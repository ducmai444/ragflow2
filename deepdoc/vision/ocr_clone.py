import logging
import copy
from posix import sendfile
import time
import os

from huggingface_hub import snapshot_download

from api.utils.file_utils import get_project_base_directory
from rag.settings import PARALLEL_DEVICES
from .operators import *
from . import operators
import math
import numpy as np
import cv2
import onnxruntime as ort

from .postprocess import build_post_process

loaded_models = {}

def transform(data, ops=None):
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data

def create_operators(op_param_list, global_config=None):
    ops = []
    for operator in op_param_list:
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = getattr(operators, op_name)(**param)
        ops.append(op)
    return ops

def load_model(model_dir, nm, device_id):
    model_file_path = os.path.join(model_dir, nm + ".onnx")
    model_cached_tag = model_file_path + str(device_id) if device_id is not None else model_file_path

    global loaded_models
    loaded_model = loaded_models.get(model_cached_tag)
    if loaded_model:
        return loaded_model
    
    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(
            model_file_path
        ))

    def cuda_is_available():
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > device_id:
            return True
    
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 2

    # Shrink GPU memory after execution
    run_options = ort.RunOptions()
    if cuda_is_available():
        cuda_provider_options = {
            "device_id": device_id,
            "gpu_mem_limit": 512 * 1024 * 1024
        }
        sess = ort.InferenceSession(
            model_file_path,
            options=options,
            providers=['CUDAExecutionProvider'],
            provider_options=[cuda_provider_options]
        )
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:" + str(device_id))
    loaded_model = (sess, run_options)
    loaded_models[model_cached_tag] = loaded_model
    return loaded_model

class TextRecognizer:
    def __init__(self, model_dir, device_id):
        self.rec_image_shape = [3, 48, 320]
        self.rec_batch_num = 16
        postprocess_params = {
            "name": "CTCLabelDecode",
            "character_dict_path": os.path.join(model_dir, "ocr.res"),
            "use_space_char": True
        }
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.run_options = load_model(model_dir, "rec", device_id)
        self.input_tensor = self.predictor.get_inputs()[0]

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape

        imgW = int(imgH * max_wh_ratio)
        w = self.input_tensor.shape[3:][0]
        if w is not None and w > 0:
            imgW = w
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2,0,1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
    
    def __call__(self, img_list):
        img_num = len(img_list)
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
            indices = np.argsort(np.array(width_list))
            rec_res = [['', 0.0]] * img_num
            batch_num = self.rec_batch_num
            st = time.time()

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0/h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no): 
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                            max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            input_dict = {}
            input_dict[self.input_tensor_name] = norm_img_batch
            for i in range(100000):
                try:
                    outputs = self.predictor.run(None, input_dict, self.run_options)
                    break
                except Exception as e:
                    if i >= 3:
                        raise e
                    time.sleep(5)
            preds = outputs[0]
            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
                
        return rec_res, time.time() - st

class TextDetector:
    def __init__(self, model_dir, device_id: int | None = None):
        pre_process_list = [{
            "DetResizeForTest": {
                "limit_side_len": 960,
                "limit_type": "max"
            }
        }, {
            "NormalizeImage": {
                "std" : [0.229, 0.224, 0.225],
                "mean" : [0.485, 0.456, 0.406],
                "scale" : "1./255.",
                "order" : "hwc"
            }
        }, {
            "ToCHWImage": None
        }, {
            "KeepKeys": {
                "keep_keys": ["image", "shape"]
            }
        }]
        postprocess_params = {"name": "DBPostProcess", "thresh": 0.3, "box_thresh": 0.45,
                            "max_candidates": 1000, "unclip_ratio": 1.5, "use_dilation": False,
                            "score_mode": "fast", "box_type": "quad"}
        
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.run_options = load_model(model_dir, "det", device_id)
        self.input_tensor = self.predictor.get_inputs()[0]

        img_h, img_w = self.input_tensor.shape[2:]
        if img_h is not None and img_w is not None and img_h > 0 and img_w > 0:
            pre_process_list[0] = {
                "DetResizeForTest": {
                    "image_shape": [img_h, img_w]
                }
            }
        self.preprocess_op = create_operators(pre_process_list)

    def order_points_clockwise(self, pts):
        