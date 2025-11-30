import logging 
import os
import math
import numpy as np
import cv2
from functools import cmp_to_key

from api.utils.file_utils import get_project_base_directory
from .operators import * 
from .operators import preprocess
from . import operators
from .ocr import load_model

class Recognizer:
    def __init__(self, label_list, task_name, model_dir=None):
        if not model_dir:
            model_dir = os.path.join(
                get_project_base_directory(),
                "rag/res/deepdoc"
            )
        self.ort_sess, self.run_options = load_model(model_dir, task_name)
        self.input_names = [node.name for node in self.ort_sess.get_inputs()]
        self.output_names = [node.name for node in self.ort_sess.get_outputs()]
        self.input_shape = self.ort_sess.get_inputs()[0].shape[2:4]
        self.label_list = label_list
    
    @staticmethod
    def sort_Y_firstly(arr, threshold):
        def cmp(c1, c2):
            diff = c1["top"] - c2["top"]
            if abs(diff) < threshold:
                diff = c1["x0"] - c2["x0"]
            return diff
        arr = sorted(arr, key=cmp_to_key(cmp))
        return arr
    
    @staticmethod
    def sort_X_firstly(arr, threshold):
        def cmp(c1, c2):
            diff = c1["x0"] - c2["x0"]
            if abs(diff) < threshold:
                diff = c1["top"] - c2["top"]
            return diff
        arr = sorted(arr, key=cmp_to_key(cmp))
        return arr

    @staticmethod
    def sort_C_firstly(arr, thr=0):
        # sort using y1 first and then x1
        # sorted(arr, key=lambda r: (r["x0"], r["top"]))
        arr = Recognizer.sort_X_firstly(arr, thr)
        for i in range(len(arr) - 1):
            for j in range(i, -1, -1):
                # restore the order using th
                if "C" not in arr[j] or "C" not in arr[j+1]:
                    continue
                if arr[j+1]["C"] < arr[j]["C"] or \
                    (arr[j+1]["C"] == arr[j]["C"] and
                        arr[j+1]["top"] > arr[j]["top"]):
                    tmp = arr[j]
                    arr[j] = arr[j+1]
                    arr[j+1] = tmp
        return arr
    
    @staticmethod
    def sort_R_firstly(arr, thr=0):
        # sort using y1 first and then x1
        # sorted(arr, key=lambda r: (r["top"], r["x0"]))
        arr = Recognizer.sort_Y_firstly(arr, thr)
        for i in range(len(arr) - 1):
            for j in range(i, -1, -1):
                if "R" not in arr[j] or "R" not in arr[j + 1]:
                    continue
                if arr[j + 1]["R"] < arr[j]["R"] \
                        or (
                        arr[j + 1]["R"] == arr[j]["R"]
                        and arr[j + 1]["x0"] < arr[j]["x0"]
                ):
                    tmp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = tmp
        return arr
    
    @staticmethod
    def overlapped_area(a, b, ratio=True):
        tp, btm, x0, x1 = a["top"], a["bottom"], a["x0"], a["x1"]
        if b["x0"] > x1 or b["x1"] < x0:
            return 0
        if b["bottom"] < tp or b["top"] > btm:
            return 0
        x0_ = max(b["x0"], x0)
        x1_ = min(b["x1"], x1)
        assert x0_ <= x1_, "Bbox mismatch! T:{}, B:{}, x0:{}, x1:{} ==> {}".format(
            tp, btm, x0, x1, b)
        tp_ = max(b["top"], tp)
        btm_ = min(b["bottom"], btm)
        assert tp_ <= btm_, "Bbox mismatch! T:{}, B:{}, x0:{}, x1:{} ==> {}".format(
            tp, btm , x0, x1, b)
        ov = (btm_ - tp_) * (x1_ - x0_) if x1 - x0 != 0 \
            and btm - tp != 0 else 0
        if ov > 0 and ratio:
            ov /= (x1 - x0) * (btm - tp)
        return ov

    
