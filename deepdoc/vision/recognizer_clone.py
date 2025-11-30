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
        arr = Recognizer.sort_X()
        pass
