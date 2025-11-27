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
            model_dir = 