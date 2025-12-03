import os
import re
from collections import Counter
from copy import deepcopy

import cv2
import numpy as np
from huggingface_hub import snapshot_download

from api.utils.file_utils import get_project_base_directory
from deepdoc.vision import Recognizer
from deepdoc.vision.operators import nms

class LayoutRecognizer(Recognizer):
    labels = [
        "_background_",
        "Text",
        "Title",
        "Figure",
        "Figure caption",
        "Table",
        "Table caption",
        "Header",
        "Footer",
        "Reference",
        "Equation"
    ]

    def __init__(self, domain):
        try:
            model_dir = os.path.join(
                get_project_base_directory(),
                "rag/res/deepdoc")
            super().__init__(self.labels, domain, model_dir)
        except Exception:
            model_dir = snapshot_download(repo_id="InfiniFlow/deepdoc",
                                          local_dir=os.path.join(get_project_base_directory(), "rag/res/deepdoc"),
                                          local_dir_use_symlinks=False)
            super().__init__(self.labels, domain, model_dir)

        self.garbage_layouts = ["footer", "header", "reference"]
        self.client = None
        if os.environ.get("TENSORRT_DLA_SVR"):
            from deepdoc.vision.dla_cli import DLAClient
            self.client = DLAClient(os.environ["TENSORRT_DLA_SVR"])
    
    def __call__(self, image_list, ocr_res, scale_factor=3, thr=0.2, batch_size=16, drop=True):
        def __is_garbage(b):
            patt = [r"^•+$", "^[0-9]{1,2} / ?[0-9]{1,2}$",
                    r"^[0-9]{1,2} of [0-9]{1,2}$", "^http://[^ ]{12,}",
                    "\\(cid *: *[0-9]+ *\\)"
                    ]
            return any([re.search(p, b["text"]) for p in patt])
        
        if self.client:
            layouts = self.client.predict(image_list)
        else:
            layouts = super().__call__(image_list, thr, batch_size)
        assert len(image_list) == len(ocr_res)
        boxes = []
        assert len(image_list) == len(layouts)
        garbages = {}
        page_layout = []
        for pn, lts in enumerate(layouts):
            bxs = ocr_res[pn]
            lts = [{"type": b["type"],
                    "score": float(b["score"]),
                    "x0": b["bbox"][0]/scale_factor, "x1": b["bbox"][2]/scale_factor,
                    "top": b["bbox"][1]/scale_factor, "bottom": b["bbox"][-1]/scale_factor,
                    "page_number": pn} for b in lts if float(b["score"]) >= 0.4 or b["type"] not in self.garbage_layouts]
            lts = self.sort_Y_firstly(lts, np.mean(
                [lt["bottom"] - lt["top"] for lt in lts]) / 2)
            lts = self.layouts_cleanup(bxs, lts)
            page_layout.append(lts)

            def findLayout(ty):
                nonlocal bxs, lts, self
                lts_ = [lt for lt in lts if lt["type"] == ty]
                i = 0
                while i < len(bxs):
                    if bxs[i].get("layout_type"):
                        i += 1
                        continue
                    if __is_garbage(bxs[i]):
                        bxs.pop(i)
                        continue
                    
                    ii = self.find_overlapped_with_threshold(bxs[i], lts_, thr=0.4)

                    if ii is None:
                        bxs[i]["layout_type"] = ""
                        i += 1
                        continue
                    lts_[ii]["visited"] = True
                    keep_feats = [
                        lts_[ii]["type"] == "footer" and bxs[i]["bottom"] < image_list[pn].size[1] * 0.9 / scale_factor,
                        lts_[ii]["type"] == "header" and bxs[i]["top"] > image_list[pn].size[1] * 0.1 / scale_factor,
                    ]
                    if drop and lts_[ii]["type"] in self.garbage_layouts and not any(keep_feats):
                        if lts_[ii]["type"] not in garbages:
                            garbages[lts_[ii]["type"]] = []
                        garbages[lts_[ii]["type"]].append(bxs[i]["text"])
                        bxs.pop(i)
                        continue
                    
                    bxs[i]["layoutno"] = f"{ty}-{ii}"
                    bxs[i]["layout_type"] = lts_[ii]["type"] if lts_[ii]["type"] != "equation" else "figure"
                    i += 1
                
            for lt in ["footer", "header", "reference", "figure caption",
                        "table caption", "title", "table", "text", "figure", "equation"]:
                findLayout(lt)
            
            # Add box to figure layouts which has not text box
            for i, lt in enumerate(
                [lt for lt in lts if lt["type"] in ["figure", "equation"]]):
                if lt.get("visited"):
                    continue
                lt = deepcopy(lt)
                del lt["type"]
                lt["text"] = ""
                lt["layout_type"] = "figure"
                lt["layoutno"] = f"figure-{i}"
                bxs.append(lt)

            boxes.extend(bxs)
        
        ocr_res = boxes

        garbage_set = set()
        for k in garbages.keys():
            garbages[k] = Counter(garbages[k])
            for g, c in garbages[k].items():
                if c > 1:
                    garbage_set.add(g)

        ocr_res = [b for b in ocr_res if b["text"].strip() not in garbage_set]
        return ocr_res, page_layout
    
    def forward(self, image_list, thr=0.7, batch_size=16):
        return super().__call__(image_list, thr, batch_size)
