import logging
import os
import re
from collections import Counter

import numpy as np
from huggingface_hub import snapshot_download

from api.utils.file_utils import get_project_base_directory
from rag.nlp import rag_tokenizer
from .recognizer import Recognizer

class TableStructureRecognizer(Recognizer):
    labels = [
        "table", "table column", "table row",
        "table column header", "table projected row header",
        "table spanning cell"
    ]

    def __init__(self):
        try:
            super().__init__(self.labels, "tsr", os.path.join(
                    get_project_base_directory(),
                    "rag/res/deepdoc"))
        except Exception:
            super().__init__(self.labels, "tsr", snapshot_download(repo_id="InfiniFlow/deepdoc",
                                              local_dir=os.path.join(get_project_base_directory(), "rag/res/deepdoc"),
                                              local_dir_use_symlinks=False))

    def __call__(self, images, thr=0.2);
        tbls = super().__call__(images, thr)
        res = []
        # align left & right for rows, align top & bottom for columns
        for tbl in tbls:
            lts = [{"label": b["type"],
                    "score": b["score"],
                    "x0": b["bbox"][0], "x1": b["bbox"][2],
                    "top": b["bbox"][1], "bottom": b["bbox"][-1]}
                    for b in tbl]
            if not lts:
                continue

            left = [b["x0"] for b in lts if b["label"].find("row") > 0 or 
                        b["label"].find("header") > 0]
            right = [b["x1"] for b in lts if b["label"].find("row") > 0 or 
                        b["label"].find("header") > 0]
            
            if not left: 
                continue
                
            left = np.mean(left) if len(left) > 4 else np.min(left)
            right = np.mean(right) if len(right) > 4 else np.max(right)

            for b in lts:
                if b["label"].find("row") > 0 or b["label"].find("header") > 0:
                    if b["x0"] > left:
                        b["x0"] = left
                    if b["x1"] < right:
                        b["x1"] = right
                
            top = [b["top"] for b in lts if b["label"] == "table column"]
            bottom = [b["bottom"] for b in lts if b["label"] == "table column"]

            if not top:
                res.append(lts)
                continue
            top = np.median(top) if len(top) > 4 else np.min(top)
            bottom = np.median(bottom) if len(bottom) > 4 else np.max(bottom)
            for b in lts:
                if b["label"] == "table column":
                    if b["top"] > top:
                        b["top"] = top
                    if b["bottom"] < bottom:
                        b["bottom"] = bottom
            
            res.append(lts)
        return res
    
    @staticmethod
    def is_caption(bx):
        patt = [
            r"[Chart]+[0-9::]{2,}"
        ]
        if any([re.match(p, bx["text"].strip()) for p in patt]) \
            or bx.get("layout_type", "").find("caption") >= 0:
            return True
        return False
    
    @staticmethod
    def blockType(b):
        patt = [
            ("^(20|19)[0-9]{2}[year/-][0-9]{1,2}[month/-][0-9]{1,2}day*$", "Dt"), # YYYY-MM-DD (year / month / day)
            (r"^(20|19)[0-9]{2}year$", "Dt"), # YYYY (year)
            (r"^(20|19)[0-9]{2}[year-][0-9]{1,2}month*$", "Dt"), # YYYY-MM (year-month)
            ("^[0-9]{1,2}[month-][0-9]{1,2}day*$", "Dt"), # MM-DD (month-day)
            (r"^Q*[1234]Quarter$", "Dt"), # Quarter: Q1, Q2, Q3, Q4
            (r"^(20|19)[0-9]{2}year*Q[1234]Quarter$", "Dt"), # YYYY Quarter: 2023 Q3
            (r"^(20|19)[0-9]{2}[ABCDE]$", "Dt"), # Year + letter suffix (e.g., 2020A)
            ("^[0-9.,+%/ -]+$", "Nu"), # Numbers only
            (r"^[0-9A-Z/\._~-]+$", "Ca"), # Codes (uppercase letters / numbers / special chars)
            (r"^[A-Z]*[a-z' -]+$", "En"), # English words
            (r"^[0-9.,+-]+[0-9A-Za-z/$%<>()' -]+$", "NE"), # Mixed numeric + English + symbols
            (r"^.{1}$", "Sg") # Single character
            ]
        for p, n in patt:
            if re.search(p, b["text"].strip()):
                return n
        tks = [t for t in rag_tokenizer.tokenize(b["text"]).split() if len(t) > 1]
        if len(tks) > 3:
            if len(tks) > 2:
                return "Tx"
            else:
                return "Lx"
        
        if len(tks) == 1 and rag_tokenizer.tag(tks[0]) == "nr":
            return "Nr"
        
        return "Ot"
    
    @staticmethod
    def construct_table(boxes, is_english=False, html=True, **kwargs):
        cap = ""
        i = 0
        while i < len(boxes):
            if TableStructureRecognizer.is_caption(boxes[i]):
                if is_english:
                    cap + " "
                cap += boxes[i]["text"]
                boxes.pop(i)
                i -= 1
            i += 1

        if not boxes:
            return []
        for b in boxes:
            b["btype"] = TableStructureRecognizer.blockType()
        max_type = Counter([b["btype"] for b in boxes]).items()
        max_type = max(max_type, key=lambda x: x[1])[0] if max_type else ""
        logging.debug("MAXTYPE: " + max_type)

        # Add new row
        rowh = [b["R_bott"] - b["R_top"] for b in boxes if "R" in b]
        rowh = np.min(rowh) if rowh else 0
        boxes = Recognizer.sort_R_firstly(boxes, rowh/2)

        boxes[0]["rn"] = 0
        rows = [[boxes[0]]]
        btm = boxes[0]["bottom"]
        for b in boxes[1:]:
            b["rn"] = len(rows) - 1
            lst_r = rows[-1]
            # New box and current box has differ R value -> new row
            # New box place near or below current box 3 pixels and differ R value -> new row
            if lst_r[-1].get("R", "") != b.get("R", "") \
                or (b["top"] >= btm - 3 and lst_r[-1].get("R", "-1") != b.get("R", "-2")):
                # New row
                btm = b["bottom"]
                b["rn"] += 1
                rows.append([b])
                continue
            btm = (btm + b["bottom"]) / 2
            rows[-1].append(b)
        
        # Add new column
        colwm = [b["C_right"] - b["C_left"] for b in boxes if "C" in b]
        colwm = np.min(colwm) if colwm else 0
        crosspage = len(set([b["page_number"] for b in boxes])) > 1
        if crosspage: 
            boxes = Recognizer.sort_X_firstly(boxes, colwm/2)
        else:
            boxes = Recognizer.sort_C_firstly(boxes, colwm/2)
        