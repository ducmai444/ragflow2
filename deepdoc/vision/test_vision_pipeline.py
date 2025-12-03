#
#  Simple test script for deepdoc.vision
#  Run OCR + (optional) table structure recognition on an image or PDF
#

import argparse
import os
import re
from typing import List

import numpy as np

from deepdoc.vision import OCR, TableStructureRecognizer, LayoutRecognizer, init_in_out
from deepdoc.vision.seeit import draw_box


def get_table_html(img, tb_cpns, ocr: OCR) -> str:
    """
    Build an HTML table from table-structure components and OCR,
    adapted from deepdoc.vision.t_recognizer.get_table_html.
    """
    from deepdoc.vision.layout_recognizer import LayoutRecognizer as _LR  # local alias

    boxes = ocr(np.array(img))
    boxes = _LR.sort_Y_firstly(
        [
            {
                "x0": b[0][0],
                "x1": b[1][0],
                "top": b[0][1],
                "text": t[0],
                "bottom": b[-1][1],
                "layout_type": "table",
                "page_number": 0,
            }
            for b, t in boxes
            if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]
        ],
        np.mean([b[-1][1] - b[0][1] for b, _ in boxes]) / 3,
    )

    def gather(kwd, fzy=10, ption=0.6):
        nonlocal boxes
        eles = _LR.sort_Y_firstly(
            [r for r in tb_cpns if re.match(kwd, r["label"])], fzy
        )
        eles = _LR.layouts_cleanup(boxes, eles, 5, ption)
        return _LR.sort_Y_firstly(eles, 0)

    headers = gather(r".*header$")
    rows = gather(r".* (row|header)")
    spans = gather(r".*spanning")
    clmns = sorted(
        [r for r in tb_cpns if re.match(r"table column$", r["label"])],
        key=lambda x: x["x0"],
    )
    clmns = _LR.layouts_cleanup(boxes, clmns, 5, 0.5)

    for b in boxes:
        ii = _LR.find_overlapped_with_threshold(b, rows, thr=0.3)
        if ii is not None:
            b["R"] = ii
            b["R_top"] = rows[ii]["top"]
            b["R_bott"] = rows[ii]["bottom"]

        ii = _LR.find_overlapped_with_threshold(b, headers, thr=0.3)
        if ii is not None:
            b["H_top"] = headers[ii]["top"]
            b["H_bott"] = headers[ii]["bottom"]
            b["H_left"] = headers[ii]["x0"]
            b["H_right"] = headers[ii]["x1"]
            b["H"] = ii

        ii = _LR.find_horizontally_tightest_fit(b, clmns)
        if ii is not None:
            b["C"] = ii
            b["C_left"] = clmns[ii]["x0"]
            b["C_right"] = clmns[ii]["x1"]

        ii = _LR.find_overlapped_with_threshold(b, spans, thr=0.3)
        if ii is not None:
            b["H_top"] = spans[ii]["top"]
            b["H_bott"] = spans[ii]["bottom"]
            b["H_left"] = spans[ii]["x0"]
            b["H_right"] = spans[ii]["x1"]
            b["SP"] = ii

    html = """
    <html>
    <head>
    <style>
    ._table_1nkzy_11 {
      margin: auto;
      width: 70%%;
      padding: 10px;
    }
    ._table_1nkzy_11 p {
      margin-bottom: 50px;
      border: 1px solid #e1e1e1;
    }

    caption {
      color: #6ac1ca;
      font-size: 20px;
      height: 50px;
      line-height: 50px;
      font-weight: 600;
      margin-bottom: 10px;
    }

    ._table_1nkzy_11 table {
      width: 100%%;
      border-collapse: collapse;
    }

    th {
      color: #fff;
      background-color: #6ac1ca;
    }

    td:hover {
      background: #c1e8e8;
    }

    tr:nth-child(even) {
      background-color: #f2f2f2;
    }

    ._table_1nkzy_11 th,
    ._table_1nkzy_11 td {
      text-align: center;
      border: 1px solid #ddd;
      padding: 8px;
    }
    </style>
    </head>
    <body>
    %s
    </body>
    </html>
    """ % TableStructureRecognizer.construct_table(boxes, html=True)
    return html


def run_pipeline(
    inputs: str,
    output_dir: str = "./vision_test_outputs",
    layout_threshold: float = 0.5,
    table_threshold: float = 0.2,
) -> None:
    """
    Run OCR + layout + (optional) table structure recognition on a PDF or image.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Reuse existing helper to load images from image/PDF path
    class _Args:
        def __init__(self, inputs, output_dir):
            self.inputs = inputs
            self.output_dir = output_dir

    images, outputs = init_in_out(_Args(inputs, output_dir))

    if not images:
        print("No images loaded from:", inputs)
        return

    print(f"Loaded {len(images)} page(s) from '{inputs}'")

    ocr = OCR()
    layout_rec = LayoutRecognizer("layout")
    tsr = TableStructureRecognizer()

    # OCR results per page
    all_ocr_boxes: List[List[dict]] = []

    # 1) OCR for each page
    for i, img in enumerate(images):
        print(f"[Page {i}] Running OCR...")
        raw_boxes = ocr(np.array(img))
        # raw_boxes is list [(b, text), ...]; convert to dicts
        bxs = [
            {
                "text": t[0],
                "bbox": [b[0][0], b[0][1], b[1][0], b[-1][1]],
                "type": "ocr",
                "score": 1.0,
            }
            for b, t in raw_boxes
            if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]
        ]
        all_ocr_boxes.append(bxs)

        # Save plain text
        txt_path = outputs[i] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join([o["text"] for o in bxs]))
        print(f"[Page {i}] Saved OCR text to {txt_path}")

    # 2) Layout recognition using OCR boxes
    print("Running layout recognition on all pages...")
    ocr_with_layout, page_layouts = layout_rec(images, all_ocr_boxes, thr=layout_threshold)

    # Save layout-annotated images
    for i, img in enumerate(images):
        lts = page_layouts[i]
        out_img = draw_box(img, lts, layout_rec.labels, layout_threshold)
        out_path = outputs[i] + ".layout.jpg"
        out_img.save(out_path, quality=95)
        print(f"[Page {i}] Saved layout visualization to {out_path}")

    # 3) Table structure recognition + HTML reconstruction
    print("Running table structure recognition on all pages...")
    table_components_per_page = tsr(images, thr=table_threshold)

    for i, img in enumerate(images):
        tb_cpns = table_components_per_page[i] if i < len(table_components_per_page) else []
        if not tb_cpns:
            print(f"[Page {i}] No table components detected.")
            continue

        # HTML reconstruction
        html = get_table_html(img, tb_cpns, ocr)
        html_path = outputs[i] + ".table.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[Page {i}] Saved table HTML to {html_path}")

        # Visualize table components
        vis_boxes = [
            {
                "type": t["label"],
                "bbox": [t["x0"], t["top"], t["x1"], t["bottom"]],
                "score": t["score"],
            }
            for t in tb_cpns
        ]
        vis_img = draw_box(img, vis_boxes, tsr.labels, table_threshold)
        vis_path = outputs[i] + ".table.jpg"
        vis_img.save(vis_path, quality=95)
        print(f"[Page {i}] Saved table visualization to {vis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test deepdoc.vision (OCR + layout + table) on a PDF or image."
    )
    parser.add_argument(
        "--inputs",
        required=True,
        help="Đường dẫn tới 1 file ảnh/PDF hoặc 1 thư mục chứa ảnh/PDF.",
    )
    parser.add_argument(
        "--output_dir",
        default="./vision_test_outputs",
        help="Thư mục lưu kết quả (ảnh annotate, HTML bảng, text).",
    )
    parser.add_argument(
        "--layout_threshold",
        type=float,
        default=0.5,
        help="Ngưỡng confidence cho layout detector.",
    )
    parser.add_argument(
        "--table_threshold",
        type=float,
        default=0.2,
        help="Ngưỡng confidence cho table-structure detector.",
    )
    args = parser.parse_args()

    run_pipeline(
        inputs=args.inputs,
        output_dir=args.output_dir,
        layout_threshold=args.layout_threshold,
        table_threshold=args.table_threshold,
    )


if __name__ == "__main__":
    main()


