#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import easyocr


# Create reader once: Simplified Chinese + English
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)


def preprocess(img):
    """
    Simple preprocessing to help OCR:
    - ensure it's reasonably large
    - slightly sharpen and increase contrast
    """
    h, w = img.shape[:2]

    # upscale if smaller than ~1200 on longest side
    scale = 1.0
    max_side = max(h, w)
    if max_side < 1200:
        scale = 1200.0 / max_side
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_LINEAR)

    # convert to grayscale, then back to 3-channel
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # histogram equalization to boost contrast
    gray = cv2.equalizeHist(gray)

    # light sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], np.float32)
    sharp = cv2.filter2D(gray, -1, kernel)

    # EasyOCR expects 3-channel
    sharp = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    return sharp


def run_ocr(img_path: str):
    # read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img_p = preprocess(img)

    # detail=1 -> we get [bbox, text, confidence]
    results = reader.readtext(img_p, detail=1)

    print(f"\nDetected {len(results)} text items in {img_path}\n")
    for i, (bbox, text, conf) in enumerate(results, 1):
        print(f"{i:02d}. {text}  (conf={conf:.3f})")
        print(f"    bbox={bbox}")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cn_ocr_easy.py image_path")
        sys.exit(1)

    run_ocr(sys.argv[1])
