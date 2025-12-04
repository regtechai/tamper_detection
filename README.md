# tamper_detection
POC for tramper detection

## ENV setup

```bash
python3 -m venv .venv
source .venv/bin/activate

Install packages: 
pip install -r requirements.txt

```

## Test Chinese OCR

I used easyOCR for Chinese and English, performance is OK but not great.

```bash

(.venv) liling@Lis-MacBook-Pro tamper_detection % python cn_ocr_reader.py img/A09.png 
Using CPU. Note: This module is much faster with a GPU.
/Users/liling/projects/law/tamper_detection/.venv/lib/python3.11/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)

Detected 10 text items in img/A09.png

01. AOg  (conf=0.352)
    bbox=[[np.int32(554), np.int32(190)], [np.int32(584), np.int32(190)], [np.int32(584), np.int32(214)], [np.int32(554), np.int32(214)]]
02. 0022-08-10  (conf=0.371)
    bbox=[[np.int32(235), np.int32(1029)], [np.int32(412), np.int32(1029)], [np.int32(412), np.int32(1078)], [np.int32(235), np.int32(1078)]]
03. 13,46  (conf=0.512)
    bbox=[[np.int32(16), np.int32(1036)], [np.int32(208), np.int32(1036)], [np.int32(208), np.int32(1114)], [np.int32(16), np.int32(1114)]]
04. 麈期三晴 27e  (conf=0.022)
    bbox=[[np.int32(237), np.int32(1079)], [np.int32(437), np.int32(1079)], [np.int32(437), np.int32(1119)], [np.int32(237), np.int32(1119)]]
05. 今8水耶  (conf=0.003)
    bbox=[[np.int32(800), np.int32(1096)], [np.int32(892), np.int32(1096)], [np.int32(892), np.int32(1122)], [np.int32(800), np.int32(1122)]]
06. 相机 -  (conf=0.905)
    bbox=[[np.int32(814), np.int32(1118)], [np.int32(886), np.int32(1118)], [np.int32(886), np.int32(1150)], [np.int32(814), np.int32(1150)]]
07. 呼和浩特市  (conf=0.886)
    bbox=[[np.int32(25), np.int32(1133)], [np.int32(169), np.int32(1133)], [np.int32(169), np.int32(1173)], [np.int32(25), np.int32(1173)]]
08. 阿木尔兆街  (conf=0.493)
    bbox=[[np.int32(184), np.int32(1132)], [np.int32(328), np.int32(1132)], [np.int32(328), np.int32(1172)], [np.int32(184), np.int32(1172)]]
09. 赓实时间  (conf=0.006)
    bbox=[[np.int32(798), np.int32(1148)], [np.int32(890), np.int32(1148)], [np.int32(890), np.int32(1178)], [np.int32(798), np.int32(1178)]]
10. 防伪 MGUHMRIMBHZNMW  (conf=0.187)
    bbox=[[np.int32(737), np.int32(1177)], [np.int32(897), np.int32(1177)], [np.int32(897), np.int32(1197)], [np.int32(737), np.int32(1197)]]

```
## Run tamper detection for two images

This script will compare giving two images, calculate similarity, OCR to extract text, calculate scored based similarity/text mismatch, and circle the text area: red - mismatch, green - match, blue - partial

```bash
(.venv) liling@Lis-MacBook-Pro tamper_detection % python fraud_detector_text_ocr.py img/A09.png img/A16.jpg 
Using CPU. Note: This module is much faster with a GPU.
[INFO] easyocr loaded with langs=['ch_sim','en']
[INFO] Perceptual hash distance: 4
[INFO] Blurred SSIM score: 0.7733
[INFO] Images are near-duplicates; checking text differences...
/Users/liling/projects/law/tamper_detection/.venv/lib/python3.11/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)
  Region 1: 文本一致 -> 13:46
  Region 2: 文本一致 -> 相机
  Region 3: 文本一致 -> 呼和浩特市
  Region 4: 文本一致 -> 阿木尔北街
  Region 5: 文本不一致 -> img1=A09 | img2=A16
[INFO] OCR mismatches: 1, partial regions: 0

[RESULT] Fraud suspicion score: 0.79 (0=clean, 1=highly suspicious)
[WARN] No Chinese-capable font found; falling back to default font (no Chinese glyphs).
[OUTPUT] OCR comparison image saved to: /Users/liling/projects/law/tamper_detection/pair_ocr_comparison.png

```

For above example, we can see that the fraud score is 0.79 and A09/A16 are marked in the comparison output. 