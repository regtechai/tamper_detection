import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os

from PIL import Image, ImageDraw, ImageFont

# ----------------- OCR setup (EasyOCR, Chinese + English) ----------------- #

try:
    import easyocr
    OCR_AVAILABLE = True
    OCR_READER = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    print("[INFO] easyocr loaded with langs=['ch_sim','en']")
except ImportError:
    OCR_AVAILABLE = False
    OCR_READER = None
    print("[WARN] easyocr not installed; OCR comparison will be skipped.")


# Paths to Chinese-capable fonts. Adjust to your environment.
FONT_PATHS = [
    "NotoSansCJK-Regular.ttc",                     # local file
    "NotoSansCJK-Regular.otf",
    "SimHei.ttf",
    "/System/Library/Fonts/PingFang.ttc",         # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

def load_chinese_font(size=22):
    for p in FONT_PATHS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    print("[WARN] No Chinese-capable font found; falling back to default font (no Chinese glyphs).")
    return ImageFont.load_default()

def preprocess_for_ocr(img_bgr, scale=2.0):
    """
    Preprocess whole image for OCR:
    - upscale (INTER_CUBIC) so small text becomes larger
    - increase local contrast with CLAHE
    Returns:
        img_bgr_upscaled, scale_factor
    """
    h, w = img_bgr.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    up = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # convert to LAB and apply CLAHE on L channel to boost contrast
    lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_eq = clahe.apply(L)
    lab_eq = cv2.merge([L_eq, A, B])
    up_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    return up_eq, scale

# ---------- Utility: perceptual hash (average hash) ----------

def average_hash(img, hash_size=16):
    """
    Simple perceptual hash.
    Returns a 1D boolean array length hash_size*hash_size.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = resized.mean()
    return resized > avg


def hamming_distance(hash1, hash2):
    return np.count_nonzero(hash1 != hash2)


# ---------- Core pipeline: load + overall similarity ----------

def load_and_align(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    if img1 is None or img2 is None:
        raise ValueError("Failed to load one of the images.")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if (h1, w1) != (h2, w2):
        img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_AREA)

    return img1, img2


def compute_similarity(img1, img2, hash_size=16, use_blur=True):
    # ---- Perceptual hash (main near-duplicate signal) ----
    h1 = average_hash(img1, hash_size=hash_size)
    h2 = average_hash(img2, hash_size=hash_size)
    phash_dist = hamming_distance(h1, h2)

    # ---- SSIM (helper signal, on blurred grayscale) ----
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if use_blur:
        gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    ssim_score = ssim(gray1, gray2, full=False)

    return phash_dist, float(ssim_score)


# ---------- OCR: full-image + box matching ----------

def run_full_image_ocr(img, min_score=0.5):
    """
    Run OCR on a preprocessed (upscaled + contrast-enhanced) version
    of the whole image with EasyOCR.

    Returns list of (box=(x,y,w,h), text, score) in ORIGINAL image
    coordinates.
    """
    if not OCR_AVAILABLE or OCR_READER is None:
        return []

    # 1) Preprocess & upscale
    pre_img, scale = preprocess_for_ocr(img, scale=2.0)  # try 2.0 or 3.0 if text is tiny
    pre_rgb = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)

    # 2) OCR on upscaled image
    ocr_raw = OCR_READER.readtext(pre_rgb, detail=1)  # [ [box, text, score], ... ]
    results = []

    for box, text, score in ocr_raw:
        if score < min_score:
            continue

        # box is 4 points in upscaled coordinates → map back to original
        pts = np.array(box, dtype=np.float32) / scale
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))

        results.append(((x, y, w, h), text.strip(), float(score)))

    return results


def box_iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter <= 0:
        return 0.0
    area1 = w1 * h1
    area2 = w2 * h2
    return inter / float(area1 + area2 - inter + 1e-6)


def find_ocr_mismatches(img1, img2,
                        min_score=0.6,
                        iou_thresh=0.3):
    """
    OCR both images, match text boxes by IoU, compare text.

    Returns:
      ocr_summary: list of dicts:
           {
             "box": (x,y,w,h),  # from image1
             "text1": str,
             "text2": str,
             "status": "match" | "mismatch" | "partial" | "none"
           }
    """
    ocr1 = run_full_image_ocr(img1, min_score=min_score)
    ocr2 = run_full_image_ocr(img2, min_score=min_score)

    ocr_summary = []
    used2 = set()

    for (box1, text1, score1) in ocr1:
        best_j = -1
        best_iou = 0.0
        for j, (box2, text2, score2) in enumerate(ocr2):
            if j in used2:
                continue
            iou = box_iou(box1, box2)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j < 0 or best_iou < iou_thresh:
            continue

        used2.add(best_j)
        box2, text2, score2 = ocr2[best_j]
        t1 = text1.strip()
        t2 = text2.strip()

        if not t1 and not t2:
            status = "none"
        elif t1 == t2:
            status = "match"
        else:
            if not t1 or not t2:
                status = "partial"
            else:
                status = "mismatch"

        ocr_summary.append({
            "box": box1,
            "text1": t1,
            "text2": t2,
            "status": status,
        })

    return ocr_summary


# ---------- Save comparison image (Chinese-safe) ----------

def save_comparison_with_ocr(img1, img2, ocr_summary, out_path):
    """
    img1, img2: original aligned images (BGR)
    ocr_summary: list of dicts above
    out_path: file to save

    Boxes drawn with OpenCV, labels drawn with PIL so Chinese works.
    """
    h, w = img1.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = img1
    canvas[:, w:] = img2

    # 1) Draw boxes in OpenCV
    for item in ocr_summary:
        x, y, bw, bh = item["box"]
        status = item["status"]

        if status == "mismatch":
            color = (0, 0, 255)      # red
        elif status == "partial":
            color = (0, 255, 255)    # yellow
        elif status == "match":
            color = (0, 255, 0)      # green
        else:
            color = (255, 255, 255)  # white / unknown

        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 3)
        cv2.rectangle(canvas, (x + w, y), (x + w + bw, y + bh), color, 3)

    # 2) Convert to PIL for text drawing
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(canvas_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = load_chinese_font(size=22)

    for item in ocr_summary:
        x, y, bw, bh = item["box"]
        status = item["status"]
        t1 = item["text1"]
        t2 = item["text2"]

        if status == "mismatch":
            txt_color = (255, 0, 0)
        elif status == "partial":
            txt_color = (255, 255, 0)
        elif status == "match":
            txt_color = (0, 255, 0)
        else:
            txt_color = (255, 255, 255)

        label = f"{t1} | {t2}"
        y_text = max(10, y - 25)
        draw.text((x, y_text), label, font=font, fill=txt_color)

    # 3) Back to OpenCV and save
    final_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, final_bgr)


# ---------- High-level fraud detection ----------

def detect_fraud(img_path1, img_path2, debug_prefix="output"):
    img1, img2 = load_and_align(img_path1, img_path2)

    # Overall similarity
    phash_dist, ssim_score = compute_similarity(img1, img2)
    print(f"[INFO] Perceptual hash distance: {phash_dist}")
    print(f"[INFO] Blurred SSIM score: {ssim_score:.4f}")

    near_duplicate = (phash_dist <= 8) or (ssim_score > 0.70)
    if near_duplicate:
        print("[INFO] Images are near-duplicates; checking text differences...")
    else:
        print("[INFO] Images are not strongly near-duplicate; fraud score will be lower.")

    # OCR-based text comparison
    ocr_summary = []
    mismatch_count = 0
    partial_count = 0

    if OCR_AVAILABLE:
        ocr_summary = find_ocr_mismatches(img1, img2)
        for i, s in enumerate(ocr_summary, 1):
            t1 = s["text1"]
            t2 = s["text2"]
            status = s["status"]
            if status == "match":
                print(f"  Region {i}: 文本一致 -> {t1}")
            elif status == "mismatch":
                mismatch_count += 1
                print(f"  Region {i}: 文本不一致 -> img1={t1} | img2={t2}")
            elif status == "partial":
                partial_count += 1
                print(f"  Region {i}: 文本不完整 -> img1={t1} | img2={t2}")
            else:
                print(f"  Region {i}: 无有效文本")
    else:
        print("[WARN] OCR not available; only overall similarity checked.")

    print(f"[INFO] OCR mismatches: {mismatch_count}, partial regions: {partial_count}")

    # Fraud scoring (OCR-aware + overall similarity)
    fraud_score = 0.0

    if near_duplicate:
        fraud_score += 0.3

    if mismatch_count + partial_count > 0:
        total = mismatch_count + partial_count
        ocr_evidence = (0.7 * mismatch_count + 0.3 * partial_count) / total
        fraud_score += 0.7 * ocr_evidence

    fraud_score = float(max(0.0, min(1.0, fraud_score)))

    print(f"\n[RESULT] Fraud suspicion score: {fraud_score:.2f} (0=clean, 1=highly suspicious)")

    # Save comparison image
    paths = {}
    comparison_path = f"{debug_prefix}_ocr_comparison.png"
    if ocr_summary:
        save_comparison_with_ocr(img1, img2, ocr_summary, comparison_path)
        paths["ocr_comparison"] = comparison_path
        print(f"[OUTPUT] OCR comparison image saved to: {os.path.abspath(comparison_path)}")

    return {
        "phash_distance": phash_dist,
        "ssim_score": ssim_score,
        "fraud_score": fraud_score,
        "near_duplicate": near_duplicate,
        "ocr_summary": ocr_summary,
        "output_paths": paths,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR-based fraud detection")
    parser.add_argument("image1", help="Path to first image")
    parser.add_argument("image2", help="Path to second image")
    parser.add_argument("--prefix", default="pair",
                        help="Output file prefix (default: 'pair')")

    args = parser.parse_args()

    detect_fraud(args.image1, args.image2, debug_prefix=args.prefix)

