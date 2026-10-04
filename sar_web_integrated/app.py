from pathlib import Path
import json
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from scipy import ndimage
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(r"D:\SAR_SHIP") if Path(r"D:\SAR_SHIP").exists() else ROOT / "SAR_SHIP"
DATASET_IMAGES = PROJECT_ROOT / "DATASET" / "images"
TEST_JSON = PROJECT_ROOT / "DATASET" / "annotations" / "test2017.json"
MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n_HRSID" / "best.pt"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# Final tested configurations used by the research pipeline.
WINDOW_SIZE = 21
GUARD_SIZE = 5
PFA = 0.001
CA_MIN_AREA = 10
OS_MIN_AREA = 20
OS_RANK_INDEX = 332  # 80% rank of 416 training cells, zero-based
OS_ALPHA = 4.35377163833898
YOLO_CONF = 0.40
YOLO_IMGSZ = 800
IOU_THRESHOLD = 0.50

_model = None
_test_annotations = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
    return _model


def load_test_annotations():
    global _test_annotations
    if _test_annotations is not None:
        return _test_annotations
    if not TEST_JSON.exists():
        _test_annotations = {}
        return _test_annotations

    with open(TEST_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    image_map = {item["file_name"]: item["id"] for item in coco["images"]}
    annotations = {}
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        annotations.setdefault(image_id, []).append(ann["bbox"])

    result = {}
    for filename, image_id in image_map.items():
        boxes = []
        for x, y, w, h in annotations.get(image_id, []):
            boxes.append([x, y, x + w, y + h])
        result[filename] = boxes

    _test_annotations = result
    return result


def alpha_ca_cfar(n, pfa):
    return n * (pfa ** (-1.0 / n) - 1.0)


def training_mask(window_size, guard_size):
    mask = np.ones((window_size, window_size), dtype=np.float32)
    center = window_size // 2
    g = guard_size // 2
    mask[center - g:center + g + 1, center - g:center + g + 1] = 0
    return mask


def connected_boxes(binary, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    boxes = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        boxes.append([x, y, x + w, y + h])
    return boxes


def ca_cfar(gray):
    mask = training_mask(WINDOW_SIZE, GUARD_SIZE)
    n = int(mask.sum())
    alpha = alpha_ca_cfar(n, PFA)
    local_sum = ndimage.convolve(gray.astype(np.float32), mask, mode="constant", cval=0)
    background = local_sum / n
    binary = (gray.astype(np.float32) > alpha * background).astype(np.uint8)
    half = WINDOW_SIZE // 2
    binary[:half, :] = 0
    binary[-half:, :] = 0
    binary[:, :half] = 0
    binary[:, -half:] = 0
    return connected_boxes(binary, CA_MIN_AREA)


def os_cfar(gray):
    background = ndimage.rank_filter(
        gray.astype(np.float32),
        rank=OS_RANK_INDEX,
        size=WINDOW_SIZE,
        mode="constant",
        cval=0,
    )
    binary = (gray.astype(np.float32) > OS_ALPHA * background).astype(np.uint8)
    half = WINDOW_SIZE // 2
    binary[:half, :] = 0
    binary[-half:, :] = 0
    binary[:, :half] = 0
    binary[:, -half:] = 0
    return connected_boxes(binary, OS_MIN_AREA)


def yolo_detect(gray):
    model = get_model()
    # Use a 3-channel representation for consistent model input.
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result = model.predict(source=bgr, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)[0]

    detections = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "label": "ship" if int(cls) == 0 else str(int(cls)),
            })
    return detections


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def match_metrics(pred_boxes, gt_boxes):
    used = set()
    tp = 0
    for pred in pred_boxes:
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(gt_boxes):
            if idx in used:
                continue
            value = iou(pred, gt)
            if value >= IOU_THRESHOLD and value > best_iou:
                best_iou = value
                best_idx = idx
        if best_idx is not None:
            used.add(best_idx)
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def draw_boxes(gray, boxes, title, box_type="detector"):
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Green for GT, red for CFAR candidates, blue for YOLO detections.
    color = (80, 255, 80) if box_type == "gt" else ((80, 80, 255) if box_type == "cfar" else (255, 120, 60))
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return canvas


def make_display_enhancement(gray):
    # Visualization-only enhancement. It is NOT fed to the evaluated detectors.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def save_image(img, name):
    path = OUTPUT_DIR / name
    cv2.imwrite(str(path), img)
    return f"/outputs/{path.name}"


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.post("/detect")
def detect():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify(success=False, error="No image file was uploaded."), 400

    suffix = Path(upload.filename).suffix.lower() or ".png"
    job_id = uuid.uuid4().hex[:10]
    input_path = OUTPUT_DIR / f"input_{job_id}{suffix}"
    upload.save(input_path)

    try:
        gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("The uploaded file could not be read as an image.")

        ca_boxes = ca_cfar(gray)
        os_boxes = os_cfar(gray)
        yolo_detections = yolo_detect(gray)
        yolo_boxes = [d["bbox"] for d in yolo_detections]

        # If this exact filename exists in HRSID test annotations, provide image-level evaluation.
        gt_boxes = load_test_annotations().get(upload.filename, None)
        if gt_boxes is not None:
            ca_metrics = match_metrics(ca_boxes, gt_boxes)
            os_metrics = match_metrics(os_boxes, gt_boxes)
            yolo_metrics = match_metrics(yolo_boxes, gt_boxes)
        else:
            ca_metrics = os_metrics = yolo_metrics = None

        original_url = save_image(gray, f"original_{job_id}.png")
        enhanced_url = save_image(make_display_enhancement(gray), f"enhanced_{job_id}.png")
        ca_img = draw_boxes(gray, ca_boxes, f"CA-CFAR: {len(ca_boxes)} candidates", "cfar")
        os_img = draw_boxes(gray, os_boxes, f"OS-CFAR: {len(os_boxes)} candidates", "cfar")
        yolo_img = draw_boxes(gray, yolo_boxes, f"YOLO11n: {len(yolo_boxes)} detections", "yolo")

        # Three-panel comparison for the web demo.
        panels = [ca_img, os_img, yolo_img]
        panel_width = max(p.shape[1] for p in panels)
        normalized = []
        for panel in panels:
            if panel.shape[1] != panel_width:
                panel = cv2.resize(panel, (panel_width, panel.shape[0]))
            normalized.append(panel)
        comparison = cv2.hconcat(normalized)

        ca_url = save_image(ca_img, f"ca_{job_id}.png")
        os_url = save_image(os_img, f"os_{job_id}.png")
        yolo_url = save_image(yolo_img, f"yolo_{job_id}.png")
        comparison_url = save_image(comparison, f"comparison_{job_id}.png")

        avg_conf = float(np.mean([d["confidence"] for d in yolo_detections])) if yolo_detections else 0.0

        # Final held-out test metrics from the trained YOLO11n model.
        # The common IoU-based evaluation at confidence 0.40 is kept
        # separately because it uses a different evaluation procedure.
        model_metrics = {
            "precision": 0.902319,
            "recall": 0.796690,
            "map50": 0.893586,
            "map5095": 0.639999,
            "common_precision": 0.891426,
            "common_recall": 0.804120,
            "common_f1": 0.845526,
        }

        response = {
            "success": True,
            "original": original_url,
            "processed": enhanced_url,
            "cfar": {
                "url": os_url,
                "ca_url": ca_url,
                "os_url": os_url,
                "ca_candidates": len(ca_boxes),
                "os_candidates": len(os_boxes),
                "detections": len(os_boxes),
                "ca_metrics": ca_metrics,
                "os_metrics": os_metrics,
            },
            "yolo": yolo_url,
            "comparison": comparison_url,
            "detections": [
                {"id": i + 1, "label": d["label"], "confidence": d["confidence"], "bbox": d["bbox"]}
                for i, d in enumerate(yolo_detections)
            ],
            "total_detections": len(yolo_detections),
            "average_confidence": avg_conf,
            "model_metrics": model_metrics,
            "evaluation_available": gt_boxes is not None,
            "ground_truth_count": len(gt_boxes) if gt_boxes is not None else None,
            "image_metrics": {
                "ca": ca_metrics,
                "os": os_metrics,
                "yolo": yolo_metrics,
            },
            "settings": {
                "ca": {"window": WINDOW_SIZE, "guard": GUARD_SIZE, "pfa": PFA, "min_area": CA_MIN_AREA},
                "os": {"window": WINDOW_SIZE, "guard": GUARD_SIZE, "pfa": PFA, "rank": "80%", "min_area": OS_MIN_AREA},
                "yolo": {"imgsz": YOLO_IMGSZ, "confidence": YOLO_CONF},
            },
        }
        return jsonify(response)

    except Exception as exc:
        app.logger.exception("Detection failed")
        return jsonify(success=False, error=str(exc)), 500
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    print("SAR Ship Detector web app")
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset images: {DATASET_IMAGES}")
    app.run(host="127.0.0.1", port=5000, debug=False)
