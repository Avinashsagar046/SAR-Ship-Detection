from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import csv

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\SAR_SHIP")

MODEL_PATH = PROJECT_DIR / "models" / "yolo11n_HRSID" / "best.pt"

IMAGE_DIR = PROJECT_DIR / "yolo_dataset" / "images" / "test"
LABEL_DIR = PROJECT_DIR / "yolo_dataset" / "labels" / "test"

OUTPUT_DIR = PROJECT_DIR / "results" / "YOLO"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "common_evaluation.csv"

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 800
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50


# ============================================================
# IoU FUNCTION
# ============================================================

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two boxes.

    Box format:
    [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


# ============================================================
# READ YOLO GROUND-TRUTH LABELS
# ============================================================

def read_ground_truth(label_path, image_width, image_height):
    """
    Read YOLO-format labels.

    YOLO format:
    class_id x_center y_center width height

    Coordinates are normalized from 0 to 1.

    Returns:
        list of boxes in [x1, y1, x2, y2] format.
    """

    ground_truth = []

    if not label_path.exists():
        return ground_truth

    with open(label_path, "r") as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id, xc, yc, w, h = map(float, values)

            # Convert normalized coordinates to pixels
            xc *= image_width
            yc *= image_height
            w *= image_width
            h *= image_height

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            ground_truth.append(
                [x1, y1, x2, y2]
            )

    return ground_truth


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("YOLO COMMON EVALUATION")
print("=" * 60)

print("\nLoading model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")


# ============================================================
# GET TEST IMAGES
# ============================================================

image_paths = sorted(
    list(IMAGE_DIR.glob("*.png")) +
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg"))
)

print(f"\nTest images found: {len(image_paths)}")

if len(image_paths) == 0:
    raise RuntimeError("No test images found.")


# ============================================================
# EVALUATION
# ============================================================

total_tp = 0
total_fp = 0
total_fn = 0

per_image_results = []


for index, image_path in enumerate(image_paths, start=1):

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"WARNING: Could not read {image_path}")
        continue

    image_height, image_width = image.shape[:2]

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    gt_boxes = read_ground_truth(
        label_path,
        image_width,
        image_height
    )

    # --------------------------------------------------------
    # YOLO prediction
    # --------------------------------------------------------

    results = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    result = results[0]

    predicted_boxes = []

    if result.boxes is not None:

        for box in result.boxes.xyxy.cpu().numpy():

            predicted_boxes.append(
                box.tolist()
            )

    # --------------------------------------------------------
    # Match predictions with ground truth
    # --------------------------------------------------------

    matched_gt = set()

    tp = 0
    fp = 0

    # Sort predictions by confidence
    if result.boxes is not None and len(result.boxes) > 0:

        confidences = result.boxes.conf.cpu().numpy()

        prediction_data = list(
            zip(predicted_boxes, confidences)
        )

        prediction_data.sort(
            key=lambda x: x[1],
            reverse=True
        )

    else:

        prediction_data = []

    # --------------------------------------------------------
    # Greedy IoU matching
    # --------------------------------------------------------

    for pred_box, confidence in prediction_data:

        best_iou = 0
        best_gt_index = -1

        for gt_index, gt_box in enumerate(gt_boxes):

            if gt_index in matched_gt:
                continue

            iou = calculate_iou(
                pred_box,
                gt_box
            )

            if iou > best_iou:

                best_iou = iou
                best_gt_index = gt_index

        if best_iou >= IOU_THRESHOLD:

            tp += 1
            matched_gt.add(best_gt_index)

        else:

            fp += 1

    # Any unmatched ground truth = FN
    fn = len(gt_boxes) - len(matched_gt)

    # --------------------------------------------------------
    # Accumulate
    # --------------------------------------------------------

    total_tp += tp
    total_fp += fp
    total_fn += fn

    per_image_results.append(
        [
            image_path.name,
            len(gt_boxes),
            len(prediction_data),
            tp,
            fp,
            fn
        ]
    )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if index % 100 == 0 or index == len(image_paths):

        print(
            f"Processed {index}/{len(image_paths)} images | "
            f"TP={total_tp} "
            f"FP={total_fp} "
            f"FN={total_fn}"
        )


# ============================================================
# FINAL METRICS
# ============================================================

precision = (
    total_tp / (total_tp + total_fp)
    if (total_tp + total_fp) > 0
    else 0
)

recall = (
    total_tp / (total_tp + total_fn)
    if (total_tp + total_fn) > 0
    else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(OUTPUT_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Image",
        "Ground_Truth",
        "Predictions",
        "TP",
        "FP",
        "FN"
    ])

    writer.writerows(per_image_results)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("FINAL COMMON EVALUATION")
print("=" * 60)

print(f"Test images : {len(image_paths)}")
print(f"TP          : {total_tp}")
print(f"FP          : {total_fp}")
print(f"FN          : {total_fn}")

print(f"\nPrecision   : {precision:.6f}")
print(f"Recall      : {recall:.6f}")
print(f"F1 Score    : {f1:.6f}")

print("\nPercentage:")
print(f"Precision   : {precision * 100:.2f}%")
print(f"Recall      : {recall * 100:.2f}%")
print(f"F1 Score    : {f1 * 100:.2f}%")

print(f"\nResults saved to:")
print(OUTPUT_FILE)

print("=" * 60)