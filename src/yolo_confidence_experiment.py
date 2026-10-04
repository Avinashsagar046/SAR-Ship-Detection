from pathlib import Path
from ultralytics import YOLO
import cv2
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

OUTPUT_FILE = OUTPUT_DIR / "confidence_experiment.csv"

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 800
IOU_THRESHOLD = 0.50

CONFIDENCE_THRESHOLDS = [
    0.10,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80
]


# ============================================================
# IoU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = (
        intersection_width * intersection_height
    )

    area1 = max(0, box1[2] - box1[0]) * \
            max(0, box1[3] - box1[1])

    area2 = max(0, box2[2] - box2[0]) * \
            max(0, box2[3] - box2[1])

    union_area = area1 + area2 - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


# ============================================================
# READ GROUND TRUTH
# ============================================================

def read_ground_truth(label_path, image_width, image_height):

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            _, xc, yc, w, h = map(float, values)

            xc *= image_width
            yc *= image_height
            w *= image_width
            h *= image_height

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            boxes.append([x1, y1, x2, y2])

    return boxes


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("YOLO CONFIDENCE THRESHOLD EXPERIMENT")
print("=" * 70)

print("\nLoading model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")


# ============================================================
# TEST IMAGES
# ============================================================

image_paths = sorted(
    list(IMAGE_DIR.glob("*.png")) +
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg"))
)

print(f"Test images: {len(image_paths)}")

if not image_paths:
    raise RuntimeError("No test images found.")


# ============================================================
# IMPORTANT:
# Run inference ONCE at confidence 0.10.
#
# Then apply the higher confidence thresholds ourselves.
# This avoids running the model 9 separate times.
# ============================================================

print("\nRunning YOLO inference...")

all_predictions = []

for index, image_path in enumerate(image_paths, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    height, width = image.shape[:2]

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    gt_boxes = read_ground_truth(
        label_path,
        width,
        height
    )

    result = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=0.10,
        verbose=False
    )[0]

    predictions = []

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, confidence in zip(boxes, confidences):

            predictions.append(
                (
                    box.tolist(),
                    float(confidence)
                )
            )

    all_predictions.append(
        (
            image_path.name,
            gt_boxes,
            predictions
        )
    )

    if index % 200 == 0 or index == len(image_paths):

        print(
            f"Processed {index}/{len(image_paths)}"
        )


print("\nInference complete.")


# ============================================================
# EVALUATE EACH CONFIDENCE THRESHOLD
# ============================================================

results = []

print("\n" + "=" * 70)
print("CONFIDENCE THRESHOLD RESULTS")
print("=" * 70)

for confidence_threshold in CONFIDENCE_THRESHOLDS:

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for image_name, gt_boxes, predictions in all_predictions:

        # Keep only predictions above threshold
        filtered_predictions = [
            (box, conf)
            for box, conf in predictions
            if conf >= confidence_threshold
        ]

        # Highest confidence first
        filtered_predictions.sort(
            key=lambda x: x[1],
            reverse=True
        )

        matched_gt = set()

        tp = 0
        fp = 0

        # ----------------------------------------------------
        # Greedy IoU matching
        # ----------------------------------------------------

        for pred_box, confidence in filtered_predictions:

            best_iou = 0.0
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

        fn = len(gt_boxes) - len(matched_gt)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    results.append(
        [
            confidence_threshold,
            total_tp,
            total_fp,
            total_fn,
            precision,
            recall,
            f1
        ]
    )

    print(
        f"Confidence {confidence_threshold:.2f} | "
        f"TP {total_tp:5d} | "
        f"FP {total_fp:5d} | "
        f"FN {total_fn:5d} | "
        f"Precision {precision:.4f} | "
        f"Recall {recall:.4f} | "
        f"F1 {f1:.4f}"
    )


# ============================================================
# SAVE CSV
# ============================================================

with open(OUTPUT_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Confidence",
        "TP",
        "FP",
        "FN",
        "Precision",
        "Recall",
        "F1"
    ])

    writer.writerows(results)


# ============================================================
# BEST F1
# ============================================================

best_result = max(
    results,
    key=lambda row: row[6]
)

print("\n" + "=" * 70)
print("BEST TESTED F1")
print("=" * 70)

print(
    f"Confidence : {best_result[0]:.2f}"
)

print(
    f"TP         : {best_result[1]}"
)

print(
    f"FP         : {best_result[2]}"
)

print(
    f"FN         : {best_result[3]}"
)

print(
    f"Precision  : {best_result[4]:.6f}"
)

print(
    f"Recall     : {best_result[5]:.6f}"
)

print(
    f"F1         : {best_result[6]:.6f}"
)

print(f"\nResults saved to:")
print(OUTPUT_FILE)

print("=" * 70)