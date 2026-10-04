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

OUTPUT_FILE = OUTPUT_DIR / "small_target_analysis.csv"


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 800

# From our confidence-threshold experiment
CONF_THRESHOLD = 0.40

IOU_THRESHOLD = 0.50

# Our experimental target-size bins
SMALL_AREA = 408
LARGE_AREA = 1380


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
# TARGET SIZE
# ============================================================

def get_size_category(box):

    width = max(0, box[2] - box[0])
    height = max(0, box[3] - box[1])

    area = width * height

    if area < SMALL_AREA:
        return "Small"

    elif area < LARGE_AREA:
        return "Medium"

    else:
        return "Large"


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

            boxes.append([
                x1, y1, x2, y2
            ])

    return boxes


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("YOLO SMALL-TARGET ANALYSIS")
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
# STATISTICS
# ============================================================

categories = [
    "Small",
    "Medium",
    "Large"
]

stats = {}

for category in categories:

    stats[category] = {
        "gt": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0
    }


# ============================================================
# EVALUATION
# ============================================================

for index, image_path in enumerate(image_paths, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    gt_boxes = read_ground_truth(
        label_path,
        width,
        height
    )

    # Determine size category of every GT ship
    gt_categories = [
        get_size_category(box)
        for box in gt_boxes
    ]

    # Count ground-truth ships
    for category in gt_categories:
        stats[category]["gt"] += 1

    # --------------------------------------------------------
    # YOLO prediction
    # --------------------------------------------------------

    result = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False
    )[0]

    predictions = []

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, confidence in zip(
            boxes,
            confidences
        ):

            predictions.append(
                (
                    box.tolist(),
                    float(confidence)
                )
            )

    # Highest confidence first
    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    matched_gt = set()

    # --------------------------------------------------------
    # Match predictions
    # --------------------------------------------------------

    for pred_box, confidence in predictions:

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

            matched_gt.add(best_gt_index)

            category = gt_categories[best_gt_index]

            stats[category]["tp"] += 1

        else:

            # False positive.
            #
            # A false positive does not have a ground-truth
            # size category, so we record it separately later.
            pass

    # --------------------------------------------------------
    # False negatives
    # --------------------------------------------------------

    for gt_index, category in enumerate(gt_categories):

        if gt_index not in matched_gt:

            stats[category]["fn"] += 1

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if index % 200 == 0 or index == len(image_paths):

        print(
            f"Processed {index}/{len(image_paths)}"
        )


# ============================================================
# CALCULATE METRICS
# ============================================================

final_results = []

print("\n" + "=" * 70)
print("SMALL-TARGET RESULTS")
print("=" * 70)

for category in categories:

    gt = stats[category]["gt"]
    tp = stats[category]["tp"]
    fn = stats[category]["fn"]

    # NOTE:
    #
    # Precision requires category-specific FP.
    # A prediction that does not match any GT is not assigned
    # a target-size category.
    #
    # Therefore, for this analysis we report:
    # Detection Recall by size.
    #
    # Precision/F1 are not calculated per size category here
    # unless false positives are explicitly categorized.

    recall = (
        tp / gt
        if gt > 0
        else 0
    )

    print(
        f"{category:8s} | "
        f"GT={gt:5d} | "
        f"TP={tp:5d} | "
        f"FN={fn:5d} | "
        f"Recall={recall:.4f}"
    )

    final_results.append([
        category,
        gt,
        tp,
        fn,
        recall
    ])


# ============================================================
# SAVE RESULTS
# ============================================================

with open(OUTPUT_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Target_Size",
        "Ground_Truth",
        "TP",
        "FN",
        "Recall"
    ])

    writer.writerows(final_results)


print("\nResults saved to:")
print(OUTPUT_FILE)

print("=" * 70)