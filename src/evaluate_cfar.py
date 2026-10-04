import json
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "DATASET"
IMAGE_DIR = DATASET_DIR / "images"
ANNOTATION_FILE = DATASET_DIR / "annotations" / "test2017.json"


# --------------------------------------------------
# CA-CFAR parameters
# --------------------------------------------------

TRAINING_WINDOW_SIZE = 21
GUARD_WINDOW_SIZE = 5
PFA = 1e-2

MIN_COMPONENT_AREA = 3

IOU_THRESHOLD = 0.50


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("JSON loaded successfully!")


# --------------------------------------------------
# IoU function
# --------------------------------------------------

def calculate_iou(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    box1_x2 = x1 + w1
    box1_y2 = y1 + h1

    box2_x2 = x2 + w2
    box2_y2 = y2 + h2

    intersection_x1 = max(x1, x2)
    intersection_y1 = max(y1, y2)

    intersection_x2 = min(box1_x2, box2_x2)
    intersection_y2 = min(box1_y2, box2_y2)

    intersection_width = max(
        0, intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0, intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    area1 = w1 * h1
    area2 = w2 * h2

    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


# --------------------------------------------------
# CA-CFAR function
# --------------------------------------------------

def run_cfar(image):

    image_float = image.astype(np.float32)

    training_area = TRAINING_WINDOW_SIZE ** 2
    guard_area = GUARD_WINDOW_SIZE ** 2

    training_cells = training_area - guard_area

    alpha = training_cells * (
        PFA ** (-1 / training_cells) - 1
    )

    outer_sum = cv2.boxFilter(
        image_float,
        ddepth=-1,
        ksize=(
            TRAINING_WINDOW_SIZE,
            TRAINING_WINDOW_SIZE
        ),
        normalize=False
    )

    inner_sum = cv2.boxFilter(
        image_float,
        ddepth=-1,
        ksize=(
            GUARD_WINDOW_SIZE,
            GUARD_WINDOW_SIZE
        ),
        normalize=False
    )

    ones = np.ones_like(image_float)

    outer_count = cv2.boxFilter(
        ones,
        ddepth=-1,
        ksize=(
            TRAINING_WINDOW_SIZE,
            TRAINING_WINDOW_SIZE
        ),
        normalize=False
    )

    inner_count = cv2.boxFilter(
        ones,
        ddepth=-1,
        ksize=(
            GUARD_WINDOW_SIZE,
            GUARD_WINDOW_SIZE
        ),
        normalize=False
    )

    training_sum = outer_sum - inner_sum
    training_count = outer_count - inner_count

    background = np.divide(
        training_sum,
        training_count,
        out=np.zeros_like(training_sum),
        where=training_count > 0
    )

    threshold = alpha * background

    detections = (
        image_float > threshold
    ).astype(np.uint8)

    # Remove border
    half_window = TRAINING_WINDOW_SIZE // 2

    detections[:half_window, :] = 0
    detections[-half_window:, :] = 0
    detections[:, :half_window] = 0
    detections[:, -half_window:] = 0

    return detections


# --------------------------------------------------
# Extract candidate bounding boxes
# --------------------------------------------------

def get_candidate_boxes(detections):

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            detections,
            connectivity=8
        )
    )

    boxes = []

    for label in range(1, num_labels):

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]

        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]

        area = stats[label, cv2.CC_STAT_AREA]

        if area < MIN_COMPONENT_AREA:
            continue

        boxes.append(
            (x, y, width, height)
        )

    return boxes


# --------------------------------------------------
# Organize ground truth annotations
# --------------------------------------------------

ground_truth = {}

for annotation in data["annotations"]:

    image_id = annotation["image_id"]

    if image_id not in ground_truth:
        ground_truth[image_id] = []

    ground_truth[image_id].append(
        tuple(annotation["bbox"])
    )


# --------------------------------------------------
# Evaluate all test images
# --------------------------------------------------

total_tp = 0
total_fp = 0
total_fn = 0

total_images = len(data["images"])


for index, image_info in enumerate(data["images"], start=1):

    image_id = image_info["id"]
    file_name = image_info["file_name"]

    image_path = IMAGE_DIR / file_name

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        print("Could not load:", file_name)
        continue

    detections = run_cfar(image)

    candidate_boxes = get_candidate_boxes(
        detections
    )

    gt_boxes = ground_truth.get(
        image_id,
        []
    )

    matched_gt = set()

    tp = 0
    fp = 0

    # Match each CFAR candidate to the
    # best available ground-truth box
    for candidate in candidate_boxes:

        best_iou = 0
        best_gt_index = None

        for gt_index, gt_box in enumerate(gt_boxes):

            if gt_index in matched_gt:
                continue

            iou = calculate_iou(
                candidate,
                gt_box
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if (
            best_iou >= IOU_THRESHOLD
            and best_gt_index is not None
        ):
            tp += 1
            matched_gt.add(best_gt_index)

        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    total_tp += tp
    total_fp += fp
    total_fn += fn

    if index % 100 == 0:
        print(
            f"Processed {index}/{total_images} images"
        )


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

precision = (
    total_tp /
    (total_tp + total_fp)
    if (total_tp + total_fp) > 0
    else 0
)

recall = (
    total_tp /
    (total_tp + total_fn)
    if (total_tp + total_fn) > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)


# --------------------------------------------------
# Display final results
# --------------------------------------------------

print("\n========================================")
print("       CA-CFAR EVALUATION RESULTS")
print("========================================")

print("Images evaluated:", total_images)

print("TP:", total_tp)
print("FP:", total_fp)
print("FN:", total_fn)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print("\nParameters:")
print("Training window:", TRAINING_WINDOW_SIZE)
print("Guard window:", GUARD_WINDOW_SIZE)
print("Pfa:", PFA)
print("Minimum component area:", MIN_COMPONENT_AREA)
print("IoU threshold:", IOU_THRESHOLD)

print("========================================")