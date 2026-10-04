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
# Parameters
# --------------------------------------------------

TRAINING_WINDOW_SIZE = 21
GUARD_WINDOW_SIZE = 5

PFA = 0.001

MIN_COMPONENT_AREA = 10

IOU_THRESHOLD = 0.50

RANK_PERCENTILE = 80


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

ground_truth = {}

for annotation in data["annotations"]:

    image_id = annotation["image_id"]

    if image_id not in ground_truth:
        ground_truth[image_id] = []

    ground_truth[image_id].append(
        tuple(annotation["bbox"])
    )


# --------------------------------------------------
# IoU
# --------------------------------------------------

def calculate_iou(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    ix1 = max(x1, x2)
    iy1 = max(y1, y2)

    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    union = (
        w1 * h1 +
        w2 * h2 -
        intersection
    )

    if union == 0:
        return 0.0

    return intersection / union


# --------------------------------------------------
# OS-CFAR
# --------------------------------------------------

def run_os_cfar(image):

    image_float = image.astype(np.float32)

    half_training = TRAINING_WINDOW_SIZE // 2
    half_guard = GUARD_WINDOW_SIZE // 2

    height, width = image.shape

    detections = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    training_cells = (
        TRAINING_WINDOW_SIZE ** 2 -
        GUARD_WINDOW_SIZE ** 2
    )

    alpha = training_cells * (
        PFA ** (-1 / training_cells) - 1
    )

    rank = int(
        training_cells *
        RANK_PERCENTILE / 100
    )

    rank = max(1, rank)

    print("Training cells:", training_cells)
    print("Selected rank:", rank)
    print("PFA:", PFA)
    print("Rank percentile:", RANK_PERCENTILE)

    # ----------------------------------------------
    # Process each valid pixel
    # ----------------------------------------------

    for y in range(
        half_training,
        height - half_training
    ):

        for x in range(
            half_training,
            width - half_training
        ):

            window = image_float[
                y - half_training:
                y + half_training + 1,

                x - half_training:
                x + half_training + 1
            ]

            # --------------------------------------
            # Remove guard region
            # --------------------------------------

            guard_start = (
                half_training -
                half_guard
            )

            guard_end = (
                half_training +
                half_guard +
                1
            )

            training_values = window.copy()

            training_values[
                guard_start:guard_end,
                guard_start:guard_end
            ] = np.nan

            training_values = (
                training_values[
                    ~np.isnan(training_values)
                ]
            )

            # --------------------------------------
            # Sort training cells
            # --------------------------------------

            sorted_values = np.sort(
                training_values
            )

            # --------------------------------------
            # Select ranked background value
            # --------------------------------------

            rank_index = min(
                rank - 1,
                len(sorted_values) - 1
            )

            background = (
                sorted_values[rank_index]
            )

            threshold = (
                alpha *
                background
            )

            # --------------------------------------
            # CUT decision
            # --------------------------------------

            if image_float[y, x] > threshold:
                detections[y, x] = 1

    return detections


# --------------------------------------------------
# Candidate boxes
# --------------------------------------------------

def get_candidate_boxes(
    detections,
    min_area
):

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            detections,
            connectivity=8
        )
    )

    boxes = []

    for label in range(1, num_labels):

        x = stats[
            label,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            label,
            cv2.CC_STAT_TOP
        ]

        width = stats[
            label,
            cv2.CC_STAT_WIDTH
        ]

        height = stats[
            label,
            cv2.CC_STAT_HEIGHT
        ]

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area < min_area:
            continue

        boxes.append(
            (x, y, width, height)
        )

    return boxes


# --------------------------------------------------
# Evaluate image
# --------------------------------------------------

def evaluate(
    candidate_boxes,
    gt_boxes
):

    matched_gt = set()

    tp = 0
    fp = 0

    for candidate in candidate_boxes:

        best_iou = 0
        best_index = None

        for i, gt in enumerate(gt_boxes):

            if i in matched_gt:
                continue

            iou = calculate_iou(
                candidate,
                gt
            )

            if iou > best_iou:
                best_iou = iou
                best_index = i

        if (
            best_iou >= IOU_THRESHOLD
            and best_index is not None
        ):
            tp += 1
            matched_gt.add(best_index)

        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn


# --------------------------------------------------
# Run test set
# --------------------------------------------------

total_tp = 0
total_fp = 0
total_fn = 0

total_images = len(data["images"])

for index, image_info in enumerate(
    data["images"],
    start=1
):

    image_id = image_info["id"]

    image_path = (
        IMAGE_DIR /
        image_info["file_name"]
    )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        continue

    detections = run_os_cfar(image)

    candidates = get_candidate_boxes(
        detections,
        MIN_COMPONENT_AREA
    )

    gt_boxes = ground_truth.get(
        image_id,
        []
    )

    tp, fp, fn = evaluate(
        candidates,
        gt_boxes
    )

    total_tp += tp
    total_fp += fp
    total_fn += fn

    if index % 100 == 0:
        print(
            f"Processed {index}/{total_images}"
        )


# --------------------------------------------------
# Metrics
# --------------------------------------------------

precision = (
    total_tp /
    (total_tp + total_fp)
    if total_tp + total_fp > 0
    else 0
)

recall = (
    total_tp /
    (total_tp + total_fn)
    if total_tp + total_fn > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall > 0
    else 0
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n===================================")
print("          OS-CFAR RESULTS")
print("===================================")

print("Images:", total_images)

print("TP:", total_tp)
print("FP:", total_fp)
print("FN:", total_fn)

print("\nPrecision:", round(precision, 6))
print("Recall:", round(recall, 6))
print("F1:", round(f1, 6))

print("\nParameters:")
print("Training:", TRAINING_WINDOW_SIZE)
print("Guard:", GUARD_WINDOW_SIZE)
print("PFA:", PFA)
print("Rank:", RANK_PERCENTILE)
print("Min area:", MIN_COMPONENT_AREA)
print("IoU:", IOU_THRESHOLD)