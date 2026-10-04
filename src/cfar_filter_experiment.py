import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "DATASET"
IMAGE_DIR = DATASET_DIR / "images"
ANNOTATION_FILE = DATASET_DIR / "annotations" / "test2017.json"

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Fixed CFAR parameters
# --------------------------------------------------

TRAINING_WINDOW_SIZE = 21
GUARD_WINDOW_SIZE = 5
PFA = 0.001

IOU_THRESHOLD = 0.50

# Only this parameter will change
MIN_AREAS = [3, 10, 20, 50, 100]


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
# CA-CFAR
# --------------------------------------------------

def run_cfar(image):

    image_float = image.astype(np.float32)

    training_cells = (
        TRAINING_WINDOW_SIZE ** 2 -
        GUARD_WINDOW_SIZE ** 2
    )

    alpha = training_cells * (
        PFA ** (-1 / training_cells) - 1
    )

    outer_sum = cv2.boxFilter(
        image_float,
        -1,
        (
            TRAINING_WINDOW_SIZE,
            TRAINING_WINDOW_SIZE
        ),
        normalize=False
    )

    inner_sum = cv2.boxFilter(
        image_float,
        -1,
        (
            GUARD_WINDOW_SIZE,
            GUARD_WINDOW_SIZE
        ),
        normalize=False
    )

    ones = np.ones_like(image_float)

    outer_count = cv2.boxFilter(
        ones,
        -1,
        (
            TRAINING_WINDOW_SIZE,
            TRAINING_WINDOW_SIZE
        ),
        normalize=False
    )

    inner_count = cv2.boxFilter(
        ones,
        -1,
        (
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

    half_window = TRAINING_WINDOW_SIZE // 2

    detections[:half_window, :] = 0
    detections[-half_window:, :] = 0
    detections[:, :half_window] = 0
    detections[:, -half_window:] = 0

    return detections


# --------------------------------------------------
# Candidate extraction
# --------------------------------------------------

def get_candidate_boxes(detections, min_area):

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

        if area < min_area:
            continue

        boxes.append(
            (x, y, width, height)
        )

    return boxes


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(candidate_boxes, gt_boxes):

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
# Run experiment
# --------------------------------------------------

results = []

for min_area in MIN_AREAS:

    print(
        f"\nRunning minimum component area = {min_area}"
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0

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

        detections = run_cfar(image)

        candidates = get_candidate_boxes(
            detections,
            min_area
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

        if index % 500 == 0:
            print(
                f"Processed {index}/"
                f"{len(data['images'])}"
            )

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

    results.append({
        "Min_Area": min_area,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


# --------------------------------------------------
# Results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n==========================================")
print("       CFAR FILTER EXPERIMENT")
print("==========================================")

print(
    results_df.to_string(index=False)
)


# --------------------------------------------------
# Save
# --------------------------------------------------

output_file = (
    RESULTS_DIR /
    "cfar_filter_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print("\nSaved to:", output_file)