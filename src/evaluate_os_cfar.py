import cv2
import numpy as np
import json
import os
import math
import pandas as pd

from scipy.ndimage import rank_filter
from scipy.optimize import brentq


# ============================================================
# 1. PATHS
# ============================================================

IMAGE_DIR = r"D:\SAR_SHIP\DATASET\images"

ANNOTATION_PATH = (
    r"D:\SAR_SHIP\DATASET\annotations\test2017.json"
)

RESULTS_DIR = r"D:\SAR_SHIP\results"


# ============================================================
# 2. OS-CFAR PARAMETERS
# ============================================================

WINDOW_SIZE = 21
GUARD_SIZE = 5

PFA = 0.001

RANK_PERCENTILE = 80

# Minimum connected-component area.
#
# We start with 1 so that we do not accidentally remove
# the very small OS-CFAR detections we observed earlier.

MIN_COMPONENT_AREA = 1

# IoU threshold used to decide whether a prediction
# matches a ground-truth ship.

IOU_THRESHOLD = 0.50


# ============================================================
# 3. CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 4. LOAD HRSID ANNOTATIONS
# ============================================================

with open(
    ANNOTATION_PATH,
    "r"
) as file:

    data = json.load(file)


print("Annotation file loaded successfully!")


# ============================================================
# 5. ORGANIZE ANNOTATIONS BY IMAGE ID
# ============================================================

annotations_by_image = {}

for annotation in data["annotations"]:

    image_id = annotation["image_id"]

    if image_id not in annotations_by_image:

        annotations_by_image[image_id] = []

    annotations_by_image[image_id].append(
        annotation["bbox"]
    )


# ============================================================
# 6. GET TEST IMAGES
# ============================================================

image_entries = data["images"]

print(
    "Number of test images:",
    len(image_entries)
)

print(
    "Number of annotations:",
    len(data["annotations"])
)


# ============================================================
# 7. OS-CFAR PARAMETERS
# ============================================================

half_window = WINDOW_SIZE // 2
half_guard = GUARD_SIZE // 2

total_cells = WINDOW_SIZE * WINDOW_SIZE

guard_cells = GUARD_SIZE * GUARD_SIZE

training_cells = (
    total_cells - guard_cells
)


# Calculate order-statistic rank.

rank = int(
    np.ceil(
        RANK_PERCENTILE
        / 100
        * training_cells
    )
)


print("\nOS-CFAR configuration")
print("--------------------------------")
print("Window size:", WINDOW_SIZE)
print("Guard size:", GUARD_SIZE)
print("Training cells:", training_cells)
print("Rank percentile:", RANK_PERCENTILE)
print("Selected rank:", rank)
print("PFA:", PFA)
print("IoU threshold:", IOU_THRESHOLD)
print("Minimum component area:", MIN_COMPONENT_AREA)
print("--------------------------------")


# ============================================================
# 8. CALCULATE OS-CFAR THRESHOLD FACTOR
# ============================================================

def log_pfa(alpha):

    value = 0.0

    for j in range(
        training_cells - rank + 1,
        training_cells + 1
    ):

        value += (
            math.log(j)
            - math.log(j + alpha)
        )

    return value


target_log_pfa = math.log(PFA)


alpha = brentq(
    lambda a:
        log_pfa(a) - target_log_pfa,
    0,
    1000
)


print(
    "OS-CFAR threshold factor:",
    alpha
)


# ============================================================
# 9. CREATE OS-CFAR FOOTPRINT
# ============================================================

# True = training cell
# False = guard cell / CUT

footprint = np.ones(
    (
        WINDOW_SIZE,
        WINDOW_SIZE
    ),
    dtype=bool
)


center = WINDOW_SIZE // 2


footprint[
    center - half_guard:
    center + half_guard + 1,

    center - half_guard:
    center + half_guard + 1
] = False


# rank_filter uses zero-based rank.
#
# Our selected rank is 333.
# Therefore rank_filter needs 332.

filter_rank = rank - 1


print(
    "Rank-filter index:",
    filter_rank
)


# ============================================================
# 10. IoU FUNCTION
# ============================================================

def calculate_iou(box_a, box_b):

    ax, ay, aw, ah = box_a

    bx, by, bw, bh = box_b


    # Convert to corner coordinates.

    ax2 = ax + aw
    ay2 = ay + ah

    bx2 = bx + bw
    by2 = by + bh


    # Intersection.

    ix1 = max(ax, bx)
    iy1 = max(ay, by)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)


    intersection_width = max(
        0,
        ix2 - ix1
    )

    intersection_height = max(
        0,
        iy2 - iy1
    )


    intersection_area = (
        intersection_width
        * intersection_height
    )


    # Individual areas.

    area_a = aw * ah

    area_b = bw * bh


    # Union.

    union_area = (
        area_a
        + area_b
        - intersection_area
    )


    if union_area <= 0:

        return 0.0


    return (
        intersection_area
        / union_area
    )


# ============================================================
# 11. EXTRACT CONNECTED-COMPONENT BOXES
# ============================================================

def detection_boxes(
    detection_map,
    min_area
):

    number_of_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            detection_map,
            connectivity=8
        )
    )


    boxes = []


    # Label 0 is the background.

    for label in range(
        1,
        number_of_labels
    ):

        x = stats[
            label,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            label,
            cv2.CC_STAT_TOP
        ]

        w = stats[
            label,
            cv2.CC_STAT_WIDTH
        ]

        h = stats[
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
            [
                float(x),
                float(y),
                float(w),
                float(h)
            ]
        )


    return boxes


# ============================================================
# 12. MATCH PREDICTIONS TO GROUND TRUTH
# ============================================================

def match_boxes(
    predictions,
    ground_truths,
    iou_threshold
):

    matches = []

    used_predictions = set()

    used_ground_truths = set()


    # Create all possible prediction/GT pairs.

    possible_matches = []


    for prediction_index, prediction in enumerate(
        predictions
    ):

        for gt_index, ground_truth in enumerate(
            ground_truths
        ):

            iou = calculate_iou(
                prediction,
                ground_truth
            )


            if iou >= iou_threshold:

                possible_matches.append(
                    (
                        iou,
                        prediction_index,
                        gt_index
                    )
                )


    # Match highest IoU first.

    possible_matches.sort(
        reverse=True
    )


    for (
        iou,
        prediction_index,
        gt_index
    ) in possible_matches:

        if prediction_index in used_predictions:

            continue


        if gt_index in used_ground_truths:

            continue


        used_predictions.add(
            prediction_index
        )

        used_ground_truths.add(
            gt_index
        )

        matches.append(
            (
                prediction_index,
                gt_index,
                iou
            )
        )


    tp = len(matches)

    fp = (
        len(predictions)
        - tp
    )

    fn = (
        len(ground_truths)
        - tp
    )


    return tp, fp, fn


# ============================================================
# 13. EVALUATION VARIABLES
# ============================================================

total_tp = 0
total_fp = 0
total_fn = 0

images_evaluated = 0

images_with_detections = 0

total_detection_pixels = 0

total_candidate_boxes = 0


# ============================================================
# 14. STORE PER-IMAGE RESULTS
# ============================================================

results = []


# ============================================================
# 15. PROCESS TEST DATASET
# ============================================================

print("\nStarting full OS-CFAR evaluation...")
print("This may take some time.\n")


for index, image_info in enumerate(
    image_entries,
    start=1
):

    image_id = image_info["id"]

    file_name = image_info["file_name"]


    image_path = os.path.join(
        IMAGE_DIR,
        file_name
    )


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )


    if image is None:

        print(
            f"WARNING: Could not load {file_name}"
        )

        continue


    image = image.astype(
        np.float32
    )


    # --------------------------------------------------------
    # Run OS-CFAR
    # --------------------------------------------------------

    # rank_filter calculates the selected order statistic
    # for every pixel.

    background_estimate = rank_filter(
        image,
        rank=filter_rank,
        footprint=footprint,
        mode="nearest"
    )


    # Calculate adaptive threshold.

    threshold = (
        alpha
        * background_estimate
    )


    # Detection decision.

    detection_map = (
        image > threshold
    ).astype(
        np.uint8
    ) * 255


    # Remove border where a complete CFAR window
    # cannot be formed.

    detection_map[
        :half_window,
        :
    ] = 0

    detection_map[
        -half_window:,
        :
    ] = 0

    detection_map[
        :,
        :half_window
    ] = 0

    detection_map[
        :,
        -half_window:
    ] = 0


    # --------------------------------------------------------
    # Count raw detection pixels
    # --------------------------------------------------------

    detection_pixels = int(
        np.sum(
            detection_map > 0
        )
    )


    total_detection_pixels += (
        detection_pixels
    )


    if detection_pixels > 0:

        images_with_detections += 1


    # --------------------------------------------------------
    # Convert pixels to candidate boxes
    # --------------------------------------------------------

    predictions = detection_boxes(
        detection_map,
        MIN_COMPONENT_AREA
    )


    total_candidate_boxes += len(
        predictions
    )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truths = annotations_by_image.get(
        image_id,
        []
    )


    # --------------------------------------------------------
    # Match predictions with ground truth
    # --------------------------------------------------------

    tp, fp, fn = match_boxes(
        predictions,
        ground_truths,
        IOU_THRESHOLD
    )


    total_tp += tp
    total_fp += fp
    total_fn += fn


    images_evaluated += 1


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append(
        {
            "image_id": image_id,
            "file_name": file_name,
            "ground_truth": len(
                ground_truths
            ),
            "candidate_boxes": len(
                predictions
            ),
            "detection_pixels": detection_pixels,
            "TP": tp,
            "FP": fp,
            "FN": fn
        }
    )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        index % 25 == 0
        or index == len(image_entries)
    ):

        print(
            f"Processed "
            f"{index}/{len(image_entries)} images | "
            f"TP={total_tp} | "
            f"FP={total_fp} | "
            f"FN={total_fn}"
        )


# ============================================================
# 16. CALCULATE FINAL METRICS
# ============================================================

if (
    total_tp + total_fp
) > 0:

    precision = (
        total_tp
        / (
            total_tp
            + total_fp
        )
    )

else:

    precision = 0.0


if (
    total_tp + total_fn
) > 0:

    recall = (
        total_tp
        / (
            total_tp
            + total_fn
        )
    )

else:

    recall = 0.0


if (
    precision + recall
) > 0:

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )

else:

    f1 = 0.0


# ============================================================
# 17. PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("OS-CFAR FULL DATASET RESULTS")
print("=" * 60)

print(
    "Images evaluated:",
    images_evaluated
)

print(
    "Images with detections:",
    images_with_detections
)

print(
    "Total detection pixels:",
    total_detection_pixels
)

print(
    "Total candidate boxes:",
    total_candidate_boxes
)

print(
    "TP:",
    total_tp
)

print(
    "FP:",
    total_fp
)

print(
    "FN:",
    total_fn
)

print(
    f"Precision: {precision:.6f}"
)

print(
    f"Recall:    {recall:.6f}"
)

print(
    f"F1 Score:  {f1:.6f}"
)

print("=" * 60)


# ============================================================
# 18. SAVE RESULTS
# ============================================================

summary = pd.DataFrame(
    [
        {
            "Method": "OS-CFAR",
            "Window_Size": WINDOW_SIZE,
            "Guard_Size": GUARD_SIZE,
            "PFA": PFA,
            "Rank_Percentile": RANK_PERCENTILE,
            "Selected_Rank": rank,
            "Training_Cells": training_cells,
            "IoU_Threshold": IOU_THRESHOLD,
            "Min_Component_Area": MIN_COMPONENT_AREA,
            "Images_Evaluated": images_evaluated,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
    ]
)


summary_path = os.path.join(
    RESULTS_DIR,
    "os_cfar_summary.csv"
)


per_image_path = os.path.join(
    RESULTS_DIR,
    "os_cfar_per_image.csv"
)


summary.to_csv(
    summary_path,
    index=False
)


pd.DataFrame(
    results
).to_csv(
    per_image_path,
    index=False
)


print("\nResults saved:")
print(
    summary_path
)

print(
    per_image_path
)