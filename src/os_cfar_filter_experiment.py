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

IOU_THRESHOLD = 0.50


# Component areas we want to test
AREA_VALUES = [1, 3, 10, 20, 50, 100]


# ============================================================
# 3. CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 4. LOAD ANNOTATIONS
# ============================================================

with open(
    ANNOTATION_PATH,
    "r"
) as file:

    data = json.load(file)


print("Annotation file loaded successfully!")

print(
    "Test images:",
    len(data["images"])
)

print(
    "Annotations:",
    len(data["annotations"])
)


# ============================================================
# 5. ORGANIZE GROUND TRUTH BY IMAGE
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
# 6. OS-CFAR PARAMETERS
# ============================================================

half_window = WINDOW_SIZE // 2

half_guard = GUARD_SIZE // 2


total_cells = (
    WINDOW_SIZE
    * WINDOW_SIZE
)


guard_cells = (
    GUARD_SIZE
    * GUARD_SIZE
)


training_cells = (
    total_cells
    - guard_cells
)


rank = int(
    np.ceil(
        RANK_PERCENTILE
        / 100
        * training_cells
    )
)


print("\nOS-CFAR configuration")
print("--------------------------------")
print("Window:", WINDOW_SIZE)
print("Guard:", GUARD_SIZE)
print("Training cells:", training_cells)
print("Rank percentile:", RANK_PERCENTILE)
print("Selected rank:", rank)
print("PFA:", PFA)
print("IoU threshold:", IOU_THRESHOLD)
print("Area values:", AREA_VALUES)
print("--------------------------------")


# ============================================================
# 7. CALCULATE OS-CFAR THRESHOLD FACTOR
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
# 8. CREATE TRAINING-CELL FOOTPRINT
# ============================================================

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


filter_rank = rank - 1


# ============================================================
# 9. IoU FUNCTION
# ============================================================

def calculate_iou(
    box_a,
    box_b
):

    ax, ay, aw, ah = box_a

    bx, by, bw, bh = box_b


    ax2 = ax + aw
    ay2 = ay + ah

    bx2 = bx + bw
    by2 = by + bh


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


    area_a = aw * ah

    area_b = bw * bh


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
# 10. EXTRACT CONNECTED COMPONENT BOXES
# ============================================================

def get_detection_boxes(
    detection_map,
    minimum_area
):

    number_of_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            detection_map,
            connectivity=8
        )
    )


    boxes = []


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


        if area < minimum_area:

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
# 11. MATCH PREDICTIONS WITH GROUND TRUTH
# ============================================================

def match_boxes(
    predictions,
    ground_truths,
    iou_threshold
):

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


    # Highest IoU matches first

    possible_matches.sort(
        reverse=True
    )


    used_predictions = set()

    used_ground_truths = set()


    matches = []


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
# 12. STORE RESULTS
# ============================================================

experiment_results = []


# ============================================================
# 13. PROCESS DATASET
# ============================================================

print("\nStarting OS-CFAR area experiment...\n")


for area_index, minimum_area in enumerate(
    AREA_VALUES,
    start=1
):

    print("\n")
    print("=" * 60)

    print(
        f"Testing minimum component area = "
        f"{minimum_area}"
    )

    print("=" * 60)


    total_tp = 0

    total_fp = 0

    total_fn = 0

    total_candidate_boxes = 0

    images_evaluated = 0


    # --------------------------------------------------------
    # Process every image
    # --------------------------------------------------------

    for image_index, image_info in enumerate(
        data["images"],
        start=1
    ):

        image_id = image_info["id"]

        file_name = image_info["file_name"]


        image_path = os.path.join(
            IMAGE_DIR,
            file_name
        )


        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )


        if image is None:

            print(
                f"WARNING: Could not load "
                f"{file_name}"
            )

            continue


        image = image.astype(
            np.float32
        )


        # ----------------------------------------------------
        # OS-CFAR
        # ----------------------------------------------------

        background_estimate = rank_filter(
            image,
            rank=filter_rank,
            footprint=footprint,
            mode="nearest"
        )


        threshold = (
            alpha
            * background_estimate
        )


        detection_map = (
            image > threshold
        ).astype(
            np.uint8
        ) * 255


        # Remove invalid image border

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


        # ----------------------------------------------------
        # Connected components
        # ----------------------------------------------------

        predictions = get_detection_boxes(
            detection_map,
            minimum_area
        )


        total_candidate_boxes += len(
            predictions
        )


        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        ground_truths = annotations_by_image.get(
            image_id,
            []
        )


        # ----------------------------------------------------
        # Match
        # ----------------------------------------------------

        tp, fp, fn = match_boxes(
            predictions,
            ground_truths,
            IOU_THRESHOLD
        )


        total_tp += tp

        total_fp += fp

        total_fn += fn


        images_evaluated += 1


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            image_index % 100 == 0
            or image_index == len(data["images"])
        ):

            print(
                f"Area {minimum_area}: "
                f"{image_index}/"
                f"{len(data['images'])} images | "
                f"TP={total_tp} | "
                f"FP={total_fp} | "
                f"FN={total_fn}"
            )


    # ========================================================
    # CALCULATE METRICS
    # ========================================================

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


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print("\n")
    print(
        f"Minimum area = {minimum_area}"
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


    # ========================================================
    # SAVE RESULT
    # ========================================================

    experiment_results.append(
        {
            "Min_Area": minimum_area,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "Candidate_Boxes": total_candidate_boxes,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
    )


# ============================================================
# 14. CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    experiment_results
)


print("\n")
print("=" * 70)
print("OS-CFAR COMPONENT AREA EXPERIMENT")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("=" * 70)


# ============================================================
# 15. SAVE CSV
# ============================================================

output_path = os.path.join(
    RESULTS_DIR,
    "os_cfar_area_experiment.csv"
)


results_df.to_csv(
    output_path,
    index=False
)


print(
    "\nResults saved to:"
)

print(
    output_path
)