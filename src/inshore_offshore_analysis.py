from pathlib import Path
from ultralytics import YOLO
import cv2
import json
import csv

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\SAR_SHIP")

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "yolo11n_HRSID"
    / "best.pt"
)

IMAGE_DIR = (
    PROJECT_DIR
    / "yolo_dataset"
    / "images"
    / "test"
)

LABEL_DIR = (
    PROJECT_DIR
    / "yolo_dataset"
    / "labels"
    / "test"
)

INSHORE_JSON = (
    PROJECT_DIR
    / "DATASET"
    / "inshore_offshore"
    / "inshore.json"
)

OFFSHORE_JSON = (
    PROJECT_DIR
    / "DATASET"
    / "inshore_offshore"
    / "offshore.json"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "YOLO"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "inshore_offshore_analysis.csv"
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 800

# Best tested threshold from our confidence experiment
CONF_THRESHOLD = 0.40

IOU_THRESHOLD = 0.50


# ============================================================
# IoU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0])
        *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        *
        max(0, box2[3] - box2[1])
    )

    union_area = (
        area1
        + area2
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


# ============================================================
# READ GROUND TRUTH
# ============================================================

def read_ground_truth(
    label_path,
    image_width,
    image_height
):

    boxes = []

    if not label_path.exists():
        return boxes

    with open(
        label_path,
        "r"
    ) as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            _, xc, yc, w, h = map(
                float,
                values
            )

            # Convert normalized YOLO coordinates
            # into pixel coordinates.

            xc *= image_width
            yc *= image_height
            w *= image_width
            h *= image_height

            x1 = xc - w / 2
            y1 = yc - h / 2

            x2 = xc + w / 2
            y2 = yc + h / 2

            boxes.append([
                x1,
                y1,
                x2,
                y2
            ])

    return boxes


# ============================================================
# LOAD SCENE INFORMATION
# ============================================================

print("=" * 70)
print("HRSID INSHORE vs OFFSHORE ANALYSIS")
print("=" * 70)

print("\nLoading scene annotations...")

with open(
    INSHORE_JSON,
    "r",
    encoding="utf-8"
) as file:

    inshore_data = json.load(file)


with open(
    OFFSHORE_JSON,
    "r",
    encoding="utf-8"
) as file:

    offshore_data = json.load(file)


# Create filename → scene mapping

scene_map = {}


for image in inshore_data["images"]:

    filename = image["file_name"]

    scene_map[filename] = "Inshore"


for image in offshore_data["images"]:

    filename = image["file_name"]

    if filename in scene_map:

        print(
            "WARNING: Image appears in both "
            f"categories: {filename}"
        )

    scene_map[filename] = "Offshore"


print(
    f"Inshore images in JSON : "
    f"{sum(1 for x in scene_map.values() if x == 'Inshore')}"
)

print(
    f"Offshore images in JSON: "
    f"{sum(1 for x in scene_map.values() if x == 'Offshore')}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLO11n...")

model = YOLO(
    str(MODEL_PATH)
)

print("Model loaded successfully.")


# ============================================================
# TEST IMAGES
# ============================================================

image_paths = sorted(
    list(IMAGE_DIR.glob("*.png"))
    +
    list(IMAGE_DIR.glob("*.jpg"))
    +
    list(IMAGE_DIR.glob("*.jpeg"))
)

print(
    f"\nTest images found: "
    f"{len(image_paths)}"
)


# ============================================================
# STATISTICS
# ============================================================

stats = {

    "Inshore": {
        "images": 0,
        "gt": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0
    },

    "Offshore": {
        "images": 0,
        "gt": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0
    }
}


unclassified = []


# ============================================================
# EVALUATION
# ============================================================

for index, image_path in enumerate(
    image_paths,
    start=1
):

    filename = image_path.name

    # --------------------------------------------------------
    # Determine scene
    # --------------------------------------------------------

    scene = scene_map.get(filename)

    if scene is None:

        unclassified.append(filename)

        continue

    stats[scene]["images"] += 1

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        continue

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    label_path = (
        LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    gt_boxes = read_ground_truth(
        label_path,
        width,
        height
    )

    stats[scene]["gt"] += len(
        gt_boxes
    )

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

    if (
        result.boxes is not None
        and len(result.boxes) > 0
    ):

        boxes = (
            result
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )

        confidences = (
            result
            .boxes
            .conf
            .cpu()
            .numpy()
        )

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

    # --------------------------------------------------------
    # Match predictions
    # --------------------------------------------------------

    matched_gt = set()

    tp = 0
    fp = 0

    for pred_box, confidence in predictions:

        best_iou = 0.0
        best_gt_index = -1

        for gt_index, gt_box in enumerate(
            gt_boxes
        ):

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

            matched_gt.add(
                best_gt_index
            )

        else:

            fp += 1

    fn = (
        len(gt_boxes)
        - len(matched_gt)
    )

    # --------------------------------------------------------
    # Accumulate
    # --------------------------------------------------------

    stats[scene]["tp"] += tp
    stats[scene]["fp"] += fp
    stats[scene]["fn"] += fn

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        index % 200 == 0
        or index == len(image_paths)
    ):

        print(
            f"Processed "
            f"{index}/{len(image_paths)}"
        )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("INSHORE vs OFFSHORE RESULTS")
print("=" * 70)


final_results = []


for scene in [
    "Inshore",
    "Offshore"
]:

    s = stats[scene]

    tp = s["tp"]
    fp = s["fp"]
    fn = s["fn"]

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0
    )

    print(
        f"\n{scene}"
    )

    print(
        f"Images    : {s['images']}"
    )

    print(
        f"GT ships  : {s['gt']}"
    )

    print(
        f"TP        : {tp}"
    )

    print(
        f"FP        : {fp}"
    )

    print(
        f"FN        : {fn}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    final_results.append([
        scene,
        s["images"],
        s["gt"],
        tp,
        fp,
        fn,
        precision,
        recall,
        f1
    ])


# ============================================================
# UNCLASSIFIED
# ============================================================

print(
    f"\nUnclassified test images: "
    f"{len(unclassified)}"
)

if unclassified:

    print(
        "\nFirst 20 unclassified:"
    )

    for filename in unclassified[:20]:

        print(
            filename
        )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Scene",
        "Images",
        "Ground_Truth",
        "TP",
        "FP",
        "FN",
        "Precision",
        "Recall",
        "F1"
    ])

    writer.writerows(
        final_results
    )


print(
    "\nResults saved to:"
)

print(
    OUTPUT_FILE
)

print("=" * 70)