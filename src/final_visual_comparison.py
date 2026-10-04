"""
Final visual comparison for SAR ship detection.

Creates one 2x2 figure using the SAME HRSID test image:
1. Ground Truth
2. CA-CFAR
3. OS-CFAR
4. YOLO11n

Final comparison settings:
CA-CFAR: 21x21 window, 5x5 guard, PFA=0.001, min area=10
OS-CFAR: 21x21 window, 5x5 guard, PFA=0.001, rank=80%, min area=20
YOLO11n: imgsz=800, confidence=0.40

Run from D:\\SAR_SHIP:
    python src\\final_visual_comparison.py
"""

from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import brentq
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_NAME = "P0001_0_800_7200_8000.png"
IMAGE_PATH = ROOT / "DATASET" / "images" / IMAGE_NAME
ANNOTATION_PATH = ROOT / "DATASET" / "annotations" / "test2017.json"
MODEL_PATH = ROOT / "models" / "yolo11n_HRSID" / "best.pt"

OUTPUT_DIR = ROOT / "results" / "visual_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "P0001_0_800_7200_8000_comparison.png"


# ============================================================
# FINAL PARAMETERS
# ============================================================

WINDOW_SIZE = 21
GUARD_SIZE = 5
PFA = 0.001

CA_MIN_AREA = 10

OS_RANK_PERCENTILE = 80
OS_MIN_AREA = 20

YOLO_CONF = 0.40
YOLO_IMGSZ = 800


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

print(f"Image: {IMAGE_NAME}")
print(f"Image shape: {image.shape}")


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

# Try exact filename first
image_record = next(
    (item for item in coco["images"]
     if item["file_name"] == IMAGE_NAME),
    None
)

# If the demo image is not in test2017.json, search all annotation
# JSON files for the same filename.
if image_record is None:

    annotation_files = [
        ROOT / "DATASET" / "annotations" / "train2017.json",
        ROOT / "DATASET" / "annotations" / "test2017.json",
        ROOT / "DATASET" / "annotations" / "train_test2017.json",
    ]

    for annotation_file in annotation_files:

        if not annotation_file.exists():
            continue

        with open(annotation_file, "r", encoding="utf-8") as f:
            temp_coco = json.load(f)

        image_record = next(
            (item for item in temp_coco.get("images", [])
             if item["file_name"] == IMAGE_NAME),
            None
        )

        if image_record is not None:
            coco = temp_coco
            print(f"Ground truth found in: {annotation_file.name}")
            break

if image_record is None:
    raise FileNotFoundError(
        f"Could not find {IMAGE_NAME} in the available COCO annotation files."
    )

image_id = image_record["id"]

gt_boxes = []

for ann in coco["annotations"]:
    if ann["image_id"] == image_id:
        x, y, w, h = ann["bbox"]

        gt_boxes.append([
            int(round(x)),
            int(round(y)),
            int(round(x + w)),
            int(round(y + h))
        ])

print(f"Image ID: {image_id}")
print(f"Ground-truth ships: {len(gt_boxes)}")


# ============================================================
# HELPERS
# ============================================================

def alpha_ca_cfar(n, pfa):
    return n * (pfa ** (-1.0 / n) - 1.0)


def make_training_mask(window_size, guard_size):
    half = window_size // 2
    mask = np.ones((window_size, window_size), dtype=np.uint8)

    center = window_size // 2
    g = guard_size // 2

    mask[
        center - g:center + g + 1,
        center - g:center + g + 1
    ] = 0

    return mask


def connected_component_boxes(binary, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8
    )

    boxes = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        boxes.append((x, y, x + w, y + h))

    return boxes


def draw_boxes(gray, boxes, title):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(
            rgb,
            (x1, y1),
            (x2, y2),
            (255, 80, 80),
            2
        )

    return rgb


# ============================================================
# CA-CFAR
# ============================================================

print("\nRunning CA-CFAR...")

half = WINDOW_SIZE // 2
guard_half = GUARD_SIZE // 2

training_mask = make_training_mask(WINDOW_SIZE, GUARD_SIZE)
N = int(training_mask.sum())

alpha = alpha_ca_cfar(N, PFA)

# Local sum of training cells
local_sum = ndimage.convolve(
    image.astype(np.float32),
    training_mask.astype(np.float32),
    mode="constant",
    cval=0
)

background = local_sum / N
threshold = alpha * background

binary_ca = (
    image.astype(np.float32) > threshold
).astype(np.uint8)

# Ignore border where the complete CFAR window is unavailable
binary_ca[:half, :] = 0
binary_ca[-half:, :] = 0
binary_ca[:, :half] = 0
binary_ca[:, -half:] = 0

ca_boxes = connected_component_boxes(binary_ca, CA_MIN_AREA)

print(f"Training cells: {N}")
print(f"Alpha: {alpha:.6f}")
print(f"CA-CFAR candidate boxes: {len(ca_boxes)}")


# ============================================================
# OS-CFAR
# ============================================================

print("\nRunning OS-CFAR...")

# Build the flattened rank filter.
# 416 training cells -> 80th percentile is one-based rank 333,
# which corresponds to zero-based rank 332 for scipy.
rank_index = 332

# Rank-filter background estimate
os_background = ndimage.rank_filter(
    image.astype(np.float32),
    rank=rank_index,
    size=WINDOW_SIZE,
    mode="constant",
    cval=0
)

# Exact factor obtained for the final OS-CFAR configuration.
os_alpha = 4.35377163833898

os_threshold = os_alpha * os_background

binary_os = (
    image.astype(np.float32) > os_threshold
).astype(np.uint8)

binary_os[:half, :] = 0
binary_os[-half:, :] = 0
binary_os[:, :half] = 0
binary_os[:, -half:] = 0

os_boxes = connected_component_boxes(binary_os, OS_MIN_AREA)

print(f"Training cells: {N}")
print(f"Rank: {rank_index + 1} / {N}")
print(f"OS-CFAR alpha: {os_alpha:.6f}")
print(f"OS-CFAR candidate boxes: {len(os_boxes)}")


# ============================================================
# YOLO11n
# ============================================================

print("\nRunning YOLO11n...")

model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=str(IMAGE_PATH),
    imgsz=YOLO_IMGSZ,
    conf=YOLO_CONF,
    verbose=False
)

yolo_boxes = []

result = results[0]

if result.boxes is not None:
    xyxy = result.boxes.xyxy.cpu().numpy()

    for box in xyxy:
        x1, y1, x2, y2 = box.astype(int)
        yolo_boxes.append((x1, y1, x2, y2))

print(f"YOLO11n detections @ conf={YOLO_CONF}: {len(yolo_boxes)}")


# ============================================================
# DRAW PANELS
# ============================================================

gt_panel = draw_boxes(image, gt_boxes, "Ground Truth")
ca_panel = draw_boxes(image, ca_boxes, "CA-CFAR")
os_panel = draw_boxes(image, os_boxes, "OS-CFAR")
yolo_panel = draw_boxes(image, yolo_boxes, "YOLO11n")


# Ground truth uses a different box appearance so it is visually
# distinguishable from detector predictions.
for x1, y1, x2, y2 in gt_boxes:
    cv2.rectangle(
        gt_panel,
        (x1, y1),
        (x2, y2),
        (80, 255, 80),
        2
    )


# ============================================================
# FINAL 2x2 FIGURE
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 14))

panels = [
    (gt_panel, f"Ground Truth ({len(gt_boxes)} ships)"),
    (ca_panel, f"CA-CFAR ({len(ca_boxes)} candidates)"),
    (os_panel, f"OS-CFAR ({len(os_boxes)} candidates)"),
    (yolo_panel, f"YOLO11n ({len(yolo_boxes)} detections, conf={YOLO_CONF})"),
]

for ax, (panel, title) in zip(axes.ravel(), panels):
    ax.imshow(panel)
    ax.set_title(title, fontsize=14)
    ax.axis("off")

fig.suptitle(
    "HRSID SAR Ship Detection — Qualitative Comparison",
    fontsize=18
)

plt.tight_layout()
plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\n========================================")
print("FINAL VISUAL COMPARISON COMPLETED")
print("========================================")
print(f"Saved to: {OUTPUT_PATH}")
print(f"CA-CFAR candidates : {len(ca_boxes)}")
print(f"OS-CFAR candidates : {len(os_boxes)}")
print(f"YOLO11n detections : {len(yolo_boxes)}")
print(f"Ground truth ships : {len(gt_boxes)}")
