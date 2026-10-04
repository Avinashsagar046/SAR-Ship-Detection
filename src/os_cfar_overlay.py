import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import math
import json


# ============================================================
# 1. PATHS
# ============================================================

IMAGE_PATH = r"D:\SAR_SHIP\DATASET\images\P0001_0_800_10190_10990.png"

ANNOTATION_PATH = r"D:\SAR_SHIP\DATASET\annotations\test2017.json"


# ============================================================
# 2. OS-CFAR PARAMETERS
# ============================================================

WINDOW_SIZE = 21
GUARD_SIZE = 5

PFA = 0.001

RANK_PERCENTILE = 80

CHUNK_ROWS = 50


# ============================================================
# 3. LOAD SAR IMAGE
# ============================================================

image = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

image = image.astype(np.float32)

height, width = image.shape

print("Image loaded successfully!")
print("Image shape:", image.shape)


# ============================================================
# 4. LOAD COCO ANNOTATIONS
# ============================================================

with open(
    ANNOTATION_PATH,
    "r"
) as file:

    data = json.load(file)


# Find the image entry using file name

image_info = None

for item in data["images"]:

    if item["file_name"] == IMAGE_PATH.split("\\")[-1]:

        image_info = item
        break


if image_info is None:

    raise ValueError(
        "Image was not found in the annotation file."
    )


image_id = image_info["id"]

print("Image ID:", image_id)


# Get annotations belonging to this image

ground_truth_boxes = []

for annotation in data["annotations"]:

    if annotation["image_id"] == image_id:

        ground_truth_boxes.append(
            annotation["bbox"]
        )


print(
    "Ground-truth ships:",
    len(ground_truth_boxes)
)


# ============================================================
# 5. CALCULATE TRAINING CELLS
# ============================================================

half_window = WINDOW_SIZE // 2
half_guard = GUARD_SIZE // 2

total_cells = WINDOW_SIZE * WINDOW_SIZE

guard_cells = GUARD_SIZE * GUARD_SIZE

training_cells = total_cells - guard_cells


rank = int(
    np.ceil(
        RANK_PERCENTILE / 100 * training_cells
    )
)


print("Training cells:", training_cells)
print("Selected rank:", rank)
print("PFA:", PFA)
print("Rank percentile:", RANK_PERCENTILE)


# ============================================================
# 6. CALCULATE OS-CFAR THRESHOLD FACTOR
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
    lambda a: log_pfa(a) - target_log_pfa,
    0,
    1000
)


print(
    "OS-CFAR threshold factor:",
    alpha
)


# ============================================================
# 7. CREATE TRAINING-CELL OFFSETS
# ============================================================

training_offsets = []


for dy in range(
    -half_window,
    half_window + 1
):

    for dx in range(
        -half_window,
        half_window + 1
    ):

        # Ignore guard cells

        if (
            abs(dy) <= half_guard
            and abs(dx) <= half_guard
        ):

            continue

        training_offsets.append(
            (dy, dx)
        )


print(
    "Training offsets:",
    len(training_offsets)
)


# ============================================================
# 8. CREATE DETECTION MAP
# ============================================================

detection_map = np.zeros(
    (height, width),
    dtype=np.uint8
)


y_start = half_window
y_end = height - half_window

x_start = half_window
x_end = width - half_window


# ============================================================
# 9. RUN OPTIMIZED OS-CFAR
# ============================================================

print("\nRunning OS-CFAR...")


for chunk_start in range(
    y_start,
    y_end,
    CHUNK_ROWS
):

    chunk_end = min(
        chunk_start + CHUNK_ROWS,
        y_end
    )

    chunk_height = chunk_end - chunk_start


    # --------------------------------------------------------
    # Collect training cells
    # --------------------------------------------------------

    training_stack = np.empty(
        (
            training_cells,
            chunk_height,
            x_end - x_start
        ),
        dtype=np.float32
    )


    for index, (dy, dx) in enumerate(
        training_offsets
    ):

        training_stack[index] = image[
            chunk_start + dy:
            chunk_end + dy,

            x_start + dx:
            x_end + dx
        ]


    # --------------------------------------------------------
    # Select order statistic
    # --------------------------------------------------------

    kth_values = np.partition(
        training_stack,
        rank - 1,
        axis=0
    )[rank - 1]


    # --------------------------------------------------------
    # Calculate threshold
    # --------------------------------------------------------

    threshold = alpha * kth_values


    # --------------------------------------------------------
    # Get CUT pixels
    # --------------------------------------------------------

    cut = image[
        chunk_start:chunk_end,
        x_start:x_end
    ]


    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    detections = cut > threshold


    detection_map[
        chunk_start:chunk_end,
        x_start:x_end
    ][detections] = 255


    print(
        f"Processed rows "
        f"{chunk_start}-{chunk_end}"
    )


# ============================================================
# 10. FIND RAW DETECTION PIXELS
# ============================================================

detection_positions = np.argwhere(
    detection_map > 0
)


print("\n--------------------------------")
print(
    "Raw OS-CFAR detections:",
    len(detection_positions)
)
print("--------------------------------")


# ============================================================
# 11. CREATE RGB OVERLAY
# ============================================================

# Convert grayscale image to RGB

overlay = cv2.cvtColor(
    image.astype(np.uint8),
    cv2.COLOR_GRAY2RGB
)


# ============================================================
# 12. DRAW GROUND-TRUTH BOXES
# ============================================================

for bbox in ground_truth_boxes:

    x, y, w, h = bbox

    x1 = int(x)
    y1 = int(y)

    x2 = int(x + w)
    y2 = int(y + h)


    # Ground truth = green

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


# ============================================================
# 13. DRAW OS-CFAR DETECTIONS
# ============================================================

for y, x in detection_positions:

    # Draw a small red circle around
    # each detected pixel so it is visible.

    cv2.circle(
        overlay,
        (int(x), int(y)),
        4,
        (255, 0, 0),
        -1
    )


# ============================================================
# 14. DISPLAY OVERLAY
# ============================================================

plt.figure(
    figsize=(10, 10)
)

plt.imshow(
    overlay
)

plt.title(
    "OS-CFAR Detections vs Ground Truth"
)

plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# 15. PRINT DETECTION COORDINATES
# ============================================================

print("\nDetection coordinates:")

for index, (y, x) in enumerate(
    detection_positions,
    start=1
):

    print(
        f"{index}. x={x}, y={y}"
    )


print("\nOS-CFAR overlay completed.")