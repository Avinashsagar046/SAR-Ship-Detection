import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import math


# ============================================================
# 1. IMAGE PATH
# ============================================================

IMAGE_PATH = r"D:\SAR_SHIP\DATASET\images\P0001_0_800_10190_10990.png"


# ============================================================
# 2. OS-CFAR PARAMETERS
# ============================================================

WINDOW_SIZE = 21
GUARD_SIZE = 5

PFA = 0.001

# Percentage of training cells used to select
# the order statistic.
RANK_PERCENTILE = 80

# Number of image rows processed at a time.
CHUNK_ROWS = 50


# ============================================================
# 3. LOAD IMAGE
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
# 4. CALCULATE TRAINING CELLS
# ============================================================

half_window = WINDOW_SIZE // 2
half_guard = GUARD_SIZE // 2

total_cells = WINDOW_SIZE * WINDOW_SIZE
guard_cells = GUARD_SIZE * GUARD_SIZE

training_cells = total_cells - guard_cells

# Calculate the selected order-statistic rank.
#
# For 416 training cells:
# 80% = 332.8
# ceil(332.8) = 333

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
# 5. CALCULATE OS-CFAR THRESHOLD FACTOR
# ============================================================

# Assuming exponentially distributed SAR intensity:
#
# Pfa = Product [j / (j + alpha)]
#
# N = number of training cells
# k = selected order statistic


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
# 6. CREATE TRAINING-CELL OFFSETS
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

        # Skip the guard region.
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
# 7. CREATE OUTPUT DETECTION MAP
# ============================================================

detection_map = np.zeros(
    (height, width),
    dtype=np.uint8
)


# Valid region where a complete CFAR window exists.

y_start = half_window
y_end = height - half_window

x_start = half_window
x_end = width - half_window


# ============================================================
# 8. OPTIMIZED OS-CFAR
# ============================================================

print("\nRunning optimized OS-CFAR...")


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
    # 8.1 Collect training cells
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
    # 8.2 Find kth smallest training-cell value
    # --------------------------------------------------------

    # np.partition is faster than completely sorting
    # all 416 training-cell values.

    kth_values = np.partition(
        training_stack,
        rank - 1,
        axis=0
    )[rank - 1]


    # --------------------------------------------------------
    # 8.3 Calculate OS-CFAR threshold
    # --------------------------------------------------------

    threshold = alpha * kth_values


    # --------------------------------------------------------
    # 8.4 Get CUT pixels
    # --------------------------------------------------------

    cut = image[
        chunk_start:chunk_end,
        x_start:x_end
    ]


    # --------------------------------------------------------
    # 8.5 Diagnostic information
    # --------------------------------------------------------

    ratio = cut / (
        kth_values + 1e-6
    )

    max_ratio = np.max(ratio)

    max_position = np.unravel_index(
        np.argmax(ratio),
        ratio.shape
    )

    print(
        "Maximum CUT / OS-background ratio:",
        max_ratio
    )

    print(
        "Position of maximum ratio:",
        max_position
    )

    print(
        "CUT value at maximum:",
        cut[max_position]
    )

    print(
        "OS background estimate at maximum:",
        kth_values[max_position]
    )

    print(
        "Threshold at maximum:",
        threshold[max_position]
    )


    # --------------------------------------------------------
    # 8.6 Detection decision
    # --------------------------------------------------------

    detections = cut > threshold


    # Store detected pixels.

    detection_map[
        chunk_start:chunk_end,
        x_start:x_end
    ][detections] = 255


    print(
        f"Processed rows "
        f"{chunk_start}-{chunk_end}"
    )


# ============================================================
# 9. CHECK RAW DETECTIONS
# ============================================================

raw_detected_pixels = np.sum(
    detection_map > 0
)

print(
    "\nRaw OS-CFAR detected pixels:",
    raw_detected_pixels
)


# ============================================================
# 10. DISPLAY RAW OS-CFAR RESULT
# ============================================================

plt.figure(
    figsize=(12, 5)
)


# ------------------------------------------------------------
# Original SAR image
# ------------------------------------------------------------

plt.subplot(
    1,
    2,
    1
)

plt.imshow(
    image,
    cmap="gray"
)

plt.title(
    "Original SAR Image"
)

plt.axis("off")


# ------------------------------------------------------------
# Raw OS-CFAR detection map
# ------------------------------------------------------------

plt.subplot(
    1,
    2,
    2
)

plt.imshow(
    detection_map,
    cmap="gray"
)

plt.title(
    "Raw OS-CFAR Detection Map"
)

plt.axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# 11. COMPLETION MESSAGE
# ============================================================

print("\n--------------------------------")
print("OS-CFAR single-image test completed!")
print(
    "Raw detected pixels:",
    raw_detected_pixels
)
print("--------------------------------")