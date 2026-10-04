from pathlib import Path
import json
import csv
import numpy as np

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\SAR_SHIP")

ANNOTATION_FILE = (
    PROJECT_DIR
    / "DATASET"
    / "annotations"
    / "test2017.json"
)

OUTPUT_DIR = PROJECT_DIR / "results" / "YOLO"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUTPUT_DIR / "target_area_distribution.csv"


# ============================================================
# LOAD COCO ANNOTATIONS
# ============================================================

print("=" * 70)
print("HRSID TARGET-AREA DISTRIBUTION ANALYSIS")
print("=" * 70)

print("\nLoading:")
print(ANNOTATION_FILE)

with open(ANNOTATION_FILE, "r") as file:
    data = json.load(file)

annotations = data["annotations"]

print(f"Annotations found: {len(annotations)}")


# ============================================================
# EXTRACT AREAS
# ============================================================

areas = []

for annotation in annotations:

    # COCO bbox:
    # [x, y, width, height]

    bbox = annotation["bbox"]

    width = bbox[2]
    height = bbox[3]

    area = width * height

    areas.append(area)


areas = np.array(areas)

print("\nBasic statistics")
print("-" * 70)

print(f"Count   : {len(areas)}")
print(f"Minimum : {areas.min():.2f}")
print(f"Maximum : {areas.max():.2f}")
print(f"Mean    : {areas.mean():.2f}")
print(f"Median  : {np.median(areas):.2f}")
print(f"Std     : {areas.std():.2f}")


# ============================================================
# PERCENTILES
# ============================================================

percentiles = [
    1,
    5,
    10,
    25,
    50,
    75,
    90,
    95,
    99
]

print("\nPercentiles")
print("-" * 70)

for p in percentiles:

    value = np.percentile(areas, p)

    print(
        f"{p:>2}th percentile : {value:10.2f} px²"
    )


# ============================================================
# FIXED AREA RANGES
# ============================================================

bins = [
    ("<256", 0, 256),
    ("256-512", 256, 512),
    ("512-1024", 512, 1024),
    ("1024-2048", 1024, 2048),
    ("2048-4096", 2048, 4096),
    ("4096-8192", 4096, 8192),
    ("8192-16384", 8192, 16384),
    (">=16384", 16384, float("inf"))
]

print("\nFixed area distribution")
print("-" * 70)

distribution = []

for name, lower, upper in bins:

    if upper == float("inf"):

        count = np.sum(
            areas >= lower
        )

    else:

        count = np.sum(
            (areas >= lower) &
            (areas < upper)
        )

    percentage = (
        count / len(areas) * 100
    )

    print(
        f"{name:12s} : "
        f"{count:5d} ships "
        f"({percentage:6.2f}%)"
    )

    distribution.append([
        name,
        lower,
        upper if upper != float("inf") else "inf",
        int(count),
        percentage
    ])


# ============================================================
# OUR CURRENT EXPERIMENTAL BINS
# ============================================================

print("\nCurrent experimental bins")
print("-" * 70)

current_bins = [
    ("Small", 0, 1024),
    ("Medium", 1024, 10000),
    ("Large", 10000, float("inf"))
]

current_distribution = []

for name, lower, upper in current_bins:

    if upper == float("inf"):

        count = np.sum(
            areas >= lower
        )

    else:

        count = np.sum(
            (areas >= lower) &
            (areas < upper)
        )

    percentage = (
        count / len(areas) * 100
    )

    print(
        f"{name:8s} : "
        f"{count:5d} ships "
        f"({percentage:6.2f}%)"
    )

    current_distribution.append([
        name,
        lower,
        upper if upper != float("inf") else "inf",
        int(count),
        percentage
    ])


# ============================================================
# QUANTILE-BASED GROUPS
# ============================================================

print("\nQuantile-based groups")
print("-" * 70)

q33 = np.percentile(areas, 33.33)
q66 = np.percentile(areas, 66.67)

print(f"33.33th percentile : {q33:.2f} px²")
print(f"66.67th percentile : {q66:.2f} px²")

small_count = np.sum(areas < q33)

medium_count = np.sum(
    (areas >= q33) &
    (areas < q66)
)

large_count = np.sum(
    areas >= q66
)

print(
    f"Small  : {small_count}"
)

print(
    f"Medium : {medium_count}"
)

print(
    f"Large  : {large_count}"
)


# ============================================================
# SAVE CSV
# ============================================================

with open(CSV_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Range",
        "Lower_Area",
        "Upper_Area",
        "Count",
        "Percentage"
    ])

    for row in distribution:
        writer.writerow(row)

    writer.writerow([])

    writer.writerow([
        "Current experimental bins"
    ])

    for row in current_distribution:
        writer.writerow(row)


print("\nDistribution saved to:")
print(CSV_FILE)

print("=" * 70)