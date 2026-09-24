# AprilTag-Based Helmet Detection Module

This directory contains a standalone module for detecting helmets using physical AprilTag markers. It is designed as an alternative or complementary system to both the neural-network-based (KPU) detector and the QR-code-based detector.

---

## Why AprilTag Over QR Code?

AprilTags offer significant advantages over traditional QR codes for embedded vision systems (such as the K210 with an OV2640 camera):

1. **Geometry-Based Detection**: AprilTag detection focuses on finding high-contrast quad shapes (the strong transition from the outer white border to the inner black border) rather than reading tiny individual modules. This makes AprilTags much easier for low-resolution/low-budget cameras to resolve at a distance.
2. **Robustness Under Lighting Varation**: AprilTag's edge-detection algorithm is far more resilient to uneven lighting, heavy shadowing, and glare.
3. **Wider Decoding Angles**: QR codes require clean alignment to resolve their finder patterns and data grid. AprilTags can be successfully localized and decoded at steep perspective tilt angles.
4. **Distance and Pose Estimation**: The mathematical structure of AprilTags allows the detection pipeline to estimate the tag's relative 3D pose (translation/rotation) relative to the camera, which is not easily computed from basic QR codes.

---

## How It Works

* **Tag Family**: The script is configured to look for the default MaixPy/OpenMV family: **TAG36H11**.
* **Resolution Restrictions**: MaixPy's built-in `find_apriltags()` function requires the total image size to be **under 64K pixels** (65,536 pixels). As a result, standard QVGA (320x240 = 76,800 pixels) will fail with a resolution limit error.
* **Optimized Resolution & Format**: To work within this hardware constraint, the script defaults to **HQVGA (240x160 = 38,400 pixels)** in **Grayscale (sensor.GRAYSCALE)** mode. Grayscale mode requires 50% less RAM, runs significantly faster, and improves overall FPS.
* **Camera Fallback**: If the camera module fails to configure to HQVGA, the script automatically falls back to **QQVGA (160x120 = 19.2K pixels)**.
* **Presence-Based Validation**: The presence of *any* valid TAG36H11 AprilTag is interpreted as a positive helmet detection.
* **Rolling Window Debouncing**: A history list of the last `N = 5` frames tracks the detection status. The system requires a majority of frames (at least `3 out of 5`) to change states. This eliminates flickering alarms due to riding vibrations.
* **Serial Debugging**: Prints frame-by-frame stats, tag ID, and 3D translation/pose parameters (`x_translation()`, `y_translation()`, `z_translation()`) if available.

---

## Comparison of Detector Types

| Dimension | ML-based Model (KPU) | QR-based Detector | AprilTag-based Detector |
|---|---|---|---|
| **CPU/RAM Overhead** | High (Requires KPU SRAM) | Very Low | Low-Medium (Slightly higher than QR, still very low) |
| **False Positives** | Low-Medium | **0%** (Cannot falsely trigger) | **0%** (Cannot falsely trigger) |
| **Angle Tolerance** | High | Low (Fails on steep angles) | **High** (Resolves up to steep angles) |
| **Lighting Tolerance** | High | Low (Highly sensitive to glare) | **Medium-High** (Resilient to glare/shadows) |
| **Distance Capability** | Depends on anchor box tuning | ~30-35cm (Requires large module print) | **~30-35cm+** (Highly resolvable at distance) |
| **Sticker Requirement**| None | Requires physical QR sticker | Requires physical AprilTag sticker |

---

## AprilTag Printing Specifications

To ensure high-precision scanning at standard motorcycle riding distances (approximately 30 to 35 cm from a handlebar-mounted camera):
1. **File**: Use the generated [helmet_apriltag.png](file:///c:/Users/sharu/jai%20shri%20ram/apriltag_helmet_detection/helmet_apriltag.png).
2. **Physical Size**: Print the tag to be approximately **4.5 cm × 4.5 cm** (calculates to 4.23 cm square when printed at 300 DPI: `500x500 pixels`).
3. **Scale**: Always print at **100% scale** (do NOT choose "Fit to Page" as it will distort dimensions).
4. **Contrast and Material**: Print with high contrast on **matte paper** or apply a **matte lamination**. **Avoid glossy materials** at all costs, as glossy surfaces reflect ambient light and create glare that blinds the camera sensor.
5. **Placement**: Stick the tag completely flat on the **front or forehead area of the helmet**. Avoid curved surfaces that distort the square geometry of the outer border.

---

## How to Run

1. **Generate Image**: Run `generate_apriltag.py` to create `helmet_apriltag.png` if it isn't already present.
2. **Format SD Card**: Make sure your micro SD card is formatted to FAT32.
3. **Copy to SD**: Copy `main_apriltag.py` and `helmet_apriltag.png` to the root folder of the SD card.
4. **Execute**:
   - **Direct Execution**: Connect the board to your PC, open MaixPy IDE, load `main_apriltag.py`, and press the **Play** button.
   - **Boot Standalone**: Rename `main_apriltag.py` to `main.py` (or `boot.py`) on the SD card so that the board launches the AprilTag detection pipeline automatically upon power-up.
