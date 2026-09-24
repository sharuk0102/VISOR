# Sipeed Maixduino (K210) YOLOv3-Lite Model Deployment Guide

This guide details the procedure for flashing, installing, and running the single-class YOLOv3-Lite helmet detection system (`no_helmet`) on the Sipeed Maixduino (K210) edge board.

---

## 1. Prerequisites & Toolchain Setup

To program and deploy the model onto the K210, you need the following utilities:
1.  **kflash_gui**: Graphical flashing tool for K210 chips. Download from [github.com/sipeed/kflash_gui/releases](https://github.com/sipeed/kflash_gui/releases).
2.  **MaixPy IDE**: Graphical development environment for compiling and debugging MicroPython scripts on the K210. Download from [dl.sipeed.com](https://dl.sipeed.com/share/maixpy/tools/maixpy-ide).
3.  **MicroPython Firmware**: Standard MaixPy bin file (minimum v0.5.0, v0.6.2 recommended). Use a firmware build that includes the `KPU` module (such as `maixpy_v0.6.2_84_g8fcd84a_minimum.bin` or `maixpy_v0.6.2_84_g8fcd84a.bin`).

---

## 2. Hardware Connections & Wiring

Ensure your Maixduino board is wired as follows:

| Periph / Component | K210 Pin | Description |
| :--- | :---: | :--- |
| **Buzzer** | `IO24` | Digital output warning indicator |
| **Red LED** | `IO12` | Bounding box alarm status indicator (D10 header) |
| **Green LED** | `IO13` | Normal/Safe status indicator (D9 header) |
| **DVP Camera** | DVP Socket | OV2640 / OV7740 camera module |
| **LCD Display** | SPI Socket | 2.4-inch $320 \times 240$ SPI LCD panel |
| **UART Connection** | `IO6` (RX), `IO7` (TX) | Serial connection to vehicle Receiver (ESP32) |

---

## 3. Deployment Method A: Micro SD Card (Recommended for Quick Iteration)

This is the easiest and fastest way to deploy the model and scripts:

1.  **Format SD Card**: Format a Micro SD card (up to 32GB) to the **FAT32** file system. (Cards formatted as exFAT or NTFS will not load).
2.  **Copy Model**: Copy the compiled `yolov3_lite.kmodel` from `deploy/` to the root folder of the SD card.
3.  **Copy Script**: Rename [boot_nn.py](file:///c:/Users/sharu/many%20helmet%20to%20one%20helmet/deploy/boot_nn.py) to `boot.py` (or `main.py`) and copy it to the root of the SD card.
4.  **Insert and Power On**: Insert the SD card into the Maixduino TF slot and power up the board via USB or power supply. The board will automatically execute `boot.py` and start the helmet detector loop.

---

## 4. Deployment Method B: Flash Memory (Recommended for Permanent Deployment)

For a standalone embedded system without an SD card, you can burn both the MicroPython firmware, script, and model directly to the K210's SPI Flash:

### Step 4.1: Model Address Layout
We map our model to the K210 flash address space:
*   **MaixPy Firmware**: Flashed to address `0x000000` (First 2MB).
*   **YOLOv3-Lite model (`yolov3_lite.kmodel`)**: Flashed to address `0x300000` (3MB offset, allowing up to 3MB of firmware size).
*   **Startup Script (`boot.py`)**: Copied to the K210's internal flash-based file system (`/flash`) using MaixPy IDE or uPyLoader.

### Step 4.2: Packaging into a `.kfpkg`
To flash both the binary executable (firmware) and the model in a single step:
1.  Open the **kflash_gui** utility.
2.  Add a file entry: Select the MaixPy firmware `.bin` file and set the address to `0x000000`.
3.  Add another file entry: Select `yolov3_lite.kmodel` and set the address to `0x300000`.
4.  Click **Save kfpkg** to package them into a single file (`helmet_interlock.kfpkg`).

### Step 4.3: Flashing the Board
1.  Connect the Maixduino board to your PC via a USB Type-C cable.
2.  Select the packaged `helmet_interlock.kfpkg` file in `kflash_gui`.
3.  Choose the correct COM port, set the board type to **Sipeed Maixduino**, and set the baud rate to **1500000** (for fast flashing).
4.  Click **Download** to burn the package to the board's flash memory.

### Step 4.4: Modifying the Code for Flash Address
If you flash the model to `0x300000` instead of reading from the SD card, you must update the load statement in `boot.py`:

```diff
-task = kpu.load("/sd/yolov3_lite.kmodel")
+task = kpu.load(0x300000)
```

Use MaixPy IDE to open the board's internal `boot.py` file (located under `/flash/boot.py`), make this change, and save.

---

## 5. Serial Telemetry Protocol

The K210 outputs status telemetry at **115,200 baud** over pins `IO6`/`IO7` for integration with external receivers (such as the vehicle ignition interlock receiver):

*   **When a Bare Head (No Helmet) is Detected**:
    `TAG:0\n`
    *(Instructs the vehicle receiver to restrict speed limits or block ignition).*
*   **When a Helmet is Detected / Compliant State**:
    `TAG:1 ID:0 X:-1 Y:-1\n`
    *(Unlocks full speed limits. Replicates the AprilTag format with default placeholders for seamless backward compatibility).*
