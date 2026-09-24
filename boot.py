# boot.py - AprilTag-Based Helmet Detection for Sipeed Maix Duino / Maix Bit (K210)
# ======================================================================
# Purpose:
#   This module provides a parallel, alternative approach to helmet detection
#   by scanning for physical AprilTags (TAG36H11 family) attached to helmets.
#   It operates independently of any KPU or kmodel neural networks.
#
# Wiring Notes (Maix Duino):
#   Buzzer -> IO24  (built-in buzzer on Maix Duino)
#   Red LED-> IO12  (GPIO warning indicator)
#   Camera -> OV2640 (DVP interface, configured for upright RGB565 QVGA)
#   LCD    -> 2.4" ILI9341 (built-in, 320x240)
# ======================================================================

import gc
import time
import math
import machine
import sensor
import image
import lcd
from Maix import FPIOA, GPIO
from machine import UART

# --- CONFIGURATION ────────────────────────────────────────────────────
PIN_BUZZER      = 24       # Buzzer GPIO Pin (Matches ML and QR project)
PIN_RED_LED     = 12       # Red LED GPIO Pin (maps to D10 header)
PIN_GREEN_LED   = 13       # Green LED GPIO Pin (maps to D9 header)
PIN_BLUE_LED    = 14       # Blue LED GPIO Pin (maps to D8 header)

# Timing Parameters for State Machine
CYCLE_INTERVAL_MS = 60000  # Duration of one full detection cycle (60 seconds)
SCAN_WINDOW_MS    = 30000  # Camera scanning window duration (30 seconds)
EARLY_PASS_MS     = 4000   # Early detection threshold for exit reference

# Camera Settings
# NOTE: MaixPy's find_apriltags() requires the total pixel resolution to be under 64K pixels (65,536 pixels).
# QVGA (320x240 = 76,800 pixels) will fail.
# We use HQVGA (240x160 = 38,400 pixels) in GRAYSCALE for sharper detection edges.
FRAME_SIZE   = sensor.HQVGA  # 240x160 (38,400 pixels). Fallback if unsupported.
PIX_FORMAT   = sensor.GRAYSCALE

# --- GPIO INITIALIZATION ──────────────────────────────────────────────
def gpio_init():
    """Configure IO pins for warning buzzer and status LEDs."""
    fpioa = FPIOA()
    fpioa.set_function(PIN_BUZZER,    FPIOA.GPIOHS0)
    fpioa.set_function(PIN_RED_LED,   FPIOA.GPIOHS1)
    fpioa.set_function(PIN_GREEN_LED, FPIOA.GPIOHS2)
    fpioa.set_function(PIN_BLUE_LED,  FPIOA.GPIOHS3)

    buzzer_gpio = GPIO(GPIO.GPIOHS0, GPIO.OUT)
    red_gpio    = GPIO(GPIO.GPIOHS1, GPIO.OUT)
    green_gpio  = GPIO(GPIO.GPIOHS2, GPIO.OUT)
    blue_gpio   = GPIO(GPIO.GPIOHS3, GPIO.OUT)

    # Set initial LED/Buzzer values
    buzzer_gpio.value(0)
    red_gpio.value(0)     
    green_gpio.value(0)   
    blue_gpio.value(0)    
    
    return buzzer_gpio, red_gpio, green_gpio, blue_gpio

# --- UART INITIALIZATION ----------------------------------------------
def uart_init():
    fpioa = FPIOA()
    # K210 IO6 connects to ESP32 U0TX (GPIO1) -> K210 RX
    fpioa.set_function(6, FPIOA.UART1_RX)
    # K210 IO7 connects to ESP32 U0RX (GPIO3) -> K210 TX
    fpioa.set_function(7, FPIOA.UART1_TX)
    # Set baudrate to 115200 to match onboard ESP32 UART0 config
    uart = UART(UART.UART1, 115200, 8, None, 1, timeout=1000, read_buf_len=64)
    return uart

# --- SEND HELMET STATUS -----------------------------------------------
last_uart_state = None

def send_helmet_signal(uart, helmet, detected_now, tag_id, cx, cy):
    global last_uart_state
    
    # 1. Format the UART payload in ASCII with newline, matching the ESP32 parser
    if helmet:
        if detected_now:
            payload = "TAG:1 ID:%d X:%d Y:%d\n" % (tag_id, cx, cy)
        else:
            payload = "TAG:1 ID:0 X:-1 Y:-1\n"
    else:
        payload = "TAG:0\n"
        
    # 2. Transmit over UART on every frame to prevent ESP32 heartbeat timeout
    try:
        uart.write(payload)
    except Exception as e:
        print("UART write error:", e)
        
    # 3. Print debug log to console only on state change to keep stdout clean
    if helmet != last_uart_state:
        if helmet:
            print("UART -> Helmet Detected")
        else:
            print("UART -> Helmet Not Detected")
        last_uart_state = helmet

# --- SENSOR & LCD INITIALIZATION ──────────────────────────────────────
def hardware_init():
    """Initialize LCD screen and OV2640 camera sensor."""
    # Initialize LCD display
    lcd.init(freq=15000000)
    lcd.clear(lcd.BLACK)
    lcd.draw_string(10, 10, "AprilTag Helmet System", lcd.WHITE, lcd.BLACK)
    lcd.draw_string(10, 30, "Initializing hardware...", lcd.WHITE, lcd.BLACK)

    # Initialize OV2640 Camera Sensor with double buffering if supported
    try:
        sensor.reset(dual_buff=True)
    except Exception:
        sensor.reset()

    # Try target resolution & pixel format, fall back to QQVGA Grayscale if unsupported
    try:
        sensor.set_pixformat(PIX_FORMAT)
        sensor.set_framesize(FRAME_SIZE)
        print("Camera configured successfully (Format=%d, Size=%d)" % (PIX_FORMAT, FRAME_SIZE))
    except Exception as e:
        print("Failed to apply camera settings:", e)
        print("Falling back to QQVGA Grayscale (160x120)...")
        sensor.set_pixformat(sensor.GRAYSCALE)
        sensor.set_framesize(sensor.QQVGA)

    sensor.set_hmirror(0)                   # Disable horizontal mirroring (required for consistent reading)
    sensor.set_vflip(0)                     # Set vertical flip to 0 (default upright orientation)
    sensor.set_contrast(3)                  # Max contrast for sharpest black/white edge separation
    sensor.set_brightness(0)                # Standard brightness

    sensor.run(1)
    sensor.skip_frames(time=1000)           # Allow AWB/AEC algorithms to warm up

# --- ALERT BANNER DRAWING ─────────────────────────────────────────────
def draw_alert_banner(img, is_helmet_ok):
    """Draw status headers on screen based on stabilized state."""
    w = img.width()
    text_scale = 1.5 if w >= 240 else 1.0
    if is_helmet_ok:
        # Green success banner (drawn in white/gray if in GRAYSCALE mode)
        img.draw_rectangle(0, 0, w, 26, color=(0, 200, 0), fill=True)
        img.draw_string(10, 4, "HELMET DETECTED", color=lcd.WHITE, scale=text_scale)
    else:
        # Red warning banner (drawn in white/gray if in GRAYSCALE mode)
        img.draw_rectangle(0, 0, w, 26, color=(200, 0, 0), fill=True)
        img.draw_string(10, 4, "NO HELMET DETECTED", color=lcd.WHITE, scale=text_scale)

# --- PROCESS FRAME & DETECT APRILTAGS ──────────────────────────────────
def process_frame(img, roi=None):
    """Scan image for AprilTags, draw markers, return scan status, tag info, and tracking ROI."""
    if not hasattr(img, "find_apriltags"):
        w = img.width()
        h = img.height()
        img.draw_rectangle(0, h // 2 - 20, w, 40, color=(200, 0, 0), fill=True)
        img.draw_string(10, h // 2 - 10, "FW DOES NOT SUPPORT APRILTAGS!", color=lcd.WHITE, scale=1.0)
        return False, [], [], None

    tags = []
    try:
        # Core MicroPython/OpenMV AprilTag detection call.
        # Pass the region of interest (ROI) to significantly speed up search.
        if roi:
            tags = img.find_apriltags(roi)
        else:
            tags = img.find_apriltags()
    except Exception as e:
        print("AprilTag scan error:", e)

    is_found = len(tags) > 0
    tag_info = []
    next_roi = None

    if is_found:
        # Sort by bounding box area (width * height) in descending order to identify the primary tag
        sorted_tags = sorted(tags, key=lambda t: t.rect()[2] * t.rect()[3], reverse=True)
        primary_tag = sorted_tags[0]
        
        # Calculate next tracking ROI around the primary tag
        rx, ry, rw, rh = primary_tag.rect()
        w_img = img.width()
        h_img = img.height()
        
        # Pad the search window to allow for motion in the next frame (e.g. 40 pixels)
        pad = 40
        roi_x = max(0, rx - pad)
        roi_y = max(0, ry - pad)
        roi_w = min(w_img - roi_x, rw + 2 * pad)
        roi_h = min(h_img - roi_y, rh + 2 * pad)
        next_roi = (roi_x, roi_y, roi_w, roi_h)

        # Draw a subtle tracking boundary indicator on screen
        if roi:
            img.draw_rectangle(roi[0], roi[1], roi[2], roi[3], color=100, thickness=1)

        for idx, tag in enumerate(sorted_tags):
            rx_t, ry_t, rw_t, rh_t = tag.rect()
            tag_id = tag.id()

            # Retrieve rough distance/pose translation data if available
            tx, ty, tz = 0.0, 0.0, 0.0
            try:
                tx = tag.x_translation()
                ty = tag.y_translation()
                tz = tag.z_translation()
            except Exception:
                pass # Pose data may not be calculated without camera lens calibration parameters

            tag_info.append({
                "id": tag_id,
                "pose": (tx, ty, tz)
            })

            is_primary = (idx == 0)
            box_color = (0, 255, 0) if is_primary else (0, 150, 255)
            line_thickness = 3 if is_primary else 1

            # Draw bounding box
            img.draw_rectangle(rx_t, ry_t, rw_t, rh_t, color=box_color, thickness=line_thickness)

            # Overlay tag ID near the bounding box
            lbl = "ID: %d" % tag_id
            img.draw_string(rx_t, max(0, ry_t - 14), lbl, color=box_color, scale=1)

    return is_found, tags, tag_info, next_roi

# ======================================================================
# TIMING & STATE MACHINE MODEL (3-LED Setup):
#   - STATE_SCANNING (0): Represents the active scanning window (first 30s of 60s cycle).
#     - PIN_BLUE_LED + PIN_RED_LED are turned ON as a visual cue to "show AprilTag".
#     - A single successful frame detection transitions the machine early to
#       STATE_RESULT_LOCKED with a PASS result.
#     - If the 30s scan window expires with zero detections, the state transitions
#       to STATE_RESULT_LOCKED with a FAIL result.
#   - STATE_RESULT_LOCKED (1): Holds the cycle result output until the 60s cycle ends.
#     - PIN_BLUE_LED is turned OFF.
#     - PASS: PIN_GREEN_LED = ON, PIN_RED_LED = OFF, Warning Buzzer = OFF.
#     - FAIL: PIN_GREEN_LED = OFF, PIN_RED_LED = ON, Warning Buzzer = ON.
#     - Starting next cycle (60s mark): Returns to STATE_SCANNING, PIN_BLUE_LED + PIN_RED_LED
#       turn ON, and the Green/Buzzer status signals reset to OFF.
# ======================================================================
STATE_SCANNING      = 0
STATE_RESULT_LOCKED = 1

# --- MAIN RUNTIME LOOP ────────────────────────────────────────────────
def main():
    # Setup alert outputs
    buzzer, red_led, green_led, blue_led = gpio_init()
    uart = uart_init()

    # Initialize camera & LCD
    hardware_init()

    print("AprilTag Helmet Detection Loop Started...")
    lcd.clear(lcd.BLACK)

    clock = time.clock()
    frame_count = 0
    last_printed_helmet = None
    tracking_roi = None

    # State Machine Initialization
    current_state = STATE_SCANNING
    cycle_start_time = time.ticks_ms()
    helmet_detected_state = False
    
    # Start cycle by scanning (Blue + Red ON during detection window)
    blue_led.value(1)
    red_led.value(1)
    buzzer.value(0)
    green_led.value(0)

    while True:
        clock.tick()
        frame_count += 1

        # Periodic garbage collection to maintain memory integrity
        if frame_count % 30 == 0:
            gc.collect()

        img = None
        try:
            img = sensor.snapshot()
        except Exception as e:
            print("Snapshot error:", e)
            time.sleep_ms(50)
            continue

        if img is None:
            continue

        # Evaluate timing in the current cycle
        now = time.ticks_ms()
        elapsed_in_cycle = time.ticks_diff(now, cycle_start_time)

        detected_now = False
        tags = []
        tag_info = []

        # --- STATE MACHINE EVALUATION ──────────────────────────────────
        if current_state == STATE_SCANNING:
            # Active scanning phase: capture frame and check for tags
            detected_now, tags, tag_info, tracking_roi = process_frame(img, tracking_roi)

            # Check PASS condition (Lenient: single frame is sufficient)
            if detected_now:
                helmet_detected_state = True
                current_state = STATE_RESULT_LOCKED
                
                # Turn OFF scanning indicators
                blue_led.value(0)
                red_led.value(0)
                
                # Set outputs for PASS (Green ON, Red OFF, Buzzer OFF)
                buzzer.value(0)
                green_led.value(1)
                
                print("Result locked: PASS (Tag detected during scan window)")
            
            # Check FAIL condition (30s timeout elapsed with zero detections)
            elif elapsed_in_cycle >= SCAN_WINDOW_MS:
                helmet_detected_state = False
                current_state = STATE_RESULT_LOCKED
                
                # Turn OFF scanning indicator
                blue_led.value(0)
                
                # Set outputs for FAIL (Green OFF, Red ON, Buzzer ON)
                buzzer.value(1)
                green_led.value(0)
                red_led.value(1)
                
                print("Result locked: FAIL (30s scan window expired with no detections)")

        elif current_state == STATE_RESULT_LOCKED:
            # Locked phase: hold states and wait for the remaining cycle duration (60s)
            if elapsed_in_cycle >= CYCLE_INTERVAL_MS:
                # Start new cycle
                cycle_start_time = now
                current_state = STATE_SCANNING
                
                # Turn ON scanning indicators (Blue + Red during detection window)
                blue_led.value(1)
                red_led.value(1)
                
                # Turn OFF outputs during active scanning window
                buzzer.value(0)
                green_led.value(0)
                
                print("New detection cycle started. Scanning...")

        # Flush the UART RX buffer to prevent buffer build-up
        if uart.any() > 0:
            try:
                uart.read(uart.any())
            except Exception:
                pass

        # Extract primary tag coordinates/ID if detected (for UART report)
        primary_id = -1
        primary_cx = -1
        primary_cy = -1
        if detected_now and len(tags) > 0:
            sorted_tags = sorted(tags, key=lambda t: t.rect()[2] * t.rect()[3], reverse=True)
            primary_tag = sorted_tags[0]
            rx, ry, rw, rh = primary_tag.rect()
            primary_id = primary_tag.id()
            primary_cx = rx + (rw // 2)
            primary_cy = ry + (rh // 2)

        # Send helmet status to onboard ESP32 (including coordinates/ID)
        send_helmet_signal(uart, helmet_detected_state, detected_now, primary_id, primary_cx, primary_cy)

        # Draw green or red headers on the LCD frame
        draw_alert_banner(img, helmet_detected_state)

        # Render FPS in the top right corner
        fps = clock.fps()
        fps_scale = 1.0 if img.width() < 240 else 1.2
        fps_offset = 65 if img.width() < 240 else 75
        img.draw_string(img.width() - fps_offset, 4, "%.1f FPS" % fps, color=lcd.YELLOW, scale=fps_scale)

        # Display frame buffer to LCD screen
        lcd.display(img)

        # Log status only when the helmet detection state changes to keep console clean
        if helmet_detected_state != last_printed_helmet:
            tag_ids = [info["id"] for info in tag_info]
            pose_str = "None"
            if len(tag_info) > 0:
                p = tag_info[0]["pose"]
                pose_str = "x=%.2f,y=%.2f,z=%.2f" % (p[0], p[1], p[2])
            print("FPS=%.1f  helmet=%s  tags_count=%d  ids=%s  lead_pose=%s" %
                  (fps, helmet_detected_state, len(tags), str(tag_ids), pose_str))
            last_printed_helmet = helmet_detected_state

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal runtime exception:", e)
        # Safely reset alert signals on crash
        try:
            buz = GPIO(GPIO.GPIOHS0, GPIO.OUT)
            r = GPIO(GPIO.GPIOHS1, GPIO.OUT)
            g = GPIO(GPIO.GPIOHS2, GPIO.OUT)
            buz.value(0)
            r.value(0)
            g.value(0)
        except:
            pass
