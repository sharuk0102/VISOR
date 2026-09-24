"""
sipeed/main.py  –  Smart Helmet Visual Interlock (Sipeed Maixduino / K210)
===========================================================================
Hardware : Sipeed Maixduino (K210 + onboard ESP32)
Runtime  : MicroPython / MaixPy firmware

Combines
────────
  1. Face Detection (YOLOv2 via KPU) + AprilTag Detection (TAG36H11)
     – Helmet WORN when face seen AND AprilTag (ID 0) is positioned
       below and within the horizontal span of the face bounding box.

  2. RGB LED + Buzzer state-machine (60 s cycle):
       STATE_SCANNING (0–30 s)        → Blue LED ON
       STATE_RESULT_LOCKED PASS       → Green LED ON
       STATE_RESULT_LOCKED FAIL       → Red LED ON + Buzzer ON
       New cycle starts at 60 s mark  → back to scanning (Blue ON)

  3. UART communication to onboard ESP32 (K210 IO7 → ESP32 GPIO3):
       Helmet WORN  → "TAG:1 ID:<id> X:<cx> Y:<cy>\\n"
       Held PASS    → "TAG:1 ID:0 X:-1 Y:-1\\n"
       No helmet    → "TAG:0\\n"
     The internal ESP32 parses these and relays via ESP-NOW to the
     receiver ESP32 (MAC A0:B7:65:47:67:7C).

Wiring (Maixduino)
──────────────────
  Buzzer   → IO24  (FPIOA.GPIOHS0)
  Red LED  → IO12  (FPIOA.GPIOHS1)  D10 header
  Green LED→ IO13  (FPIOA.GPIOHS2)  D9  header
  Blue LED → IO14  (FPIOA.GPIOHS3)  D8  header
  UART TX  → IO7   (FPIOA.UART1_TX) → ESP32 GPIO3 RX

Model
─────
  face.kmodel on SD card (place in the root of SD card as /sd/face.kmodel)
  Fallback: SPI flash address 0x300000
"""

import sensor
import image
import lcd
import KPU as kpu
import gc
import time
import math
import machine
from Maix import FPIOA, GPIO
from machine import UART

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# ── Face Detection Tuning ────────────────────────────────────────────────────
CONF_THRESHOLD  = 0.18   # Very low – catches face at any distance / angle
NMS_THRESHOLD   = 0.3    # Non-Maximum Suppression threshold
SMOOTHING       = 0.15   # Heavy EMA – box is extremely stable (0=max, 1=none)
MAX_LIFETIME    = 25     # Hold face box 25 frames after loss – eliminates blink-outs

# Eye-region crop and axis alignment
# EYE_CROP_RATIO: how much of the face box height to use as eye region.
#   Higher value = threshold line is lower = more room for tag below eyes.
EYE_CROP_RATIO  = 0.65   # Relaxed from 0.45 → threshold line now at mid-face
# AXIS_TOLERANCE: fraction of face width the tag center X must fall within.
#   1.0 = full width (no side restriction), 0.8 = inner 80% (strict)
AXIS_TOLERANCE  = 1.0    # Full width – only vertical position matters now

# Tag stability hold: once the tag passes the position check, hold the
# "tag detected" state for this many frames even if the tag blinks out.
# Eliminates rapid on/off flickering caused by AprilTag detection jitter.
TAG_HOLD_FRAMES = 15     # ~0.6 s at 25 fps – increase if still flickering

# YOLOv2 anchor values – must match training configuration of face.kmodel
ANCHORS = (1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658,
           5.155437, 6.92275, 6.718375, 9.01025)

# ── Camera / LCD Settings ────────────────────────────────────────────────────
CAMERA_VFLIP    = 0
CAMERA_HMIRROR  = 0
LCD_ROTATION    = 2

# ── GPIO Pin Assignments (Maixduino) ─────────────────────────────────────────
PIN_BUZZER      = 24   # IO24  Buzzer (built-in on Maixduino)
PIN_RED_LED     = 12   # IO12  Red LED   (D10 header)
PIN_GREEN_LED   = 13   # IO13  Green LED (D9  header)
PIN_BLUE_LED    = 14   # IO14  Blue LED  (D8  header)

# ── Timing Parameters for State Machine ──────────────────────────────────────
CYCLE_INTERVAL_MS = 60000  # Full cycle duration (60 seconds)
SCAN_WINDOW_MS    = 30000  # Active scanning window (30 seconds)

# ── State Machine States ──────────────────────────────────────────────────────
STATE_SCANNING      = 0
STATE_RESULT_LOCKED = 1

# ── Colours (RGB565 for MaixPy image drawing) ─────────────────────────────────
GREEN  = (0,   255, 0)
RED    = (255, 0,   0)
ORANGE = (255, 165, 0)
WHITE  = (255, 255, 255)
YELLOW = (255, 255, 0)
BLACK  = (0,   0,   0)

# ─────────────────────────────────────────────────────────────────────────────
# 2. GPIO INITIALISATION (Buzzer + RGB LEDs)
# ─────────────────────────────────────────────────────────────────────────────
def gpio_init():
    """Configure IO pins for buzzer and RGB status LEDs."""
    fpioa = FPIOA()
    fpioa.set_function(PIN_BUZZER,    FPIOA.GPIOHS0)
    fpioa.set_function(PIN_RED_LED,   FPIOA.GPIOHS1)
    fpioa.set_function(PIN_GREEN_LED, FPIOA.GPIOHS2)
    fpioa.set_function(PIN_BLUE_LED,  FPIOA.GPIOHS3)

    buzzer_gpio = GPIO(GPIO.GPIOHS0, GPIO.OUT)
    red_gpio    = GPIO(GPIO.GPIOHS1, GPIO.OUT)
    green_gpio  = GPIO(GPIO.GPIOHS2, GPIO.OUT)
    blue_gpio   = GPIO(GPIO.GPIOHS3, GPIO.OUT)

    # All outputs start LOW (off)
    buzzer_gpio.value(0)
    red_gpio.value(0)
    green_gpio.value(0)
    blue_gpio.value(0)

    return buzzer_gpio, red_gpio, green_gpio, blue_gpio

# ─────────────────────────────────────────────────────────────────────────────
# 3. UART INITIALISATION (K210 IO7 → Internal ESP32 GPIO3)
# ─────────────────────────────────────────────────────────────────────────────
def uart_init():
    """
    Initialise UART1 on the K210.
    IO6 = UART1_RX  (receives from ESP32 GPIO1, used for SEND_OK/SEND_FAIL acks)
    IO7 = UART1_TX  (transmits to ESP32 GPIO3 – the primary data channel)
    Baud rate 115200 matches the internal ESP32 Serial2 configuration.
    """
    fpioa = FPIOA()
    fpioa.set_function(6, FPIOA.UART1_RX)
    fpioa.set_function(7, FPIOA.UART1_TX)
    uart = UART(UART.UART1, 115200, 8, None, 1, timeout=1000, read_buf_len=64)
    return uart

# ─────────────────────────────────────────────────────────────────────────────
# 4. UART PAYLOAD SENDER
# ─────────────────────────────────────────────────────────────────────────────
_last_uart_state = None

def send_helmet_signal(uart, helmet, detected_now, tag_id, cx, cy):
    """
    Transmit helmet status to the internal ESP32 over UART every frame.

    Payload formats (parsed by internal_esp32.ino parseK210Line()):
      TAG:1 ID:<id> X:<cx> Y:<cy>\\n   – helmet worn, tag seen this frame
      TAG:1 ID:0 X:-1 Y:-1\\n           – helmet worn (state held), no live tag
      TAG:0\\n                           – helmet not worn
    """
    global _last_uart_state

    if helmet:
        if detected_now:
            payload = "TAG:1 ID:%d X:%d Y:%d\n" % (tag_id, cx, cy)
        else:
            payload = "TAG:1 ID:0 X:-1 Y:-1\n"
    else:
        payload = "TAG:0\n"

    try:
        uart.write(payload)
    except Exception as e:
        print("UART write error:", e)

    # Log only on state change to keep console clean
    if helmet != _last_uart_state:
        print("UART →", "Helmet Detected" if helmet else "Helmet Not Detected")
        _last_uart_state = helmet

    # Flush any ACK bytes sent back from the ESP32 (SEND_OK / SEND_FAIL)
    if uart.any() > 0:
        try:
            uart.read(uart.any())
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# 5. LCD & CAMERA INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
def hardware_init():
    """Initialise LCD and OV2640 camera in RGB565 QVGA mode for face detection."""
    lcd.init(freq=15000000)
    lcd.rotation(LCD_ROTATION)
    lcd.clear(lcd.BLACK)
    lcd.draw_string(10, 10, "Smart Helmet System",    lcd.WHITE, lcd.BLACK)
    lcd.draw_string(10, 30, "Initialising camera...", lcd.WHITE, lcd.BLACK)

    try:
        sensor.reset()   # single buffer – saves 150KB pool for find_apriltags
    except Exception:
        sensor.reset()

    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)       # 320 × 240
    sensor.set_hmirror(CAMERA_HMIRROR)
    sensor.set_vflip(CAMERA_VFLIP)
    # Low-light optimisation (from reference project)
    sensor.set_contrast(2)              # Max contrast – sharpens edges in dark frames
    sensor.set_brightness(1)            # Slight brightness boost
    sensor.set_auto_gain(True)
    sensor.set_gainceiling(8)           # Allow up to 8x gain (helps in low light)
    sensor.set_auto_exposure(True)
    sensor.set_auto_whitebal(True)
    sensor.run(1)
    sensor.skip_frames(time=1000)       # Let AWB/AEC warm up

# ─────────────────────────────────────────────────────────────────────────────
# 6. KPU MODEL LOAD (face.kmodel)
# ─────────────────────────────────────────────────────────────────────────────
def load_kpu_model():
    """
    Load the face detection YOLOv2 kmodel.
    IMPORTANT: Call this BEFORE hardware_init() so KPU gets maximum free memory.
    Priority: SD card /sd/face.kmodel → SPI flash 0x300000
    """
    task = None
    # Aggressively free memory before every load attempt
    gc.collect()
    gc.collect()

    # Try SD card first (place face.kmodel in root of SD card)
    for path in ["/sd/face.kmodel", "/sd/facedetect_main.kmodel", "/sd/facedetectmain.kmodel"]:
        gc.collect()   # free between attempts
        try:
            task = kpu.load(path)
            print("KPU: loaded from SD card:", path)
            break
        except Exception as e:
            print("SD load failed (%s):" % path, e)

    # Fallback to SPI flash
    if task is None:
        gc.collect()
        try:
            task = kpu.load(0x300000)
            print("KPU: loaded from SPI Flash @ 0x300000")
        except Exception as e:
            print("KPU Flash load failed:", e)

    if task is None:
        # LCD may not be init yet – try both paths
        try:
            lcd.clear(lcd.BLACK)
            lcd.draw_string(10, 10, "Model Load Error!", lcd.RED, lcd.WHITE)
        except Exception:
            pass
        raise SystemExit("Cannot load face detection model.")

    kpu.init_yolo2(task, CONF_THRESHOLD, NMS_THRESHOLD, 5, ANCHORS)
    print("YOLOv2 face detection initialised.")
    return task

# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── KPU model loaded FIRST – before LCD/sensor consume memory ────────────
    # ERR_NO_MEM fix: load model while maximum SRAM is available.
    task = load_kpu_model()

    # ── Hardware Init (after model load) ──────────────────────────────────────
    buzzer, red_led, green_led, blue_led = gpio_init()
    uart  = uart_init()
    hardware_init()

    lcd.clear(lcd.BLACK)
    print("Smart Helmet Detection Loop Started...")

    # ── Face tracking state ───────────────────────────────────────────────────
    last_face     = None   # smoothed [x, y, w, h]
    face_lifetime = 0
    display_score = 0.0

    # ── Rolling history buffers for debouncing flicker ───────────────────────
    # Keeps last HISTORY_LEN frames of face/tag True/False results.
    # Only needs HISTORY_HIT positives out of HISTORY_LEN to report PASS.
    # This smooths out single-frame dropouts without adding lag.
    face_history = []
    tag_history  = []
    HISTORY_LEN  = 12   # wide rolling window – very smooth transitions
    HISTORY_HIT  = 2    # only need 2/12 positives – extremely easy to trigger PASS

    # ── State machine init ────────────────────────────────────────────────────
    current_state         = STATE_SCANNING
    cycle_start_time      = time.ticks_ms()
    helmet_detected_state = False   # locked result for current cycle

    clock       = time.clock()
    frame_count = 0

    # Start first cycle: Blue ON, everything else OFF
    blue_led.value(1)
    buzzer.value(0)
    red_led.value(0)
    green_led.value(0)

    try:
        while True:
            clock.tick()
            frame_count += 1

            # Periodic GC to prevent MicroPython SRAM fragmentation
            if frame_count % 30 == 0:
                gc.collect()

            # ── Capture frame ─────────────────────────────────────────────────
            try:
                img = sensor.snapshot()
            except Exception as e:
                print("Snapshot error:", e)
                time.sleep_ms(50)
                continue

            # ── Face Detection (YOLOv2 via KPU) ──────────────────────────────
            detections = kpu.run_yolo2(task, img)

            best_det = None
            if detections:
                best_det = detections[0]
                for det in detections:
                    if det.value() > best_det.value():
                        best_det = det

            if best_det is not None:
                nx, ny, nw, nh = best_det.rect()
                display_score = best_det.value()
                if last_face is None:
                    last_face = [nx, ny, nw, nh]
                else:
                    # Exponential Moving Average – reduces box jitter
                    a = SMOOTHING
                    last_face[0] = int(a * nx + (1 - a) * last_face[0])
                    last_face[1] = int(a * ny + (1 - a) * last_face[1])
                    last_face[2] = int(a * nw + (1 - a) * last_face[2])
                    last_face[3] = int(a * nh + (1 - a) * last_face[3])
                face_lifetime = MAX_LIFETIME
            else:
                if face_lifetime > 0:
                    face_lifetime -= 1
                else:
                    last_face = None

            face_seen = last_face is not None

            # ── Rolling face debounce ────────────────────────────────────────
            face_history.append(face_seen)
            if len(face_history) > HISTORY_LEN:
                face_history.pop(0)
            face_seen_stable = face_history.count(True) >= HISTORY_HIT

            # ── Compute EYE BOX for positional check ─────────────────────────
            face_left_x   = None
            face_right_x  = None
            face_bottom_y = None
            eye_fx        = None
            eye_fy        = None
            eye_fw        = None
            eye_fh        = None
            if face_seen:
                fx, fy, fw, fh = last_face
                fx = max(0, min(fx, 320))
                fy = max(0, min(fy, 240))
                fw = max(0, min(fw, 320 - fx))
                fh = max(0, min(fh, 240 - fy))
                eye_fx        = fx
                eye_fy        = fy
                eye_fw        = fw
                eye_fh        = max(10, int(fh * 0.45))
                face_left_x   = fx
                face_right_x  = fx + fw
                face_bottom_y = fy + eye_fh

            # ── AprilTag Detection – TIGHT ROI to fit in memory ───────────────
            # find_apriltags() allocates ~12 bytes/pixel internally.
            # 320×200 ROI = 64K pixels → ~786KB → OOM on this device.
            # Face-width × 100px ≈ 10K pixels → ~120KB → FITS ✅
            # ROI is tight to the face column so any tag found is already
            # in the correct horizontal alignment.
            tag_detected  = False
            tag_id_valid  = 0
            tag_cx_valid  = -1
            tag_cy_valid  = -1
            _apriltag_err = None

            if face_seen and face_bottom_y is not None:
                _roi_x = max(0, face_left_x - 25)
                _roi_y = max(0, face_bottom_y)
                _roi_w = min(face_right_x - face_left_x + 50, 320 - _roi_x)
                _roi_h = min(240 - _roi_y, 120)   # max 120 rows, ~14K pixels – fits in pool

                if _roi_w > 10 and _roi_h > 10:
                    gc.collect()   # free pool before allocating tag buffer
                    try:
                        _tags = img.find_apriltags(
                            roi=(_roi_x, _roi_y, _roi_w, _roi_h),
                            families=image.TAG36H11)
                        for tag in _tags:
                            if tag.id() != 0:
                                continue
                            tcx, tcy           = tag.cx(), tag.cy()
                            tx, ty, tw_t, th_t = tag.rect()
                            if tcy > face_bottom_y:
                                # ✅ In column (ROI-guaranteed) and below eye box
                                tag_detected = True
                                tag_id_valid = tag.id()
                                tag_cx_valid = tcx
                                tag_cy_valid = tcy
                                img.draw_rectangle(tx, ty, tw_t, th_t, color=GREEN, thickness=2)
                                img.draw_line(tcx - 6, tcy,     tcx + 6, tcy,     color=GREEN)
                                img.draw_line(tcx,     tcy - 6, tcx,     tcy + 6, color=GREEN)
                                img.draw_string(tx, max(ty - 14, 0), "TAG OK", color=GREEN, scale=1)
                            else:
                                img.draw_rectangle(tx, ty, tw_t, th_t, color=ORANGE, thickness=2)
                                img.draw_string(tx, max(ty - 14, 0), "NOT BELOW", color=ORANGE, scale=1)
                    except Exception as e:
                        _apriltag_err = str(e)



            # ── Rolling tag debounce ────────────────────────────────────────────
            tag_history.append(tag_detected)
            if len(tag_history) > HISTORY_LEN:
                tag_history.pop(0)
            tag_seen = tag_history.count(True) >= HISTORY_HIT

            # ── Combined helmet-worn decision ─────────────────────────────────
            # HELMET WORN = face seen (stable) AND tag seen (stable via history)
            helmet_worn_now = face_seen_stable and tag_seen

            # ── State Machine ─────────────────────────────────────────────────
            now             = time.ticks_ms()
            elapsed_in_cycle = time.ticks_diff(now, cycle_start_time)

            if current_state == STATE_SCANNING:
                if helmet_worn_now:
                    # ── PASS: lock result immediately ──────────────────────────
                    helmet_detected_state = True
                    current_state         = STATE_RESULT_LOCKED
                    buzzer.value(0)
                    print("Result locked: PASS (helmet worn detected)")

                elif elapsed_in_cycle >= SCAN_WINDOW_MS:
                    # ── FAIL: 30 s window expired with no detection ────────────
                    helmet_detected_state = False
                    current_state         = STATE_RESULT_LOCKED
                    buzzer.value(1)
                    print("Result locked: FAIL (30 s scan window expired)")

            elif current_state == STATE_RESULT_LOCKED:
                if elapsed_in_cycle >= CYCLE_INTERVAL_MS:
                    # ── New cycle: reset back to scanning ─────────────────────
                    cycle_start_time = now
                    current_state    = STATE_SCANNING
                    buzzer.value(0)
                    print("New detection cycle started. Scanning...")

            # ── LED update every frame (always accurate) ───────────────────────
            # BLUE  = detection cycle is scanning (not yet locked)
            # GREEN = both face AND tag confirmed → helmet WORN
            # RED   = scanning but not yet confirmed, or result is FAIL
            if current_state == STATE_SCANNING:
                # Actively scanning: BLUE on, others off
                blue_led.value(1)
                green_led.value(0)
                red_led.value(0)
            else:
                # Result locked
                blue_led.value(0)
                if helmet_detected_state:
                    # Both face + tag passed → GREEN
                    green_led.value(1)
                    red_led.value(0)
                else:
                    # Failed → RED
                    green_led.value(0)
                    red_led.value(1)



            # ── UART: send helmet status to internal ESP32 every frame ────────
            send_helmet_signal(uart, helmet_detected_state,
                               tag_detected,        # detected this frame?
                               tag_id_valid,
                               tag_cx_valid,
                               tag_cy_valid)

            # ── LCD: draw eye-region box + alignment guides ──────────────────
            if face_seen:
                box_color  = GREEN if tag_detected else RED
                face_label = "HELMET WORN" if tag_detected else "NO HELMET"

                score_str = ("%d%%" % int(display_score * 100)
                             if face_lifetime == MAX_LIFETIME else "tracking")

                # Draw eye-region box (cropped height = EYE_CROP_RATIO of full face)
                img.draw_rectangle(eye_fx, eye_fy, eye_fw, eye_fh,
                                   color=box_color, thickness=2)

                # Yellow threshold line = bottom of eye box = tag must be below this
                img.draw_line(eye_fx, eye_fy + eye_fh,
                              eye_fx + eye_fw, eye_fy + eye_fh,
                              color=YELLOW, thickness=2)

                # Vertical axis guide lines (inner AXIS_TOLERANCE band edges)
                # These show the exact X-range the tag center must fall within
                ax_left  = eye_fx + int(eye_fw * (1.0 - AXIS_TOLERANCE) / 2)
                ax_right = eye_fx + eye_fw - int(eye_fw * (1.0 - AXIS_TOLERANCE) / 2)
                img.draw_line(ax_left,  eye_fy, ax_left,  eye_fy + eye_fh,
                              color=YELLOW, thickness=1)
                img.draw_line(ax_right, eye_fy, ax_right, eye_fy + eye_fh,
                              color=YELLOW, thickness=1)

                # Label above the box
                img.draw_string(eye_fx, max(eye_fy - 14, 0),
                                face_label + " " + score_str,
                                color=box_color, scale=1)

            # ── LCD: HUD banner ───────────────────────────────────────────────
            tag_txt  = "AprilTag: PASS" if tag_detected else "AprilTag: FAIL"
            face_txt = "Face:     PASS" if face_seen    else "Face:     FAIL"
            img.draw_string(2, 2,  tag_txt,  color=(GREEN if tag_detected else RED), scale=1)
            img.draw_string(2, 16, face_txt, color=(GREEN if face_seen    else RED), scale=1)

            # Show tag error if any
            if _apriltag_err is not None:
                img.draw_string(2, 30, "ERR:" + _apriltag_err[:24], color=YELLOW, scale=1)

            # Final status banner at bottom of screen
            if helmet_detected_state:
                img.draw_string(2, 220, "=== HELMET WORN ===",   color=GREEN, scale=1)
            else:
                img.draw_string(2, 220, "=== NO HELMET ===",     color=RED,   scale=1)

            # FPS counter
            fps = clock.fps()
            img.draw_string(img.width() - 75, 4,
                            "%.1f FPS" % fps, color=YELLOW, scale=1)

            # Push frame to LCD
            lcd.display(img)

            # GC at end of each frame (reference project pattern)
            gc.collect()


    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        if task is not None:
            kpu.deinit(task)
            print("KPU deinitialised.")
        # Safe shutdown: all outputs OFF
        buzzer.value(0)
        red_led.value(0)
        green_led.value(0)
        blue_led.value(0)
        lcd.clear(lcd.BLACK)
        print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal runtime exception:", e)
        # Emergency safe-shutdown of outputs
        try:
            buz = GPIO(GPIO.GPIOHS0, GPIO.OUT)
            r   = GPIO(GPIO.GPIOHS1, GPIO.OUT)
            g   = GPIO(GPIO.GPIOHS2, GPIO.OUT)
            b   = GPIO(GPIO.GPIOHS3, GPIO.OUT)
            buz.value(0)
            r.value(0)
            g.value(0)
            b.value(0)
        except Exception:
            pass
