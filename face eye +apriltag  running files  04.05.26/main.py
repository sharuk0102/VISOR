"""
maixduino_main.py – Smart Helmet Visual Interlock
Hardware : Sipeed Maixduino (K210)
Runtime  : MicroPython / OpenMV firmware (Sipeed MaixPy)

Wiring / Setup
──────────────
1. Flash the face-detection YOLOv2 kmodel to SPI Flash at address 0x300000
   OR copy facedetect_main.kmodel onto a microSD card.
2. Stick an AprilTag (TAG36H11, ID = 0) on the outside-front of the helmet,
   roughly at chin / jaw level so it appears BELOW the face bounding box in
   the camera feed.
3. Power the board; the LCD will show LIVE camera feed with detections.

Logic
──────
  HELMET WORN  →  face detected  AND  AprilTag (ID 0) center is:
                    • below   the bottom edge of the face bounding box
                    • within  the horizontal span of the face bounding box
  Any other combination → HELMET NOT WORN

LED / LCD feedback
──────────────────
  Green box + "HELMET WORN"   → compliant
  Red box   + "NO HELMET"     → non-compliant
  Orange outline              → tag found but in wrong position
"""

import sensor
import image
import lcd
import KPU as kpu
import gc

# ─────────────────────────────────────────────────────────────────────────────
# Tunable Configuration
# ─────────────────────────────────────────────────────────────────────────────
CONF_THRESHOLD  = 0.35   # Face detection confidence threshold (0.30–0.50)
NMS_THRESHOLD   = 0.3    # Non-Maximum Suppression threshold
SMOOTHING       = 0.5    # EMA smoothing factor for face box (0=heavy, 1=none)
MAX_LIFETIME    = 6      # Frames to keep face box after losing detection

# Camera settings
CAMERA_VFLIP    = 0
CAMERA_HMIRROR  = 0
LCD_ROTATION    = 2

# Colours (RGB565 packed integers for MaixPy)
GREEN  = image.Color.from_rgb(0,   255, 0)
RED    = image.Color.from_rgb(255, 0,   0)
ORANGE = image.Color.from_rgb(255, 165, 0)
WHITE  = image.Color.from_rgb(255, 255, 255)
YELLOW = image.Color.from_rgb(255, 255, 0)
BLACK  = image.Color.from_rgb(0,   0,   0)

# YOLOv2 anchor values – must match what was used during training
ANCHORS = (1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658,
           5.155437, 6.92275, 6.718375, 9.01025)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LCD
# ─────────────────────────────────────────────────────────────────────────────
lcd.init()
lcd.rotation(LCD_ROTATION)
lcd.clear(lcd.BLACK)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Camera
# ─────────────────────────────────────────────────────────────────────────────
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)   # 320 × 240
sensor.set_hmirror(CAMERA_HMIRROR)
sensor.set_vflip(CAMERA_VFLIP)
sensor.set_contrast(1)
sensor.set_brightness(0)
sensor.set_auto_gain(True)
sensor.set_auto_exposure(True)
sensor.set_auto_whitebal(True)
sensor.run(1)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load KPU Face Detection Model
# ─────────────────────────────────────────────────────────────────────────────
task = None
try:
    task = kpu.load(0x300000)
    print("KPU: loaded from SPI Flash @ 0x300000")
except Exception as e:
    print("KPU Flash load failed:", e)
    for path in ["/sd/facedetect_main.kmodel", "/sd/facedetectmain.kmodel"]:
        try:
            task = kpu.load(path)
            print("KPU: loaded from SD card:", path)
            break
        except Exception as e2:
            print("SD load failed (", path, "):", e2)

if task is None:
    lcd.draw_string(10, 10, "Model Load Error!", lcd.RED, lcd.WHITE)
    raise SystemExit("Cannot load face detection model.")

kpu.init_yolo2(task, CONF_THRESHOLD, NMS_THRESHOLD, 5, ANCHORS)
print("YOLOv2 initialised.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Tracking state
# ─────────────────────────────────────────────────────────────────────────────
last_face     = None   # smoothed [x, y, w, h]
face_lifetime = 0
display_score = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Loop
# ─────────────────────────────────────────────────────────────────────────────
try:
    while True:
        img = sensor.snapshot()

        # ── Face Detection (YOLOv2 via KPU) ──────────────────────────────────
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
                # Exponential Moving Average smoothing to reduce box jitter
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

        # ── Determine face bounding region for positional check ───────────────
        face_left_x   = None
        face_right_x  = None
        face_bottom_y = None
        if face_seen:
            fx, fy, fw, fh = last_face
            # Clamp to frame bounds
            fx = max(0, min(fx, 320))
            fy = max(0, min(fy, 240))
            fw = max(0, min(fw, 320 - fx))
            fh = max(0, min(fh, 240 - fy))
            face_left_x   = fx
            face_right_x  = fx + fw
            face_bottom_y = fy + fh

        # ── AprilTag Detection ────────────────────────────────────────────────
        # image.find_apriltags() is built into MaixPy – no extra library needed.
        # It returns a list of apriltag objects with .rect(), .cx(), .cy(), .id()
        tag_detected  = False
        tag_valid_cx  = None
        tag_valid_cy  = None

        try:
            tags = img.find_apriltags(families=image.TAG36H11)
            for tag in tags:
                if tag.id() != 0:
                    continue   # Only accept Tag ID 0

                tcx, tcy = tag.cx(), tag.cy()

                in_column  = (face_left_x  is not None
                              and face_right_x is not None
                              and face_left_x <= tcx <= face_right_x)
                below_face = (face_bottom_y is not None
                              and tcy > face_bottom_y)

                tx, ty, tw_t, th_t = tag.rect()

                if below_face and in_column:
                    # ✅ Valid position: tag is directly below the face box
                    tag_detected = True
                    tag_valid_cx = tcx
                    tag_valid_cy = tcy
                    img.draw_rectangle(tx, ty, tw_t, th_t,
                                       color=GREEN, thickness=2)
                    img.draw_cross(tcx, tcy, color=GREEN, size=6)
                    img.draw_string(tx, max(ty - 14, 0),
                                    "TAG ID 0 OK", color=GREEN, scale=1)
                else:
                    # ❌ Tag found but wrong position – show in orange
                    reason = "NOT BELOW" if not below_face else "WRONG COL"
                    img.draw_rectangle(tx, ty, tw_t, th_t,
                                       color=ORANGE, thickness=2)
                    img.draw_string(tx, max(ty - 14, 0),
                                    reason, color=ORANGE, scale=1)
        except Exception as e:
            # find_apriltags can occasionally raise on unusual frames
            pass

        # ── Draw Face Box ─────────────────────────────────────────────────────
        if face_seen:
            fx, fy, fw, fh = last_face
            box_color  = GREEN if tag_detected else RED
            face_label = "HELMET WORN" if tag_detected else "NO HELMET"

            if face_lifetime == MAX_LIFETIME:
                score_str = "%d%%" % int(display_score * 100)
            else:
                score_str = "tracking"

            img.draw_rectangle(fx, fy, fw, fh, color=box_color, thickness=2)
            # Yellow bottom-edge threshold line
            img.draw_line(fx, fy + fh, fx + fw, fy + fh,
                          color=YELLOW, thickness=1)
            img.draw_string(fx, max(fy - 14, 0),
                            face_label + " " + score_str,
                            color=box_color, scale=1)

        # ── HUD Banner ────────────────────────────────────────────────────────
        tag_txt  = "AprilTag: PASS" if tag_detected else "AprilTag: FAIL"
        face_txt = "Face:     PASS" if face_seen    else "Face:     FAIL"
        tag_col  = GREEN if tag_detected else RED
        face_col = GREEN if face_seen    else RED

        img.draw_string(2, 2,  tag_txt,  color=tag_col,  scale=1)
        img.draw_string(2, 16, face_txt, color=face_col, scale=1)

        if tag_detected and face_seen:
            img.draw_string(2, 220, "=== HELMET WORN ===", color=GREEN, scale=1)
        else:
            img.draw_string(2, 220, "=== NO HELMET ===",   color=RED,   scale=1)

        # ── Display ───────────────────────────────────────────────────────────
        lcd.display(img)

        # Explicit GC to prevent MicroPython SRAM fragmentation
        gc.collect()

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    if task is not None:
        kpu.deinit(task)
        print("KPU deinitialised.")
    lcd.clear(lcd.BLACK)
    print("Done.")
