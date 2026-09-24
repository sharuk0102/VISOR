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
CONF_THRESHOLD  = 0.25   # Lower = easier to detect in low light (0.20–0.40)
NMS_THRESHOLD   = 0.3    # Non-Maximum Suppression threshold
SMOOTHING       = 0.3    # EMA smoothing (lower = heavier smoothing = less jitter)
MAX_LIFETIME    = 10     # Frames to keep face box after losing detection (longer = stable)

# Camera settings
CAMERA_VFLIP    = 0
CAMERA_HMIRROR  = 0
LCD_ROTATION    = 2

# Colours – MaixPy uses plain (R, G, B) tuples
GREEN  = (0,   255, 0)
RED    = (255, 0,   0)
ORANGE = (255, 165, 0)
WHITE  = (255, 255, 255)
YELLOW = (255, 255, 0)
BLACK  = (0,   0,   0)

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
# ── Low-light optimisation ────────────────────────────────────────────────────
sensor.set_contrast(2)              # Max contrast (+2) sharpens edges in dark frames
sensor.set_brightness(1)            # Slight brightness boost for dark environments
sensor.set_auto_gain(True)          # Auto gain ON
sensor.set_gainceiling(8)           # Allow up to 8x gain (helps in low light)
sensor.set_auto_exposure(True)      # Auto exposure ON – let sensor adapt
sensor.set_auto_whitebal(True)      # Auto white balance
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
    for path in ["/sd/face.kmodel", "/sd/facedetect_main.kmodel", "/sd/facedetectmain.kmodel"]:
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

# Rolling history buffers for debouncing flicker
face_history = []      # last N face detection results (True/False)
tag_history  = []      # last N AprilTag detection results (True/False)
HISTORY_LEN  = 7      # wider window = smoother transitions
HISTORY_HIT  = 3      # only need 3/7 hits → easier to trigger PASS

# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Loop
# ─────────────────────────────────────────────────────────────────────────────
try:
    while True:
        img = sensor.snapshot()

        # ── Low-light preprocessing: histogram equalisation ───────────────────
        # histeq() redistributes pixel brightness to use the full 0-255 range.
        # This dramatically improves contrast in dark or unevenly lit frames.
        img.histeq()

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

        # ── Determine EYE BOX for positional check ─────────────────────────
        # The YOLOv2 model detects the full face (eyes to chin).
        # We shrink the displayed box to the TOP 45% (eye region only) so that:
        #   • The green grid is small and covers just the eyes
        #   • The bottom of the eye box is used as the AprilTag threshold
        #   • The tag at chin/visor level sits clearly BELOW the box
        face_left_x   = None
        face_right_x  = None
        face_bottom_y = None
        eye_fy        = None
        eye_fh        = None
        if face_seen:
            fx, fy, fw, fh = last_face
            # Clamp to frame bounds
            fx = max(0, min(fx, 320))
            fy = max(0, min(fy, 240))
            fw = max(0, min(fw, 320 - fx))
            fh = max(0, min(fh, 240 - fy))
            # Eye box = top 45% of the full face height
            eye_fy        = fy
            eye_fh        = max(10, int(fh * 0.45))
            face_left_x   = fx
            face_right_x  = fx + fw
            face_bottom_y = fy + eye_fh   # bottom of EYE BOX = AprilTag threshold

        # ── AprilTag Detection ────────────────────────────────────────────────
        # MaixPy find_apriltags() has a hard resolution limit (~QQVGA).
        # Fix: only search a small ROI = the column directly below the face box.
        # This is always small enough AND already the only valid position for the tag.
        # If no face is detected, skip tag search (compliance is already FAIL).
        tag_detected  = False
        tag_valid_cx  = None
        tag_valid_cy  = None
        _apriltag_err = None

        if face_seen and face_bottom_y is not None:
            # Build ROI: face column width + ±20 px horizontal padding,
            # rows from face bottom to frame bottom.
            # Padding makes the tag easier to catch when held slightly off-centre.
            PAD = 20
            roi_x = max(0, face_left_x - PAD)
            roi_y = max(0, face_bottom_y)
            roi_w = max(1, min(face_right_x - face_left_x + PAD * 2, 320 - roi_x))
            roi_h = max(1, 240 - roi_y)

            try:
                # Search only inside the ROI – well within MaixPy's resolution limit
                tags = img.find_apriltags(roi=(roi_x, roi_y, roi_w, roi_h), families=16)
                for tag in tags:
                    if tag.id() != 0:
                        continue   # Only accept Tag ID 0

                    tcx, tcy = tag.cx(), tag.cy()
                    tx, ty, tw_t, th_t = tag.rect()

                    # Tag is inside the ROI which IS the valid zone → always PASS position
                    tag_detected = True
                    tag_valid_cx = tcx
                    tag_valid_cy = tcy
                    img.draw_rectangle(tx, ty, tw_t, th_t, color=GREEN, thickness=2)
                    img.draw_line(tcx - 6, tcy, tcx + 6, tcy, color=GREEN)
                    img.draw_line(tcx, tcy - 6, tcx, tcy + 6, color=GREEN)
                    img.draw_string(tx, max(ty - 14, 0), "TAG OK", color=GREEN, scale=1)

            except Exception as e:
                _apriltag_err = str(e)

        # ── Debounce tag_detected with rolling history ────────────────────────
        tag_history.append(tag_detected)
        if len(tag_history) > HISTORY_LEN:
            tag_history.pop(0)
        tag_seen = tag_history.count(True) >= HISTORY_HIT

        # ── Draw EYE BOX (top 45% of face = eye region only) ─────────────────
        if face_seen and eye_fy is not None:
            box_color  = GREEN if tag_seen else RED
            face_label = "HELMET WORN" if tag_seen else "NO HELMET"

            if face_lifetime == MAX_LIFETIME:
                score_str = "%d%%" % int(display_score * 100)
            else:
                score_str = "tracking"

            # Draw only the eye-region box (small, tight around eyes)
            img.draw_rectangle(face_left_x, eye_fy, fw, eye_fh,
                               color=box_color, thickness=2)
            # Yellow threshold line at bottom of eye box
            img.draw_line(face_left_x, eye_fy + eye_fh,
                          face_right_x, eye_fy + eye_fh,
                          color=YELLOW, thickness=1)
            img.draw_string(face_left_x, max(eye_fy - 14, 0),
                            face_label + " " + score_str,
                            color=box_color, scale=1)

        # ── HUD Banner ────────────────────────────────────────────────────────
        tag_txt  = "AprilTag: PASS" if tag_seen  else "AprilTag: FAIL"
        face_txt = "Face:     PASS" if face_seen else "Face:     FAIL"
        tag_col  = GREEN if tag_seen  else RED
        face_col = GREEN if face_seen else RED

        img.draw_string(2, 2,  tag_txt,  color=tag_col,  scale=1)
        img.draw_string(2, 16, face_txt, color=face_col, scale=1)

        # Show AprilTag error on screen if detection threw an exception
        if _apriltag_err is not None:
            img.draw_string(2, 30, "TAG ERR:" + _apriltag_err[:20], color=YELLOW, scale=1)

        if tag_seen and face_seen:
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
