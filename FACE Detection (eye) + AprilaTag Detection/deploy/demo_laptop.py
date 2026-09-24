import sys
import time
import numpy as np
import cv2

# ─────────────────────────────────────────────────────────────────────────────
# Speed-tuning knobs
# ─────────────────────────────────────────────────────────────────────────────
# Camera capture resolution – lower = faster grab + decode.
CAM_W, CAM_H = 320, 240

# Detection scale – all detectors run on a smaller image; results are
# scaled back up for display.  0.5 = half of CAM_W × CAM_H (160 × 120).
DETECT_SCALE = 0.5
# ─────────────────────────────────────────────────────────────────────────────

# ── AprilTag import ──────────────────────────────────────────────────────────
try:
    from pupil_apriltags import Detector
    HAS_APRILTAG = True
    APRILTAG_TYPE = "pupil"
except ImportError:
    try:
        import apriltag
        HAS_APRILTAG = True
        APRILTAG_TYPE = "standard"
    except ImportError:
        print("Warning: Neither 'pupil-apriltags' nor 'apriltag' is installed. Tag tracking is disabled.")
        HAS_APRILTAG = False
        APRILTAG_TYPE = None


def run_laptop_demo():
    global HAS_APRILTAG

    det_w = int(CAM_W * DETECT_SCALE)
    det_h = int(CAM_H * DETECT_SCALE)
    # Scale-up factors for mapping small-image coords → display coords
    sx = 1.0 / DETECT_SCALE   # e.g. 2.0
    sy = 1.0 / DETECT_SCALE

    # ── 1. AprilTag Detector ─────────────────────────────────────────────────
    detector = None
    if HAS_APRILTAG:
        try:
            if APRILTAG_TYPE == "pupil":
                # 4 threads + quad_decimate=2 makes detection ~4x faster
                detector = Detector(
                    families="tag36h11",
                    nthreads=4,
                    quad_decimate=2.0,   # subsample quads – big speed win
                    quad_sigma=0.0,
                    refine_edges=False,  # skip expensive edge refinement
                    decode_sharpening=0.25,
                )
                print("pupil-apriltags ready (4 threads, quad_decimate=2).")
            else:
                options = apriltag.DetectorOptions(families="tag36h11")
                detector = apriltag.Detector(options)
                print("apriltag detector ready.")
        except Exception as e:
            print("Failed to initialize AprilTag detector:", e)
            HAS_APRILTAG = False

    # ── 2. Haar Cascade Classifiers ──────────────────────────────────────────
    # Tuned for speed: larger scaleFactor = fewer pyramid levels = faster.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    print("Occlusion-robust Face and Eye Classifiers loaded.")

    # ── 3. Webcam – force low resolution + drain stale buffer ────────────────
    print("Starting webcam stream. Press 'q' to exit.")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # CAP_DSHOW avoids extra latency on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)       # 1-frame buffer = always get the latest frame
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return

    # Drain any frames already sitting in the driver buffer
    for _ in range(5):
        cap.read()

    print(f"Webcam ready at {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}.")

    # Initialize rolling window history buffer for face debouncing
    face_history = []
    prev_time    = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Downscale once; run ALL detectors on this small image ─────────────
        small = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_NEAREST)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # ── PART 1: Collect raw AprilTag hits (positional check comes later) ──
        raw_tags     = []
        tag_detected = False
        tag_center   = None

        if HAS_APRILTAG and detector is not None:
            try:
                if APRILTAG_TYPE == "pupil":
                    tags = detector.detect(gray, estimate_tag_pose=False)
                    for tag in tags:
                        family_str = tag.tag_family.decode() if isinstance(tag.tag_family, bytes) else tag.tag_family
                        if family_str == "tag36h11" and tag.tag_id == 0:
                            raw_tags.append(tag)
                else:
                    tags = detector.detect(gray)
                    for tag in tags:
                        if tag.tag_family.decode() == "tag36h11" and tag.tag_id == 0:
                            raw_tags.append(tag)
            except Exception as e:
                print("AprilTag detection step error:", e)

        # ── PART 2: Face & Eye Detection (faster params on small image) ───────
        # Larger scaleFactor = fewer pyramid levels = faster sweep
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40))
        eyes  = eye_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=4, minSize=(15, 15))

        raw_face_detected = (len(faces) > 0) or (len(eyes) > 0)
        face_history.append(raw_face_detected)
        if len(face_history) > 5:
            face_history.pop(0)
        face_seen = face_history.count(True) >= 3

        # ── Determine face bounding region (on small image) ───────────────────
        face_bottom_y = face_left_x = face_right_x = None
        if len(faces) > 0:
            best_face = max(faces, key=lambda f: f[2] * f[3])
            bfx, bfy, bfw, bfh = best_face
            face_bottom_y = bfy + bfh
            face_left_x   = bfx
            face_right_x  = bfx + bfw
        elif len(eyes) > 0:
            face_bottom_y = max(ey + eh for (ex, ey, ew, eh) in eyes)
            face_left_x   = min(ex       for (ex, ey, ew, eh) in eyes)
            face_right_x  = max(ex + ew  for (ex, ey, ew, eh) in eyes)

        # ── PART 1b: Positional AprilTag Validation ───────────────────────────
        # An AprilTag PASSES only when its center satisfies ALL three conditions:
        #   1. Below  the bottom edge of the face box  (cy > face_bottom_y)
        #   2. To the right of the left edge           (cx >= face_left_x)
        #   3. To the left of the right edge           (cx <= face_right_x)
        # i.e. the tag must be in the same vertical column as the face.
        for tag in raw_tags:
            cx_s, cy_s = int(tag.center[0]), int(tag.center[1])   # small-image coords

            in_column  = (face_left_x  is not None and face_right_x is not None
                          and face_left_x <= cx_s <= face_right_x)
            below_face = (face_bottom_y is not None and cy_s > face_bottom_y)

            # Scale corners + center back to display-frame coordinates
            corners_disp = (np.array(tag.corners) * [sx, sy]).astype(np.int32)
            cx_d = int(cx_s * sx)
            cy_d = int(cy_s * sy)

            if below_face and in_column:
                tag_detected = True
                tag_center   = (cx_d, cy_d)
                cv2.polylines(frame, [corners_disp], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.circle(frame, tag_center, 5, (0, 255, 0), -1)
                cv2.putText(frame, "HELMET TAG ID 0", (cx_d - 70, cy_d - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                reason = "NOT BELOW" if not below_face else "OUT OF COLUMN"
                cv2.polylines(frame, [corners_disp], isClosed=True, color=(0, 165, 255), thickness=2)
                cv2.putText(frame, f"TAG ({reason})", (cx_d - 60, cy_d - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        # ── Draw face / eye boxes (scale small-image coords back to display) ──
        if face_seen:
            color = (0, 255, 0) if tag_detected else (0, 0, 255)
            label = "FACE OK (HELMET WORN)" if tag_detected else "NO HELMET (BARE FACE)"
            for (fx, fy, fw, fh) in faces:
                dx, dy = int(fx * sx), int(fy * sy)
                dw, dh = int(fw * sx), int(fh * sy)
                cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), color, 2)
                # Yellow threshold line at bottom of face box
                cv2.line(frame, (dx, dy + dh), (dx + dw, dy + dh), (255, 255, 0), 1)
                cv2.putText(frame, label, (dx, max(dy - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            for (ex, ey, ew, eh) in eyes:
                dx, dy = int(ex * sx), int(ey * sy)
                dw, dh = int(ew * sx), int(eh * sy)
                cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), color, 1)
                cv2.putText(frame, "EYE", (dx, max(dy - 4, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                start_x = max(0, (w_disp - crop_w) // 2)
                start_y = max(0, (h_disp - crop_h) // 2)
                cropped = gray_frame[start_y:start_y+crop_h, start_x:start_x+crop_w]
                model_in_img = cv2.resize(cropped, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
                
                model_input = model_in_img.astype(np.float32) / 255.0
                model_input = np.expand_dims(np.expand_dims(model_input, axis=0), axis=-1)
                
                interpreter.set_tensor(input_details[0]['index'], model_input)
                interpreter.invoke()
                y_pred_flat = interpreter.get_tensor(output_details[0]['index'])
                tflite_boxes = decode_output(y_pred_flat, threshold=0.5)
            except Exception as e:
                pass
                
        # --- PART 4: Two-Component Safety Interlock Logic ---
        # 1. AprilTag check
        tag_ok = tag_detected
        
        # 2. Face Detection Model check
        face_model_ok = face_seen
        
        # Strict Compliance: Helmet is worn ONLY if BOTH AprilTag and Face Detection Model pass
        is_compliant = tag_ok and face_model_ok
            
        banner_h = 60  # Increase banner height for double row status
        overlay = display_frame.copy()
        
        # Draw background banner
        cv2.rectangle(overlay, (0, 0), (w_disp, banner_h), (30, 30, 30), -1)
        
        # Status Text colors
        tag_color = (0, 255, 0) if tag_ok else (0, 0, 255)
        model_color = (0, 255, 0) if face_model_ok else (0, 0, 255)
        
        # Draw status parameters
        cv2.putText(overlay, f"1. AprilTag: {'PASS' if tag_ok else 'FAIL'}", (15, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, tag_color, 2)
        cv2.putText(overlay, f"2. Face Detection Model: {'PASS' if face_model_ok else 'FAIL'}", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, model_color, 2)
                    
        # Draw compliance overlay banner on the right side
        comp_color = (0, 180, 0) if is_compliant else (0, 0, 200)
        cv2.rectangle(overlay, (w_disp - 220, 0), (w_disp, banner_h), comp_color, -1)
        cv2.putText(overlay, "HELMET WORN" if is_compliant else "HELMET NOT WORN", (w_disp - 210, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
        cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0, display_frame)
        
        # Display HUD status
        if not HAS_APRILTAG:
            cv2.putText(display_frame, "AprilTag tracking disabled (missing library)", (10, h_disp - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            cv2.putText(display_frame, "AprilTag tracking: ENABLED (pupil-apriltags)", (10, h_disp - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        
        cv2.imshow("Smart Helmet Visual interlock - Laptop Demo", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_laptop_demo()
