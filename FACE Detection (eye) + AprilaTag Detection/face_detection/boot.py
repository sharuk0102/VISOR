# boot.py
# Face Detection demo for Sipeed Maixduino (K210 + OV2640)
# Loaded from SD Card

import sensor
import image
import lcd
import KPU as kpu
import time
import gc

# 1. Initialize the LCD Display
lcd.init(freq=15000000)
lcd.rotation(2)  # Adjust rotation if screen is upside down (0-3)
lcd.clear(lcd.RED)

# Draw loading status
img_status = image.Image()
img_status.draw_string(20, 100, "Initializing Camera...", color=(255, 255, 255), scale=2)
lcd.display(img_status)

# 2. Initialize the Camera Sensor (OV2640)
try:
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565) # RGB565 is required for KPU
    sensor.set_framesize(sensor.QVGA)   # QVGA is 320x240, required by the face model
    
    # Configure camera orientation
    sensor.set_hmirror(0) # Change to 1 if image is horizontally mirrored
    sensor.set_vflip(0)   # Change to 1 if image is vertically upside down
    
    sensor.run(1)
    sensor.skip_frames(time=2000) # Let sensor adjust to lighting
except Exception as e:
    img_status.clear()
    img_status.draw_string(10, 80, "Camera Error!", color=(255, 0, 0), scale=2)
    img_status.draw_string(10, 120, str(e), color=(255, 255, 255), scale=1)
    lcd.display(img_status)
    while True:
        time.sleep(1)

# Update loading status
img_status.clear()
img_status.draw_string(20, 100, "Loading face.kmodel...", color=(255, 255, 255), scale=2)
lcd.display(img_status)

# 3. Load the YOLOv2 Face Detection Model from SD Card
model_path = "/sd/face.kmodel"
task = None

try:
    # Load model from the root of the micro SD card
    task = kpu.load(model_path)
except Exception as e:
    img_status.clear()
    img_status.draw_string(10, 60, "Model Load Fail!", color=(255, 0, 0), scale=2)
    img_status.draw_string(10, 100, "Ensure 'face.kmodel' is on SD card.", color=(255, 255, 255), scale=1)
    img_status.draw_string(10, 130, "Err: " + str(e), color=(255, 255, 255), scale=1)
    lcd.display(img_status)
    while True:
        time.sleep(1)

# 4. Initialize YOLOv2 Network Parameters
# These anchor values are specifically tuned for the standard face detection model.
anchors = (1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437, 6.92275, 6.718375, 9.01025)
# Parameters: task, probability threshold, NMS threshold, anchor count, anchor values
kpu.init_yolo2(task, 0.5, 0.3, 5, anchors)

# Set screen title bar style (optional UI)
info_font_color = (0, 255, 0)
rect_color = (0, 255, 0) # Green for detected face

print("Face detection initialized successfully. Starting loop...")
fps_clock = time.clock()

# 5. Main Inference Loop
while True:
    fps_clock.tick()
    
    # Capture camera frame
    img = sensor.snapshot()
    
    # Run face detection on the frame
    # This runs inference on the KPU accelerator
    objects = kpu.run_yolo2(task, img)
    
    if objects:
        for obj in objects:
            # Draw bounding rectangle around the face
            # obj.rect() returns (x, y, w, h)
            img.draw_rectangle(obj.rect(), color=rect_color, thickness=2)
            
            # Display confidence score above the box
            confidence = int(obj.value() * 100)
            img.draw_string(obj.x(), obj.y() - 15, "Face: %d%%" % confidence, color=info_font_color, scale=1)
            
    # Calculate and draw FPS on the screen
    fps = fps_clock.fps()
    img.draw_string(10, 10, "FPS: %.1f" % fps, color=(255, 255, 0), scale=1)
    
    # Display the final frame on the LCD screen
    lcd.display(img)
    
    # Run garbage collection periodically to prevent memory leaks
    gc.collect()

# Deinitialize KPU task (unreachable in this infinite loop, but good practice)
kpu.deinit(task)
