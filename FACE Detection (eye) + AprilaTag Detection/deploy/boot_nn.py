# boot_nn.py - YOLOv3-Lite Helmet Detection for Sipeed Maixduino (K210)
# ======================================================================
# Purpose:
#   Runs the lightweight 2-scale YOLOv3-Lite helmet detection model
#   to identify bare heads (class: no_helmet) in real-time.
#   It triggers physical alarms (buzzer/LED) and transmits status over UART.
# ======================================================================

import gc
import time
import math
import machine
import sensor
import image
import lcd
import KPU as kpu
from Maix import FPIOA, GPIO
from machine import UART

# --- CONFIGURATION ────────────────────────────────────────────────────
PIN_BUZZER      = 24       # Buzzer GPIO Pin (Matches AprilTag setup)
PIN_RED_LED     = 12       # Red LED Warning GPIO Pin
PIN_GREEN_LED   = 13       # Green LED Success GPIO Pin

# Image Dimensions
IMG_W, IMG_H = 224, 160

# Model Output Grid Sizes
GRID_S16_W, GRID_S16_H = 14, 10
GRID_S32_W, GRID_S32_H = 7, 5

# Hardcoded Anchors (must match model/anchors.txt values)
# Scale 1 (stride 16) gets anchors 0,1,2; Scale 2 (stride 32) gets anchors 3,4,5
ANCHORS_S16 = [(38, 45), (42, 50), (47, 54)]
ANCHORS_S32 = [(51, 58), (56, 62), (60, 68)]

# Pre-calculated threshold for raw objectness value
# Since confidence = sigmoid(raw_val) * sigmoid(class_raw_val) > 0.5,
# we can early-filter if raw_val <= 0.0 (saving exponential/division operations in MicroPython)
OBJ_RAW_THRESH = 0.0

# --- GPIO INITIALIZATION ──────────────────────────────────────────────
def gpio_init():
    fpioa = FPIOA()
    fpioa.set_function(PIN_BUZZER,    FPIOA.GPIOHS0)
    fpioa.set_function(PIN_RED_LED,   FPIOA.GPIOHS1)
    fpioa.set_function(PIN_GREEN_LED, FPIOA.GPIOHS2)

    buzzer_gpio = GPIO(GPIO.GPIOHS0, GPIO.OUT)
    red_gpio    = GPIO(GPIO.GPIOHS1, GPIO.OUT)
    green_gpio  = GPIO(GPIO.GPIOHS2, GPIO.OUT)

    # Set initial states (Buzzer/Red off, Green on)
    buzzer_gpio.value(0)
    red_gpio.value(0)
    green_gpio.value(1)
    return buzzer_gpio, red_gpio, green_gpio

# --- UART INITIALIZATION ──────────────────────────────────────────────
def uart_init():
    fpioa = FPIOA()
    # K210 IO6 connects to ESP32 U0TX -> K210 RX
    fpioa.set_function(6, FPIOA.UART1_RX)
    # K210 IO7 connects to ESP32 U0RX -> K210 TX
    fpioa.set_function(7, FPIOA.UART1_TX)
    uart = UART(UART.UART1, 115200, 8, None, 1, timeout=1000, read_buf_len=64)
    return uart

# --- COORDINATES DECODER ──────────────────────────────────────────────
def sigmoid(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.0 if x < 0 else 1.0

def decode_output(fmap):
    """Parses flat 1D output of shape (3150,) into list of bounding boxes."""
    boxes = []
    
    # --- PART 1: Stride 16 Scale (Index 0 to 2519) ---
    # Shape: (10, 14, 3 anchors, 6 values) -> step coefficients:
    # index = gy * 252 + gx * 18 + a * 6 + c
    for gy in range(10):
        for gx in range(14):
            for a in range(3):
                base_idx = gy * 252 + gx * 18 + a * 6
                
                # Retrieve raw objectness score
                raw_obj = fmap[base_idx + 4]
                if raw_obj > OBJ_RAW_THRESH:
                    # Check class score (index 5)
                    raw_cls = fmap[base_idx + 5]
                    if raw_cls > OBJ_RAW_THRESH:
                        obj_score = sigmoid(raw_obj)
                        cls_score = sigmoid(raw_cls)
                        confidence = obj_score * cls_score
                        
                        if confidence > 0.5:
                            # Decode box coordinates
                            tx = sigmoid(fmap[base_idx + 0])
                            ty = sigmoid(fmap[base_idx + 1])
                            tw = fmap[base_idx + 2]
                            th = fmap[base_idx + 3]
                            
                            anchor_w, anchor_h = ANCHORS_S16[a]
                            
                            cx = (gx + tx) / 14.0
                            cy = (gy + ty) / 10.0
                            w = (anchor_w * math.exp(tw)) / 224.0
                            h = (anchor_h * math.exp(th)) / 160.0
                            
                            xmin = cx - w / 2.0
                            ymin = cy - h / 2.0
                            xmax = cx + w / 2.0
                            ymax = cy + h / 2.0
                            
                            boxes.append((xmin, ymin, xmax, ymax, confidence))
                            
    # --- PART 2: Stride 32 Scale (Index 2520 to 3149) ---
    # Shape: (5, 7, 3 anchors, 6 values) -> step coefficients:
    # index = 2520 + gy * 126 + gx * 18 + a * 6 + c
    for gy in range(5):
        for gx in range(7):
            for a in range(3):
                base_idx = 2520 + gy * 126 + gx * 18 + a * 6
                
                raw_obj = fmap[base_idx + 4]
                if raw_obj > OBJ_RAW_THRESH:
                    raw_cls = fmap[base_idx + 5]
                    if raw_cls > OBJ_RAW_THRESH:
                        obj_score = sigmoid(raw_obj)
                        cls_score = sigmoid(raw_cls)
                        confidence = obj_score * cls_score
                        
                        if confidence > 0.5:
                            tx = sigmoid(fmap[base_idx + 0])
                            ty = sigmoid(fmap[base_idx + 1])
                            tw = fmap[base_idx + 2]
                            th = fmap[base_idx + 3]
                            
                            anchor_w, anchor_h = ANCHORS_S32[a]
                            
                            cx = (gx + tx) / 7.0
                            cy = (gy + ty) / 5.0
                            w = (anchor_w * math.exp(tw)) / 224.0
                            h = (anchor_h * math.exp(th)) / 160.0
                            
                            xmin = cx - w / 2.0
                            ymin = cy - h / 2.0
                            xmax = cx + w / 2.0
                            ymax = cy + h / 2.0
                            
                            boxes.append((xmin, ymin, xmax, ymax, confidence))
                            
    return boxes

def apply_nms(boxes, iou_threshold=0.4):
    """Executes Non-Maximum Suppression to remove duplicate candidate boxes."""
    if not boxes:
        return []
        
    # Sort boxes by confidence in descending order
    sorted_boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    selected_boxes = []
    
    while len(sorted_boxes) > 0:
        best_box = sorted_boxes[0]
        selected_boxes.append(best_box)
        sorted_boxes = sorted_boxes[1:]
        
        remaining_boxes = []
        for box in sorted_boxes:
            # Calculate Intersection over Union (IoU)
            ixmin = max(best_box[0], box[0])
            iymin = max(best_box[1], box[1])
            ixmax = min(best_box[2], box[2])
            iymax = min(best_box[3], box[3])
            
            iw = max(0.0, ixmax - ixmin)
            ih = max(0.0, iymax - iymin)
            inter_area = iw * ih
            
            area_best = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])
            area_box = (box[2] - box[0]) * (box[3] - box[1])
            union_area = area_best + area_box - inter_area
            
            iou = inter_area / union_area if union_area > 0 else 0.0
            
            if iou < iou_threshold:
                remaining_boxes.append(box)
                
        sorted_boxes = remaining_boxes
        
    return selected_boxes

# --- MAIN EXECUTION ───────────────────────────────────────────────────
def main():
    # 1. Initialize hardware
    lcd.init(freq=15000000)
    lcd.clear(lcd.BLACK)
    lcd.draw_string(10, 10, "YOLOv3-Lite Helmet System", lcd.WHITE, lcd.BLACK)
    lcd.draw_string(10, 30, "Initializing hardware...", lcd.WHITE, lcd.BLACK)
    
    buzzer_gpio, red_gpio, green_gpio = gpio_init()
    uart = uart_init()
    
    # 2. Camera Setup (GRAYSCALE, HQVGA 240x160)
    sensor.reset()
    sensor.set_pixformat(sensor.GRAYSCALE)
    sensor.set_framesize(sensor.HQVGA)
    
    # Crop to 224x160 in hardware (8 pixels from left and right borders)
    sensor.set_windowing((8, 0, 224, 160))
    sensor.run(1)
    sensor.skip_frames(time=1000)
    
    # 3. Load Model
    lcd.clear(lcd.BLACK)
    lcd.draw_string(10, 30, "Loading KPU model...", lcd.WHITE, lcd.BLACK)
    try:
        task = kpu.load("/sd/yolov3_lite.kmodel")
        lcd.draw_string(10, 50, "KPU Loaded successfully!", lcd.GREEN, lcd.BLACK)
        time.sleep(1)
    except Exception as e:
        print("Model load error:", e)
        lcd.draw_string(10, 50, "KPU Load Failed!", lcd.RED, lcd.BLACK)
        while True:
            time.sleep(1)

    print("Starting inference loop...")
    while True:
        # Capture frame
        img = sensor.snapshot()
        
        # Sync pixel values with KPU AI memory
        img.pix_to_ai()
        
        # Run forward pass
        fmap = kpu.forward(task, img)
        
        # Decode output coordinates
        raw_boxes = decode_output(fmap)
        final_boxes = apply_nms(raw_boxes, iou_threshold=0.4)
        
        no_helmet_detected = len(final_boxes) > 0
        
        # Draw results on camera feed
        for box in final_boxes:
            xmin, ymin, xmax, ymax, conf = box
            
            # Map normalized coordinates back to actual pixel sizes
            x = int(xmin * IMG_W)
            y = int(ymin * IMG_H)
            w = int((xmax - xmin) * IMG_W)
            h = int((ymax - ymin) * IMG_H)
            
            # Draw box and label (color maps to a gray scale value on screen)
            img.draw_rectangle(x, y, w, h, color=180, thickness=2)
            img.draw_string(x, y - 15, "NO HELMET", color=180, scale=1.0)
            
        # Draw status banner
        if no_helmet_detected:
            img.draw_rectangle(0, 0, IMG_W, 20, color=50, fill=True)
            img.draw_string(10, 2, "ALERT: NO HELMET DETECTED", color=255, scale=1.0)
            
            # Trigger alarms
            buzzer_gpio.value(1)
            red_gpio.value(1)
            green_gpio.value(0)
            
            # Send alert flag over UART (TAG:0 means non-compliant/no tag)
            uart.write("TAG:0\n")
        else:
            img.draw_rectangle(0, 0, IMG_W, 20, color=200, fill=True)
            img.draw_string(10, 2, "COMPLIANT (HELMET OK)", color=0, scale=1.0)
            
            # Turn off alarms
            buzzer_gpio.value(0)
            red_gpio.value(0)
            green_gpio.value(1)
            
            # Send compliant flag over UART
            uart.write("TAG:1 ID:0 X:-1 Y:-1\n")
            
        # Display image on LCD screen
        lcd.display(img)
        
        # Force garbage collection to prevent memory leaks
        gc.collect()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Application Error:", e)
        # De-initialize KPU
        kpu.deinit(task)
