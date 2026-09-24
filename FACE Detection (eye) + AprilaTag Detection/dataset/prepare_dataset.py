import os
import random
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from augment import augment_image

# Set random seed
random.seed(42)

# Define directories
RAW_IMAGES_DIR = "raw_images"
RAW_ANN_DIR = "raw_annotations"

OUT_DIR = "processed_dataset"
TRAIN_IMG_DIR = os.path.join(OUT_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(OUT_DIR, "train", "labels")
VAL_IMG_DIR = os.path.join(OUT_DIR, "val", "images")
VAL_LBL_DIR = os.path.join(OUT_DIR, "val", "labels")

# Create output directories
os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(TRAIN_LBL_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(VAL_LBL_DIR, exist_ok=True)

# Define input resolution
IMG_W, IMG_H = 224, 160
TARGET_CLASS = "no_helmet"  # Only detect riders NOT wearing a helmet

def parse_voc_xml(xml_path):
    """Parse VOC XML file and extract bounding boxes for TARGET_CLASS."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    bboxes = []
    
    # Get original image dimensions
    size_elem = root.find("size")
    orig_w = int(size_elem.find("width").text)
    orig_h = int(size_elem.find("height").text)
    
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        
        # We only keep the 'no_helmet' class for single-class target detection
        if class_name != TARGET_CLASS:
            continue
            
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        
        # Scale bounding box to match our target width/height if they differ
        xmin_s = (xmin / orig_w) * IMG_W
        ymin_s = (ymin / orig_h) * IMG_H
        xmax_s = (xmax / orig_w) * IMG_W
        ymax_s = (ymax / orig_h) * IMG_H
        
        bboxes.append([xmin_s, ymin_s, xmax_s, ymax_s])
        
    return bboxes

def save_yolo_labels(bboxes, filepath):
    """Save bounding boxes in YOLO format (class_idx center_x center_y width height)."""
    with open(filepath, "w") as f:
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            
            # Compute center and normalized width/height
            cx = (xmin + xmax) / 2.0 / IMG_W
            cy = (ymin + ymax) / 2.0 / IMG_H
            w = (xmax - xmin) / IMG_W
            h = (ymax - ymin) / IMG_H
            
            # Class index is 0 (since it's a single class)
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def process_and_save(img_name, xml_name, split, aug_factor=0):
    img_path = os.path.join(RAW_IMAGES_DIR, img_name)
    xml_path = os.path.join(RAW_ANN_DIR, xml_name)
    
    # Load image and force conversion to 224x160 Grayscale
    img = Image.open(img_path).convert("L")
    img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    
    bboxes = parse_voc_xml(xml_path)
    
    base_name = os.path.splitext(img_name)[0]
    
    if split == "train":
        img_out_dir = TRAIN_IMG_DIR
        lbl_out_dir = TRAIN_LBL_DIR
    else:
        img_out_dir = VAL_IMG_DIR
        lbl_out_dir = VAL_LBL_DIR
        
    # Save original sample (in split folder)
    img.save(os.path.join(img_out_dir, f"{base_name}.jpg"))
    save_yolo_labels(bboxes, os.path.join(lbl_out_dir, f"{base_name}.txt"))
    
    # Generate augmented versions if training split and aug_factor > 0
    if split == "train" and aug_factor > 0:
        for i in range(1, aug_factor + 1):
            aug_img, aug_bboxes = augment_image(img, bboxes)
            aug_img.save(os.path.join(img_out_dir, f"{base_name}_aug{i}.jpg"))
            save_yolo_labels(aug_bboxes, os.path.join(lbl_out_dir, f"{base_name}_aug{i}.txt"))

if __name__ == "__main__":
    print("Preparing dataset (train/val split and data augmentation)...")
    
    # Find all XML files
    xml_files = glob.glob(os.path.join(RAW_ANN_DIR, "*.xml"))
    xml_basenames = [os.path.basename(f) for f in xml_files]
    
    # Shuffle for split
    random.shuffle(xml_basenames)
    
    # 8:2 Split
    split_idx = int(len(xml_basenames) * 0.8)
    train_xmls = xml_basenames[:split_idx]
    val_xmls = xml_basenames[split_idx:]
    
    print(f"Total samples: {len(xml_basenames)}")
    print(f"Train samples (base): {len(train_xmls)}")
    print(f"Val samples: {len(val_xmls)}")
    
    # Process validation set (no augmentation)
    for xml_name in val_xmls:
        img_name = xml_name.replace(".xml", ".jpg")
        process_and_save(img_name, xml_name, split="val", aug_factor=0)
        
    # Process training set with 3x data augmentation factor
    for xml_name in train_xmls:
        img_name = xml_name.replace(".xml", ".jpg")
        process_and_save(img_name, xml_name, split="train", aug_factor=3)
        
    # Count generated files
    train_images = len(os.listdir(TRAIN_IMG_DIR))
    val_images = len(os.listdir(VAL_IMG_DIR))
    
    print("Dataset preparation complete!")
    print(f"Processed training samples (including augmentations): {train_images}")
    print(f"Processed validation samples: {val_images}")
    print(f"Dataset path: {os.path.abspath(OUT_DIR)}")
