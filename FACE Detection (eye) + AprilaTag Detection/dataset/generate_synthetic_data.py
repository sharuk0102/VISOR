import os
import random
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

# Create directories
os.makedirs("raw_images", exist_ok=True)
os.makedirs("raw_annotations", exist_ok=True)

# Set seed for reproducibility
random.seed(42)

def create_voc_xml(filename, img_w, img_h, bboxes, classes, xml_path):
    root = ET.Element("annotation")
    
    folder = ET.SubElement(root, "folder")
    folder.text = "raw_images"
    
    fname = ET.SubElement(root, "filename")
    fname.text = filename
    
    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(img_w)
    height = ET.SubElement(size, "height")
    height.text = str(img_h)
    depth = ET.SubElement(size, "depth")
    depth.text = "1"  # Grayscale
    
    for bbox, cls_name in zip(bboxes, classes):
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        name.text = cls_name
        
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(bbox[0]))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(bbox[1]))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(bbox[2]))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(bbox[3]))
        
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        
    tree = ET.ElementTree(root)
    # Write with indentation for readability
    ET.indent(tree, space="  ", level=0)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

def generate_sample(idx, is_helmet):
    # Dimensions: 224x160 Grayscale
    w, h = 224, 160
    
    # Create background: gray gradient with noise to look like a road scene
    img = Image.new("L", (w, h), color=0)
    draw = ImageDraw.Draw(img)
    
    # Draw simple background (e.g. road perspective)
    for y in range(h):
        # Gradient background
        color = int(40 + (y / h) * 60)
        draw.line([(0, y), (w, y)], fill=color)
        
    # Add random pixel noise
    pixels = img.load()
    for x in range(w):
        for y in range(h):
            noise = random.randint(-15, 15)
            val = max(0, min(255, pixels[x, y] + noise))
            pixels[x, y] = val
            
    # Add road shoulder line (simulated perspective lines)
    draw.line([(0, h - 20), (w, h - 50)], fill=150, width=1)
    
    # Bounding box coordinates list
    bboxes = []
    classes = []
    
    # Generate head/helmet parameters
    cx = random.randint(40, w - 40)
    cy = random.randint(40, h - 40)
    r = random.randint(18, 26)
    
    # Head color (skin tone equivalent in grayscale)
    head_color = random.randint(180, 220)
    # Draw neck
    draw.rectangle([cx - 6, cy + r - 5, cx + 6, cy + r + 15], fill=head_color - 30)
    # Draw head
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=head_color)
    
    if is_helmet:
        # Draw helmet dome on top of the head
        r_h = r + 2
        # Helmet colors: bright white/yellow in grayscale (240-255) or black (30-50)
        helmet_color = random.choice([40, 255])
        draw.chord([cx - r_h, cy - r_h - 2, cx + r_h, cy + r_h], start=180, end=360, fill=helmet_color)
        # Draw chin strap (simple line)
        draw.line([cx - r, cy, cx, cy + r + 5], fill=50, width=2)
        draw.line([cx + r, cy, cx, cy + r + 5], fill=50, width=2)
        
        # Bounding box coordinates around helmet
        xmin = max(0, cx - r_h)
        ymin = max(0, cy - r_h - 2)
        xmax = min(w, cx + r_h)
        ymax = min(h, cy + r + 5)
        
        bboxes.append([xmin, ymin, xmax, ymax])
        classes.append("helmet")
    else:
        # Draw hair (bare head)
        hair_color = random.randint(20, 50)
        # Draw some curves representing hair on top
        draw.arc([cx - r - 2, cy - r - 2, cx + r + 2, cy], start=180, end=360, fill=hair_color, width=4)
        
        # Bounding box coordinates around bare head
        xmin = max(0, cx - r)
        ymin = max(0, cy - r - 2)
        xmax = min(w, cx + r)
        ymax = min(h, cy + r + 5)
        
        bboxes.append([xmin, ymin, xmax, ymax])
        classes.append("no_helmet")
        
    # Save image
    img_name = f"img_{idx:04d}.jpg"
    img_path = os.path.join("raw_images", img_name)
    img.save(img_path)
    
    # Save XML
    xml_name = f"img_{idx:04d}.xml"
    xml_path = os.path.join("raw_annotations", xml_name)
    create_voc_xml(img_name, w, h, bboxes, classes, xml_path)

if __name__ == "__main__":
    print("Generating synthetic helmet dataset...")
    # Generate 200 samples total (100 helmet, 100 no_helmet)
    for i in range(1, 201):
        is_helmet = (i % 2 == 0)
        generate_sample(i, is_helmet)
    print(f"Generated 200 samples in {os.path.abspath('raw_images')} and {os.path.abspath('raw_annotations')}")
