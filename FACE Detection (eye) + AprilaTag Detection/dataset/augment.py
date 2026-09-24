import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

def horizontal_flip(img, bboxes):
    """Flip image horizontally and adjust bounding boxes."""
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w, h = img.size
    flipped_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        new_xmin = w - xmax
        new_xmax = w - xmin
        flipped_bboxes.append([new_xmin, ymin, new_xmax, ymax])
    return flipped_img, flipped_bboxes

def random_translation(img, bboxes, max_shift_x=15, max_shift_y=10):
    """Translate image randomly and adjust bounding boxes."""
    w, h = img.size
    shift_x = random.randint(-max_shift_x, max_shift_x)
    shift_y = random.randint(-max_shift_y, max_shift_y)
    
    # Create new background image
    translated_img = Image.new("L", (w, h), color=int(np.mean(img)))
    translated_img.paste(img, (shift_x, shift_y))
    
    translated_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        new_xmin = max(0, min(w, xmin + shift_x))
        new_ymin = max(0, min(h, ymin + shift_y))
        new_xmax = max(0, min(w, xmax + shift_x))
        new_ymax = max(0, min(h, ymax + shift_y))
        
        # Keep box only if it's still valid
        if (new_xmax - new_xmin) > 5 and (new_ymax - new_ymin) > 5:
            translated_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
            
    # If translation pushed all bounding boxes out, return original
    if not translated_bboxes and bboxes:
        return img, bboxes
        
    return translated_img, translated_bboxes

def random_rotate(img, bboxes, max_angle=10):
    """Rotate image and adjust bounding boxes (approximate rotation)."""
    angle = random.uniform(-max_angle, max_angle)
    # Rotate with bilinear filtering to keep edges smooth, filling background with mean intensity
    bg_color = int(np.mean(img))
    rotated_img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=bg_color)
    
    w, h = img.size
    cx, cy = w / 2.0, h / 2.0
    rad = np.radians(-angle)  # PIL rotates counter-clockwise
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    
    rotated_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        # Define 4 corners
        corners = [
            (xmin, ymin), (xmax, ymin),
            (xmin, ymax), (xmax, ymax)
        ]
        
        # Rotate corners around center
        rot_corners_x = []
        rot_corners_y = []
        for x, y in corners:
            tx = x - cx
            ty = y - cy
            rx = tx * cos_a - ty * sin_a + cx
            ry = tx * sin_a + ty * cos_a + cy
            rot_corners_x.append(rx)
            rot_corners_y.append(ry)
            
        new_xmin = max(0, min(w, min(rot_corners_x)))
        new_ymin = max(0, min(h, min(rot_corners_y)))
        new_xmax = max(0, min(w, max(rot_corners_x)))
        new_ymax = max(0, min(h, max(rot_corners_y)))
        
        if (new_xmax - new_xmin) > 5 and (new_ymax - new_ymin) > 5:
            rotated_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
            
    if not rotated_bboxes and bboxes:
        return img, bboxes
        
    return rotated_img, rotated_bboxes

def random_crop(img, bboxes, min_crop_ratio=0.85):
    """Crop image randomly and adjust/clip bounding boxes."""
    w, h = img.size
    crop_w = int(w * random.uniform(min_crop_ratio, 1.0))
    crop_h = int(h * random.uniform(min_crop_ratio, 1.0))
    
    # We want to make sure the crop contains the bounding box(es) if possible
    if bboxes:
        # Find union of all bounding boxes to keep them visible
        min_x = min(b[0] for b in bboxes)
        min_y = min(b[1] for b in bboxes)
        max_x = max(b[2] for b in bboxes)
        max_y = max(b[3] for b in bboxes)
        
        box_w = max_x - min_x
        box_h = max_y - min_y
        
        # Ensure crop size is at least as large as the bounding box union
        crop_w = max(crop_w, int(box_w + 10))
        crop_h = max(crop_h, int(box_h + 10))
        
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        
        # Choose crop coordinates enclosing the box
        x_min_bound = max(0, int(max_x - crop_w))
        x_max_bound = min(w - crop_w, int(min_x))
        y_min_bound = max(0, int(max_y - crop_h))
        y_max_bound = min(h - crop_h, int(min_y))
        
        left = random.randint(x_min_bound, x_max_bound) if x_min_bound <= x_max_bound else 0
        top = random.randint(y_min_bound, y_max_bound) if y_min_bound <= y_max_bound else 0
    else:
        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)
        
    right = left + crop_w
    bottom = top + crop_h
    
    cropped_img = img.crop((left, top, right, bottom))
    # Resize back to original size
    resized_img = cropped_img.resize((w, h), Image.BILINEAR)
    
    # Adjust bounding boxes for crop + resize scale factors
    scale_x = w / crop_w
    scale_y = h / crop_h
    
    cropped_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        
        # Translate to crop space
        new_xmin = (xmin - left) * scale_x
        new_ymin = (ymin - top) * scale_y
        new_xmax = (xmax - left) * scale_x
        new_ymax = (ymax - top) * scale_y
        
        # Clip to dimensions
        new_xmin = max(0, min(w, new_xmin))
        new_ymin = max(0, min(h, new_ymin))
        new_xmax = max(0, min(w, new_xmax))
        new_ymax = max(0, min(h, new_ymax))
        
        if (new_xmax - new_xmin) > 5 and (new_ymax - new_ymin) > 5:
            cropped_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
            
    if not cropped_bboxes and bboxes:
        return img, bboxes
        
    return resized_img, cropped_bboxes

def adjust_brightness_contrast(img):
    """Jitter brightness and contrast."""
    # Jitter brightness
    enhancer_b = ImageEnhance.Brightness(img)
    img = enhancer_b.enhance(random.uniform(0.7, 1.3))
    # Jitter contrast
    enhancer_c = ImageEnhance.Contrast(img)
    img = enhancer_c.enhance(random.uniform(0.7, 1.3))
    return img

def add_noise(img):
    """Add Gaussian or salt-and-pepper noise."""
    arr = np.array(img, dtype=np.float32)
    noise_type = random.choice(["none", "gaussian", "salt_pepper"])
    
    if noise_type == "gaussian":
        std = random.uniform(2.0, 10.0)
        noise = np.random.normal(0, std, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
    elif noise_type == "salt_pepper":
        # Salt & Pepper
        prob = random.uniform(0.005, 0.02)
        random_matrix = np.random.random(arr.shape)
        arr[random_matrix < prob / 2.0] = 0.0     # Pepper
        arr[random_matrix > 1.0 - prob / 2.0] = 255.0  # Salt
        
    return Image.fromarray(arr.astype(np.uint8))

def motion_blur(img):
    """Simulate camera motion blur using a linear filter."""
    if random.random() < 0.5:
        # 50% chance to skip motion blur
        return img
        
    size = random.choice([3, 5])
    # Create horizontal, vertical, or diagonal motion blur kernel
    kernel = np.zeros((size, size))
    direction = random.choice(["h", "v", "d1", "d2"])
    
    if direction == "h":
        kernel[size // 2, :] = 1.0
    elif direction == "v":
        kernel[:, size // 2] = 1.0
    elif direction == "d1":
        for i in range(size):
            kernel[i, i] = 1.0
    else:  # d2
        for i in range(size):
            kernel[i, size - 1 - i] = 1.0
            
    kernel = kernel / np.sum(kernel)
    
    # Run filter using PIL
    # ImageFilter.Kernel requires 1D flat list of float weights
    flat_kernel = kernel.flatten().tolist()
    return img.filter(ImageFilter.Kernel((size, size), flat_kernel, scale=1, offset=0))

def augment_image(img, bboxes):
    """Main pipeline for image augmentation."""
    # Spatial transforms
    if random.random() < 0.5:
        img, bboxes = horizontal_flip(img, bboxes)
    if random.random() < 0.4:
        img, bboxes = random_crop(img, bboxes)
    if random.random() < 0.4:
        img, bboxes = random_translation(img, bboxes)
    if random.random() < 0.4:
        img, bboxes = random_rotate(img, bboxes)
        
    # Pixel transforms
    img = adjust_brightness_contrast(img)
    img = add_noise(img)
    img = motion_blur(img)
    
    return img, bboxes
