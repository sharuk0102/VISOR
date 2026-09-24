import os
import shutil
import glob

def prepare_calibration_data():
    val_images_dir = os.path.join("processed_dataset", "val", "images")
    cal_dir = os.path.join("convert", "calibration_dataset")
    
    os.makedirs(cal_dir, exist_ok=True)
    
    val_images = glob.glob(os.path.join(val_images_dir, "*.jpg"))
    
    if not val_images:
        print(f"Error: No validation images found in {val_images_dir}! Run training first.")
        return
        
    print(f"Copying {min(len(val_images), 50)} images for calibration...")
    # Copy up to 50 validation images to the calibration folder
    for i, img_path in enumerate(val_images[:50]):
        dest = os.path.join(cal_dir, os.path.basename(img_path))
        shutil.copy(img_path, dest)
        
    print(f"Calibration dataset prepared at {cal_dir}")

if __name__ == "__main__":
    prepare_calibration_data()
