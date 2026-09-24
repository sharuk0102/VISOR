import os
from PIL import Image

def generate_tag36h11_id0():
    # 1. Define the exact 10x10 matrix for TAG36H11 ID 0.
    # 1 represents white, 0 represents black.
    # The grid contains a 1-module outer white border, a 1-module inner black border,
    # and the 6x6 data region representing the code 0xd5d628584.
    grid = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
    
    # 2. Configure box_size (pixels per module) to achieve a 4-5 cm print size at 300 DPI.
    # - Matrix size is 10x10 modules.
    # - We choose box_size = 50 pixels per module.
    # - Total pixel size will be: 10 modules * 50 pixels/module = 500x500 pixels.
    box_size = 50
    width = len(grid[0]) * box_size
    height = len(grid) * box_size
    
    # Create the base 10x10 image in grayscale ('L') mode
    base_img = Image.new('L', (len(grid[0]), len(grid)))
    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            color = 255 if val == 1 else 0
            base_img.putpixel((x, y), color)
            
    # Resize utilizing NEAREST neighbor scaling to prevent anti-aliasing blur
    tag_img = base_img.resize((width, height), Image.Resampling.NEAREST)
    
    # Physical Print Size Math (at 300 DPI):
    # ----------------------------------------------------
    # - Total tag dimensions = 500 x 500 pixels.
    # - Print resolution = 300 DPI (Dots Per Inch).
    # - Printed size in inches = 500 pixels / 300 DPI = 1.67 inches.
    # - Printed size in centimeters = 1.67 inches * 2.54 cm/inch = 4.23 cm.
    # - 4.23 cm is approximately in the middle of the requested 4-5 cm range,
    #   making it the ideal dimension for detection at a 30-35 cm range.
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    path_tag = os.path.join(output_dir, "helmet_apriltag.png")
    
    tag_img.save(path_tag)
    
    print(f"AprilTag (TAG36H11 ID 0) generated successfully:")
    print(f" - Image size: {width}x{height} pixels")
    print(f" - Calculated print size at 300 DPI: 4.23 cm (1.67 inches)")
    print(f" - Saved to: {path_tag}")

if __name__ == "__main__":
    generate_tag36h11_id0()
