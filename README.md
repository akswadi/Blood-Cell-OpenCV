# Blood Cell Detector using OpenCV

## Crop and Rotate Images

This script processes images by cropping and rotating them.

### What the Script Does

1. **Read Images**: It reads all images with `.jpg`, `.jpeg`, or `.png` extensions from the specified input directory (`input_folder`).
2. **WBC_Detection**: Each image is scanned for WBC's using a given colour range
2. **RBC_Detection**: Each image is scanned for RBC's using a given colour range
4. **Save Processed Images**: The cropped and rotated images are saved to the specified output directory (`output_folder`).

### Usage Instructions

1. Place the images you want to process in the `Images` directory.
2. Find out the ideal colour range for WBC/RBC detection using 'HSV_VALS.py' Copy and paste those values in the original code
3. Run the script, and it will process the images and save the results in two separate directories for WBC and RBC

