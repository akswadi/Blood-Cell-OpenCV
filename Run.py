import cv2
import numpy as np
import os

def color_threshold(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([103, 67, 182])
    upper_hsv = np.array([180, 180, 250])
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    kernel = np.ones((30, 30), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result, mask

input_dir = 'Images'
output_dir = 'Results'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_img_path = os.path.join(input_dir, filename)
        image_bgr = cv2.imread(input_img_path)
        if image_bgr is not None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb = cv2.GaussianBlur(image_rgb, (15, 15), 0)
            thresholded_image, mask = color_threshold(image_bgr)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 340:
                    cv2.drawContours(image_bgr, [contour], -1, (0, 255, 0), 2)
            thresholded_image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
            output_img_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_img_path, thresholded_image_bgr)
            print(f"Processed: {filename}")

print("Image processing and contour detection completed. Images with contours saved in:", output_dir)
