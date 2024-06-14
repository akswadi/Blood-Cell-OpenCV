import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def color_threshold(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([103, 67, 181])
    upper_hsv = np.array([179, 172, 250])
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    result = cv2.bitwise_or(image, image, mask=mask)
    return result, mask
def merge_overlapping_boxes(boxes, overlap_threshold):
    merged_boxes = []
    
    for current_box in boxes:
        x1, y1, w1, h1 = current_box
        merged = False
        
        for i, merged_box in enumerate(merged_boxes):
            x2, y2, w2, h2 = merged_box
            it_x1 = max(x1, x2)
            it_y1 = max(y1, y2)
            it_x2 = min(x1 + w1, x2 + w2)
            it_y2 = min(y1 + h1, y2 + h2)
            
            it_area = max(0, it_x2 - it_x1 + 1) * max(0, it_y2 - it_y1 + 1)
            area1 = w1 * h1
            area2 = w2 * h2
            overlap_ratio = it_area / min(area1, area2)
            
            if overlap_ratio > overlap_threshold:
                x_merge = min(x1, x2)
                y_merge = min(y1, y2)
                w_merge = max(x1 + w1, x2 + w2) - x_merge
                h_merge = max(y1 + h1, y2 + h2) - y_merge
                merged_boxes[i] = (x_merge, y_merge, w_merge, h_merge)
                merged = True
                break
        
        if not merged:
            merged_boxes.append(current_box)
    
    return merged_boxes

input_dir = 'Images'
output_dir = 'Results'
os.makedirs(output_dir, exist_ok=True)

min_contour_area = 1100 
overlap_threshold = 0.2  

result_images = []

for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_img_path = os.path.join(input_dir, filename)
        image_bgr = cv2.imread(input_img_path)
        if image_bgr is not None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tim, mask = color_threshold(image_rgb)
            

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            

            boxes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    boxes.append((x-20, y-20, w+20, h+20))

            merged_boxes = merge_overlapping_boxes(boxes, overlap_threshold)
            

            for (x, y, w, h) in merged_boxes:
                cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 55, 0), 2)
                cv2.putText(image_rgb,"WBC",(x+5,y+23),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,55,0),2)
            tim_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            output_img_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_img_path, tim_bgr)
            result_images.append(tim_bgr)
            print(f"Processed: {filename}")