import cv2
import numpy as np
import os

def calculate_brightness_contrast(image):
    brightness = np.mean(image)
    contrast = np.std(image)
    return brightness, contrast

def adjust_brightness_contrast(image, target_brightness, target_contrast):
    brightness = np.mean(image)
    contrast = np.std(image)
    adjusted_image = (image - brightness) * (target_contrast / contrast) + target_brightness
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
    
    return adjusted_image



def crop_image(image, target_width=640, target_height=480):
    height, width = image.shape[:2]
    if width == target_width and height == target_height:
        return image

    center_x, center_y = width // 2, height // 2
    start_x = max(center_x - target_width // 2, 0)
    start_y = max(center_y - target_height // 2, 0)
    end_x = start_x + target_width
    end_y = start_y + target_height

    cropped_image = image[start_y:end_y, start_x:end_x]


    if cropped_image.shape[1] < target_width or cropped_image.shape[0] < target_height:
        cropped_image = cv2.copyMakeBorder(
            cropped_image,
            top=(target_height - cropped_image.shape[0]) // 2,
            bottom=(target_height - cropped_image.shape[0] + 1) // 2,
            left=(target_width - cropped_image.shape[1]) // 2,
            right=(target_width - cropped_image.shape[1] + 1) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

    return cropped_image


def WBC(input_dir, output_dir, min_contour_area=3000,max_c_area=39000, overlap_threshold=0.2):
    def color_threshold(image):
        image = crop_image(image)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([37, 58, 138])
        upper_hsv = np.array([179, 172, 255])
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

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            input_img_path = os.path.join(input_dir, filename)
            image_bgr = cv2.imread(input_img_path)
            if image_bgr is not None:
                image_bgr = crop_image(image_bgr)
                image_bgr = adjust_brightness_contrast(image_bgr, target_brightness, target_contrast)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                tim, mask = color_threshold(image_rgb)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                boxes = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > min_contour_area and area<max_c_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        boxes.append((x-20, y-20, w+20, h+20))

                merged_boxes = merge_overlapping_boxes(boxes, overlap_threshold)

                for (x, y, w, h) in merged_boxes:
                    #cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 55, 0), 2)
                    #cv2.putText(image_rgb, "WBC", (x+5, y+23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 55, 0), 2)
                    abcd=1
                
                tim_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                output_img_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_img_path, tim_bgr)
                print(f"Processed: {filename}")


def RBC(input_dir, output_dir, min_contour_area=3500):
    def color_threshold(image):
        image = crop_image(image)
        image = 225 - image
        image = cv2.GaussianBlur(image, (7, 7), 1)
        image = cv2.medianBlur(image, 5)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([1, 2, 36])
        upper_hsv = np.array([177, 172, 225])
        hsv_image = cv2.GaussianBlur(hsv_image, (51, 51), 1)
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        result = cv2.bitwise_not(image, image, mask=mask)
        return result, mask
    
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            input_img_path = os.path.join(input_dir, filename)
            image_bgr = cv2.imread(input_img_path)
            if image_bgr is not None:
                image_bgr = adjust_brightness_contrast(image_bgr, target_brightness, target_contrast)
                image_bgr = crop_image(image_bgr)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                tim, mask = color_threshold(image_rgb)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
              
                for contour in contours:
                    if cv2.contourArea(contour) >= min_contour_area:
                        cv2.drawContours(image_bgr, [contour], -1, (40, 40, 255), 2)

                output_img_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_img_path, image_bgr)
                print(f"Processed: {filename}")

reference_image_path = "Images/BloodImage_00000.jpg"
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)
target_brightness, target_contrast = calculate_brightness_contrast(reference_image)
input_dir = 'Images'
output_dir = 'Results_WBC'
output_dir2 = 'Results_RBC'
WBC(input_dir, output_dir)
RBC(input_dir, output_dir2)