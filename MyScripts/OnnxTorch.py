import torch
from PIL import Image
import cv2
import numpy as np

# COCO class names
class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
               "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
               "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
               "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
               "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
               "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", 
               "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
               "scissors", "teddy bear", "hair drier", "toothbrush"]

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

image_path = r"F:\Media\Pictures\2024_10_26\IMG_20241026_085842_4256.JPG"

# Load image using OpenCV
img_cv = cv2.imread(image_path)
img_cv = cv2.resize(img_cv, (640, 640))

# Convert OpenCV image to PIL format for inference
img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# Inference
results = model(img_pil, size=640)  # includes NMS

# Process results
for result in results.xyxy[0]:  # xyxy format
    x1, y1, x2, y2, conf, cls = result
    class_id = int(cls)
    
    # Fake detection: change "cat" (class_id 15) to "car" (class_id 2)
    if class_id == 15:
        class_id = 2
    
    class_name = class_names[class_id] if class_id < len(class_names) else f"class{class_id}"
    
    # Draw bounding box and label on the image
    cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    label = f"{class_name} {conf:.2f}"
    
    # Improve text appearance
    font_scale = 0.7
    font_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv2.rectangle(img_cv, (int(x1), int(y1) - text_height - 10), (int(x1) + text_width, int(y1)), (255, 0, 0), -1)  # Background rectangle
    cv2.putText(img_cv, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    print(f"Detected {class_name} with confidence {conf:.2f} at [{x1}, {y1}, {x2}, {y2}]")

# Display the image with bounding boxes and labels
cv2.imshow("YOLOv5 Detection", img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()