import torch
from PIL import Image
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
img = Image.open(image_path)  # PIL image
img = img.resize((640, 640))

# Inference
results = model(img, size=640)  # includes NMS

# Process results
for result in results.xyxy[0]:  # xyxy format
    x1, y1, x2, y2, conf, cls = result
    class_id = int(cls)
    class_name = class_names[class_id] if class_id < len(class_names) else f"class{class_id}"
    
    print(f"Detected {class_name} with confidence {conf:.2f} at [{x1}, {y1}, {x2}, {y2}]")

results.print()  # print results to screen
results.show()  # display results