import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

onnx_path = r"F:\P\HK7\IOT\Yolo\yolov5s.onnx"
image_path = r"F:\Media\Pictures\2024_10_26\IMG_20241026_085842_4256.JPG"

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

# Load ONNX model using ONNX Runtime
ort_session = ort.InferenceSession(onnx_path)

# Preprocess image
img = Image.open(image_path).resize((640, 640))
img = np.array(img).astype('float32') / 255.0
img = img.transpose(2, 0, 1)  # HWC to CHW
img = img[None, :, :, :]  # Add batch dimension

# Run inference
ort_inputs = {ort_session.get_inputs()[0].name: img}
ort_outs = ort_session.run(None, ort_inputs)

# Process outputs (this part depends on the model's output format)
# Assuming the output format is [batch, num_boxes, 85] where 85 = [x, y, w, h, conf, class_scores...]
boxes = ort_outs[0][0][:, :4]  # x, y, w, h
scores = ort_outs[0][0][:, 4]  # confidence
class_ids = np.argmax(ort_outs[0][0][:, 5:], axis=1)  # class scores

# Filter boxes with a confidence threshold
confidence_threshold = 0.5
indices = np.where(scores > confidence_threshold)[0]

# Load the original image for display
original_img = cv2.imread(image_path)
original_img = cv2.resize(original_img, (640, 640))

print(boxes)
for i in indices:
    box = boxes[i]
    x, y, w, h = box
    x1 = int((x - w / 2) * original_img.shape[1])
    y1 = int((y - h / 2) * original_img.shape[0])
    x2 = int((x + w / 2) * original_img.shape[1])
    y2 = int((y + h / 2) * original_img.shape[0])

    # Get class name
    class_id = class_ids[i]
    class_name = class_names[class_id] if class_id < len(class_names) else f"class{class_id}"
    # print(f"Detected class {class_name} with confidence {scores[i]:.2f} at [{x1}, {y1}, {x2}, {y2}]")
    # print(f"Position: [{x1}, {y1}, {x2}, {y2}]")

    # Draw bounding box and label on the image
    cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{class_name}: {scores[i]:.2f}"
    cv2.putText(original_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image
cv2.imshow("YOLOv5 Detection", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()