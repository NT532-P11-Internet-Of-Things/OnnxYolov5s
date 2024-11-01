import cv2
import onnxruntime as ort
import numpy as np

# Load ONNX model
onnx_path = "yolov5s.onnx"
ort_session = ort.InferenceSession(onnx_path)

# Open video file
video_path = r"F:\Media\Pictures\Videos\2024-11-01 09-33-35.mkv"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for ONNX model
    img = cv2.resize(frame, (640, 640))
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = img.astype('float32') / 255.0
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

    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        x1 = int((x - w / 2) * frame.shape[1])
        y1 = int((y - h / 2) * frame.shape[0])
        x2 = int((x + w / 2) * frame.shape[1])
        y2 = int((y + h / 2) * frame.shape[0])

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class {class_ids[i]}: {scores[i]:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()