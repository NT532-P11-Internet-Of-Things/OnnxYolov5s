import cv2
import torch

model = torch.hub.load("ultralytics/yolov5", "yolov5s")
video_path = r"F:\P\HK7\XLA\1741510124859903443.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

target_classes = ['car', 'motorcycle']
# Loop through each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)
    filtered_results = [det for det in results.pred[0] if model.names[int(det[5])] in target_classes]
    # Display the results
    for det in filtered_results:
        # Extract bounding box coordinates
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]
        
        # Draw bounding box and label on frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("YOLOv5 Detection", frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()