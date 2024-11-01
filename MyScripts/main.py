import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

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

    # Perform inference on the frame
    results = model(frame)

    # Display the results
    results.render()  # Render the results on the frame
    cv2.imshow("YOLOv5 Detection", frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()