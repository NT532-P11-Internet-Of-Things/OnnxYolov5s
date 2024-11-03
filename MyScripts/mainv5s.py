import torch
import cv2
import warnings
import numpy as np
import time
warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Open video file
video_path = r"F:\P\HK7\IOT\Project\datas\video.mkv"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Assuming class ID for car is 2
car_class_id = 2
# Assuming class ID for motorcycle is 3
motorcycle_class_id = 3

# Initialize trackers
left_tracker = cv2.TrackerCSRT_create()
right_tracker = cv2.TrackerCSRT_create()
left_initBB = None
right_initBB = None

# Start time for tracking
start_time = None

# Loop through each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Filter results to only include cars and motorcycles
    filtered_results = []
    for i in range(len(results.xyxy[0])):
        if int(results.xyxy[0][i, 5]) == car_class_id or int(results.xyxy[0][i, 5]) == motorcycle_class_id:
            filtered_results.append(results.xyxy[0][i])

    # Count the number of vehicles
    vehicle_count = len(filtered_results)

    # Find the most left and most right vehicles
    if len(filtered_results) > 0:
        left_vehicle = min(filtered_results, key=lambda x: x[0])
        right_vehicle = max(filtered_results, key=lambda x: x[2])

        # Initialize left tracker
        if left_initBB is None:
            x1, y1, x2, y2, conf, cls = left_vehicle
            left_initBB = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            left_tracker.init(frame, left_initBB)
            start_time = time.time()  # Start the timer

        # Initialize right tracker
        if right_initBB is None:
            x1, y1, x2, y2, conf, cls = right_vehicle
            right_initBB = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            right_tracker.init(frame, right_initBB)
            start_time = time.time()  # Start the timer

    # Update trackers and draw bounding boxes
    elapsed_time = time.time() - start_time if start_time else 0
    if elapsed_time <= 2:
        if left_initBB is not None:
            (left_success, left_box) = left_tracker.update(frame)
            if left_success:
                (x, y, w, h) = [int(v) for v in left_box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Left Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if right_initBB is not None:
            (right_success, right_box) = right_tracker.update(frame)
            if right_success:
                (x, y, w, h) = [int(v) for v in right_box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Right Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # Draw a line across the full screen
        height, width = frame.shape[:2]
        cv2.line(frame, (0, height // 2), (width, height // 2), (0, 0, 255), 2)

    # Display the vehicle count on the top-left corner
    count_label = f'Vehicles: {vehicle_count}'
    cv2.putText(frame, count_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the results
    cv2.namedWindow("YOLOv5 Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv5 Detection", 960, 540)
    cv2.imshow("YOLOv5 Detection", frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()