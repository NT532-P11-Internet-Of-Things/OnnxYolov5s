import torch
import cv2
import warnings
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

    # Render the filtered results on the frame
    for result in filtered_results:
        x1, y1, x2, y2, conf, cls = result
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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