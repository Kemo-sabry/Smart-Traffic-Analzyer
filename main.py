import cv2
import cvzone
import math
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

# Initialize the Model
model = YOLO('yolov8n.pt')

# Class Names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
              "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
              "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

traffic_classes = ["car", "motorbike", "bus", "truck", "bicycle"]

# Open Video
input_video_path = "traffic_video3.mp4"
cap = cv2.VideoCapture(input_video_path)

# Data Logging List
traffic_data = []
frame_count = 0

print("Starting analysis... Press 'q' to stop and save data.")

while True:
    success, img = cap.read()
    if not success:
        # Loop video for demonstration
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    frame_count += 1
    # Run YOLO detection
    results = model(img, stream=True)

    vehicle_count_in_frame = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in traffic_classes and conf > 0.3:
                # Increment count for this frame
                vehicle_count_in_frame += 1

                # Draw Visuals
                color = (255, 0, 255) # Default Purple
                if currentClass == "car": color = (255, 100, 0) # Blue-ish
                elif currentClass == "bus": color = (0, 255, 0) # Green
                
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=color)
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3)

    # --- TRAFFIC ANALYSIS LOGIC ---
    status = "LOW"
    status_color = (0, 255, 0) # Green

    if vehicle_count_in_frame > 5:
        status = "MEDIUM"
        status_color = (0, 255, 255) # Yellow
    
    if vehicle_count_in_frame > 15:
        status = "HIGH"
        status_color = (0, 0, 255) # Red

    # Log Data: [Frame Number, Vehicle Count, Status]
    # We log every 10th frame to keep data size manageable and less noisy, or every frame for precision
    # Let's log every frame for smooth graphs
    traffic_data.append({
        "Frame": frame_count,
        "VehicleCount": vehicle_count_in_frame,
        "Status": status
    })

    # --- DASHBOARD DISPLAY ---
    cv2.rectangle(img, (0, 0), (350, 150), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, f'Vehicles: {vehicle_count_in_frame}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f'Traffic: {status}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    cv2.imshow("Smart Traffic Analyzer", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save Data on Exit
print("Analysis Stopped. Saving Data...")
df = pd.DataFrame(traffic_data)
df.to_csv("traffic_data.csv", index=False)
print("Data saved to 'traffic_data.csv'. Run 'generate_report.py' to see plots.")

cap.release()
cv2.destroyAllWindows()
