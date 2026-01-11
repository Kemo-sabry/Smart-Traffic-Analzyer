import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO

class TrafficAnalyzer:
    def __init__(self, model_path, input_video, output_video=None):
        self.cap = cv2.VideoCapture(input_video)
        self.model = YOLO(model_path)
        self.output_video = output_video
        
        # Define the Counting Line (StartPoint, EndPoint)
        # Adjust these coordinates based on your video resolution and camera angle
        self.line_start = (300, 400)
        self.line_end = (900, 400)
        
        # Tracking Data
        self.tracked_ids = set()
        self.vehicle_counts = {
            "car": 0,
            "bus": 0,
            "truck": 0,
            "motorbike": 0
        }
        
        # Class mapping (Update this if your model has different IDs)
        # Using standard COCO IDs for the demo, but mapped to our specific categories
        self.class_map = {
            2: "car",
            3: "motorbike", 
            5: "bus",
            7: "truck"
        }
        
    def process_video(self):
        print("Starting Video Analysis...")
        
        # Setup Video Writer if output is requested
        out = None
        if self.output_video:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_video, fourcc, fps, (width, height))

        while True:
            success, frame = self.cap.read()
            if not success:
                break
                
            # 1. Detection & Tracking
            # persist=True enables the internal tracker (ByteTrack by default)
            results = self.model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
            
            # 2. Process Tracks
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                cls_ids = results[0].boxes.cls.int().cpu().tolist()
                
                for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                    self.update_counts(frame, box, track_id, cls_id)
            
            # 3. Draw UI
            self.draw_ui(frame)
            
            # Output
            if out:
                out.write(frame)
            cv2.imshow("Smart Traffic Analyzer v2", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print("\nAnalysis Complete.")
        print("Total Counts:", self.vehicle_counts)

    def update_counts(self, frame, box, track_id, cls_id):
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        
        # Get class name
        # If using standard YOLOv8n, we map COCO IDs. 
        # If using custom trained model, use self.model.names[cls_id]
        class_name = self.model.names[cls_id]
        
        # Only track relevant vehicles
        if class_name not in self.vehicle_counts:
            # Try mapping from COCO if using raw yolov8n
            if cls_id in self.class_map:
                class_name = self.class_map[cls_id]
            else:
                return # Ignore irrelevent classes like 'person'
        
        # Draw Bounding Box & Label
        color = (0, 200, 255) # Gold
        if class_name == "truck": color = (255, 100, 0)
        elif class_name == "bus": color = (0, 255, 100)
        
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=color)
        cvzone.putTextRect(frame, f'{track_id} - {class_name}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3, colorT=(255,255,255), colorR=color)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Counting Logic (Line Crossing)
        # We define a line and check if the center point crosses it
        # Simple proximity check for this demo (better is vector cross product)
        
        line_y = self.line_start[1]
        offset = 15 # Margin of error
        
        if line_y - offset < cy < line_y + offset:
            if track_id not in self.tracked_ids:
                self.tracked_ids.add(track_id)
                self.vehicle_counts[class_name] += 1
                cv2.line(frame, self.line_start, self.line_end, (0, 255, 0), 5) # Flash green

    def draw_ui(self, frame):
        # Draw Counting Line
        cv2.line(frame, self.line_start, self.line_end, (0, 0, 255), 3)
        
        # Draw Dashboard
        cvzone.putTextRect(frame, "Traffic Counter", (20, 40), scale=2, thickness=2, offset=5)
        
        y_pos = 100
        for vehicle, count in self.vehicle_counts.items():
            text = f"{vehicle.capitalize()}: {count}"
            cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_pos += 40

if __name__ == "__main__":
    # Example Usage
    # Ensure you have 'traffic_video3.mp4' in the directory or change path
    # Using 'yolov8n.pt' for demo. Replace with 'runs/detect/train/weights/best.pt' after training.
    analyzer = TrafficAnalyzer(r'traffic_analysis_project\yolov8n_custom_traffic5\weights\best.pt', 'traffic_video2.mp4', 'output_analysis.mp4')
    analyzer.process_video()
