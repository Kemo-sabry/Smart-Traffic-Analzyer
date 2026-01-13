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
        
        # Traffic Levels (for Low/Medium/High classification)
        self.limits = {
            "low": 5,
            "medium": 15
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
            
            current_on_screen_count = 0
            
            # 2. Process Tracks
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                cls_ids = results[0].boxes.cls.int().cpu().tolist()
                
                current_on_screen_count = len(boxes)
                
                for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                    self.update_counts(frame, box, track_id, cls_id)
            
            # 3. Determine Traffic Status
            if current_on_screen_count < self.limits["low"]:
                status = "Low"
                status_color = (0, 255, 0) # Green
            elif current_on_screen_count < self.limits["medium"]:
                status = "Medium"
                status_color = (0, 255, 255) # Yellow
            else:
                status = "High"
                status_color = (0, 0, 255) # Red
            
            # 4. Draw UI
            self.draw_ui(frame, status, status_color, current_on_screen_count)
            
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
        class_name = self.model.names[cls_id]
        
        # Only track relevant vehicles
        if class_name not in self.vehicle_counts:
            # Try mapping from COCO if using raw yolov8n
            if cls_id in self.class_map:
                class_name = self.class_map[cls_id]
            else:
                return # Ignore irrelevant classes
        
        # Draw Bounding Box & Label
        color = (0, 200, 255) # Gold
        if class_name == "truck": color = (255, 100, 0)
        elif class_name == "bus": color = (0, 255, 100)
        
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=color)
        cvzone.putTextRect(frame, f'{track_id} - {class_name}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3, colorT=(255,255,255), colorR=color)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Counting Logic (ANY new track seen is counted)
        if track_id not in self.tracked_ids:
            self.tracked_ids.add(track_id)
            self.vehicle_counts[class_name] += 1

    def draw_ui(self, frame, status, status_color, current_count):
        # Draw Dashboard Background
        cv2.rectangle(frame, (0, 0), (350, 300), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(frame, (0, 0), (350, 300), (255, 255, 255), 2)
        
        # Title
        cvzone.putTextRect(frame, "Traffic Monitor", (20, 40), scale=2, thickness=2, offset=5, colorR=(0,0,0))
        
        # Status
        cv2.putText(frame, f"Status: {status}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, f"On Screen: {current_count}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Detailed Counts
        y_pos = 170
        for vehicle, count in self.vehicle_counts.items():
            text = f"{vehicle.capitalize()}: {count}"
            cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30

if __name__ == "__main__":
    # Example Usage
    # Ensure you have 'traffic_video3.mp4' in the directory or change path
    # Using 'yolov8n.pt' for demo. Replace with 'runs/detect/train/weights/best.pt' after training.
    analyzer = TrafficAnalyzer(r'traffic_analysis_project\yolov8n_custom_traffic5\weights\best.pt', 'traffic_video3.mp4', 'output_analysis.mp4')
    analyzer.process_video()
