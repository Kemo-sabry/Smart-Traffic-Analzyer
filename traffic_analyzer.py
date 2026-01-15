import cv2
import cvzone
import math
import numpy as np
import csv
import os
from ultralytics import YOLO

class TrafficAnalyzer:
    def __init__(self, model_path, input_video, output_video=None):
        self.cap = cv2.VideoCapture(input_video)
        self.model = YOLO(model_path)
        self.output_video = output_video
        
        # Traffic Density classes
        self.class_map = {
            2: "car",
            3: "motorbike", 
            5: "bus",
            7: "truck"
        }
        

        
        # CSV Logging Setup
        self.csv_file = "traffic_features.csv"
        self.init_csv()

    def init_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['width', 'height', 'aspect_ratio', 'area', 'avg_color_r', 'avg_color_g', 'avg_color_b', 'confidence', 'class_label'])
        
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
                
            # Reset counts for this frame
            current_counts = {
                "car": 0,
                "bus": 0,
                "truck": 0,
                "motorbike": 0
            }

            # 1. Detection & Tracking
            # persist=True enables the internal tracker (ByteTrack by default)
            results = self.model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
            
            # 2. Process Tracks
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                cls_ids = results[0].boxes.cls.int().cpu().tolist()
                
                for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                    self.update_counts(frame, box, track_id, cls_id, current_counts)
            
            # Calculate Traffic Status
            total_vehicles = sum(current_counts.values())
            if total_vehicles < 5:
                status = "Low"
                color_status = (0, 255, 0) # Green
            elif total_vehicles < 15:
                status = "Medium"
                color_status = (0, 165, 255) # Orange
            else:
                status = "High"
                color_status = (0, 0, 255) # Red

            # 3. Draw UI
            self.draw_ui(frame, current_counts, status, color_status)
            
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

    def update_counts(self, frame, box, track_id, cls_id, current_counts):
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        
        # Get class name
        # If using standard YOLOv8n, we map COCO IDs. 
        # If using custom trained model, use self.model.names[cls_id]
        class_name = self.model.names[cls_id]
        
        # Only track relevant vehicles
        if class_name not in current_counts:
            # Try mapping from COCO if using raw yolov8n
            if cls_id in self.class_map:
                class_name = self.class_map[cls_id]
            else:
                return # Ignore irrelevent classes like 'person'
        
        # Increment Count
        current_counts[class_name] += 1
        
        # Draw Bounding Box & Label
        color = (0, 200, 255) # Gold
        if class_name == "truck": color = (255, 100, 0)
        elif class_name == "bus": color = (0, 255, 100)
        
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=color)
        cvzone.putTextRect(frame, f'{track_id} - {class_name}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3, colorT=(255,255,255), colorR=color)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Feature Extraction for ML (Commented out for Performance)
        # try:
        #     # Get Crop for Color Analysis
        #     crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        #     if crop.size > 0:
        #         avg_bgra = cv2.mean(crop)
        #         avg_b, avg_g, avg_r = avg_bgra[0], avg_bgra[1], avg_bgra[2]
                
        #         # Calculate Geometry
        #         aspect_ratio = w / h if h != 0 else 0
        #         area = w * h
        #         confidence = float(box.conf[0]) if hasattr(box, 'conf') else 0.8 # Default if not passed
                
        #         # Log to CSV
        #         # with open(self.csv_file, mode='a', newline='') as f:
        #         #     writer = csv.writer(f)
        #         #     writer.writerow([w, h, aspect_ratio, area, avg_r, avg_g, avg_b, confidence, class_name])
        # except Exception as e:
        #     print(f"Error logging features: {e}")



    def draw_ui(self, frame, current_counts, status, color_status):
        # Draw Dashboard
        cvzone.putTextRect(frame, f"Status: {status}", (20, 50), scale=2, thickness=2, offset=10, colorR=color_status)
        
        y_pos = 110
        for vehicle, count in current_counts.items():
            text = f"{vehicle.capitalize()}: {count}"
            cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_pos += 40

if __name__ == "__main__":
    # Example Usage
    # Ensure you have 'traffic_video3.mp4' in the directory or change path
    # Using 'yolov8n.pt' for demo. Replace with 'runs/detect/train/weights/best.pt' after training.
    analyzer = TrafficAnalyzer(r'traffic_analysis_project\yolov8n_custom_traffic5\weights\best.pt', 'traffic_video3.mp4', 'output_analysis.mp4')
    analyzer.process_video()
