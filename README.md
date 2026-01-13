# Smart Traffic Analyzer v2 - Project Documentation

## 1. System Pipeline
The system follows a standard computer vision pipeline:
1.  **Detection (YOLOv8)**: Scans each video frame to detect objects. We fine-tune this model on specific traffic classes (Car, Bus, Truck, Motorcycle).
2.  **Tracking (ByteTrack)**: Associates detections across frames to assign unique IDs to each vehicle. This prevents double-counting the same car.
3.  **Counting (Line Crossing)**: A virtual line is defined. When a tracked vehicle's center point crosses this line, the counter is incremented based on its class.
4.  **Analysis**: Real-time stats are displayed on the dashboard and saved to an output video.

## 2. Directory Structure
```
Smart Traffic Analyzer/
│
├── datasets/                   # Training Data
│   └── traffic_data/           # Created by prepare_dataset.py
│       ├── train/              # Training split (images + labels)
│       └── val/                # Validation split
│
├── runs/                       # Training Results (weights, charts)
├── data.yaml                   # YOLO dataset config
├── prepare_dataset.py          # Script to auto-generate data
├── train_model.py              # Script to train YOLOv8
├── traffic_analyzer.py         # Main Analysis Application
└── traffic_video3.mp4          # Input Video
```

## 3. How to Train
We use Transfer Learning to fine-tune YOLOv8n.
1.  **Generate Dataset**:
    ```bash
    python prepare_dataset.py
    ```
    *Extracts frames from video and auto-labels them.*

2.  **Run Training**:
    ```bash
    python train_model.py
    ```
    *Trains for 50 epochs. Best weights saved to `traffic_analysis_project/.../best.pt`*

## 4. How to Run Inference
Run the analyzer with the pretrained model (or your trained model):
```bash
python traffic_analyzer.py
```
*Outputs `output_analysis.mp4` with tracking lines and counters.*

## 5. Key Technologies
-   **Ultralytics YOLOv8**: SOTA Object Detection.
-   **ByteTrack**: High-performance multi-object tracking.
-   **OpenCV**: Video processing and visualization.
