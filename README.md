# Smart Traffic Analyzer 🚗

A Computer Vision project using **YOLOv8** and **OpenCV** to detect and count vehicles in traffic videos.

## Features
- Detects Cars, Motorbikes, Buses, Trucks, and Bicycles.
- Visualizes detection with bounding boxes.
- (In Progress) Line-based counting logic.

## Setup Instructions

### 1. Install Dependencies
Open your terminal in this folder and run:
```bash
pip install -r requirements.txt
```

### 2. Get a Video
You need a traffic video to analyze.
1.  Go to [Pexels Traffic Videos](https://www.pexels.com/search/videos/traffic/) or use YouTube.
2.  Download a video (Top-down view or street view works best).
3.  Rename it to `traffic_video.mp4` and put it in this folder.
    *   *Note: If you name it something else, update the `input_video_path` variable in `main.py`.*

### 3. Run the Analyzer
```bash
python main.py
```

## How it Works (For your Report)
1.  **Input:** Reads the video frame by frame.
2.  **Detection:** Uses the YOLOv8 (You Only Look Once) deep learning model to identify objects.
3.  **Filtering:** Ignores non-vehicle objects (like people or birds) to focus on traffic.
4.  **Counting:** (To be fully implemented) Tracks objects as they cross a specific line on the street.

## Troubleshooting
- **Video closes immediately:** Check if the video path in `main.py` is correct.
- **Slow performance:** Your computer might be slow. You can try resizing the frame in the code.
