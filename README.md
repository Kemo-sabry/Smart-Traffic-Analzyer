# Smart Traffic Analyzer

A comprehensive machine learning system for real-time traffic analysis and classification using computer vision and machine learning techniques.

## 📋 Project Overview

This project implements a dual-approach traffic analysis system:
1. **Real-time Video Analysis**: Uses YOLOv8 for vehicle detection and tracking with live traffic density classification
2. **Traffic Situation Classification**: Employs multiple ML classifiers to predict traffic conditions from historical data

## 🎯 Features

### Video Analysis Module
- **Vehicle Detection & Tracking**: Real-time detection of cars, buses, trucks, and motorcycles using YOLOv8 with ByteTrack tracking
- **Traffic Density Classification**: Automatic classification of traffic as Low, Medium, or High based on vehicle count
- **Visual Dashboard**: Real-time display of vehicle counts and traffic status with color-coded indicators
- **Video Output**: Generates annotated videos with bounding boxes, tracking IDs, and statistics

### Traffic Situation Prediction Module
- **Multi-Model Comparison**: Trains and evaluates 4 different classifiers:
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Decision Tree
- **Feature Engineering**: Extracts temporal features (hour, minute, day of week) and vehicle counts
- **Performance Analysis**: Generates confusion matrices and detailed classification reports

## 📁 Project Structure

```
Smart Traffic Analyzer/
│
├── dataset2/                       # Traffic situation dataset
│   └── train/
│       └── TrafficTwoMonth.csv    # Historical traffic data
│
├── datasets/                       # Training data for YOLOv8
│   └── traffic_data/              # Created by prepare_dataset.py
│       ├── train/                 # Training split (images + labels)
│       └── val/                   # Validation split
│
├── traffic_analysis_project/       # Training results folder
│   └── yolov8n_custom_traffic*/   # Model weights and training metrics
│       └── weights/
│           └── best.pt            # Best trained model weights
│
├── data.yaml                       # YOLOv8 dataset configuration
├── prepare_dataset.py              # Auto-generates labeled dataset from video
├── train_model.py                  # YOLOv8 model training script
├── traffic_analyzer.py             # Main video analysis application
├── train_classifiers.py            # ML classifier training & comparison
├── requirements.txt                # Python dependencies
│
├── confusion_matrix_*.png          # Generated confusion matrices
├── model_comparison_results.csv    # Classifier performance comparison
└── output_analysis.mp4             # Processed video output
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Smart-Traffic-Analyzer
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Video Analysis Pipeline

**Step 1: Generate Training Dataset** (Optional - for custom training)
```bash
python prepare_dataset.py
```
This script:
- Extracts frames from `traffic_video3.mp4`
- Auto-labels vehicles using pretrained YOLOv8
- Saves images and labels to `datasets/traffic_data/`
- Supports resumable operation (continues from last saved frame)

**Step 2: Train Custom Model** (Optional)
```bash
python train_model.py
```
- Fine-tunes YOLOv8n on traffic data
- Trains for 50 epochs
- Saves best weights to `traffic_analysis_project/yolov8n_custom_traffic*/weights/best.pt`

**Step 3: Run Traffic Analysis**
```bash
python traffic_analyzer.py
```
- Processes input video frame-by-frame
- Detects and tracks vehicles
- Classifies traffic density:
  - **Low**: < 5 vehicles (Green)
  - **Medium**: 5-14 vehicles (Orange)
  - **High**: ≥ 15 vehicles (Red)
- Outputs annotated video as `output_analysis.mp4`

#### 2. Traffic Situation Classification

```bash
python train_classifiers.py
```
This script:
- Loads `dataset2/train/TrafficTwoMonth.csv`
- Performs feature engineering and selection
- Trains 4 different classifiers
- Generates confusion matrices and performance reports
- Saves results to `model_comparison_results.csv`

## 🛠️ Technologies Used

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8 (Ultralytics) |
| Tracking Algorithm | ByteTrack |
| Video Processing | OpenCV |
| Machine Learning | scikit-learn |
| Data Analysis | pandas, numpy |
| Visualization | matplotlib, seaborn, cvzone |

## 📊 System Pipeline

### Video Analysis Flow
```
Input Video → Frame Extraction → YOLOv8 Detection → ByteTrack Tracking
→ Vehicle Counting → Density Classification → Annotated Output
```

### ML Classification Flow
```
CSV Data → Feature Engineering → Feature Selection → Train/Val/Test Split
→ Scaling → Model Training → Evaluation → Performance Comparison
```

## 🎓 Academic Context

This project was developed as part of a Machine Learning course to demonstrate:
- Transfer learning with state-of-the-art models
- Real-time computer vision applications
- Multi-classifier comparison and evaluation
- Feature engineering and selection
- End-to-end ML pipeline implementation

## 📈 Results

### Video Analysis
- Real-time vehicle detection and tracking
- Accurate density classification with visual feedback
- Smooth video processing with annotated output

### Traffic Situation Classification
Results vary by model and are documented in `model_comparison_results.csv`:
- Detailed accuracy metrics for validation and test sets
- Confusion matrices saved as PNG images
- Classification reports for each model

## 🔧 Configuration

### Traffic Analyzer Settings
Edit `traffic_analyzer.py` to customize:
- Model path (line 168): Change to your trained model
- Input video (line 168): Specify your video file
- Density thresholds (lines 75-82): Adjust Low/Medium/High boundaries

### Training Parameters
Edit `train_model.py` to modify:
- Epochs (line 21): Increase for better accuracy
- Batch size (line 23): Adjust based on GPU memory
- Image size (line 22): Higher = more accurate but slower
- Device (line 26): Use 'cpu' or '0' for GPU

## 📝 Notes

- The system supports resumable dataset preparation for efficient data generation
- Custom class mapping handles both COCO and custom-trained models
- Feature extraction for ML analysis is available but commented out for performance
- All confusion matrices and performance metrics are automatically saved

## 🤝 Contributing

This is an academic project. Feel free to fork and extend for educational purposes.

## 📄 License

This project is developed for educational purposes as part of university coursework.

## 👨‍💻 Authors

Developed for Machine Learning Course - Fall 2025

---

**Note**: Ensure you have sufficient disk space for video processing and model training. GPU acceleration is recommended for training but not required.
