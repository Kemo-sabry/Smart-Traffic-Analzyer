from ultralytics import YOLO

def train():
    # 1. Load the model. 
    # We use 'yolov8n.pt' (Nano) for speed, but you can use 'yolov8s.pt' or 'yolov8m.pt' for better accuracy.
    # The model will download automatically if not present.
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt') 

    # 2. Train the model
    # data: Path to data.yaml
    # epochs: Number of training epochs (50-100 is usually good for initial results)
    # imgsz: Image size (640 is standard)
    # batch: Batch size (reduce if you run out of GPU memory)
    # project: Name of the project folder where results are saved
    # name: Name of the experiment run
    print("Starting training...")
    try:
        results = model.train(
            data='data.yaml',
            epochs=50,
            imgsz=640,
            batch=16,
            project='traffic_analysis_project',
            name='yolov8n_custom_traffic',
            device='cpu', # Use '0' for GPU, 'cpu' for CPU
            verbose=True
        )
        print("Training completed successfully!")
        print(f"Best model saved at: traffic_analysis_project/yolov8n_custom_traffic/weights/best.pt")
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Note: Ensure you have a GPU available and properly configured (part of ultralytics/pytorch requirements).")
        print("If you want to train on CPU, change device='0' to device='cpu'.")

if __name__ == '__main__':
    train()
