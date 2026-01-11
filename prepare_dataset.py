import cv2
import os
import shutil
from ultralytics import YOLO

def prepare_dataset(video_path, output_dir, num_train=100, num_val=20):
    print(f"Preparing dataset from {video_path}...")
    
    # 1. Setup Directories
    dirs = [
        f"{output_dir}/train/images", f"{output_dir}/train/labels",
        f"{output_dir}/val/images", f"{output_dir}/val/labels"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
    # 2. Check for existing dataset to resume index
    existing_indices = []
    for split in ['train', 'val']:
        img_dir = f"{output_dir}/{split}/images"
        if os.path.exists(img_dir):
            for f in os.listdir(img_dir):
                if f.startswith("frame_") and f.endswith(".jpg"):
                    try:
                        idx = int(f.split("_")[1].split(".")[0])
                        existing_indices.append(idx)
                    except ValueError:
                        pass
    
    start_index = max(existing_indices) + 1 if existing_indices else 0
    print(f"Resuming dataset preparation from index {start_index}...")

    # 3. Load Pretrained Model for Auto-Labeling (Pseudo-labeling)
    model = YOLO('yolov8n.pt')
    
    # Class Mapping: COCO ID -> Our Dataset ID
    # COCO: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
    # Ours: 0=Car, 1=Bus, 2=Truck, 3=Motorcycle (defined in data.yaml)
    coco_to_custom = {
        2: 0, # Car -> Car
        5: 1, # Bus -> Bus
        7: 2, # Truck -> Truck
        3: 3  # Motorbike -> Motorcycle
    }

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    
    # Calculate step to get diverse frames
    target_count = num_train + num_val
    step = max(1, total_frames // target_count)
    
    frame_count = 0        # Frame counter in current video
    saved_count_local = 0  # How many saved from THIS video (for split logic)
    global_index = start_index # Index for filename uniqueness
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        if frame_count % step == 0 and saved_count_local < target_count:
            # Determine split based on local count to ensure we get requested distribution from THIS video
            subset = "train" if saved_count_local < num_train else "val"
            
            # Filename uses GLOBAL index to avoid overwriting
            filename = f"frame_{global_index:04d}"
            img_path = f"{output_dir}/{subset}/images/{filename}.jpg"
            label_path = f"{output_dir}/{subset}/labels/{filename}.txt"
            
            # Save Image
            cv2.imwrite(img_path, frame)
            
            # Auto-Label
            results = model(frame, verbose=False)
            
            with open(label_path, 'w') as f:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    
                    if cls_id in coco_to_custom:
                        # Convert to normalized xywh
                        x, y, w, h = box.xywhn[0].tolist()
                        new_cls = coco_to_custom[cls_id]
                        
                        f.write(f"{new_cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
            saved_count_local += 1
            global_index += 1
            print(f"Saved {subset} sample: {filename}")
            
        frame_count += 1
        if saved_count_local >= target_count:
            break
            
    cap.release()
    print("\nDataset preparation complete!")
    print(f"Images and Labels saved to {output_dir}")
    print(f"Total images in dataset: {global_index}")
    print("You can now run 'train_model.py'.")

if __name__ == "__main__":
    # Ensure video path matches your file
    prepare_dataset("traffic_video3.mp4", "datasets/traffic_data")
