import os
from .image_preproccesing_functions import Image_Processing_main
import yaml, glob
from PIL import Image

global model
model = YOLO('yolov8n.pt')  # nano (smallest/fastest)
from ultralytics import YOLO
import sys


# Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "yolo_dataset", "dataset.yaml")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
# Train

def normal_train(epochs: int):
    results = model.train(
        data=DATA_PATH,
        epochs=epochs,
        imgsz=640,
        name="baseline",
        project=OUTPUTS_DIR  # YOLO saves directly here
    )
    return results

def enhanced_train(epochs: int):
    def preprocess_dataset_images(yaml_file):
        with open(yaml_file, 'r') as f:
            ds = yaml.safe_load(f)

        for split in ("train", "val", "test"):
            if split not in ds:
                continue

        path = ds[split]
        # if the YAML points to a directory, glob images; if it points to a list/file pattern, glob that
        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, '**', '*.*'), recursive=True)
        else:
            files = glob.glob(path, recursive=True)

        for img_path in files:
            try:
                _, _, img_enhanced, _, _ = Image_Processing_main(img_path)
                # save enhanced image to a separate folder (preserve relative structure where possible)
                out_root = os.path.join("artifacts", "enhanced_images")
                try:
                    rel = os.path.relpath(img_path, start=path)
                except Exception:
                    rel = os.path.basename(img_path)
                out_dir = os.path.join(out_root, os.path.dirname(rel))
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, os.path.basename(img_path))
                Image.fromarray(img_enhanced).save(out_path)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    preprocess_dataset_images(DATA_PATH)
        
    # prefer the enhanced images directory if it exists, otherwise fall back to the original dataset yaml
    enhanced_dir = os.path.join("artifacts", "enhanced_images")
    train_data = enhanced_dir if os.path.exists(enhanced_dir) else DATA_PATH
    results = model.train(
        data=train_data, 
        epochs=epochs,           # More epochs for enhanced training
        imgsz=640,           # Image size
        batch=16,            # Adjust based on GPU memory
        device=0,            # GPU device (0, 1, 2...) or 'cpu'
        project='artifacts',  # Save to 'artifacts/' directory
        name='enhanced_exp1',
        augment=True,
        lr0=0.01,
        momentum=0.937,      # Momentum
        weight_decay=0.0005  # Weight decay
    )

    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <normal|enhanced> [epochs]")
        sys.exit(1)

    mode = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    if mode == "normal":
        normal_train(epochs)
    elif mode == "enhanced":
        enhanced_train(epochs)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)