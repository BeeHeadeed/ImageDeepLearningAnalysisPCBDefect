import os
import yaml, glob
from PIL import Image 
from ultralytics import YOLO
import sys
from image_preproccesing_functions import preprocess_dataset_images,resolve_enhanced_labels,resolve_enhanced_yaml

model = YOLO('yolov8n.pt')  # nano (smallest/fastest)
# Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "yolo_dataset", "dataset.yaml")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
ENHANCED_DIR = os.path.join(BASE_DIR, "data", "enhanced_images")
ENHANCED_YAML = os.path.join(BASE_DIR, "data", "yolo_dataset", "enhanced_images.yaml")

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
    # Prepare enhanced images if not already done
    # if not os.path.exists(ENHANCED_YAML):
    #     preprocess_dataset_images(DATA_PATH)
    #     resolve_enhanced_labels()
    #     resolve_enhanced_yaml()

    results = model.train(
        data=ENHANCED_YAML,
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=0,
        project=OUTPUTS_DIR,
        name='enhanced_exp1',
        augment=True,
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005
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