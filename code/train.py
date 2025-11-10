import os
from ultralytics import YOLO
from eval_metrics import export_results

# Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "yolo_dataset", "dataset.yaml")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Training 
model = YOLO("yolov8n.pt")

results_base = model.train(
    data=DATA_PATH,
    epochs=50,
    imgsz=640,
    name="baseline",
    project=OUTPUTS_DIR  # YOLO saves directly here
)

export_results(OUTPUTS_DIR,METRICS_DIR,"baseline")



