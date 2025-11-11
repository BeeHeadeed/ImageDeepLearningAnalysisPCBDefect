from ultralytics import YOLO
import os

model = YOLO('yolov8n.pt')  # nano (smallest/fastest)



data_path = os.path.join("data", "yolo_dataset", "dataset.yaml")

# Train
results = model.train(
    data= data_path, 
    epochs=50,           # Start with 50-100 for baseline
    imgsz=640,           # Image size
    batch=16,            # Adjust based on GPU memory
    device=0,            # GPU device (0, 1, 2...) or 'cpu'
    project='artifacts',  # Save to 'artifacts/' directory
    name='baseline_exp1'
)
