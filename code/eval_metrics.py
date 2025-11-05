from ultralytics import YOLO

models = {
    "baseline": "experiments/baseline/weights/best.pt",
    "enhanced": "experiments/enhanced/weights/best.pt"
}

for name, path in models.items():
    print(f"\nEvaluating {name}:")
    model = YOLO(path)
    metrics = model.val()
    print(metrics)
