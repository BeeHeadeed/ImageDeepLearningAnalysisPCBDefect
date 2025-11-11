from ultralytics import YOLO
import sys

model_path = sys.argv[1]
image_path = sys.argv[2]

model = YOLO(model_path)
model.predict(source=image_path, save=True)
