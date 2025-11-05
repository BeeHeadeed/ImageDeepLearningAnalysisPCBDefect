from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        augment=False, 
        name="baseline",
        project="experiments"
    )

if __name__ == "__main__":
    main()
