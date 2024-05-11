from ultralytics import YOLO
import torch

if __name__ == "__main__":

    # Load a model
    model = YOLO("Yolo-Weights/yolov8s.pt")
    print(torch.cuda.is_available())

    # Train the model
    results = model.train(data="config.yaml", epochs=100, imgsz=640)
