from ultralytics import YOLO
import torch

if __name__ == "__main__":

    # Load a model
    model = YOLO("Yolo-Weights/yolov8s.pt")
    print(torch.cuda.is_available())
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print ("MPS device not found.")
    # Train the model
    results = model.train(data="config.yaml", epochs=100, imgsz=640,device=torch.device("mps"))
