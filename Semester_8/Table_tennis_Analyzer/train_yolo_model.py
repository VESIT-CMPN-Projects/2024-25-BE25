# train_yolo_model.py

from ultralytics import YOLO

def train_yolo_model(yaml_path, epochs=100, img_size=640, batch_size=16):
    """
    Train YOLOv8 model on the table tennis dataset
    Args:
        yaml_path: Path to the YAML configuration file
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Training batch size
    """
    model = YOLO('yolov8n.pt')  # nano version for speed

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name="table_tennis_model"
    )
    return model

if __name__ == "__main__":
    yaml_path = "E:/Table tennis analyser for changes/dataset/table_tennis.yaml"
    model = train_yolo_model(yaml_path, epochs=100)
