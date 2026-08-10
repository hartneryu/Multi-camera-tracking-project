import os

# Keep Ultralytics settings inside this project.
YOLO_CONFIG_FOLDER = os.path.abspath("outputs/ultralytics_config")
os.makedirs(YOLO_CONFIG_FOLDER, exist_ok=True)
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    YOLO_CONFIG_FOLDER,
)

from ultralytics import YOLO


def load_detector(model_path):
    return YOLO(model_path)


def result_to_detections(result):
    detections = []
    if result.boxes is None:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    for box, score in zip(boxes, scores):
        detections.append(
            {
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
                "confidence": float(score),
            }
        )
    return detections


def detect_people_batch(detector, images, confidence, image_size):
    results = detector.predict(
        images,
        classes=[0],
        conf=confidence,
        imgsz=image_size,
        device=0,
        verbose=False,
    )
    return [result_to_detections(result) for result in results]
