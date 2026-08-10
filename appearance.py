import warnings

import cv2
import numpy as np
import torch
from torch.nn import functional as torch_functional


def load_appearance_model(model_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from torchreid.reid.models import osnet_x0_25
        from torchreid.reid.utils import load_pretrained_weights

    model = osnet_x0_25(num_classes=4101, pretrained=False)
    load_pretrained_weights(model, model_path)
    model.eval()
    model.cuda()
    return model


def prepare_person_crop(image, detection):
    image_height, image_width = image.shape[:2]
    x1 = max(0, min(image_width - 1, int(detection["x1"])))
    y1 = max(0, min(image_height - 1, int(detection["y1"])))
    x2 = max(x1 + 1, min(image_width, int(detection["x2"])))
    y2 = max(y1 + 1, min(image_height, int(detection["y2"])))

    crop = image[y1:y2, x1:x2]
    crop = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = crop.astype(np.float32) / 255.0
    crop = (crop - np.array([0.485, 0.456, 0.406])) / np.array(
        [0.229, 0.224, 0.225]
    )
    return crop.transpose(2, 0, 1).astype(np.float32)


def extract_appearance_embeddings(model, image, detections):
    if not detections:
        return []

    crops = [prepare_person_crop(image, detection) for detection in detections]
    batch = torch.from_numpy(np.stack(crops)).cuda()

    with torch.inference_mode():
        embeddings = model(batch)
        embeddings = torch_functional.normalize(embeddings, dim=1)

    return [embedding.cpu().numpy() for embedding in embeddings]
