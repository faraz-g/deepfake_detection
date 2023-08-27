import json
import os
from os import cpu_count
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
from facenet_pytorch.models.mtcnn import MTCNN, extract_face
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from deepfake_detection.defaults import DATA_PATH


def load_image(image_path: str) -> Image:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)    
    
    return image

class ImageDataset(Dataset):
    def __init__(self, images_data_path: str | Path, data_path: str | Path) -> None:
        self.images_df = pd.read_csv(os.path.join(data_path, images_data_path))
        self.drive_path = data_path

    def __getitem__(self, index: int) -> Any:
        image_name = self.images_df.iloc[index, 0]
        img_path = os.path.join(self.drive_path, image_name)
        image = load_image(image_path=img_path)

        return image, image_name

    def __len__(self) -> int:
        return len(self.images_df)
    
detector = MTCNN(select_largest=True, keep_all=False, post_process=False, device="cuda:0")

def draw_image(image, boxes, points):
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    for i, (box, point) in enumerate(zip(boxes, points)):
        draw.rectangle(box, width=5)
        extract_face(image, box, save_path='detected_face_{}.png'.format(i))
    img_draw.save('annotated_faces.png')

def detect_faces_in_images(
    images_data_path: str | Path,
    data_path: str | Path
):
    dataset = ImageDataset(images_data_path=images_data_path, data_path=data_path)
    data_loader = DataLoader(dataset=dataset, shuffle=False, num_workers=cpu_count() - 1, collate_fn=lambda x: x)
    
    all_results = {}
    for item in tqdm(data_loader):
        image, image_name = item[0]
        boxes, probs, points = detector.detect(image, landmarks=True)
        if boxes is not None:
            boxes = [box.tolist() if box is not None else box for box in boxes]
            boxes = boxes[0]
        if points is not None:
            points = [point.tolist() if point is not None else point for point in points]
            points = points[0]
        if probs is not None:
            probs = probs.tolist()
            probs = probs[0]
            
        image_result = {
            "boxes" : boxes,
            "probs": probs,
            "points" : points
        }
        
        all_results[image_name] = image_result

    return all_results

if __name__ == "__main__":
    data_path = DATA_PATH
    images_data_path = "images_data.csv"

    results = detect_faces_in_images(images_data_path=images_data_path, data_path=data_path)

    with open(DATA_PATH / "extracted_faces_two.json", 'w') as f:
        json.dump(results, f)