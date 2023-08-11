from facenet_pytorch.models.mtcnn import MTCNN
import json
from os import cpu_count

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from deepfake_detection.data.image_dataset import ImageDataset
from deepfake_detection.defaults import DATA_PATH
from pathlib import Path

def detect_faces_in_images(
    images_data_path: str | Path,
    data_path: str | Path
):
    detector = MTCNN(select_largest=True, keep_all=False, post_process=False, device="cuda:0")
    dataset = ImageDataset(images_data_path=images_data_path, data_path=data_path)
    data_loader = DataLoader(dataset=dataset, shuffle=False, num_workers=cpu_count() - 2, collate_fn=lambda x: x)
    
    all_results = []
    for item in tqdm(data_loader):
        image, image_name = item[0]
        boxes, probs, points = detector.detect(image, landmarks=True)
        if boxes is not None:
            boxes = [box.tolist() if box is not None else box for box in boxes]
        if points is not None:
            points = [point.tolist() if point is not None else point for point in points]
        
        image_result = {
            image_name : {
                "boxes" : boxes,
                "probs": probs,
                "points" : points
            }
        }
        all_results.append(image_result)

    return all_results

if __name__ == "__main__":
    data_path = DATA_PATH
    images_data_path = "images_data.csv"

    results = detect_faces_in_images(images_data_path=images_data_path, data_path=data_path)

    with open(DATA_PATH / "extracted_faces.json", 'w') as f:
        json.dump(results, f)