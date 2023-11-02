from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import os
from PIL import Image
import cv2
from typing import Any
from albumentations.pytorch.functional import img_to_tensor
from albumentations import Compose
import torch


class DeepFakeDetectionDataset(Dataset):
    def __init__(
        self,
        mode: str,
        data_path: str | Path,
        data_folder_path: str | Path,
        augmentations: Compose | None,
        dataset: str = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        df = pd.read_csv(os.path.join(data_folder_path, data_path))
        df = df.reset_index()
        if dataset is not None:
            df = df[df["dataset"] == dataset]
        self.images_df = df[["cropped_face_path", "label", "dataset"]]
        self.data_folder_path = data_folder_path
        self.augmentations = augmentations

    def __getitem__(self, index: int) -> Any:
        image_data = self.images_df.iloc[index]
        cropped_face_path = image_data["cropped_face_path"]
        label = image_data["label"]
        image_path = os.path.join(self.data_folder_path, cropped_face_path)

        image = cv2.imread(image_path)

        if self.augmentations is not None:
            image = self.augmentations(image=image)["image"]

        # os.makedirs("test_outputs", exist_ok=True)
        # debug_out_path = os.path.join("test_outputs", f"{label}_{os.path.basename(cropped_face_path)}")
        # cv2.imwrite(debug_out_path, image)

        image = img_to_tensor(image)
        label = 1 if label.lower() == "fake" else 0
        label = torch.tensor(label, dtype=torch.float)

        return image, label

    def __len__(self) -> int:
        return len(self.images_df)
