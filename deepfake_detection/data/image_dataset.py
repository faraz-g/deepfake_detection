from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd 
import os 
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, images_data_path: str | Path, data_path: str | Path) -> None:
        self.images_df = pd.read_csv(os.path.join(data_path, images_data_path))
        self.drive_path = data_path

    def __getitem__(self, index: int):
        image_name = self.images_df.iloc[index, 0]
        img_path = os.path.join(self.drive_path, image_name)
        image = Image.open(img_path)

        return image, image_name

    def __len__(self) -> int:
        return len(self.images_df)