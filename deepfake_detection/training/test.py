import torch
from deepfake_detection.defaults import TEST_OUT_PATH, DATA_PATH
import argparse
import os 
from deepfake_detection.training.configurator import get_config, TrainingConfig
from deepfake_detection.training.training_utils import Classifier, get_optimizer, get_scheduler
from torch.nn.modules.loss import BCEWithLogitsLoss
from deepfake_detection.training.data_loader import DeepFakeDetectionDataset
from torch.utils.data.dataloader import DataLoader
from deepfake_detection.training.augmentations import train_augmentations, val_augmentations
import json
from tqdm import tqdm
from sklearn.metrics import log_loss
import numpy as np 


def test(
    saved_model_path: str,
    config_name: str,
    out_dir: str
):
    os.makedirs(out_dir, exist_ok=True)
    config = get_config(config_name=config_name)
    model = Classifier(model_key=config.model_key)
    saved_model = torch.load(saved_model_path)

    model.load_state_dict(saved_model['state'])

    data_val = DeepFakeDetectionDataset(
        mode="val",
        data_path="val.csv",
        data_folder_path=DATA_PATH,
        augmentations=val_augmentations(height=config.img_height, width=config.img_width)
    )    
    val_loader = DataLoader(
        data_val, 
        batch_size=config.batch_size, 
        pin_memory=False,
        shuffle=True,
        drop_last=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model_path', type=str)
    parser.add_argument('--config_name', type=str)
    parser.add_argument('--out_dir', type=str, default=TEST_OUT_PATH)

    args = parser.parse_args()

    test(**vars(args))
