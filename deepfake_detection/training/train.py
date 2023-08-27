import torch
from deepfake_detection.defaults import DATA_PATH, MODEL_OUT_PATH
import argparse
import os 
from deepfake_detection.training.configurator import get_config
from deepfake_detection.training.training_utils import Classifier, get_optimizer, get_scheduler
from torch.nn.modules.loss import BCEWithLogitsLoss
from deepfake_detection.training.data_loader import DeepFakeDetectionDataset
from torch.utils.data.dataloader import DataLoader
from deepfake_detection.training.augmentations import train_augmentations, val_augmentations


def train(
    num_epochs: int,
    resume: str,
    config_name: str,
    data_dir: str,
    out_dir: str 
):
    os.makedirs(out_dir, exist_ok=True)
    config = get_config(config_name=config_name)

    print("Training config:\n", config)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device} for training")

    model = Classifier(model_key=config.model_key)
    model = model.to(device)
    loss = BCEWithLogitsLoss().to(device)

    optimizer = get_optimizer(
        model=model, 
        optim_type=config.optim_config.optim_type,
        lr=config.optim_config.learning_rate,
        momentum=config.optim_config.momentum,
        weight_decay=config.optim_config.weight_decay
    )

    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_type=config.scheduler_config.scheduler_type,
        params=config.scheduler_config.params
    )

    loss = 1

    batch_size = config.batch_size

    data_train = DeepFakeDetectionDataset(
        mode="train",
        data_path="train.csv",
        data_folder_path=DATA_PATH,
        augmentations=train_augmentations(height=config.img_height, width=config.img_width)
    )    
    train_loader = DataLoader(
        data_train, 
        batch_size=config.batch_size, 
        pin_memory=False,
        shuffle=True,
        drop_last=True
    )

    start_epoch = 0
    max_epoch = config.max_epochs
    for data in train_loader:
        images = data[0]
        labels = data[1]

        images = images.to(device)
        labels = labels.to(device)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--config_name', type=str, default="default_config")
    parser.add_argument('--data_dir', type=str, default=DATA_PATH)
    parser.add_argument('--out_dir', type=str, default=MODEL_OUT_PATH)

    args = parser.parse_args()

    train(**vars(args))
