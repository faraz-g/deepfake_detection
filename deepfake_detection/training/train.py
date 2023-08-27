import torch
from deepfake_detection.defaults import DATA_PATH, MODEL_OUT_PATH
import argparse
import os 
from deepfake_detection.training.configurator import get_config, TrainingConfig
from deepfake_detection.training.training_utils import Classifier, get_optimizer, get_scheduler
from torch.nn.modules.loss import BCEWithLogitsLoss
from deepfake_detection.training.data_loader import DeepFakeDetectionDataset
from torch.utils.data.dataloader import DataLoader
from deepfake_detection.training.augmentations import train_augmentations, val_augmentations
import json

torch.backends.cudnn.benchmark = True

class LossTracker:
    def __init__(self) -> None:
        self.initialise()
    
    def initialise(self):
        self.loss_value = 0
        self.average = 0
        self.sum = 0
        self.count_iters = 0
    
    def update(self, loss_value: float, iterations: int):
        self.loss_value = loss_value
        self.sum += loss_value * iterations
        self.count_iters += iterations
        self.average = self.sum / self.count_iters

def _single_epoch(
    epoch: int,
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler.LRScheduler, 
    loss_function: BCEWithLogitsLoss,
    train_loader: DataLoader, 
    config: TrainingConfig,
    device: torch.device
):
    fake_loss_tracker = LossTracker()
    real_loss_tracker = LossTracker()
    all_loss_tracker = LossTracker()

    max_batches = config.batches_per_epoch

    model = model.train()

    for i, data in enumerate(train_loader):
        images = data[0]
        labels = data[1]

        num_images = images.size(0)

        images = images.to(device)
        labels = labels.to(device)

        out_labels = model(images)
        out_labels = out_labels.squeeze(1)
        fake_image_indexes = labels == 1
        real_image_indexes = labels == 0

        if torch.sum(fake_image_indexes * 1) > 0:
            fake_loss = loss_function(out_labels[fake_image_indexes], labels[fake_image_indexes])
        else:
            fake_loss = 0

        if torch.sum(real_image_indexes * 1) > 0:
            real_loss = loss_function(out_labels[real_image_indexes], labels[real_image_indexes])
        else:
            real_loss = 0

        all_loss = (fake_loss + real_loss) / 2

        all_loss_tracker.update(all_loss.item(), num_images)
        real_loss_tracker.update(0 if real_loss == 0 else real_loss.item(), num_images)
        fake_loss_tracker.update(0 if fake_loss == 0 else fake_loss.item(), num_images)

        print(f"E:{epoch}I:{i} LR: {scheduler.get_lr()[0]} Total Loss: {all_loss_tracker.average} Real Loss: {real_loss_tracker.average} Fake Loss: {fake_loss_tracker.average}")
        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()
        scheduler.step()    

        if i + 1 == max_batches:
            break

def _evaluate(
    epoch: int,
    model: torch.nn.Module,
    val_loader: DataLoader, 
):
    model = model.eval()
    # TODO Implement this.

def train(
    resume: str,
    prefix: str,
    config_name: str,
    data_dir: str,
    out_dir: str 
):
    os.makedirs(out_dir, exist_ok=True)
    config = get_config(config_name=config_name)

    print("Training with config: \n", json.dumps(config.model_dump(), indent=1))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device} for training")

    model = Classifier(model_key=config.model_key)
    model = model.to(device)
    loss_function = BCEWithLogitsLoss().to(device)

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

    data_val = DeepFakeDetectionDataset(
        mode="val",
        data_path="val.csv",
        data_folder_path=DATA_PATH,
        augmentations=val_augmentations(height=config.img_height, width=config.img_width)
    )    
    val_loader = DataLoader(
        data_val, 
        batch_size=config.batch_size * 2, 
        pin_memory=False,
        shuffle=False,
        drop_last=True
    )

    start_epoch = 1
    max_epoch = config.max_epochs

    for epoch in range(start_epoch, max_epoch):
        print(f"Epoch: {epoch}")
        _single_epoch(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            train_loader=train_loader,
            config=config,
            device=device
        )

        out_name = f"{prefix}{config.model_key}_{epoch}"
        out_path = os.path.join(out_dir, out_name)
        torch.save({"epoch": epoch, "state": model.state_dict()}, out_path)     
        if epoch % config.evaluation_frequency == 0:
            _evaluate(
                epoch=epoch,
                model=model,
                val_loader=val_loader
            )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--prefix', default='', type=str)
    parser.add_argument('--config_name', type=str, default="default_config")
    parser.add_argument('--data_dir', type=str, default=DATA_PATH)
    parser.add_argument('--out_dir', type=str, default=MODEL_OUT_PATH)

    args = parser.parse_args()

    train(**vars(args))
