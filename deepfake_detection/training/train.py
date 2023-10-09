import torch
from deepfake_detection.defaults import DATA_PATH, MODEL_OUT_PATH
import argparse
import os
from deepfake_detection.training.configurator import get_config, TrainingConfig
from deepfake_detection.training.training_utils import (
    Classifier,
    get_optimizer,
    get_scheduler,
)
from torch.nn.modules.loss import BCEWithLogitsLoss
from deepfake_detection.training.data_loader import DeepFakeDetectionDataset
from torch.utils.data.dataloader import DataLoader
from deepfake_detection.training.augmentations import (
    train_augmentations,
    val_augmentations,
)
import json
from tqdm import tqdm
from sklearn.metrics import log_loss
import numpy as np

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
    device: torch.device,
):
    fake_loss_tracker = LossTracker()
    real_loss_tracker = LossTracker()
    all_loss_tracker = LossTracker()

    max_batches = config.batches_per_epoch

    model = model.train()

    with tqdm(total=(max_batches), desc=f"Training Epoch: {epoch}", ncols=0) as pbar:
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

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix_str(
                f"LR: {scheduler.get_last_lr()[0]:.4f} Total Loss: {all_loss_tracker.average:.4f} Real Loss: {real_loss_tracker.average:.4f} Fake Loss: {fake_loss_tracker.average:.4f}"
            )
            pbar.update()

            if i + 1 == max_batches:
                break


def _evaluate(
    epoch: int,
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
):
    model = model.eval()

    predictions = []
    ground_truth = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validating Epoch: {epoch}", ncols=0)
        for data in pbar:
            images = data[0]
            labels = data[1]

            images = images.to(device)
            out_labels = model(images)

            labels = labels.to("cpu").numpy().tolist()
            preds = torch.sigmoid(out_labels).squeeze(1).to("cpu").numpy().tolist()

            predictions.extend(preds)
            ground_truth.extend(labels)

    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)

    fake_image_indexes = ground_truth == 1
    real_image_indexes = ground_truth == 0

    fake_loss = log_loss(ground_truth[fake_image_indexes], predictions[fake_image_indexes], labels=[0, 1])
    real_loss = log_loss(ground_truth[real_image_indexes], predictions[real_image_indexes], labels=[0, 1])

    combined_loss = (fake_loss + real_loss) / 2

    return combined_loss


def train(resume: str, prefix: str, config_name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    config = get_config(config_name=config_name)
    print("Training with config: \n", json.dumps(config.model_dump(), indent=1))
    model = Classifier(model_key=config.model_key)

    if resume:
        saved_model = torch.load(os.path.join(out_dir, resume))
        start_epoch = saved_model["epoch"] + 1
        model.load_state_dict(saved_model["state"])
        best_val_loss = saved_model["best_val_loss"]
    else:
        start_epoch = 1
        best_val_loss = float("inf")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for training")
    model = model.to(device)
    loss_function = BCEWithLogitsLoss().to(device)

    optimizer = get_optimizer(
        model=model,
        optim_type=config.optim_config.optim_type,
        lr=config.optim_config.learning_rate,
        momentum=config.optim_config.momentum,
        weight_decay=config.optim_config.weight_decay,
    )

    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_type=config.scheduler_config.scheduler_type,
        params=config.scheduler_config.params,
    )

    data_train = DeepFakeDetectionDataset(
        mode="train",
        data_path="train.csv",
        data_folder_path=DATA_PATH,
        augmentations=train_augmentations(height=config.img_height, width=config.img_width),
    )
    train_loader = DataLoader(
        data_train,
        batch_size=config.batch_size,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )

    data_val = DeepFakeDetectionDataset(
        mode="val",
        data_path="val.csv",
        data_folder_path=DATA_PATH,
        augmentations=val_augmentations(height=config.img_height, width=config.img_width),
    )
    val_loader = DataLoader(
        data_val,
        batch_size=config.batch_size * 2,
        pin_memory=False,
        shuffle=False,
        drop_last=True,
    )

    max_epoch = config.max_epochs

    out_name = f"{prefix}_{config.model_key}"
    out_path = os.path.join(out_dir, out_name)
    os.makedirs(out_path, exist_ok=True)
    if len(os.listdir(out_path)) > 0 and not resume:
        raise NotImplementedError("Need to resume if out_dir is not empty")

    epochs_since_best_loss = 0
    for epoch in range(start_epoch, max_epoch):
        _single_epoch(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        if epoch % config.evaluation_frequency == 0:
            validation_loss = _evaluate(
                epoch=epoch,
                model=model,
                val_loader=val_loader,
                device=device,
            )
            if validation_loss < best_val_loss:
                epochs_since_best_loss = 0
                best_val_loss = validation_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "state": model.state_dict(),
                        "best_val_loss": best_val_loss,
                    },
                    os.path.join(out_path, "best"),
                )
            else:
                epochs_since_best_loss += 1

            print(f"Validation Loss at epoch {epoch}: {validation_loss}. Best Loss: {best_val_loss}")
        torch.save(
            {
                "epoch": epoch,
                "state": model.state_dict(),
                "best_val_loss": best_val_loss,
            },
            os.path.join(out_path, f"{epoch}"),
        )
        torch.save(
            {
                "epoch": epoch,
                "state": model.state_dict(),
                "best_val_loss": best_val_loss,
            },
            os.path.join(out_path, "last"),
        )
        if epochs_since_best_loss > config.early_stopping_threshold:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--prefix", default="", type=str)
    parser.add_argument("--config_name", type=str, default="default_config")
    parser.add_argument("--out_dir", type=str, default=MODEL_OUT_PATH)

    args = parser.parse_args()

    train(**vars(args))
