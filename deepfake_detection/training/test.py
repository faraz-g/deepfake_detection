import torch
from deepfake_detection.defaults import TEST_OUT_PATH, DATA_PATH, MODEL_OUT_PATH
import argparse
import os
from deepfake_detection.training.configurator import get_config
from deepfake_detection.training.training_utils import Classifier
from deepfake_detection.training.data_loader import DeepFakeDetectionDataset
from torch.utils.data.dataloader import DataLoader
from deepfake_detection.training.augmentations import test_augmentations
from tqdm import tqdm
import pandas as pd
from pathlib import Path


def test(checkpoint_path: str, config_name: str, model_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    config = get_config(config_name=config_name)
    model = Classifier(model_key=config.model_key)
    checkpoint_path = Path(checkpoint_path)
    saved_model = torch.load(model_dir / checkpoint_path)

    model.load_state_dict(saved_model["state"])

    data_test = DeepFakeDetectionDataset(
        mode="test",
        data_path="test.csv",
        data_folder_path=DATA_PATH,
        augmentations=test_augmentations(height=config.img_height, width=config.img_width),
    )
    test_loader = DataLoader(
        data_test,
        batch_size=config.batch_size * 2,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model = model.eval()
    out_dict = {"ground_truth": [], "predictions": []}
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing Model"):
            images = data[0]
            labels = data[1]

            images = images.to(device)

            out_labels = model(images)

            labels = labels.to("cpu").numpy().tolist()
            preds = torch.sigmoid(out_labels).squeeze(1).to("cpu").numpy().tolist()

            out_dict["predictions"].extend(preds)
            out_dict["ground_truth"].extend(labels)

    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(os.path.join(out_dir, f"{checkpoint_path.parent}.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--config_name", type=str, default="default_config")
    parser.add_argument("--out_dir", type=str, default=TEST_OUT_PATH)
    parser.add_argument("--model_dir", type=str, default=MODEL_OUT_PATH)

    args = parser.parse_args()

    test(**vars(args))
