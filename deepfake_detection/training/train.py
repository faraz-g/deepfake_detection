from deepfake_detection.defaults import DATA_PATH, MODEL_OUT_PATH
import argparse
import os 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--data_dir', type=str, default=DATA_PATH)
    parser.add_argument('--out_dir', type=str, default=MODEL_OUT_PATH)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
