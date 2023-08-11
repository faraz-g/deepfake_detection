import pandas as pd 
from deepfake_detection.defaults import DRIVE_PATH
import os
import math
from multiprocessing.pool import Pool
from os import cpu_count
from functools import partial
import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from tqdm import tqdm
import json
from pathlib import Path 

def extract_crop_from_image(
        image_path: str, 
        out_dir: str
    ):
        pass

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DRIVE_PATH / "metadata.csv")
    videos_paths = df['video_path'].apply(lambda x: str(DRIVE_PATH / x))
    videos_paths = videos_paths.to_list()

    results_map = {}
    with Pool(processes=cpu_count() - 2) as p:
        with tqdm(total=len(videos_paths)) as pbar:
            for v in p.imap_unordered(partial(extract_frames_from_video, number_of_frames_to_extract=NUM_FRAMES_TO_EXTRACT, out_dir=OUT_DIR), videos_paths, chunksize=4):
                pbar.update()
                results_map[v[0]] = v[1]
    
    save_results_map(DRIVE_PATH, results_map)