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

def extract_frames_from_video(
        video_path: str, 
        number_of_frames_to_extract: int,
        out_dir: str
    ):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_to_extract = [math.floor(i * total_frames / number_of_frames_to_extract) for i in range(number_of_frames_to_extract)]
        
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    saved_frame_paths = []
    for frame_number in frames_to_extract:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        if not success:
            continue
        
        file_name = f"{video_base_name}_{frame_number}.jpg"
        out_path = os.path.join(out_dir, file_name)
        if not os.path.exists(out_path):
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        saved_frame_paths.append(out_path)

    return video_path, saved_frame_paths

def save_results_map(out_dir, results):
    with open(out_dir / "exported_images.json", 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    OUT_DIR = Path('D:\exported_images')
    NUM_FRAMES_TO_EXTRACT = 2

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