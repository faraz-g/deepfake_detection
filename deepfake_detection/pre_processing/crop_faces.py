import pandas as pd
from deepfake_detection.defaults import DATA_PATH
import os
from multiprocessing.pool import Pool
from os import cpu_count
from functools import partial
import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from tqdm import tqdm
from pathlib import Path


def extract_crop_from_image(img_data, data_dir: str, out_dir: str):
    image_name = img_data[0]
    bboxes = img_data[1]
    img = cv2.imread(os.path.join(data_dir, image_name))
    if bboxes is None or len(bboxes) == 0:
        return None
    xmin, ymin, xmax, ymax = [int(b) for b in bboxes]
    w = xmax - xmin
    h = ymax - ymin
    p_h = h // 3
    p_w = w // 3
    crop = img[max(ymin - p_h, 0) : ymax + p_h, max(xmin - p_w, 0) : xmax + p_w]
    h, w = crop.shape[:2]

    cv2.imwrite(os.path.join(out_dir, f"{Path(image_name).stem}.png"), crop)


if __name__ == "__main__":
    OUT_DIR = DATA_PATH / "cropped_faces"
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_json(DATA_PATH / "combined_extracted_data.json", orient="index")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "image_path"}, inplace=True)
    img_data = list(zip(df["image_path"], df["boxes"]))
    with Pool(processes=cpu_count() - 2) as p:
        with tqdm(total=len(img_data)) as pbar:
            for v in p.imap_unordered(
                partial(extract_crop_from_image, data_dir=DATA_PATH, out_dir=OUT_DIR),
                img_data,
                chunksize=4,
            ):
                pbar.update()
