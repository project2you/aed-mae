import glob
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif"]

def compute_gradients(data_root_folder, step, folder):
    extension = None
    for ext in IMG_EXTENSIONS:
        if len(list(glob.glob(os.path.join(data_root_folder, f"{folder}/frames", f"*/*{ext}")))) > 0:
            extension = ext
            break

    if extension is None:
        print("No valid image extensions found.")
        return

    dirs = list(glob.glob(os.path.join(data_root_folder, folder, "frames", "*")))
    for video in tqdm(dirs, desc=f"Processing {folder} videos"):
        img_paths = list(glob.glob(os.path.join(video, f"*{extension}")))
        img_paths = sorted(img_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))

        for i, img_path in enumerate(img_paths):
            previous = max(0, i - step)
            next = min(len(img_paths) - 1, i + step)
            
            previous_img = cv2.imread(img_paths[previous])
            previous_img = previous_img.astype(np.int32)
            next_img = cv2.imread(img_paths[next])
            next_img = next_img.astype(np.int32)
            
            gradient = np.abs(previous_img - next_img).astype(np.uint8)
            os.makedirs(os.path.join(data_root_folder, f"{folder}/gradients2/{os.path.basename(video)}"), exist_ok=True)
            
            gradient = cv2.cvtColor(gradient, cv2.COLOR_BGR2RGB)
            Image.fromarray(gradient).save(os.path.join(data_root_folder, f"{folder}/gradients2/{os.path.basename(video)}", os.path.basename(img_path)))

if __name__ == "__main__":
    root_folder_avenue = "/notebooks/aed-mae/data/avenue"
    compute_gradients(root_folder_avenue, 1, "training")
    compute_gradients(root_folder_avenue, 1, "testing")
