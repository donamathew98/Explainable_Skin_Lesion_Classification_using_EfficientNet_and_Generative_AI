# src/preprocess.py

import os
import shutil
import pandas as pd

def organise_dataset():
    metadata_path = "data/metadata.csv"
    image_dir = "data/all_images/"
    output_dir = "data/images/"

    df = pd.read_csv(metadata_path)

    for _, row in df.iterrows():
        label = row['dx']
        img_name = row['image_id'] + ".jpg"

        src = os.path.join(image_dir, img_name)
        dst_dir = os.path.join(output_dir, label)

        os.makedirs(dst_dir, exist_ok=True)

        if os.path.exists(src):
            shutil.copy(src, os.path.join(dst_dir, img_name))

if __name__ == "__main__":
    organise_dataset()