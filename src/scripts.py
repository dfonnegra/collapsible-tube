import os
import shutil
import time

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from img_preprocessing import get_center_circle


def _aux_generate_sets(filenames, train_size, dev_size):
    train_files, test_files = train_test_split(filenames, train_size=train_size)
    dev_files, test_files = train_test_split(
        test_files, train_size=(dev_size / (1 - train_size))
    )
    return train_files, dev_files, test_files


def generate_train_dev_test_sets(train_size=0.8, dev_size=0.1, with_preprocess=False):
    for src_folder, dst_folder in zip(
        ["Limpias", "Raras", "Sucias"], ["clean", "rare", "dirty"]
    ):
        src_dir_path = f"../img/{src_folder}"
        filenames = np.array(os.listdir(src_dir_path))
        train_files, dev_files, test_files = _aux_generate_sets(
            filenames, train_size, dev_size
        )
        for file_list, folder in zip(
            [train_files, dev_files, test_files], ["train", "dev", "test"]
        ):
            [
                shutil.copyfile(
                    os.path.join(src_dir_path, file),
                    f"../img/{folder}/{dst_folder}/{file}",
                )
                for file in file_list
            ]


def compute_dataset_stats():
    paths = [f"../img/train/clean/{file}" for file in os.listdir("../img/train/clean/")]
    paths.extend(
        f"../img/train/dirty/{file}" for file in os.listdir("../img/train/dirty/")
    )
    start_t = time.time()
    x_pos, y_pos, small_rads, med_rads, big_rads = [], [], [], [], []
    for index, path in enumerate(paths):
        try:
            x, y, rad = get_center_circle(cv2.imread(path))
            x_pos.append(x)
            y_pos.append(y)
            if rad > 180:
                big_rads.append(rad)
            elif rad > 80:
                med_rads.append(rad)
            else:
                small_rads.append(rad)
        except TypeError:
            pass
        if index % 500 == 0:
            print(
                f"Remaining time: {int((time.time() - start_t) / (index + 1) * (len(paths) - index - 1))}s"
            )
    x_pos, y_pos, small_rads, med_rads, big_rads = (
        np.array(x_pos),
        np.array(y_pos),
        np.array(small_rads),
        np.array(med_rads),
        np.array(big_rads),
    )

    print(f"x = {x_pos.mean()} +- {x_pos.std()}")
    print(f"y = {y_pos.mean()} +- {y_pos.std()}")
    print(f"r = {small_rads.mean()} +- {small_rads.std()}")
    print(f"r = {med_rads.mean()} +- {med_rads.std()}")
    print(f"r = {big_rads.mean()} +- {big_rads.std()}")


if __name__ == "__main__":
    compute_dataset_stats()
