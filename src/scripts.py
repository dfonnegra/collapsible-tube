import multiprocessing
import os
import shutil
import time

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from img_preprocessing import (
    dirt_img,
    get_center_circle,
    mask_circle,
    mask_circle_and_wrap_polar,
)


def _aux_generate_sets(filenames, train_size, dev_size):
    train_files, test_files = train_test_split(filenames, train_size=train_size)
    dev_files, test_files = train_test_split(
        test_files, train_size=(dev_size / (1 - train_size))
    )
    return train_files, dev_files, test_files


def generate_train_dev_test_sets(train_size=0.8, dev_size=0.1, with_preprocess=False):
    for src_folder, dst_folder in zip(["Limpias", "Sucias"], ["clean", "dirty"]):
        src_dir_path = f"../img/{src_folder}"
        filenames = np.array(os.listdir(src_dir_path))
        train_files, dev_files, test_files = _aux_generate_sets(
            filenames, train_size, dev_size
        )
        for file_list, folder in zip(
            [train_files, dev_files, test_files], ["train", "dev", "test"]
        ):
            dst_dir = f"../img/{folder}/{dst_folder}/"
            shutil.rmtree(dst_dir)
            os.mkdir(dst_dir)
            for file in file_list:
                src_path = os.path.join(src_dir_path, file)
                dst_path = os.path.join(dst_dir, file)
                if not with_preprocess:
                    shutil.copyfile(src_path, dst_path)
                else:
                    img = np.asarray(Image.open(src_path))
                    img = cv2.resize(
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR), dsize=(640, 480)
                    )
                    img = mask_circle(img)
                    cv2.imwrite(dst_path, img)


def generate_train_dev_test_dirt_images():
    for folder in ["train", "dev", "test"]:
        [
            os.remove(f"../img/{folder}/dirty/{filename}")
            for filename in os.listdir(f"../img/{folder}/dirty")
            if filename.startswith("FAKE_")
        ]
        dir_path = f"../img/{folder}/clean"
        for filename in os.listdir(dir_path):
            if np.random.choice([True, False], p=[0.9, 0.1]):
                continue
            file_path = f"{dir_path}/{filename}"
            dst_path = f"../img/{folder}/dirty/FAKE_{filename}"
            cv2.imwrite(dst_path, dirt_img(cv2.imread(file_path)))


def compute_dataset_stats():
    paths = [f"../img/Limpias/{file}" for file in os.listdir("../img/Limpias/")]
    start_t = time.time()
    x_pos, y_pos, rads = [], [], []
    for index, path in enumerate(paths):
        try:
            x, y, rad = get_center_circle(cv2.imread(path))
            x_pos.append(x)
            y_pos.append(y)
            rads.append(rad)
        except TypeError:
            pass
        if index % 500 == 0:
            print(
                f"Remaining time: {int((time.time() - start_t) / (index + 1) * (len(paths) - index - 1))}s"
            )
    x_pos, y_pos, rads = (
        np.array(x_pos),
        np.array(y_pos),
        np.array(rads),
    )

    print(f"x = {x_pos.mean()} +- {x_pos.std()}")
    print(f"y = {y_pos.mean()} +- {y_pos.std()}")
    print(f"r = {rads.mean()} +- {rads.std()}")


def compute_error(params):
    circles = []

    paths = [
        f"../img/Limpias/{path}"
        for path in os.listdir("../img/Limpias")
        if np.random.choice([True, False], p=[0.5, 0.5])
    ]

    for path in paths:
        try:
            img = cv2.cvtColor(np.asarray(Image.open(path)), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, dsize=(640, 480))
            circles.append(get_center_circle(img, **params))
        except TypeError:
            pass
    error = (
        np.std(circles, axis=0).sum() * (len(paths) / (len(circles) + 0.000001)) ** 2
    )
    if len(circles) < 5:
        error = 1e9
    print(
        f"Error for param1: {params['param1']} and param2: {params['param2']} is: {error}"
    )
    return error


def compute_hough_circles_params():
    param_set = [
        {
            "param1": int(np.random.rand() * 200) + 1,
            "param2": int(np.random.rand() * 70) + 1,
        }
        for _ in range(104)
    ]
    with multiprocessing.Pool(processes=8) as pool:
        errors = pool.map(compute_error, param_set)
    best_params = param_set[np.argmin(errors)]
    print(best_params)
    print(errors)


if __name__ == "__main__":
    generate_train_dev_test_sets(with_preprocess=True)
    # compute_hough_circles_params()
    # compute_dataset_stats()
    # generate_train_dev_test_dirt_images()
