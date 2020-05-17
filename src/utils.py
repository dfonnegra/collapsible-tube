from os.path import splitext, join, isdir, islink, isfile
from os import listdir, unlink
import shutil
import numpy as np

CLEAN_IMG_DIR = "../clean_img"
GENERATED_CLEAN_DIR = "clean"
GENERATED_DIRTY_DIR = "dirty"
LITTLE_CIRCLE_SCALE = 80


def generate_path(dir_path, filename, suffix, index):
    name, ext = splitext(filename)
    train_val_test = np.random.choice(['train', 'validation', 'test'], p=[0.7, 0.15, 0.15])
    return f"../img/{train_val_test}/{join(dir_path, name)}_{suffix}_{index}{ext}"


def clean_dir(path):
    for filename in listdir(path):
        file_path = join(path, filename)
        try:
            if isfile(file_path) or islink(file_path):
                unlink(file_path)
            elif isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
