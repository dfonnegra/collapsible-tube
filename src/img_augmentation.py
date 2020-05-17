from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from utils import generate_path, GENERATED_CLEAN_DIR, clean_dir, CLEAN_IMG_DIR, LITTLE_CIRCLE_SCALE
from dirt_img import dirt_img


def augment_flip(img):
    return [
        cv2.flip(img, 0),
        cv2.flip(img, 1),
        cv2.flip(img, -1)
    ]


def augment_rotation(img):
    rows, cols, _ = img.shape
    images = []
    for theta in range(0, 360, 20):
        M = cv2.getRotationMatrix2D((cols // 2, rows // 2), theta, 1)
        images.append(cv2.warpAffine(img, M, (cols, rows)))
    return images


def augment_brightness(img):
    images = []
    for gamma in (np.random.rand(20) * 1.5 + 0.5):
        images.append((255 * (img / 255) ** gamma).astype(np.uint8))
    return images


def init_image(img):
    img = cv2.resize(img, (640, 480))
    gray_img = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), sigmaX=2, sigmaY=2)
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 30, minRadius=15, maxRadius=120, param1=100, param2=50)
    if circles is None:
        return None
    rows, cols, _ = img.shape
    x, y, rad = circles[0][0]
    dx = int(x) - img.shape[1] // 2
    dy = int(y) - img.shape[0] // 2
    trans_img = np.zeros(img.shape)
    if dx > 0 and dy > 0:
        trans_img[:rows - dy, :cols - dx, :] = img[dy:, dx:, :]
    elif (dx < 0) and (dy > 0):
        trans_img[:rows - dy, -dx:, :] = img[dy:, :cols + dx, :]
    elif (dx > 0) and (dy < 0):
        trans_img[-dy:, :cols - dx, :] = img[:rows + dy, dx:, :]
    else:
        trans_img[:rows + dy, :cols + dx, :] = img[-dy:, -dx:, :]
    scale = LITTLE_CIRCLE_SCALE / rad
    new_rows, new_cols = int(round(scale * rows)), int(round(scale * cols))
    trans_img = cv2.resize(trans_img, (new_cols, new_rows))
    old_center = (cols // 2, rows // 2)
    new_center = (new_cols // 2, new_rows // 2)
    if scale > 1:
        start_range = (new_center[1] - old_center[1], new_center[0] - old_center[0])
        end_range = (start_range[0] + rows, start_range[1] + cols)
        result_img = trans_img[start_range[0]:end_range[0], start_range[1]:end_range[1], :]
    else:
        start_range = (-new_center[1] + old_center[1], -new_center[0] + old_center[0])
        end_range = (start_range[0] + new_rows, start_range[1] + new_cols)
        result_img = np.zeros(img.shape)
        result_img[start_range[0]:end_range[0], start_range[1]:end_range[1], :] = trans_img
    return result_img


augment_funcs = [augment_flip, augment_rotation, augment_brightness]


def augment_imgs(path):
    for filename in listdir(path):
        file_path = join(path, filename)
        if not isfile(file_path):
            continue
        img = init_image(cv2.imread(file_path))
        if img is None:
            continue
        images = [img]
        for augmentor in augment_funcs:
            new_imgs = []
            for img in images:
                new_imgs.extend(augmentor(img))
            images.extend(new_imgs)
        print(len(images))
        for index, img in enumerate(images):
            cv2.imwrite(generate_path(GENERATED_CLEAN_DIR, filename, "aug", index), img)


def dirt_images(path):
    types = ["train", "validation", "test"]
    for t in types:
        for filename in listdir(f"../img/{t}/{path}"):
            file_path = join(f"../img/{t}/{path}", filename)
            if not isfile(file_path):
                continue
            img = cv2.imread(file_path)
            if img is not None:
                dirt_img(img, filename)


if __name__ == "__main__":
    types = ["train", "validation", "test"]
    for t in types:
       clean_dir(f"../img/{t}/clean")
       clean_dir(f"../img/{t}/dirty")
    augment_imgs(CLEAN_IMG_DIR)
    dirt_images(GENERATED_CLEAN_DIR)
