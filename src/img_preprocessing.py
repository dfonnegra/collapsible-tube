import cv2
import numpy as np

import settings
from utils import LITTLE_CIRCLE_SCALE


def generate_random_color():
    return [np.random.randint(0, 255) for _ in range(3)]


def generate_dust(img):
    dust_num = np.random.randint(0, 100)
    random_dust = np.random.rand(dust_num, 3)
    img_copy = img.copy()
    rows, cols, _ = img.shape
    dust_color = generate_random_color()
    for dust in random_dust:
        x, y, rad = dust
        x = int(x * cols)
        y = int(y * rows)
        rad = int(rad * 6)
        cv2.circle(img_copy, (x, y), rad, color=dust_color, thickness=cv2.FILLED)
    return img_copy


def get_random_point_with_dist_ref(bias_x, bias_y, dist_max, dist_min=0):
    x_start = int(np.random.rand() * dist_max)
    if x_start < dist_min:
        sign = np.random.choice([-1, 1])
        y_start = int(
            np.sqrt(dist_min ** 2 - x_start ** 2)
            + (
                np.sqrt(dist_max ** 2 - x_start ** 2)
                - np.sqrt(dist_min ** 2 - x_start ** 2)
            )
            * np.random.rand()
        )
        if sign < 0:
            y_start = int(
                -np.sqrt(dist_max ** 2 - x_start ** 2)
                + (
                    np.sqrt(dist_min ** 2 - x_start ** 2)
                    - np.sqrt(dist_max ** 2 - x_start ** 2)
                )
                * np.random.rand()
            )
    else:
        y_start = int(
            np.random.choice([-1, 1])
            * np.random.rand()
            * np.sqrt(dist_max ** 2 - x_start ** 2)
        )
    return x_start + bias_x, y_start + bias_y


def generate_noise(img):
    rows, cols, _ = img.shape
    shape_type = np.random.choice(["dust", "circle", "rect", "poly"])
    figure_color = generate_random_color()
    filled = np.random.choice([True, False])
    img_copy = img.copy()
    x_start, y_start = get_random_point_with_dist_ref(
        cols // 2, rows // 2, LITTLE_CIRCLE_SCALE + 50, LITTLE_CIRCLE_SCALE
    )
    if shape_type == "dust":
        img_copy = generate_dust(img_copy)
    elif shape_type == "circle":
        radius = np.random.randint(10, 25)
        thickness = cv2.FILLED if filled else np.random.randint(3, 10)
        cv2.circle(
            img_copy, (x_start, y_start), radius, figure_color, thickness=thickness
        )
    elif shape_type == "rect":
        thickness = cv2.FILLED if filled else np.random.randint(3, 10)
        x_end, y_end = get_random_point_with_dist_ref(
            x_start, y_start, LITTLE_CIRCLE_SCALE // 3
        )
        cv2.rectangle(
            img_copy, (x_start, y_start), (x_end, y_end), figure_color, thickness
        )
    elif shape_type == "poly":
        num_of_points = np.random.randint(2, 5)
        points = [[x_start, y_start]]
        for _ in range(num_of_points):
            points.append(
                get_random_point_with_dist_ref(*points[-1], LITTLE_CIRCLE_SCALE // 3)
            )
        points = np.array(points, np.int32).reshape(-1, 1, 2)
        if filled:
            cv2.fillPoly(img_copy, [points], figure_color)
        else:
            thickness = np.random.randint(3, 10)
            cv2.polylines(img_copy, [points], False, figure_color, thickness)
    return img_copy


def dirt_img(img):
    return generate_noise(img)


def get_center_circle(img, **hough_params):
    cvt_img = img
    if len(img.shape) == 3:
        cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cvt_img = cv2.GaussianBlur(cvt_img, (11, 11), sigmaX=2, sigmaY=2)
    cvt_img = 255 - cvt_img
    hough_params.setdefault("param1", settings.PARAM1_HOUGH)
    hough_params.setdefault("param2", settings.PARAM2_HOUGH)
    circles = cv2.HoughCircles(
        cvt_img, cv2.HOUGH_GRADIENT, 1, 10, minRadius=45, maxRadius=55, **hough_params
    )
    # cv2.circle(
    #     img, (circles[0, 0, 0], circles[0, 0, 1]), circles[0, 0, 2], (0, 0, 255), 5
    # )
    return circles[0][0]


def translate_image(img, dx, dy):
    new_img = np.zeros(img.shape, dtype=np.uint8) + img.mean(axis=(0, 1)).astype(
        np.uint8
    )

    if dx > 0 and dy > 0:
        new_img[dy:, dx:] = img[:-dy, :-dx]
    elif dx > 0 and dy < 0:
        new_img[:dy, dx:] = img[-dy:, :-dx]
    elif dx < 0 and dy > 0:
        new_img[dy:, :dx] = img[:-dy, -dx:]
    elif dx < 0 and dy < 0:
        new_img[:dy, :dx] = img[-dy:, -dx:]
    elif dx == 0 and dy > 0:
        new_img[dy:, :] = img[:-dy, :]
    elif dx == 0 and dy < 0:
        new_img[:dy, :] = img[-dy:, :]
    elif dx > 0 and dy == 0:
        new_img[:, dx:] = img[:, :-dx]
    elif dx < 0 and dy == 0:
        new_img[:, :dx] = img[:, -dx:]
    else:
        new_img = img
    return new_img


def mask_circle(img):
    try:
        x, y, rad = get_center_circle(img)
        if abs(rad - settings.SMALL_RADIUS_SIZE) < abs(rad - settings.MED_RADIUS_SIZE):
            rad = settings.BIG_RADIUS_SIZE * settings.SMALL_RADIUS_SIZE / rad
        elif abs(rad - settings.MED_RADIUS_SIZE) < abs(rad - settings.BIG_RADIUS_SIZE):
            rad = settings.BIG_RADIUS_SIZE * settings.MED_RADIUS_SIZE / rad
    except TypeError:
        x, y = settings.AVERAGE_CENTER
        rad = settings.BIG_RADIUS_SIZE
    x, y, rad = int(x), int(y), int(settings.CIRCLE_PERC_FILTER * rad)
    dim = settings.OUTPUT_IMG_DIMENSION
    img = translate_image(img, img.shape[1] // 2 - x, img.shape[0] // 2 - y)
    y_dim = min(img.shape[0] // 2 + rad, img.shape[0]) - max(img.shape[0] // 2 - rad, 0)
    x_dim = min(img.shape[1] // 2 + rad, img.shape[1]) - max(img.shape[1] // 2 - rad, 0)
    filter_dim = min(x_dim, y_dim)
    img = img[
        img.shape[0] // 2 - filter_dim // 2 : img.shape[0] // 2 + filter_dim // 2,
        img.shape[1] // 2 - filter_dim // 2 : img.shape[1] // 2 + filter_dim // 2,
    ]
    circle_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(circle_mask, (img.shape[1] // 2, img.shape[0] // 2), rad, 255, -1)
    img_mean = img.mean(axis=(0, 1))
    img = cv2.bitwise_and(img, img, mask=circle_mask)
    background = cv2.bitwise_or(img, img_mean, mask=cv2.bitwise_not(circle_mask))
    img = cv2.bitwise_or(img, background)
    return cv2.resize(img, dsize=(dim, dim))


def mask_circle_and_wrap_polar(img):
    try:
        x, y, rad = get_center_circle(img)
        if abs(rad - settings.SMALL_RADIUS_SIZE) < abs(rad - settings.MED_RADIUS_SIZE):
            rad = settings.BIG_RADIUS_SIZE / settings.SMALL_RADIUS_SIZE * rad
        elif abs(rad - settings.MED_RADIUS_SIZE) < abs(rad - settings.BIG_RADIUS_SIZE):
            rad = settings.BIG_RADIUS_SIZE / settings.MED_RADIUS_SIZE * rad
    except TypeError:
        x, y = settings.AVERAGE_CENTER
        rad = settings.BIG_RADIUS_SIZE
    dim = settings.OUTPUT_IMG_DIMENSION
    return cv2.warpPolar(
        img,
        (dim, dim),
        (x, y),
        int(settings.CIRCLE_PERC_FILTER * rad),
        cv2.WARP_POLAR_LINEAR,
    )
