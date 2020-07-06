import cv2
import numpy as np

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


def get_center_circle(img):
    cvt_img = img
    if len(cvt_img.shape) == 3:
        cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cvt_img = cv2.GaussianBlur(cvt_img, (11, 11), sigmaX=2, sigmaY=2)
    circles = cv2.HoughCircles(cvt_img, cv2.HOUGH_GRADIENT, 1, 10, minRadius=40)
    radius_list = circles[0, :, 2]
    min_radius = np.min(radius_list)
    main_radius = min_radius
    main_circle = circles[0, np.where(radius_list == main_radius)[0][0], :]
    x, y, rad = main_circle
    return x, y, rad
