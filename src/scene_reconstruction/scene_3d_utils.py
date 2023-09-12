import cv2
import numpy as np


def draw_lines(img1, img2, lines, pts1, pts2):
    """
    Draw lines between two images given two points.
    :param img1: Gray scale image
    :param img2: Gray scale image
    :param lines: Corresponding epilines
    :param pts1: Points form first images
    :param pts2: Points from second images
    :return: Stereo images
    """
    r, c, _ = img1.shape

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[0] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c / r[1])])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2
