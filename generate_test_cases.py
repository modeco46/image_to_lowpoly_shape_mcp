"""
generate_test_cases.py — генератор тестовых изображений для проверки ретопологии.

Раздел 8 ТЗ:
1. S-лента с петлёй и ответвлением
2. X-перекрёстие (две линии толщиной 26 px)
3. Y-развилка (3 ленты по 120°)
4. Широкие ленты 55-60 px с 4 развилками на холсте 1000x1000
5. Синтетическая подвеска с 3 отверстиями
"""

import math
import cv2
import numpy as np


def make_x_cross(filename="test_x_cross.png", size=500, thickness=26):
    img = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    cv2.line(img, (50, 50), (size - 50, size - 50), 255, thickness)
    cv2.line(img, (50, size - 50), (size - 50, 50), 255, thickness)
    cv2.imwrite(filename, img)
    return filename


def make_y_junction(filename="test_y_junction.png", size=500, thickness=26):
    img = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = size // 2 - 50
    angles = [90, 210, 330]
    for deg in angles:
        rad = math.radians(deg)
        end_pt = (
            int(center[0] + radius * math.cos(rad)),
            int(center[1] + radius * math.sin(rad)),
        )
        cv2.line(img, center, end_pt, 255, thickness)
    cv2.imwrite(filename, img)
    return filename


def make_s_ribbon(filename="test_s_ribbon.png", size=600, thickness=30):
    img = np.zeros((size, size), dtype=np.uint8)
    pts = []
    for t in np.linspace(0, 2 * math.pi, 200):
        x = int(size / 2 + (size / 3) * math.sin(t))
        y = int(50 + (size - 100) * (t / (2 * math.pi)))
        pts.append((x, y))

    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], 255, thickness)

    branch_start = pts[len(pts) // 2]
    branch_end = (branch_start[0] + 120, branch_start[1] - 40)
    cv2.line(img, branch_start, branch_end, 255, thickness)

    cv2.circle(img, (pts[0][0] - 20, pts[0][1] - 20), 40, 255, thickness)

    cv2.imwrite(filename, img)
    return filename


def make_wide_ribbons_4junctions(filename="test_wide_ribbons.png", size=1000, thickness=58):
    img = np.zeros((size, size), dtype=np.uint8)
    j1 = (250, 250)
    j2 = (750, 250)
    j3 = (750, 750)
    j4 = (250, 750)

    cv2.line(img, j1, j2, 255, thickness)
    cv2.line(img, j2, j3, 255, thickness)
    cv2.line(img, j3, j4, 255, thickness)
    cv2.line(img, j4, j1, 255, thickness)

    cv2.line(img, j1, j3, 255, thickness)

    cv2.line(img, j1, (80, 80), 255, thickness)
    cv2.line(img, j2, (920, 80), 255, thickness)
    cv2.line(img, j3, (920, 920), 255, thickness)
    cv2.line(img, j4, (80, 920), 255, thickness)

    cv2.imwrite(filename, img)
    return filename


def make_pendant_3holes(filename="test_pendant_3holes.png", size=600):
    img = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.ellipse(img, center, (220, 260), 0, 0, 360, 255, -1)

    cv2.circle(img, (size // 2, size // 2 - 120), 45, 0, -1)
    cv2.circle(img, (size // 2 - 90, size // 2 + 60), 40, 0, -1)
    cv2.circle(img, (size // 2 + 90, size // 2 + 60), 40, 0, -1)

    cv2.imwrite(filename, img)
    return filename


def generate_all():
    files = [
        make_x_cross(),
        make_y_junction(),
        make_s_ribbon(),
        make_wide_ribbons_4junctions(),
        make_pendant_3holes(),
    ]
    print("Generated synthetic test files:", files)
    return files


if __name__ == "__main__":
    generate_all()
