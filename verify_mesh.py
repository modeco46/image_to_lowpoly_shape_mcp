"""Строгая проверка качества сетки. Используется для приёмки, а не для
"на глаз посмотрели и вроде норм"."""

import sys
from itertools import combinations

import numpy as np


def _seg_cross(p1, p2, p3, p4, eps=1e-12) -> bool:
    """Строгое пересечение отрезков.

    eps обязателен: у коллинеарных рёбер (а их в квадовой сетке много —
    соседние квады полосы часто дают рёбра на одной прямой) величины
    ориентации равны нулю с точностью до округления, и сравнение знаков
    даёт ложное "пересечение". Проверено: пара непересекающихся
    коллинеарных рёбер давала ориентации ~1e-18 разных знаков.
    """
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = ccw(p3, p4, p1), ccw(p3, p4, p2)
    d3, d4 = ccw(p1, p2, p3), ccw(p1, p2, p4)
    if abs(d1) < eps or abs(d2) < eps or abs(d3) < eps or abs(d4) < eps:
        return False                      # коллинеарно/касание — не пересечение
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _signed_area(pts) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def verify(mesh, name="mesh", log=True) -> dict:
    """ВАЖНО: вывод идёт только в stderr. MCP по stdio использует stdout
    для JSON-RPC — любая печать туда ломает протокол и сервер падает с
    "Connection closed"."""
    V = mesh.vertices[:, :2]
    F = mesh.faces

    non_quads = [f for f in F if len(f) != 4]
    degenerate = [f for f in F if abs(_signed_area(V[list(f)])) < 1e-9]

    signs = [_signed_area(V[list(f)]) > 0 for f in F]
    winding_ok = all(signs) or not any(signs)

    edge_count: dict[tuple[int, int], int] = {}
    for f in F:
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            k = (a, b) if a < b else (b, a)
            edge_count[k] = edge_count.get(k, 0) + 1
    hanging = [k for k, c in edge_count.items() if c > 2]

    edges = list(edge_count.keys())
    crossings = 0
    for (a, b), (c, d) in combinations(edges, 2):
        if len({a, b, c, d}) < 4:
            continue
        if _seg_cross(V[a], V[b], V[c], V[d]):
            crossings += 1

    # "квадратность" граней: 1.0 = идеальный квадрат
    quality = []
    for f in F:
        if len(f) != 4:
            continue
        pts = V[list(f)]
        e = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
        if min(e) < 1e-9:
            continue
        aspect = min(e) / max(e)
        dev = 0.0
        for i in range(4):
            v1 = pts[i - 1] - pts[i]
            v2 = pts[(i + 1) % 4] - pts[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
            dev += abs(ang - 90.0)
        quality.append(0.5 * aspect + 0.5 * max(0.0, 1.0 - (dev / 4) / 90.0))

    quads_count = len(F) - len(non_quads)
    quad_pct = (quads_count / len(F) * 100.0) if len(F) > 0 else 0.0

    boundary_mask = np.zeros(len(V), dtype=bool)
    for (a, b), c in edge_count.items():
        if c == 1:
            boundary_mask[a] = True
            boundary_mask[b] = True
    interior_verts_count = int(np.sum(~boundary_mask))

    report = {
        "faces": len(F),
        "verts": len(V),
        "quads": quads_count,
        "non_quads": len(non_quads),
        "quad_pct": round(quad_pct, 1),
        "interior_verts": interior_verts_count,
        "degenerate": len(degenerate),
        "winding_consistent": winding_ok,
        "hanging_edges": len(hanging),
        "edge_crossings": crossings,
        "quad_quality_mean": round(float(np.mean(quality)), 3) if quality else None,
        "quad_quality_min": round(float(np.min(quality)), 3) if quality else None,
    }

    ok = (
        report["degenerate"] == 0
        and report["winding_consistent"]
        and report["hanging_edges"] == 0
        and report["edge_crossings"] == 0
        and report["quad_pct"] >= 85.0
    )
    report["PASS"] = ok
    if log:
        print(f"{name}: {'PASS' if ok else 'FAIL'} {report}", file=sys.stderr)
    return report


def silhouette_fidelity(mesh, image_path, scale, threshold=127.0, invert=False,
                        morph_size=3) -> float:
    """IoU построенной сетки с исходной маской.

    Ловит класс ошибок, который структурные проверки пропускают: если
    алгоритм построил лишь ФРАГМЕНТ фигуры, фрагмент сам по себе
    безупречен (все квады, ноль пересечений) и verify() честно говорит
    PASS. Только сравнение с исходной маской показывает, что построена
    не та фигура. Реальный случай: маркер отверстия попадал в тело
    фигуры, triangle вырезал материал, IoU падал до 0.007 при PASS.
    """
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1.0
    _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    if invert:
        mask = 255 - mask
    if morph_size and morph_size > 0:
        k = np.ones((morph_size, morph_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    src = mask > 0

    ys, xs = np.nonzero(src)
    if len(xs) == 0:
        return -1.0

    V = mesh.vertices[:, :2] / scale
    V = V.copy()
    V[:, 1] = -V[:, 1]                       # обратно в систему изображения
    bb_min, bb_max = V.min(axis=0), V.max(axis=0)
    src_min = np.array([xs.min(), ys.min()], float)
    src_max = np.array([xs.max(), ys.max()], float)

    # Восстановление координат изображения
    # Do not stretch or distort coordinates via bounding-box min/max mapping,
    # as non-uniform component filtering or centering shifts canvas coordinates.
    if bb_min[0] < -1.0 or bb_min[1] < -1.0:
        src_center = (src_min + src_max) / 2.0
        V = V + src_center
    else:
        pass

    H, W = src.shape
    render = np.zeros((H, W), np.uint8)
    polys = [np.round(V[list(f)]).astype(np.int32) for f in mesh.faces]
    cv2.fillPoly(render, polys, 255)
    got = render > 0

    inter = np.logical_and(src, got).sum()
    union = np.logical_or(src, got).sum()
    return float(inter) / float(union) if union else 0.0
