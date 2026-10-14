"""
server.py — Layer B (MCP-обёртка) + Layer C (доставка картинки)

Отдельный MCP-сервер, работает РЯДОМ с официальным Blender MCP, ничего
в нём не трогает. Даёт один тул: image_to_lowpoly_shape.

Layer C реализован МАКСИМАЛЬНО просто: единая папка-обменник на диске
(по умолчанию C:\\Temp\\ai_bridge, как и ваш OBJ-обмен с Rhino).
    - image_path="AUTO"  -> берётся самый свежий файл по mtime из папки
    - image_path="<путь>" -> используется как есть (никакого поиска)

Никакого LM Studio Plugin SDK, никакой TypeScript-сборки, никакой гонки
между процессами. Картинку в папку кладёте вы (drag&drop) или, отдельно,
можно доложить автоматизацию (plugin) поверх этого — но сервер не
зависит от неё.

Сервер НЕ трогает bpy напрямую. Он отдаёт:
    - готовый .obj на диске
    - статистику по сетке (компоненты/дырки/вершины/квады/тримы)
    - готовый фрагмент bpy-кода для импорта, который агент передаёт
      в уже существующий официальный Blender MCP (execute_blender_code)

Так слой "картинка -> геометрия" остаётся полностью независимым и
тестируемым без Blender и без LM Studio (см. lowpoly_algo.py, там же
CLI для автономного прогона).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from lowpoly_algo import Params, process_image, process_image_quad
from quad_mesh import QuadParams, process_image_allquad
from ribbon_topology import RibbonParams, process_image_ribbon
from verify_mesh import silhouette_fidelity, verify as verify_mesh

# --------------------------------------------------------------------------
# Конфигурация Layer C (папка-обменник)
# --------------------------------------------------------------------------

BRIDGE_DIR = Path(os.environ.get("LOWPOLY_BRIDGE_DIR", r"C:\Temp\ai_bridge"))
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
OUTPUT_DIR = Path(os.environ.get("LOWPOLY_OUTPUT_DIR", str(BRIDGE_DIR / "out")))

mcp = FastMCP("image_to_lowpoly_shape")


def _resolve_auto() -> Path:
    if not BRIDGE_DIR.exists():
        raise FileNotFoundError(
            f"Папка-обменник не найдена: {BRIDGE_DIR}. "
            f"Создайте её и положите туда референс-изображение, "
            f"либо задайте переменную окружения LOWPOLY_BRIDGE_DIR."
        )
    candidates = [
        p for p in BRIDGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if not candidates:
        raise FileNotFoundError(
            f"В папке {BRIDGE_DIR} нет изображений ({', '.join(sorted(IMAGE_EXTS))}). "
            f"Положите туда референс и повторите вызов с image_path='AUTO'."
        )
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest


def _make_bpy_snippet(obj_path: str, object_name: str, add_subdiv: bool, subdiv_level: int) -> str:
    obj_path_escaped = obj_path.replace("\\", "\\\\")
    lines = [
        "import bpy",
        f'bpy.ops.wm.obj_import(filepath=r"{obj_path_escaped}")',
        "imported = bpy.context.selected_objects[0]",
        f'imported.name = "{object_name}"',
    ]
    if add_subdiv:
        lines += [
            'mod = imported.modifiers.new(name="Subdivision", type="SUBSURF")',
            f"mod.levels = {subdiv_level}",
            f"mod.render_levels = {subdiv_level}",
        ]
    lines.append("bpy.context.view_layer.objects.active = imported")
    return "\n".join(lines)


@mcp.tool()
def image_to_lowpoly_shape(
    image_path: str = "AUTO",
    topology: str = "allquad",
    threshold: float = 127.0,
    invert: bool = False,
    blur: int = 0,
    morph_size: int = 3,
    simplify_px: float = 3.0,
    min_area_px: float = 80.0,
    target_edge_px: float = 30.0,
    corner_epsilon_px: float = 4.0,
    width_factor: float = 2.0,
    min_edge_px: float = 6.0,
    smooth_iterations: int = 12,
    steiner_points: bool = False,
    prune_px: float = 8.0,
    merge_junction_px: float = 10.0,
    merge_junction_width_factor: float = 3.0,
    min_width_px: float = 1.5,
    scale: float = 0.01,
    center: bool = True,
    max_parts: int = 32,
    subdivision: bool = True,
    subdiv_level: int = 2,
    object_name: str = "LowPolyShape",
) -> dict:
    """
    Строит плоскую (Z=0) низкополигональную сетку по силуэту на изображении
    (завиток, орнамент, буква, ювелирная деталь и т.п.) и сохраняет .obj.

    topology:
        "allquad" (по умолчанию, РЕКОМЕНДУЕТСЯ) — ГАРАНТИРОВАННО 100%
            квадов: ни одного треугольника, ни одного n-угольника, ни
            одного самопересечения, ни одного висячего узла. Достигается
            не эвристикой, а тождеством: после quad-dominant слияния
            применяется топологическое измельчение Катмулла-Кларка
            (n-угольник -> ровно n квадов), после чего внутренние
            вершины сглаживаются лапласовым фильтром (граница закреплена,
            силуэт не смещается). Шаг точек по контуру следует локальной
            ширине фигуры (width_factor), поэтому квады выходят близкими
            к квадратам, а не длинными тонкими прямоугольниками.
            Результат самопроверяется — см. поле "verification" в ответе.
        "quad" — v5: CDT + слияние по качеству, БЕЗ гарантии 100% квадов
            Delaunay триангуляция РЕАЛЬНОГО контура (самопересечение
            математически невозможно, пока контур простой — а он всегда
            простой), затем контуры передискретизируются с сохранением
            острых углов (target_edge_px/corner_epsilon_px) и треугольники
            сливаются в квады по приоритету "квадратности", а не как
            попало. Пришёл на смену topology="ribbon": тот
            skeleton-подход синтезировал геометрию заново и оказался
            систематически подвержен самопересечениям на реальных
            картинках (несколько независимых источников багов).
        "ribbon" — старый skeleton/medial-axis режим. Оставлен только для
            сравнения — НЕ рекомендуется, известны случаи самопересечений
            на некоторых реальных изображениях (см. README).
        "fill" — старый режим earcut + жадное (не по качеству) слияние в
            квады. Тоже вытеснен "quad", оставлен для сравнения.

    image_path: путь к файлу ИЛИ "AUTO" — тогда берётся самый свежий файл
        из папки-обменника (по умолчанию C:\\Temp\\ai_bridge, см.
        LOWPOLY_BRIDGE_DIR). Положите туда референс перед вызовом.

    target_edge_px / corner_epsilon_px — только для topology="quad":
        желаемый шаг точек на изогнутых участках контура (управляет
        плотностью/числом квадов — больше значение = меньше квадов) и
        порог определения "углов" (они всегда остаются отдельными
        вершинами, Subdivision их не тронет).

    prune_px / min_width_px / merge_junction_px / merge_junction_width_factor —
        только для topology="ribbon" (не рекомендуется, см. выше).

    Возвращает путь к .obj, статистику по сетке и готовый bpy-код для
    импорта через официальный Blender MCP (execute_blender_code).
    Сам этот сервер bpy не вызывает.
    """
    if image_path == "AUTO":
        src = _resolve_auto()
    else:
        src = Path(image_path)
        if not src.exists():
            raise FileNotFoundError(f"Файл не найден: {src}")

    if topology == "allquad":
        params = QuadParams(
            threshold=threshold,
            invert=invert,
            blur=blur,
            morph_size=morph_size,
            min_area_px=min_area_px,
            target_edge_px=target_edge_px,
            corner_epsilon_px=corner_epsilon_px,
            width_factor=width_factor,
            min_edge_px=min_edge_px,
            smooth_iterations=smooth_iterations,
            steiner_points=steiner_points,
            scale=scale,
            center=center,
            max_parts=max_parts,
        )
        mesh = process_image_allquad(src, params)
    elif topology == "quad":
        params = Params(
            threshold=threshold,
            invert=invert,
            blur=blur,
            morph_size=morph_size,
            min_area_px=min_area_px,
            target_edge_px=target_edge_px,
            corner_epsilon_px=corner_epsilon_px,
            scale=scale,
            center=center,
            max_parts=max_parts,
        )
        mesh = process_image_quad(src, params)
    elif topology == "ribbon":
        params = RibbonParams(
            threshold=threshold,
            invert=invert,
            blur=blur,
            morph_size=morph_size,
            simplify_px=simplify_px,
            min_area_px=min_area_px,
            prune_px=prune_px,
            merge_junction_px=merge_junction_px,
            merge_junction_width_factor=merge_junction_width_factor,
            min_width_px=min_width_px,
            scale=scale,
            center=center,
            max_parts=max_parts,
        )
        mesh = process_image_ribbon(src, params)
    elif topology == "fill":
        params = Params(
            threshold=threshold,
            invert=invert,
            blur=blur,
            morph_size=morph_size,
            simplify_px=simplify_px,
            min_area_px=min_area_px,
            scale=scale,
            center=center,
            max_parts=max_parts,
        )
        mesh = process_image(src, params)
    else:
        raise ValueError(
            f"Неизвестный topology='{topology}', "
            f"допустимо: 'allquad' | 'quad' | 'ribbon' | 'fill'"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    obj_path = OUTPUT_DIR / f"{object_name}_{stamp}.obj"
    mesh.to_obj(obj_path)

    stats = mesh.stats()
    bpy_code = _make_bpy_snippet(str(obj_path), object_name, subdivision, subdiv_level)

    # Самопроверка: тул сам себя проверяет и честно сообщает вердикт,
    # вместо того чтобы отдавать сетку молча и оставлять оценку на глаз.
    try:
        report = verify_mesh(mesh, object_name, log=False)
        # Структурные проверки не ловят случай "построен лишь фрагмент
        # фигуры": фрагмент сам по себе безупречен. Сверка с исходной
        # маской — единственное, что это ловит.
        iou = silhouette_fidelity(mesh, src, scale, threshold, invert, morph_size)
        report["silhouette_iou"] = round(iou, 4)
        if 0.0 <= iou < 0.85:
            report["PASS"] = False
            report["warning"] = (
                f"Силуэт совпадает с исходной маской лишь на {iou:.1%} — "
                f"вероятно построена только часть фигуры либо неверно "
                f"подобраны threshold/invert. Сверьтесь со скриншотом."
            )
    except Exception as exc:  # проверка не должна ронять основную работу
        report = {"error": str(exc)}

    return {
        "verification": report,
        "source_image": str(src),
        "obj_path": str(obj_path),
        "stats": stats,
        "next_step": (
            "Передайте поле 'blender_python' в execute_blender_code "
            "официального Blender MCP, чтобы импортировать результат. "
            "После импорта сделайте Top Orthographic скриншот и сверьте "
            "с референсом, прежде чем считать результат окончательным."
        ),
        "blender_python": bpy_code,
    }


@mcp.tool()
def list_bridge_folder() -> dict:
    """
    Показывает, что сейчас лежит в папке-обменнике (Layer C) и что будет
    выбрано при image_path='AUTO'. Полезно для отладки перед основным
    вызовом image_to_lowpoly_shape.
    """
    if not BRIDGE_DIR.exists():
        return {"bridge_dir": str(BRIDGE_DIR), "exists": False, "files": []}
    files = [
        {"name": p.name, "mtime": p.stat().st_mtime}
        for p in BRIDGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    files.sort(key=lambda f: -f["mtime"])
    auto_pick = files[0]["name"] if files else None
    return {
        "bridge_dir": str(BRIDGE_DIR),
        "exists": True,
        "files": files,
        "would_pick_for_AUTO": auto_pick,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
