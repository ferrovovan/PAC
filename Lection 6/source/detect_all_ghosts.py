#!/usr/bin/python3

import cv2
from detection_algorithms import find_template


def detect_all_ghosts(scene_path, ghosts_paths, output_path):
    """
    ЛР 6.2: Поиск всех призраков на изображении, учитывая масштабирование и отражение.

    Args:
        scene_path (str): Путь к изображению сцены.
        ghosts_paths (list): Список путей к изображениям призраков.
        output_path (str): Путь для сохранения результата.
    """
    # Загрузка изображения сцены
    scene = cv2.imread(scene_path)
    if scene is None:
        raise ValueError(f"Не удалось загрузить изображение сцены: {scene_path}")

    # Для каждого шаблона найти призраков
    for ghost_path in ghosts_paths:
        scene = find_template(scene, ghost_path)

    # Сохранение результата
    cv2.imwrite(output_path, scene)
    print(f"Результат сохранён в: {output_path}")


if __name__ == "__main__":
    # Пути к файлам
    scene_file = "images/lab7.png"
    ghost_files = ["images/candy_ghost.png", "images/pampkin_ghost.png", "images/scary_ghost.png"]

    # ЛР 6.2: Поиск всех призраков
    detect_all_ghosts(scene_file, ghost_files, "results/all_ghosts.png")

