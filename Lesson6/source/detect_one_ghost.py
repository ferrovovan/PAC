#!/usr/bin/python3

import cv2
from detection_algorithms import find_template


def detect_one_ghost(scene_path, ghost_path, output_path):
    """
    ЛР 6.1: Поиск одного призрака на изображении.

    Args:
        scene_path (str): Путь к изображению сцены.
        ghost_path (str): Путь к изображению призрака.
        output_path (str): Путь для сохранения результата.
    """
    # Загрузка изображения сцены
    scene = cv2.imread(scene_path)
    if scene is None:
        raise ValueError(f"Не удалось загрузить изображение сцены: {scene_path}")

    # Проверка наличия призрака на сцене, включая отражённую версию
    scene_with_fond_ghost = find_template(scene, ghost_path)

    # Сохранение результата
    cv2.imwrite(output_path, scene_with_fond_ghost)
    print(f"Результат сохранён в: {output_path}")


if __name__ == "__main__":
    # Пути к файлам
    scene_file = "images/lab7.png"
    ghost_files = ["images/candy_ghost.png", "images/pampkin_ghost.png", "images/scary_ghost.png"]
    
    # ЛР 6.1: Поиск одного призрака
    detect_one_ghost(scene_file, ghost_files[1], "results/one_ghost.png")

