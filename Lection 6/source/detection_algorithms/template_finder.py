import cv2
import numpy as np
from .image_rotation import rotate_image
from .main_algorithm import detect_and_draw 


def find_template(scene: np.ndarray, template_path: str) -> np.ndarray:
    """
    Поиск объектов на сцене по заданному шаблону, включая повороты и отражения по горизонтали.
    Возвращает обновлённые scene и scene_without_fond_ghosts.
    """
    # Загрузка шаблона
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    scene_without_fond_ghosts: np.ndarray = scene.copy()

    if template is None:
        raise ValueError(f"Не удалось загрузить изображение шаблона: {template_path}")

    # Перебираем повороты шаблона
    for angle in range(0, 360, 45):  # Шаг 45°; можно уменьшить до 15° для более точного поиска
        rotated_template = rotate_image(template, angle)
        flipped_rotated_template = cv2.flip(rotated_template, 1)

        # Обновляем сцену и сцену без найденных объектов
        scene, scene_without_fond_ghosts = detect_and_draw(scene, scene_without_fond_ghosts, rotated_template)
        scene, scene_without_fond_ghosts = detect_and_draw(scene, scene_without_fond_ghosts, flipped_rotated_template)

    return scene

