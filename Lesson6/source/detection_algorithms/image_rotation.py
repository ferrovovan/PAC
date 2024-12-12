import cv2
import numpy as np


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """
    Функция для поворота изображения на заданный угол.
    Возвращает повернутое изображение.
    """
    # Получаем центр изображения
    center = (image.shape[1] // 2, image.shape[0] // 2)
    
    # Получаем матрицу поворота
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Поворачиваем изображение
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    
    return rotated_image

