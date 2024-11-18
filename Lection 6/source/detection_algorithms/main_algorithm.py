import cv2
import numpy as np


def create_detector_algorithm(name: str = "ORB") -> cv2.Feature2D:
    """
    # ORB сочетает два алгоритма: FAST (Features from Accelerated Segment Test) для обнаружения ключевых точек и BRIEF (Binary Robust Independent Elementary Features) для вычисления дескрипторов. Оба алгоритма являются открытыми и не защищены патентами.
    #         Алгоритм ORB был разработан с учётом потребности в быстром и эффективном решении для обнаружения и описания ключевых точек, при этом не нарушая патенты.
    # SIFT (Scale-Invariant Feature Transform) — это алгоритм для обнаружения ключевых точек (особенностей) изображения, инвариантных к масштабу и вращению.
    #  Патент на алгоритм принадлежит Южноафриканскому университету
    """
    if name == "ORB":
        return cv2.ORB_create()
    elif name == "SIFT":
        return cv2.SIFT_create()
    else:
        raise ValueError(f"Неизвестное имя алгоритма: {name}")


# def find_keypoints_and_descriptors(key_points_algorithm, template: np.ndarray, scene_without_fond_ghosts: np.ndarray) -> tuple, tuple:
#    return (keypoints_scene, keypoints_template), (descriptors_scene, descriptors_template)

# 3. Сопоставление ключевых точек
def BruteForce_Matching_k2(descriptors_template, descriptors_scene):
    """
    # BFMatcher — "Brute-Force Matcher", алгоритм, который сопоставляет дескрипторы двух изображений, находя ближайшие пары точек.
    # knnMatch():
    #  + Находит для каждого дескриптора template двух ближайших соседей в дескрипторах scene.
    #  +  Параметр k=2 указывает, что для каждой точки нужно найти две лучших совпадающих точки.
    #
    # Результат:
    #  + matches — список пар (по 2) ближайших соседей для каждой ключевой точки шаблона.
    """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_template, descriptors_scene, k=2) # k-ближайших соседей (k-NN)
    
    return matches

# 4. Фильтрация хороших совпадений
def filter_matches(matches) -> list:
    """
    # Идея:
    #  + Для каждой пары ближайших соседей m и n проверяется, насколько первая точка (ближайший сосед) лучше второй.
    #  + Метрика качества определяется расстоянием (distance) между дескрипторами.
    #  + Если первая точка на 25% (или больше) ближе, чем вторая, то совпадение считается "хорошим".
    #
    # Результат: в good_matches сохраняются только те совпадения, которые проходят фильтр.
    """
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches


# 8. Итоговая раскраска
def process_scene_with_homography(
    scene: np.ndarray, 
    scene_without_fond_ghosts: np.ndarray, 
    M: np.ndarray,
    template_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Рисует рамку на изображении и закрашивает область, соответствующую найденному шаблону.
    
    Параметры:
        scene (np.ndarray): Изображение, на котором будет нарисована рамка.
        scene_without_fond_ghosts (np.ndarray): Изображение для закрашивания области шаблона.
        M (np.ndarray): Матрица гомографии для преобразования координат.

    Возвращает:
        tuple[np.ndarray, np.ndarray]: 
            - Обновлённое изображение `scene` с рамкой.
            - Обновлённое изображение `scene_without_fond_ghosts` с закрашенной областью.
    """
    if M is not None:
        h, w = template_shape
        # Определяем координаты углов исходного шаблона (прямоугольника)
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        # Преобразуем их в соответствии с матрицей гомографии
        dst = cv2.perspectiveTransform(pts, M)

        # Рисуем рамку на изображении `scene`
        green=(0, 255, 0)
        scene = cv2.polylines(scene, [np.int32(dst)], isClosed=True, color=green, thickness=3, lineType=cv2.LINE_AA)

        # Преобразуем точки в целые числа
        dst_int = np.int32(dst).reshape(-1, 2)
        x_min, y_min = dst_int.min(axis=0)
        x_max, y_max = dst_int.max(axis=0)

        # Рисуем закрашенный прямоугольник на `scene_without_fond_ghosts`
        scene_without_fond_ghosts = cv2.rectangle(scene_without_fond_ghosts, (x_min, y_min), (x_max, y_max), (255, 0, 0), -1)

    return scene, scene_without_fond_ghosts

    

def detect_and_draw(scene: np.ndarray, scene_without_fond_ghosts: np.ndarray, template: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Выполняет обнаружение объекта на сцене и отрисовывает рамки вокруг найденных объектов.

    Returns:
        np.ndarray: Сцена с нанесёнными рамками вокруг найденных объектов.
    """
    
    # 1. Создание детектора точек
    key_points_algorithm = create_detector_algorithm("SIFT")

    # 2. Обнаружение ключевых точек и дескрипторов
    keypoints_template, descriptors_template = key_points_algorithm.detectAndCompute(template, None)
    keypoints_scene, descriptors_scene = key_points_algorithm.detectAndCompute(scene_without_fond_ghosts, None)
    # detectAndCompute():
    #  + Находит ключевые точки — это особенные области изображения (например, углы, края), которые устойчивы к изменению масштаба, поворота и освещения.
    #  + Вычисляет дескрипторы — численные векторы, описывающие найденные ключевые точки.
    #
    # Результат:
    #  + keypoints_scene и keypoints_template — списки ключевых точек для каждого изображения.
    #  + descriptors_scene и descriptors_template — дескрипторы для каждой ключевой точки.

    # 3. Сопоставление ключевых точек
    matches = BruteForce_Matching_k2(descriptors_template, descriptors_scene)

    # 4. Фильтрация хороших совпадений
    good_matches = filter_matches(matches)

    # Проверка достаточности совпадений
    if len(good_matches) < 10:
        return scene, scene_without_fond_ghosts

    # 6. Определение гомографии
    src_pts = np.float32([keypoints_template[m.queryIdx].pt  for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([   keypoints_scene[m.trainIdx].pt  for m in good_matches]).reshape(-1, 1, 2)
    # Гомография — это преобразование, которое связывает координаты точек на одном изображении с координатами на другом. Она используется для нахождения положения объекта.
    #  + Вычисляет дескрипторы — численные векторы, описывающие найденные ключевые точки.
    #
    # Результат:
    #  + keypoints_scene и keypoints_template — списки ключевых точек для каждого изображения.
    #  + descriptors_scene и descriptors_template — дескрипторы для каждой ключевой точки.

    # 7. Поиск матрицы гомографии
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # Находит матрицу преобразования (гомографии) M, которая преобразует точки src_pts в dst_pts
    # Используется метод RANSAC (Random Sample Consensus), чтобы исключить выбросы (ошибочные соответствия).
    # Параметр 5.0 задаёт допустимое расстояние для соответствия.
    # 
    # Выход:
    # M — матрица гомографии.
    # mask — двоичный массив, указывающий, какие совпадения из good_matches являются согласованными.
    
    # 8. Итоговая раскраска
    scene, scene_without_fond_ghosts = process_scene_with_homography(scene, scene_without_fond_ghosts, M, template.shape)
    
    # посмотреть закрашенные области
    show_subprocess_image(scene_without_fond_ghosts)

    return scene, scene_without_fond_ghosts

def show_subprocess_image(scene_without_fond_ghosts) -> None:
    import matplotlib.pyplot as plt
    plt.imshow(scene_without_fond_ghosts)
    plt.show()

#def is_symmetric(image: np.ndarray) -> bool:
#    """
#    Проверка на симметричность изображения относительно вертикальной оси.
#    """
#    flipped = cv2.flip(image, 1)  # Отражаем изображение по вертикали
#    diff = cv2.absdiff(image, flipped)  # Находим разницу между оригиналом и отражением
#    return np.sum(diff) < 0.01 * image.size  # Если разница малая, значит изображение симметрично
