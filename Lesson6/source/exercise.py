#!/usr/bin/python3

import os
import cv2
import numpy as np
import random

def nails_data_generator(data_path, batch_size, image_size=(128, 128)):
    """
    Генератор данных для датасета Nails Segmentation.
    
    Args:
        data_path (str): Путь к папке с подкаталогами "images" и "labels".
        batch_size (int): Количество изображений и масок в одной итерации.
        image_size (tuple): Размер изображений после обработки (ширина, высота).
    
    Yields:
        tuple: Пара списков (изображения, маски), оба списки размером batch_size.
    """
    images_path = os.path.join(data_path, "images")
    labels_path = os.path.join(data_path, "labels")

    # Список пар изображений и масок
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    label_files = sorted([f for f in os.listdir(labels_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    paired_files = [(os.path.join(images_path, img), os.path.join(labels_path, lbl))
                    for img, lbl in zip(image_files, label_files) if img == lbl]

    if not paired_files:
        raise ValueError("Не найдены парные файлы изображений и масок!")
    
    while True:
        # Перемешиваем пары
        random.shuffle(paired_files)

        for i in range(0, len(paired_files), batch_size):
            batch_pairs = paired_files[i:i + batch_size]
            images, masks = [], []

            for img_path, mask_path in batch_pairs:
                # Считываем изображение и маску
                image = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if image is None or mask is None:
                    continue

                # Аугментация
                image, mask = apply_augmentation(image, mask)

                # Изменяем размер
                image = cv2.resize(image, image_size)
                mask = cv2.resize(mask, image_size)

                # Добавляем в текущую выборку
                images.append(image)
                masks.append(mask)

            yield np.array(images), np.array(masks)


def apply_augmentation(image, mask):
    """
    Применяет случайные аугментации к изображению и маске.

    Args:
        image (numpy.ndarray): Исходное изображение.
        mask (numpy.ndarray): Исходная маска.
    
    Returns:
        tuple: Аугментированное изображение и маска.
    """
    # Поворот на случайный угол
    if random.random() > 0.5:
        angle = random.uniform(-30, 30)
        image, mask = rotate(image, mask, angle)

    # Случайное отражение
    if random.random() > 0.5:
        image = cv2.flip(image, 1)  # Отражение по горизонтали
        mask = cv2.flip(mask, 1)
    if random.random() > 0.5:
        image = cv2.flip(image, 0)  # Отражение по вертикали
        mask = cv2.flip(mask, 0)

    # Вырезание случайной части
    if random.random() > 0.5:
        image, mask = crop_random(image, mask)

    # Размытие
    if random.random() > 0.5:
        kernel_size = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return image, mask


def rotate(image, mask, angle):
    """Поворот изображения и маски на заданный угол."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    image_rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR)
    mask_rotated = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST)
    return image_rotated, mask_rotated


def crop_random(image, mask, crop_size=(50, 50)):
    """Вырезание случайной части изображения и маски."""
    h, w = image.shape[:2]
    ch, cw = crop_size
    if h <= ch or w <= cw:
        return image, mask

    x = random.randint(0, w - cw)
    y = random.randint(0, h - ch)

    image_cropped = image[y:y + ch, x:x + cw]
    mask_cropped  =  mask[y:y + ch, x:x + cw]
    return cv2.resize(image_cropped, (w, h)), cv2.resize(mask_cropped, (w, h))


# Тестирование генератора
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    generator = nails_data_generator("nails_segmentation", batch_size=4)

    # Получаем первую выборку
    images, masks = next(generator)

    # Отображаем результаты
    for i in range(len(images)):
        plt.subplot(2, len(images), i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.subplot(2, len(images), i + 1 + len(images))
        plt.imshow(masks[i], cmap='gray')
        plt.axis('off')

    plt.show()

