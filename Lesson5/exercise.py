import os
import cv2

# nails_processing

# Путь к папке с данными
DATA_PATH = "nails_segmentation"
IMAGES_PATH = os.path.join(DATA_PATH, "images")
LABELS_PATH = os.path.join(DATA_PATH, "labels")

# Считываем список файлов из папок
image_files = sorted([f for f in os.listdir(IMAGES_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))])
label_files = sorted([f for f in os.listdir(LABELS_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))])

# Проверяем, что парные изображения и маски имеют одинаковые названия
paired_files = [(os.path.join(IMAGES_PATH, img), os.path.join(LABELS_PATH, lbl))
                for img, lbl in zip(image_files, label_files) if img == lbl]

if not paired_files:
    print("Не найдены парные файлы изображений и масок!")
    exit()

# Выводим пары изображений и масок, выделяем контуры и отрисовываем их
for img_path, mask_path in paired_files:
    # Считываем изображения и маски
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Проверка на успешную загрузку
    if image is None or mask is None:
        print(f"Ошибка чтения файлов: {img_path}, {mask_path}")
        continue

    # Находим контуры на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отрисовываем контуры на изображении
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    # Отображаем результат
    combined = cv2.hconcat([image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
    cv2.imshow("Image and Mask with Contours", combined)

    # Ждём нажатия клавиши для перехода к следующей паре
    if cv2.waitKey(0) == 27:  # Нажатие ESC завершает просмотр
        break

cv2.destroyAllWindows()

# Воспроизведение видео в градациях серого
VIDEO_PATH = "/home/vovan/Видео/Episode 3.mkv"  # Укажите путь к вашему видеофайлу

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Ошибка открытия видеофайла: {VIDEO_PATH}")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Конвертируем кадр в градации серого
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Отображаем кадр
    cv2.imshow("Video in Grayscale", gray_frame)

    # Выход по нажатию клавиши ESC
    if cv2.waitKey(25) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

