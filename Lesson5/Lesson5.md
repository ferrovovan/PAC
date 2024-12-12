# Matplotlib
## Работа с графиками (в двух словах)
```
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

np.random.seed(42)
xs = np.arange(-5, 5)
ys = np.random.randint(0, 5, size=10)
print(ys)
```
[3 4 2 4 4 1 2 2 2 4]
```
plt.plot(xs, ys)
plt.show()
```
![]()

## Форматирование графика
```
plt.plot(xs, ys, 'g')
plt.show()
plt.plot(xs, ys, 'y-*')
plt.show()
plt.plot(xs, ys, linewidth=5)
plt.show()
```
![]()
```
plt.scatter(xs, ys)
plt.show()
plt.plot(xs, ys)
plt.scatter(xs, ys, c='red', marker='v')
plt.grid()
plt.show()
```
![]()
## Несколько графиков
```
plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)  # Выбор подграфика (кол-во строк, кол-во столбцов, № графика)
plt.plot(xs, ys)
plt.subplot(1, 2, 2)
plt.scatter(xs, ys, c='red', marker='v')
plt.grid()
plt.show()
```
# Matplotlib + numpy
```
image = np.zeros((50, 50))
plt.imshow(image)
plt.show()
```
![]()
```
image[:10, 10:20] = 1
plt.imshow(image)
plt.show()
```
![]()
```
plt.imshow(image)
plt.colorbar()  # Показать соответствие цвета и значения
plt.show()
```
## Смена цветовой карты на чёрно-белую
```
plt.imshow(image, cmap='gray')
plt.colorbar()
plt.show()
```
![]()
## Картинка из функции
```
image = np.fromfunction(lambda y, x: x, shape=(50, 50))
plt.imshow(image, cmap='gray')
plt.colorbar()
plt.show()
image = np.fromfunction(lambda y, x: x + y / 2, shape=(50, 50))
plt.imshow(image, cmap='gray')
plt.colorbar()
plt.show()
```
![]()
## Трёхканальные изображения (RGB)
Необходимо, чтобы числа были от 0 до 1, либо от 0 до 255 и быть типа uint8
```
image_rgb = np.stack([image, image, image], axis=-1)
image_rgb = (image_rgb / image_rgb.max()) * 255
image_rgb = image_rgb.astype(np.uint8)
plt.imshow(image_rgb)
plt.colorbar()
plt.show()
```
![]()
```
# 1 канал - красный, 2 - зелёный, 3 - синий
for i in range(3):
    image_out = np.zeros_like(image_rgb)  # Копирование размера и типа данных
    image_out[:, :, i] = image_rgb[:, :, i]
    
    plt.subplot(1, 3, i+1)
    plt.title(f'{image_out[-1, -1]}')
    plt.imshow(image_out)

plt.show()
```
![]()
```
for i in range(3):
    image_out = image_rgb.copy()
    image_out[:, :, i] = 0  # Зануляем один из каналов
    
    plt.subplot(1, 3, i+1)
    plt.title(f'{image_out[-1, -1]}')
    plt.imshow(image_out)

plt.show()
```
![]()

# Библиотека OpenCV
![logo](./images/)
**OpenCV** (Open Source Computer Vision Library) — библиотека алгоритмов компьютерного зрения, обработки изображений и численных алгоритмов общего назначения с открытым кодом. Реализована на C/C++, также разрабатывается для Python, Java, Ruby, Matlab, Lua и других языков.
## Пример отображения
```
import matplotlib.pyplot as plt
# Чтение картинки (чтение происходит в цветовой модели BGR)

img_bgr = cv2.imread('data/Vanya.jpg')
print(img_bgr.shape)
plt.imshow(img_bgr)
plt.show()
```
(256, 256, 3)
![]()
### Пример отображения
```
# Для перевода в RGB можно либо воспользоваться функцией opencv, либо инвертировать каналы
image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
image = img_bgr[:, :, ::-1]  # Каналы - третье измерение изображения
plt.imshow(image)
plt.show()
```
(256, 256, 3)
![]()
### image - np.range
```
# Массив остаётся массивом
crop = image[50:150, 100: 150]
plt.imshow(crop)
plt.show()
```
![]()
### Чтение в цветовой модели Grayscale
```
img_gray = cv2.imread('data/Vanya.jpg', 0)
print(img_gray.shape)
plt.imshow(img_gray, cmap='gray')
plt.show()
```
![]()
## Цветовые пространства
![]()
**YUV** — цветовая модель, в которой цвет состоит из трёх компонентов — яркость (Y) и два цветоразностных компонента (U и V)
Компоненты YUV на основе компонент RGB:
![]()
### Перевод в YUV цветовую модель
```
img_gray = cv2.imread('data/Vanya.jpg', 0)
print(img_gray.shape)
plt.imshow(img_gray, cmap='gray')
plt.show()
```
![]()
## HSV
![]()
HSV (англ. Hue, Saturation, Value — тон, насыщенность, значение) или HSB (англ. Hue, Saturation, Brightness — тон, насыщенность, яркость) — цветовая модель, в которой координатами цвета являются:
- Hue — цветовой тон, (например, красный, зелёный или сине-голубой). Варьируется в пределах 0—360°, однако иногда приводится к диапазону 0—100 или 0—1.
- Saturation — насыщенность. Варьируется в пределах 0—100 или 0—1. Чем больше этот параметр, тем «чище» цвет, поэтому этот параметр иногда называют чистотой цвета. А чем ближе этот параметр к нулю, тем ближе цвет к нейтральному серому.
- Value (значение цвета) или Brightness — яркость. Также задаётся в пределах 0—100 или 0—1.
### Использование цветовой модели HSV
```
image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
print(image.shape)
plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(image[:, :, i], cmap='gray')
plt.show()
```
![]()

## Гистограмма
Гистограмма распределения частот - это столбиковая диаграмма, каждый столбец которой опирается на конкретное значение признака или разрядный интервал (для сгруппированных частот). Высота столбика пропорциональна частоте встречаемости соответствующего значения.
### Перевод изображения в гистограмму
```
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.subplot(1, 2, 2)
plt.hist(img_gray.ravel(), range=(0, 255), bins=256)
plt.show()
```
![]()
### Выравнивание гистограммы 
```
img_eq = cv2.equalizeHist(img_gray) 

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_eq, cmap='gray')
plt.subplot(1, 2, 2)
plt.hist(img_gray.ravel(), range=(0, 255), bins=256)
plt.show()
```
![]()
### Выравнивание гистограммы по окрестности
```
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))  # Выравнивание по окрестности
# подробнее об методе https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
img_eq = clahe.apply(img_gray)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_eq, cmap='gray')
plt.subplot(1, 2, 2)
plt.hist(img_gray.ravel(), range=(0, 255), bins=256)
plt.show()
```
![]()

## Работа с изображениями с использованием свёрток
### увеличение чёткости
```
kernel = np.array([
    [-0.1, -0.1, -0.1],
    [-0.1,    2, -0.1],
    [-0.1, -0.1, -0.1],
])
img_out = cv2.filter2D(image, -1, kernel)
plt.imshow(img_out)
plt.show()
```
![]()

### затемнение
```
kernel = np.array([
    [-0.1,  0.1, -0.1],
    [ 0.1,  0.5,  0.1],
    [-0.1,  0.1, -0.1],
])
img_out = cv2.filter2D(image, -1, kernel)
plt.imshow(img_out)
plt.show()
```
![]()
### увеличение яркости
```
kernel = np.array([
    [-0.1,  0.2, -0.1],
    [ 0.2,    1,  0.2],
    [-0.1,  0.2, -0.1],
])
img_out = cv2.filter2D(image, -1, kernel)
plt.imshow(img_out)
plt.show()
```
![]()
Выделение границ
### фильтр Робертса
```
kernel = np.array([
    [1,  0],
    [0, -1],
])
img_out = cv2.filter2D(img_gray, -1, kernel)
plt.imshow(img_out, cmap='gray')
plt.show()
```
![]()
### фильтр Превитт
```
kernel = np.array([
    [-1,  0, 1],
    [-1,  0, 1],
    [-1,  0, 1],
])
img_out = cv2.filter2D(img_gray, -1, kernel)
plt.imshow(img_out, cmap='gray')
plt.show()
```
![]()
### фильтр Собеля
```
kernel = np.array([
    [-1,  0, 1],
    [-2,  0, 2],
    [-1,  0, 1],
])
img_out = cv2.filter2D(img_gray, -1, kernel)
plt.imshow(img_out, cmap='gray')
plt.show()
```
![]()

### фильтр Собеля горизонтальный
```
kernel = np.array([
    [ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1],
])
img_out = cv2.filter2D(img_gray, -1, kernel)
plt.imshow(img_out, cmap='gray')
plt.show()
```
![]()

## Работа с видео
```
# Получение видео. Объект VideoCapture позволяет работать с видеопотоком с камеры или из файла
# Для работы с файлом указывается путь к нему, для работы с вебкамерой указывается её номер, начиная с 0
cap = cv2.VideoCapture(0)
# Кадр получается с помощью метода read (возвращается флаг успешной операции и кадр)
# ret - returns : bool
ret, frame = cap.read()
# После работы с камерой VideoCapture освобождается
cap.release()
```
```
# Изображение получается в формате BGR
plt.imshow(frame)
plt.show()
```
![]()
```
# OpenCV позволяет выводить кадры в отдельном окне. Для этого служит функция imshow
# На вход подаётся изображение в формате BGR
# Первым аргументом укзывается имя окна
cv2.imshow('Frame', frame)
# Для задержки показа используется функция waitKey
# Аргумент указывает задержку в мс. Если 0, то ждёт нажатия любой клавиши
# Функция waitKey возвращает код нажатой клавиши
cv2.waitKey(0)
# После работы с ним окно удаляется
cv2.destroyWindow('Frame')
```
### Воспроизведение видео
```
# OpenCV позволяет выводить кадры в отдельном окне. Для этого служит функция imshow
# На вход подаётся изображение в формате BGR
# Первым аргументом укзывается имя окна
cv2.imshow('Frame', frame)
# Для задержки показа используется функция waitKey
# Аргумент указывает задержку в мс. Если 0, то ждёт нажатия любой клавиши
# Функция waitKey возвращает код нажатой клавиши
cv2.waitKey(0)
# После работы с ним окно удаляется
cv2.destroyWindow('Frame')
```
### Рисование
Прямоугольник
```
image2 = image.copy()
cv2.rectangle(image2, (0, 0), (100, 200), (255, 0, 0, 55), 4)
plt.imshow(image2)
plt.show()
```
![]()

Круг
```
image2 = image.copy()
cv2.circle(image2, (100, 100), 30, (255, 0, 0), 4)
plt.imshow(image2)
plt.show()
```
![]()

Контур
```
image2 = image.copy()
contours = [
    np.array([(0, 0), (0, 100), (100, 100), (100, 50)]),
    np.array([(200, 200), (100, 100), (200, 100)]),
]
cv2.drawContours(image2, contours, -1, (255, 0, 0), 4)

plt.imshow(image2)
plt.show()
```
![]()

Текст
```
image2 = image.copy()
cv2.putText(image2, 'Text It', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
plt.imshow(image2)
plt.show()
```
![]()

### Работа с изображением
Перевод в градации серого
```
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(image_gray, cmap='gray')
plt.show()
```
![]()

Бинаризация
```
_, thresh = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)
plt.imshow(thresh, cmap='gray')
plt.show()
```
![]()

Поиск контуров
```
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours), contours[2])
```
![]()

Рисование контуров
```
image_out = np.zeros_like(image)
image_out = cv2.drawContours(image_out, contours, -1, (0, 255, 0), 2)
plt.imshow(image_out)
plt.show()
```
![]()


# Задания
1. Для данных [Nails segmentation](https://www.kaggle.com/vpapenko/nails-segmentation) объедините пары изображение-маска (список файлов получить с помощью библиотеки os название парных изображений совпадают).
2. Выведите по очереди эти пары с помощью OpenCV (переключение по нажатию клавиши).
3. Выделите контуры на масках и отрисуйте их на изображениях.
4. Воспроизведите любой видеофайл с помощью OpenCV в градациях серого.

## Лабораторная работа 5. Красный свет / зелёный свет
Необходимо средствами OpenCV реализовать детскую корейскую игру “Красный свет / зелёный свет”.
![]()

**Лабораторная 5.1**
Программа должна реализовывать следующий функционал:

1. Покадровое получение видеопотока с камеры. Использовать камеру ноутбука, вебкамеру или записать видео файл с вебкамеры товарища и использовать его.
2. Реализовать обнаружение движения в видеопотоке: попарно сравнивать текущий и предыдущий кадры.
3. По мере проигрывания видео в отдельном окне отрисовывать двухцветную карту с результатом: красное - есть движение, зелёное - нет движения
4. Добавить таймер, по которому включается и выключается обнаружение движения. О текущем режиме программы сообщать текстом с краю изображения: “Красный свет” - движение обнаруживается, “Зелёный свет” - движение не обнаруживается.
5. Реализовать более сложный алгоритм обнаружения движения, устойчивый к шумам вебкамеры (OpticalFlow)
