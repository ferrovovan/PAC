## Сверточная нейронная сеть

![Архитектура](https://upload.wikimedia.org/wikipedia/commons/5/55/%D0%90%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80%D0%B0_%D1%81%D0%B2%D0%B5%D1%80%D1%82%D0%BE%D1%87%D0%BD%D0%BE%D0%B9_%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%BE%D0%B9_%D1%81%D0%B5%D1%82%D0%B8.png)



## Датасет ImageNet
ImageNet — набор данных, состоящий из более чем 15 миллионов размеченных высококачественных изображений, разделенных на 22000 категорий. Изображения были взяты из интернета и размечены вручную людьми-разметчиками 

<img src="images/cifar10.png" alt="Dataset" height=50% width=50% />

На август 2017 года в ImageNet 14 197 122 изображения, разбитых на 21 841 категорию.


<img src="images/imagenet_err.png" width=60% height=60%>

# Архитектуры нейронных сетей

## AlexNet
Архитектура AlexNet состоит из пяти свёрточных слоёв, между которыми располагаются pooling-слои и слои нормализации, а завершают нейросеть три полносвязных слоя.  
На схеме архитектуры все выходные изображения делятся на два одинаковых участка — это связано с тем, что нейросеть обучалась на старых GPU GTX580, у которых было всего 3 ГБ видеопамяти. Для обработки использовались две видеокарты, чтобы параллельно выполнять операции над двумя частями изображения.
<img src="images/alexnet.png" width=80% height=80%>

# VGG-16
VGG16 — одна из самых знаменитых моделей, отправленных на соревнование ILSVRC-2014. Она является улучшенной версией AlexNet, в которой заменены большие фильтры (размера 11 и 5 в первом и втором сверточном слое, соответственно) на несколько фильтров размера 3х3, следующих один за другим. Сеть VGG16 обучалась на протяжении нескольких недель при использовании видеокарт NVIDIA TITAN BLACK.

<img src="https://neurohive.io/wp-content/uploads/2018/11/vgg16-neural-network-1-e1542973058418.jpg" alt="VGG" height=50% width=50% />
<img src="https://neurohive.io/wp-content/uploads/2018/11/vgg16-2.png" alt="VGG Plain" height=50% width=50% />



## VGG-19
Основная идея VGG-архитектур — использование большего числа слоёв с фильтрами меньшего размера. Существуют версии VGG-16 и VGG-19 с 16 и 19 слоями соответственно.  
С маленькими фильтрами получается не так много параметров, но при этом мы сможем гораздо эффективнее обрабатывать их.
<img src="images/vgg.png" width=50% height=50%>

## ResNet
Основой сети ResNet является Residual-блок (остаточный блок) с shortcut-соединением, через которое данные проходят без изменений. Res-блок представляет собой несколько сверточных слоев с активациями, которые преобразуют входной сигнал x в F(x). Shortcut-соединение — это тождественное преобразование x -> x.

Создатели ResNet решили не складывать слои друг на друга для изучения отображения нужной функции напрямую, а использовать остаточные блоки, 
которые пытаются «подогнать» это отображение. Так ResNet стала первой остаточной нейронной сетью. 
Иначе говоря, она «перепрыгивает» через некоторые слои. Они больше не содержат признаков и используются для нахождения остаточной функции H(x) = F(x) + x вместо того, чтобы искать H(x) напрямую.
<img src="images/res_block.png" width=80% height=80%>

В результате экспериментов с ResNet выяснилось, что очень глубокие сети действительно можно обучить без ухудшения точности. Нейросеть достигла наименьшей ошибки в задачах классификации, которая превзошла даже человеческий результат.


<img src="images/resnet.png" width=50% height=50%>

## MobileNet V2

- Особенностью данной архитектуры является отсутствие max pooling-слоёв. Вместо них для снижения пространственной размерности используется свёртка с параметром stride равным 2. Также изменение свёртки уменьшает количество операций сети

<img src="https://habrastorage.org/webt/wl/yo/sz/wlyoszqnws58itd4ojt1cqt7sng.png" alt="MobBlock" height=30% width=30% />


# Сегментация изображений
<img src="images/segment.png" width=80% height=80%>

<img src="images/segment2.png" width=80% height=80%>

## Fully Convolutional Network
<img src="images/FCN8.ppm" width=80% height=80%>

# SegNet
<img src="images/segnet.png" width=80% height=80%>

<img src="images/denoise.png" width=80% height=80%>

Upsampling:

https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html

# UNet
U-Net — это свёрточная нейронная сеть, которая была создана в 2015 году для сегментации биомедицинских изображений в отделении Computer Science Фрайбургского университета. Архитектура сети представляет собой полносвязную свёрточную сеть, модифицированную так, чтобы она могла работать с меньшим количеством примеров (обучающих образов) и делала более точную сегментацию.

<img src="images/unet.png" width=60% height=60%>

## Метрика
Intersection over union (IoU)
<img src="images/iou.png" width=40% height=40%>

## Функция потерь

* Cross entropy
* Weighted cross entropy
* DICE
$$ DC = 1 - {2\sum y_i p_i \over \sum y_i + \sum p_i} $$

# Детекция
<img src="images/detect.png" width=80% height=80%>

## Non-maximum Suppression (NMS)
Алгоритм:  
1) Выбрать область с наибольшией уверенностью и запомнить его  
2) Найти IoU с оставшимися областями  
3) Удалить области с IoU больше заданного порога  
4) Повторять пп. 1-3, пока все области не отфильтруются  
<img src="images/nms.png" width=80% height=80%>

# Single Shot Detectors

## Yolo
Изображение делится на части и предсказывается отдельно для каждой части
<img src="images/yolo_grid.jpeg" width=80% height=80%>

Каждая часть предсказывает координаты, уверенность и вероятность каждого из 20 классов
<img src="images/yolo.png" width=80% height=80%>

## SSD
<img src="images/ssd.jpeg" width=80% height=80%>

# Two Shot Detectors

https://habr.com/ru/company/jetinfosystems/blog/498294/

## R-CNN
<img src="images/rcnn.png" width=80% height=80%>

## R-CNN
<img src="images/rcnn_conv.png" width=50% height=50%>

https://habr.com/ru/company/jetinfosystems/blog/498652/

## Fast R-CNN
<img src="images/fast_rcnn.png" width=80% height=80%>

## Faster R-CNN
<img src="images/faster_rcnn.png" width=50% height=50%>

## Сравнение скоростей
<img src="images/rcnn_speed.png" width=80% height=80%>


```python
# define model
model = torchvision.models.resnet18(pretrained=True)
avgpool_emb = None

def get_embedding(module, inputs, output):
    global avgpool_emb
    avgpool_emb = output

model.avgpool.register_forward_hook(get_embedding)
model.eval()
```

## Лабораторная работа 15.
Задание:
Сделать визуализацию работы Resnet50 (как на видео https://www.youtube.com/watch?v=fZvOy0VXWAI&t=85s)
1. Установить torch, torchvision, скачать Resnet (pretrained=True)
2. Написать функцию get_features_map() (установить hook), которая "возвращает" результаты работы слоя layer4 (находится перед слоем avgpool)
3. Достать из модели матрицу весов "W" последнего (полносвязного) слоя (fc).
4. Сложить карты признаков (2048 штук для ResNet50), с коэффициентами w_i, как показано в "статье" - https://alexisbcook.github.io/posts/global-average-pooling-layers-for-object-localization/


