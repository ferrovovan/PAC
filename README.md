# Позаимствованный исходный код
Исходный код с 12 по 15 (вкл) задачи позаимствован отсюда:  
https://github.com/SampiNova/nsu_py

16 Лабораторная  
https://github.com/Zea-Zee/HWs/blob/main/PHC/Lesson13.ipynb

# PAC
Оригинал: https://github.com/vvabi-sabi/PAC/tree/main

# Структура лекций
#### Введение в Python
1. Основы + PEP8
2. Типы данных
3. ООП

#### Изучение часто используемых библиотек
4. *numpy*
5. *Matplotlib* + *OpenCV*
6. *OpenCV*
7. *Pandas*

#### Введение в Машинное обучение
8. Часть 1. Знакомство с *Scikit-learn*
9. Часть 2. Применение *Scikit-learn*

#### Нейронные сети
10. Перцептрон
11. Скрытые слои.
12. Backpropogation

#### Фреймворк pytorch
13. Введение в PyTorch
14. Создание свёртки
15. CNN архитектура
16. RNN - Рекуррентные нейронные сети

# Перевод из `.ipynb` в `.md`
```
sudo apt install  python3-nbconvert

python3 -m nbconvert --to markdown Lesson${i}.ipynb
```
# Установка библиотек через apt
```
sudo apt update
sudo apt install python3-opencv python3-pandas python3-sklearn python3-xgboost
```
### Для 16 лабораторной
```
sudo apt install python3-pip python3-dev libtorch-dev

# torch (PyTorch, для машинного обучения и нейронных сетей)
pip install torch
# nltk (Natural Language Toolkit для обработки текста)
pip install nltk
# gensim (Для работы с моделями типа Word2Vec)
pip install gensim
```
