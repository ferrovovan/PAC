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
