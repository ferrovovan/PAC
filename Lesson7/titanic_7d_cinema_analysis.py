import pandas as pd
import numpy as np

# Загрузка данных
cinema_sessions = pd.read_csv('cinema_sessions.csv')
titanic_with_labels = pd.read_csv('titanic_with_labels.csv')

# Очистка заголовков
titanic_with_labels.columns = titanic_with_labels.columns.str.strip()
cinema_sessions.columns = cinema_sessions.columns.str.strip()

# 1) Фильтрация и преобразование пола
# Преобразование пола: 'ж' = 0, 'м' = 1, 'M' = 1
titanic_with_labels = titanic_with_labels[titanic_with_labels['sex'].notna()]
titanic_with_labels['sex'] = titanic_with_labels['sex'].replace({'ж': 0, 'м': 1, 'M': 1})

# 2) Заполнение номера ряда
max_row_number = titanic_with_labels['row_number'].max()
titanic_with_labels['row_number'].fillna(max_row_number, inplace=True)

# 3) Обработка количества выпитого
mean_liters_drunk = titanic_with_labels['liters_drunk'].loc[
    (titanic_with_labels['liters_drunk'] >= 0) & 
    (titanic_with_labels['liters_drunk'] < 50)].mean()
titanic_with_labels['liters_drunk'] = titanic_with_labels['liters_drunk'].where(
    (titanic_with_labels['liters_drunk'] >= 0) & (titanic_with_labels['liters_drunk'] < 50), 
    mean_liters_drunk
)

# 4) Группировка возраста
titanic_with_labels['age_children'] = (titanic_with_labels['age'] < 18).astype(int)
titanic_with_labels['age_adults'] = ((titanic_with_labels['age'] >= 18) & (titanic_with_labels['age'] <= 50)).astype(int)
titanic_with_labels['age_seniors'] = (titanic_with_labels['age'] > 50).astype(int)

# Удаление старого столбца с возрастом
titanic_with_labels.drop(columns=['age'], inplace=True)

# 5) Преобразование напитка
titanic_with_labels['drink'] = titanic_with_labels['drink'].replace({'Cola': 0, 'Fanta': 0, 'Beer': 1})

# 6) Кодирование номера чека
# Сначала преобразуем check_number в строку для правильного объединения
titanic_with_labels['check_number'] = titanic_with_labels['check_number'].astype(str)
cinema_sessions['check_number'] = cinema_sessions['check_number'].astype(str)

# Объединение таблиц для сопоставления времени сеанса
merged_sessions = titanic_with_labels.merge(cinema_sessions, on='check_number', how='left')

# Преобразование session_start в datetime для удобства
merged_sessions['session_start'] = pd.to_datetime(merged_sessions['session_start'], format='%H:%M:%S.%f')

# Кодирование времени сеанса
merged_sessions['morning'] = (merged_sessions['session_start'].dt.hour < 12).astype(int)
merged_sessions['day'] = ((merged_sessions['session_start'].dt.hour >= 12) & (merged_sessions['session_start'].dt.hour < 18)).astype(int)
merged_sessions['evening'] = (merged_sessions['session_start'].dt.hour >= 18).astype(int)

# Проверка результата
print(merged_sessions.head())

# Сохранение обработанных данных в новый файл, если необходимо
merged_sessions.to_csv('processed_titanic_cinema_analysis.csv', index=False)

