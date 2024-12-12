#!/usr/bin/python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier  # Импорт XGBoost


# Лабораторная 9.1

# 1. Загрузка данных и разделение на train/test
def load_and_split_data(filepath):
    """
    Загружает данные из CSV и разделяет их на train/test.
    10% данных выбираются случайно для тестовой выборки.
    
    Args:
        filepath (str): Путь к файлу.
    Returns:
        pd.DataFrame, pd.DataFrame: Тренировочная и тестовая выборки.
    """
    data = pd.read_csv(filepath)
    # Удаляем ненужные столбцы
    data = data.drop(columns=["Unnamed: 0", "row_number", "check_number"], errors="ignore")
    # Делим данные на train/test
    train, test = train_test_split(data, test_size=0.1, random_state=42)  # 10% тестовых данных.
    return train, test

# 2. Обучение моделей
def train_models(train):
    """
    Обучает Decision Tree, Logistic Regression и XGBoost на тренировочных данных.
    
    Args:
        train (pd.DataFrame): Тренировочные данные.
    Returns:
        dict: Обученные модели.
    """
    X_train = train.drop(columns=["label"])  # Признаки
    y_train = train["label"]  # Целевая переменная
    
    # Решающее дерево
    dt_model = DecisionTreeClassifier(random_state=42)  # Инициализация модели.
    dt_model.fit(X_train, y_train)  # Обучение модели на данных.
    
    # Логистическая регрессия
    lr_model = LogisticRegression(max_iter=500, random_state=42)  # Инициализация модели.
    lr_model.fit(X_train, y_train)  # Обучение модели на данных.
    
    # XGBoost
    xgb_model = XGBClassifier(eval_metric="logloss", random_state=42)
    xgb_model.fit(X_train, y_train)  # Обучение модели на данных.
    
    return {"Decision Tree": dt_model, "Logistic Regression": lr_model, "XGBoost": xgb_model}

# 3. Замер Accuracy
def evaluate_models(models, test):
    """
    Вычисляет точность для каждой модели на тестовых данных.
    
    Args:
        models (dict): Словарь обученных моделей.
        test (pd.DataFrame): Тестовые данные.
    Returns:
        dict: Точность моделей.
    """
    X_test = test.drop(columns=["label"])
    y_test = test["label"]
    
    accuracies = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)  # Предсказания модели.
        accuracies[name] = accuracy_score(y_test, y_pred)  # Доля правильных предсказаний.
    return accuracies

# 5. Выбор двух самых важных признаков
def select_top_features(train, test):
    """
    Выбирает два самых важных признака с помощью Decision Tree и проверяет точность на них.
    
    Args:
        train (pd.DataFrame): Тренировочные данные.
        test (pd.DataFrame): Тестовые данные.
    Returns:
        dict: Точность модели на выбранных признаках.
    """
    X_train = train.drop(columns=["label"])
    y_train = train["label"]
    X_test = test.drop(columns=["label"])
    y_test = test["label"]
    
    # Обучение решающего дерева
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Извлечение важности признаков
    feature_importances = pd.Series(dt_model.feature_importances_, index=X_train.columns)
    top_features = feature_importances.nlargest(2).index.tolist()  # Два самых важных признака.
    
    # Обучение модели на выбранных признаках
    dt_model_top = DecisionTreeClassifier(random_state=42)
    dt_model_top.fit(X_train[top_features], y_train)
    
    # Оценка точности
    y_pred = dt_model_top.predict(X_test[top_features])
    accuracy = accuracy_score(y_test, y_pred)
    return {"Top Features": top_features, "Accuracy": accuracy}


# Лабораторная 9.2

# Реализация собственного Random Forest
class MyRandomForest:
    """
    Реализация алгоритма Случайного леса.
    Использует DecisionTreeClassifier из sklearn.
    """
    def __init__(self, n_estimators=10, max_features="sqrt", random_state=None):
        """
        Args:
            n_estimators (int): Количество деревьев в лесу.
            max_features (str): Максимальное количество признаков для каждого узла дерева.
            Возможные значения: 
                         "auto" (по умолчанию) эквивалентно sqrt(n_features),
                         "sqrt": На каждом узле дерева выбираются признаки, равные квадратному корню от общего числа признаков.
                         "log2": Используется логарифм по основанию 2 от числа признаков. Это чаще всего приводит к меньшему количеству признаков на разбиение, что может быть полезно для борьбы с переобучением.
                         int: Вы можете задать целое число, указывающее точное количество признаков, которое будет использоваться для каждого разбиения. Например, если у вас 10 признаков, и вы хотите, чтобы на каждом разбиении использовались только 5 признаков, установите max_features=5.
                         float: Это значение определяет долю признаков от общего числа признаков. Например, если у вас 10 признаков и вы установите max_features=0.5, будет использоваться 5 признаков на каждом разбиении.
                         None: Если не указано, то используется все количество признаков для каждого разбиения.

            random_state (int): Фиксация случайности для воспроизводимости.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.features = []

    def fit(self, X, y):
        """
        Обучает случайный лес на данных.
        
        Args:
            X (pd.DataFrame): Признаки.
            y (pd.Series): Целевая переменная.
        """
        np.random.seed(self.random_state)
        for _ in range(self.n_estimators):
            # Случайная выборка строк и признаков
            bootstrap_idx = np.random.choice(len(X), size=len(X), replace=True)
            selected_features = np.random.choice(X.columns, 
                                                  size=int(np.sqrt(len(X.columns))), 
                                                  replace=False)
            X_bootstrap = X.iloc[bootstrap_idx][selected_features]
            y_bootstrap = y.iloc[bootstrap_idx]
            
            # Обучение одного дерева
            tree = DecisionTreeClassifier(max_features=self.max_features, random_state=self.random_state)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            self.features.append(selected_features)

    def predict(self, X):
        """
        Предсказывает значения с помощью случайного леса.
        
        Args:
            X (pd.DataFrame): Признаки.
        Returns:
            np.array: Предсказания.
        """
        predictions = []
        for tree, features in zip(self.trees, self.features):
            predictions.append(tree.predict(X[features]))
        # Мажоритарное голосование
        predictions = np.array(predictions).T
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)


# Основной блок
if __name__ == "__main__":
    # 9.1.1 Загрузка данных
    train, test = load_and_split_data("data/titanic_prepared.csv")

    # 9.1.2 Обучение моделей
    models = train_models(train)

    # 9.1.3 Оценка точности
    accuracies = evaluate_models(models, test)
    print("Accuracies of models:", accuracies)

    # 9.1.5 Выбор двух самых важных признаков
    top_features_result = select_top_features(train, test)
    print("Top features accuracy:", top_features_result)

    # 9.2.2 Реализация Random Forest
    X_train = train.drop(columns=["label"])
    y_train = train["label"]
    X_test = test.drop(columns=["label"])
    y_test = test["label"]

    rf = MyRandomForest(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print("MyRandomForest accuracy:", rf_accuracy)

