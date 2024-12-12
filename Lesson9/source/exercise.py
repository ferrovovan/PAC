#!/usr/bin/python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def preprocess_data(filepath):
    """
    Загружает данные и выполняет их предобработку.
    - Заполняет пропуски медианой или модой.
    - Кодирует категориальные признаки в числовые.
    
    Args:
        filepath (str): Путь к CSV-файлу с данными.
    Returns:
        pd.DataFrame: Матрица признаков X.
        pd.Series: Целевая переменная y.
    """
    data = pd.read_csv(filepath)
    data['Age'].fillna(data['Age'].median(), inplace=True)  # Заполняем пропуски в возрасте медианой.
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)  # Заполняем пропуски модой.
    data['Fare'].fillna(data['Fare'].median(), inplace=True)  # Заполняем пропуски в цене медианой.
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)  # Преобразуем текстовые категории.
    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    X = data[features]
    y = data['Survived']
    return X, y


def split_data(X, y):
    """
    Делит данные на тренировочную, валидационную и тестовую выборки.
    
    Args:
        X (pd.DataFrame): Матрица признаков.
        y (pd.Series): Целевая переменная.
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def grid_search_models(X_train, y_train):
    """
    Подбирает гиперпараметры для разных моделей.
    
    Args:
        X_train (pd.DataFrame): Тренировочные признаки.
        y_train (pd.Series): Тренировочная цель.
    Returns:
        dict: Словарь с результатами подбора гиперпараметров и точностью на тесте.
    """
    models = {
        "Random Forest": (RandomForestClassifier(random_state=42), {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=42), {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }),
        "Logistic Regression": (LogisticRegression(max_iter=500), {
            'C': [0.01, 0.1, 1, 10, 100]
        }),
        "KNN": (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        })
    }

    results = {}
    for name, (model, params) in models.items():
        grid = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)  # Перебор параметров с кросс-валидацией.
        grid.fit(X_train, y_train)
        results[name] = {"Best Params": grid.best_params_, "Model": grid.best_estimator_}
    return results


def evaluate_models(models, X_test, y_test):
    """
    Оценивает точность моделей на тестовой выборке.
    
    Args:
        models (dict): Словарь моделей с лучшими гиперпараметрами.
        X_test (pd.DataFrame): Тестовые признаки.
        y_test (pd.Series): Тестовая цель.
    Returns:k-ближайших соседей (k-NN)
        dict: Точность моделей.
    """
    accuracies = {}
    for name, data in models.items():
        model = data["Model"]
        y_pred = model.predict(X_test)
        accuracies[name] = accuracy_score(y_test, y_pred)  # Считаем точность.
    return accuracies


def feature_importance_selection(X_train, y_train, X_test, y_test, n_values=[2, 4, 8]):
    """
    Оценивает точность модели RandomForest на самых важных признаках.
    
    Args:
        X_train (pd.DataFrame): Тренировочные признаки.
        y_train (pd.Series): Тренировочная цель.
        X_test (pd.DataFrame): Тестовые признаки.
        y_test (pd.Series): Тестовая цель.
        n_values (list): Список количества отобранных признаков.
    Returns:
        dict: Результаты точности на отобранных признаках.
    """
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    results = {}
    for n in n_values:
        top_features = importances.head(n).index.tolist()
        rf_top = RandomForestClassifier(random_state=42)
        rf_top.fit(X_train[top_features], y_train)
        y_pred = rf_top.predict(X_test[top_features])
        results[f"Top {n} Features"] = {
            "Features": top_features,
            "Accuracy": accuracy_score(y_test, y_pred)
        }
    return results


if __name__ == "__main__":
    # Загрузка и предобработка данных
    X, y = preprocess_data("data/train.csv")

    # Разделение данных
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Подбор гиперпараметров для моделей
    models = grid_search_models(X_train, y_train)

    # Оценка точности моделей на тестовой выборке
    test_accuracies = evaluate_models(models, X_test, y_test)
    print("Model Test Accuracies:")
    for name, acc in test_accuracies.items():
        print(f"{name}: {acc:.4f}")

    # Оценка точности на самых важных признаках
    important_features_results = feature_importance_selection(X_train, y_train, X_test, y_test)
    print("\nВажнейшие особенности:")
    for key, value in important_features_results.items():
        print(f"{key}: Features: {value['Features']}, Accuracy: {value['Accuracy']:.4f}")



