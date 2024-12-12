#!/usr/bin/python3

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
from torchvision import transforms

def calculate_avg_digits(X_train, y_train):
    avg_digits = []
    for digit in range(10):
        mask = (y_train == digit)  # Фильтруем только данные этого класса
        avg_digit = np.mean(X_train[mask], axis=0)  # Среднее по строкам
        avg_digits.append(avg_digit)
    return np.array(avg_digits)

def classify_digit(image, avg_digit, bias=0):
    score = np.dot(image, avg_digit) + bias  # Скалярное произведение + bias
    return 1 if score > 0 else 0  # Бинарная классификация

def calculate_accuracy(X_test, y_test, avg_digits):
    accuracies = []
    for digit in range(10):
        predictions = [classify_digit(x, avg_digits[digit]) for x in X_test]
        true_labels = (y_test == digit).astype(int)  # Истинные метки
        accuracy = np.mean(predictions == true_labels)
        accuracies.append(accuracy)
    return accuracies

def combined_model(image, avg_digits, biases):
    return np.array([classify_digit(image, avg_digits[i], biases[i]) for i in range(10)])

def calculate_metrics(X_test, y_test, avg_digits):
    precision_scores = []
    recall_scores = []
    for digit in range(10):
        predictions = [classify_digit(x, avg_digits[digit]) for x in X_test]
        true_positives = sum((predictions == 1) & (y_test == digit))
        false_positives = sum((predictions == 1) & (y_test != digit))
        false_negatives = sum((predictions == 0) & (y_test == digit))

        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0
        if (true_positives + false_negatives) > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0

        precision_scores.append(precision)
        recall_scores.append(recall)
    return precision_scores, recall_scores

def visualize_tsne(X, y, n_samples=30):
    # Выбираем n_samples изображений каждого класса
    sampled_indices = []
    for digit in range(10):
        indices = np.where(y == digit)[0][:n_samples]
        sampled_indices.extend(indices)
    X_sampled = X[sampled_indices]
    y_sampled = y[sampled_indices]

    # Применяем t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X_sampled)

    # Визуализация
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y_sampled, palette='tab10', legend='full')
    plt.title("t-SNE визуализация исходных данных")
    plt.show()

def check_validity(data):
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")
    return True


def visualize_model_output_tsne(X_test, avg_digits, biases):
    outputs = np.array([combined_model(x, avg_digits, biases) for x in X_test])
    
    # Проверка данных на NaN и бесконечности
    try:
        check_validity(outputs)
    except ValueError as e:
        print(e)
        return

    tsne = TSNE(n_components=2, random_state=42)
    outputs_embedded = tsne.fit_transform(outputs)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=outputs_embedded[:, 0], y=outputs_embedded[:, 1], palette='tab10', legend=None)
    plt.title("t-SNE визуализация выходов модели")
    plt.show()


if __name__ == "__main__":
    import sys
    # 1. Загрузка данных MNIST через torchvision
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Загрузка обучающих и тестовых данных
    print("1. Загрузка обучающих и тестовых данных", file=sys.stderr)
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Преобразуем изображения в numpy-массивы
    X_train = np.array([train_dataset[i][0].numpy().flatten() for i in range(len(train_dataset))])
    y_train = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

    X_test = np.array([test_dataset[i][0].numpy().flatten() for i in range(len(test_dataset))])
    y_test = np.array([test_dataset[i][1] for i in range(len(test_dataset))])

    # 2. Вычисление средних векторов для каждого класса
    print("2. Вычисление средних векторов для каждого класса", file=sys.stderr)

    avg_digits = calculate_avg_digits(X_train, y_train)

    # 3. Рассчёт точности классификаторов
    #accuracies = calculate_accuracy(X_test, y_test, avg_digits)
    #print("Точности для каждого класса:", accuracies)

    # 4. Рассчёт precision и recall
    #precision_scores, recall_scores = calculate_metrics(X_test, y_test, avg_digits)
    #print("Precision для каждого класса:", precision_scores)
    #print("Recall для каждого класса:", recall_scores)

    # 5. Визуализация t-SNE исходных данных
    visualize_tsne(X_test, y_test)

    # 6. Визуализация t-SNE выходов модели
    biases = np.zeros(10)  # Для простоты bias = 0
    visualize_model_output_tsne(X_test, avg_digits, biases)

