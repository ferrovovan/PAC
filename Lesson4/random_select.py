import sys
import random
import numpy as np

def read_numbers_from_file(file_path):
    with open(file_path, 'r') as file:
        return list(map(int, file.read().strip().split()))


def random_select_classic(real_data, synthetic_data, probability):
    mixed_data = []
    for real, synthetic in zip(real_data, synthetic_data):
        if random.random() < probability:
            mixed_data.append(synthetic)
        else:
            mixed_data.append(real)
    return mixed_data

def random_select_mask(real_data, synthetic_data, probability):
    # Создаем новый список, заполненный реальными данными
    mixed_data = [real for real in real_data]
    
    # Определяем количество элементов, которые будем заменять
    count_to_replace = int(len(real_data) * probability)
    
    # Выбираем случайные индексы для замены
    indices_to_replace = random.sample(range(len(real_data)), count_to_replace)
    
    # Заменяем элементы в mixed_data на синтетические
    for index in indices_to_replace:
        mixed_data[index] = synthetic_data[index]
    
    return mixed_data

def random_select_numpy_mask(real_data, synthetic_data, probability):
    # Генерируем случайные числа от 0 до 1 и сравниваем с вероятностью
    mask = np.random.rand(len(real_data)) < probability
    # Используем маску для выбора данных
    mixed_data = np.where(mask, synthetic_data, real_data)
    return mixed_data


def main():
    if len(sys.argv) != 4:
        print("Использование: python random_select.py <file_1.txt> <file_2.txt> <probability>")
        sys.exit(1)

    # Считывание
    file_1 = sys.argv[1]
    file_2 = sys.argv[2]
    probability = float(sys.argv[3])

    if probability < 0 or probability > 1:
        print("Вероятность должна быть в диапазоне от 0 до 1.")
        sys.exit(1)

    real_data = read_numbers_from_file(file_1)
    synthetic_data = read_numbers_from_file(file_2)

    if len(real_data) != len(synthetic_data):
        print("Длина массивов должна быть одинаковой.")
        sys.exit(1)


    # Метод 1: Через random и if else
    result_method_1 = random_select_classic(real_data, synthetic_data, probability)
    print("Результат методом 1:", result_method_1)

    # Метод 2: Через random и маску
    result_method_2 = random_select_mask(real_data, synthetic_data, probability)
    print("Результат методом 2:", result_method_2)
    
    # Метод 3: Использование NumPy и маски
    result_method_3 = random_select_numpy_mask(real_data, synthetic_data, probability)
    print("Результат методом 3:", result_method_3)


if __name__ == "__main__":
    main()

