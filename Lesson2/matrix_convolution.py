import argparse



def read_matrices_from_file(file_path):
    """Чтение матриц из файла и их преобразование в списки списков"""
    matrix1 = []
    matrix2 = []
    reading_matrix_1 = True

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                # Если строка пустая, начинаем чтение второй матрицы
                reading_matrix_1 = False
                continue

            if reading_matrix_1:
                matrix1.append(list(map(int, line.split())))
            else:
                matrix2.append(list(map(int, line.split())))

    return matrix1, matrix2

def convolve2d(matrix1, matrix2):
    """Выполнение операции свёртки (2D)"""
    m1_height = len(matrix1)
    m1_width = len(matrix1[0])
    k_height = len(matrix2)
    k_width = len(matrix2[0])

    # Определяем размеры результирующей матрицы
    result_height = m1_height - k_height + 1
    result_width = m1_width - k_width + 1

    # Результирующая матрица
    result = [[0 for _ in range(result_width)] for _ in range(result_height)]

    # Применение свёртки
    for i in range(result_height):
        for j in range(result_width):
            sum_val = 0
            for ki in range(k_height):
                for kj in range(k_width):
                    sum_val += matrix1[i + ki][j + kj] * matrix2[ki][kj]
            result[i][j] = sum_val

    return result

def write_matrix_to_file(matrix, file_path):
    """Запись результирующей матрицы в файл"""
    with open(file_path, 'w') as f:
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Convolution operation script.')
    parser.add_argument('input_file', type=str, help='Input file containing matrices')
    parser.add_argument('output_file', type=str, help='Output file to save result')
    args = parser.parse_args()

    # Чтение матриц
    matrix1, matrix2 = read_matrices_from_file(args.input_file)

    # Проверка размеров матриц
    if len(matrix2) > len(matrix1) or len(matrix2[0]) > len(matrix1[0]):
        raise ValueError("Размер второй матрицы не может быть больше первой.")

    # Выполнение свёртки
    result_matrix = convolve2d(matrix1, matrix2)

    # Запись результата в файл
    write_matrix_to_file(result_matrix, args.output_file)

if __name__ == "__main__":
    main()

