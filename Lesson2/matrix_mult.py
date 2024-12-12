import argparse

parser = argparse.ArgumentParser(description='Multiply matricies script.')
parser.add_argument('input_file' , type=str, help='input file' )
parser.add_argument('output_file', type=str, help='output file')
args = parser.parse_args()


# Считывание матриц
matrix1 = []
matrix2 = []
reading_matrix_1 = True

with open(args.input_file, 'r') as f:
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

# Проверка на совместимость матриц
m1_height = len(matrix1)
m1_width = len(matrix1[0])
m2_height = len(matrix2)
m2_width = len(matrix2[0])

if m1_width != m2_height:
    raise ValueError("Количество столбцов первой матрицы должно быть равно количеству строк второй матрицы.")

# Перемножение матриц
matrix_res = [[0 for _ in range(m2_width)] for _ in range(m1_height)]

for i in range(m1_height):
    for j in range(m2_width):
        for k in range(m2_height):
            matrix_res[i][j] += matrix1[i][k] * matrix2[k][j]

# Запись результирующей матрицы
with open(args.output_file, 'w') as f:
    for row in matrix_res:
        f.write(' '.join(map(str, row)) + '\n')
