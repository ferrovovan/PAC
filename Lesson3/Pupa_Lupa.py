def read_matrix(filename) -> list:
    matrix = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            matrix.append(list(map(int, line.split())))
    return matrix

def can_add_or_subtract(matrix1: list, matrix2: list) -> bool:
    # Проверяем, одинаковое ли количество строк
    if len(matrix1) != len(matrix2):
        return False

    # Проверяем, одинаковое ли количество столбцов в каждой строке
    for row1, row2 in zip(matrix1, matrix2):
        if len(row1) != len(row2):
            return False

    return True


class Worker:
    def __init__(self, name):
        self.name = name
        self._salary = 0

    def take_salary(self, amount):
        self._salary += amount
        print(f"{self.name} получил зарплату {amount}. Его зарплата равна {self._salary}")

class Pupa(Worker):
    def do_work(self, filename1, filename2):
        """Суммирование матриц"""
        matrix1 = read_matrix(filename1)
        matrix2 = read_matrix(filename2)
        if not can_add_or_subtract(matrix1, matrix2):
            print(f"{self.name} не может выполнить суммирование матриц.")
            return False

        result = [[matrix1[i][j] + matrix2[i][j] for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
        print(f"{self.name} выполнил суммирование матриц. Результат:")
        for row in result:
            print(row)
        return True

class Lupa(Worker):
    def do_work(self, filename1, filename2):
        """Вычитание матриц"""
        matrix1 = read_matrix(filename1)
        matrix2 = read_matrix(filename2)
        if not can_add_or_subtract(matrix1, matrix2):
            print(f"{self.name} не может провести вычитание матриц.")
            return False

        result = [[matrix1[i][j] - matrix2[i][j] for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
        print(f"{self.name} выполняет вычитание матриц. Результат:")
        for row in result:
            print(row)
        return True

class Accountant:
    def give_salary(self, worker, amount):
        worker.take_salary(amount)


if __name__ == "__main__":
    pupa = Pupa("Пупа")
    lupa = Lupa("Лупа")
    
    accountant = Accountant()


    is_result = pupa.do_work("matrix1.txt", "matrix1.txt")
    if is_result:
        accountant.give_salary(pupa, 150)

    is_result = lupa.do_work("matrix1.txt", "matrix2.txt")
    if is_result:
        accountant.give_salary(lupa, 100)

