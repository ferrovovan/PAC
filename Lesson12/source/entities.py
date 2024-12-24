# domain.entities.py
from functions import Sigmoid
import numpy as np

def reshape(array):
	array = np.asarray(array)
	# Убедиться, что массив array имеет хотя бы две оси (включая ось "длина батча").
	if len(array.shape) < 2:
		return np.expand_dims(array, 0)
	return array

def xavier_normal(shape: tuple[int, int], gain: float = 1.0) -> np.ndarray:
	"""
	Функция xavier_normal инициализирует веса нейронной сети случайным образом,
	используя метод Ксавье (Xavier Initialization).
	Этот метод позволяет достичь более стабильной скорости сходимости модели,
	особенно при использовании сигмоидной или гиперболической тангенс функций активации.

	shape: — кортеж из двух целых чисел, представляющих размеры матрицы весов (число входов и выходов).
	gain — это коэффициент масштабирования, который позволяет подстроить разброс весов в зависимости от функции активации. Например:
	* Для гиперболического тангенса gain=1.0 (стандартное значение).
	* Для ReLU часто используют     gain=√2 .
​.

	Эта функция генерирует веса, которые:
	1. Не слишком велики (чтобы избежать взрывного роста значений при прямом проходе).
	2. Не слишком малы (чтобы избежать затухания градиентов при обратном проходе).
	3. Обеспечивают равномерное распределение мощности сигнала между слоями.
	"""
	# Разделение размерности на количество входов и выходов.
	fan_in, fan_out = shape[0], shape[1]
	# Вычисляем среднеквадратическое отклонение (stddev) для нормального распределения.
	stddev = gain * np.sqrt(2.0 / (fan_in + fan_out))
	# Возвращаем случайные значения из нормального распределения
	return np.random.normal(0, stddev, size=shape)


class Neuron:
	def __init__(self, input_weights, bias, name: str):
		self.weights = np.asarray(input_weights)
		self.bias = np.asarray(bias)
		self.activation = Sigmoid
		self.name = name

		# State variables for forward and backpropagation
		self.X = None
		self.db = None
		self.dw = None
		self.t = None
		self.h = None

	def forward(self, X):
		# 1. Сохраняем вход X.
		self.X = reshape(X)
		# 2. Вычисляем взвешенную сумму входов с учетом весов и смещения (т.н. pre-activation).
		self.t = self.X @ self.weights + self.bias
		# 3. Применяем функцию активации, чтобы получить выход нейрона (активированный ответ).
		self.h = self.activation.func(self.t)
		# Возвращаем активированный ответ.
		return self.h

	def backprop(self, dEdh):
		"""
		Выполняет обратное распространение ошибки.
		Вход dEdh - производная ошибки по выходу нейрона.
		"""
		# 1. Вычисляем производную функции активации по pre-activation t.
		df_dt = self.activation.derivative(self.t)[0]
		# 2. Вычисляем вклад ошибки по смещению (bias).
		# self.db = dEdh * self.df(self.t)[0]
		self.db = dEdh * df_dt  # производная ошибки по bias
		# print("db:", self.db)
		# 3. Вычисляем вклад ошибки по весам.
		self.dw = self.X.T @ self.db  # матричное умножение входа на производную
		self.dw = reshape(self.dw).T
		# print("dw:", self.dw)
		# 4. Возвращаем ошибку по входу (для предыдущих нейронов в слое).
		return self.db @ self.weights.T

	def update(self, lr):
		"""
		Обновляем параметры нейрона.
		Вход lr - коэффициент обучения.
		"""
		self.bias -= lr * self.db
		self.weights -= lr * self.dw

	def __repr__(self):
		return self.name


class Model:
	def __init__(self, architecture: list[int]):
		self.layers: list[list[Neuron]] = []
		for i in range(1, len(architecture)):
			neurons = []
			weights = xavier_normal( (architecture[i - 1], architecture[i]) )
			biases = xavier_normal(  (architecture[i], 1) )
			for j in range(architecture[i]):
				neurons.append(
					Neuron(reshape(weights[:, j]).T, reshape(biases[j]), f"{i}-{j}")
				)
			self.layers.append(neurons)

	def forward(self, X):
		output = np.asarray(X)
		for layer in self.layers:
			output = np.asarray([neuron.forward(output)[0] for neuron in layer]).T
		return output

	def backprop(self, dEdh):
		output_dx = np.asarray(dEdh)
		for layer in reversed(self.layers):
			new_dx = np.asarray(
				[neuron.backprop(dx_row) for neuron, dx_row in zip(layer, output_dx)]
			)
			# print(new_dx)
			output_dx = np.sum(new_dx, 0)
			# print(output_dx)

	def update(self, lr):
		for layer in self.layers:
			for neuron in layer:
				neuron.update(lr)
