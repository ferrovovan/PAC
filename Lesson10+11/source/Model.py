import random
import numpy as np
import os
from PIL import Image


class DigitClassifier:
	"""Классификатор для распознавания конкретной цифры."""
	def __init__(self, avg_digit: np.ndarray, bias: int):
		"""
		Инициализирует классификатор.
		:param avg_digit: Матрица среднего изображения для цифры.
		:param bias: Смещение (bias) для классификатора.
		"""
		self.bias = bias
		self.weights = np.transpose(avg_digit)

	@staticmethod
	def binary_step(x: float) -> bool:
		return x >= 0

	def predict_digit(self, image: np.ndarray) -> bool:
		"""
		Предсказывает, является ли изображение целевой цифрой.
		:param image: Входное изображение в виде вектора.
		:return: True, если изображение распознано как целевая цифра, иначе False.
		"""
		W = self.weights
		res = (np.dot(W, image) - self.bias) / np.linalg.norm(W)
		return self.binary_step(res)


class Perceptron:
	"""Класс Перцептрона."""
	def __init__(self, train_data, load_trained: bool, trained_path: str):
		if load_trained and os.path.exists(trained_path):
			self.avg_digits = self._load_trained_data(trained_path)
		else:
			self.avg_digits = [self._compute_average_digit(train_data, digit) for digit in range(10)]

		# biases = random.choices(population=range(40, 70), k=10)
		biases = self._compute_average_biases(train_data)
		# Создаем классификаторы для каждой цифры
		self.digitClassifiers = [
			DigitClassifier(self.avg_digits[i], biases[i]) for i in range(10)
		]


	@staticmethod
	def _compute_average_digit(data, digit):
		filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
		return np.mean(filtered_data, axis=0)

	def _compute_average_biases(self, data):
		biases = []
		for num in range(10):
			W = np.transpose(self.avg_digits[num])
			filtered_data = [x[0] for x in data if np.argmax(x[1]) == num]  # Выбираем данные для текущей цифры
			avg_dot = np.mean([np.dot(W, image) for image in filtered_data])  # Средний результат скалярного произведения
			biases.append(avg_dot)
		return np.array(biases)

	def predict(self, image) -> list[int]:
		"""Получает логиты модели для входного изображения."""
		prediction = [0] * 10
		for digit in range(10):
			prediction[digit] = self.digitClassifiers[digit].predict_digit(image)
		return prediction

	def _load_trained_data(self, path):
		avg_digits = []
		for digit in range(10):
			avg_file_path = os.path.join(path, f"avg_digit_{digit}.png")
			
			if not os.path.exists(avg_file_path):
				raise FileNotFoundError(f"Trained data file for digit {digit} not found: {avg_file_path})")
			
			# Загружаем средние изображения как картинки и преобразуем обратно в массив
			avg_img = Image.open(avg_file_path).convert("L")  # Конвертируем в grayscale
			avg_array = np.array(avg_img).astype(float) / 255  # Возвращаем в диапазон [0, 1]
			avg_digits.append(avg_array.reshape(784, 1))

		return avg_digits

	def save_trained_data(self, save_path: str):
		os.makedirs(save_path, exist_ok=True)
		for digit, avg in enumerate(self.avg_digits):
			avg_file_path = os.path.join(save_path, f"avg_digit_{digit}.png")
			weight_file_path = os.path.join(save_path, f"weight_digit_{digit}.npy")
			
			# Преобразуем среднее изображение (784, 1) в (28, 28) и сохраняем как изображение
			normalized_avg = 255 * avg / np.max(avg)  # Масштабируем в диапазон [0, 255]
			normalized_img = np.reshape(normalized_avg, (28, 28)).astype(np.uint8)
			avg_img = Image.fromarray(normalized_img)
			avg_img.save(avg_file_path)


