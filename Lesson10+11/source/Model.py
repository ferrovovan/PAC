import random
import numpy as np
import os
from PIL import Image


class Perceptron:
	"""Класс Перцептрона."""
	def __init__(self, train_data, load_trained: bool, trained_path: str):
		if load_trained and os.path.exists(trained_path):
			self.avg_digits = self._load_trained_data(trained_path)
		else:
			self.avg_digits = [self._compute_average_digit(train_data, digit) for digit in range(10)]
			
		self.weights = [np.transpose(avg) for avg in self.avg_digits]
		self.biases = random.choices(population=range(40, 70), k=10)

	@staticmethod
	def _compute_average_digit(data, digit):
		filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
		return np.mean(filtered_data, axis=0)

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

	def predict_digit(self, image, digit) -> bool:
		W = self.weights[digit]
		bias: int = self.biases[digit]
		
		def binary_step(x):
			return True if x >= 0 else False
		
		res = (np.dot(W, image) - bias) / np.linalg.norm(W)
		return binary_step(res)

	def predict(self, image):
		prediction = [0] * 10
		for digit in range(10):
			if self.predict_digit(image, digit):
				prediction[digit] = 1
				break
		return prediction


