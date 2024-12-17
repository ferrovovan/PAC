import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Visualizer_with_tsne:
	"""Класс визуализаций с помощью алгоритма t-SNE"""
	def __init__(self, model):
		self.model: Perceptron = model

	@classmethod
	def visualize_with_tsne(cls, data):
		images = np.array([x[0] for x in data])
		labels = np.array([np.argmax(x[1]) for x in data])

		cls.plot_figure(images, labels, 'Samples from Training Data')

	def visualize_logits_with_tsne(self, data: list[np.array, np.array]):
		"""Визуализирует логиты (результаты) модели с помощью t-SNE."""

		# Заполняем словарь: по 30 изображений для каждой цифры
		filtered_data: list[list[np.ndarray]] = [[] for _ in range(10)]
		for image, label in data:
			digit = np.argmax(label)  # Определяем цифру по метке
			if len(filtered_data[digit]) < 30:  # Проверяем лимит на 30 изображений
				filtered_data[digit].append(image)

		# Объединяем собранные изображения и метки
		images = []
		labels = []
		for digit, images_list in enumerate(filtered_data):
			images.extend(images_list)                 # Добавляем изображения в общий список
			labels.extend([digit] * len(images_list))  # Добавляем соответствующие метки

		# Преобразуем списки в numpy массивы
		images = np.array(images)
		labels = np.array(labels)

		# Получаем логиты для всех изображений
		logits = []
		for image in images:
			prediction: np.ndarray[float] = self.model.predict(image)
			noisy_prediction = prediction + np.where(prediction == 1, np.random.uniform(-0.001, 0.001), 0)
			logits.append(noisy_prediction)
		logits = np.array(logits)

		self.plot_figure(logits, labels, 'Results on Training Data')

	@staticmethod
	def plot_figure(data, labels, title: str):
		# Применяем t-SNE
		tsne = TSNE(n_components=2, random_state=42)
		transformed = tsne.fit_transform(data)

		# Визуализация
		plt.figure(figsize=(10, 8))
		scatter = plt.scatter(transformed[:, 0], transformed[:, 1], c=labels, cmap='tab10')
		plt.title(title)
		plt.colorbar(scatter)
		plt.tight_layout()
		plt.show()

