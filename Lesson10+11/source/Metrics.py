import numpy as np
from Model import Perceptron

class Evaluator:
	"""Класс для оценки модели."""
	def __init__(self, model):
		self.model: Perceptron = model

	def evaluate(self, data) -> dict:
		"""Оценивает модель и возвращает словарь с метриками."""
		metrics = {
			"tp": np.zeros(10, dtype=int),
			"tn": np.zeros(10, dtype=int),
			"fp": np.zeros(10, dtype=int),
			"fn": np.zeros(10, dtype=int),
		}

		for image, label in data:
			real_digit = np.argmax(label)  # Истинная цифра
			predictions = self.model.predict(image)  # Вектор логитов, полученный от модели

			for digit in range(10):
				predicted_positive = predictions[digit] == 1  # Прогноз модели на эту цифру
				actual_positive = (real_digit == digit)       # Истинная метка

				if predicted_positive:
					if actual_positive:
						metrics["tp"][digit] += 1
					else:
						metrics["fp"][digit] += 1
				else:
					if actual_positive:
						metrics["fn"][digit] += 1
					else:
						metrics["tn"][digit] += 1

		tp = metrics["tp"]
		fp = metrics["fp"]
		fn = metrics["fn"]
		tn = metrics["tn"]

		precision = Evaluator.calculate_precision(tp, fp)
		recall = Evaluator.calculate_recall(tp, fn)
		accuracy = Evaluator.calculate_accuracy(tp, tn, fp, fn)

		return {
			"precision": precision.tolist(),
			"recall": recall.tolist(),
			"accuracy": accuracy.tolist(),
			"tp": tp.tolist(),
			"fp": fp.tolist(),
			"fn": fn.tolist(),
			"tn": tn.tolist(),
		}

	@staticmethod
	def calculate_precision(tp: np.ndarray, fp: np.ndarray) -> np.ndarray:
		return np.divide(tp, tp + fp,
				out=np.zeros_like(tp, dtype=float),
				where=(tp + fp) != 0
			)

	@staticmethod
	def calculate_recall(tp: np.ndarray, fn: np.ndarray) -> np.ndarray:
		return np.divide(tp, tp + fn,
				out=np.zeros_like(tp, dtype=float),
				where=(tp + fn) != 0
			)

	@staticmethod
	def calculate_accuracy(
		tp: np.ndarray, tn: np.ndarray,
		fp: np.ndarray, fn: np.ndarray
	) -> np.ndarray:
		return np.divide(tp + tn, tp + tn + fp + fn,
			out=np.zeros_like(tp, dtype=float),
			where=(tp + tn + fp + fn) != 0
		)


class ReportGenerator:
	"""Класс для формирования отчётов."""
	@staticmethod
	def show_metrics(metrics: dict, dataset_name: str):
		"""Формирует и выводит отчёт по метрикам."""
		precision = metrics["precision"]
		recall = metrics["recall"]
		accuracy = metrics["accuracy"]

		# precision_message = "\n".join([f"Precision for {i}: {p:.3f}" for i, p in enumerate(precision)])
		# recall_message = "\n".join([f"Recall for {i}: {r:.3f}" for i, r in enumerate(recall)])
		accuracy_message = "\n".join([f"Accuracy for {i}: {a:.3f}" for i, a in enumerate(accuracy)])
		
		total_precision = np.mean(precision)
		total_recall = np.mean(recall)
		total_accuracy = np.mean(accuracy)

		print(f"\nDataset: {dataset_name}")
		# print(precision_message)
		# print(recall_message)
		print(accuracy_message)
		print("----------")
		print(f"Total Precision: {total_precision:.3f}")
		print(f"Total Recall: {total_recall:.3f}")
		print(f"Total Accuracy: {total_accuracy:.3f}")

