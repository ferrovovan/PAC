#!/usr/bin/python3

from Model import Perceptron
from Metrics import Evaluator, ReportGenerator, Visualizer_with_tsne
from Mnist_loader import MNISTLoader
from sklearn.metrics import precision_score, recall_score


import os
import pickle


def echo_green(text: str):
	print(f"\033[0;32m{text}\033[0m")

def echo_yellow(text: str):
    print(f"\033[0;33m{text}\033[0m")

def save_processed_data(processed_data, filename):
	"""Сохраняет обработанные данные в файл с использованием pickle."""
	with open(filename, 'wb') as f:
		pickle.dump(processed_data, f)

def load_processed_data(filename):
	"""Загружает обработанные данные из файла с использованием pickle."""
	with open(filename, 'rb') as f:
		return pickle.load(f)

def create_processed_data_folder(folder: str):
    """Создаёт папку './processed_data', если её ещё не существует."""
    os.makedirs(folder, exist_ok=True)


def main():
	load_data  = 0
	data_dir = "./processed_data"
	load_model = 0
	save_model = 0
	evaluate_model = 1
	visualize_with_tsne = 1

	train_data: list[np.array, np.array]
	test_data:  list[np.array, np.array]
	if load_data:
		echo_green("Загрузка обработанных данных...")
		train_data = load_processed_data(f"{data_dir}/train_data.pkl")
		test_data  = load_processed_data(f"{data_dir}/test_data.pkl")
	else:
		echo_green("Загрузка данных...")
		loader = MNISTLoader(root=".")
		train_data = loader.load_data(train=True)
		test_data  = loader.load_data(train=False)

		create_processed_data_folder(data_dir)
		save_processed_data(train_data, f"{data_dir}/train_data.pkl")
		save_processed_data(test_data, f"{data_dir}/test_data.pkl")

	echo_green("Создание модели и тренировка...")
	model = Perceptron(train_data=train_data, load_trained=load_model, trained_path="./trained_data")
	if save_model:
		echo_green("Сохрание модели...")
		create_processed_data_folder(data_dir)
		model.save_trained_data("./trained_data")

	if evaluate_model:  # Рассчитать 𝑝𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛 и 𝑟𝑒𝑐𝑎𝑙𝑙 получившейся модели на тестовом наборе
		evaluator = Evaluator(model=model)
		echo_yellow("Считаем метрики...")
		test_metrics = evaluator.evaluate(test_data)

		echo_yellow("Печать статистики...")
		report = ReportGenerator()
		report.show_metrics(test_metrics, "Test Dataset")
	
	if visualize_with_tsne:
		visualizer = Visualizer_with_tsne(model=model)
		echo_yellow("Визуализация данных с помощью t-SNE")
		echo_yellow("Необработанные изображения...")
		visualizer.visualize_with_tsne(train_data[:300])

		echo_yellow("Предсказания модели...")
		visualizer.visualize_logits_with_tsne(test_data[:300])


if __name__ == "__main__":
	main()

"""
Укажу тебе статистику счётов, и назначаемые байесы, ты скажи какой поставить.
"""
