#!/usr/bin/python3

from Model import Perceptron
from Metrics import Evaluator, ReportGenerator
from Mnist_loader import MNISTLoader
from sklearn.metrics import precision_score, recall_score


import os
import pickle


def echo_green(text: str):
	print(f"\033[0;32m{text}\033[0m")

def save_processed_data(processed_data, filename):
    """Сохраняет обработанные данные в файл с использованием pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(processed_data, f)

def load_processed_data(filename):
    """Загружает обработанные данные из файла с использованием pickle."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
	load_data = 1
	data_dir = "./processed_data"
    
	if load_data:
		echo_green("Загрузка обработанных данных...")
		train_data = load_processed_data(f"{data_dir}/train_data.pkl")
		test_data  = load_processed_data(f"{data_dir}/test_data.pkl")
		
	else:
		echo_green("Загрузка данных...")
		loader = MNISTLoader(root=".")
		train_data = loader.load_data(train=True)
		test_data  = loader.load_data(train=False)
        
		# Сохранение обработанных данных после загрузки
		save_processed_data(train_data, f"{data_dir}/train_data.pkl")
		save_processed_data(test_data, f"{data_dir}/test_data.pkl")

	echo_green("Создание модели и тренировка...")
	model = Perceptron(train_data=train_data, load_trained=True, trained_path="./trained_data")
	# model = Perceptron(train_data=train_data, load_trained=False, trained_path="./trained_data")
	if 0:
		echo_green("Сохрание модели...")
		model.save_trained_data("./trained_data")
	
	
	

	
	evaluator = Evaluator(model=model)
	
	echo_green("Визуализация данных с помощью t-SNE (необработанные изображения)...")
	evaluator.visualize_with_tsne(train_data[:100])

	echo_green("Визуализация данных с помощью t-SNE (предсказания модели)...")
	evaluator.visualize_predictions_with_tsne(test_data[:100])
	
	echo_green("Считаем метрики...")
	train_metrics = evaluator.evaluate(train_data)
	test_metrics = evaluator.evaluate(test_data)

	echo_green("Печать статистики...")
	report = ReportGenerator()
	report.show_metrics(train_metrics, "Train Dataset")
	report.show_metrics(test_metrics, "Test Dataset")


if __name__ == "__main__":
	main()


