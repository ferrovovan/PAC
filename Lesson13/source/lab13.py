#!/usr/bin/python3

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import torchvision
import numpy as np
from torchvision.datasets import MNIST


def echo_green(text: str):
	print(f"\033[0;32m{text}\033[0m")


transform = transforms.Compose([
	transforms.ToTensor()
])


class MnistModel(nn.Module):
	def __init__(self):
		super(MnistModel, self).__init__()
		# Свёрточные слои с нормализацией
		self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		# Полносвязный слой
		self.fc = nn.Linear(32 * 7 * 7, 10)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.max_pool2d(x, 2, 2)
		x = torch.flatten(x, 1)
		x = self.fc(x)
		return x

if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	echo_green("Загрузка данных...")
	train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
	test_dataset = datasets.MNIST('../data', train=False, transform=transform)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

	echo_green("Создание моделей...")
	model = MnistModel().to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	criterion = nn.CrossEntropyLoss()

	echo_green("Цикл обучения...")
	epochs = 5
	for epoch in range(epochs):
		running_loss = 0.0
		for i, data in enumerate(train_loader):
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)

			optimizer.zero_grad()

			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')


	echo_green("Вычисление метрик...")
	correct = 0
	total   = 0
	with torch.no_grad():
		for data in test_loader:
			images, labels = data
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	echo_green("Результат...")
	print(f'Accuracy: {100 * correct / total}%')
