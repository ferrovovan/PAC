#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 25, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


def train(net, trainloader, criterion, optimizer, device, epochs=5, print_every=2000):
	net = net.to(device)
	for epoch in range(epochs):
		running_loss = 0.0
		correct, total = 0, 0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)

			# Обнуление градиентов
			optimizer.zero_grad()

			# Прямой проход, расчёт потерь, обратное распространение
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# Обновление статистики
			running_loss += loss.item()
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

			# Вывод промежуточных результатов
			if (i + 1) % print_every == 0:
				print(f'[Epoch {epoch + 1}, Batch {i + 1}] '
					f'Loss: {running_loss / print_every:.3f} '
					f'Accuracy: {100 * correct / total:.2f}%'
				)
				running_loss = 0.0
	print('Finished Training')


def evaluate_accuracy(net, testloader, device):
	"""
	Оценивает точность модели на тестовом наборе данных.

	:param net: torch.nn.Module, обученная модель.
	:param testloader: DataLoader, тестовый набор данных.
	:param device: str, устройство ('cpu' или 'cuda').
	:return: float, точность модели в процентах.
	"""
	net.eval()  # Перевод модели в режим оценки
	correct, total = 0, 0

	with torch.no_grad():
		for data in testloader:
			images, labels = data
			images, labels = images.to(device), labels.to(device)

			outputs = net(images)
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	accuracy = 100 * correct / total
	return accuracy


if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

	testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

	net = Net()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	train(net, trainloader, criterion, optimizer, device)

	accuracy = evaluate_accuracy(net, testloader, device)
	print(f'Accuracy: {accuracy:.2f}%')

