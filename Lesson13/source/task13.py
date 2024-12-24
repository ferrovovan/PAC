#!/usr/bin/python3

import torch
import torch.nn as nn

class SimpleModelLinear(nn.Module):
	def __init__(self):
		super(SimpleModelLinear, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(256, 64),
			nn.ReLU(),
			nn.Linear(64, 16),
			nn.Tanh(),
			nn.Linear(16, 4),
			nn.Softmax(dim=1)
		)

	def forward(self, x):
		return self.model(x)


class SimpleModelConvolutional(nn.Module):
	def __init__(self):
		super(SimpleModelConvolutional, self).__init__()
		self.model = nn.Sequential(
			nn.Conv2d(3, 8, kernel_size=1, padding=0),
			nn.ReLU(), # отрицательные значения зануляются
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(8, 16, kernel_size=3, padding=1),
			nn.ReLU(), # отрицательные значения зануляются
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Flatten() # Преобразует многомерный вход (например, 8×8×16) в одномерный вектор (1024).
		)

	def forward(self, x):
		return self.model(x)


class CombinedModel(nn.Module):
	def __init__(self):
		super(CombinedModel, self).__init__()
		self.linear_layers = SimpleModelLinear()
		self.conv_layers = SimpleModelConvolutional()

		test_input = torch.randn(1, 3, 19, 19)
		with torch.no_grad():  # отключает автоматическое вычисление градиентов. Не обновляет параметры сети.
			conv_output = self.conv_layers(test_input)
		# используется позже для определения входного размера полносвязного (nn.Linear) слоя.
		self.linear_input_size = conv_output.shape[-1]  # размер последней оси — ширина тензора

	def forward(self, x):
		# Свёрточные слои извлекают признаки из данных, формируя многомерный выход.
		x = self.conv_layers(x)
		# Этот шаг подготавливает данные для полносвязных слоев, которые ожидают вход в виде векторов.
		x = x.view(-1, self.linear_input_size)  # Меняется форма тензора с многомерного  на двумерный.
		# Это преобразование приводит признаки к конечным выходным значениям, например, вероятностям классов для задачи классификации.
		x = self.linear_layers(x)
		return x


def linear_test():
	"""
	nn.Sequential от nn.Module в данном случае не отличаются
	"""
	linear_model = SimpleModelLinear()
	linear_input = torch.randn(1, 256)
	linear_output = linear_model(linear_input)
	print("Linear prediction:", linear_output)

def convolution_test():
	conv_model = SimpleModelConvolutional()

	# 1 — размер мини-батча (число изображений, обрабатываемых за раз). Здесь это одиночное изображение.
	# 3 — число каналов, например:
	  # В цветном изображении: RGB-каналы (красный, зелёный, синий).
	  # В других случаях: могут быть другие данные, разделённые по каналам (например, тепло или глубина).
	# 19 x 19 — пространственные размеры (высота × ширина).
	conv_input = torch.randn(1, 3, 19, 19)
	conv_output = conv_model(conv_input)
	print("Convolution prediction:", conv_output)

def combined_test():
	combined_model = CombinedModel()
	combined_input = torch.randn(1, 3, 19, 19)
	combined_output = combined_model(combined_input)
	print("Combinated prediction:", combined_output)

if __name__ == "__main__":
	linear_test()
	convolution_test()
	combined_test()

