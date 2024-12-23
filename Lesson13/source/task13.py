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
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(8, 16, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Flatten()
		)

	def forward(self, x):
		return self.model(x)


class CombinedModel(nn.Module):
	def __init__(self):
		super(CombinedModel, self).__init__()
		self.conv_layers = SimpleModelConvolutional()
		self.linear_layers = SimpleModelLinear()

		test_input = torch.randn(1, 3, 19, 19)
		with torch.no_grad():
			conv_output = self.conv_layers(test_input)
		self.linear_input_size = conv_output.shape[-1]

	def forward(self, x):
		x = self.conv_layers(x)
		x = x.view(-1, self.linear_input_size)
		x = self.linear_layers(x)
		return x


if __name__ == "__main__":
	linear_model = SimpleModelLinear()
	conv_model = SimpleModelConvolutional()
	combined_model = CombinedModel()

	linear_input = torch.randn(1, 256)
	linear_output = linear_model(linear_input)
	print("Linear:", linear_output)

	conv_input = torch.randn(1, 3, 19, 19)
	conv_output = conv_model(conv_input)
	print("Conv:", conv_output)

	combined_input = torch.randn(1, 3, 19, 19)
	combined_output = combined_model(combined_input)
	print("Comb:", combined_output)

