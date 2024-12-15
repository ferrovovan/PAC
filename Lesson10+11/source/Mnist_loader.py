import torchvision
import numpy as np
from torchvision.datasets import MNIST


class MNISTLoader:
	"""Класс для выдачи датасета MNIST."""
	def __init__(self, root: str):
		self.root = root

	def load_data(self, train: bool):
		dataset = torchvision.datasets.MNIST(
			root=self.root,
			train=train,
			transform=torchvision.transforms.ToTensor(),
			download=False                                      
		)
		
		return self._process_data(dataset)

	@staticmethod
	def _process_data(data: MNIST) -> list[np.array, np.array]:
		# np.eye(10) создает матрицу размером 10x10, где по диагонали стоят единицы.
		labels = np.eye(10)[[x[1] for x in data]].astype(float)

		features = np.array([np.reshape(x[0][0].numpy(), (784,)) for x in data])
		return list(zip(features, labels))
