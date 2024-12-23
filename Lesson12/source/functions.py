import numpy as np

# loss_functions
def mse_loss(predicted, actual):
	predicted, actual = np.asarray(predicted), np.asarray(actual)
	return (predicted - actual) ** 2

def mse_derivative(predicted, actual):
	predicted, actual = np.asarray(predicted), np.asarray(actual)
	return 2 * (predicted - actual)


# activation_functions

class Sigmoid:
	@staticmethod
	def func(x):
		return 1 / (1 + np.exp(-x))

	@staticmethod
	def derivative(x): # производная
		sigmoid_x = Sigmoid.func(x)
		return sigmoid_x * (1 - sigmoid_x)

