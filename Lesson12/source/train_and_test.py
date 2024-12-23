#!/usr/bin/python3

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from functions import mse_loss, mse_derivative
from entities import Model


def train_and_evaluate(
	model: Model, train_data,
	epochs: int, lr: float, patience=0.75
):
	loss_history, accuracy_history = [], []
	best_accuracy = 0

	for epoch in range(epochs):
		epoch_loss, predictions = [], []
		for X, y in train_data:
			pred = model.forward(X)
			model.backprop(np.round(
				mse_derivative(pred, y))
			)
			model.update(lr)

			epoch_loss.append(mse_loss(pred, y)[0, 0])
			predictions.append(np.round(pred[0, 0]))

		mean_loss = np.mean(epoch_loss)
		accuracy: float = accuracy_score(y_train, predictions)
		loss_history.append(mean_loss)
		accuracy_history.append(accuracy)

		if accuracy > best_accuracy:
			best_accuracy = accuracy
			if best_accuracy >= patience:
				break

	return model, loss_history, accuracy_history

def plot_metrics(loss_history, accuracy_history):
	epochs = range(1, len(loss_history) + 1)
	plt.plot(epochs, loss_history, label="Loss")
	plt.plot(epochs, accuracy_history, label="Accuracy")
	plt.legend()
	plt.show()

if __name__ == "__main__":
	X_train = np.asarray(
		[ [0, 0], [0, 1],
		  [1, 0], [1, 1] ],
		dtype=np.float64
	)
	y_train = np.asarray(
		[ 0, 1,
		  1, 0 ],
		dtype=np.float64
	)

	lr = 0.1 # learning rate
	epochs = 1000
	model = Model([2, 2, 1])
	print("start training and evaluating")
	model, loss, accuracy = train_and_evaluate(
		model, zip(X_train, y_train),
		epochs=epochs, lr=lr, patience=0.8
	)
	print("Final Weights and Biases:")
	for layer in model.layers:
		for neuron in layer:
			print("neuron", neuron)
			print("weight: ", neuron.weights)
			print("bias: ", neuron.bias)

	print("predictions:")
	my_pred = []
	for x in X_train:
		pred = model.forward(x)[0][0]
		my_pred.append(pred)
		print(pred)

	my_pred = np.round(np.asarray(my_pred))
	print("accuracy_score:", accuracy_score(y_train, my_pred))

	plot_metrics(loss, accuracy)

