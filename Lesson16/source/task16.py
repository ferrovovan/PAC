#!/usr/bin/python3

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# Загрузка данных
df = pd.read_csv('./data/production.csv')
df.head()

# Подготовка данных по добыче
resources = df.groupby('API')[['Liquid', "Gas", "Water"]].apply(lambda df_: df_.reset_index(drop=True))
print(resources.head())

# Нормализация данных
resources["Liquid"] /= resources["Liquid"].max()
resources["Gas"] /= resources["Gas"].max()
resources["Water"] /= resources["Water"].max()
print(resources.head())

# Преобразование данных в массив
data = resources.to_numpy()
data = data.reshape((50, 24, 3))
print(data.shape)

# Разделение на обучающую и тестовую выборки
train = data[:40]
test = data[40:]
print(train.shape, test.shape)

# Формирование выборок X и Y
x_data = [train[:, i:i+12] for i in range(11)]
y_data = [train[:, i+1:i+13] for i in range(11)]

# Объединение выборок
x_data = np.concatenate(x_data, axis=0)
y_data = np.concatenate(y_data, axis=0)
print(x_data.shape, y_data.shape)

# Перевод данных в тензоры
tensor_x = torch.Tensor(x_data)  # Преобразуем в тензор
tensor_y = torch.Tensor(y_data)

# Создание Dataset и DataLoader
oil_dataset = TensorDataset(tensor_x, tensor_y)
oil_dataloader = DataLoader(oil_dataset, batch_size=16)

# Проверка DataLoader
for x_t, y_t in oil_dataloader:
	break
print(x_t.shape, y_t.shape)


# Определение модели
class OilModel(nn.Module):
	def __init__(self, timesteps=12, units=32):
		super(OilModel, self).__init__()
		self.lstm1 = nn.LSTM(3, units, 2, batch_first=True)
		self.dense = nn.Linear(units, 3)
		self.relu = nn.ReLU()

	def forward(self, x):
		h, _ = self.lstm1(x)
		outs = []
		for i in range(h.shape[0]):
			outs.append(self.relu(self.dense(h[i])))
		out = torch.stack(outs, dim=0)
		return out


# Инициализация модели и оптимизатора
model = OilModel()
opt = optim.Adam(model.parameters())
criterion = nn.MSELoss()

# Обучение
NUM_EPOCHS = 500

for epoch in range(NUM_EPOCHS):
	running_loss = 0.0
	num = 0
	for x_t, y_t in oil_dataloader:
		opt.zero_grad()  # обнуляем градиенты

		# Прямой проход, расчет ошибки и оптимизация
		outputs = model(x_t)
		loss = criterion(outputs, y_t)
		loss.backward()
		opt.step()

		# Собираем статистику по потерям
		running_loss += loss.item()
		num += 1

	if (epoch + 1) % 10 == 0:
		print(f'[Epoch: {epoch + 1:2d}] loss: {running_loss:.3f}')
		print(outputs.shape)

print('Finished Training')

# Тестирование модели
x_tst = test[:, :12]
predicts = np.zeros((x_tst.shape[0], 0, x_tst.shape[2]))
print(x_tst.shape)

for i in range(12):
	x = np.concatenate((x_tst[:, i:], predicts), axis=1)
	x_t = torch.from_numpy(x).float()
	print(x_t.shape)
	pred = model(x_t).detach().numpy()
	last_pred = pred[:, -1:]  # Нас интересует только последний месяц
	predicts = np.concatenate((predicts, last_pred), axis=1)

# Визуализация результатов
plt.figure(figsize=(10, 6))
for iapi in range(4):
	plt.subplot(2, 2, iapi + 1)
	plt.plot(np.arange(x_tst.shape[1]), x_tst[iapi, :, 0], label='Actual')
	plt.plot(np.arange(predicts.shape[1]) + x_tst.shape[1], predicts[iapi, :, 0], label='Prediction')
	plt.legend()

plt.show()

