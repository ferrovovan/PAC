#!/usr/bin/python3

import os
import codecs
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd

nltk.download('punkt_tab')

# Фильтрация текста
def filter_text(input_path, output_path, encoding='utf-8'):
	with open(input_path, "r", encoding=encoding) as inp, open(output_path, "w", encoding=encoding) as otp:
		content = inp.read().lower()
		filtered = ''.join(c for c in content if ('а' <= c and c <= 'я') or c.isspace())
		otp.write(filtered)

# Обучение Word2Vec
def train_word2vec(file_path, params):
	return Word2Vec(LineSentence(file_path), **params)

# Получение предложений фиксированной длины
def get_sentences(text, exact_sentence_len):
	sentences = sent_tokenize(text)
	result_sentences = []
	current_sentence = ""

	for sentence in sentences:
		words = word_tokenize(sentence)
		for word in words:
			if len(current_sentence) + len(word) + 1 <= exact_sentence_len:
				current_sentence += word + " "
			else:
				if len(current_sentence) == exact_sentence_len:
					result_sentences.append(current_sentence.strip())
				current_sentence = word + " "

	if current_sentence and len(current_sentence) == exact_sentence_len:
		result_sentences.append(current_sentence.strip())

	return result_sentences

def one_hot_encoding(sentence, alphabet):
	char_to_index = {char: i for i, char in enumerate(alphabet)}
	num_unique_chars = len(alphabet)
	one_hot_matrix = np.zeros((len(sentence), num_unique_chars))
	for i, char in enumerate(sentence):
		one_hot_matrix[i, char_to_index[char]] = 1
	return one_hot_matrix


class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()
		self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
		self.fc = nn.Linear(hidden_size, output_size)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x):
		out, _ = self.rnn(x)
		out = self.fc(out)
		out = self.softmax(out)
		return out


# Генерация последовательности
def generate_sequence_2(model, start_sequence, length, alphabet, device):
	generated_sequence = list(start_sequence)
	for _ in range(length):
		input_sequence = torch.tensor(
			[one_hot_encoding(generated_sequence, alphabet)], dtype=torch.float32
		)
		input_sequence = input_sequence.to(device)
		output_probs = model(input_sequence)

		# Перемещаем тензор на CPU перед использованием .numpy()
		predicted_probs = output_probs[0][-1].detach().cpu().numpy()
		predicted_probs /= np.sum(predicted_probs)
		next_char_index = np.random.choice(len(predicted_probs), p=predicted_probs)
		next_char = alphabet[next_char_index]
		generated_sequence.append(next_char)
	return generated_sequence


if __name__ == "__main__":

	# Фильтрация текста
	input_path  = "../data/text.txt"
	output_path = "../data/filtered.txt"
	filter_text(input_path, output_path)


	# Обучение Word2Vec
	params = {
		'vector_size': 100,
		'window': 5,
		'min_count': 1,
		'workers': multiprocessing.cpu_count(),
	}
	model = train_word2vec(output_path, params)
	print(model.wv.most_similar('он', topn=5))

	# Подготовка данных для RNN
	with open(output_path, "r", encoding="utf-8") as f:
		text = f.read().lower()
	exact_sentence_len = 10
	sentences = get_sentences(text, exact_sentence_len)

	alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя "
	encoded_sentences = [one_hot_encoding(sentence, alphabet) for sentence in sentences]
	encoded_array = np.array(encoded_sentences)
	encoded_array_torch = torch.tensor(encoded_array, dtype=torch.float32)

	# Настройка RNN
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model_path  = "../data/rnn.pth"
	load_model = True

	input_size = encoded_array.shape[2]
	hidden_size = 128
	output_size = input_size
	model = RNN(input_size, hidden_size, output_size)
	if load_model:
		model.load_state_dict(torch.load(model_path, weights_only=True))
	model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	encoded_array_torch = encoded_array_torch.to(device)

	if not load_model:
		# Обучение RNN
		num_epochs = 1000
		print_every = 50
		for epoch in range(num_epochs):
			outputs = model(encoded_array_torch)
			loss = criterion(
				outputs.view(-1, output_size),
				encoded_array_torch.view(-1, output_size),
			)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if (epoch + 1) % 100 == 0:
				print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

		# Сохранение модели
		torch.save(model.state_dict(), model_path)
		print("Модель сохранена.")

	# Генерация последовательности
	start_sequence = "привет как дела"
	generated_sequence: list = generate_sequence_2(model, start_sequence, 50, alphabet, device)
	print("Generated Sequence:", "".join(generated_sequence))

