#!/usr/bin/python3

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

def im2col(image, kernel_size=1, stride=1, padding=0):
	N, C, H, W = image.shape

	padded = F.pad(image, [padding] * 4, mode='constant', value=0)

	_, _, padded_H, padded_W = padded.shape

	output_H = (padded_H - kernel_size) // stride + 1
	output_W = (padded_W - kernel_size) // stride + 1

	cols = torch.zeros((N, C * kernel_size * kernel_size, output_H * output_W), device=image.device)
	for i in range(output_H):
		for j in range(output_W):
			patch = padded[:, :, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size]
			cols[:, :, i * output_W + j] = patch.reshape(N, -1)

	return cols

def conv2d_no_loops(image, kernel, stride=1, padding=0):
	N, C, H, W = image.shape
	K, _, KH, KW = kernel.shape

	cols = im2col(image, kernel_size=KH, stride=stride, padding=padding)
	kernel_reshaped = kernel.reshape(K, -1)

	output_H = (H + 2 * padding - KH) // stride + 1
	output_W = (W + 2 * padding - KW) // stride + 1

	output = kernel_reshaped @ cols
	output = output.reshape(K, output_H, output_W, N).permute(3, 0, 1, 2)

	return output

if __name__ == "__main__":
	batch_size = 1
	channels = 3
	height = 28
	width = 28
	out_channels = 32
	k_size = 3

	input_data = torch.randn(batch_size, channels, height, width)
	for c in range(channels):
		for i in range(height):
			for j in range(width):
				input_data[0, c, i, j] = i + j  + 1
	kernel = torch.randn(out_channels, channels, k_size, k_size)

	no_loop_conv = conv2d_no_loops(input_data, kernel=kernel, padding=1)

	conv_layer = nn.Conv2d(
		in_channels=channels, out_channels=out_channels,
		kernel_size=k_size, stride=1, padding=1, bias=False
	)
	conv_layer.weight.data = kernel
	torch_conv = conv_layer(input_data)

	print(no_loop_conv.mean(), torch_conv.mean())
	print(no_loop_conv.max(), torch_conv.max())
	print(no_loop_conv.min(), torch_conv.min())
	print(torch.mean(torch.abs(no_loop_conv - torch_conv)))

	plt.subplot(1, 2, 1)
	plt.imshow(no_loop_conv[0, 0, :, :].detach().numpy())
	plt.subplot(1, 2, 2)
	plt.imshow(torch_conv[0, 0, :, :].detach().numpy())

	plt.show()
