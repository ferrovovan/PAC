#!/usr/bin/python3

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy.ndimage import zoom
import cv2



def get_features_map(model, layer_name):
	def hook(module, input, output):
		features_map.append(output)

	features_map = []
	hook_handle = getattr(model, layer_name).register_forward_hook(hook)

	return features_map, hook_handle


def process_frame(frame, model, transform):
	img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	img_tensor = transform(img).unsqueeze(0).to(device)

	features_maps, hook_handle = get_features_map(model, 'layer4')
	with torch.no_grad():
		outputs = model(img_tensor)
		_, predicted_class = torch.max(outputs, 1)
		predicted_class = predicted_class.item()
		weights = model.fc.weight.cpu().detach().numpy()

	features_maps_numpy = features_maps[0][0].cpu().detach().numpy()
	cam = np.zeros_like(features_maps_numpy[0, :, :])

	for i in range(features_maps_numpy.shape[0]):
		cam += weights[predicted_class, i] * features_maps_numpy[i, :, :]

	img_size = img.size
	cam = zoom(cam, (img_size[1] / cam.shape[0], img_size[0] / cam.shape[1]), order=1)

	cam = np.maximum(cam, 0)
	cam = cam / cam.max()

	cam = cv2.cvtColor((cam * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

	heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
	result = cv2.addWeighted(frame, 0.8, heatmap, 0.3, 0)

	return result, predicted_class


if __name__ == "__main__":
	device_name = "cuda" if torch.cuda.is_available() else "cpu"
	device = torch.device(device_name)
	print(f"Starting with {device_name}")

	model = models.resnet50(pretrained=True).to(device)
	model.eval()


	with open("imagenet_classes.txt", "r") as file:
		classes = list(map(lambda x: x[:-1], file.readlines()))

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("Не удалось открыть камеру")
		exit()

	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(
			# Это средние значения по всем пикселям в каждом канале (R, G, B) для всех изображений из ImageNet.
			mean=[0.485, 0.456, 0.406],
			# Это стандартное отклонение интенсивностей в каждом канале RGB для ImageNet.
			std=[0.229, 0.224, 0.225]),
	])
	while True:
		ret, frame = cap.read()
		if not ret:
			print("Не удалось получить кадр")
			break

		result, pred = process_frame(frame, model, transform)
		cv2.putText(result, f"Class: {classes[pred - 1]}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
					(255, 255, 255), 2, cv2.LINE_AA)

		cv2.imshow('CAM', result)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
