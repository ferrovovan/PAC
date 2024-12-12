import argparse
import random

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


parser = argparse.ArgumentParser(description='Bubble sort script.')
parser.add_argument('length', type=int, help='Length of the list to sort')
args = parser.parse_args()

random_list = [random.random() for _ in range(args.length)]

print("Исходный список:", random_list)

bubble_sort(random_list)

print("Отсортированный список:", random_list)
