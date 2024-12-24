#!/usr/bin/bash

echo_green() {
	local text="$1"
	echo -e "\n\033[0;32m$text\033[0m"  # Вывод текста с зелёным цветом
}

cd source
echo_green "Task"
./source/task16.py
echo_green "Lab"
./source/lab16.py
