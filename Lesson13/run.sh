#!/usr/bin/bash

echo_green() {
	local text="$1"
	echo -e "\n\033[0;32m$text\033[0m"  # Вывод текста с зелёным цветом
}

cd source
echo_green "Task"
./task13.py
echo_green "Lab"
./lab13.py
