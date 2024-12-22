#!/bin/bash

# Проверка на количество аргументов
if [ "$#" -ne 2 ]; then
  echo "Использование: $0 <путь к уроку> <путь к исходной директории изображений>"
  exit 1
fi

# Путь к уроку и исходной директории изображений
lesson_dir="$1"
source_dir="$2"

# Находим единственный markdown файл в директории урока
md_file=$(find "$lesson_dir" -type f -name "*.md" -print -quit)

# Если markdown файл не найден
if [ -z "$md_file" ]; then
  echo "Ошибка: markdown файл (.md) не найден в директории $lesson_dir."
  exit 1
fi

# Определяем целевую директорию для изображений
target_dir="$lesson_dir/images"

# Если целевая директория не существует, создаем её
if [ ! -d "$target_dir" ]; then
  echo "Создаю директорию для изображений: $target_dir"
  mkdir -p "$target_dir"
fi

echo "Обработать файл $md_file и изображения из $source_dir, сохранять в $target_dir"



# Читаем исходный Markdown файл
while IFS= read -r line
do
  # Проверяем, если строка содержит ссылку на картинку
# Читаем строку из Markdown файла
  if echo "$line" | grep -q '<img src="images/LessonsII/' ; then
    # Извлекаем имя файла с помощью 'sed' для получения нужной части строки
    img_name=$(echo "$line" | sed -E 's/.*<img src="images\/LessonsII\/([^"]*)".*/\1/')

    # Проверяем, существует ли изображение в исходной директории
    if [ -f "$source_dir/$img_name" ]; then
      # Копируем картинку в целевую директорию
      cp "$source_dir/$img_name" "$target_dir/"

      # Заменяем строку с путём на новый
      line="${line/images\/LessonsII/images}"
    else
      echo "Изображение $img_name не найдено в $source_dir"
    fi
  fi
  # Записываем изменённую строку обратно в файл
  echo "$line"
done < "$md_file" > "${md_file}.tmp"

# Перезаписываем оригинальный Markdown файл изменёнными данными
mv "${md_file}.tmp" "$md_file"

echo "Готово! Картинки скопированы и ссылки обновлены."

