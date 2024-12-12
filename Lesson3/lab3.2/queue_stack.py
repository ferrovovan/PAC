class Queue(list):
    def enqueue(self, item):
        """Добавление элемента в конец очереди."""
        self.append(item)

    def dequeue(self):
        """Получение первого элемента из очереди."""
        if not self:
            raise IndexError("Очередь пуста")
        return self.pop(0)

    def get_list(self):
        """Возвращает копию очереди."""
        return self[:]  # Копия текущего состояния очереди


class Stack(list):
    def push(self, item):
        """Добавление элемента на верх стека."""
        self.append(item)

    def pop(self):
        """Получение верхнего элемента из стека."""
        if not self:
            raise IndexError("Стек пуст")
        return super().pop()

    def top(self):
        """Возвращает верхний элемент стека без удаления."""
        if not self:
            raise IndexError("Стек пуст")
        return self[-1]


class Container:
    def __init__(self):
        self._queue = Queue()

    def add_to_queue(self, item):
        """Добавление элемента в очередь."""
        self._queue.enqueue(item)

    def get_from_queue(self):
        """Получение первого элемента из очереди."""
        return self._queue.dequeue()

    def get_queue_list(self):
        """Получает копию очереди."""
        return self._queue.get_list()

