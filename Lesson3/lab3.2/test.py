from queue_stack import *


def test_stack():
    stack = Stack() # Создаем экземпляр стека

    assert len(stack) == 0, "Стек должен быть пустым"

    stack.push(10)
    stack.push(20)
    stack.push(30)

    assert len(stack) == 3, "Стек должен содержать 3 элемента"

    assert stack.top() == 30, "Верхний элемент стека должен быть 30"

    assert stack.pop() == 30, "Должен быть удален элемент 30"
    assert len(stack) == 2, "Стек должен содержать 2 элемента после удаления"

    assert stack.top() == 20, "Теперь верхний элемент стека должен быть 20"

    assert stack.pop() == 20, "Должен быть удален элемент 20"
    assert stack.pop() == 10, "Должен быть удален элемент 10"

    assert len(stack) == 0, "Стек должен быть пустым после удаления всех элементов"

    try:
        stack.pop()
    except IndexError:
        print("Ошибка: Стек пуст, как и ожидалось.")

    print("Все проверки стека пройдены успешно!")

def test_queue():
    queue = Queue()

    assert len(queue) == 0, "Очередь должна быть пустой"

    queue.enqueue(1)
    queue.enqueue(2)
    queue.enqueue(3)

    assert len(queue) == 3, "Очередь должна содержать 3 элемента"

    assert queue.dequeue() == 1, "Первым элементом должна быть 1"
    assert len(queue) == 2, "Очередь должна содержать 2 элемента после удаления"

    assert queue.dequeue() == 2, "Теперь первым элементом должна быть 2"

    assert len(queue) == 1, "Очередь должна содержать 1 элемент"

    assert queue.dequeue() == 3, "Теперь первым элементом должна быть 3"

    assert len(queue) == 0, "Очередь должна быть пустой после удаления всех элементов"

    try:
        queue.dequeue()
    except IndexError:
        print("Ошибка: Очередь пуста, как и ожидалось.")

    print("Все проверки очереди пройдены успешно!")

def test_container():
    container = Container()

    container.add_to_queue(5)
    container.add_to_queue(10)
    container.add_to_queue(15)

    assert len(container.get_queue_list()) == 3, "Очередь должна содержать 3 элемента"

    assert container.get_from_queue() == 5, "Первым элементом должна быть 5"

    assert len(container.get_queue_list()) == 2, "Очередь должна содержать 2 элемента после удаления"

    queue_copy = container.get_queue_list()
    assert queue_copy == [10, 15], "Копия очереди должна содержать [10, 15]"

    queue_copy.append(20)
    assert queue_copy == [10, 15, 20], "Измененная копия должна содержать [10, 15, 20]"

    assert container.get_queue_list() == [10, 15], "Оригинальная очередь должна оставаться без изменений"

    print("Все проверки контейнера пройдены успешно!")

if __name__ == "__main__":
    test_stack()
    print()
    test_queue()
    print()
    test_container()
