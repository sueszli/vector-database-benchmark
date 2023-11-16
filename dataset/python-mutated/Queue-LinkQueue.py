class Node:

    def __init__(self, value):
        if False:
            return 10
        self.value = value
        self.next = None

class Queue:

    def __init__(self):
        if False:
            while True:
                i = 10
        head = Node(0)
        self.front = head
        self.rear = head

    def is_empty(self):
        if False:
            while True:
                i = 10
        return self.front == self.rear

    def enqueue(self, value):
        if False:
            i = 10
            return i + 15
        node = Node(value)
        self.rear.next = node
        self.rear = node

    def dequeue(self):
        if False:
            print('Hello World!')
        if self.is_empty():
            raise Exception('Queue is empty')
        else:
            node = self.front.next
            self.front.next = node.next
            if self.rear == node:
                self.rear = self.front
            value = node.value
            del node
            return value

    def front_value(self):
        if False:
            i = 10
            return i + 15
        if self.is_empty():
            raise Exception('Queue is empty')
        else:
            return self.front.next.value

    def rear_value(self):
        if False:
            while True:
                i = 10
        if self.is_empty():
            raise Exception('Queue is empty')
        else:
            return self.rear.value
queue = Queue()
queue.enqueue(1)
print(queue.front_value())
print(queue.rear_value())
queue.dequeue()
queue.enqueue(2)
queue.enqueue(3)
queue.dequeue()
print(queue.front_value())
print(queue.rear_value())
queue.dequeue()
print(queue.front_value())
print(queue.rear_value())