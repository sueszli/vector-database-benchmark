class Queue:

    def __init__(self, size=100):
        if False:
            while True:
                i = 10
        self.size = size + 1
        self.queue = [None for _ in range(size + 1)]
        self.front = 0
        self.rear = 0

    def is_empty(self):
        if False:
            i = 10
            return i + 15
        return self.front == self.rear

    def is_full(self):
        if False:
            while True:
                i = 10
        return (self.rear + 1) % self.size == self.front

    def enqueue(self, value):
        if False:
            while True:
                i = 10
        if self.is_full():
            raise Exception('Queue is full')
        else:
            self.rear = (self.rear + 1) % self.size
            self.queue[self.rear] = value

    def dequeue(self):
        if False:
            while True:
                i = 10
        if self.is_empty():
            raise Exception('Queue is empty')
        else:
            self.queue[self.front] = None
            self.front = (self.front + 1) % self.size
            return self.queue[self.front]

    def front_value(self):
        if False:
            return 10
        if self.is_empty():
            raise Exception('Queue is empty')
        else:
            value = self.queue[(self.front + 1) % self.size]
            return value

    def rear_value(self):
        if False:
            while True:
                i = 10
        if self.is_empty():
            raise Exception('Queue is empty')
        else:
            value = self.queue[self.rear]
            return value
queue = Queue(size=2)
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(queue.front_value())
print(queue.rear_value())