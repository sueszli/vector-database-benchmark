class Queue:

    def __init__(self, size=100):
        if False:
            while True:
                i = 10
        self.size = size
        self.queue = [None for _ in range(size)]
        self.front = -1
        self.rear = -1

    def is_empty(self):
        if False:
            while True:
                i = 10
        return self.front == self.rear

    def is_full(self):
        if False:
            i = 10
            return i + 15
        return self.rear + 1 == self.size

    def enqueue(self, value):
        if False:
            i = 10
            return i + 15
        if self.is_full():
            raise Exception('Queue is full')
        else:
            self.rear += 1
            self.queue[self.rear] = value

    def dequeue(self):
        if False:
            for i in range(10):
                print('nop')
        if self.is_empty():
            raise Exception('Queue is empty')
        else:
            self.queue[self.front] = None
            self.front += 1
            return self.queue[self.front]

    def front_value(self):
        if False:
            print('Hello World!')
        if self.is_empty():
            raise Exception('Queue is empty')
        else:
            return self.queue[self.front + 1]

    def rear_value(self):
        if False:
            i = 10
            return i + 15
        if self.is_empty():
            raise Exception('Queue is empty')
        else:
            return self.queue[self.rear]
queue = Queue(size=2)
queue.enqueue(1)
print(queue.front_value())
print(queue.rear_value())
queue.enqueue(2)
queue.enqueue(3)
print(queue.front_value())
print(queue.rear_value())