class Node(object):

    def __init__(self, data):
        if False:
            while True:
                i = 10
        self.data = data
        self.next = None

class Queue(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.head = None
        self.tail = None

    def enqueue(self, data):
        if False:
            print('Hello World!')
        node = Node(data)
        if self.head is None and self.tail is None:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node

    def dequeue(self):
        if False:
            return 10
        if self.head is None and self.tail is None:
            return None
        data = self.head.data
        if self.head == self.tail:
            self.head = None
            self.tail = None
        else:
            self.head = self.head.next
        return data