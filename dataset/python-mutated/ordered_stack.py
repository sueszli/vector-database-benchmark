class OrderedStack:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.items = []

    def is_empty(self):
        if False:
            while True:
                i = 10
        return self.items == []

    def push_t(self, item):
        if False:
            i = 10
            return i + 15
        self.items.append(item)

    def push(self, item):
        if False:
            i = 10
            return i + 15
        temp_stack = OrderedStack()
        if self.is_empty() or item > self.peek():
            self.push_t(item)
        else:
            while item < self.peek() and (not self.is_empty()):
                temp_stack.push_t(self.pop())
            self.push_t(item)
            while not temp_stack.is_empty():
                self.push_t(temp_stack.pop())

    def pop(self):
        if False:
            while True:
                i = 10
        if self.is_empty():
            raise IndexError('Stack is empty')
        return self.items.pop()

    def peek(self):
        if False:
            i = 10
            return i + 15
        return self.items[len(self.items) - 1]

    def size(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.items)