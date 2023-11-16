class Stack:

    def __init__(self, size=100):
        if False:
            print('Hello World!')
        self.stack = []
        self.size = size
        self.top = -1

    def is_empty(self):
        if False:
            while True:
                i = 10
        return self.top == -1

    def is_full(self):
        if False:
            while True:
                i = 10
        return self.top + 1 == self.size

    def push(self, value):
        if False:
            return 10
        if self.is_full():
            raise Exception('Stack is full')
        else:
            self.stack.append(value)
            self.top += 1

    def pop(self):
        if False:
            return 10
        if self.is_empty():
            raise Exception('Stack is empty')
        else:
            self.top -= 1
            self.stack.pop()

    def peek(self):
        if False:
            return 10
        if self.is_empty():
            raise Exception('Stack is empty')
        else:
            return self.stack[self.top]
S = Stack(10)
for i in range(5):
    S.push(i)
for i in range(3):
    S.pop()
print(S.peek())