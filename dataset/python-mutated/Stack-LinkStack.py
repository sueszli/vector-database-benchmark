class Node:

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self.value = value
        self.next = None

class Stack:

    def __init__(self):
        if False:
            print('Hello World!')
        self.top = None

    def is_empty(self):
        if False:
            for i in range(10):
                print('nop')
        return self.top == None

    def push(self, value):
        if False:
            while True:
                i = 10
        cur = Node(value)
        cur.next = self.top
        self.top = cur

    def pop(self):
        if False:
            while True:
                i = 10
        if self.is_empty():
            raise Exception('Stack is empty')
        else:
            cur = self.top
            self.top = self.top.next
            del cur

    def peek(self):
        if False:
            return 10
        if self.is_empty():
            raise Exception('Stack is empty')
        else:
            return self.top.value
stack = Stack()
for i in range(5):
    stack.push(i)
for i in range(3):
    stack.pop()
print(stack.peek())