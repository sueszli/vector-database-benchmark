"""
Stack Abstract Data Type (ADT)
Stack() creates a new stack that is empty.
   It needs no parameters and returns an empty stack.
push(item) adds a new item to the top of the stack.
   It needs the item and returns nothing.
pop() removes the top item from the stack.
   It needs no parameters and returns the item. The stack is modified.
peek() returns the top item from the stack but does not remove it.
   It needs no parameters. The stack is not modified.
is_empty() tests to see whether the stack is empty.
   It needs no parameters and returns a boolean value.
"""
from abc import ABCMeta, abstractmethod

class AbstractStack(metaclass=ABCMeta):
    """Abstract Class for Stacks."""

    def __init__(self):
        if False:
            return 10
        self._top = -1

    def __len__(self):
        if False:
            return 10
        return self._top + 1

    def __str__(self):
        if False:
            print('Hello World!')
        result = ' '.join(map(str, self))
        return 'Top-> ' + result

    def is_empty(self):
        if False:
            print('Hello World!')
        return self._top == -1

    @abstractmethod
    def __iter__(self):
        if False:
            return 10
        pass

    @abstractmethod
    def push(self, value):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def pop(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def peek(self):
        if False:
            return 10
        pass

class ArrayStack(AbstractStack):

    def __init__(self, size=10):
        if False:
            i = 10
            return i + 15
        '\n        Initialize python List with size of 10 or user given input.\n        Python List type is a dynamic array, so we have to restrict its\n        dynamic nature to make it work like a static array.\n        '
        super().__init__()
        self._array = [None] * size

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        probe = self._top
        while True:
            if probe == -1:
                return
            yield self._array[probe]
            probe -= 1

    def push(self, value):
        if False:
            i = 10
            return i + 15
        self._top += 1
        if self._top == len(self._array):
            self._expand()
        self._array[self._top] = value

    def pop(self):
        if False:
            print('Hello World!')
        if self.is_empty():
            raise IndexError('Stack is empty')
        value = self._array[self._top]
        self._top -= 1
        return value

    def peek(self):
        if False:
            for i in range(10):
                print('nop')
        'returns the current top element of the stack.'
        if self.is_empty():
            raise IndexError('Stack is empty')
        return self._array[self._top]

    def _expand(self):
        if False:
            for i in range(10):
                print('nop')
        '\n         expands size of the array.\n         Time Complexity: O(n)\n        '
        self._array += [None] * len(self._array)

class StackNode:
    """Represents a single stack node."""

    def __init__(self, value):
        if False:
            return 10
        self.value = value
        self.next = None

class LinkedListStack(AbstractStack):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.head = None

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        probe = self.head
        while True:
            if probe is None:
                return
            yield probe.value
            probe = probe.next

    def push(self, value):
        if False:
            while True:
                i = 10
        node = StackNode(value)
        node.next = self.head
        self.head = node
        self._top += 1

    def pop(self):
        if False:
            for i in range(10):
                print('nop')
        if self.is_empty():
            raise IndexError('Stack is empty')
        value = self.head.value
        self.head = self.head.next
        self._top -= 1
        return value

    def peek(self):
        if False:
            return 10
        if self.is_empty():
            raise IndexError('Stack is empty')
        return self.head.value