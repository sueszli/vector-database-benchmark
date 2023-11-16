"""Thread-local stack."""
import threading

class ThreadLocalStack(threading.local):
    """A thread-local stack of objects for providing implicit defaults."""

    def __init__(self):
        if False:
            print('Hello World!')
        super(ThreadLocalStack, self).__init__()
        self._stack = []

    def peek(self):
        if False:
            for i in range(10):
                print('nop')
        return self._stack[-1] if self._stack else None

    def push(self, ctx):
        if False:
            while True:
                i = 10
        return self._stack.append(ctx)

    def pop(self):
        if False:
            print('Hello World!')
        self._stack.pop()