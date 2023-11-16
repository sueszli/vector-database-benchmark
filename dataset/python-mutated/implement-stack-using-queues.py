import collections

class Queue(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.data = collections.deque()

    def push(self, x):
        if False:
            while True:
                i = 10
        self.data.append(x)

    def peek(self):
        if False:
            return 10
        return self.data[0]

    def pop(self):
        if False:
            return 10
        return self.data.popleft()

    def size(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.data)

    def empty(self):
        if False:
            i = 10
            return i + 15
        return len(self.data) == 0

class Stack(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.q_ = Queue()

    def push(self, x):
        if False:
            print('Hello World!')
        self.q_.push(x)
        for _ in xrange(self.q_.size() - 1):
            self.q_.push(self.q_.pop())

    def pop(self):
        if False:
            i = 10
            return i + 15
        self.q_.pop()

    def top(self):
        if False:
            return 10
        return self.q_.peek()

    def empty(self):
        if False:
            return 10
        return self.q_.empty()

class Stack2(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.q_ = Queue()
        self.top_ = None

    def push(self, x):
        if False:
            while True:
                i = 10
        self.q_.push(x)
        self.top_ = x

    def pop(self):
        if False:
            while True:
                i = 10
        for _ in xrange(self.q_.size() - 1):
            self.top_ = self.q_.pop()
            self.q_.push(self.top_)
        self.q_.pop()

    def top(self):
        if False:
            while True:
                i = 10
        return self.top_

    def empty(self):
        if False:
            for i in range(10):
                print('nop')
        return self.q_.empty()