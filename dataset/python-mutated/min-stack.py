class MinStack(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.min = None
        self.stack = []

    def push(self, x):
        if False:
            i = 10
            return i + 15
        if not self.stack:
            self.stack.append(0)
            self.min = x
        else:
            self.stack.append(x - self.min)
            if x < self.min:
                self.min = x

    def pop(self):
        if False:
            print('Hello World!')
        x = self.stack.pop()
        if x < 0:
            self.min = self.min - x

    def top(self):
        if False:
            while True:
                i = 10
        x = self.stack[-1]
        if x > 0:
            return x + self.min
        else:
            return self.min

    def getMin(self):
        if False:
            return 10
        return self.min

class MinStack2(object):

    def __init__(self):
        if False:
            print('Hello World!')
        (self.stack, self.minStack) = ([], [])

    def push(self, x):
        if False:
            i = 10
            return i + 15
        self.stack.append(x)
        if len(self.minStack):
            if x < self.minStack[-1][0]:
                self.minStack.append([x, 1])
            elif x == self.minStack[-1][0]:
                self.minStack[-1][1] += 1
        else:
            self.minStack.append([x, 1])

    def pop(self):
        if False:
            for i in range(10):
                print('nop')
        x = self.stack.pop()
        if x == self.minStack[-1][0]:
            self.minStack[-1][1] -= 1
            if self.minStack[-1][1] == 0:
                self.minStack.pop()

    def top(self):
        if False:
            print('Hello World!')
        return self.stack[-1]

    def getMin(self):
        if False:
            print('Hello World!')
        return self.minStack[-1][0]

class MinStack3(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.stack = []

    def push(self, x):
        if False:
            print('Hello World!')
        if self.stack:
            current_min = min(x, self.stack[-1][0])
            self.stack.append((current_min, x))
        else:
            self.stack.append((x, x))

    def pop(self):
        if False:
            return 10
        return self.stack.pop()[1]

    def top(self):
        if False:
            return 10
        return self.stack[-1][1]

    def getMin(self):
        if False:
            for i in range(10):
                print('nop')
        return self.stack[-1][0]