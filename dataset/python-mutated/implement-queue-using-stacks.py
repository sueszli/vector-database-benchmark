class MyQueue(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        (self.A, self.B) = ([], [])

    def push(self, x):
        if False:
            return 10
        '\n        :type x: int\n        :rtype: None\n        '
        self.A.append(x)

    def pop(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype: int\n        '
        self.peek()
        return self.B.pop()

    def peek(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :rtype: int\n        '
        if not self.B:
            while self.A:
                self.B.append(self.A.pop())
        return self.B[-1]

    def empty(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype: bool\n        '
        return not self.A and (not self.B)