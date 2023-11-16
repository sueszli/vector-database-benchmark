def myfunction(x, y=2):
    if False:
        for i in range(10):
            print('nop')
    a = x - y
    return a + x * y

def _helper(a):
    if False:
        print('Hello World!')
    return a + 1

class A:

    def __init__(self, b=0):
        if False:
            print('Hello World!')
        self.a = 3
        self.b = b

    def foo(self, x):
        if False:
            i = 10
            return i + 15
        print(x + _helper(1.0))