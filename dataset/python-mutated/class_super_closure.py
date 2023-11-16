class A:

    def __init__(self):
        if False:
            return 10
        self.val = 4

    def foo(self):
        if False:
            i = 10
            return i + 15
        return list(range(self.val))

class B(A):

    def foo(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.bar(i) for i in super().foo()]

    def bar(self, x):
        if False:
            for i in range(10):
                print('nop')
        return 2 * x
print(A().foo())
print(B().foo())