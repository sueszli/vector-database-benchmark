class Fib(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        (self.a, self.b) = (0, 1)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        (self.a, self.b) = (self.b, self.a + self.b)
        if self.a > 100000:
            raise StopIteration()
        return self.a
for n in Fib():
    print(n)