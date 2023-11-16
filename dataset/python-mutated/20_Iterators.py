class Fibonacci:

    def __init__(self, limit):
        if False:
            i = 10
            return i + 15
        self.previous = 0
        self.current = 1
        self.n = 1
        self.limit = limit

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __next__(self):
        if False:
            return 10
        if self.n < self.limit:
            result = self.previous + self.current
            self.previous = self.current
            self.current = result
            self.n += 1
            return result
        else:
            raise StopIteration
fib_iterator = iter(Fibonacci(5))
while True:
    try:
        print(next(fib_iterator))
    except StopIteration:
        break