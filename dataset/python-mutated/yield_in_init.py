def a():
    if False:
        while True:
            i = 10
    yield

def __init__():
    if False:
        print('Hello World!')
    yield

class A:

    def __init__(self):
        if False:
            print('Hello World!')
        yield

class B:

    def __init__(self):
        if False:
            return 10
        yield from self.gen()

    def gen(self):
        if False:
            print('Hello World!')
        yield 5