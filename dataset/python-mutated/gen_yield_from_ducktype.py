class MyGen:

    def __init__(self):
        if False:
            return 10
        self.v = 0

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        self.v += 1
        if self.v > 5:
            raise StopIteration
        return self.v

def gen():
    if False:
        print('Hello World!')
    yield from MyGen()

def gen2():
    if False:
        i = 10
        return i + 15
    yield from gen()
print(list(gen()))
print(list(gen2()))

class Incrementer:

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        return self.send(None)

    def send(self, val):
        if False:
            return 10
        if val is None:
            return 'Incrementer initialized'
        return val + 1

def gen3():
    if False:
        for i in range(10):
            print('nop')
    yield from Incrementer()
g = gen3()
print(next(g))
print(g.send(5))
print(g.send(100))

class MyIter:

    def __iter__(self):
        if False:
            print('Hello World!')
        return self

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        raise StopIteration(42)

def gen4():
    if False:
        for i in range(10):
            print('nop')
    global ret
    ret = (yield from MyIter())
    1 // 0
ret = None
try:
    print(list(gen4()))
except ZeroDivisionError:
    print('ZeroDivisionError')
print(ret)