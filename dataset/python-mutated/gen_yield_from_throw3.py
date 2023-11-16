class Iter:

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __next__(self):
        if False:
            return 10
        return 1

    def throw(self, x):
        if False:
            return 10
        print('throw', x)
        return 456

def gen():
    if False:
        return 10
    yield from Iter()
g = gen()
print(next(g))
g.close()
g = gen()
print(next(g))
print(g.throw(123))
g = gen()
print(next(g))
print(g.throw(ZeroDivisionError))

class Iter2:

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __next__(self):
        if False:
            while True:
                i = 10
        return 1

def gen2():
    if False:
        print('Hello World!')
    yield from Iter2()
g = gen2()
print(next(g))
try:
    g.throw(ValueError)
except:
    print('ValueError')
g = gen2()
print(next(g))
try:
    g.throw(123)
except TypeError:
    print('TypeError')