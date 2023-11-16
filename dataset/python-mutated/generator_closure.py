def f():
    if False:
        print('Hello World!')
    x = 1

    def g():
        if False:
            while True:
                i = 10
        yield x
        yield (x + 1)
    return g()
for i in f():
    print(i)

def f():
    if False:
        print('Hello World!')
    x = 1

    def g():
        if False:
            return 10
        return x + 1
    yield g()
    x = 2
    yield g()
for i in f():
    print(i)
generator_of_generators = (((x, y) for x in range(2)) for y in range(3))
for i in generator_of_generators:
    for j in i:
        print(j)

def genc():
    if False:
        while True:
            i = 10
    foo = 1
    repr(lambda : (yield foo))
genc()