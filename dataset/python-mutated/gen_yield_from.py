def gen():
    if False:
        i = 10
        return i + 15
    yield 1
    yield 2
    return 3

def gen2():
    if False:
        while True:
            i = 10
    print('here1')
    print((yield from gen()))
    print('here2')
g = gen2()
print(list(g))

def gen7(x):
    if False:
        for i in range(10):
            print('nop')
    if x < 3:
        return x
    else:
        raise StopIteration(444)

def gen8():
    if False:
        while True:
            i = 10
    print((yield from map(gen7, range(100))))
g = gen8()
print(list(g))