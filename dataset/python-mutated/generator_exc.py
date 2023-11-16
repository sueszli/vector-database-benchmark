def gen():
    if False:
        return 10
    try:
        yield 1
        raise ValueError
    except ValueError:
        print('Caught')
    yield 2
for i in gen():
    print(i)

def gen2():
    if False:
        i = 10
        return i + 15
    yield 1
    raise ValueError
    yield 2
    yield 3
g = gen2()
print(next(g))
try:
    print(next(g))
except ValueError:
    print('ValueError')
try:
    print(next(g))
except StopIteration:
    print('StopIteration')

def gen3():
    if False:
        for i in range(10):
            print('nop')
    yield 1
    try:
        yield 2
    except ValueError:
        print('ValueError received')
        yield 3
    yield 4
    yield 5
g = gen3()
print(next(g))
print(next(g))
print('out of throw:', g.throw(ValueError))
print(next(g))
try:
    print('out of throw2:', g.throw(ValueError))
except ValueError:
    print('Boomerang ValueError caught')