def gen1():
    if False:
        return 10
    yield 1
    yield 2
g = gen1()
print(g.close())
try:
    next(g)
except StopIteration:
    print('StopIteration')
g = gen1()
print(next(g))
print(g.close())
try:
    next(g)
    print('No StopIteration')
except StopIteration:
    print('StopIteration')
g = gen1()
print(list(g))
print(g.close())
try:
    next(g)
    print('No StopIteration')
except StopIteration:
    print('StopIteration')

def gen2():
    if False:
        for i in range(10):
            print('nop')
    try:
        yield 1
        yield 2
    except:
        print('raising GeneratorExit')
        raise GeneratorExit
g = gen2()
next(g)
print(g.close())

def gen3():
    if False:
        i = 10
        return i + 15
    try:
        yield 1
        yield 2
    except:
        raise ValueError
g = gen3()
next(g)
try:
    print(g.close())
except ValueError:
    print('ValueError')