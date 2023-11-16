@micropython.native
def gen1(x):
    if False:
        print('Hello World!')
    yield x
    yield (x + 1)
    return x + 2
g = gen1(3)
print(next(g))
print(next(g))
try:
    next(g)
except StopIteration as e:
    print(e.args[0])

@micropython.native
def gen2(x):
    if False:
        print('Hello World!')
    yield from range(x)
print(list(gen2(3)))

@micropython.native
def gen3():
    if False:
        print('Hello World!')
    try:
        yield 1
        yield 2
    except Exception as er:
        print('caught', repr(er))
        yield 3
g = gen3()
print(next(g))
print(g.throw(ValueError(42)))

@micropython.native
def gen4():
    if False:
        print('Hello World!')
    try:
        yield 1
    except:
        print('raising GeneratorExit')
        raise GeneratorExit
g = gen4()
print(next(g))
print(g.close())