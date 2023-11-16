def gen():
    if False:
        return 10
    try:
        yield 123
    except GeneratorExit:
        print('GeneratorExit')

def gen2():
    if False:
        print('Hello World!')
    try:
        yield from gen()
    except GeneratorExit:
        print('GeneratorExit outer')
    yield 789
g = gen2()
print(next(g))
print(g.throw(GeneratorExit))
g = gen2()
print(next(g))
print(g.throw(GeneratorExit()))