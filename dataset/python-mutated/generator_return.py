def gen():
    if False:
        for i in range(10):
            print('nop')
    yield 1
    return 42
g = gen()
print(next(g))
try:
    print(next(g))
except StopIteration as e:
    print(type(e), e.args)
try:
    print(next(g))
except StopIteration as e:
    print(type(e), e.args)