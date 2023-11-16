def f():
    if False:
        print('Hello World!')
    yield 1
    yield from g
g = f()
print(next(g))
try:
    next(g)
except ValueError:
    print('ValueError')