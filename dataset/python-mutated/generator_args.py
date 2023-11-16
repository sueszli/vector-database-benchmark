def gen(v=5):
    if False:
        print('Hello World!')
    for i in range(v):
        yield i
print(list(gen()))
print(list(gen(v=10)))

def g(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    for i in args:
        yield i
    for (k, v) in kwargs.items():
        yield (k, v)
print(list(g(1, 2, 3, foo='bar')))