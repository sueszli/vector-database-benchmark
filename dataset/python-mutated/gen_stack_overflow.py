def gen():
    if False:
        return 10
    yield from gen()
try:
    print(list(gen()))
except RuntimeError:
    print('RuntimeError')