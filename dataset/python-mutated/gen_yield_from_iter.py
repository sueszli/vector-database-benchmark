def gen():
    if False:
        return 10
    yield from (1, 2, 3)

def gen2():
    if False:
        print('Hello World!')
    yield from gen()

def gen3():
    if False:
        print('Hello World!')
    yield from (4, 5)
    yield 6
print(list(gen()))
print(list(gen2()))
print(list(gen3()))