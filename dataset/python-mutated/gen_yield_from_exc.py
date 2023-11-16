def gen():
    if False:
        while True:
            i = 10
    yield 1
    yield 2
    raise ValueError

def gen2():
    if False:
        i = 10
        return i + 15
    try:
        print((yield from gen()))
    except ValueError:
        print('caught ValueError from downstream')
g = gen2()
print(list(g))