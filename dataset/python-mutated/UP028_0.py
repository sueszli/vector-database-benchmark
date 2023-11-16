def f():
    if False:
        return 10
    for x in y:
        yield x

def g():
    if False:
        print('Hello World!')
    for (x, y) in z:
        yield (x, y)

def h():
    if False:
        i = 10
        return i + 15
    for x in [1, 2, 3]:
        yield x

def i():
    if False:
        while True:
            i = 10
    for x in {x for x in y}:
        yield x

def j():
    if False:
        i = 10
        return i + 15
    for x in (1, 2, 3):
        yield x

def k():
    if False:
        i = 10
        return i + 15
    for (x, y) in {3: 'x', 6: 'y'}:
        yield (x, y)

def f():
    if False:
        while True:
            i = 10
    for (x, y) in {3: 'x', 6: 'y'}:
        yield (x, y)

def f():
    if False:
        for i in range(10):
            print('nop')
    for (x, y) in [{3: (3, [44, 'long ss']), 6: 'y'}]:
        yield (x, y)

def f():
    if False:
        while True:
            i = 10
    for (x, y) in z():
        yield (x, y)

def f():
    if False:
        for i in range(10):
            print('nop')

    def func():
        if False:
            while True:
                i = 10
        for (x, y) in z():
            yield (x, y)

def g():
    if False:
        for i in range(10):
            print('nop')
    print(3)

def f():
    if False:
        while True:
            i = 10
    for x in y:
        yield x
    for z in x:
        yield z

def f():
    if False:
        while True:
            i = 10
    for (x, y) in z():
        yield (x, y)
    x = 1

def _serve_method(fn):
    if False:
        i = 10
        return i + 15
    for h in TaggedText.from_file(args.input).markup(highlight=args.region):
        yield h