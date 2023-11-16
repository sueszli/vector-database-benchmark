if (foo := 0):
    pass
if (foo := 1):
    pass
if (y := (5 + 5)):
    pass
y = (x := 0)
y += (x := 0)
(y := (5 + 5))
test: int = (test2 := 2)
(a, b) = (test := (1, 2))
assert (foo := (42 - 12))
foo(x=(y := f(x)))

def foo(answer=(p := 42)):
    if False:
        i = 10
        return i + 15
    ...

def foo2(answer: (p := 42)=5):
    if False:
        print('Hello World!')
    ...
lambda : (x := 1)
a[(x := 12)]
a[:(x := 13)]
f'{(x := 10)}'

def a():
    if False:
        while True:
            i = 10
    return (x := 3)
    await (b := 1)
    yield (a := 2)
    raise (c := 3)

def this_is_so_dumb() -> (please := no):
    if False:
        for i in range(10):
            print('nop')
    pass

async def await_the_walrus():
    with (x := y):
        pass
    with (x := y) as z, (a := b) as c:
        pass
    with (x := (await y)):
        pass
    with ((x := (await a)), (y := (await b))):
        pass
    with ((x := (await a)), (y := (await b))):
        pass
    with (x := (await a)), (y := (await b)):
        pass
if (foo := 0):
    pass
if (foo := 1):
    pass
if (y := (5 + 5)):
    pass
y = (x := 0)
y += (x := 0)
(y := (5 + 5))
test: int = (test2 := 2)
(a, b) = (test := (1, 2))
assert (foo := (42 - 12))
foo(x=(y := f(x)))

def foo(answer=(p := 42)):
    if False:
        for i in range(10):
            print('nop')
    ...

def foo2(answer: (p := 42)=5):
    if False:
        print('Hello World!')
    ...
lambda : (x := 1)
a[(x := 12)]
a[:(x := 13)]
f'{(x := 10)}'

def a():
    if False:
        i = 10
        return i + 15
    return (x := 3)
    await (b := 1)
    yield (a := 2)
    raise (c := 3)

def this_is_so_dumb() -> (please := no):
    if False:
        while True:
            i = 10
    pass

async def await_the_walrus():
    with (x := y):
        pass
    with (x := y) as z, (a := b) as c:
        pass
    with (x := (await y)):
        pass
    with ((x := (await a)), (y := (await b))):
        pass
    with ((x := (await a)), (y := (await b))):
        pass
    with (x := (await a)), (y := (await b)):
        pass