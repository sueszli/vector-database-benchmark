def f12(x, y=os.pathsep) -> None:
    if False:
        while True:
            i = 10
    ...

def f11(*, x='x') -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f13(x=['foo', 'bar', 'baz']) -> None:
    if False:
        i = 10
        return i + 15
    ...

def f14(x=('foo', 'bar', 'baz')) -> None:
    if False:
        while True:
            i = 10
    ...

def f15(x={'foo', 'bar', 'baz'}) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f151(x={1: 2}) -> None:
    if False:
        while True:
            i = 10
    ...

def f152(x={1: 2, **{3: 4}}) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f153(x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) -> None:
    if False:
        i = 10
        return i + 15
    ...

def f154(x=('foo', ('bar', 'baz'))) -> None:
    if False:
        return 10
    ...

def f141(x=[*range(10)]) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f142(x=list(range(10))) -> None:
    if False:
        i = 10
        return i + 15
    ...

def f16(x=frozenset({b'foo', b'bar', b'baz'})) -> None:
    if False:
        i = 10
        return i + 15
    ...

def f17(x='foo' + 'bar') -> None:
    if False:
        print('Hello World!')
    ...

def f18(x=b'foo' + b'bar') -> None:
    if False:
        print('Hello World!')
    ...

def f19(x='foo' + 4) -> None:
    if False:
        print('Hello World!')
    ...

def f20(x=5 + 5) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f21(x=3j - 3j) -> None:
    if False:
        while True:
            i = 10
    ...

def f22(x=-42.5j + 4.3j) -> None:
    if False:
        i = 10
        return i + 15
    ...