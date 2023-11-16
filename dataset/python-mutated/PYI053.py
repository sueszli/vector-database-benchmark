def f1(x: str='50 character stringggggggggggggggggggggggggggggggg') -> None:
    if False:
        return 10
    ...

def f2(x: str='51 character stringgggggggggggggggggggggggggggggggg') -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f3(x: str='50 character stringggggggggggggggggggggggggggggg😀') -> None:
    if False:
        while True:
            i = 10
    ...

def f4(x: str='51 character stringgggggggggggggggggggggggggggggg😀') -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f5(x: bytes=b'50 character byte stringgggggggggggggggggggggggggg') -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f6(x: bytes=b'51 character byte stringgggggggggggggggggggggggggg') -> None:
    if False:
        print('Hello World!')
    ...

def f7(x: bytes=b'50 character byte stringggggggggggggggggggggggggg\xff') -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f8(x: bytes=b'50 character byte stringgggggggggggggggggggggggggg\xff') -> None:
    if False:
        return 10
    ...
foo: str = '50 character stringggggggggggggggggggggggggggggggg'
bar: str = '51 character stringgggggggggggggggggggggggggggggggg'
baz: bytes = b'50 character byte stringgggggggggggggggggggggggggg'
qux: bytes = b'51 character byte stringggggggggggggggggggggggggggg\xff'

class Demo:
    """Docstrings are excluded from this rule. Some padding."""

def func() -> None:
    if False:
        while True:
            i = 10
    'Docstrings are excluded from this rule. Some padding.'