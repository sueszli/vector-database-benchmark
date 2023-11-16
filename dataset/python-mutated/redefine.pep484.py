def f1(x) -> Union[int, str]:
    if False:
        print('Hello World!')
    return 1

def f1(x) -> Union[int, str]:
    if False:
        print('Hello World!')
    return 'foo'

def f2(x) -> None:
    if False:
        for i in range(10):
            print('nop')
    pass

def f2(x, y) -> None:
    if False:
        i = 10
        return i + 15
    pass

def f3(x: int) -> int:
    if False:
        i = 10
        return i + 15
    return 'asd' + x

def f3(x: int) -> int:
    if False:
        while True:
            i = 10
    return 1 + x