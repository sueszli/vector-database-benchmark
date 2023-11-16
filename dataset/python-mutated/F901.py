def f() -> None:
    if False:
        print('Hello World!')
    raise NotImplemented()

def g() -> None:
    if False:
        i = 10
        return i + 15
    raise NotImplemented