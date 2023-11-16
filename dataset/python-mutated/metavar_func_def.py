def foo():
    if False:
        return 10
    foo(1, 2)

def bar(bar1, bar2, bar3):
    if False:
        i = 10
        return i + 15
    bar(1, 2, 3)

def foobar(bar1) -> int:
    if False:
        i = 10
        return i + 15
    bar(1, 2, 3)
    foo()
    return 3