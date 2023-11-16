def test1():
    if False:
        print('Hello World!')
    source()
    foo()
    bar()
    sink()

def test2():
    if False:
        for i in range(10):
            print('nop')
    source()
    foo()
    bar()
    if baz():
        sink()

def test2():
    if False:
        print('Hello World!')
    if foo():
        source()
    bar()
    if baz():
        sink()