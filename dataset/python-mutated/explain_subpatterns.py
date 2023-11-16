def test():
    if False:
        return 10
    foo(1)
    stuff()
    bar(1)

def test2():
    if False:
        while True:
            i = 10
    foo(1)
    stuff()
    bar(2)