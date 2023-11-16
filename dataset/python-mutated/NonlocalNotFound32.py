def testNonlocal():
    if False:
        i = 10
        return i + 15
    x = 0
    y = 0

    def f():
        if False:
            for i in range(10):
                print('nop')
        nonlocal z
    f()
testNonlocal()