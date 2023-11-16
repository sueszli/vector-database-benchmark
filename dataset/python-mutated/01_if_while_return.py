def f1():
    if False:
        i = 10
        return i + 15
    if True:
        while True:
            return 5

def f2():
    if False:
        return 10
    if True:
        while 1:
            return 6
assert f1() == 5 and f2() == 6