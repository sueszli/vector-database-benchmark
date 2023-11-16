@test([1, 2, 3, 4])
def f():
    if False:
        for i in range(10):
            print('nop')
    pass

@test([1, 2, 3, 4])
def f():
    if False:
        while True:
            i = 10
    pass

@test([1, 2, 3, 4])
def f():
    if False:
        for i in range(10):
            print('nop')
    pass

@test([1, 2, 3, 4])
def f():
    if False:
        return 10
    pass