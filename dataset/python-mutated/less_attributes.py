@alpha
@beta
@charlie('xyz')
def foo():
    if False:
        for i in range(10):
            print('nop')
    x = 42

@beta
@alpha
@charlie('xyz')
def foo():
    if False:
        i = 10
        return i + 15
    x = 42