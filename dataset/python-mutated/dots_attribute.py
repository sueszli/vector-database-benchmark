@alpha
@beta
@charlie('xyz')
def foo():
    if False:
        i = 10
        return i + 15
    x = 42

@beta
@alpha
@charlie('bla')
def foo():
    if False:
        return 10
    x = 42