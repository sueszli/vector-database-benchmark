@(Hello := (lambda x: x))
def f():
    if False:
        return 10
    return None

@(omega := (lambda x: x(x)))
def g():
    if False:
        i = 10
        return i + 15
    return None

@(omega := x[lambda x: x(x)].hello)
def h():
    if False:
        print('Hello World!')
    return None