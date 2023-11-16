class C:

    def __init__(self):
        if False:
            print('Hello World!')
        print('__init__')

    def __enter__(self):
        if False:
            print('Hello World!')
        print('__enter__')

    def __exit__(self, a, b, c):
        if False:
            i = 10
            return i + 15
        print('__exit__', a, b, c)

@micropython.viper
def f():
    if False:
        return 10
    with C():
        print(1)
f()

@micropython.viper
def f():
    if False:
        i = 10
        return i + 15
    try:
        with C():
            print(1)
            fail
            print(2)
    except NameError:
        print('NameError')
f()