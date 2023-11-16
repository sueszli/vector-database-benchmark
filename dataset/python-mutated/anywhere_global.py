MAKE_BAR_INSECURE = True

def foo():
    if False:
        while True:
            i = 10
    bar()
    foo()
    if MAKE_BAR_INSECURE:
        print('Hello, world!')
    return 2
bar()