def foo():
    if False:
        print('Hello World!')
    foo('whatever sequence of chars')
    foo(f'constant string')
    foo(f'string {var} interpolation')