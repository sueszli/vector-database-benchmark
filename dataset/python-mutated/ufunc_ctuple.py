import cython

@cython.ufunc
@cython.cfunc
def add_one_add_two(x: cython.int) -> tuple[cython.int, cython.int]:
    if False:
        print('Hello World!')
    return (x + 1, x + 2)