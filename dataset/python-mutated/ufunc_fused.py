import cython

@cython.ufunc
@cython.cfunc
def generic_add_one(x: cython.numeric) -> cython.numeric:
    if False:
        print('Hello World!')
    return x + 1