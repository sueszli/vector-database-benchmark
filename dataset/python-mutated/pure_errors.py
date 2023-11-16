import cython

@cython.cfunc
@cython.locals(x=cython.int)
@cython.returns(cython.int)
def cdef_needs_gil(x):
    if False:
        for i in range(10):
            print('nop')
    return x + 1

@cython.cfunc
@cython.nogil
@cython.locals(x=cython.int)
@cython.returns(cython.int)
def cdef_nogil(x):
    if False:
        for i in range(10):
            print('nop')
    return x + 1

@cython.cfunc
@cython.nogil(True)
@cython.locals(x=cython.int)
@cython.returns(cython.int)
def cdef_nogil_true(x):
    if False:
        print('Hello World!')
    return x + 1

@cython.cfunc
@cython.nogil(False)
@cython.locals(x=cython.int)
@cython.returns(cython.int)
def cdef_nogil_false(x):
    if False:
        while True:
            i = 10
    return x + 1

@cython.locals(x=cython.int)
def test_cdef_nogil(x):
    if False:
        for i in range(10):
            print('nop')
    cdef_nogil(x)
    cdef_nogil_false(x)
    cdef_nogil_true(x)
    with cython.nogil:
        cdef_nogil_true(x)
        cdef_needs_gil(x)
        cdef_nogil_false(x)

@cython.nogil
def pyfunc(x):
    if False:
        return 10
    return x + 1

@cython.exceptval(-1)
@cython.cfunc
def test_cdef_return_object_broken(x: object) -> object:
    if False:
        for i in range(10):
            print('nop')
    return x

@cython.ccall
@cython.cfunc
def test_contradicting_decorators1(x: object) -> object:
    if False:
        return 10
    return x

@cython.cfunc
@cython.ccall
def test_contradicting_decorators2(x: object) -> object:
    if False:
        while True:
            i = 10
    return x

@cython.cfunc
@cython.ufunc
def add_one(x: cython.double) -> cython.double:
    if False:
        i = 10
        return i + 15
    return x + 1
_ERRORS = "\n44:22: Calling gil-requiring function not allowed without gil\n45:24: Calling gil-requiring function not allowed without gil\n48:0: Python functions cannot be declared 'nogil'\n53:0: Exception clause not allowed for function returning Python object\n59:0: cfunc and ccall directives cannot be combined\n65:0: cfunc and ccall directives cannot be combined\n71:0: Cannot apply @cfunc to @ufunc, please reverse the decorators.\n"
_WARNINGS = "\n30:0: Directive does not change previous value (nogil=False)\n# bugs:\n59:0: 'test_contradicting_decorators1' redeclared\n65:0: 'test_contradicting_decorators2' redeclared\n"