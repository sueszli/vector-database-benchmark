from cython.cimports.cpython.ref import PyObject

def main():
    if False:
        for i in range(10):
            print('nop')
    python_string = 'foo'
    ptr = cython.cast(cython.p_void, python_string)
    adress_in_c = cython.cast(Py_intptr_t, ptr)
    address_from_void = adress_in_c
    ptr2 = cython.cast(cython.pointer(PyObject), python_string)
    address_in_c2 = cython.cast(Py_intptr_t, ptr2)
    address_from_PyObject = address_in_c2
    assert address_from_void == address_from_PyObject == id(python_string)
    print(cython.cast(object, ptr))
    print(cython.cast(object, ptr2))