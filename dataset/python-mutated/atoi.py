from cython.cimports.libc.stdlib import atoi

@cython.cfunc
def parse_charptr_to_py_int(s: cython.p_char):
    if False:
        i = 10
        return i + 15
    assert s is not cython.NULL, 'byte string value is NULL'
    return atoi(s)