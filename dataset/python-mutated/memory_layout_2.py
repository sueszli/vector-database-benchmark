from cython.cimports.cython import view

def main():
    if False:
        print('Hello World!')
    a: cython.int[::view.indirect, ::1, :]
    b: cython.int[::view.indirect, :, ::1]
    c: cython.int[::view.indirect_contiguous, ::1, :]
    d: cython.int[::view.contiguous, ::view.indirect, :]
    e: cython.int[::1, ::view.indirect, :]
_ERRORS = u'\n12:17: Only dimension 2 may be contiguous and direct\n'