from cython.cimports.cython import view

def main():
    if False:
        return 10
    a: cython.int[:, ::view.contiguous]
    b: cython.int[::view.indirect_contiguous, ::1]
    c: cython.int[::view.generic, :]