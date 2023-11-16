import cython
from cython.cimports.my_module import Shrubbery

@cython.cfunc
def widen_shrubbery(sh: Shrubbery, extra_width):
    if False:
        for i in range(10):
            print('nop')
    sh.width = sh.width + extra_width