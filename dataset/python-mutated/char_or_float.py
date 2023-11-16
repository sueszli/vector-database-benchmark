from __future__ import print_function
char_or_float = cython.fused_type(cython.char, cython.float)

@cython.ccall
def plus_one(var: char_or_float) -> char_or_float:
    if False:
        while True:
            i = 10
    return var + 1

def show_me():
    if False:
        print('Hello World!')
    a: cython.char = 127
    b: cython.float = 127
    print('char', plus_one(a))
    print('float', plus_one(b))