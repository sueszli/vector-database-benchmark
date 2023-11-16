import cython
from cython.cimports.dishes import spamdish, sausage

@cython.cfunc
def prepare(d: cython.pointer(spamdish)) -> cython.void:
    if False:
        i = 10
        return i + 15
    d.oz_of_spam = 42
    d.filler = sausage

def serve():
    if False:
        while True:
            i = 10
    d: spamdish
    prepare(cython.address(d))
    print(f'{d.oz_of_spam} oz spam, filler no. {d.filler}')