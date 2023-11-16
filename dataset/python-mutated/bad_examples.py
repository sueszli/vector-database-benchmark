"""These are bad examples and raise errors in in the classicalfunction compiler"""
from qiskit.circuit.classicalfunction.types import Int1, Int2

def id_no_type_arg(a) -> Int1:
    if False:
        return 10
    return a

def id_no_type_return(a: Int1):
    if False:
        print('Hello World!')
    return a

def id_bad_return(a: Int1) -> Int2:
    if False:
        for i in range(10):
            print('nop')
    return a

def out_of_scope(a: Int1) -> Int1:
    if False:
        while True:
            i = 10
    return a & c

def bit_not(a: Int1) -> Int1:
    if False:
        for i in range(10):
            print('nop')
    return ~a