my_fused_type = cython.fused_type(cython.int, cython.float)

@cython.cfunc
def func(a: cython.pointer(my_fused_type)):
    if False:
        return 10
    print(a[0])

def main():
    if False:
        i = 10
        return i + 15
    a: cython.int = 3
    b: cython.float = 5.0
    func(cython.address(a))
    func(cython.address(b))