import cython
double_or_object = cython.fused_type(cython.double, object)

def increment(x: double_or_object):
    if False:
        for i in range(10):
            print('nop')
    with cython.nogil(double_or_object is not object):
        x = x + 1
    return x
increment(5.0)
increment(5)