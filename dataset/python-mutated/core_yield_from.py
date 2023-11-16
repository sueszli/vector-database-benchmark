def recursive_yield_from(depth, iter_):
    if False:
        for i in range(10):
            print('nop')
    if depth <= 0:
        for i in iter_:
            yield i
    else:
        yield from recursive_yield_from(depth - 1, iter_)

def test(n):
    if False:
        print('Hello World!')
    global result
    result = 0
    for i in recursive_yield_from(10, range(n)):
        result += i
bm_params = {(100, 10): (2000,), (1000, 10): (20000,), (5000, 10): (100000,)}

def bm_setup(params):
    if False:
        print('Hello World!')
    (nloop,) = params
    return (lambda : test(nloop), lambda : (nloop // 100, result))