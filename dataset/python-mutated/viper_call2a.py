@micropython.viper
def f2a(x, y):
    if False:
        i = 10
        return i + 15
    return x

@micropython.native
def call(r):
    if False:
        print('Hello World!')
    f = f2a
    for _ in r:
        f(1, 2)
bm_params = {(50, 10): (15000,), (100, 10): (30000,), (1000, 10): (300000,), (5000, 10): (1500000,)}

def bm_setup(params):
    if False:
        i = 10
        return i + 15
    return (lambda : call(range(params[0])), lambda : (params[0] // 1000, None))