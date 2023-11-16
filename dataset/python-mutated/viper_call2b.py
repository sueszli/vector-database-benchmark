@micropython.viper
def f2b(x: int, y: int) -> int:
    if False:
        while True:
            i = 10
    return x + y

@micropython.native
def call(r):
    if False:
        return 10
    f = f2b
    for _ in r:
        f(1, 2)
bm_params = {(50, 10): (15000,), (100, 10): (30000,), (1000, 10): (300000,), (5000, 10): (1500000,)}

def bm_setup(params):
    if False:
        for i in range(10):
            print('nop')
    return (lambda : call(range(params[0])), lambda : (params[0] // 1000, None))