@micropython.viper
def f1b(x) -> int:
    if False:
        i = 10
        return i + 15
    return int(x)

@micropython.native
def call(r):
    if False:
        while True:
            i = 10
    f = f1b
    for _ in r:
        f(1)
bm_params = {(50, 10): (15000,), (100, 10): (30000,), (1000, 10): (300000,), (5000, 10): (1500000,)}

def bm_setup(params):
    if False:
        for i in range(10):
            print('nop')
    return (lambda : call(range(params[0])), lambda : (params[0] // 1000, None))