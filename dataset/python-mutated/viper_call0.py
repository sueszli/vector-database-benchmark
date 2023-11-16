@micropython.viper
def f0():
    if False:
        return 10
    pass

@micropython.native
def call(r):
    if False:
        return 10
    f = f0
    for _ in r:
        f()
bm_params = {(50, 10): (15000,), (100, 10): (30000,), (1000, 10): (300000,), (5000, 10): (1500000,)}

def bm_setup(params):
    if False:
        print('Hello World!')
    return (lambda : call(range(params[0])), lambda : (params[0] // 1000, None))