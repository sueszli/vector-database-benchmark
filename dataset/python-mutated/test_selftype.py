import gc

def leak():
    if False:
        for i in range(10):
            print('nop')

    class T(type):
        pass

    class U(type, metaclass=T):
        pass
    U.__class__ = U
    del U
    gc.collect()
    gc.collect()
    gc.collect()