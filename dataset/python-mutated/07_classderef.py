def CFUNCTYPE(argtypes):
    if False:
        return 10

    class CFunctionType(object):
        _argtypes_ = argtypes
    return CFunctionType