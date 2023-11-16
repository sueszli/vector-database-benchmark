import torch._C._lazy_ts_backend

def init():
    if False:
        for i in range(10):
            print('nop')
    'Initializes the lazy Torchscript backend'
    torch._C._lazy_ts_backend._init()