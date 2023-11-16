import torch

def div_int_nofuture():
    if False:
        i = 10
        return i + 15
    return 1 / 2

def div_float_nofuture():
    if False:
        for i in range(10):
            print('nop')
    return 3.14 / 0.125