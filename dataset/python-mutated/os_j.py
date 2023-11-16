import os
import bjam
__OS = bjam.call('peek', [], 'OS')[0]

def name():
    if False:
        while True:
            i = 10
    return __OS

def environ(keys):
    if False:
        return 10
    return [os.environ[key] for key in keys if key in os.environ]