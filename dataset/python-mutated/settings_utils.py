import os
from distutils.util import strtobool

def force_bool(val):
    if False:
        while True:
            i = 10
    return strtobool(str(val))

def environ_get_list(names, default=None):
    if False:
        for i in range(10):
            print('nop')
    for name in names:
        if name in os.environ:
            return os.environ[name]
    return default