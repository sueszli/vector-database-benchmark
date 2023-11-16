import sys
import os.path
from pyi_testmod_gettemp import gettemp

def get_data_dir():
    if False:
        for i in range(10):
            print('nop')
    if getattr(sys, 'frozen', False):
        return gettemp('data')
    else:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')