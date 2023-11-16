"""
Get a full absolute path a file
"""
import os

def full_path(file):
    if False:
        for i in range(10):
            print('nop')
    return os.path.abspath(os.path.expanduser(file))