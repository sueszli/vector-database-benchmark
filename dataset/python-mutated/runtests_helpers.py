"""
Wheel functions for integration tests
"""

def failure():
    if False:
        while True:
            i = 10
    __context__['retcode'] = 1
    return False

def success():
    if False:
        i = 10
        return i + 15
    return True