from __future__ import print_function
print('This is deep brother module talking.', __name__)

def someBrotherFunction():
    if False:
        return 10
    pass
print('The __module__ of function here is', someBrotherFunction.__module__)