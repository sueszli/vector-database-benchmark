"""Module with bad __all__

To test https://github.com/ipython/ipython/issues/9678
"""

def evil():
    if False:
        return 10
    pass

def puppies():
    if False:
        print('Hello World!')
    pass
__all__ = [evil, 'puppies']