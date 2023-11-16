"""Raise the given error at evaluation time"""

def reraise(error):
    if False:
        for i in range(10):
            print('nop')
    'Return a function that raises the given error when evaluated'

    def local_function(*args, **kwargs):
        if False:
            return 10
        raise error
    return local_function