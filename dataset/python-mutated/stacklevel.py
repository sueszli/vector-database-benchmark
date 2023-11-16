import warnings

def outer(message, stacklevel=1):
    if False:
        print('Hello World!')
    inner(message, stacklevel)

def inner(message, stacklevel=1):
    if False:
        i = 10
        return i + 15
    warnings.warn(message, stacklevel=stacklevel)