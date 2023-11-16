import warnings

def always_warn(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        warnings.warn(*args, **kwargs)