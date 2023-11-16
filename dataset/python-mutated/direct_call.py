"""
Direct call executor module
"""

def execute(opts, data, func, args, kwargs):
    if False:
        print('Hello World!')
    '\n    Directly calls the given function with arguments\n    '
    return func(*args, **kwargs)