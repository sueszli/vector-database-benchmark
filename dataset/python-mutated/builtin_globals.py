assert 'NEW' not in globals()
globals().update(NEW=True)
assert 'NEW' in globals()

def default_args(value=NEW):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> default_args()\n    True\n    '
    return value