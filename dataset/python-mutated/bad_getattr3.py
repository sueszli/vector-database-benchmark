def __getattr__(name):
    if False:
        for i in range(10):
            print('nop')
    if name != 'delgetattr':
        raise AttributeError
    del globals()['__getattr__']
    raise AttributeError