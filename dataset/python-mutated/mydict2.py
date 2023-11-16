class Dict(dict):
    """
    Simple dict but also support access as x.y style.

    >>> d1 = Dict()
    >>> d1['x'] = 100
    >>> d1.x
    100
    >>> d1.y = 200
    >>> d1['y']
    200
    >>> d2 = Dict(a=1, b=2, c='3')
    >>> d2.c
    '3'
    >>> d2['empty']
    Traceback (most recent call last):
        ...
    KeyError: 'empty'
    >>> d2.empty
    Traceback (most recent call last):
        ...
    AttributeError: 'Dict' object has no attribute 'empty'
    """

    def __init__(self, **kw):
        if False:
            while True:
                i = 10
        super(Dict, self).__init__(**kw)

    def __getattr__(self, key):
        if False:
            print('Hello World!')
        try:
            return self[key]
        except KeyError:
            raise AttributeError("'Dict' object has no attribute '%s'" % key)

    def __setattr__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        self[key] = value
if __name__ == '__main__':
    import doctest
    doctest.testmod()