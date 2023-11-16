"""A convenience class replicating some lua table syntax with a python dict.

In general, should behave like a dictionary except that we can use dot notation
 to access keys. Users should be careful to only provide keys suitable for
 instance variable names.

Nota bene: do not use the key "keys" since it will collide with the method keys.

Usage example:

>>> t = T(a=5,b='kaw', c=T(v=[],x=33))
>>> t.a
5
>>> t.z = None
>>> print t
T(a=5, z=None, c=T(x=33, v=[]), b='kaw')

>>> t2 = T({'h':'f','x':4})
>>> t2
T(h='f', x=4)
>>> t2['x']
4
"""

class T(object):
    """Class for emulating lua tables."""

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        if len(args) > 1 or (len(args) == 1 and len(kwargs) > 0):
            errmsg = 'constructor only allows a single dict as a positional\n      argument or keyword arguments'
            raise ValueError(errmsg)
        if len(args) == 1 and isinstance(args[0], dict):
            self.__dict__.update(args[0])
        else:
            self.__dict__.update(kwargs)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        fmt = ', '.join(('%s=%s' for i in range(len(self.__dict__))))
        kwargstr = fmt % tuple((x for tup in self.__dict__.items() for x in [str(tup[0]), repr(tup[1])]))
        return 'T(' + kwargstr + ')'

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return self.__dict__[key]

    def __setitem__(self, key, val):
        if False:
            for i in range(10):
                print('nop')
        self.__dict__[key] = val

    def __delitem__(self, key):
        if False:
            return 10
        del self.__dict__[key]

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self.__dict__)

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.__dict__)

    def keys(self):
        if False:
            while True:
                i = 10
        return self.__dict__.keys()

    def iteritems(self):
        if False:
            i = 10
            return i + 15
        return self.__dict__.iteritems()