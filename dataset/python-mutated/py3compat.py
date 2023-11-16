"""
spyder.py3compat
----------------

Transitional module providing compatibility functions intended to help
migrating from Python 2 to Python 3.
"""
import operator
import pickle
TEXT_TYPES = (str,)
INT_TYPES = (int,)

def is_type_text_string(obj):
    if False:
        print('Hello World!')
    'Return True if `obj` is type text string, False if it is anything else,\n    like an instance of a class that extends the basestring class.'
    return type(obj) in [str, bytes]

def is_text_string(obj):
    if False:
        while True:
            i = 10
    'Return True if `obj` is a text string, False if it is anything else,\n    like binary data (Python 3) or QString (PyQt API #1)'
    return isinstance(obj, str)

def is_binary_string(obj):
    if False:
        while True:
            i = 10
    'Return True if `obj` is a binary string, False if it is anything else'
    return isinstance(obj, bytes)

def is_string(obj):
    if False:
        i = 10
        return i + 15
    'Return True if `obj` is a text or binary Python string object,\n    False if it is anything else, like a QString (PyQt API #1)'
    return is_text_string(obj) or is_binary_string(obj)

def to_text_string(obj, encoding=None):
    if False:
        return 10
    'Convert `obj` to (unicode) text string'
    if encoding is None:
        return str(obj)
    elif isinstance(obj, str):
        return obj
    else:
        return str(obj, encoding)

def to_binary_string(obj, encoding='utf-8'):
    if False:
        while True:
            i = 10
    'Convert `obj` to binary string (bytes)'
    return bytes(obj, encoding)

def qbytearray_to_str(qba):
    if False:
        i = 10
        return i + 15
    'Convert QByteArray object to str in a way compatible with Python 3'
    return str(bytes(qba.toHex().data()).decode())

def iterkeys(d, **kw):
    if False:
        print('Hello World!')
    return iter(d.keys(**kw))

def itervalues(d, **kw):
    if False:
        i = 10
        return i + 15
    return iter(d.values(**kw))

def iteritems(d, **kw):
    if False:
        print('Hello World!')
    return iter(d.items(**kw))

def iterlists(d, **kw):
    if False:
        for i in range(10):
            print('nop')
    return iter(d.lists(**kw))
viewkeys = operator.methodcaller('keys')
viewvalues = operator.methodcaller('values')
viewitems = operator.methodcaller('items')
if __name__ == '__main__':
    pass