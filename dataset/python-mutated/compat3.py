import sys
__all__ = ['b', 'basestring_', 'bytes', 'unicode_', 'next', 'is_unicode']
if sys.version < '3':
    b = bytes = str
    basestring_ = basestring
    unicode_ = unicode
else:

    def b(s):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(s, str):
            return s.encode('latin1')
        return bytes(s)
    basestring_ = (bytes, str)
    bytes = bytes
    unicode_ = str
text = str
if sys.version < '3':

    def next(obj):
        if False:
            while True:
                i = 10
        return obj.next()
else:
    next = next
if sys.version < '3':

    def is_unicode(obj):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(obj, unicode)
else:

    def is_unicode(obj):
        if False:
            i = 10
            return i + 15
        return isinstance(obj, str)

def coerce_text(v):
    if False:
        i = 10
        return i + 15
    if not isinstance(v, basestring_):
        if sys.version < '3':
            attr = '__unicode__'
        else:
            attr = '__str__'
        if hasattr(v, attr):
            return unicode(v)
        else:
            return bytes(v)
    return v