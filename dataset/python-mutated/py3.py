from __future__ import absolute_import, division, print_function, unicode_literals
import itertools
import sys
PY2 = sys.version_info.major == 2
if PY2:
    try:
        import _winreg as winreg
    except ImportError:
        winreg = None
    MAXINT = sys.maxint
    MININT = -sys.maxint - 1
    MAXFLOAT = sys.float_info.max
    MINFLOAT = sys.float_info.min
    string_types = (str, unicode)
    integer_types = (int, long)
    filter = itertools.ifilter
    map = itertools.imap
    range = xrange
    zip = itertools.izip
    long = long
    cmp = cmp
    bytes = bytes
    bstr = bytes
    from io import StringIO
    from urllib2 import urlopen, ProxyHandler, build_opener, install_opener
    from urllib import quote as urlquote

    def iterkeys(d):
        if False:
            i = 10
            return i + 15
        return d.iterkeys()

    def itervalues(d):
        if False:
            for i in range(10):
                print('nop')
        return d.itervalues()

    def iteritems(d):
        if False:
            print('Hello World!')
        return d.iteritems()

    def keys(d):
        if False:
            i = 10
            return i + 15
        return d.keys()

    def values(d):
        if False:
            i = 10
            return i + 15
        return d.values()

    def items(d):
        if False:
            while True:
                i = 10
        return d.items()
    import Queue as queue
else:
    try:
        import winreg
    except ImportError:
        winreg = None
    MAXINT = sys.maxsize
    MININT = -sys.maxsize - 1
    MAXFLOAT = sys.float_info.max
    MINFLOAT = sys.float_info.min
    string_types = (str,)
    integer_types = (int,)
    filter = filter
    map = map
    range = range
    zip = zip
    long = int

    def cmp(a, b):
        if False:
            for i in range(10):
                print('nop')
        return (a > b) - (a < b)

    def bytes(x):
        if False:
            i = 10
            return i + 15
        return x.encode('utf-8')

    def bstr(x):
        if False:
            while True:
                i = 10
        return str(x)
    from io import StringIO
    from urllib.request import urlopen, ProxyHandler, build_opener, install_opener
    from urllib.parse import quote as urlquote

    def iterkeys(d):
        if False:
            i = 10
            return i + 15
        return iter(d.keys())

    def itervalues(d):
        if False:
            i = 10
            return i + 15
        return iter(d.values())

    def iteritems(d):
        if False:
            i = 10
            return i + 15
        return iter(d.items())

    def keys(d):
        if False:
            print('Hello World!')
        return list(d.keys())

    def values(d):
        if False:
            i = 10
            return i + 15
        return list(d.values())

    def items(d):
        if False:
            for i in range(10):
                print('nop')
        return list(d.items())
    import queue as queue

def with_metaclass(meta, *bases):
    if False:
        i = 10
        return i + 15
    'Create a base class with a metaclass.'

    class metaclass(meta):

        def __new__(cls, name, this_bases, d):
            if False:
                while True:
                    i = 10
            return meta(name, bases, d)
    return type.__new__(metaclass, str('temporary_class'), (), {})