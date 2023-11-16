"""
Python2.5 and Python3.3 compatibility shim

Heavily inspirted by the "six" library.
https://pypi.python.org/pypi/six
"""
import sys
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
if PY3:

    def b(s):
        if False:
            i = 10
            return i + 15
        if isinstance(s, bytes):
            return s
        return s.encode('utf8', 'replace')

    def u(s):
        if False:
            i = 10
            return i + 15
        if isinstance(s, bytes):
            return s.decode('utf8', 'replace')
        return s
    from io import BytesIO as StringIO
    from configparser import RawConfigParser
else:

    def b(s):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(s, unicode):
            return s.encode('utf8', 'replace')
        return s

    def u(s):
        if False:
            print('Hello World!')
        if isinstance(s, unicode):
            return s
        if isinstance(s, int):
            s = str(s)
        return unicode(s, 'utf8', 'replace')
    from StringIO import StringIO
    from ConfigParser import RawConfigParser
b.__doc__ = 'Ensure we have a byte string'
u.__doc__ = 'Ensure we have a unicode string'