__doc__ = "\nhashlib backwards-compatibility module for older (pre-2.5) Python versions\n\nThis does not not NOT (repeat, *NOT*) provide complete hashlib\nfunctionality.  It only wraps the portions of MD5 functionality used\nby SCons, in an interface that looks like hashlib (or enough for our\npurposes, anyway).  In fact, this module will raise an ImportError if\nthe underlying md5 module isn't available.\n"
__revision__ = 'src/engine/SCons/compat/_scons_hashlib.py  2014/07/05 09:42:21 garyo'
import md5
from string import hexdigits

class md5obj(object):
    md5_module = md5

    def __init__(self, name, string=''):
        if False:
            print('Hello World!')
        if not name in ('MD5', 'md5'):
            raise ValueError('unsupported hash type')
        self.name = 'md5'
        self.m = self.md5_module.md5()

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<%s HASH object @ %#x>' % (self.name, id(self))

    def copy(self):
        if False:
            return 10
        import copy
        result = copy.copy(self)
        result.m = self.m.copy()
        return result

    def digest(self):
        if False:
            print('Hello World!')
        return self.m.digest()

    def update(self, arg):
        if False:
            i = 10
            return i + 15
        return self.m.update(arg)

    def hexdigest(self):
        if False:
            for i in range(10):
                print('nop')
        return self.m.hexdigest()
new = md5obj

def md5(string=''):
    if False:
        i = 10
        return i + 15
    return md5obj('md5', string)