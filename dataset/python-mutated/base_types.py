"""Definition of cross-platform types and structures"""
import six
from ctypes import Structure as Struct
from ctypes import memmove
from ctypes import addressof

def _construct(typ, buf):
    if False:
        print('Hello World!')
    obj = typ.__new__(typ)
    memmove(addressof(obj), buf, len(buf))
    return obj

def _reduce(self):
    if False:
        print('Hello World!')
    return (_construct, (self.__class__, bytes(memoryview(self))))

class StructureMixIn(object):
    """Define printing and comparison behaviors to be used for the Structure class from ctypes"""

    def __str__(self):
        if False:
            while True:
                i = 10
        'Print out the fields of the ctypes Structure'
        lines = []
        for (field_name, _) in getattr(self, '_fields_', []):
            lines.append('%20s\t%s' % (field_name, getattr(self, field_name)))
        return '\n'.join(lines)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        'Return True if the two instances have the same coordinates'
        fields = getattr(self, '_fields_', [])
        if isinstance(other, Struct):
            try:
                if len(fields) != len(getattr(other, '_fields_', [])):
                    return False
                for (field_name, _) in fields:
                    if getattr(self, field_name) != getattr(other, field_name):
                        return False
                return True
            except AttributeError:
                return False
        elif isinstance(other, (list, tuple)):
            if len(fields) != len(other):
                return False
            try:
                for (i, (field_name, _)) in enumerate(fields):
                    if getattr(self, field_name) != other[i]:
                        return False
                return True
            except Exception:
                return False
        return False

    def __ne__(self, other):
        if False:
            return 10
        'Return False if the two instances have the same coordinates'
        return not self.__eq__(other)
    __hash__ = None

class Structure(Struct, StructureMixIn):
    """Override the Structure class from ctypes to add printing and comparison"""
    pass

class PointIteratorMixin(object):
    """Add iterator functionality to POINT structure"""
    x = None
    y = None

    def __iter__(self):
        if False:
            print('Hello World!')
        'Allow iteration through coordinates'
        yield self.x
        yield self.y

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        'Allow indexing of coordinates'
        if key == 0 or key == -2:
            return self.x
        elif key == 1 or key == -1:
            return self.y
        else:
            raise IndexError('Illegal index')

class RectExtMixin(object):
    """Wrap the RECT structure and add extra functionality"""
    _RECT = type(None)
    _POINT = type(None)

    def __init__(self, other=0, top=0, right=0, bottom=0):
        if False:
            print('Hello World!')
        'Provide a constructor for _RECT structures\n\n        A _RECT can be constructed by:\n        - Another RECT (each value will be copied)\n        - Values for left, top, right and bottom\n\n        e.g. my_rect = _RECT(otherRect)\n        or   my_rect = _RECT(10, 20, 34, 100)\n        '
        if isinstance(other, self._RECT):
            self.left = other.left
            self.right = other.right
            self.top = other.top
            self.bottom = other.bottom
        else:
            long_int = six.integer_types[-1]
            self.left = long_int(other)
            self.right = long_int(right)
            self.top = long_int(top)
            self.bottom = long_int(bottom)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a string representation of the _RECT'
        return '(L%d, T%d, R%d, B%d)' % (self.left, self.top, self.right, self.bottom)

    def __repr__(self):
        if False:
            return 10
        'Return some representation of the _RECT'
        return '<RECT L%d, T%d, R%d, B%d>' % (self.left, self.top, self.right, self.bottom)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        'Allow iteration through coordinates'
        yield self.left
        yield self.top
        yield self.right
        yield self.bottom

    def __sub__(self, other):
        if False:
            while True:
                i = 10
        'Return a new rectangle which is offset from the one passed in'
        new_rect = self._RECT()
        new_rect.left = self.left - other.left
        new_rect.right = self.right - other.left
        new_rect.top = self.top - other.top
        new_rect.bottom = self.bottom - other.top
        return new_rect

    def __add__(self, other):
        if False:
            return 10
        'Allow two rects to be added using +'
        new_rect = self._RECT()
        new_rect.left = self.left + other.left
        new_rect.right = self.right + other.left
        new_rect.top = self.top + other.top
        new_rect.bottom = self.bottom + other.top
        return new_rect

    def width(self):
        if False:
            print('Hello World!')
        'Return the width of the rect'
        return self.right - self.left

    def height(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the height of the rect'
        return self.bottom - self.top

    def mid_point(self):
        if False:
            return 10
        'Return a POINT structure representing the mid point'
        pt = self._POINT()
        pt.x = self.left + int(float(self.width()) / 2.0)
        pt.y = self.top + int(float(self.height()) / 2.0)
        return pt
    __reduce__ = _reduce