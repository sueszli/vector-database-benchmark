import builtins
from . import Image, _imagingmath

def _isconstant(v):
    if False:
        print('Hello World!')
    return isinstance(v, (int, float))

class _Operand:
    """Wraps an image operand, providing standard operators"""

    def __init__(self, im):
        if False:
            while True:
                i = 10
        self.im = im

    def __fixup(self, im1):
        if False:
            while True:
                i = 10
        if isinstance(im1, _Operand):
            if im1.im.mode in ('1', 'L'):
                return im1.im.convert('I')
            elif im1.im.mode in ('I', 'F'):
                return im1.im
            else:
                msg = f'unsupported mode: {im1.im.mode}'
                raise ValueError(msg)
        elif _isconstant(im1) and self.im.mode in ('1', 'L', 'I'):
            return Image.new('I', self.im.size, im1)
        else:
            return Image.new('F', self.im.size, im1)

    def apply(self, op, im1, im2=None, mode=None):
        if False:
            for i in range(10):
                print('nop')
        im1 = self.__fixup(im1)
        if im2 is None:
            out = Image.new(mode or im1.mode, im1.size, None)
            im1.load()
            try:
                op = getattr(_imagingmath, op + '_' + im1.mode)
            except AttributeError as e:
                msg = f"bad operand type for '{op}'"
                raise TypeError(msg) from e
            _imagingmath.unop(op, out.im.id, im1.im.id)
        else:
            im2 = self.__fixup(im2)
            if im1.mode != im2.mode:
                if im1.mode != 'F':
                    im1 = im1.convert('F')
                if im2.mode != 'F':
                    im2 = im2.convert('F')
            if im1.size != im2.size:
                size = (min(im1.size[0], im2.size[0]), min(im1.size[1], im2.size[1]))
                if im1.size != size:
                    im1 = im1.crop((0, 0) + size)
                if im2.size != size:
                    im2 = im2.crop((0, 0) + size)
            out = Image.new(mode or im1.mode, im1.size, None)
            im1.load()
            im2.load()
            try:
                op = getattr(_imagingmath, op + '_' + im1.mode)
            except AttributeError as e:
                msg = f"bad operand type for '{op}'"
                raise TypeError(msg) from e
            _imagingmath.binop(op, out.im.id, im1.im.id, im2.im.id)
        return _Operand(out)

    def __bool__(self):
        if False:
            i = 10
            return i + 15
        return self.im.getbbox() is not None

    def __abs__(self):
        if False:
            print('Hello World!')
        return self.apply('abs', self)

    def __pos__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __neg__(self):
        if False:
            print('Hello World!')
        return self.apply('neg', self)

    def __add__(self, other):
        if False:
            return 10
        return self.apply('add', self, other)

    def __radd__(self, other):
        if False:
            return 10
        return self.apply('add', other, self)

    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.apply('sub', self, other)

    def __rsub__(self, other):
        if False:
            print('Hello World!')
        return self.apply('sub', other, self)

    def __mul__(self, other):
        if False:
            print('Hello World!')
        return self.apply('mul', self, other)

    def __rmul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.apply('mul', other, self)

    def __truediv__(self, other):
        if False:
            while True:
                i = 10
        return self.apply('div', self, other)

    def __rtruediv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.apply('div', other, self)

    def __mod__(self, other):
        if False:
            i = 10
            return i + 15
        return self.apply('mod', self, other)

    def __rmod__(self, other):
        if False:
            return 10
        return self.apply('mod', other, self)

    def __pow__(self, other):
        if False:
            print('Hello World!')
        return self.apply('pow', self, other)

    def __rpow__(self, other):
        if False:
            return 10
        return self.apply('pow', other, self)

    def __invert__(self):
        if False:
            while True:
                i = 10
        return self.apply('invert', self)

    def __and__(self, other):
        if False:
            return 10
        return self.apply('and', self, other)

    def __rand__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.apply('and', other, self)

    def __or__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.apply('or', self, other)

    def __ror__(self, other):
        if False:
            return 10
        return self.apply('or', other, self)

    def __xor__(self, other):
        if False:
            return 10
        return self.apply('xor', self, other)

    def __rxor__(self, other):
        if False:
            return 10
        return self.apply('xor', other, self)

    def __lshift__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.apply('lshift', self, other)

    def __rshift__(self, other):
        if False:
            return 10
        return self.apply('rshift', self, other)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return self.apply('eq', self, other)

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return self.apply('ne', self, other)

    def __lt__(self, other):
        if False:
            return 10
        return self.apply('lt', self, other)

    def __le__(self, other):
        if False:
            print('Hello World!')
        return self.apply('le', self, other)

    def __gt__(self, other):
        if False:
            i = 10
            return i + 15
        return self.apply('gt', self, other)

    def __ge__(self, other):
        if False:
            print('Hello World!')
        return self.apply('ge', self, other)

def imagemath_int(self):
    if False:
        while True:
            i = 10
    return _Operand(self.im.convert('I'))

def imagemath_float(self):
    if False:
        print('Hello World!')
    return _Operand(self.im.convert('F'))

def imagemath_equal(self, other):
    if False:
        while True:
            i = 10
    return self.apply('eq', self, other, mode='I')

def imagemath_notequal(self, other):
    if False:
        print('Hello World!')
    return self.apply('ne', self, other, mode='I')

def imagemath_min(self, other):
    if False:
        for i in range(10):
            print('nop')
    return self.apply('min', self, other)

def imagemath_max(self, other):
    if False:
        for i in range(10):
            print('nop')
    return self.apply('max', self, other)

def imagemath_convert(self, mode):
    if False:
        return 10
    return _Operand(self.im.convert(mode))
ops = {}
for (k, v) in list(globals().items()):
    if k[:10] == 'imagemath_':
        ops[k[10:]] = v

def eval(expression, _dict={}, **kw):
    if False:
        i = 10
        return i + 15
    '\n    Evaluates an image expression.\n\n    :param expression: A string containing a Python-style expression.\n    :param options: Values to add to the evaluation context.  You\n                    can either use a dictionary, or one or more keyword\n                    arguments.\n    :return: The evaluated expression. This is usually an image object, but can\n             also be an integer, a floating point value, or a pixel tuple,\n             depending on the expression.\n    '
    args = ops.copy()
    args.update(_dict)
    args.update(kw)
    for (k, v) in args.items():
        if hasattr(v, 'im'):
            args[k] = _Operand(v)
    compiled_code = compile(expression, '<string>', 'eval')

    def scan(code):
        if False:
            print('Hello World!')
        for const in code.co_consts:
            if type(const) is type(compiled_code):
                scan(const)
        for name in code.co_names:
            if name not in args and name != 'abs':
                msg = f"'{name}' not allowed"
                raise ValueError(msg)
    scan(compiled_code)
    out = builtins.eval(expression, {'__builtins': {'abs': abs}}, args)
    try:
        return out.im
    except AttributeError:
        return out