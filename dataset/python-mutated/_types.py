import locale
from visidata import options, TypedWrapper, vd, VisiData
vd.help_float_fmt = "\n- fmt starting with `'%'` (like `%0.2f`) will use [:onclick https://docs.python.org/3.6/library/locale.html#locale.format_string]locale.format_string[/]\n- other fmt (like `{:.02f}` is passed to Python [:onclick https://docs.python.org/3/library/string.html#custom-string-formatting)]string.format][/]\n"
vd.help_int_fmt = "\n- fmt starting with `'%'` (like `%04d`) will use [:onclick https://docs.python.org/3.6/library/locale.html#locale.format_string]locale.format_string[/]\n- other fmt (like `{:4d}` is passed to Python [:onclick https://docs.python.org/3/library/string.html#custom-string-formatting)]string.format[/]\n"
vd.option('disp_float_fmt', '{:.02f}', 'default fmtstr to format float values', replay=True, help=vd.help_float_fmt)
vd.option('disp_int_fmt', '{:d}', 'default fmtstr to format int values', replay=True, help=vd.help_int_fmt)
vd.numericTypes = [int, float]

def anytype(r=None):
    if False:
        print('Hello World!')
    'minimalist "any" passthrough type'
    return r
anytype.__name__ = ''

@VisiData.global_api
def numericFormatter(vd, fmtstr, typedval):
    if False:
        for i in range(10):
            print('nop')
    try:
        fmtstr = fmtstr or options['disp_' + type(typedval).__name__ + '_fmt']
        if fmtstr[0] == '%':
            return locale.format_string(fmtstr, typedval, grouping=False)
        else:
            return fmtstr.format(typedval)
    except ValueError:
        return str(typedval)

@VisiData.api
def numericType(vd, icon='', fmtstr='', formatter=vd.numericFormatter):
    if False:
        i = 10
        return i + 15
    'Decorator for numeric types.'

    def _decorator(f):
        if False:
            print('Hello World!')
        vd.addType(f, icon=icon, fmtstr=fmtstr, formatter=formatter)
        vd.numericTypes.append(f)
        vd.addGlobals({f.__name__: f})
        return f
    return _decorator

class VisiDataType:
    """Register *typetype* in the typemap."""

    def __init__(self, typetype=None, icon=None, fmtstr='', formatter=vd.numericFormatter, key='', name=None):
        if False:
            return 10
        self.typetype = typetype or anytype
        self.name = name or getattr(typetype, '__name__', str(typetype))
        self.icon = icon
        self.fmtstr = fmtstr
        self.formatter = formatter
        self.key = key

@VisiData.api
def addType(vd, typetype=None, icon=None, fmtstr='', formatter=vd.numericFormatter, key='', name=None):
    if False:
        for i in range(10):
            print('nop')
    'Add type to type map.\n\n    - *typetype*: actual type class *TYPE* above\n    - *icon*: unicode character in column header\n    - *fmtstr*: format string to use if fmtstr not given\n    - *formatter*: formatting function to call as ``formatter(fmtstr, typedvalue)``\n    '
    t = VisiDataType(typetype=typetype, icon=icon, fmtstr=fmtstr, formatter=formatter, key=key, name=name)
    if typetype:
        vd.typemap[typetype] = t
    return t
vdtype = vd.addType
vd.typemap = {}

@VisiData.api
def getType(vd, typetype):
    if False:
        while True:
            i = 10
    return vd.typemap.get(typetype) or VisiDataType()
vdtype(None, '∅')
vdtype(anytype, '', formatter=lambda _, v: str(v))
vdtype(str, '~', formatter=lambda _, v: v)
vdtype(int, '#')
vdtype(float, '%')
vdtype(dict, '')
vdtype(list, '')

@VisiData.api
def isNumeric(vd, col):
    if False:
        i = 10
        return i + 15
    return col.type in vd.numericTypes

def deduceType(v):
    if False:
        i = 10
        return i + 15
    if isinstance(v, (float, int)):
        return type(v)
    else:
        return anytype

@vd.numericType('%')
def floatlocale(*args):
    if False:
        print('Hello World!')
    'Calculate float() using system locale set in LC_NUMERIC.'
    if not args:
        return 0.0
    return locale.atof(*args)

@vd.numericType('♯', fmtstr='%d')
class vlen(int):

    def __new__(cls, v=0):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(v, (vlen, int, float)):
            return super(vlen, cls).__new__(cls, v)
        else:
            return super(vlen, cls).__new__(cls, len(v))

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self
vd.addGlobals(anytype=anytype, vdtype=vdtype, deduceType=deduceType)