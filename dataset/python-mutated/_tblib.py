import re
import sys
from types import CodeType
__version__ = '2.0.0'
__all__ = ('Traceback', 'TracebackParseError', 'Frame', 'Code')
FRAME_RE = re.compile('^\\s*File "(?P<co_filename>.+)", line (?P<tb_lineno>\\d+)(, in (?P<co_name>.+))?$')

class _AttrDict(dict):
    __slots__ = ()

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

class __traceback_maker(Exception):
    pass

class TracebackParseError(Exception):
    pass

class Code:
    """
    Class that replicates just enough of the builtin Code object to enable serialization and traceback rendering.
    """
    co_code = None

    def __init__(self, code):
        if False:
            i = 10
            return i + 15
        self.co_filename = code.co_filename
        self.co_name = code.co_name
        self.co_argcount = 0
        self.co_kwonlyargcount = 0
        self.co_varnames = ()
        self.co_nlocals = 0
        self.co_stacksize = 0
        self.co_flags = 64
        self.co_firstlineno = 0

class Frame:
    """
    Class that replicates just enough of the builtin Frame object to enable serialization and traceback rendering.
    """

    def __init__(self, frame):
        if False:
            return 10
        self.f_locals = {}
        self.f_globals = {k: v for (k, v) in frame.f_globals.items() if k in ('__file__', '__name__')}
        self.f_code = Code(frame.f_code)
        self.f_lineno = frame.f_lineno

    def clear(self):
        if False:
            i = 10
            return i + 15
        '\n        For compatibility with PyPy 3.5;\n        clear() was added to frame in Python 3.4\n        and is called by traceback.clear_frames(), which\n        in turn is called by unittest.TestCase.assertRaises\n        '

class Traceback:
    """
    Class that wraps builtin Traceback objects.
    """
    tb_next = None

    def __init__(self, tb):
        if False:
            for i in range(10):
                print('nop')
        self.tb_frame = Frame(tb.tb_frame)
        self.tb_lineno = int(tb.tb_lineno)
        tb = tb.tb_next
        prev_traceback = self
        cls = type(self)
        while tb is not None:
            traceback = object.__new__(cls)
            traceback.tb_frame = Frame(tb.tb_frame)
            traceback.tb_lineno = int(tb.tb_lineno)
            prev_traceback.tb_next = traceback
            prev_traceback = traceback
            tb = tb.tb_next

    def as_traceback(self):
        if False:
            print('Hello World!')
        '\n        Convert to a builtin Traceback object that is usable for raising or rendering a stacktrace.\n        '
        current = self
        top_tb = None
        tb = None
        while current:
            f_code = current.tb_frame.f_code
            code = compile('\n' * (current.tb_lineno - 1) + 'raise __traceback_maker', current.tb_frame.f_code.co_filename, 'exec')
            if hasattr(code, 'replace'):
                code = code.replace(co_argcount=0, co_filename=f_code.co_filename, co_name=f_code.co_name, co_freevars=(), co_cellvars=())
            else:
                code = CodeType(0, code.co_kwonlyargcount, code.co_nlocals, code.co_stacksize, code.co_flags, code.co_code, code.co_consts, code.co_names, code.co_varnames, f_code.co_filename, f_code.co_name, code.co_firstlineno, code.co_lnotab, (), ())
            try:
                exec(code, dict(current.tb_frame.f_globals), {})
            except Exception:
                next_tb = sys.exc_info()[2].tb_next
                if top_tb is None:
                    top_tb = next_tb
                if tb is not None:
                    tb.tb_next = next_tb
                tb = next_tb
                del next_tb
            current = current.tb_next
        try:
            return top_tb
        finally:
            del top_tb
            del tb
    to_traceback = as_traceback

    def as_dict(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts to a dictionary representation. You can serialize the result to JSON as it only has\n        builtin objects like dicts, lists, ints or strings.\n        '
        if self.tb_next is None:
            tb_next = None
        else:
            tb_next = self.tb_next.to_dict()
        code = {'co_filename': self.tb_frame.f_code.co_filename, 'co_name': self.tb_frame.f_code.co_name}
        frame = {'f_globals': self.tb_frame.f_globals, 'f_code': code, 'f_lineno': self.tb_frame.f_lineno}
        return {'tb_frame': frame, 'tb_lineno': self.tb_lineno, 'tb_next': tb_next}
    to_dict = as_dict

    @classmethod
    def from_dict(cls, dct):
        if False:
            i = 10
            return i + 15
        '\n        Creates an instance from a dictionary with the same structure as ``.as_dict()`` returns.\n        '
        if dct['tb_next']:
            tb_next = cls.from_dict(dct['tb_next'])
        else:
            tb_next = None
        code = _AttrDict(co_filename=dct['tb_frame']['f_code']['co_filename'], co_name=dct['tb_frame']['f_code']['co_name'])
        frame = _AttrDict(f_globals=dct['tb_frame']['f_globals'], f_code=code, f_lineno=dct['tb_frame']['f_lineno'])
        tb = _AttrDict(tb_frame=frame, tb_lineno=dct['tb_lineno'], tb_next=tb_next)
        return cls(tb)

    @classmethod
    def from_string(cls, string, strict=True):
        if False:
            i = 10
            return i + 15
        '\n        Creates an instance by parsing a stacktrace. Strict means that parsing stops when lines are not indented by at least two spaces\n        anymore.\n        '
        frames = []
        header = strict
        for line in string.splitlines():
            line = line.rstrip()
            if header:
                if line == 'Traceback (most recent call last):':
                    header = False
                continue
            frame_match = FRAME_RE.match(line)
            if frame_match:
                frames.append(frame_match.groupdict())
            elif line.startswith('  '):
                pass
            elif strict:
                break
        if frames:
            previous = None
            for frame in reversed(frames):
                previous = _AttrDict(frame, tb_frame=_AttrDict(frame, f_globals=_AttrDict(__file__=frame['co_filename'], __name__='?'), f_code=_AttrDict(frame), f_lineno=int(frame['tb_lineno'])), tb_next=previous)
            return cls(previous)
        else:
            raise TracebackParseError('Could not find any frames in %r.' % string)
import sys
from types import TracebackType

def unpickle_traceback(tb_frame, tb_lineno, tb_next):
    if False:
        return 10
    ret = object.__new__(Traceback)
    ret.tb_frame = tb_frame
    ret.tb_lineno = tb_lineno
    ret.tb_next = tb_next
    return ret.as_traceback()

def pickle_traceback(tb):
    if False:
        while True:
            i = 10
    return (unpickle_traceback, (Frame(tb.tb_frame), tb.tb_lineno, tb.tb_next and Traceback(tb.tb_next)))

def unpickle_exception(func, args, cause, tb):
    if False:
        while True:
            i = 10
    inst = func(*args)
    inst.__cause__ = cause
    inst.__traceback__ = tb
    return inst

def pickle_exception(obj):
    if False:
        print('Hello World!')
    rv = obj.__reduce_ex__(3)
    if isinstance(rv, str):
        raise TypeError('str __reduce__ output is not supported')
    assert isinstance(rv, tuple)
    assert len(rv) >= 2
    return (unpickle_exception, rv[:2] + (obj.__cause__, obj.__traceback__)) + rv[2:]

def _get_subclasses(cls):
    if False:
        i = 10
        return i + 15
    to_visit = [cls]
    while to_visit:
        this = to_visit.pop()
        yield this
        to_visit += list(this.__subclasses__())

def install(*exc_classes_or_instances):
    if False:
        return 10
    import copyreg
    copyreg.pickle(TracebackType, pickle_traceback)
    if sys.version_info.major < 3:
        if len(exc_classes_or_instances) == 1:
            exc = exc_classes_or_instances[0]
            if isinstance(exc, type) and issubclass(exc, BaseException):
                return exc
        return
    if not exc_classes_or_instances:
        for exception_cls in _get_subclasses(BaseException):
            copyreg.pickle(exception_cls, pickle_exception)
        return
    for exc in exc_classes_or_instances:
        if isinstance(exc, BaseException):
            while exc is not None:
                copyreg.pickle(type(exc), pickle_exception)
                exc = exc.__cause__
        elif isinstance(exc, type) and issubclass(exc, BaseException):
            copyreg.pickle(exc, pickle_exception)
            if len(exc_classes_or_instances) == 1:
                return exc
        else:
            raise TypeError('Expected subclasses or instances of BaseException, got %s' % type(exc))
_installed = False

def dump_traceback(tb):
    if False:
        i = 10
        return i + 15
    from pickle import dumps
    if tb is None:
        return dumps(None)
    tb = Traceback(tb)
    return dumps(tb.to_dict())

def load_traceback(s):
    if False:
        print('Hello World!')
    from pickle import loads
    as_dict = loads(s)
    if as_dict is None:
        return None
    tb = Traceback.from_dict(as_dict)
    return tb.as_traceback()