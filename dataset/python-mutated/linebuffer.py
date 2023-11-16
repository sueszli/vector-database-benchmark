"""

.. module:: linebuffer

Classes that hold the buffer for a *line* and can operate on it
with appends, forwarding, rewinding, resetting and other

.. moduleauthor:: Daniel Rodriguez

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import array
import collections
import datetime
from itertools import islice
import math
from .utils.py3 import range, with_metaclass, string_types
from .lineroot import LineRoot, LineSingle, LineMultiple
from . import metabase
from .utils import num2date, time2num
NAN = float('NaN')

class LineBuffer(LineSingle):
    """
    LineBuffer defines an interface to an "array.array" (or list) in which
    index 0 points to the item which is active for input and output.

    Positive indices fetch values from the past (left hand side)
    Negative indices fetch values from the future (if the array has been
    extended on the right hand side)

    With this behavior no index has to be passed around to entities which have
    to work with the current value produced by other entities: the value is
    always reachable at "0".

    Likewise storing the current value produced by "self" is done at 0.

    Additional operations to move the pointer (home, forward, extend, rewind,
    advance getzero) are provided

    The class can also hold "bindings" to other LineBuffers. When a value
    is set in this class
    it will also be set in the binding.
    """
    (UnBounded, QBuffer) = (0, 1)

    def __init__(self):
        if False:
            return 10
        self.lines = [self]
        self.mode = self.UnBounded
        self.bindings = list()
        self.reset()
        self._tz = None

    def get_idx(self):
        if False:
            i = 10
            return i + 15
        return self._idx

    def set_idx(self, idx, force=False):
        if False:
            print('Hello World!')
        if self.mode == self.QBuffer:
            if force or self._idx < self.lenmark:
                self._idx = idx
        else:
            self._idx = idx
    idx = property(get_idx, set_idx)

    def reset(self):
        if False:
            i = 10
            return i + 15
        ' Resets the internal buffer structure and the indices\n        '
        if self.mode == self.QBuffer:
            self.array = collections.deque(maxlen=self.maxlen + self.extrasize)
            self.useislice = True
        else:
            self.array = array.array(str('d'))
            self.useislice = False
        self.lencount = 0
        self.idx = -1
        self.extension = 0

    def qbuffer(self, savemem=0, extrasize=0):
        if False:
            print('Hello World!')
        self.mode = self.QBuffer
        self.maxlen = self._minperiod
        self.extrasize = extrasize
        self.lenmark = self.maxlen - (not self.extrasize)
        self.reset()

    def getindicators(self):
        if False:
            while True:
                i = 10
        return []

    def minbuffer(self, size):
        if False:
            while True:
                i = 10
        'The linebuffer must guarantee the minimum requested size to be\n        available.\n\n        In non-dqbuffer mode, this is always true (of course until data is\n        filled at the beginning, there are less values, but minperiod in the\n        framework should account for this.\n\n        In dqbuffer mode the buffer has to be adjusted for this if currently\n        less than requested\n        '
        if self.mode != self.QBuffer or self.maxlen >= size:
            return
        self.maxlen = size
        self.lenmark = self.maxlen - (not self.extrasize)
        self.reset()

    def __len__(self):
        if False:
            print('Hello World!')
        return self.lencount

    def buflen(self):
        if False:
            return 10
        ' Real data that can be currently held in the internal buffer\n\n        The internal buffer can be longer than the actual stored data to\n        allow for "lookahead" operations. The real amount of data that is\n        held/can be held in the buffer\n        is returned\n        '
        return len(self.array) - self.extension

    def __getitem__(self, ago):
        if False:
            for i in range(10):
                print('nop')
        return self.array[self.idx + ago]

    def get(self, ago=0, size=1):
        if False:
            return 10
        ' Returns a slice of the array relative to *ago*\n\n        Keyword Args:\n            ago (int): Point of the array to which size will be added\n            to return the slice size(int): size of the slice to return,\n            can be positive or negative\n\n        If size is positive *ago* will mark the end of the iterable and vice\n        versa if size is negative\n\n        Returns:\n            A slice of the underlying buffer\n        '
        if self.useislice:
            start = self.idx + ago - size + 1
            end = self.idx + ago + 1
            return list(islice(self.array, start, end))
        return self.array[self.idx + ago - size + 1:self.idx + ago + 1]

    def getzeroval(self, idx=0):
        if False:
            for i in range(10):
                print('nop')
        ' Returns a single value of the array relative to the real zero\n        of the buffer\n\n        Keyword Args:\n            idx (int): Where to start relative to the real start of the buffer\n            size(int): size of the slice to return\n\n        Returns:\n            A slice of the underlying buffer\n        '
        return self.array[idx]

    def getzero(self, idx=0, size=1):
        if False:
            return 10
        ' Returns a slice of the array relative to the real zero of the buffer\n\n        Keyword Args:\n            idx (int): Where to start relative to the real start of the buffer\n            size(int): size of the slice to return\n\n        Returns:\n            A slice of the underlying buffer\n        '
        if self.useislice:
            return list(islice(self.array, idx, idx + size))
        return self.array[idx:idx + size]

    def __setitem__(self, ago, value):
        if False:
            return 10
        ' Sets a value at position "ago" and executes any associated bindings\n\n        Keyword Args:\n            ago (int): Point of the array to which size will be added to return\n            the slice\n            value (variable): value to be set\n        '
        self.array[self.idx + ago] = value
        for binding in self.bindings:
            binding[ago] = value

    def set(self, value, ago=0):
        if False:
            return 10
        ' Sets a value at position "ago" and executes any associated bindings\n\n        Keyword Args:\n            value (variable): value to be set\n            ago (int): Point of the array to which size will be added to return\n            the slice\n        '
        self.array[self.idx + ago] = value
        for binding in self.bindings:
            binding[ago] = value

    def home(self):
        if False:
            i = 10
            return i + 15
        ' Rewinds the logical index to the beginning\n\n        The underlying buffer remains untouched and the actual len can be found\n        out with buflen\n        '
        self.idx = -1
        self.lencount = 0

    def forward(self, value=NAN, size=1):
        if False:
            i = 10
            return i + 15
        ' Moves the logical index foward and enlarges the buffer as much as needed\n\n        Keyword Args:\n            value (variable): value to be set in new positins\n            size (int): How many extra positions to enlarge the buffer\n        '
        self.idx += size
        self.lencount += size
        for i in range(size):
            self.array.append(value)

    def backwards(self, size=1, force=False):
        if False:
            while True:
                i = 10
        ' Moves the logical index backwards and reduces the buffer as much as needed\n\n        Keyword Args:\n            size (int): How many extra positions to rewind and reduce the\n            buffer\n        '
        self.set_idx(self._idx - size, force=force)
        self.lencount -= size
        for i in range(size):
            self.array.pop()

    def rewind(self, size=1):
        if False:
            print('Hello World!')
        self.idx -= size
        self.lencount -= size

    def advance(self, size=1):
        if False:
            while True:
                i = 10
        ' Advances the logical index without touching the underlying buffer\n\n        Keyword Args:\n            size (int): How many extra positions to move forward\n        '
        self.idx += size
        self.lencount += size

    def extend(self, value=NAN, size=0):
        if False:
            for i in range(10):
                print('nop')
        ' Extends the underlying array with positions that the index will not reach\n\n        Keyword Args:\n            value (variable): value to be set in new positins\n            size (int): How many extra positions to enlarge the buffer\n\n        The purpose is to allow for lookahead operations or to be able to\n        set values in the buffer "future"\n        '
        self.extension += size
        for i in range(size):
            self.array.append(value)

    def addbinding(self, binding):
        if False:
            for i in range(10):
                print('nop')
        ' Adds another line binding\n\n        Keyword Args:\n            binding (LineBuffer): another line that must be set when this line\n            becomes a value\n        '
        self.bindings.append(binding)
        binding.updateminperiod(self._minperiod)

    def plot(self, idx=0, size=None):
        if False:
            i = 10
            return i + 15
        ' Returns a slice of the array relative to the real zero of the buffer\n\n        Keyword Args:\n            idx (int): Where to start relative to the real start of the buffer\n            size(int): size of the slice to return\n\n        This is a variant of getzero which unless told otherwise returns the\n        entire buffer, which is usually the idea behind plottint (all must\n        plotted)\n\n        Returns:\n            A slice of the underlying buffer\n        '
        return self.getzero(idx, size or len(self))

    def plotrange(self, start, end):
        if False:
            return 10
        if self.useislice:
            return list(islice(self.array, start, end))
        return self.array[start:end]

    def oncebinding(self):
        if False:
            return 10
        '\n        Executes the bindings when running in "once" mode\n        '
        larray = self.array
        blen = self.buflen()
        for binding in self.bindings:
            binding.array[0:blen] = larray[0:blen]

    def bind2lines(self, binding=0):
        if False:
            return 10
        '\n        Stores a binding to another line. "binding" can be an index or a name\n        '
        if isinstance(binding, string_types):
            line = getattr(self._owner.lines, binding)
        else:
            line = self._owner.lines[binding]
        self.addbinding(line)
        return self
    bind2line = bind2lines

    def __call__(self, ago=None):
        if False:
            return 10
        'Returns either a delayed verison of itself in the form of a\n        LineDelay object or a timeframe adapting version with regards to a ago\n\n        Param: ago (default: None)\n\n          If ago is None or an instance of LineRoot (a lines object) the\n          returned valued is a LineCoupler instance\n\n          If ago is anything else, it is assumed to be an int and a LineDelay\n          object will be returned\n        '
        from .lineiterator import LineCoupler
        if ago is None or isinstance(ago, LineRoot):
            return LineCoupler(self, ago)
        return LineDelay(self, ago)

    def _makeoperation(self, other, operation, r=False, _ownerskip=None):
        if False:
            return 10
        return LinesOperation(self, other, operation, r=r, _ownerskip=_ownerskip)

    def _makeoperationown(self, operation, _ownerskip=None):
        if False:
            return 10
        return LineOwnOperation(self, operation, _ownerskip=_ownerskip)

    def _settz(self, tz):
        if False:
            return 10
        self._tz = tz

    def datetime(self, ago=0, tz=None, naive=True):
        if False:
            while True:
                i = 10
        return num2date(self.array[self.idx + ago], tz=tz or self._tz, naive=naive)

    def date(self, ago=0, tz=None, naive=True):
        if False:
            i = 10
            return i + 15
        return num2date(self.array[self.idx + ago], tz=tz or self._tz, naive=naive).date()

    def time(self, ago=0, tz=None, naive=True):
        if False:
            while True:
                i = 10
        return num2date(self.array[self.idx + ago], tz=tz or self._tz, naive=naive).time()

    def dt(self, ago=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        return numeric date part of datetimefloat\n        '
        return math.trunc(self.array[self.idx + ago])

    def tm_raw(self, ago=0):
        if False:
            i = 10
            return i + 15
        '\n        return raw numeric time part of datetimefloat\n        '
        return math.modf(self.array[self.idx + ago])[0]

    def tm(self, ago=0):
        if False:
            while True:
                i = 10
        '\n        return numeric time part of datetimefloat\n        '
        return time2num(num2date(self.array[self.idx + ago]).time())

    def tm_lt(self, other, ago=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        return numeric time part of datetimefloat\n        '
        dtime = self.array[self.idx + ago]
        (tm, dt) = math.modf(dtime)
        return dtime < dt + other

    def tm_le(self, other, ago=0):
        if False:
            print('Hello World!')
        '\n        return numeric time part of datetimefloat\n        '
        dtime = self.array[self.idx + ago]
        (tm, dt) = math.modf(dtime)
        return dtime <= dt + other

    def tm_eq(self, other, ago=0):
        if False:
            return 10
        '\n        return numeric time part of datetimefloat\n        '
        dtime = self.array[self.idx + ago]
        (tm, dt) = math.modf(dtime)
        return dtime == dt + other

    def tm_gt(self, other, ago=0):
        if False:
            print('Hello World!')
        '\n        return numeric time part of datetimefloat\n        '
        dtime = self.array[self.idx + ago]
        (tm, dt) = math.modf(dtime)
        return dtime > dt + other

    def tm_ge(self, other, ago=0):
        if False:
            print('Hello World!')
        '\n        return numeric time part of datetimefloat\n        '
        dtime = self.array[self.idx + ago]
        (tm, dt) = math.modf(dtime)
        return dtime >= dt + other

    def tm2dtime(self, tm, ago=0):
        if False:
            print('Hello World!')
        '\n        Returns the given ``tm`` in the frame of the (ago bars) datatime.\n\n        Useful for external comparisons to avoid precision errors\n        '
        return int(self.array[self.idx + ago]) + tm

    def tm2datetime(self, tm, ago=0):
        if False:
            i = 10
            return i + 15
        '\n        Returns the given ``tm`` in the frame of the (ago bars) datatime.\n\n        Useful for external comparisons to avoid precision errors\n        '
        return num2date(int(self.array[self.idx + ago]) + tm)

class MetaLineActions(LineBuffer.__class__):
    """
    Metaclass for Lineactions

    Scans the instance before init for LineBuffer (or parentclass LineSingle)
    instances to calculate the minperiod for this instance

    postinit it registers the instance to the owner (remember that owner has
    been found in the base Metaclass for LineRoot)
    """
    _acache = dict()
    _acacheuse = False

    @classmethod
    def cleancache(cls):
        if False:
            i = 10
            return i + 15
        cls._acache = dict()

    @classmethod
    def usecache(cls, onoff):
        if False:
            for i in range(10):
                print('nop')
        cls._acacheuse = onoff

    def __call__(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if not cls._acacheuse:
            return super(MetaLineActions, cls).__call__(*args, **kwargs)
        ckey = (cls, tuple(args), tuple(kwargs.items()))
        try:
            return cls._acache[ckey]
        except TypeError:
            return super(MetaLineActions, cls).__call__(*args, **kwargs)
        except KeyError:
            pass
        _obj = super(MetaLineActions, cls).__call__(*args, **kwargs)
        return cls._acache.setdefault(ckey, _obj)

    def dopreinit(cls, _obj, *args, **kwargs):
        if False:
            print('Hello World!')
        (_obj, args, kwargs) = super(MetaLineActions, cls).dopreinit(_obj, *args, **kwargs)
        _obj._clock = _obj._owner
        if isinstance(args[0], LineRoot):
            _obj._clock = args[0]
        _obj._datas = [x for x in args if isinstance(x, LineRoot)]
        _minperiods = [x._minperiod for x in args if isinstance(x, LineSingle)]
        mlines = [x.lines[0] for x in args if isinstance(x, LineMultiple)]
        _minperiods += [x._minperiod for x in mlines]
        _minperiod = max(_minperiods or [1])
        _obj.updateminperiod(_minperiod)
        return (_obj, args, kwargs)

    def dopostinit(cls, _obj, *args, **kwargs):
        if False:
            while True:
                i = 10
        (_obj, args, kwargs) = super(MetaLineActions, cls).dopostinit(_obj, *args, **kwargs)
        _obj._owner.addindicator(_obj)
        return (_obj, args, kwargs)

class PseudoArray(object):

    def __init__(self, wrapped):
        if False:
            for i in range(10):
                print('nop')
        self.wrapped = wrapped

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return self.wrapped

    @property
    def array(self):
        if False:
            for i in range(10):
                print('nop')
        return self

class LineActions(with_metaclass(MetaLineActions, LineBuffer)):
    """
    Base class derived from LineBuffer intented to defined the
    minimum interface to make it compatible with a LineIterator by
    providing operational _next and _once interfaces.

    The metaclass does the dirty job of calculating minperiods and registering
    """
    _ltype = LineBuffer.IndType

    def getindicators(self):
        if False:
            return 10
        return []

    def qbuffer(self, savemem=0):
        if False:
            i = 10
            return i + 15
        super(LineActions, self).qbuffer(savemem=savemem)
        for data in self._datas:
            data.minbuffer(size=self._minperiod)

    @staticmethod
    def arrayize(obj):
        if False:
            i = 10
            return i + 15
        if isinstance(obj, LineRoot):
            if not isinstance(obj, LineSingle):
                obj = obj.lines[0]
        else:
            obj = PseudoArray(obj)
        return obj

    def _next(self):
        if False:
            return 10
        clock_len = len(self._clock)
        if clock_len > len(self):
            self.forward()
        if clock_len > self._minperiod:
            self.next()
        elif clock_len == self._minperiod:
            self.nextstart()
        else:
            self.prenext()

    def _once(self):
        if False:
            while True:
                i = 10
        self.forward(size=self._clock.buflen())
        self.home()
        self.preonce(0, self._minperiod - 1)
        self.oncestart(self._minperiod - 1, self._minperiod)
        self.once(self._minperiod, self.buflen())
        self.oncebinding()

def LineDelay(a, ago=0, **kwargs):
    if False:
        while True:
            i = 10
    if ago <= 0:
        return _LineDelay(a, ago, **kwargs)
    return _LineForward(a, ago, **kwargs)

def LineNum(num):
    if False:
        i = 10
        return i + 15
    return LineDelay(PseudoArray(num))

class _LineDelay(LineActions):
    """
    Takes a LineBuffer (or derived) object and stores the value from
    "ago" periods effectively delaying the delivery of data
    """

    def __init__(self, a, ago):
        if False:
            return 10
        super(_LineDelay, self).__init__()
        self.a = a
        self.ago = ago
        self.addminperiod(abs(ago) + 1)

    def next(self):
        if False:
            while True:
                i = 10
        self[0] = self.a[self.ago]

    def once(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        dst = self.array
        src = self.a.array
        ago = self.ago
        for i in range(start, end):
            dst[i] = src[i + ago]

class _LineForward(LineActions):
    """
    Takes a LineBuffer (or derived) object and stores the value from
    "ago" periods from the future
    """

    def __init__(self, a, ago):
        if False:
            for i in range(10):
                print('nop')
        super(_LineForward, self).__init__()
        self.a = a
        self.ago = ago
        if ago > self.a._minperiod:
            self.addminperiod(ago - self.a._minperiod + 1)

    def next(self):
        if False:
            return 10
        self[-self.ago] = self.a[0]

    def once(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        dst = self.array
        src = self.a.array
        ago = self.ago
        for i in range(start, end):
            dst[i - ago] = src[i]

class LinesOperation(LineActions):
    """
    Holds an operation that operates on a two operands. Example: mul

    It will "next"/traverse the array applying the operation on the
    two operands and storing the result in self.

    To optimize the operations and avoid conditional checks the right
    next/once is chosen using the operation direction (normal or reversed)
    and the nature of the operands (LineBuffer vs non-LineBuffer)

    In the "once" operations "map" could be used as in:

        operated = map(self.operation, srca[start:end], srcb[start:end])
        self.array[start:end] = array.array(str(self.typecode), operated)

    No real execution time benefits were appreciated and therefore the loops
    have been kept in place for clarity (although the maps are not really
    unclear here)
    """

    def __init__(self, a, b, operation, r=False):
        if False:
            while True:
                i = 10
        super(LinesOperation, self).__init__()
        self.operation = operation
        self.a = a
        self.b = b
        self.r = r
        self.bline = isinstance(b, LineBuffer)
        self.btime = isinstance(b, datetime.time)
        self.bfloat = not self.bline and (not self.btime)
        if r:
            (self.a, self.b) = (b, a)

    def next(self):
        if False:
            i = 10
            return i + 15
        if self.bline:
            self[0] = self.operation(self.a[0], self.b[0])
        elif not self.r:
            if not self.btime:
                self[0] = self.operation(self.a[0], self.b)
            else:
                self[0] = self.operation(self.a.time(), self.b)
        else:
            self[0] = self.operation(self.a, self.b[0])

    def once(self, start, end):
        if False:
            while True:
                i = 10
        if self.bline:
            self._once_op(start, end)
        elif not self.r:
            if not self.btime:
                self._once_val_op(start, end)
            else:
                self._once_time_op(start, end)
        else:
            self._once_val_op_r(start, end)

    def _once_op(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        dst = self.array
        srca = self.a.array
        srcb = self.b.array
        op = self.operation
        for i in range(start, end):
            dst[i] = op(srca[i], srcb[i])

    def _once_time_op(self, start, end):
        if False:
            return 10
        dst = self.array
        srca = self.a.array
        srcb = self.b
        op = self.operation
        tz = self._tz
        for i in range(start, end):
            dst[i] = op(num2date(srca[i], tz=tz).time(), srcb)

    def _once_val_op(self, start, end):
        if False:
            return 10
        dst = self.array
        srca = self.a.array
        srcb = self.b
        op = self.operation
        for i in range(start, end):
            dst[i] = op(srca[i], srcb)

    def _once_val_op_r(self, start, end):
        if False:
            while True:
                i = 10
        dst = self.array
        srca = self.a
        srcb = self.b.array
        op = self.operation
        for i in range(start, end):
            dst[i] = op(srca, srcb[i])

class LineOwnOperation(LineActions):
    """
    Holds an operation that operates on a single operand. Example: abs

    It will "next"/traverse the array applying the operation and storing
    the result in self
    """

    def __init__(self, a, operation):
        if False:
            while True:
                i = 10
        super(LineOwnOperation, self).__init__()
        self.operation = operation
        self.a = a

    def next(self):
        if False:
            for i in range(10):
                print('nop')
        self[0] = self.operation(self.a[0])

    def once(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        dst = self.array
        srca = self.a.array
        op = self.operation
        for i in range(start, end):
            dst[i] = op(srca[i])