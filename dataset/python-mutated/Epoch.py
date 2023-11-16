"""Epoch module."""
import functools
import operator
import math
import datetime as DT
from matplotlib import _api
from matplotlib.dates import date2num

class Epoch:
    allowed = {'ET': {'UTC': +64.1839}, 'UTC': {'ET': -64.1839}}

    def __init__(self, frame, sec=None, jd=None, daynum=None, dt=None):
        if False:
            i = 10
            return i + 15
        "\n        Create a new Epoch object.\n\n        Build an epoch 1 of 2 ways:\n\n        Using seconds past a Julian date:\n        #   Epoch('ET', sec=1e8, jd=2451545)\n\n        or using a matplotlib day number\n        #   Epoch('ET', daynum=730119.5)\n\n        = ERROR CONDITIONS\n        - If the input units are not in the allowed list, an error is thrown.\n\n        = INPUT VARIABLES\n        - frame     The frame of the epoch.  Must be 'ET' or 'UTC'\n        - sec        The number of seconds past the input JD.\n        - jd         The Julian date of the epoch.\n        - daynum    The matplotlib day number of the epoch.\n        - dt         A python datetime instance.\n        "
        if sec is None and jd is not None or (sec is not None and jd is None) or (daynum is not None and (sec is not None or jd is not None)) or (daynum is None and dt is None and (sec is None or jd is None)) or (daynum is not None and dt is not None) or (dt is not None and (sec is not None or jd is not None)) or (dt is not None and (not isinstance(dt, DT.datetime))):
            raise ValueError('Invalid inputs.  Must enter sec and jd together, daynum by itself, or dt (must be a python datetime).\nSec = %s\nJD  = %s\ndnum= %s\ndt  = %s' % (sec, jd, daynum, dt))
        _api.check_in_list(self.allowed, frame=frame)
        self._frame = frame
        if dt is not None:
            daynum = date2num(dt)
        if daynum is not None:
            jd = float(daynum) + 1721425.5
            self._jd = math.floor(jd)
            self._seconds = (jd - self._jd) * 86400.0
        else:
            self._seconds = float(sec)
            self._jd = float(jd)
            deltaDays = math.floor(self._seconds / 86400)
            self._jd += deltaDays
            self._seconds -= deltaDays * 86400.0

    def convert(self, frame):
        if False:
            return 10
        if self._frame == frame:
            return self
        offset = self.allowed[self._frame][frame]
        return Epoch(frame, self._seconds + offset, self._jd)

    def frame(self):
        if False:
            i = 10
            return i + 15
        return self._frame

    def julianDate(self, frame):
        if False:
            while True:
                i = 10
        t = self
        if frame != self._frame:
            t = self.convert(frame)
        return t._jd + t._seconds / 86400.0

    def secondsPast(self, frame, jd):
        if False:
            print('Hello World!')
        t = self
        if frame != self._frame:
            t = self.convert(frame)
        delta = t._jd - jd
        return t._seconds + delta * 86400

    def _cmp(self, op, rhs):
        if False:
            i = 10
            return i + 15
        'Compare Epochs *self* and *rhs* using operator *op*.'
        t = self
        if self._frame != rhs._frame:
            t = self.convert(rhs._frame)
        if t._jd != rhs._jd:
            return op(t._jd, rhs._jd)
        return op(t._seconds, rhs._seconds)
    __eq__ = functools.partialmethod(_cmp, operator.eq)
    __ne__ = functools.partialmethod(_cmp, operator.ne)
    __lt__ = functools.partialmethod(_cmp, operator.lt)
    __le__ = functools.partialmethod(_cmp, operator.le)
    __gt__ = functools.partialmethod(_cmp, operator.gt)
    __ge__ = functools.partialmethod(_cmp, operator.ge)

    def __add__(self, rhs):
        if False:
            while True:
                i = 10
        '\n        Add a duration to an Epoch.\n\n        = INPUT VARIABLES\n        - rhs     The Epoch to subtract.\n\n        = RETURN VALUE\n        - Returns the difference of ourselves and the input Epoch.\n        '
        t = self
        if self._frame != rhs.frame():
            t = self.convert(rhs._frame)
        sec = t._seconds + rhs.seconds()
        return Epoch(t._frame, sec, t._jd)

    def __sub__(self, rhs):
        if False:
            return 10
        "\n        Subtract two Epoch's or a Duration from an Epoch.\n\n        Valid:\n        Duration = Epoch - Epoch\n        Epoch = Epoch - Duration\n\n        = INPUT VARIABLES\n        - rhs     The Epoch to subtract.\n\n        = RETURN VALUE\n        - Returns either the duration between to Epoch's or the a new\n          Epoch that is the result of subtracting a duration from an epoch.\n        "
        import matplotlib.testing.jpl_units as U
        if isinstance(rhs, U.Duration):
            return self + -rhs
        t = self
        if self._frame != rhs._frame:
            t = self.convert(rhs._frame)
        days = t._jd - rhs._jd
        sec = t._seconds - rhs._seconds
        return U.Duration(rhs._frame, days * 86400 + sec)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Print the Epoch.'
        return f'{self.julianDate(self._frame):22.15e} {self._frame}'

    def __repr__(self):
        if False:
            return 10
        'Print the Epoch.'
        return str(self)

    @staticmethod
    def range(start, stop, step):
        if False:
            return 10
        '\n        Generate a range of Epoch objects.\n\n        Similar to the Python range() method.  Returns the range [\n        start, stop) at the requested step.  Each element will be a\n        Epoch object.\n\n        = INPUT VARIABLES\n        - start     The starting value of the range.\n        - stop      The stop value of the range.\n        - step      Step to use.\n\n        = RETURN VALUE\n        - Returns a list containing the requested Epoch values.\n        '
        elems = []
        i = 0
        while True:
            d = start + i * step
            if d >= stop:
                break
            elems.append(d)
            i += 1
        return elems