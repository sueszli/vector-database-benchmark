"""Duration module."""
import functools
import operator
from matplotlib import _api

class Duration:
    """Class Duration in development."""
    allowed = ['ET', 'UTC']

    def __init__(self, frame, seconds):
        if False:
            for i in range(10):
                print('nop')
        "\n        Create a new Duration object.\n\n        = ERROR CONDITIONS\n        - If the input frame is not in the allowed list, an error is thrown.\n\n        = INPUT VARIABLES\n        - frame     The frame of the duration.  Must be 'ET' or 'UTC'\n        - seconds  The number of seconds in the Duration.\n        "
        _api.check_in_list(self.allowed, frame=frame)
        self._frame = frame
        self._seconds = seconds

    def frame(self):
        if False:
            i = 10
            return i + 15
        'Return the frame the duration is in.'
        return self._frame

    def __abs__(self):
        if False:
            print('Hello World!')
        'Return the absolute value of the duration.'
        return Duration(self._frame, abs(self._seconds))

    def __neg__(self):
        if False:
            return 10
        'Return the negative value of this Duration.'
        return Duration(self._frame, -self._seconds)

    def seconds(self):
        if False:
            print('Hello World!')
        'Return the number of seconds in the Duration.'
        return self._seconds

    def __bool__(self):
        if False:
            print('Hello World!')
        return self._seconds != 0

    def _cmp(self, op, rhs):
        if False:
            while True:
                i = 10
        '\n        Check that *self* and *rhs* share frames; compare them using *op*.\n        '
        self.checkSameFrame(rhs, 'compare')
        return op(self._seconds, rhs._seconds)
    __eq__ = functools.partialmethod(_cmp, operator.eq)
    __ne__ = functools.partialmethod(_cmp, operator.ne)
    __lt__ = functools.partialmethod(_cmp, operator.lt)
    __le__ = functools.partialmethod(_cmp, operator.le)
    __gt__ = functools.partialmethod(_cmp, operator.gt)
    __ge__ = functools.partialmethod(_cmp, operator.ge)

    def __add__(self, rhs):
        if False:
            return 10
        '\n        Add two Durations.\n\n        = ERROR CONDITIONS\n        - If the input rhs is not in the same frame, an error is thrown.\n\n        = INPUT VARIABLES\n        - rhs     The Duration to add.\n\n        = RETURN VALUE\n        - Returns the sum of ourselves and the input Duration.\n        '
        import matplotlib.testing.jpl_units as U
        if isinstance(rhs, U.Epoch):
            return rhs + self
        self.checkSameFrame(rhs, 'add')
        return Duration(self._frame, self._seconds + rhs._seconds)

    def __sub__(self, rhs):
        if False:
            print('Hello World!')
        '\n        Subtract two Durations.\n\n        = ERROR CONDITIONS\n        - If the input rhs is not in the same frame, an error is thrown.\n\n        = INPUT VARIABLES\n        - rhs     The Duration to subtract.\n\n        = RETURN VALUE\n        - Returns the difference of ourselves and the input Duration.\n        '
        self.checkSameFrame(rhs, 'sub')
        return Duration(self._frame, self._seconds - rhs._seconds)

    def __mul__(self, rhs):
        if False:
            print('Hello World!')
        '\n        Scale a UnitDbl by a value.\n\n        = INPUT VARIABLES\n        - rhs     The scalar to multiply by.\n\n        = RETURN VALUE\n        - Returns the scaled Duration.\n        '
        return Duration(self._frame, self._seconds * float(rhs))
    __rmul__ = __mul__

    def __str__(self):
        if False:
            print('Hello World!')
        'Print the Duration.'
        return f'{self._seconds:g} {self._frame}'

    def __repr__(self):
        if False:
            return 10
        'Print the Duration.'
        return f"Duration('{self._frame}', {self._seconds:g})"

    def checkSameFrame(self, rhs, func):
        if False:
            return 10
        '\n        Check to see if frames are the same.\n\n        = ERROR CONDITIONS\n        - If the frame of the rhs Duration is not the same as our frame,\n          an error is thrown.\n\n        = INPUT VARIABLES\n        - rhs     The Duration to check for the same frame\n        - func    The name of the function doing the check.\n        '
        if self._frame != rhs._frame:
            raise ValueError(f'Cannot {func} Durations with different frames.\nLHS: {self._frame}\nRHS: {rhs._frame}')