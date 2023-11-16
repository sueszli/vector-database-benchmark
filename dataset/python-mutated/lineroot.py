"""

.. module:: lineroot

Definition of the base class LineRoot and base classes LineSingle/LineMultiple
to define interfaces and hierarchy for the real operational classes

.. moduleauthor:: Daniel Rodriguez

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import operator
from .utils.py3 import range, with_metaclass
from . import metabase

class MetaLineRoot(metabase.MetaParams):
    """
    Once the object is created (effectively pre-init) the "owner" of this
    class is sought
    """

    def donew(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        (_obj, args, kwargs) = super(MetaLineRoot, cls).donew(*args, **kwargs)
        ownerskip = kwargs.pop('_ownerskip', None)
        _obj._owner = metabase.findowner(_obj, _obj._OwnerCls or LineMultiple, skip=ownerskip)
        return (_obj, args, kwargs)

class LineRoot(with_metaclass(MetaLineRoot, object)):
    """
    Defines a common base and interfaces for Single and Multiple
    LineXXX instances

        Period management
        Iteration management
        Operation (dual/single operand) Management
        Rich Comparison operator definition
    """
    _OwnerCls = None
    _minperiod = 1
    _opstage = 1
    (IndType, StratType, ObsType) = range(3)

    def _stage1(self):
        if False:
            for i in range(10):
                print('nop')
        self._opstage = 1

    def _stage2(self):
        if False:
            for i in range(10):
                print('nop')
        self._opstage = 2

    def _operation(self, other, operation, r=False, intify=False):
        if False:
            for i in range(10):
                print('nop')
        if self._opstage == 1:
            return self._operation_stage1(other, operation, r=r, intify=intify)
        return self._operation_stage2(other, operation, r=r)

    def _operationown(self, operation):
        if False:
            while True:
                i = 10
        if self._opstage == 1:
            return self._operationown_stage1(operation)
        return self._operationown_stage2(operation)

    def qbuffer(self, savemem=0):
        if False:
            while True:
                i = 10
        'Change the lines to implement a minimum size qbuffer scheme'
        raise NotImplementedError

    def minbuffer(self, size):
        if False:
            i = 10
            return i + 15
        'Receive notification of how large the buffer must at least be'
        raise NotImplementedError

    def setminperiod(self, minperiod):
        if False:
            i = 10
            return i + 15
        '\n        Direct minperiod manipulation. It could be used for example\n        by a strategy\n        to not wait for all indicators to produce a value\n        '
        self._minperiod = minperiod

    def updateminperiod(self, minperiod):
        if False:
            return 10
        "\n        Update the minperiod if needed. The minperiod will have been\n        calculated elsewhere\n        and has to take over if greater that self's\n        "
        self._minperiod = max(self._minperiod, minperiod)

    def addminperiod(self, minperiod):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a minperiod to own ... to be defined by subclasses\n        '
        raise NotImplementedError

    def incminperiod(self, minperiod):
        if False:
            return 10
        '\n        Increment the minperiod with no considerations\n        '
        raise NotImplementedError

    def prenext(self):
        if False:
            i = 10
            return i + 15
        '\n        It will be called during the "minperiod" phase of an iteration.\n        '
        pass

    def nextstart(self):
        if False:
            return 10
        '\n        It will be called when the minperiod phase is over for the 1st\n        post-minperiod value. Only called once and defaults to automatically\n        calling next\n        '
        self.next()

    def next(self):
        if False:
            print('Hello World!')
        '\n        Called to calculate values when the minperiod is over\n        '
        pass

    def preonce(self, start, end):
        if False:
            i = 10
            return i + 15
        '\n        It will be called during the "minperiod" phase of a "once" iteration\n        '
        pass

    def oncestart(self, start, end):
        if False:
            while True:
                i = 10
        '\n        It will be called when the minperiod phase is over for the 1st\n        post-minperiod value\n\n        Only called once and defaults to automatically calling once\n        '
        self.once(start, end)

    def once(self, start, end):
        if False:
            return 10
        '\n        Called to calculate values at "once" when the minperiod is over\n        '
        pass

    def _makeoperation(self, other, operation, r=False, _ownerskip=None):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def _makeoperationown(self, operation, _ownerskip=None):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def _operationown_stage1(self, operation):
        if False:
            while True:
                i = 10
        '\n        Operation with single operand which is "self"\n        '
        return self._makeoperationown(operation, _ownerskip=self)

    def _roperation(self, other, operation, intify=False):
        if False:
            print('Hello World!')
        '\n        Relies on self._operation to and passes "r" True to define a\n        reverse operation\n        '
        return self._operation(other, operation, r=True, intify=intify)

    def _operation_stage1(self, other, operation, r=False, intify=False):
        if False:
            return 10
        "\n        Two operands' operation. Scanning of other happens to understand\n        if other must be directly an operand or rather a subitem thereof\n        "
        if isinstance(other, LineMultiple):
            other = other.lines[0]
        return self._makeoperation(other, operation, r, self)

    def _operation_stage2(self, other, operation, r=False):
        if False:
            i = 10
            return i + 15
        '\n        Rich Comparison operators. Scans other and returns either an\n        operation with other directly or a subitem from other\n        '
        if isinstance(other, LineRoot):
            other = other[0]
        if r:
            return operation(other, self[0])
        return operation(self[0], other)

    def _operationown_stage2(self, operation):
        if False:
            while True:
                i = 10
        return operation(self[0])

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._operation(other, operator.__add__)

    def __radd__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._roperation(other, operator.__add__)

    def __sub__(self, other):
        if False:
            i = 10
            return i + 15
        return self._operation(other, operator.__sub__)

    def __rsub__(self, other):
        if False:
            while True:
                i = 10
        return self._roperation(other, operator.__sub__)

    def __mul__(self, other):
        if False:
            while True:
                i = 10
        return self._operation(other, operator.__mul__)

    def __rmul__(self, other):
        if False:
            return 10
        return self._roperation(other, operator.__mul__)

    def __div__(self, other):
        if False:
            i = 10
            return i + 15
        return self._operation(other, operator.__div__)

    def __rdiv__(self, other):
        if False:
            print('Hello World!')
        return self._roperation(other, operator.__div__)

    def __floordiv__(self, other):
        if False:
            return 10
        return self._operation(other, operator.__floordiv__)

    def __rfloordiv__(self, other):
        if False:
            return 10
        return self._roperation(other, operator.__floordiv__)

    def __truediv__(self, other):
        if False:
            print('Hello World!')
        return self._operation(other, operator.__truediv__)

    def __rtruediv__(self, other):
        if False:
            while True:
                i = 10
        return self._roperation(other, operator.__truediv__)

    def __pow__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._operation(other, operator.__pow__)

    def __rpow__(self, other):
        if False:
            print('Hello World!')
        return self._roperation(other, operator.__pow__)

    def __abs__(self):
        if False:
            return 10
        return self._operationown(operator.__abs__)

    def __neg__(self):
        if False:
            print('Hello World!')
        return self._operationown(operator.__neg__)

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        return self._operation(other, operator.__lt__)

    def __gt__(self, other):
        if False:
            print('Hello World!')
        return self._operation(other, operator.__gt__)

    def __le__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._operation(other, operator.__le__)

    def __ge__(self, other):
        if False:
            return 10
        return self._operation(other, operator.__ge__)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return self._operation(other, operator.__eq__)

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._operation(other, operator.__ne__)

    def __nonzero__(self):
        if False:
            while True:
                i = 10
        return self._operationown(bool)
    __bool__ = __nonzero__
    __hash__ = object.__hash__

class LineMultiple(LineRoot):
    """
    Base class for LineXXX instances that hold more than one line
    """

    def reset(self):
        if False:
            return 10
        self._stage1()
        self.lines.reset()

    def _stage1(self):
        if False:
            i = 10
            return i + 15
        super(LineMultiple, self)._stage1()
        for line in self.lines:
            line._stage1()

    def _stage2(self):
        if False:
            while True:
                i = 10
        super(LineMultiple, self)._stage2()
        for line in self.lines:
            line._stage2()

    def addminperiod(self, minperiod):
        if False:
            return 10
        '\n        The passed minperiod is fed to the lines\n        '
        for line in self.lines:
            line.addminperiod(minperiod)

    def incminperiod(self, minperiod):
        if False:
            while True:
                i = 10
        '\n        The passed minperiod is fed to the lines\n        '
        for line in self.lines:
            line.incminperiod(minperiod)

    def _makeoperation(self, other, operation, r=False, _ownerskip=None):
        if False:
            for i in range(10):
                print('nop')
        return self.lines[0]._makeoperation(other, operation, r, _ownerskip)

    def _makeoperationown(self, operation, _ownerskip=None):
        if False:
            while True:
                i = 10
        return self.lines[0]._makeoperationown(operation, _ownerskip)

    def qbuffer(self, savemem=0):
        if False:
            print('Hello World!')
        for line in self.lines:
            line.qbuffer(savemem=1)

    def minbuffer(self, size):
        if False:
            i = 10
            return i + 15
        for line in self.lines:
            line.minbuffer(size)

class LineSingle(LineRoot):
    """
    Base class for LineXXX instances that hold a single line
    """

    def addminperiod(self, minperiod):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add the minperiod (substracting the overlapping 1 minimum period)\n        '
        self._minperiod += minperiod - 1

    def incminperiod(self, minperiod):
        if False:
            for i in range(10):
                print('nop')
        '\n        Increment the minperiod with no considerations\n        '
        self._minperiod += minperiod