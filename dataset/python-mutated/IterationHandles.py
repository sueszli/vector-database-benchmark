""" Node for Iteration Handles.

"""
import math
from abc import abstractmethod
from nuitka.__past__ import xrange
from nuitka.utils.SlotMetaClasses import getMetaClassBase

class IterationHandleBase(getMetaClassBase('IterationHandle', require_slots=True)):
    """Base class for Iteration Handles."""
    __slots__ = ()

    @abstractmethod
    def getNextValueExpression(self):
        if False:
            for i in range(10):
                print('nop')
        'Abstract method to get next iteration value.'

    @abstractmethod
    def getIterationValueWithIndex(self, value_index):
        if False:
            i = 10
            return i + 15
        'Abstract method for random access of the expression.'

    def getNextValueTruth(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns truth value of the next expression or Stops the\n        iteration handle if end is reached.\n        '
        iteration_value = self.getNextValueExpression()
        if iteration_value is None:
            return StopIteration
        return iteration_value.getTruthValue()

    def getAllElementTruthValue(self):
        if False:
            i = 10
            return i + 15
        "Returns truth value for 'all' on 'lists'. It returns\n        True: if all the elements of the list are True,\n        False: if any element in the list is False,\n        None: if number of elements in the list is greater than\n        256 or any element is Unknown.\n        "
        all_true = True
        count = 0
        while True:
            truth_value = self.getNextValueTruth()
            if truth_value is StopIteration:
                break
            if count > 256:
                return None
            if truth_value is False:
                return False
            if truth_value is None:
                all_true = None
            count += 1
        return all_true

class ConstantIterationHandleBase(IterationHandleBase):
    """Base class for the Constant Iteration Handles.

    Attributes
    ----------
    constant_node : node_object
        Instance of the calling node.

    Methods
    -------
    __repr__()
        Prints representation of the ConstantIterationHandleBase
        and it's children objects
    getNextValueExpression()
        Returns the next iteration value
    getNextValueTruth()
        Returns the boolean value of the next handle
    """
    __slots__ = ('constant_node', 'iter')

    def __init__(self, constant_node):
        if False:
            print('Hello World!')
        assert constant_node.isIterableConstant()
        self.constant_node = constant_node
        self.iter = iter(self.constant_node.constant)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<%s of %r>' % (self.__class__.__name__, self.constant_node)

    def getNextValueExpression(self):
        if False:
            while True:
                i = 10
        'Returns truth value of the next expression or Stops the iteration handle\n        and returns None if end is reached.\n        '
        try:
            from .ConstantRefNodes import makeConstantRefNode
            return makeConstantRefNode(constant=next(self.iter), source_ref=self.constant_node.source_ref)
        except StopIteration:
            return None

    def getNextValueTruth(self):
        if False:
            print('Hello World!')
        'Return the truth value of the next iteration value or StopIteration.'
        try:
            iteration_value = next(self.iter)
        except StopIteration:
            return StopIteration
        return bool(iteration_value)

    def getIterationValueWithIndex(self, value_index):
        if False:
            return 10
        return None

class ConstantIndexableIterationHandle(ConstantIterationHandleBase):
    """Class for the constants that are indexable.

    Attributes
    ----------
    constant_node : node_object
        Instance of the calling node.

    Methods
    -------
    getIterationValueWithIndex(value_index)
        Sequential access of the constants
    """
    __slots__ = ()

    def getIterationValueWithIndex(self, value_index):
        if False:
            i = 10
            return i + 15
        'Tries to return constant value at the given index.\n\n        Parameters\n        ----------\n        value_index : int\n            Index value of the element to be returned\n        '
        try:
            from .ConstantRefNodes import makeConstantRefNode
            return makeConstantRefNode(constant=self.constant_node.constant[value_index], source_ref=self.constant_node.source_ref)
        except IndexError:
            return None

class ConstantTupleIterationHandle(ConstantIndexableIterationHandle):
    __slots__ = ()

class ConstantListIterationHandle(ConstantIndexableIterationHandle):
    __slots__ = ()

class ConstantStrIterationHandle(ConstantIndexableIterationHandle):
    __slots__ = ()

class ConstantUnicodeIterationHandle(ConstantIndexableIterationHandle):
    __slots__ = ()

class ConstantBytesIterationHandle(ConstantIndexableIterationHandle):
    __slots__ = ()

class ConstantBytearrayIterationHandle(ConstantIndexableIterationHandle):
    __slots__ = ()

class ConstantRangeIterationHandle(ConstantIndexableIterationHandle):
    __slots__ = ()

class ConstantSetAndDictIterationHandleBase(ConstantIterationHandleBase):
    """Class for the set and dictionary constants."""
    __slots__ = ()

class ConstantSetIterationHandle(ConstantSetAndDictIterationHandleBase):
    __slots__ = ()

class ConstantFrozensetIterationHandle(ConstantSetAndDictIterationHandleBase):
    __slots__ = ()

class ConstantDictIterationHandle(ConstantSetAndDictIterationHandleBase):
    __slots__ = ()

class ListAndTupleContainerMakingIterationHandle(IterationHandleBase):
    """Class for list and tuple container making expression

    Attributes
    ----------
    constant_node : node_object
        Instance of the calling node.

    Methods
    -------
    __repr__()
        Prints representation of the ListAndTupleContainerMakingIterationHandle
        object
    getNextValueExpression()
        Returns the next iteration value
    getNextValueTruth()
        Returns the boolean value of the next handle
    getIterationValueWithIndex(value_index)
        Sequential access of the expression
    """
    __slots__ = ('elements', 'iter')

    def __init__(self, elements):
        if False:
            return 10
        self.elements = elements
        self.iter = iter(self.elements)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<%s of %r>' % (self.__class__.__name__, self.elements)

    def getNextValueExpression(self):
        if False:
            while True:
                i = 10
        'Return the next iteration value or StopIteration exception\n        if the iteration has reached the end\n        '
        try:
            return next(self.iter)
        except StopIteration:
            return None

    def getIterationValueWithIndex(self, value_index):
        if False:
            while True:
                i = 10
        'Tries to return constant value at the given index.\n\n        Parameters\n        ----------\n        value_index : int\n            Index value of the element to be returned\n        '
        try:
            return self.elements[value_index]
        except IndexError:
            return None

class RangeIterationHandleBase(IterationHandleBase):
    """Iteration handle class for range nodes

    Attributes
    ----------
    low : int
        Optional. An integer number specifying at which position to start. Default is 0
    high : int
        Optional. An integer number specifying at which position to end.
    step : int
        Optional. An integer number specifying the increment. Default is 1
    """
    step = 1
    __slots__ = ('low', 'iter', 'source_ref')

    def __init__(self, low_value, range_value, source_ref):
        if False:
            return 10
        self.low = low_value
        self.iter = iter(range_value)
        self.source_ref = source_ref

    def getNextValueExpression(self):
        if False:
            i = 10
            return i + 15
        'Return the next iteration value or StopIteration exception\n        if the iteration has reached the end\n        '
        try:
            from .ConstantRefNodes import makeConstantRefNode
            return makeConstantRefNode(constant=next(self.iter), source_ref=self.source_ref)
        except StopIteration:
            return None

    @abstractmethod
    def getIterationLength(self):
        if False:
            i = 10
            return i + 15
        'return length'

    def getIterationValueWithIndex(self, value_index):
        if False:
            return 10
        'Tries to return constant value at the given index.\n\n        Parameters\n        ----------\n        value_index : int\n            Index value of the element to be returned\n        '
        if value_index < self.getIterationLength():
            from .ConstantRefNodes import makeConstantRefNode
            return makeConstantRefNode(constant=value_index * self.step + self.low, source_ref=self.source_ref)
        else:
            return IndexError

    def getNextValueTruth(self):
        if False:
            print('Hello World!')
        'Return the boolean value of the next iteration handle.'
        try:
            iteration_value = next(self.iter)
        except StopIteration:
            return StopIteration
        return bool(iteration_value)

    @staticmethod
    def getAllElementTruthValue():
        if False:
            while True:
                i = 10
        return True

class IterationHandleRange1(RangeIterationHandleBase):
    """Iteration handle for range(low,)"""
    __slots__ = ()

    def __init__(self, low_value, source_ref):
        if False:
            return 10
        RangeIterationHandleBase.__init__(self, low_value, xrange(low_value), source_ref)

    def getIterationLength(self):
        if False:
            i = 10
            return i + 15
        return max(0, self.low)

    @staticmethod
    def getAllElementTruthValue():
        if False:
            while True:
                i = 10
        return False

class IterationHandleRange2(RangeIterationHandleBase):
    """Iteration handle for ranges(low, high)"""
    __slots__ = ('high',)

    def __init__(self, low_value, high_value, source_ref):
        if False:
            i = 10
            return i + 15
        RangeIterationHandleBase.__init__(self, low_value, xrange(low_value, high_value), source_ref)
        self.high = high_value

    def getIterationLength(self):
        if False:
            i = 10
            return i + 15
        return max(0, self.high - self.low)

class IterationHandleRange3(RangeIterationHandleBase):
    """Iteration handle for ranges(low, high, step)"""
    __slots__ = ('high', 'step')

    def __init__(self, low_value, high_value, step_value, source_ref):
        if False:
            print('Hello World!')
        RangeIterationHandleBase.__init__(self, low_value, xrange(low_value, high_value, step_value), source_ref)
        self.high = high_value
        self.step = step_value

    def getIterationLength(self):
        if False:
            while True:
                i = 10
        if self.low < self.high:
            if self.step < 0:
                estimate = 0
            else:
                estimate = math.ceil(float(self.high - self.low) / self.step)
        elif self.step > 0:
            estimate = 0
        else:
            estimate = math.ceil(float(self.high - self.low) / self.step)
        assert estimate >= 0
        return int(estimate)