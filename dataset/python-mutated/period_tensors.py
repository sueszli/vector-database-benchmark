"""PeriodTensor definition."""
import tensorflow.compat.v2 as tf
from tf_quant_finance.datetime import constants
from tf_quant_finance.datetime import tensor_wrapper

def day():
    if False:
        while True:
            i = 10
    return days(1)

def days(n):
    if False:
        print('Hello World!')
    return PeriodTensor(n, constants.PeriodType.DAY)

def week():
    if False:
        i = 10
        return i + 15
    return weeks(1)

def weeks(n):
    if False:
        i = 10
        return i + 15
    return PeriodTensor(n, constants.PeriodType.WEEK)

def month():
    if False:
        return 10
    return months(1)

def months(n):
    if False:
        print('Hello World!')
    return PeriodTensor(n, constants.PeriodType.MONTH)

def year():
    if False:
        for i in range(10):
            print('nop')
    return years(1)

def years(n):
    if False:
        i = 10
        return i + 15
    return PeriodTensor(n, constants.PeriodType.YEAR)

class PeriodTensor(tensor_wrapper.TensorWrapper):
    """Represents a tensor of time periods."""

    def __init__(self, quantity, period_type):
        if False:
            print('Hello World!')
        'Initializer.\n\n    Args:\n      quantity: A Tensor of type tf.int32, representing the quantities\n        of period types (e.g. how many months). Can be both positive and\n        negative.\n      period_type: A PeriodType (a day, a month, etc). Currently only one\n        PeriodType per instance of PeriodTensor is supported.\n\n    Example:\n    ```python\n    two_weeks = PeriodTensor(2, PeriodType.WEEK)\n\n    months = [3, 6, 9, 12]\n    periods = PeriodTensor(months, PeriodType.MONTH)\n    ```\n    '
        self._quantity = tf.convert_to_tensor(quantity, dtype=tf.int32, name='pt_quantity')
        self._period_type = period_type

    def period_type(self):
        if False:
            while True:
                i = 10
        'Returns the PeriodType of this PeriodTensor.'
        return self._period_type

    def quantity(self):
        if False:
            print('Hello World!')
        'Returns the quantities tensor of this PeriodTensor.'
        return self._quantity

    def __mul__(self, multiplier):
        if False:
            i = 10
            return i + 15
        'Multiplies PeriodTensor by a Tensor of ints.'
        multiplier = tf.convert_to_tensor(multiplier, tf.int32)
        return PeriodTensor(self._quantity * multiplier, self._period_type)

    def __add__(self, other):
        if False:
            while True:
                i = 10
        'Adds another PeriodTensor of the same type.'
        if other.period_type() != self._period_type:
            raise ValueError('Mixing different period types is not supported')
        return PeriodTensor(self._quantity + other.quantity(), self._period_type)

    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Subtracts another PeriodTensor of the same type.'
        if other.period_type() != self._period_type:
            raise ValueError('Mixing different period types is not supported')
        return PeriodTensor(self._quantity - other.quantity(), self._period_type)

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        return self._quantity.shape

    @property
    def rank(self):
        if False:
            print('Hello World!')
        return tf.rank(self._quantity)

    @classmethod
    def _apply_sequence_to_tensor_op(cls, op_fn, tensor_wrappers):
        if False:
            print('Hello World!')
        q = op_fn([t.quantity() for t in tensor_wrappers])
        period_type = tensor_wrappers[0].period_type()
        if not all((t.period_type() == period_type for t in tensor_wrappers[1:])):
            raise ValueError('Combined PeriodTensors must have the same PeriodType')
        return PeriodTensor(q, period_type)

    def _apply_op(self, op_fn):
        if False:
            i = 10
            return i + 15
        q = op_fn(self._quantity)
        return PeriodTensor(q, self._period_type)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        output = 'PeriodTensor: shape={}'.format(self.shape)
        if tf.executing_eagerly():
            return output + ', quantities={}'.format(repr(self._quantity.numpy()))
        return output
__all__ = ['day', 'days', 'month', 'months', 'week', 'weeks', 'year', 'years', 'PeriodTensor']