"""HolidayCalendar definition."""
import abc
from tf_quant_finance.datetime import constants

class HolidayCalendar(abc.ABC):
    """Represents a holiday calendar.

  Provides methods for manipulating the dates taking into account the holidays,
  and the business day roll conventions. Weekends are treated as holidays.
  """

    @abc.abstractmethod
    def is_business_day(self, date_tensor):
        if False:
            while True:
                i = 10
        'Returns a tensor of bools for whether given dates are business days.'
        pass

    @abc.abstractmethod
    def roll_to_business_day(self, date_tensor, roll_convention):
        if False:
            for i in range(10):
                print('nop')
        'Rolls the given dates to business dates according to given convention.\n\n    Args:\n      date_tensor: DateTensor of dates to roll from.\n      roll_convention: BusinessDayConvention. Determines how to roll a date that\n        falls on a holiday.\n\n    Returns:\n      The resulting DateTensor.\n    '
        pass

    @abc.abstractmethod
    def add_period_and_roll(self, date_tensor, period_tensor, roll_convention=constants.BusinessDayConvention.NONE):
        if False:
            print('Hello World!')
        'Adds given periods to given dates and rolls to business days.\n\n    The original dates are not rolled prior to addition.\n\n    Args:\n      date_tensor: DateTensor of dates to add to.\n      period_tensor: PeriodTensor broadcastable to `date_tensor`.\n      roll_convention: BusinessDayConvention. Determines how to roll a date that\n        falls on a holiday.\n\n    Returns:\n      The resulting DateTensor.\n    '
        pass

    @abc.abstractmethod
    def add_business_days(self, date_tensor, num_days, roll_convention=constants.BusinessDayConvention.NONE):
        if False:
            return 10
        'Adds given number of business days to given dates.\n\n    Note that this is different from calling `add_period_and_roll` with\n    PeriodType.DAY. For example, adding 5 business days to Monday gives the next\n    Monday (unless there are holidays on this week or next Monday). Adding 5\n    days and rolling means landing on Saturday and then rolling either to next\n    Monday or to Friday of the same week, depending on the roll convention.\n\n    If any of the dates in `date_tensor` are not business days, they will be\n    rolled to business days before doing the addition. If `roll_convention` is\n    `NONE`, and any dates are not business days, an exception is raised.\n\n    Args:\n      date_tensor: DateTensor of dates to advance from.\n      num_days: Tensor of int32 type broadcastable to `date_tensor`.\n      roll_convention: BusinessDayConvention. Determines how to roll a date that\n        falls on a holiday.\n\n    Returns:\n      The resulting DateTensor.\n    '
        pass

    @abc.abstractmethod
    def subtract_period_and_roll(self, date_tensor, period_tensor, roll_convention=constants.BusinessDayConvention.NONE):
        if False:
            i = 10
            return i + 15
        'Subtracts given periods from given dates and rolls to business days.\n\n    The original dates are not rolled prior to subtraction.\n\n    Args:\n      date_tensor: DateTensor of dates to subtract from.\n      period_tensor: PeriodTensor broadcastable to `date_tensor`.\n      roll_convention: BusinessDayConvention. Determines how to roll a date that\n        falls on a holiday.\n\n    Returns:\n      The resulting DateTensor.\n    '
        pass

    @abc.abstractmethod
    def subtract_business_days(self, date_tensor, num_days, roll_convention=constants.BusinessDayConvention.NONE):
        if False:
            while True:
                i = 10
        'Adds given number of business days to given dates.\n\n    Note that this is different from calling `subtract_period_and_roll` with\n    PeriodType.DAY. For example, subtracting 5 business days from Friday gives\n    the previous Friday (unless there are holidays on this week or previous\n    Friday). Subtracting 5 days and rolling means landing on Sunday and then\n    rolling either to Monday or to Friday, depending on the roll convention.\n\n    If any of the dates in `date_tensor` are not business days, they will be\n    rolled to business days before doing the subtraction. If `roll_convention`\n    is `NONE`, and any dates are not business days, an exception is raised.\n\n    Args:\n      date_tensor: DateTensor of dates to advance from.\n      num_days: Tensor of int32 type broadcastable to `date_tensor`.\n      roll_convention: BusinessDayConvention. Determines how to roll a date that\n        falls on a holiday.\n\n    Returns:\n      The resulting DateTensor.\n    '
        pass

    @abc.abstractmethod
    def business_days_in_period(self, date_tensor, period_tensor):
        if False:
            while True:
                i = 10
        'Calculates number of business days in a period.\n\n    Includes the dates in `date_tensor`, but excludes final dates resulting from\n    addition of `period_tensor`.\n\n    Args:\n      date_tensor: DateTensor of starting dates.\n      period_tensor: PeriodTensor, should be broadcastable to `date_tensor`.\n\n    Returns:\n       An int32 Tensor with the number of business days in given periods that\n       start at given dates.\n\n    '
        pass

    @abc.abstractmethod
    def business_days_between(self, from_dates, to_dates):
        if False:
            while True:
                i = 10
        'Calculates number of business between pairs of dates.\n\n    For each pair, the initial date is included in the difference, and the final\n    date is excluded. If the final date is the same or earlier than the initial\n    date, zero is returned.\n\n    Args:\n      from_dates: DateTensor of initial dates.\n      to_dates: DateTensor of final dates, should be broadcastable to\n        `from_dates`.\n\n    Returns:\n       An int32 Tensor with the number of business days between the\n       corresponding pairs of dates.\n    '
        pass