"""HolidayCalendar definition."""
import tensorflow.compat.v2 as tf
from tf_quant_finance.datetime import constants
from tf_quant_finance.datetime import date_tensor as dt
from tf_quant_finance.datetime import date_utils as du
from tf_quant_finance.datetime import holiday_calendar
from tf_quant_finance.datetime import holiday_utils as hol
from tf_quant_finance.datetime import periods

class UnboundedHolidayCalendar(holiday_calendar.HolidayCalendar):
    """HolidayCalendar implementation.

  Unlike BoundedHolidayCalendar, doesn't require specifying calendar bounds, and
  supports weekends and holiday supplied as `Tensor`s. However, it is
  (potentially significantly) slower than the dates.HolidayCalendar
  implementation.
  """

    def __init__(self, weekend_mask=None, holidays=None):
        if False:
            print('Hello World!')
        'Initializer.\n\n    Args:\n      weekend_mask: Boolean `Tensor` of 7 elements one for each day of the week\n        starting with Monday at index 0. A `True` value indicates the day is\n        considered a weekend day and a `False` value implies a week day.\n        Default value: None which means no weekends are applied.\n      holidays: Defines the holidays that are added to the weekends defined by\n        `weekend_mask`. An instance of `dates.DateTensor` or an object\n        convertible to `DateTensor`.\n        Default value: None which means no holidays other than those implied by\n          the weekends (if any).\n    '
        if weekend_mask is not None:
            weekend_mask = tf.cast(weekend_mask, dtype=tf.bool)
        if holidays is not None:
            holidays = dt.convert_to_date_tensor(holidays).ordinal()
        (self._to_biz_space, self._from_biz_space) = hol.business_day_mappers(weekend_mask=weekend_mask, holidays=holidays)

    def is_business_day(self, date_tensor):
        if False:
            for i in range(10):
                print('nop')
        'Returns a tensor of bools for whether given dates are business days.'
        ordinals = dt.convert_to_date_tensor(date_tensor).ordinal()
        return self._to_biz_space(ordinals)[1]

    def roll_to_business_day(self, date_tensor, roll_convention):
        if False:
            return 10
        'Rolls the given dates to business dates according to given convention.\n\n    Args:\n      date_tensor: `DateTensor` of dates to roll from.\n      roll_convention: BusinessDayConvention. Determines how to roll a date that\n        falls on a holiday.\n\n    Returns:\n      The resulting `DateTensor`.\n    '
        if roll_convention == constants.BusinessDayConvention.NONE:
            return date_tensor
        ordinals = dt.convert_to_date_tensor(date_tensor).ordinal()
        (biz_days, is_bizday) = self._to_biz_space(ordinals)
        biz_days_rolled = self._apply_roll_biz_space(date_tensor, biz_days, is_bizday, roll_convention)
        return dt.from_ordinals(self._from_biz_space(biz_days_rolled))

    def _apply_roll_biz_space(self, date_tensor, biz_days, is_bizday, roll_convention):
        if False:
            return 10
        'Applies roll in business day space.'
        if roll_convention == constants.BusinessDayConvention.NONE:
            return biz_days
        if roll_convention == constants.BusinessDayConvention.FOLLOWING:
            return tf.where(is_bizday, biz_days, biz_days + 1)
        if roll_convention == constants.BusinessDayConvention.PRECEDING:
            return biz_days
        if roll_convention == constants.BusinessDayConvention.MODIFIED_FOLLOWING:
            maybe_prev_biz_day = biz_days
            maybe_next_biz_day = tf.where(is_bizday, biz_days, biz_days + 1)
            maybe_next_biz_ordinal = self._from_biz_space(maybe_next_biz_day)
            take_previous = tf.not_equal(_get_month(maybe_next_biz_ordinal), date_tensor.month())
            return tf.where(take_previous, maybe_prev_biz_day, maybe_next_biz_day)
        if roll_convention == constants.BusinessDayConvention.MODIFIED_PRECEDING:
            maybe_prev_biz_day = biz_days
            maybe_next_biz_day = tf.where(is_bizday, biz_days, biz_days + 1)
            maybe_prev_biz_ordinal = self._from_biz_space(maybe_prev_biz_day)
            take_next = tf.not_equal(_get_month(maybe_prev_biz_ordinal), date_tensor.month())
            return tf.where(take_next, maybe_next_biz_day, maybe_prev_biz_day)
        raise ValueError('Unsupported roll convention: {}'.format(roll_convention))

    def add_period_and_roll(self, date_tensor, period_tensor, roll_convention=constants.BusinessDayConvention.NONE):
        if False:
            print('Hello World!')
        'Adds given periods to given dates and rolls to business days.\n\n    The original dates are not rolled prior to addition.\n\n    Args:\n      date_tensor: `DateTensor` of dates to add to.\n      period_tensor: PeriodTensor broadcastable to `date_tensor`.\n      roll_convention: BusinessDayConvention. Determines how to roll a date that\n        falls on a holiday.\n\n    Returns:\n      The resulting `DateTensor`.\n    '
        return self.roll_to_business_day(date_tensor + period_tensor, roll_convention)

    def add_business_days(self, date_tensor, num_days, roll_convention=constants.BusinessDayConvention.NONE):
        if False:
            print('Hello World!')
        'Adds given number of business days to given dates.\n\n    Note that this is different from calling `add_period_and_roll` with\n    PeriodType.DAY. For example, adding 5 business days to Monday gives the next\n    Monday (unless there are holidays on this week or next Monday). Adding 5\n    days and rolling means landing on Saturday and then rolling either to next\n    Monday or to Friday of the same week, depending on the roll convention.\n\n    If any of the dates in `date_tensor` are not business days, they will be\n    rolled to business days before doing the addition. If `roll_convention` is\n    `NONE`, and any dates are not business days, an exception is raised.\n\n    Args:\n      date_tensor: `DateTensor` of dates to advance from.\n      num_days: Tensor of int32 type broadcastable to `date_tensor`.\n      roll_convention: BusinessDayConvention. Determines how to roll a date that\n        falls on a holiday.\n\n    Returns:\n      The resulting `DateTensor`.\n    '
        control_deps = []
        (biz_days, is_bizday) = self._to_biz_space(dt.convert_to_date_tensor(date_tensor).ordinal())
        if roll_convention == constants.BusinessDayConvention.NONE:
            control_deps.append(tf.debugging.assert_equal(is_bizday, True, message='Non business starting day with no roll convention.'))
        with tf.compat.v1.control_dependencies(control_deps):
            biz_days_rolled = self._apply_roll_biz_space(date_tensor, biz_days, is_bizday, roll_convention)
            return dt.from_ordinals(self._from_biz_space(biz_days_rolled + num_days))

    def subtract_period_and_roll(self, date_tensor, period_tensor, roll_convention=constants.BusinessDayConvention.NONE):
        if False:
            return 10
        'Subtracts given periods from given dates and rolls to business days.\n\n    The original dates are not rolled prior to subtraction.\n\n    Args:\n      date_tensor: `DateTensor` of dates to subtract from.\n      period_tensor: PeriodTensor broadcastable to `date_tensor`.\n      roll_convention: BusinessDayConvention. Determines how to roll a date that\n        falls on a holiday.\n\n    Returns:\n      The resulting `DateTensor`.\n    '
        minus_period_tensor = periods.PeriodTensor(-period_tensor.quantity(), period_tensor.period_type())
        return self.add_period_and_roll(date_tensor, minus_period_tensor, roll_convention)

    def subtract_business_days(self, date_tensor, num_days, roll_convention=constants.BusinessDayConvention.NONE):
        if False:
            return 10
        'Adds given number of business days to given dates.\n\n    Note that this is different from calling `subtract_period_and_roll` with\n    PeriodType.DAY. For example, subtracting 5 business days from Friday gives\n    the previous Friday (unless there are holidays on this week or previous\n    Friday). Subtracting 5 days and rolling means landing on Sunday and then\n    rolling either to Monday or to Friday, depending on the roll convention.\n\n    If any of the dates in `date_tensor` are not business days, they will be\n    rolled to business days before doing the subtraction. If `roll_convention`\n    is `NONE`, and any dates are not business days, an exception is raised.\n\n    Args:\n      date_tensor: `DateTensor` of dates to advance from.\n      num_days: Tensor of int32 type broadcastable to `date_tensor`.\n      roll_convention: BusinessDayConvention. Determines how to roll a date that\n        falls on a holiday.\n\n    Returns:\n      The resulting `DateTensor`.\n    '
        return self.add_business_days(date_tensor, -num_days, roll_convention)

    def business_days_in_period(self, date_tensor, period_tensor):
        if False:
            while True:
                i = 10
        'Calculates number of business days in a period.\n\n    Includes the dates in `date_tensor`, but excludes final dates resulting from\n    addition of `period_tensor`.\n\n    Args:\n      date_tensor: `DateTensor` of starting dates.\n      period_tensor: PeriodTensor, should be broadcastable to `date_tensor`.\n\n    Returns:\n       An int32 Tensor with the number of business days in given periods that\n       start at given dates.\n\n    '
        return self.business_days_between(date_tensor, date_tensor + period_tensor)

    def business_days_between(self, from_dates, to_dates):
        if False:
            return 10
        'Calculates number of business between pairs of dates.\n\n    For each pair, the initial date is included in the difference, and the final\n    date is excluded. If the final date is the same or earlier than the initial\n    date, zero is returned.\n\n    Args:\n      from_dates: `DateTensor` of initial dates.\n      to_dates: `DateTensor` of final dates, should be broadcastable to\n        `from_dates`.\n\n    Returns:\n       An int32 Tensor with the number of business days between the\n       corresponding pairs of dates.\n    '
        (from_biz, from_is_bizday) = self._to_biz_space(dt.convert_to_date_tensor(from_dates).ordinal())
        (to_biz, to_is_bizday) = self._to_biz_space(dt.convert_to_date_tensor(to_dates).ordinal())
        from_biz = tf.where(from_is_bizday, from_biz, from_biz + 1)
        to_biz = tf.where(to_is_bizday, to_biz, to_biz + 1)
        return tf.math.maximum(to_biz - from_biz, 0)

def _get_month(ordinals):
    if False:
        i = 10
        return i + 15
    return du.ordinal_to_year_month_day(ordinals)[1]