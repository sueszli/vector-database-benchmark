"""DateTensor definition."""
import collections.abc
import datetime
import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance.datetime import constants
from tf_quant_finance.datetime import date_utils
from tf_quant_finance.datetime import periods
from tf_quant_finance.datetime import tensor_wrapper
_DAYS_IN_MONTHS_NON_LEAP = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_DAYS_IN_MONTHS_LEAP = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_DAYS_IN_MONTHS_COMBINED = [0] + _DAYS_IN_MONTHS_NON_LEAP + _DAYS_IN_MONTHS_LEAP
_ORDINAL_OF_1_1_1970 = 719163

class DateTensor(tensor_wrapper.TensorWrapper):
    """Represents a tensor of dates."""

    def __init__(self, ordinals, years, months, days):
        if False:
            while True:
                i = 10
        "Initializer.\n\n    This initializer is primarily for internal use. More convenient construction\n    methods are available via 'dates_from_*' functions.\n\n    Args:\n      ordinals: Tensor of type int32. Each value is number of days since 1 Jan\n        0001. 1 Jan 0001 has `ordinal=1`. `years`, `months` and `days` must\n        represent the same dates as `ordinals`.\n      years: Tensor of type int32, of same shape as `ordinals`.\n      months: Tensor of type int32, of same shape as `ordinals`\n      days: Tensor of type int32, of same shape as `ordinals`.\n    "
        self._ordinals = tf.convert_to_tensor(ordinals, dtype=tf.int32, name='dt_ordinals')
        self._years = tf.convert_to_tensor(years, dtype=tf.int32, name='dt_years')
        self._months = tf.convert_to_tensor(months, dtype=tf.int32, name='dt_months')
        self._days = tf.convert_to_tensor(days, dtype=tf.int32, name='dt_days')
        self._day_of_year = None

    def day(self):
        if False:
            while True:
                i = 10
        'Returns an int32 tensor of days since the beginning the month.\n\n    The result is one-based, i.e. yields 1 for first day of the month.\n\n    #### Example\n\n    ```python\n    dates = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])\n    dates.day()  # [25, 2]\n    ```\n    '
        return self._days

    def day_of_week(self):
        if False:
            while True:
                i = 10
        'Returns an int32 tensor of weekdays.\n\n    The result is zero-based according to Python datetime convention, i.e.\n    Monday is "0".\n\n    #### Example\n\n    ```python\n    dates = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])\n    dates.day_of_week()  # [5, 1]\n    ```\n    '
        return (self._ordinals - 1) % 7

    def month(self):
        if False:
            while True:
                i = 10
        'Returns an int32 tensor of months.\n\n    #### Example\n\n    ```python\n    dates = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])\n    dates.month()  # [1, 3]\n    ```\n    '
        return self._months

    def year(self):
        if False:
            return 10
        'Returns an int32 tensor of years.\n\n    #### Example\n\n    ```python\n    dates = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])\n    dates.year()  # [2019, 2020]\n    ```\n    '
        return self._years

    def ordinal(self):
        if False:
            while True:
                i = 10
        'Returns an int32 tensor of ordinals.\n\n    Ordinal is the number of days since 1st Jan 0001.\n\n    #### Example\n\n    ```python\n    dates = tff.datetime.dates_from_tuples([(2019, 3, 25), (1, 1, 1)])\n    dates.ordinal()  # [737143, 1]\n    ```\n    '
        return self._ordinals

    def to_tensor(self):
        if False:
            while True:
                i = 10
        'Packs the dates into a single Tensor.\n\n    The Tensor has shape `date_tensor.shape() + (3,)`, where the last dimension\n    represents years, months and days, in this order.\n\n    This can be convenient when the dates are the final result of a computation\n    in the graph mode: a `tf.function` can return `date_tensor.to_tensor()`, or,\n    if one uses `tf.compat.v1.Session`, they can call\n    `session.run(date_tensor.to_tensor())`.\n\n    Returns:\n      A Tensor of shape `date_tensor.shape() + (3,)`.\n\n    #### Example\n\n    ```python\n    dates = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])\n    dates.to_tensor()  # tf.Tensor with contents [[2019, 1, 25], [2020, 3, 2]].\n    ```\n    '
        return tf.stack((self.year(), self.month(), self.day()), axis=-1)

    def day_of_year(self):
        if False:
            return 10
        'Calculates the number of days since the beginning of the year.\n\n    Returns:\n      Tensor of int32 type with elements in range [1, 366]. January 1st yields\n      "1".\n\n    #### Example\n\n    ```python\n    dt = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])\n    dt.day_of_year()  # [25, 62]\n    ```\n    '
        if self._day_of_year is None:
            cumul_days_in_month_nonleap = tf.math.cumsum(_DAYS_IN_MONTHS_NON_LEAP, exclusive=True)
            cumul_days_in_month_leap = tf.math.cumsum(_DAYS_IN_MONTHS_LEAP, exclusive=True)
            days_before_month_non_leap = tf.gather(cumul_days_in_month_nonleap, self.month() - 1)
            days_before_month_leap = tf.gather(cumul_days_in_month_leap, self.month() - 1)
            days_before_month = tf.where(date_utils.is_leap_year(self.year()), days_before_month_leap, days_before_month_non_leap)
            self._day_of_year = days_before_month + self.day()
        return self._day_of_year

    def days_until(self, target_date_tensor):
        if False:
            while True:
                i = 10
        'Computes the number of days until the target dates.\n\n    Args:\n      target_date_tensor: A DateTensor object broadcastable to the shape of\n        "self".\n\n    Returns:\n      An int32 tensor with numbers of days until the target dates.\n\n     #### Example\n\n     ```python\n    dates = tff.datetime.dates_from_tuples([(2020, 1, 25), (2020, 3, 2)])\n    target = tff.datetime.dates_from_tuples([(2020, 3, 5)])\n    dates.days_until(target) # [40, 3]\n\n    targets = tff.datetime.dates_from_tuples([(2020, 2, 5), (2020, 3, 5)])\n    dates.days_until(targets)  # [11, 3]\n    ```\n    '
        return target_date_tensor.ordinal() - self._ordinals

    def period_length_in_days(self, period_tensor):
        if False:
            return 10
        'Computes the number of days in each period.\n\n    Args:\n      period_tensor: A PeriodTensor object broadcastable to the shape of "self".\n\n    Returns:\n      An int32 tensor with numbers of days each period takes.\n\n    #### Example\n\n    ```python\n    dates = tff.datetime.dates_from_tuples([(2020, 2, 25), (2020, 3, 2)])\n    dates.period_length_in_days(month())  # [29, 31]\n\n    periods = tff.datetime.months([1, 2])\n    dates.period_length_in_days(periods)  # [29, 61]\n    ```\n    '
        return (self + period_tensor).ordinal() - self._ordinals

    def is_end_of_month(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a bool Tensor indicating whether dates are at ends of months.'
        return tf.math.equal(self._days, _num_days_in_month(self._months, self._years))

    def to_end_of_month(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a new DateTensor with each date shifted to the end of month.'
        days = _num_days_in_month(self._months, self._years)
        return from_year_month_day(self._years, self._months, days, validate=False)

    @property
    def shape(self):
        if False:
            print('Hello World!')
        return self._ordinals.shape

    @property
    def rank(self):
        if False:
            return 10
        return tf.rank(self._ordinals)

    def __add__(self, period_tensor):
        if False:
            print('Hello World!')
        'Adds a tensor of periods.\n\n    When adding months or years, the resulting day of the month is decreased\n    to the largest valid value if necessary. E.g. 31.03.2020 + 1 month =\n    30.04.2020, 29.02.2020 + 1 year = 28.02.2021.\n\n    Args:\n      period_tensor: A `PeriodTensor` object broadcastable to the shape of\n      "self".\n\n    Returns:\n      The new instance of DateTensor.\n\n    #### Example\n    ```python\n    dates = tff.datetime.dates_from_tuples([(2020, 2, 25), (2020, 3, 31)])\n    new_dates = dates + tff.datetime.month()\n    # DateTensor([(2020, 3, 25), (2020, 4, 30)])\n\n    new_dates = dates + tff.datetime.month([1, 2])\n    # DateTensor([(2020, 3, 25), (2020, 5, 31)])\n    ```\n    '
        period_type = period_tensor.period_type()
        if period_type == constants.PeriodType.DAY:
            ordinals = self._ordinals + period_tensor.quantity()
            return from_ordinals(ordinals)
        if period_type == constants.PeriodType.WEEK:
            return self + periods.PeriodTensor(period_tensor.quantity() * 7, constants.PeriodType.DAY)

        def adjust_day(year, month, day):
            if False:
                i = 10
                return i + 15
            return tf.math.minimum(day, _num_days_in_month(month, year))
        if period_type == constants.PeriodType.MONTH:
            m = self._months - 1 + period_tensor.quantity()
            y = self._years + m // 12
            m = m % 12 + 1
            d = adjust_day(y, m, self._days)
            return from_year_month_day(y, m, d, validate=False)
        if period_type == constants.PeriodType.YEAR:
            y = self._years + period_tensor.quantity()
            m = tf.broadcast_to(self._months, tf.shape(y))
            d = adjust_day(y, m, self._days)
            return from_year_month_day(y, m, d, validate=False)
        raise ValueError('Unrecognized period type: {}'.format(period_type))

    def __sub__(self, period_tensor):
        if False:
            i = 10
            return i + 15
        'Subtracts a tensor of periods.\n\n    When subtracting months or years, the resulting day of the month is\n    decreased to the largest valid value if necessary. E.g. 31.03.2020 - 1 month\n    = 29.02.2020, 29.02.2020 - 1 year = 28.02.2019.\n\n    Args:\n      period_tensor: a PeriodTensor object broadcastable to the shape of "self".\n\n    Returns:\n      The new instance of DateTensor.\n    '
        return self + periods.PeriodTensor(-period_tensor.quantity(), period_tensor.period_type())

    def __eq__(self, other):
        if False:
            return 10
        'Compares two DateTensors by "==", returning a Tensor of bools.'
        return tf.math.equal(self._ordinals, other.ordinal())

    def __ne__(self, other):
        if False:
            return 10
        'Compares two DateTensors by "!=", returning a Tensor of bools.'
        return tf.math.not_equal(self._ordinals, other.ordinal())

    def __gt__(self, other):
        if False:
            print('Hello World!')
        'Compares two DateTensors by ">", returning a Tensor of bools.'
        return self._ordinals > other.ordinal()

    def __ge__(self, other):
        if False:
            print('Hello World!')
        'Compares two DateTensors by ">=", returning a Tensor of bools.'
        return self._ordinals >= other.ordinal()

    def __lt__(self, other):
        if False:
            print('Hello World!')
        'Compares two DateTensors by "<", returning a Tensor of bools.'
        return self._ordinals < other.ordinal()

    def __le__(self, other):
        if False:
            while True:
                i = 10
        'Compares two DateTensors by "<=", returning a Tensor of bools.'
        return self._ordinals <= other.ordinal()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        output = 'DateTensor: shape={}'.format(self.shape)
        if tf.executing_eagerly():
            contents_np = np.stack((self._years.numpy(), self._months.numpy(), self._days.numpy()), axis=-1)
            return output + ', contents={}'.format(repr(contents_np))
        return output

    @classmethod
    def _apply_sequence_to_tensor_op(cls, op_fn, tensor_wrappers):
        if False:
            while True:
                i = 10
        o = op_fn([t.ordinal() for t in tensor_wrappers])
        y = op_fn([t.year() for t in tensor_wrappers])
        m = op_fn([t.month() for t in tensor_wrappers])
        d = op_fn([t.day() for t in tensor_wrappers])
        return DateTensor(o, y, m, d)

    def _apply_op(self, op_fn):
        if False:
            print('Hello World!')
        (o, y, m, d) = (op_fn(t) for t in (self._ordinals, self._years, self._months, self._days))
        return DateTensor(o, y, m, d)

def _num_days_in_month(month, year):
    if False:
        for i in range(10):
            print('nop')
    'Returns number of days in a given month of a given year.'
    days_in_months = tf.constant(_DAYS_IN_MONTHS_COMBINED, tf.int32)
    is_leap = date_utils.is_leap_year(year)
    return tf.gather(days_in_months, month + 12 * tf.dtypes.cast(is_leap, tf.int32))

def convert_to_date_tensor(date_inputs):
    if False:
        while True:
            i = 10
    "Converts supplied data to a `DateTensor` if possible.\n\n  Args:\n    date_inputs: One of the supported types that can be converted to a\n      DateTensor. The following input formats are supported. 1. Sequence of\n      `datetime.datetime`, `datetime.date`, or any other structure with data\n      attributes called 'year', 'month' and 'day'. 2. A numpy array of\n      `datetime64` type. 3. Sequence of (year, month, day) Tuples. Months are\n      1-based (with January as 1) and constants.Months enum may be used instead\n      of ints. Days are also 1-based. 4. A tuple of three int32 `Tensor`s\n      containing year, month and date as positive integers in that order. 5. A\n      single int32 `Tensor` containing ordinals (i.e. number of days since 31\n      Dec 0 with 1 being 1 Jan 1.)\n\n  Returns:\n    A `DateTensor` object representing the supplied dates.\n\n  Raises:\n    ValueError: If conversion fails for any reason.\n  "
    if isinstance(date_inputs, DateTensor):
        return date_inputs
    if hasattr(date_inputs, 'year'):
        return from_datetimes(date_inputs)
    if isinstance(date_inputs, np.ndarray):
        date_inputs = date_inputs.astype('datetime64[D]')
        return from_np_datetimes(date_inputs)
    if tf.is_tensor(date_inputs):
        return from_ordinals(date_inputs)
    if isinstance(date_inputs, collections.abc.Sequence):
        if not date_inputs:
            return from_ordinals([])
        test_element = date_inputs[0]
        if hasattr(test_element, 'year'):
            return from_datetimes(date_inputs)
        if isinstance(test_element, collections.abc.Sequence):
            return from_tuples(date_inputs)
        if len(date_inputs) == 3:
            return from_year_month_day(date_inputs[0], date_inputs[1], date_inputs[2])
    try:
        as_ordinals = tf.convert_to_tensor(date_inputs, dtype=tf.int32)
        return from_ordinals(as_ordinals)
    except ValueError as e:
        raise ValueError('Failed to convert inputs to DateTensor. Unrecognized format. Error: ' + e)

def from_datetimes(datetimes):
    if False:
        i = 10
        return i + 15
    'Creates DateTensor from a sequence of Python datetime objects.\n\n  Args:\n    datetimes: Sequence of Python datetime objects.\n\n  Returns:\n    DateTensor object.\n\n  #### Example\n\n  ```python\n  import datetime\n\n  dates = [datetime.date(2015, 4, 15), datetime.date(2017, 12, 30)]\n  date_tensor = tff.datetime.dates_from_datetimes(dates)\n  ```\n  '
    if isinstance(datetimes, (datetime.date, datetime.datetime)):
        return from_year_month_day(datetimes.year, datetimes.month, datetimes.day, validate=False)
    years = tf.constant([dt.year for dt in datetimes], dtype=tf.int32)
    months = tf.constant([dt.month for dt in datetimes], dtype=tf.int32)
    days = tf.constant([dt.day for dt in datetimes], dtype=tf.int32)
    return from_year_month_day(years, months, days, validate=False)

def from_np_datetimes(np_datetimes):
    if False:
        i = 10
        return i + 15
    'Creates DateTensor from a Numpy array of dtype datetime64.\n\n  Args:\n    np_datetimes: Numpy array of dtype datetime64.\n\n  Returns:\n    DateTensor object.\n\n  #### Example\n\n  ```python\n  import datetime\n  import numpy as np\n\n  date_tensor_np = np.array(\n    [[datetime.date(2019, 3, 25), datetime.date(2020, 6, 2)],\n     [datetime.date(2020, 9, 15), datetime.date(2020, 12, 27)]],\n     dtype=np.datetime64)\n\n  date_tensor = tff.datetime.dates_from_np_datetimes(date_tensor_np)\n  ```\n  '
    ordinals = tf.constant(np_datetimes, dtype=tf.int32) + _ORDINAL_OF_1_1_1970
    return from_ordinals(ordinals, validate=False)

def from_tuples(year_month_day_tuples, validate=True):
    if False:
        for i in range(10):
            print('nop')
    'Creates DateTensor from a sequence of year-month-day Tuples.\n\n  Args:\n    year_month_day_tuples: Sequence of (year, month, day) Tuples. Months are\n      1-based; constants from Months enum can be used instead of ints. Days are\n      also 1-based.\n    validate: Whether to validate the dates.\n\n  Returns:\n    DateTensor object.\n\n  #### Example\n\n  ```python\n  date_tensor = tff.datetime.dates_from_tuples([(2015, 4, 15), (2017, 12, 30)])\n  ```\n  '
    (years, months, days) = ([], [], [])
    for t in year_month_day_tuples:
        years.append(t[0])
        months.append(t[1])
        days.append(t[2])
    years = tf.constant(years, dtype=tf.int32)
    months = tf.constant(months, dtype=tf.int32)
    days = tf.constant(days, dtype=tf.int32)
    return from_year_month_day(years, months, days, validate)

def from_year_month_day(year, month, day, validate=True):
    if False:
        while True:
            i = 10
    'Creates DateTensor from tensors of years, months and days.\n\n  Args:\n    year: Tensor of int32 type. Elements should be positive.\n    month: Tensor of int32 type of same shape as `year`. Elements should be in\n      range `[1, 12]`.\n    day: Tensor of int32 type of same shape as `year`. Elements should be in\n      range `[1, 31]` and represent valid dates together with corresponding\n      elements of `month` and `year` Tensors.\n    validate: Whether to validate the dates.\n\n  Returns:\n    DateTensor object.\n\n  #### Example\n\n  ```python\n  year = tf.constant([2015, 2017], dtype=tf.int32)\n  month = tf.constant([4, 12], dtype=tf.int32)\n  day = tf.constant([15, 30], dtype=tf.int32)\n  date_tensor = tff.datetime.dates_from_year_month_day(year, month, day)\n  ```\n  '
    year = tf.convert_to_tensor(year, tf.int32)
    month = tf.convert_to_tensor(month, tf.int32)
    day = tf.convert_to_tensor(day, tf.int32)
    control_deps = []
    if validate:
        control_deps.append(tf.debugging.assert_positive(year, message='Year must be positive.'))
        control_deps.append(tf.debugging.assert_greater_equal(month, constants.Month.JANUARY.value, message=f'Month must be >= {constants.Month.JANUARY.value}'))
        control_deps.append(tf.debugging.assert_less_equal(month, constants.Month.DECEMBER.value, message='Month must be <= {constants.Month.JANUARY.value}'))
        control_deps.append(tf.debugging.assert_positive(day, message='Day must be positive.'))
        is_leap = date_utils.is_leap_year(year)
        days_in_months = tf.constant(_DAYS_IN_MONTHS_COMBINED, tf.int32)
        max_days = tf.gather(days_in_months, month + 12 * tf.dtypes.cast(is_leap, np.int32))
        control_deps.append(tf.debugging.assert_less_equal(day, max_days, message='Invalid day-month pairing.'))
        with tf.compat.v1.control_dependencies(control_deps):
            year = tf.identity(year)
            month = tf.identity(month)
            day = tf.identity(day)
    with tf.compat.v1.control_dependencies(control_deps):
        ordinal = date_utils.year_month_day_to_ordinal(year, month, day)
        return DateTensor(ordinal, year, month, day)

def from_ordinals(ordinals, validate=True):
    if False:
        return 10
    'Creates DateTensor from tensors of ordinals.\n\n  Args:\n    ordinals: Tensor of type int32. Each value is number of days since 1 Jan\n      0001. 1 Jan 0001 has `ordinal=1`.\n    validate: Whether to validate the dates.\n\n  Returns:\n    DateTensor object.\n\n  #### Example\n\n  ```python\n  ordinals = tf.constant([\n    735703,  # 2015-4-12\n    736693   # 2017-12-30\n  ], dtype=tf.int32)\n\n  date_tensor = tff.datetime.dates_from_ordinals(ordinals)\n  ```\n  '
    ordinals = tf.convert_to_tensor(ordinals, dtype=tf.int32)
    control_deps = []
    if validate:
        control_deps.append(tf.debugging.assert_positive(ordinals, message='Ordinals must be positive.'))
        with tf.compat.v1.control_dependencies(control_deps):
            ordinals = tf.identity(ordinals)
    with tf.compat.v1.control_dependencies(control_deps):
        (years, months, days) = date_utils.ordinal_to_year_month_day(ordinals)
        return DateTensor(ordinals, years, months, days)

def from_tensor(tensor, validate=True):
    if False:
        print('Hello World!')
    'Creates DateTensor from a single tensor containing years, months and days.\n\n  This function is complementary to DateTensor.to_tensor: given an int32 Tensor\n  of shape (..., 3), creates a DateTensor. The three elements of the last\n  dimension are years, months and days, in this order.\n\n  Args:\n    tensor: Tensor of type int32 and shape (..., 3).\n    validate: Whether to validate the dates.\n\n  Returns:\n    DateTensor object.\n\n  #### Example\n\n  ```python\n  tensor = tf.constant([[2015, 4, 15], [2017, 12, 30]], dtype=tf.int32)\n  date_tensor = tff.datetime.dates_from_tensor(tensor)\n  ```\n  '
    tensor = tf.convert_to_tensor(tensor, dtype=tf.int32)
    return from_year_month_day(year=tensor[..., 0], month=tensor[..., 1], day=tensor[..., 2], validate=validate)

def random_dates(*, start_date, end_date, size=1, seed=None, name=None):
    if False:
        i = 10
        return i + 15
    "Generates random dates between the supplied start and end dates.\n\n  Generates specified number of random dates between the given start and end\n  dates. The start and end dates are supplied as `DateTensor` objects. The dates\n  uniformly distributed between the start date (inclusive) and end date\n  (exclusive). Note that the dates are uniformly distributed over the calendar\n  range, i.e. no holiday calendar is taken into account.\n\n  Args:\n    start_date: DateTensor of arbitrary shape. The start dates of the range from\n      which to sample. The start dates are themselves included in the range.\n    end_date: DateTensor of shape compatible with the `start_date`. The end date\n      of the range from which to sample. The end dates are excluded from the\n      range.\n    size: Positive scalar int32 Tensor. The number of dates to draw between the\n      start and end date.\n      Default value: 1.\n    seed: Optional seed for the random generation.\n    name: Optional str. The name to give to the ops created by this function.\n      Default value: 'random_dates'.\n\n  Returns:\n    A DateTensor of shape [size] + dates_shape where dates_shape is the common\n    broadcast shape for (start_date, end_date).\n\n  #### Example\n\n  ```python\n  # Note that the start and end dates need to be of broadcastable shape (though\n  # not necessarily the same shape).\n  # In this example, the start dates are of shape [2] and the end dates are\n  # of a compatible but non-identical shape [1].\n  start_dates = tff.datetime.dates_from_tuples([\n    (2020, 5, 16),\n    (2020, 6, 13)\n  ])\n  end_dates = tff.datetime.dates_from_tuples([(2021, 5, 21)])\n  size = 3  # Generate 3 dates for each pair of (start, end date).\n  sample = tff.datetime.random_dates(start_date=start_dates, end_date=end_dates,\n                              size=size)\n  # sample is a DateTensor of shape [3, 2]. The [3] is from the size and [2] is\n  # the common broadcast shape of start and end date.\n  ```\n  "
    with tf.name_scope(name or 'random_dates'):
        size = tf.reshape(tf.convert_to_tensor(size, dtype=tf.int32, name='size'), [-1])
        start_date = convert_to_date_tensor(start_date)
        end_date = convert_to_date_tensor(end_date)
        ordinal_range = tf.cast(end_date.ordinal() - start_date.ordinal(), dtype=tf.float64)
        sample_shape = tf.concat((size, tf.shape(ordinal_range)), axis=0)
        ordinal_sample = tf.cast(tf.random.uniform(sample_shape, maxval=ordinal_range, seed=seed, name='ordinal_sample', dtype=tf.float64), dtype=tf.int32)
        return from_ordinals(start_date.ordinal() + ordinal_sample, validate=False)