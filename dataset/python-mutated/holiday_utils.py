"""Utils to manipulate holidays."""
import tensorflow.compat.v2 as tf
_DAYOFWEEK_0 = 6

def business_day_mappers(weekend_mask=None, holidays=None):
    if False:
        while True:
            i = 10
    'Returns functions to map from ordinal to biz day and back.'
    if weekend_mask is None and holidays is None:
        return (lambda x: (x, tf.ones_like(x, dtype=tf.bool)), lambda x: x)
    (weekday_fwd, weekday_back) = _week_day_mappers(weekend_mask)
    if holidays is None:
        return (weekday_fwd, weekday_back)
    holidays_raw = tf.convert_to_tensor(holidays, dtype=tf.int32)
    (holidays, is_weekday) = weekday_fwd(holidays_raw)
    holidays = holidays[is_weekday]
    holidays = tf.concat([[0], holidays], axis=0)
    reverse_holidays = tf.reverse(-holidays, axis=[0])
    num_holidays = tf.size(holidays) - 1

    def bizday_fwd(x):
        if False:
            for i in range(10):
                print('nop')
        'Calculates business day ordinal and whether it is a business day.'
        left = tf.searchsorted(holidays, x, side='left')
        right = num_holidays - tf.searchsorted(reverse_holidays, -x, side='left')
        is_bizday = tf.not_equal(left, right)
        bizday_ordinal = x - right
        return (bizday_ordinal, is_bizday)
    cum_holidays = tf.range(num_holidays + 1, dtype=holidays.dtype)
    bizday_at_holidays = holidays - cum_holidays

    def bizday_back(x):
        if False:
            for i in range(10):
                print('nop')
        left = tf.searchsorted(bizday_at_holidays, x, side='left')
        ordinal = x + left - 1
        return ordinal

    def from_ordinal(ordinals):
        if False:
            while True:
                i = 10
        'Maps ordinals to business day and whether it is a work day.'
        ordinals = tf.convert_to_tensor(ordinals, dtype=tf.int32)
        (weekday_values, is_weekday) = weekday_fwd(ordinals)
        (biz_ordinal, is_bizday) = bizday_fwd(weekday_values)
        return (biz_ordinal, is_weekday & is_bizday)

    def to_ordinal(biz_values):
        if False:
            while True:
                i = 10
        'Maps from business day count to ordinals.'
        return weekday_back(bizday_back(biz_values))
    return (from_ordinal, to_ordinal)

def _week_day_mappers(weekend_mask):
    if False:
        i = 10
        return i + 15
    'Creates functions to map from ordinals to week days and inverse.\n\n  Creates functions to map from ordinal space (i.e. days since 31 Dec 0) to\n  week days. The function assigns the value of 0 to the first non weekend\n  day in the week starting on Sunday, 31 Dec 1 through to Saturday, 6 Jan 1 and\n  the value assigned to each successive work day is incremented by 1. For a day\n  that is not a week day, this count is not incremented from the previous week\n  day (hence, multiple ordinal days may have the same week day value).\n\n  Args:\n    weekend_mask: A bool `Tensor` of length 7 or None. The weekend mask.\n\n  Returns:\n    A tuple of callables.\n      `forward`: Takes one `Tensor` argument containing ordinals and returns a\n        tuple of two `Tensor`s of the same shape as the input. The first\n        `Tensor` is of type `int32` and contains the week day value. The second\n        is a bool `Tensor` indicating whether the supplied ordinal was a weekend\n        day (i.e. True where the day is a weekend day and False otherwise).\n      `backward`: Takes one int32 `Tensor` argument containing week day values\n        and returns an int32 `Tensor` containing ordinals for those week days.\n  '
    if weekend_mask is None:
        default_forward = lambda x: (x, tf.zeros_like(x, dtype=tf.bool))
        identity = lambda x: x
        return (default_forward, identity)
    weekend_mask = tf.convert_to_tensor(weekend_mask, dtype=tf.bool)
    weekend_mask = tf.roll(weekend_mask, -_DAYOFWEEK_0, axis=0)
    weekday_mask = tf.logical_not(weekend_mask)
    weekday_offsets = tf.cumsum(tf.cast(weekday_mask, dtype=tf.int32))
    num_workdays = weekday_offsets[-1]
    weekday_offsets -= 1
    ordinal_offsets = tf.convert_to_tensor([0, 1, 2, 3, 4, 5, 6], dtype=tf.int32)
    ordinal_offsets = ordinal_offsets[weekday_mask]

    def forward(ordinals):
        if False:
            for i in range(10):
                print('nop')
        'Adjusts the ordinals by removing the number of weekend days so far.'
        (mod, remainder) = (ordinals // 7, ordinals % 7)
        weekday_values = mod * num_workdays + tf.gather(weekday_offsets, remainder)
        is_weekday = tf.gather(weekday_mask, remainder)
        return (weekday_values, is_weekday)

    def backward(weekday_values):
        if False:
            while True:
                i = 10
        'Converts from weekend adjusted values to ordinals.'
        return weekday_values // num_workdays * 7 + tf.gather(ordinal_offsets, weekday_values % num_workdays)
    return (forward, backward)