"""
Behavioral based tests for offsets and date_range.

This file is adapted from https://github.com/pandas-dev/pandas/pull/18761 -
which was more ambitious but less idiomatic in its use of Hypothesis.

You may wish to consult the previous version for inspiration on further
tests, or when trying to pin down the bugs exposed by the tests below.
"""
from hypothesis import assume, given
import pytest
import pytz
import pandas as pd
from pandas._testing._hypothesis import DATETIME_JAN_1_1900_OPTIONAL_TZ, YQM_OFFSET

@pytest.mark.arm_slow
@given(DATETIME_JAN_1_1900_OPTIONAL_TZ, YQM_OFFSET)
def test_on_offset_implementations(dt, offset):
    if False:
        for i in range(10):
            print('nop')
    assume(not offset.normalize)
    try:
        compare = dt + offset - offset
    except (pytz.NonExistentTimeError, pytz.AmbiguousTimeError):
        assume(False)
    assert offset.is_on_offset(dt) == (compare == dt)

@given(YQM_OFFSET)
def test_shift_across_dst(offset):
    if False:
        i = 10
        return i + 15
    assume(not offset.normalize)
    dti = pd.date_range(start='2017-10-30 12:00:00', end='2017-11-06', freq='D', tz='US/Eastern')
    assert (dti.hour == 12).all()
    res = dti + offset
    assert (res.hour == 12).all()