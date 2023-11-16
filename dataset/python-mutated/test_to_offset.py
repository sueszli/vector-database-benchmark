import re
import pytest
from pandas._libs.tslibs import Timedelta, offsets, to_offset

@pytest.mark.parametrize('freq_input,expected', [(to_offset('10us'), offsets.Micro(10)), (offsets.Hour(), offsets.Hour()), ('2h30min', offsets.Minute(150)), ('2h 30min', offsets.Minute(150)), ('2h30min15s', offsets.Second(150 * 60 + 15)), ('2h 60min', offsets.Hour(3)), ('2h 20.5min', offsets.Second(8430)), ('1.5min', offsets.Second(90)), ('0.5s', offsets.Milli(500)), ('15ms500us', offsets.Micro(15500)), ('10s75ms', offsets.Milli(10075)), ('1s0.25ms', offsets.Micro(1000250)), ('1s0.25ms', offsets.Micro(1000250)), ('2800ns', offsets.Nano(2800)), ('2SM', offsets.SemiMonthEnd(2)), ('2SM-16', offsets.SemiMonthEnd(2, day_of_month=16)), ('2SMS-14', offsets.SemiMonthBegin(2, day_of_month=14)), ('2SMS-15', offsets.SemiMonthBegin(2))])
def test_to_offset(freq_input, expected):
    if False:
        return 10
    result = to_offset(freq_input)
    assert result == expected

@pytest.mark.parametrize('freqstr,expected', [('-1s', -1), ('-2SM', -2), ('-1SMS', -1), ('-5min10s', -310)])
def test_to_offset_negative(freqstr, expected):
    if False:
        for i in range(10):
            print('nop')
    result = to_offset(freqstr)
    assert result.n == expected

@pytest.mark.parametrize('freqstr', ['2h20m', 'us1', '-us', '3us1', '-2-3us', '-2D:3h', '1.5.0s', '2SMS-15-15', '2SMS-15D', '100foo', '+-1d', '-+1h', '+1', '-7', '+d', '-m', 'SM-0', 'SM-28', 'SM-29', 'SM-FOO', 'BSM', 'SM--1', 'SMS-1', 'SMS-28', 'SMS-30', 'SMS-BAR', 'SMS-BYR', 'BSMS', 'SMS--2'])
def test_to_offset_invalid(freqstr):
    if False:
        while True:
            i = 10
    msg = re.escape(f'Invalid frequency: {freqstr}')
    with pytest.raises(ValueError, match=msg):
        to_offset(freqstr)

def test_to_offset_no_evaluate():
    if False:
        return 10
    msg = str(('', ''))
    with pytest.raises(TypeError, match=msg):
        to_offset(('', ''))

def test_to_offset_tuple_unsupported():
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError, match='pass as a string instead'):
        to_offset((5, 'T'))

@pytest.mark.parametrize('freqstr,expected', [('2D 3h', offsets.Hour(51)), ('2 D3 h', offsets.Hour(51)), ('2 D 3 h', offsets.Hour(51)), ('  2 D 3 h  ', offsets.Hour(51)), ('   h    ', offsets.Hour()), (' 3  h    ', offsets.Hour(3))])
def test_to_offset_whitespace(freqstr, expected):
    if False:
        i = 10
        return i + 15
    result = to_offset(freqstr)
    assert result == expected

@pytest.mark.parametrize('freqstr,expected', [('00h 00min 01s', 1), ('-00h 03min 14s', -194)])
def test_to_offset_leading_zero(freqstr, expected):
    if False:
        return 10
    result = to_offset(freqstr)
    assert result.n == expected

@pytest.mark.parametrize('freqstr,expected', [('+1d', 1), ('+2h30min', 150)])
def test_to_offset_leading_plus(freqstr, expected):
    if False:
        while True:
            i = 10
    result = to_offset(freqstr)
    assert result.n == expected

@pytest.mark.parametrize('kwargs,expected', [({'days': 1, 'seconds': 1}, offsets.Second(86401)), ({'days': -1, 'seconds': 1}, offsets.Second(-86399)), ({'hours': 1, 'minutes': 10}, offsets.Minute(70)), ({'hours': 1, 'minutes': -10}, offsets.Minute(50)), ({'weeks': 1}, offsets.Day(7)), ({'hours': 1}, offsets.Hour(1)), ({'hours': 1}, to_offset('60min')), ({'microseconds': 1}, offsets.Micro(1)), ({'microseconds': 0}, offsets.Nano(0))])
def test_to_offset_pd_timedelta(kwargs, expected):
    if False:
        print('Hello World!')
    td = Timedelta(**kwargs)
    result = to_offset(td)
    assert result == expected

@pytest.mark.parametrize('shortcut,expected', [('W', offsets.Week(weekday=6)), ('W-SUN', offsets.Week(weekday=6)), ('QE', offsets.QuarterEnd(startingMonth=12)), ('QE-DEC', offsets.QuarterEnd(startingMonth=12)), ('QE-MAY', offsets.QuarterEnd(startingMonth=5)), ('SM', offsets.SemiMonthEnd(day_of_month=15)), ('SM-15', offsets.SemiMonthEnd(day_of_month=15)), ('SM-1', offsets.SemiMonthEnd(day_of_month=1)), ('SM-27', offsets.SemiMonthEnd(day_of_month=27)), ('SMS-2', offsets.SemiMonthBegin(day_of_month=2)), ('SMS-27', offsets.SemiMonthBegin(day_of_month=27))])
def test_anchored_shortcuts(shortcut, expected):
    if False:
        for i in range(10):
            print('nop')
    result = to_offset(shortcut)
    assert result == expected