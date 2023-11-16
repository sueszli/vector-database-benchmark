from datetime import datetime
import pytest
from pandas._libs import tslib
from pandas import Timestamp

@pytest.mark.parametrize('date_str, exp', [('2011-01-02', datetime(2011, 1, 2)), ('2011-1-2', datetime(2011, 1, 2)), ('2011-01', datetime(2011, 1, 1)), ('2011-1', datetime(2011, 1, 1)), ('2011 01 02', datetime(2011, 1, 2)), ('2011.01.02', datetime(2011, 1, 2)), ('2011/01/02', datetime(2011, 1, 2)), ('2011\\01\\02', datetime(2011, 1, 2)), ('2013-01-01 05:30:00', datetime(2013, 1, 1, 5, 30)), ('2013-1-1 5:30:00', datetime(2013, 1, 1, 5, 30)), ('2013-1-1 5:30:00+01:00', Timestamp(2013, 1, 1, 5, 30, tz='UTC+01:00'))])
def test_parsers_iso8601(date_str, exp):
    if False:
        return 10
    actual = tslib._test_parse_iso8601(date_str)
    assert actual == exp

@pytest.mark.parametrize('date_str', ['2011-01/02', '2011=11=11', '201401', '201111', '200101', '2005-0101', '200501-01', '20010101 12:3456', '20010101 1234:56', '20010101 1', '20010101 123', '20010101 12345', '20010101 12345Z'])
def test_parsers_iso8601_invalid(date_str):
    if False:
        while True:
            i = 10
    msg = f'Error parsing datetime string "{date_str}"'
    with pytest.raises(ValueError, match=msg):
        tslib._test_parse_iso8601(date_str)

def test_parsers_iso8601_invalid_offset_invalid():
    if False:
        i = 10
        return i + 15
    date_str = '2001-01-01 12-34-56'
    msg = f'Timezone hours offset out of range in datetime string "{date_str}"'
    with pytest.raises(ValueError, match=msg):
        tslib._test_parse_iso8601(date_str)

def test_parsers_iso8601_leading_space():
    if False:
        print('Hello World!')
    (date_str, expected) = ('2013-1-1 5:30:00', datetime(2013, 1, 1, 5, 30))
    actual = tslib._test_parse_iso8601(' ' * 200 + date_str)
    assert actual == expected

@pytest.mark.parametrize('date_str, timespec, exp', [('2023-01-01 00:00:00', 'auto', '2023-01-01T00:00:00'), ('2023-01-01 00:00:00', 'seconds', '2023-01-01T00:00:00'), ('2023-01-01 00:00:00', 'milliseconds', '2023-01-01T00:00:00.000'), ('2023-01-01 00:00:00', 'microseconds', '2023-01-01T00:00:00.000000'), ('2023-01-01 00:00:00', 'nanoseconds', '2023-01-01T00:00:00.000000000'), ('2023-01-01 00:00:00.001', 'auto', '2023-01-01T00:00:00.001000'), ('2023-01-01 00:00:00.001', 'seconds', '2023-01-01T00:00:00'), ('2023-01-01 00:00:00.001', 'milliseconds', '2023-01-01T00:00:00.001'), ('2023-01-01 00:00:00.001', 'microseconds', '2023-01-01T00:00:00.001000'), ('2023-01-01 00:00:00.001', 'nanoseconds', '2023-01-01T00:00:00.001000000'), ('2023-01-01 00:00:00.000001', 'auto', '2023-01-01T00:00:00.000001'), ('2023-01-01 00:00:00.000001', 'seconds', '2023-01-01T00:00:00'), ('2023-01-01 00:00:00.000001', 'milliseconds', '2023-01-01T00:00:00.000'), ('2023-01-01 00:00:00.000001', 'microseconds', '2023-01-01T00:00:00.000001'), ('2023-01-01 00:00:00.000001', 'nanoseconds', '2023-01-01T00:00:00.000001000'), ('2023-01-01 00:00:00.000000001', 'auto', '2023-01-01T00:00:00.000000001'), ('2023-01-01 00:00:00.000000001', 'seconds', '2023-01-01T00:00:00'), ('2023-01-01 00:00:00.000000001', 'milliseconds', '2023-01-01T00:00:00.000'), ('2023-01-01 00:00:00.000000001', 'microseconds', '2023-01-01T00:00:00.000000'), ('2023-01-01 00:00:00.000000001', 'nanoseconds', '2023-01-01T00:00:00.000000001'), ('2023-01-01 00:00:00.000001001', 'auto', '2023-01-01T00:00:00.000001001'), ('2023-01-01 00:00:00.000001001', 'seconds', '2023-01-01T00:00:00'), ('2023-01-01 00:00:00.000001001', 'milliseconds', '2023-01-01T00:00:00.000'), ('2023-01-01 00:00:00.000001001', 'microseconds', '2023-01-01T00:00:00.000001'), ('2023-01-01 00:00:00.000001001', 'nanoseconds', '2023-01-01T00:00:00.000001001')])
def test_iso8601_formatter(date_str: str, timespec: str, exp: str):
    if False:
        for i in range(10):
            print('nop')
    ts = Timestamp(date_str)
    assert ts.isoformat(timespec=timespec) == exp