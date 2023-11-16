import pytest
from datetime import datetime, timezone
from faust.utils._iso8601_python import InvalidTZ, parse, parse_tz

def test_python():
    if False:
        print('Hello World!')
    dt1 = datetime.now().astimezone(timezone.utc)
    dt2 = parse(dt1.isoformat())
    assert dt1 == dt2

def test_timezone_no_sep():
    if False:
        for i in range(10):
            print('nop')
    dt = parse('2018-12-04T19:36:08-0500')
    assert dt.tzinfo
    assert str(dt.tzinfo) == 'UTC-05:00'

def test_parse_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        parse('foo')

@pytest.mark.parametrize('tz', ['Z', '+00:10', '-01:20', '+0300', '-0600'])
def test_parse_tz(tz):
    if False:
        print('Hello World!')
    assert parse_tz(tz) is not None

def test_parse_tz__no_match():
    if False:
        print('Hello World!')
    with pytest.raises(InvalidTZ):
        parse_tz('foo')