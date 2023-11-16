import platform
import pytest
from hypothesis import given, strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.strategies._internal.datetime import zoneinfo
from tests.common.debug import assert_no_examples, find_any, minimal

def test_utc_is_minimal():
    if False:
        return 10
    assert minimal(st.timezones()) is zoneinfo.ZoneInfo('UTC')

def test_can_generate_non_utc():
    if False:
        i = 10
        return i + 15
    find_any(st.datetimes(timezones=st.timezones()).filter(lambda d: d.tzinfo.key != 'UTC'))

@given(st.data(), st.datetimes(), st.datetimes())
def test_datetimes_stay_within_naive_bounds(data, lo, hi):
    if False:
        print('Hello World!')
    if lo > hi:
        (lo, hi) = (hi, lo)
    out = data.draw(st.datetimes(lo, hi, timezones=st.timezones()))
    assert lo <= out.replace(tzinfo=None) <= hi

@pytest.mark.parametrize('kwargs', [{'no_cache': 1}])
def test_timezones_argument_validation(kwargs):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(InvalidArgument):
        st.timezones(**kwargs).validate()

@pytest.mark.parametrize('kwargs', [{'allow_prefix': 1}])
def test_timezone_keys_argument_validation(kwargs):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(InvalidArgument):
        st.timezone_keys(**kwargs).validate()

@pytest.mark.skipif(platform.system() != 'Linux', reason='platform-specific')
def test_can_generate_prefixes_if_allowed_and_available():
    if False:
        return 10
    '\n    This is actually kinda fiddly: we may generate timezone keys with the\n    "posix/" or "right/" prefix if-and-only-if they are present on the filesystem.\n\n    This immediately rules out Windows (which uses the tzdata package instead),\n    along with OSX (which doesn\'t seem to have prefixed keys).  We believe that\n    they are present on at least most Linux distros, but have not done exhaustive\n    testing.\n\n    It\'s fine to just patch this test out if it fails - passing in the\n    Hypothesis CI demonstrates that the feature works on *some* systems.\n    '
    find_any(st.timezone_keys(), lambda s: s.startswith('posix/'))
    find_any(st.timezone_keys(), lambda s: s.startswith('right/'))

def test_can_disallow_prefixes():
    if False:
        print('Hello World!')
    assert_no_examples(st.timezone_keys(allow_prefix=False), lambda s: s.startswith(('posix/', 'right/')))