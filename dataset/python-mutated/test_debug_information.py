import re
import pytest
from hypothesis import Verbosity, given, settings, strategies as st
from tests.common.utils import capture_out

def test_reports_passes():
    if False:
        return 10

    @given(st.integers())
    @settings(verbosity=Verbosity.debug, max_examples=1000)
    def test(i):
        if False:
            return 10
        assert i < 10
    with capture_out() as out:
        with pytest.raises(AssertionError):
            test()
    value = out.getvalue()
    assert 'minimize_individual_blocks' in value
    assert 'calls' in value
    assert 'shrinks' in value
    shrinks_info = re.compile('call(s?) of which ([0-9]+) shrank')
    for l in value.splitlines():
        m = shrinks_info.search(l)
        if m is not None and int(m.group(2)) != 0:
            break
    else:
        pytest.xfail(reason='Sometimes the first failure is 10, and cannot shrink.')