import datetime
import sys
sys.path.insert(0, '.')
import pytest

@pytest.fixture
def uptime(xession, load_xontrib):
    if False:
        return 10
    load_xontrib('coreutils')
    return xession.aliases['uptime']

def test_uptime(uptime):
    if False:
        return 10
    out = uptime([])
    delta = datetime.timedelta(seconds=float(out))
    assert delta > datetime.timedelta(microseconds=1)