import platform
import pytest

@pytest.fixture
def uname(xession, load_xontrib):
    if False:
        i = 10
        return i + 15
    load_xontrib('coreutils')
    return xession.aliases['uname']

def test_uname_without_args(uname):
    if False:
        return 10
    out = uname(['-a'])
    assert out.startswith(platform.uname().system)