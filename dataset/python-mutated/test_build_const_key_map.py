import pytest
from xdis.version_info import PYTHON_VERSION_TRIPLE, IS_PYPY
from validate import validate_uncompyle

@pytest.mark.skipif(PYTHON_VERSION_TRIPLE < (3, 6) or IS_PYPY, reason='need at least Python 3.6 and not PyPY')
@pytest.mark.parametrize('text', ("{0.: 'a', -1: 'b'}", "{'a':'b'}", '{0: 1}', "{b'0':1, b'2':3}", '{0: 1, 2: 3}', "{'a':'b','c':'d'}", '{0: 1, 2: 3}', "{'a': 1, 'b': 2}", "{'a':'b','c':'d'}", "{0.0:'b',0.1:'d'}"))
def test_build_const_key_map(text):
    if False:
        while True:
            i = 10
    validate_uncompyle(text)