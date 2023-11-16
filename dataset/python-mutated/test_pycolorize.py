"""Test suite for our color utilities.

Authors
-------

* Min RK
"""
import sys
from IPython.utils.PyColorize import Parser
import io
import pytest

@pytest.fixture(scope='module', params=('Linux', 'NoColor', 'LightBG', 'Neutral'))
def style(request):
    if False:
        return 10
    yield request.param
sample = '\ndef function(arg, *args, kwarg=True, **kwargs):\n    \'\'\'\n    this is docs\n    \'\'\'\n    pass is True\n    False == None\n\n    with io.open(ru\'unicode\', encoding=\'utf-8\'):\n        raise ValueError("escape \r sequence")\n\n    print("wěird ünicoðe")\n\nclass Bar(Super):\n\n    def __init__(self):\n        super(Bar, self).__init__(1**2, 3^4, 5 or 6)\n'

def test_parse_sample(style):
    if False:
        for i in range(10):
            print('nop')
    'and test writing to a buffer'
    buf = io.StringIO()
    p = Parser(style=style)
    p.format(sample, buf)
    buf.seek(0)
    f1 = buf.read()
    assert 'ERROR' not in f1

def test_parse_error(style):
    if False:
        i = 10
        return i + 15
    p = Parser(style=style)
    f1 = p.format('\\ ' if sys.version_info >= (3, 12) else ')', 'str')
    if style != 'NoColor':
        assert 'ERROR' in f1