from __future__ import annotations
import pytest
from pyupgrade._data import Settings
from pyupgrade._main import _fix_plugins

@pytest.mark.parametrize('s', ('from six import MAXSIZE as text_type\nisinstance(s, text_type)\n', '(\n    six\n).text_type(u)\n', pytest.param('from .six import text_type\nisinstance("foo", text_type)\n', id='relative import might not be six'), pytest.param('foo.range(3)', id='Range, but not from six.moves')))
def test_six_simple_noop(s):
    if False:
        i = 10
        return i + 15
    assert _fix_plugins(s, settings=Settings()) == s

@pytest.mark.parametrize(('s', 'expected'), (('isinstance(s, six.text_type)', 'isinstance(s, str)'), pytest.param('isinstance(s, six   .    string_types)', 'isinstance(s, str)', id='weird spacing on six.attr'), ('isinstance(s, six.string_types)', 'isinstance(s, str)'), ('issubclass(tp, six.string_types)', 'issubclass(tp, str)'), ('STRING_TYPES = six.string_types', 'STRING_TYPES = (str,)'), ('from six import string_types\nisinstance(s, string_types)\n', 'from six import string_types\nisinstance(s, str)\n'), ('from six import string_types\nSTRING_TYPES = string_types\n', 'from six import string_types\nSTRING_TYPES = (str,)\n'), pytest.param('six.moves.range(3)\n', 'range(3)\n', id='six.moves.range'), pytest.param('six.moves.xrange(3)\n', 'range(3)\n', id='six.moves.xrange'), pytest.param('from six.moves import xrange\nxrange(3)\n', 'from six.moves import xrange\nrange(3)\n', id='six.moves.xrange, from import')))
def test_fix_six_simple(s, expected):
    if False:
        print('Hello World!')
    ret = _fix_plugins(s, settings=Settings())
    assert ret == expected