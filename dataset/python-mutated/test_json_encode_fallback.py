from __future__ import annotations
import pytest
from datetime import datetime, timedelta, tzinfo
from ansible.module_utils.common.text.converters import _json_encode_fallback

class timezone(tzinfo):
    """Simple timezone implementation for use until we drop Python 2.7 support."""

    def __init__(self, offset):
        if False:
            print('Hello World!')
        self._offset = offset

    def utcoffset(self, dt):
        if False:
            for i in range(10):
                print('nop')
        return self._offset

@pytest.mark.parametrize('test_input,expected', [(set([1]), [1]), (datetime(2019, 5, 14, 13, 39, 38, 569047), '2019-05-14T13:39:38.569047'), (datetime(2019, 5, 14, 13, 47, 16, 923866), '2019-05-14T13:47:16.923866'), (datetime(2019, 6, 15, 14, 45, tzinfo=timezone(timedelta(0))), '2019-06-15T14:45:00+00:00'), (datetime(2019, 6, 15, 14, 45, tzinfo=timezone(timedelta(hours=1, minutes=40))), '2019-06-15T14:45:00+01:40')])
def test_json_encode_fallback(test_input, expected):
    if False:
        return 10
    '\n    Test for passing expected objects to _json_encode_fallback().\n    '
    assert _json_encode_fallback(test_input) == expected

@pytest.mark.parametrize('test_input', [1, 1.1, u'string', b'string', [1, 2], True, None, {1: 1}, (1, 2)])
def test_json_encode_fallback_default_behavior(test_input):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for _json_encode_fallback() default behavior.\n\n    It must fail with TypeError.\n    '
    with pytest.raises(TypeError, match='Cannot json serialize'):
        _json_encode_fallback(test_input)