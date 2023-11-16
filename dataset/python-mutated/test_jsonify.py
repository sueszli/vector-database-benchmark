from __future__ import annotations
import pytest
from ansible.module_utils.common.text.converters import jsonify

@pytest.mark.parametrize('test_input,expected', [(1, '1'), (u'string', u'"string"'), (u'くらとみ', u'"\\u304f\\u3089\\u3068\\u307f"'), (u'café', u'"caf\\u00e9"'), (b'string', u'"string"'), (False, u'false'), (u'string'.encode('utf-8'), u'"string"')])
def test_jsonify(test_input, expected):
    if False:
        while True:
            i = 10
    'Test for jsonify().'
    assert jsonify(test_input) == expected