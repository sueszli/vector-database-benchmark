from __future__ import annotations
import pytest
pytest
from unittest.mock import MagicMock, patch
import bokeh.util.deprecation as dep
import bokeh.util.warnings as warn

@patch('warnings.warn')
def test_find_stack_level(mock_warn: MagicMock) -> None:
    if False:
        for i in range(10):
            print('nop')
    assert warn.find_stack_level() == 1
    warn.warn('test')
    assert mock_warn.call_count == 1
    assert mock_warn.call_args[1] == {'stacklevel': 2}
    dep.deprecated((1, 2, 3), old='foo', new='bar', extra='baz')
    assert mock_warn.call_count == 2
    assert mock_warn.call_args[1] == {'stacklevel': 3}