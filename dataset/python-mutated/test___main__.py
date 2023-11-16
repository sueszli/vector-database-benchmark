from __future__ import annotations
import pytest
pytest
from unittest.mock import MagicMock, patch
from tests.support.util.api import verify_all
import bokeh.__main__ as bm
ALL = ('main',)
Test___all__ = verify_all(bm, ALL)

@patch('bokeh.command.bootstrap.main')
def test_main(mock_main: MagicMock) -> None:
    if False:
        return 10
    import sys
    old_argv = sys.argv
    sys.argv = ['foo', 'bar']
    bm.main()
    assert mock_main.call_count == 1
    assert mock_main.call_args[0] == (['foo', 'bar'],)
    assert mock_main.call_args[1] == {}
    sys.argv = old_argv