from __future__ import annotations
import pytest
pytest
from unittest.mock import MagicMock
import bokeh.command.subcommand as sc

class _Bad(sc.Subcommand):
    pass

class _Good(sc.Subcommand):

    def invoke(self, args):
        if False:
            i = 10
            return i + 15
        pass

def test_is_abstract() -> None:
    if False:
        return 10
    with pytest.raises(TypeError):
        _Bad()

def test_missing_args() -> None:
    if False:
        while True:
            i = 10
    p = MagicMock()
    _Good(p)
    p.add_argument.assert_not_called()

def test_no_args() -> None:
    if False:
        while True:
            i = 10
    _Good.args = ()
    p = MagicMock()
    _Good(p)
    p.add_argument.assert_not_called()

def test_one_arg() -> None:
    if False:
        return 10
    _Good.args = (('foo', sc.Argument(nargs=1, help='foo')),)
    p = MagicMock()
    _Good(p)
    assert p.add_argument.call_count == 1

def test_args() -> None:
    if False:
        return 10
    _Good.args = (('foo', sc.Argument(nargs=1, help='foo')), ('bar', sc.Argument(nargs=2, help='bar')))
    p = MagicMock()
    _Good(p)
    assert p.add_argument.call_count == 2

def test_base_invoke() -> None:
    if False:
        i = 10
        return i + 15
    with pytest.raises(NotImplementedError):
        p = MagicMock()
        obj = _Good(p)
        super(_Good, obj).invoke('foo')