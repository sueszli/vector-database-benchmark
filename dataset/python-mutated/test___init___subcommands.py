from __future__ import annotations
import pytest
pytest
import bokeh.command.subcommands as sc

def test_all() -> None:
    if False:
        for i in range(10):
            print('nop')
    assert hasattr(sc, 'all')
    assert isinstance(sc.all, list)

def test_all_types() -> None:
    if False:
        return 10
    from bokeh.command.subcommand import Subcommand
    assert all((issubclass(x, Subcommand) for x in sc.all))

def test_all_count() -> None:
    if False:
        i = 10
        return i + 15
    from os import listdir
    from os.path import dirname
    files = listdir(dirname(sc.__file__))
    pyfiles = [x for x in files if x.endswith('.py')]
    assert len(sc.all) == len(pyfiles) - 2