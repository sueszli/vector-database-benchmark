from __future__ import annotations
import pytest
pytest
import bokeh.command as command

def test_doc() -> None:
    if False:
        for i in range(10):
            print('nop')
    import bokeh.command.subcommands as sc
    assert len(command.__doc__.split('\n')) == 5 + 3 * len(sc.all)
    for x in sc.all:
        assert x.name + '\n    ' + x.help in command.__doc__