from __future__ import annotations
import pytest
pytest
from bokeh.command.bootstrap import main
from tests.support.util.types import Capture
import bokeh.command.subcommands.secret as scsecret

def test_create() -> None:
    if False:
        for i in range(10):
            print('nop')
    import argparse
    from bokeh.command.subcommand import Subcommand
    obj = scsecret.Secret(parser=argparse.ArgumentParser())
    assert isinstance(obj, Subcommand)

def test_name() -> None:
    if False:
        print('Hello World!')
    assert scsecret.Secret.name == 'secret'

def test_help() -> None:
    if False:
        while True:
            i = 10
    assert scsecret.Secret.help == 'Create a Bokeh secret key for use with Bokeh server'

def test_args() -> None:
    if False:
        while True:
            i = 10
    assert scsecret.Secret.args == ()

def test_run(capsys: Capture) -> None:
    if False:
        i = 10
        return i + 15
    main(['bokeh', 'secret'])
    (out, err) = capsys.readouterr()
    assert err == ''
    assert len(out) == 45
    assert out[-1] == '\n'