from __future__ import annotations
import pytest
pytest
from bokeh.command.bootstrap import main
from tests.support.util.types import Capture
import bokeh.command.subcommands.sampledata as scsample
did_call_download = False

def test_create() -> None:
    if False:
        i = 10
        return i + 15
    import argparse
    from bokeh.command.subcommand import Subcommand
    obj = scsample.Sampledata(parser=argparse.ArgumentParser())
    assert isinstance(obj, Subcommand)

def test_name() -> None:
    if False:
        for i in range(10):
            print('nop')
    assert scsample.Sampledata.name == 'sampledata'

def test_help() -> None:
    if False:
        while True:
            i = 10
    assert scsample.Sampledata.help == 'Download the bokeh sample data sets'

def test_args() -> None:
    if False:
        i = 10
        return i + 15
    assert scsample.Sampledata.args == ()

def test_run(capsys: Capture) -> None:
    if False:
        i = 10
        return i + 15
    main(['bokeh', 'sampledata'])
    assert did_call_download is True

def _mock_download():
    if False:
        for i in range(10):
            print('nop')
    global did_call_download
    did_call_download = True
scsample.sampledata.download = _mock_download