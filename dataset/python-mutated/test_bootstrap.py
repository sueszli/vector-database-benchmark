from __future__ import annotations
import pytest
pytest
from bokeh import __version__
from tests.support.util.types import Capture
from bokeh.command.bootstrap import main

def _assert_version_output(capsys: Capture):
    if False:
        i = 10
        return i + 15
    (out, err) = capsys.readouterr()
    err_expected = ''
    out_expected = f'{__version__}\n'
    assert err == err_expected
    assert out == out_expected

def test_no_subcommand(capsys: Capture) -> None:
    if False:
        i = 10
        return i + 15
    with pytest.raises(SystemExit):
        main(['bokeh'])
    (out, err) = capsys.readouterr()
    assert err == 'ERROR: Must specify subcommand, one of: build, info, init, json, sampledata, secret, serve or static\n'
    assert out == ''

def test_version(capsys: Capture) -> None:
    if False:
        print('Hello World!')
    with pytest.raises(SystemExit):
        main(['bokeh', '--version'])
    _assert_version_output(capsys)

def test_version_short(capsys: Capture) -> None:
    if False:
        i = 10
        return i + 15
    with pytest.raises(SystemExit):
        main(['bokeh', '-v'])
    _assert_version_output(capsys)

def test_error(capsys: Capture) -> None:
    if False:
        while True:
            i = 10
    from bokeh.command.subcommands.info import Info
    old_invoke = Info.invoke

    def err(x, y):
        if False:
            return 10
        raise RuntimeError('foo')
    Info.invoke = err
    with pytest.raises(SystemExit):
        main(['bokeh', 'info'])
    (out, err) = capsys.readouterr()
    assert err == 'ERROR: foo\n'
    Info.invoke = old_invoke