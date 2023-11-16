from __future__ import annotations
import pytest
pytest
import argparse
import os
from bokeh.command.bootstrap import main
from bokeh.command.subcommand import Argument
from tests.support.util.filesystem import TmpDir, WorkingDir, with_directory_contents
from tests.support.util.types import Capture
from _util_subcommands import basic_scatter_script
import bokeh.command.subcommands.json as scjson

def test_create() -> None:
    if False:
        return 10
    import argparse
    from bokeh.command.subcommand import Subcommand
    obj = scjson.JSON(parser=argparse.ArgumentParser())
    assert isinstance(obj, Subcommand)

def test_name() -> None:
    if False:
        i = 10
        return i + 15
    assert scjson.JSON.name == 'json'

def test_help() -> None:
    if False:
        for i in range(10):
            print('nop')
    assert scjson.JSON.help == 'Create JSON files for one or more applications'

def test_args() -> None:
    if False:
        i = 10
        return i + 15
    assert scjson.JSON.args == (('files', Argument(metavar='DIRECTORY-OR-SCRIPT', nargs='+', help='The app directories or scripts to generate JSON for', default=None)), ('--indent', Argument(metavar='LEVEL', type=int, help='indentation to use when printing', default=None)), (('-o', '--output'), Argument(metavar='FILENAME', action='append', type=str, help='Name of the output file or - for standard output.')), ('--args', Argument(metavar='COMMAND-LINE-ARGS', nargs=argparse.REMAINDER, help='Any command line arguments remaining are passed on to the application handler')))

def test_no_script(capsys: Capture) -> None:
    if False:
        while True:
            i = 10
    with TmpDir(prefix='bokeh-json-no-script') as dirname:
        with WorkingDir(dirname):
            with pytest.raises(SystemExit):
                main(['bokeh', 'json'])
        (out, err) = capsys.readouterr()
        too_few = 'the following arguments are required: DIRECTORY-OR-SCRIPT'
        assert err == f'usage: bokeh json [-h] [--indent LEVEL] [-o FILENAME] [--args ...]\n                  DIRECTORY-OR-SCRIPT [DIRECTORY-OR-SCRIPT ...]\nbokeh json: error: {too_few}\n'
        assert out == ''

def test_basic_script(capsys: Capture) -> None:
    if False:
        for i in range(10):
            print('nop')

    def run(dirname: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        with WorkingDir(dirname):
            main(['bokeh', 'json', 'scatter.py'])
        (out, err) = capsys.readouterr()
        assert err == ''
        assert out == ''
        assert {'scatter.json', 'scatter.py'} == set(os.listdir(dirname))
    with_directory_contents({'scatter.py': basic_scatter_script}, run)

def test_basic_script_with_output_after(capsys: Capture) -> None:
    if False:
        while True:
            i = 10

    def run(dirname: str) -> None:
        if False:
            i = 10
            return i + 15
        with WorkingDir(dirname):
            main(['bokeh', 'json', 'scatter.py', '--output', 'foo.json'])
        (out, err) = capsys.readouterr()
        assert err == ''
        assert out == ''
        assert {'foo.json', 'scatter.py'} == set(os.listdir(dirname))
    with_directory_contents({'scatter.py': basic_scatter_script}, run)

def test_basic_script_with_output_before(capsys: Capture) -> None:
    if False:
        for i in range(10):
            print('nop')

    def run(dirname: str) -> None:
        if False:
            i = 10
            return i + 15
        with WorkingDir(dirname):
            main(['bokeh', 'json', '--output', 'foo.json', 'scatter.py'])
        (out, err) = capsys.readouterr()
        assert err == ''
        assert out == ''
        assert {'foo.json', 'scatter.py'} == set(os.listdir(dirname))
    with_directory_contents({'scatter.py': basic_scatter_script}, run)