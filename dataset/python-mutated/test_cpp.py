import pytest
from argparse import Namespace
from os import path
import tempfile
from grc.compiler import main

def test_cpp(capsys):
    if False:
        print('Hello World!')
    args = Namespace(output=tempfile.gettempdir(), user_lib_dir=False, grc_files=[path.join(path.dirname(__file__), 'resources', 'test_cpp.grc')], run=True)
    main(args)
    (out, err) = capsys.readouterr()
    assert not err