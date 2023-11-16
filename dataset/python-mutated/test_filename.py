from panda3d.core import Filename
import sys, os
import pytest

def test_filename_fspath():
    if False:
        return 10
    fn = Filename.from_os_specific(__file__)
    assert os.fspath(fn) == fn.to_os_specific_w()

def test_filename_open():
    if False:
        return 10
    fn = Filename.from_os_specific(__file__)
    open(fn, 'rb')

def test_filename_ctor_pathlib():
    if False:
        while True:
            i = 10
    pathlib = pytest.importorskip('pathlib')
    path = pathlib.Path(__file__)
    fn = Filename(path)
    assert fn.to_os_specific_w().lower() == str(path).lower()