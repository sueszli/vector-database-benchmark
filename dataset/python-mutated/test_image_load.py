import logging
import os
import pytest
from PIL import Image
from .helper import hopper

def test_sanity():
    if False:
        print('Hello World!')
    im = hopper()
    pix = im.load()
    assert pix[0, 0] == (20, 20, 70)

def test_close():
    if False:
        print('Hello World!')
    im = Image.open('Tests/images/hopper.gif')
    im.close()
    with pytest.raises(ValueError):
        im.load()
    with pytest.raises(ValueError):
        im.getpixel((0, 0))

def test_close_after_load(caplog):
    if False:
        i = 10
        return i + 15
    im = Image.open('Tests/images/hopper.gif')
    im.load()
    with caplog.at_level(logging.DEBUG):
        im.close()
    assert len(caplog.records) == 0

def test_contextmanager():
    if False:
        print('Hello World!')
    fn = None
    with Image.open('Tests/images/hopper.gif') as im:
        fn = im.fp.fileno()
        os.fstat(fn)
    with pytest.raises(OSError):
        os.fstat(fn)

def test_contextmanager_non_exclusive_fp():
    if False:
        i = 10
        return i + 15
    with open('Tests/images/hopper.gif', 'rb') as fp:
        with Image.open(fp):
            pass
        assert not fp.closed