import sys
from io import BytesIO, StringIO
import pytest
from PIL import Image, IptcImagePlugin
from .helper import hopper
TEST_FILE = 'Tests/images/iptc.jpg'

def test_getiptcinfo_jpg_none():
    if False:
        i = 10
        return i + 15
    with hopper() as im:
        iptc = IptcImagePlugin.getiptcinfo(im)
    assert iptc is None

def test_getiptcinfo_jpg_found():
    if False:
        return 10
    with Image.open(TEST_FILE) as im:
        iptc = IptcImagePlugin.getiptcinfo(im)
    assert isinstance(iptc, dict)
    assert iptc[2, 90] == b'Budapest'
    assert iptc[2, 101] == b'Hungary'

def test_getiptcinfo_fotostation():
    if False:
        return 10
    with open(TEST_FILE, 'rb') as fp:
        data = bytearray(fp.read())
    data[86] = 240
    f = BytesIO(data)
    with Image.open(f) as im:
        iptc = IptcImagePlugin.getiptcinfo(im)
    for tag in iptc.keys():
        if tag[0] == 240:
            return
    pytest.fail('FotoStation tag not found')

def test_getiptcinfo_zero_padding():
    if False:
        i = 10
        return i + 15
    with Image.open(TEST_FILE) as im:
        im.info['photoshop'][1028] += b'\x00\x00\x00'
        iptc = IptcImagePlugin.getiptcinfo(im)
    assert isinstance(iptc, dict)
    assert len(iptc) == 3

def test_getiptcinfo_tiff_none():
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/hopper.tif') as im:
        iptc = IptcImagePlugin.getiptcinfo(im)
    assert iptc is None

def test_i():
    if False:
        while True:
            i = 10
    c = b'a'
    ret = IptcImagePlugin.i(c)
    assert ret == 97

def test_dump():
    if False:
        while True:
            i = 10
    c = b'abc'
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    IptcImagePlugin.dump(c)
    sys.stdout = old_stdout
    assert mystdout.getvalue() == '61 62 63 \n'