from fractions import Fraction
from PIL import Image, TiffImagePlugin, features
from PIL.TiffImagePlugin import IFDRational
from .helper import hopper

def _test_equal(num, denom, target):
    if False:
        return 10
    t = IFDRational(num, denom)
    assert target == t
    assert t == target

def test_sanity():
    if False:
        print('Hello World!')
    _test_equal(1, 1, 1)
    _test_equal(1, 1, Fraction(1, 1))
    _test_equal(2, 2, 1)
    _test_equal(1.0, 1, Fraction(1, 1))
    _test_equal(Fraction(1, 1), 1, Fraction(1, 1))
    _test_equal(IFDRational(1, 1), 1, 1)
    _test_equal(1, 2, Fraction(1, 2))
    _test_equal(1, 2, IFDRational(1, 2))
    _test_equal(7, 5, 1.4)

def test_ranges():
    if False:
        while True:
            i = 10
    for num in range(1, 10):
        for denom in range(1, 10):
            assert IFDRational(num, denom) == IFDRational(num, denom)

def test_nonetype():
    if False:
        while True:
            i = 10
    xres = IFDRational(72)
    yres = IFDRational(72)
    assert xres._val is not None
    assert xres.numerator is not None
    assert xres.denominator is not None
    assert yres._val is not None
    assert xres and 1
    assert xres and yres

def test_ifd_rational_save(tmp_path):
    if False:
        i = 10
        return i + 15
    methods = (True, False)
    if not features.check('libtiff'):
        methods = (False,)
    for libtiff in methods:
        TiffImagePlugin.WRITE_LIBTIFF = libtiff
        im = hopper()
        out = str(tmp_path / 'temp.tiff')
        res = IFDRational(301, 1)
        im.save(out, dpi=(res, res), compression='raw')
        with Image.open(out) as reloaded:
            assert float(IFDRational(301, 1)) == float(reloaded.tag_v2[282])