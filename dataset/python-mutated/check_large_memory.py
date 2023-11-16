import sys
import pytest
from PIL import Image
try:
    import numpy
except ImportError:
    numpy = None
YDIM = 32769
XDIM = 48000
pytestmark = pytest.mark.skipif(sys.maxsize <= 2 ** 32, reason='requires 64-bit system')

def _write_png(tmp_path, xdim, ydim):
    if False:
        while True:
            i = 10
    f = str(tmp_path / 'temp.png')
    im = Image.new('L', (xdim, ydim), 0)
    im.save(f)

def test_large(tmp_path):
    if False:
        i = 10
        return i + 15
    'succeeded prepatch'
    _write_png(tmp_path, XDIM, YDIM)

def test_2gpx(tmp_path):
    if False:
        print('Hello World!')
    'failed prepatch'
    _write_png(tmp_path, XDIM, XDIM)

@pytest.mark.skipif(numpy is None, reason='Numpy is not installed')
def test_size_greater_than_int():
    if False:
        while True:
            i = 10
    arr = numpy.ndarray(shape=(16394, 16394))
    Image.fromarray(arr)