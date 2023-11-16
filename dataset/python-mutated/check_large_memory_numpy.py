import sys
import pytest
from PIL import Image
np = pytest.importorskip('numpy', reason='NumPy not installed')
YDIM = 32769
XDIM = 48000
pytestmark = pytest.mark.skipif(sys.maxsize <= 2 ** 32, reason='requires 64-bit system')

def _write_png(tmp_path, xdim, ydim):
    if False:
        print('Hello World!')
    dtype = np.uint8
    a = np.zeros((xdim, ydim), dtype=dtype)
    f = str(tmp_path / 'temp.png')
    im = Image.fromarray(a, 'L')
    im.save(f)

def test_large(tmp_path):
    if False:
        return 10
    'succeeded prepatch'
    _write_png(tmp_path, XDIM, YDIM)

def test_2gpx(tmp_path):
    if False:
        return 10
    'failed prepatch'
    _write_png(tmp_path, XDIM, XDIM)