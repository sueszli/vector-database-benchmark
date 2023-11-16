import pytest
from PIL import Image
TEST_FILE = 'Tests/images/libtiff_segfault.tif'

def test_libtiff_segfault():
    if False:
        for i in range(10):
            print('nop')
    'This test should not segfault. It will on Pillow <= 3.1.0 and\n    libtiff >= 4.0.0\n    '
    with pytest.raises(OSError):
        with Image.open(TEST_FILE) as im:
            im.load()