import pytest
from .helper import assert_image_equal, fromstring, hopper

def test_sanity():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError):
        hopper().tobitmap()
    im1 = hopper().convert('1')
    bitmap = im1.tobitmap()
    assert isinstance(bitmap, bytes)
    assert_image_equal(im1, fromstring(bitmap))