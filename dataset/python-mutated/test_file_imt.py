import io
import pytest
from PIL import Image, ImtImagePlugin
from .helper import assert_image_equal_tofile

def test_sanity():
    if False:
        i = 10
        return i + 15
    with Image.open('Tests/images/bw_gradient.imt') as im:
        assert_image_equal_tofile(im, 'Tests/images/bw_gradient.png')

@pytest.mark.parametrize('data', (b'\n', b'\n-', b'width 1\n'))
def test_invalid_file(data):
    if False:
        i = 10
        return i + 15
    with io.BytesIO(data) as fp:
        with pytest.raises(SyntaxError):
            ImtImagePlugin.ImtImageFile(fp)