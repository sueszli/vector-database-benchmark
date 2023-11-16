import pytest
from PIL import ImageQt
from .helper import assert_image_equal, assert_image_equal_tofile, hopper
pytestmark = pytest.mark.skipif(not ImageQt.qt_is_installed, reason='Qt bindings are not installed')
if ImageQt.qt_is_installed:
    from PIL.ImageQt import QImage

@pytest.mark.parametrize('mode', ('RGB', 'RGBA', 'L', 'P', '1'))
def test_sanity(mode, tmp_path):
    if False:
        while True:
            i = 10
    src = hopper(mode)
    data = ImageQt.toqimage(src)
    assert isinstance(data, QImage)
    assert not data.isNull()
    rt = ImageQt.fromqimage(data)
    if mode in ('L', 'P', '1'):
        assert_image_equal(rt, src.convert('RGB'))
    else:
        assert_image_equal(rt, src)
    if mode == '1':
        return
    tempfile = str(tmp_path / f'temp_{mode}.png')
    data.save(tempfile)
    assert_image_equal_tofile(src, tempfile)