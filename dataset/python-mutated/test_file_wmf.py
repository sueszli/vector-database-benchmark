import pytest
from PIL import Image, WmfImagePlugin
from .helper import assert_image_similar_tofile, hopper

def test_load_raw():
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/drawing.emf') as im:
        if hasattr(Image.core, 'drawwmf'):
            im.load()
            assert_image_similar_tofile(im, 'Tests/images/drawing_emf_ref.png', 0)
    with Image.open('Tests/images/drawing.wmf') as im:
        if hasattr(Image.core, 'drawwmf'):
            im.load()
            assert_image_similar_tofile(im, 'Tests/images/drawing_wmf_ref.png', 2.0)

def test_load():
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/drawing.emf') as im:
        if hasattr(Image.core, 'drawwmf'):
            assert im.load()[0, 0] == (255, 255, 255)

def test_register_handler(tmp_path):
    if False:
        while True:
            i = 10

    class TestHandler:
        methodCalled = False

        def save(self, im, fp, filename):
            if False:
                print('Hello World!')
            self.methodCalled = True
    handler = TestHandler()
    original_handler = WmfImagePlugin._handler
    WmfImagePlugin.register_handler(handler)
    im = hopper()
    tmpfile = str(tmp_path / 'temp.wmf')
    im.save(tmpfile)
    assert handler.methodCalled
    WmfImagePlugin.register_handler(original_handler)

def test_load_float_dpi():
    if False:
        print('Hello World!')
    with Image.open('Tests/images/drawing.emf') as im:
        assert im.info['dpi'] == 1423.7668161434979

def test_load_set_dpi():
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/drawing.wmf') as im:
        assert im.size == (82, 82)
        if hasattr(Image.core, 'drawwmf'):
            im.load(144)
            assert im.size == (164, 164)
            assert_image_similar_tofile(im, 'Tests/images/drawing_wmf_ref_144.png', 2.1)

@pytest.mark.parametrize('ext', ('.wmf', '.emf'))
def test_save(ext, tmp_path):
    if False:
        return 10
    im = hopper()
    tmpfile = str(tmp_path / ('temp' + ext))
    with pytest.raises(OSError):
        im.save(tmpfile)