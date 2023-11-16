import pytest
from PIL import Image, ImageShow
from .helper import hopper, is_win32, on_ci

def test_sanity():
    if False:
        while True:
            i = 10
    dir(Image)
    dir(ImageShow)

def test_register():
    if False:
        while True:
            i = 10
    ImageShow.register('not a class')
    ImageShow._viewers.pop()

@pytest.mark.parametrize('order', [-1, 0])
def test_viewer_show(order):
    if False:
        return 10

    class TestViewer(ImageShow.Viewer):

        def show_image(self, image, **options):
            if False:
                return 10
            self.methodCalled = True
            return True
    viewer = TestViewer()
    ImageShow.register(viewer, order)
    for mode in ('1', 'I;16', 'LA', 'RGB', 'RGBA'):
        viewer.methodCalled = False
        with hopper(mode) as im:
            assert ImageShow.show(im)
        assert viewer.methodCalled
    ImageShow._viewers.pop(0)

@pytest.mark.skipif(not on_ci() or is_win32(), reason='Only run on CIs; hangs on Windows CIs')
@pytest.mark.parametrize('mode', ('1', 'I;16', 'LA', 'RGB', 'RGBA'))
def test_show(mode):
    if False:
        return 10
    im = hopper(mode)
    assert ImageShow.show(im)

def test_show_without_viewers():
    if False:
        i = 10
        return i + 15
    viewers = ImageShow._viewers
    ImageShow._viewers = []
    with hopper() as im:
        assert not ImageShow.show(im)
    ImageShow._viewers = viewers

def test_viewer():
    if False:
        while True:
            i = 10
    viewer = ImageShow.Viewer()
    assert viewer.get_format(None) is None
    with pytest.raises(NotImplementedError):
        viewer.get_command(None)

@pytest.mark.parametrize('viewer', ImageShow._viewers)
def test_viewers(viewer):
    if False:
        return 10
    try:
        viewer.get_command('test.jpg')
    except NotImplementedError:
        pass

def test_ipythonviewer():
    if False:
        for i in range(10):
            print('nop')
    pytest.importorskip('IPython', reason='IPython not installed')
    for viewer in ImageShow._viewers:
        if isinstance(viewer, ImageShow.IPythonViewer):
            test_viewer = viewer
            break
    else:
        pytest.fail()
    im = hopper()
    assert test_viewer.show(im) == 1