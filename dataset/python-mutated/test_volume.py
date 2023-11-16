import cairocffi
import pytest
from libqtile import bar, images
from libqtile.widget import Volume
from test.widgets.conftest import TEST_DIR

def test_images_fail():
    if False:
        i = 10
        return i + 15
    vol = Volume(theme_path=TEST_DIR)
    with pytest.raises(images.LoadingError):
        vol.setup_images()

def test_images_good(tmpdir, fake_bar, svg_img_as_pypath):
    if False:
        for i in range(10):
            print('nop')
    names = ('audio-volume-high.svg', 'audio-volume-low.svg', 'audio-volume-medium.svg', 'audio-volume-muted.svg')
    for name in names:
        target = tmpdir.join(name)
        svg_img_as_pypath.copy(target)
    vol = Volume(theme_path=str(tmpdir))
    vol.bar = fake_bar
    vol.length_type = bar.STATIC
    vol.length = 0
    vol.setup_images()
    assert len(vol.surfaces) == len(names)
    for (name, surfpat) in vol.surfaces.items():
        assert isinstance(surfpat, cairocffi.SurfacePattern)

def test_emoji():
    if False:
        return 10
    vol = Volume(emoji=True)
    vol.volume = -1
    vol._update_drawer()
    assert vol.text == '🔇'
    vol.volume = 29
    vol._update_drawer()
    assert vol.text == '🔈'
    vol.volume = 79
    vol._update_drawer()
    assert vol.text == '🔉'
    vol.volume = 80
    vol._update_drawer()
    assert vol.text == '🔊'

def test_text():
    if False:
        for i in range(10):
            print('nop')
    fmt = 'Volume: {}'
    vol = Volume(fmt=fmt)
    vol.volume = -1
    vol._update_drawer()
    assert vol.text == 'M'
    vol.volume = 50
    vol._update_drawer()
    assert vol.text == '50%'