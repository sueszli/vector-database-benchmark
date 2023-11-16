import io
import os.path
import platform
from pathlib import Path
import PIL
import pytest
from pyboy import PyBoy

@pytest.mark.parametrize('rom', ['LYC.gb', 'sprite_suite.gb'])
def test_shonumi(rom, shonumi_dir):
    if False:
        return 10
    pyboy = PyBoy(shonumi_dir + rom, window_type='headless', color_palette=(16777215, 10066329, 6316128, 0))
    pyboy.set_emulation_speed(0)
    for _ in range(60 + 23 + 48):
        pyboy.tick()
    png_path = Path(f'tests/test_results/GB Tests/{rom}.png')
    png_path.parents[0].mkdir(parents=True, exist_ok=True)
    image = pyboy.botsupport_manager().screen().screen_image()
    old_image = PIL.Image.open(png_path)
    old_image = old_image.resize(image.size, resample=PIL.Image.Dither.NONE)
    diff = PIL.ImageChops.difference(image, old_image)
    if diff.getbbox() and (not os.environ.get('TEST_CI')):
        image.show()
        old_image.show()
        diff.show()
    assert not diff.getbbox(), f'Images are different! {rom}'
    pyboy.stop(save=False)