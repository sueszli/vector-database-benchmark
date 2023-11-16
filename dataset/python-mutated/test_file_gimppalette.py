import pytest
from PIL.GimpPaletteFile import GimpPaletteFile

def test_sanity():
    if False:
        for i in range(10):
            print('nop')
    with open('Tests/images/test.gpl', 'rb') as fp:
        GimpPaletteFile(fp)
    with open('Tests/images/hopper.jpg', 'rb') as fp:
        with pytest.raises(SyntaxError):
            GimpPaletteFile(fp)
    with open('Tests/images/bad_palette_file.gpl', 'rb') as fp:
        with pytest.raises(SyntaxError):
            GimpPaletteFile(fp)
    with open('Tests/images/bad_palette_entry.gpl', 'rb') as fp:
        with pytest.raises(ValueError):
            GimpPaletteFile(fp)

def test_get_palette():
    if False:
        print('Hello World!')
    with open('Tests/images/custom_gimp_palette.gpl', 'rb') as fp:
        palette_file = GimpPaletteFile(fp)
    (palette, mode) = palette_file.getpalette()
    assert mode == 'RGB'