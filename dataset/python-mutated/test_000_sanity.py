from PIL import Image

def test_sanity():
    if False:
        return 10
    Image.core.new('L', (100, 100))
    im = Image.new('1', (100, 100))
    assert (im.mode, im.size) == ('1', (100, 100))
    assert len(im.tobytes()) == 1300
    Image.new('L', (100, 100))
    Image.new('P', (100, 100))
    Image.new('RGB', (100, 100))
    Image.new('I', (100, 100))
    Image.new('F', (100, 100))