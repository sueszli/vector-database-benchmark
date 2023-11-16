import os
from PIL import Image
from faces import main
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

def test_main(tmpdir):
    if False:
        print('Hello World!')
    out_file = os.path.join(tmpdir.dirname, 'face-output.jpg')
    in_file = os.path.join(RESOURCES, 'face-input.jpg')
    im = Image.open(in_file)
    pixels = im.getdata()
    greens = sum((1 for (r, g, b) in pixels if r == 0 and g == 255 and (b == 0)))
    assert greens < 1
    main(in_file, out_file, 10)
    im = Image.open(out_file)
    pixels = im.getdata()
    greens = sum((1 for (r, g, b) in pixels if r == 0 and g == 255 and (b == 0)))
    assert greens > 10