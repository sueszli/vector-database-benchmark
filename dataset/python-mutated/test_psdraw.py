import os
import sys
from io import BytesIO
import pytest
from PIL import Image, PSDraw

def _create_document(ps):
    if False:
        return 10
    title = 'hopper'
    box = (1 * 72, 2 * 72, 7 * 72, 10 * 72)
    ps.begin_document(title)
    ps.line((1 * 72, 2 * 72), (7 * 72, 10 * 72))
    ps.line((7 * 72, 2 * 72), (1 * 72, 10 * 72))
    with Image.open('Tests/images/hopper.ppm') as im:
        ps.image(box, im, 75)
    ps.rectangle(box)
    ps.setfont('Courier', 36)
    ps.text((3 * 72, 4 * 72), title)
    ps.end_document()

def test_draw_postscript(tmp_path):
    if False:
        print('Hello World!')
    tempfile = str(tmp_path / 'temp.ps')
    with open(tempfile, 'wb') as fp:
        ps = PSDraw.PSDraw(fp)
        _create_document(ps)
    assert os.path.isfile(tempfile)
    assert os.path.getsize(tempfile) > 0

@pytest.mark.parametrize('buffer', (True, False))
def test_stdout(buffer):
    if False:
        print('Hello World!')
    old_stdout = sys.stdout
    if buffer:

        class MyStdOut:
            buffer = BytesIO()
        mystdout = MyStdOut()
    else:
        mystdout = BytesIO()
    sys.stdout = mystdout
    ps = PSDraw.PSDraw()
    _create_document(ps)
    sys.stdout = old_stdout
    if buffer:
        mystdout = mystdout.buffer
    assert mystdout.getvalue() != b''