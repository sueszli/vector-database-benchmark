from PIL import Image

def test_white():
    if False:
        i = 10
        return i + 15
    with Image.open('Tests/images/lab.tif') as i:
        i.load()
        assert i.mode == 'LAB'
        assert i.getbands() == ('L', 'A', 'B')
        k = i.getpixel((0, 0))
        L = i.getdata(0)
        a = i.getdata(1)
        b = i.getdata(2)
    assert k == (255, 128, 128)
    assert list(L) == [255] * 100
    assert list(a) == [128] * 100
    assert list(b) == [128] * 100

def test_green():
    if False:
        while True:
            i = 10
    with Image.open('Tests/images/lab-green.tif') as i:
        k = i.getpixel((0, 0))
    assert k == (128, 28, 128)

def test_red():
    if False:
        print('Hello World!')
    with Image.open('Tests/images/lab-red.tif') as i:
        k = i.getpixel((0, 0))
    assert k == (128, 228, 128)