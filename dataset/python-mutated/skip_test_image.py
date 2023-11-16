import os
import cv2
import numpy as np
import pytest
from PIL import Image
import uiautomator2.image as u2image
TESTDIR = os.path.dirname(os.path.abspath(__file__)) + '/testdata'

@pytest.fixture
def path_ae86():
    if False:
        i = 10
        return i + 15
    filepath = os.path.join(TESTDIR, './AE86.jpg')
    return filepath

@pytest.fixture
def im_ae86(path_ae86: str) -> np.ndarray:
    if False:
        return 10
    ' 使用opencv打开的图片 '
    im = cv2.imread(path_ae86)
    return im

def test_imread(im_ae86, path_ae86):
    if False:
        while True:
            i = 10
    im = u2image.imread(path_ae86)
    assert im.shape == (193, 321, 3)
    im = u2image.imread('https://www.baidu.com/img/bd_logo1.png')
    assert im.shape == (258, 540, 3)
    im = u2image.imread(im_ae86)
    assert im.shape == (193, 321, 3), '图片格式变化'
    pilim = Image.open(path_ae86)
    im = u2image.imread(pilim)
    assert pilim.size == (321, 193)
    assert im.shape == (193, 321, 3), '图片格式变化'

@pytest.mark.skip('missing test images')
def test_image_match():
    if False:
        print('Hello World!')

    class MockDevice:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x = None
            self.y = None

        def click(self, x, y):
            if False:
                while True:
                    i = 10
            self.x = x
            self.y = y

        def screenshot(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            return cv2.imread(TESTDIR + '/screenshot.jpg')
    d = MockDevice()
    ix = u2image.ImageX(d)
    template = Image.open(TESTDIR + '/template.jpg')
    res = ix.match(template)
    (x, y) = res['point']
    assert (x, y) == (409, 659), 'Match position is wrong'
    ix.click(template)
    assert d.x == 409
    assert d.y == 659
    if False:
        pim = Image.open(TESTDIR + '/screenshot.jpg')
        nim = u2image.draw_point(pim, x, y)
        nim.show()