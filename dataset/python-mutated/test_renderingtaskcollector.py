import os
import random
import numpy
import cv2
import pytest
from golem.tools.testdirfixture import TestDirFixture
from apps.rendering.resources.renderingtaskcollector import RenderingTaskCollector
from apps.rendering.resources.imgrepr import OpenCVImgRepr, OpenCVError

def make_test_img(img_path, size=(10, 10), color=(255, 0, 0)):
    if False:
        print('Hello World!')
    img = numpy.zeros((size[0], size[1], 3), numpy.uint8)
    img[:] = tuple(reversed(color))
    cv2.imwrite(img_path, img)

def make_test_img_16bits(img_path, width, height, color=(0, 0, 255)):
    if False:
        return 10
    img = numpy.zeros((height, width, 3), numpy.uint16)
    img[0:height, 0:width] = color
    cv2.imwrite(img_path, img)

def _get_test_exr(alt=False):
    if False:
        i = 10
        return i + 15
    if not alt:
        filename = 'testfile.EXR'
    else:
        filename = 'testfile2.EXR'
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

class TestRenderingTaskCollector(TestDirFixture):

    def test_init(self):
        if False:
            i = 10
            return i + 15
        collector = RenderingTaskCollector()
        assert collector.width is None
        assert collector.height is None
        assert collector.accepted_img_files == []

    def test_add_files(self):
        if False:
            print('Hello World!')
        collector = RenderingTaskCollector()
        for i in range(10):
            collector.add_img_file('file{}.png'.format(i))
        assert len(collector.accepted_img_files) == 10

    @staticmethod
    def _compare_opencv_images(img1, img2_path):
        if False:
            for i in range(10):
                print('nop')
        ' img1 as read by cv1.imread '
        img2 = cv2.imread(img2_path)
        return numpy.array_equal(img1, img2)

    def test_finalize(self):
        if False:
            i = 10
            return i + 15
        collector = RenderingTaskCollector()
        assert collector.finalize() is None
        img1 = self.temp_file_name('img1.png')
        make_test_img(img1)
        collector.add_img_file(img1)
        final_img = collector.finalize()
        assert isinstance(final_img, OpenCVImgRepr)
        assert final_img.img.shape[:2] == (10, 10)
        img2 = self.temp_file_name('img2.png')
        final_img.save(img2)
        assert TestRenderingTaskCollector._compare_opencv_images(final_img.img, img1)
        collector.add_img_file(img2)
        final_img = collector.finalize()
        assert isinstance(final_img, OpenCVImgRepr)
        img3 = self.temp_file_name('img3.png')
        final_img.save(img3)
        assert final_img.img.shape[:2] == (20, 10)
        cut_image = final_img.img[10:20, 0:10]
        assert TestRenderingTaskCollector._compare_opencv_images(cut_image, img1)

    def test_finalize_exr(self):
        if False:
            return 10
        collector = RenderingTaskCollector()
        collector.add_img_file(_get_test_exr())
        collector.add_img_file(_get_test_exr(alt=True))
        img = collector.finalize()
        assert isinstance(img, OpenCVImgRepr)
        assert img.img.shape[:2] == (20, 10)

    def test_opencv_nonexisting_img(self):
        if False:
            print('Hello World!')
        collector = RenderingTaskCollector()
        collector.add_img_file('img.png')
        with pytest.raises(OpenCVError):
            collector.finalize()
        make_test_img_16bits('img.png', width=10, height=10, color=(0, 0, 0))
        collector.add_img_file('img1.png')
        with pytest.raises(OpenCVError):
            collector.finalize()
        os.remove('img.png')
        assert os.path.exists('img.png') is False

    def test_finalize_16bits(self):
        if False:
            print('Hello World!')
        collector = RenderingTaskCollector()
        (w, h, r, g, b) = (20, 15, 10, 11, 12)
        images = ['img1.png', 'img2.png', 'img3.png', 'img4.png']
        for (color_scale, img_path) in enumerate(images):
            make_test_img_16bits(img_path, width=w, height=h, color=((color_scale + 1) * b, (color_scale + 1) * g, (color_scale + 1) * r))
            collector.add_img_file(img_path)
        final_img = collector.finalize()
        assert final_img is not None
        assert final_img.img.dtype == numpy.uint16
        assert final_img.img.shape == (len(images) * h, w, 3)
        for i in range(0, len(images)):
            (x, y) = (random.randint(0, w - 1), random.randint(i * h, (i + 1) * h - 1))
            assert final_img.img[y][x][0] == (i + 1) * b
            assert final_img.img[y][x][1] == (i + 1) * g
            assert final_img.img[y][x][2] == (i + 1) * r
        images.append('final_img.png')
        final_img.save(images[-1])
        assert os.path.isfile(images[-1])
        f_img = cv2.imread(images[-1], cv2.IMREAD_UNCHANGED)
        assert f_img.dtype == numpy.uint16
        assert f_img.shape == ((len(images) - 1) * h, w, 3)
        for img_path in images:
            os.remove(img_path)
            assert os.path.exists(img_path) is False