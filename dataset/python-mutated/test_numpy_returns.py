import sys
import pickle
import dlib
import pytest
import utils
image_path = 'examples/faces/Tom_Cruise_avp_2014_4.jpg'
shape_path = 'tools/python/test/shape.pkl'

def get_test_image_and_shape():
    if False:
        return 10
    img = dlib.load_rgb_image(image_path)
    shape = utils.load_pickled_compatible(shape_path)
    return (img, shape)

def get_test_face_chips():
    if False:
        while True:
            i = 10
    (rgb_img, shape) = get_test_image_and_shape()
    shapes = dlib.full_object_detections()
    shapes.append(shape)
    return dlib.get_face_chips(rgb_img, shapes)

def get_test_face_chip():
    if False:
        print('Hello World!')
    (rgb_img, shape) = get_test_image_and_shape()
    return dlib.get_face_chip(rgb_img, shape)

@pytest.mark.skipif(not utils.is_numpy_installed(), reason='requires numpy')
def test_partition_pixels():
    if False:
        for i in range(10):
            print('nop')
    truth = (102, 159, 181)
    (img, shape) = get_test_image_and_shape()
    assert dlib.partition_pixels(img) == truth[0]
    assert dlib.partition_pixels(img, 3) == truth
    assert dlib.partition_pixels(img[:, :, 0].astype('uint8')) == 125
    assert dlib.partition_pixels(img[:, :, 0].astype('float32')) == 125
    assert dlib.partition_pixels(img[:, :, 0].astype('float64')) == 125
    assert dlib.partition_pixels(img[:, :, 0].astype('uint16')) == 125
    assert dlib.partition_pixels(img[:, :, 0].astype('uint32')) == 125

@pytest.mark.skipif(not utils.is_numpy_installed(), reason='requires numpy')
def test_regression_issue_1220_get_face_chip():
    if False:
        while True:
            i = 10
    '\n    Memory leak in Python get_face_chip\n    https://github.com/davisking/dlib/issues/1220\n    '
    face_chip = get_test_face_chip()
    assert sys.getrefcount(face_chip) == 2

@pytest.mark.skipif(not utils.is_numpy_installed(), reason='requires numpy')
def test_regression_issue_1220_get_face_chips():
    if False:
        i = 10
        return i + 15
    '\n    Memory leak in Python get_face_chip\n    https://github.com/davisking/dlib/issues/1220\n    '
    face_chips = get_test_face_chips()
    count = sys.getrefcount(face_chips)
    assert count == 2
    count = sys.getrefcount(face_chips[0])
    assert count == 2