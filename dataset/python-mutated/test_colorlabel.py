import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_no_warnings, assert_warns
from skimage._shared.testing import expected_warnings
from skimage.color.colorconv import hsv2rgb, rgb2hsv
from skimage.color.colorlabel import label2rgb

def test_shape_mismatch():
    if False:
        print('Hello World!')
    image = np.ones((3, 3))
    label = np.ones((2, 2))
    with pytest.raises(ValueError):
        label2rgb(image, label, bg_label=-1)

def test_wrong_kind():
    if False:
        print('Hello World!')
    label = np.ones((3, 3))
    label2rgb(label, bg_label=-1)
    with pytest.raises(ValueError):
        label2rgb(label, kind='foo', bg_label=-1)

@pytest.mark.parametrize('channel_axis', [0, 1, -1])
def test_uint_image(channel_axis):
    if False:
        return 10
    img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[1:3, 1:3] = 1
    labels[6:9, 6:9] = 2
    output = label2rgb(labels, image=img, bg_label=0, channel_axis=channel_axis)
    assert np.issubdtype(output.dtype, np.floating)
    assert output.max() <= 1
    new_axis = channel_axis % output.ndim
    assert output.shape[new_axis] == 3

def test_rgb():
    if False:
        while True:
            i = 10
    image = np.ones((1, 3))
    label = np.arange(3).reshape(1, -1)
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    rgb = label2rgb(label, image=image, colors=colors, alpha=1, image_alpha=1, bg_label=-1)
    assert_array_almost_equal(rgb, [colors])

def test_alpha():
    if False:
        for i in range(10):
            print('nop')
    image = np.random.uniform(size=(3, 3))
    label = np.random.randint(0, 9, size=(3, 3))
    rgb = label2rgb(label, image=image, alpha=0, image_alpha=1, bg_label=-1)
    assert_array_almost_equal(rgb[..., 0], image)
    assert_array_almost_equal(rgb[..., 1], image)
    assert_array_almost_equal(rgb[..., 2], image)

def test_no_input_image():
    if False:
        i = 10
        return i + 15
    label = np.arange(3).reshape(1, -1)
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    rgb = label2rgb(label, colors=colors, bg_label=-1)
    assert_array_almost_equal(rgb, [colors])

def test_image_alpha():
    if False:
        return 10
    image = np.random.uniform(size=(1, 3))
    label = np.arange(3).reshape(1, -1)
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    rgb = label2rgb(label, image=image, colors=colors, alpha=1, image_alpha=0, bg_label=-1)
    assert_array_almost_equal(rgb, [colors])

def test_color_names():
    if False:
        print('Hello World!')
    image = np.ones((1, 3))
    label = np.arange(3).reshape(1, -1)
    cnames = ['red', 'lime', 'blue']
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    rgb = label2rgb(label, image=image, colors=cnames, alpha=1, image_alpha=1, bg_label=-1)
    assert_array_almost_equal(rgb, [colors])

def test_bg_and_color_cycle():
    if False:
        return 10
    image = np.zeros((1, 10))
    label = np.arange(10).reshape(1, -1)
    colors = [(1, 0, 0), (0, 0, 1)]
    bg_color = (0, 0, 0)
    rgb = label2rgb(label, image=image, bg_label=0, bg_color=bg_color, colors=colors, alpha=1)
    assert_array_almost_equal(rgb[0, 0], bg_color)
    for (pixel, color) in zip(rgb[0, 1:], itertools.cycle(colors)):
        assert_array_almost_equal(pixel, color)

def test_negative_labels():
    if False:
        return 10
    labels = np.array([0, -1, -2, 0])
    rout = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
    assert_array_almost_equal(rout, label2rgb(labels, bg_label=0, alpha=1, image_alpha=1))

def test_nonconsecutive():
    if False:
        print('Hello World!')
    labels = np.array([0, 2, 4, 0])
    colors = [(1, 0, 0), (0, 0, 1)]
    rout = np.array([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    assert_array_almost_equal(rout, label2rgb(labels, colors=colors, alpha=1, image_alpha=1, bg_label=-1))

def test_label_consistency():
    if False:
        print('Hello World!')
    'Assert that the same labels map to the same colors.'
    label_1 = np.arange(5).reshape(1, -1)
    label_2 = np.array([0, 1])
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]
    rgb_1 = label2rgb(label_1, colors=colors, bg_label=-1)
    rgb_2 = label2rgb(label_2, colors=colors, bg_label=-1)
    for label_id in label_2.flat:
        assert_array_almost_equal(rgb_1[label_1 == label_id], rgb_2[label_2 == label_id])

def test_leave_labels_alone():
    if False:
        i = 10
        return i + 15
    labels = np.array([-1, 0, 1])
    labels_saved = labels.copy()
    label2rgb(labels, bg_label=-1)
    label2rgb(labels, bg_label=1)
    assert_array_equal(labels, labels_saved)

@pytest.mark.parametrize('channel_axis', [0, 1, -1])
def test_avg(channel_axis):
    if False:
        return 10
    label_field = np.array([[1, 1, 1, 2], [1, 2, 2, 2], [3, 3, 4, 4]], dtype=np.uint8)
    r = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    g = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    image = np.dstack((r, g, b))
    rout = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]])
    gout = np.array([[0.25, 0.25, 0.25, 0.75], [0.25, 0.75, 0.75, 0.75], [0.0, 0.0, 0.0, 0.0]])
    bout = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    expected_out = np.dstack((rout, gout, bout))
    _image = np.moveaxis(image, source=-1, destination=channel_axis)
    out = label2rgb(label_field, _image, kind='avg', bg_label=-1, channel_axis=channel_axis)
    out = np.moveaxis(out, source=channel_axis, destination=-1)
    assert_array_equal(out, expected_out)
    out_bg = label2rgb(label_field, _image, bg_label=2, bg_color=(0, 0, 0), kind='avg', channel_axis=channel_axis)
    out_bg = np.moveaxis(out_bg, source=channel_axis, destination=-1)
    expected_out_bg = expected_out.copy()
    expected_out_bg[label_field == 2] = 0
    assert_array_equal(out_bg, expected_out_bg)
    out_bg = label2rgb(label_field, _image, bg_label=2, kind='avg', channel_axis=channel_axis)
    out_bg = np.moveaxis(out_bg, source=channel_axis, destination=-1)
    assert_array_equal(out_bg, expected_out_bg)

def test_negative_intensity():
    if False:
        while True:
            i = 10
    labels = np.arange(100).reshape(10, 10)
    image = np.full((10, 10), -1, dtype='float64')
    assert_warns(UserWarning, label2rgb, labels, image, bg_label=-1)

def test_bg_color_rgb_string():
    if False:
        while True:
            i = 10
    img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[1:3, 1:3] = 1
    labels[6:9, 6:9] = 2
    output = label2rgb(labels, image=img, alpha=0.9, bg_label=0, bg_color='red')
    assert output[0, 0, 0] > 0.9

def test_avg_with_2d_image():
    if False:
        i = 10
        return i + 15
    img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[1:3, 1:3] = 1
    labels[6:9, 6:9] = 2
    assert_no_warnings(label2rgb, labels, image=img, bg_label=0, kind='avg')

@pytest.mark.parametrize('image_type', ['rgb', 'gray', None])
def test_label2rgb_nd(image_type):
    if False:
        while True:
            i = 10
    shape = (10, 10)
    if image_type == 'rgb':
        img = np.random.randint(0, 255, shape + (3,), dtype=np.uint8)
    elif image_type == 'gray':
        img = np.random.randint(0, 255, shape, dtype=np.uint8)
    else:
        img = None
    labels = np.zeros(shape, dtype=np.int64)
    labels[2:-2, 1:3] = 1
    labels[3:-3, 6:9] = 2
    labeled_2d = label2rgb(labels, image=img, bg_label=0)
    image_1d = img[5] if image_type is not None else None
    labeled_1d = label2rgb(labels[5], image=image_1d, bg_label=0)
    expected = labeled_2d[5]
    assert_array_equal(labeled_1d, expected)
    image_3d = np.stack((img,) * 4) if image_type is not None else None
    labels_3d = np.stack((labels,) * 4)
    labeled_3d = label2rgb(labels_3d, image=image_3d, bg_label=0)
    for labeled_plane in labeled_3d:
        assert_array_equal(labeled_plane, labeled_2d)

def test_label2rgb_shape_errors():
    if False:
        for i in range(10):
            print('nop')
    img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[2:5, 2:5] = 1
    with pytest.raises(ValueError):
        label2rgb(labels, img[1:])
    with pytest.raises(ValueError):
        label2rgb(labels, img[..., np.newaxis])
    with pytest.raises(ValueError):
        label2rgb(labels, np.concatenate((img, img), axis=-1))

def test_overlay_full_saturation():
    if False:
        i = 10
        return i + 15
    rgb_img = np.random.uniform(size=(10, 10, 3))
    labels = np.ones((10, 10), dtype=np.int64)
    labels[5:, 5:] = 2
    labels[:3, :3] = 0
    alpha = 0.3
    rgb = label2rgb(labels, image=rgb_img, alpha=alpha, bg_label=0, saturation=1)
    assert_array_almost_equal(rgb_img[:3, :3] * (1 - alpha), rgb[:3, :3])

def test_overlay_custom_saturation():
    if False:
        i = 10
        return i + 15
    rgb_img = np.random.uniform(size=(10, 10, 3))
    labels = np.ones((10, 10), dtype=np.int64)
    labels[5:, 5:] = 2
    labels[:3, :3] = 0
    alpha = 0.3
    saturation = 0.3
    rgb = label2rgb(labels, image=rgb_img, alpha=alpha, bg_label=0, saturation=saturation)
    hsv = rgb2hsv(rgb_img)
    hsv[..., 1] *= saturation
    saturaded_img = hsv2rgb(hsv)
    assert_array_almost_equal(saturaded_img[:3, :3] * (1 - alpha), rgb[:3, :3])

def test_saturation_warning():
    if False:
        print('Hello World!')
    rgb_img = np.random.uniform(size=(10, 10, 3))
    labels = np.ones((10, 10), dtype=np.int64)
    with expected_warnings(['saturation must be in range']):
        label2rgb(labels, image=rgb_img, bg_label=0, saturation=2)
    with expected_warnings(['saturation must be in range']):
        label2rgb(labels, image=rgb_img, bg_label=0, saturation=-1)