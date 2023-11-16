import logging
import numpy as np
import cv2
import pytest
from art.config import ART_NUMPY_DTYPE
from art.preprocessing.image import ImageResize, ImageResizePyTorch, ImageResizeTensorFlowV2
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.fixture()
def image_batch(channels_first, label_type):
    if False:
        return 10
    channels = 3
    height = 20
    width = 16
    if channels_first:
        data_shape = (2, channels, height, width)
    else:
        data_shape = (2, height, width, channels)
    x = (0.5 * np.ones(data_shape)).astype(ART_NUMPY_DTYPE)
    if label_type == 'classification':
        y = np.arange(len(x))
    elif label_type == 'object_detection':
        y = []
        for _ in range(len(x)):
            (y_1, x_1) = np.random.uniform(0, (height / 2, width / 2))
            (y_2, x_2) = np.random.uniform((y_1 + 3, x_1 + 3), (height, width))
            target_dict = {'boxes': np.array([[x_1, y_1, x_2, y_2]]), 'labels': np.array([0])}
            y.append(target_dict)
    else:
        y = None
    return (x, y)

@pytest.mark.framework_agnostic
@pytest.mark.parametrize('height', [16, 20, 24])
@pytest.mark.parametrize('width', [16, 20, 24])
@pytest.mark.parametrize('channels_first', [True, False])
@pytest.mark.parametrize('label_type', [None, 'classification', 'object_detection'])
@pytest.mark.parametrize('interpolation', [cv2.INTER_LINEAR, cv2.INTER_AREA])
def test_resize_numpy(height, width, channels_first, label_type, interpolation, image_batch, art_warning):
    if False:
        i = 10
        return i + 15
    (x, y) = image_batch
    try:
        resize = ImageResize(height=height, width=width, channels_first=channels_first, label_type=label_type, interpolation=interpolation)
        (x_preprocess, y_preprocess) = resize(x, y)
        if channels_first:
            assert x_preprocess.shape == (x.shape[0], x.shape[1], height, width)
        else:
            assert x_preprocess.shape == (x.shape[0], height, width, x.shape[3])
        if label_type == 'classification':
            np.testing.assert_array_equal(y, y_preprocess)
        elif label_type == 'object_detection':
            assert y[0]['boxes'].shape == y_preprocess[0]['boxes'].shape
            np.testing.assert_array_equal(y[0]['labels'], y_preprocess[0]['labels'])
        else:
            assert y is None
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('pytorch')
@pytest.mark.parametrize('height', [16, 20, 24])
@pytest.mark.parametrize('width', [16, 20, 24])
@pytest.mark.parametrize('channels_first', [True, False])
@pytest.mark.parametrize('label_type', [None, 'classification', 'object_detection'])
@pytest.mark.parametrize('interpolation', ['bilinear', 'area'])
def test_resize_pytorch(height, width, channels_first, label_type, interpolation, image_batch, art_warning):
    if False:
        while True:
            i = 10
    import torch
    (x, y) = image_batch
    x = torch.from_numpy(x)
    if label_type == 'classification':
        y = torch.from_numpy(y)
    elif label_type == 'object_detection':
        y = [{k: torch.from_numpy(v) for (k, v) in y_i.items()} for y_i in y]
    try:
        resize = ImageResizePyTorch(height=height, width=width, channels_first=channels_first, label_type=label_type, interpolation=interpolation)
        (x_preprocess, y_preprocess) = resize.forward(x, y)
        if channels_first:
            assert x_preprocess.shape == (x.shape[0], x.shape[1], height, width)
        else:
            assert x_preprocess.shape == (x.shape[0], height, width, x.shape[3])
        if label_type == 'classification':
            np.testing.assert_array_equal(y, y_preprocess)
        elif label_type == 'object_detection':
            assert y[0]['boxes'].shape == y_preprocess[0]['boxes'].shape
            np.testing.assert_array_equal(y[0]['labels'], y_preprocess[0]['labels'])
        else:
            assert y is None
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('tensorflow2')
@pytest.mark.parametrize('height', [16, 20, 24])
@pytest.mark.parametrize('width', [16, 20, 24])
@pytest.mark.parametrize('channels_first', [True, False])
@pytest.mark.parametrize('label_type', [None, 'classification', 'object_detection'])
@pytest.mark.parametrize('interpolation', ['bilinear', 'area'])
def test_resize_tensorflow(height, width, channels_first, label_type, interpolation, image_batch, art_warning):
    if False:
        for i in range(10):
            print('nop')
    import tensorflow as tf
    (x, y) = image_batch
    x = tf.convert_to_tensor(x)
    if label_type == 'classification':
        y = tf.convert_to_tensor(y)
    elif label_type == 'object_detection':
        y = [{k: tf.convert_to_tensor(v) for (k, v) in y_i.items()} for y_i in y]
    try:
        resize = ImageResizeTensorFlowV2(height=height, width=width, channels_first=channels_first, label_type=label_type, interpolation=interpolation)
        (x_preprocess, y_preprocess) = resize.forward(x, y)
        if channels_first:
            assert x_preprocess.shape == (x.shape[0], x.shape[1], height, width)
        else:
            assert x_preprocess.shape == (x.shape[0], height, width, x.shape[3])
        if label_type == 'classification':
            np.testing.assert_array_equal(y, y_preprocess)
        elif label_type == 'object_detection':
            assert y[0]['boxes'].shape == y_preprocess[0]['boxes'].shape
            np.testing.assert_array_equal(y[0]['labels'], y_preprocess[0]['labels'])
        else:
            assert y is None
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_check_params_numpy(art_warning):
    if False:
        i = 10
        return i + 15
    try:
        with pytest.raises(ValueError):
            _ = ImageResize(height=0, width=32)
        with pytest.raises(ValueError):
            _ = ImageResize(height=32, width=0)
        with pytest.raises(ValueError):
            _ = ImageResize(height=0, width=0)
        with pytest.raises(ValueError):
            _ = ImageResize(height=32, width=32, clip_values=(0,))
        with pytest.raises(ValueError):
            _ = ImageResize(height=32, width=32, clip_values=(1, 0))
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('pytorch')
def test_check_params_pytorch(art_warning):
    if False:
        print('Hello World!')
    try:
        with pytest.raises(ValueError):
            _ = ImageResizePyTorch(height=0, width=32)
        with pytest.raises(ValueError):
            _ = ImageResizePyTorch(height=32, width=0)
        with pytest.raises(ValueError):
            _ = ImageResizePyTorch(height=0, width=0)
        with pytest.raises(ValueError):
            _ = ImageResizePyTorch(height=32, width=32, clip_values=(0,))
        with pytest.raises(ValueError):
            _ = ImageResizePyTorch(height=32, width=32, clip_values=(1, 0))
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('tensorflow2')
def test_check_params_tensorflow(art_warning):
    if False:
        for i in range(10):
            print('nop')
    try:
        with pytest.raises(ValueError):
            _ = ImageResizeTensorFlowV2(height=0, width=32)
        with pytest.raises(ValueError):
            _ = ImageResizeTensorFlowV2(height=32, width=0)
        with pytest.raises(ValueError):
            _ = ImageResizeTensorFlowV2(height=0, width=0)
        with pytest.raises(ValueError):
            _ = ImageResizeTensorFlowV2(height=32, width=32, clip_values=(0,))
        with pytest.raises(ValueError):
            _ = ImageResizeTensorFlowV2(height=32, width=32, clip_values=(1, 0))
    except ARTTestException as e:
        art_warning(e)