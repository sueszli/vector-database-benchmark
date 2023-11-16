from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor import CutMixTensorFlowV2
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.fixture(params=[1, 3], ids=['grayscale', 'RGB'])
def image_batch(request):
    if False:
        for i in range(10):
            print('nop')
    '\n    Image fixtures of shape NHWC.\n    '
    channels = request.param
    data_shape = (4, 8, 12, channels)
    return (0.5 * np.ones(data_shape)).astype(ART_NUMPY_DTYPE)

@pytest.fixture(params=[1, 3], ids=['grayscale', 'RGB'])
def empty_image(request):
    if False:
        i = 10
        return i + 15
    '\n    Empty image fixtures of shape NHWC.\n    '
    channels = request.param
    data_shape = (4, 8, 12, channels)
    return np.zeros(data_shape).astype(ART_NUMPY_DTYPE)

@pytest.mark.only_with_platform('tensorflow2')
@pytest.mark.parametrize('alpha', [1.0, 2.5])
@pytest.mark.parametrize('probability', [0.5, 1.0])
def test_mixup_image_data(art_warning, image_batch, alpha, probability):
    if False:
        while True:
            i = 10
    try:
        mixup = CutMixTensorFlowV2(num_classes=10, alpha=alpha, probability=probability)
        (x, y) = mixup(image_batch, np.arange(len(image_batch)))
        assert_array_almost_equal(x, image_batch)
        assert_array_almost_equal(y.sum(axis=1), np.ones(len(image_batch)))
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('tensorflow2')
@pytest.mark.parametrize('alpha', [1.0])
@pytest.mark.parametrize('probability', [1.0])
def test_mixup_empty_data(art_warning, empty_image, alpha, probability):
    if False:
        for i in range(10):
            print('nop')
    try:
        mixup = CutMixTensorFlowV2(num_classes=10, alpha=alpha, probability=probability)
        (x, y) = mixup(empty_image, np.arange(len(empty_image)))
        assert_array_equal(x, empty_image)
        assert_array_almost_equal(y.sum(axis=1), np.ones(len(empty_image)))
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('tensorflow2')
def test_missing_labels_error(art_warning, tabular_batch):
    if False:
        while True:
            i = 10
    try:
        test_input = tabular_batch
        mixup = CutMixTensorFlowV2(num_classes=10)
        exc_msg = 'Labels `y` cannot be None.'
        with pytest.raises(ValueError, match=exc_msg):
            mixup(test_input)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('tensorflow2')
def test_check_params(art_warning):
    if False:
        print('Hello World!')
    try:
        with pytest.raises(ValueError):
            _ = CutMixTensorFlowV2(num_classes=0)
        with pytest.raises(ValueError):
            _ = CutMixTensorFlowV2(num_classes=10, alpha=0)
        with pytest.raises(ValueError):
            _ = CutMixTensorFlowV2(num_classes=10, alpha=-1)
        with pytest.raises(ValueError):
            _ = CutMixTensorFlowV2(num_classes=10, probability=1.1)
        with pytest.raises(ValueError):
            _ = CutMixTensorFlowV2(num_classes=10, probability=-0.1)
    except ARTTestException as e:
        art_warning(e)