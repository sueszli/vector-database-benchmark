from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from art.defences.preprocessor.spatial_smoothing_tensorflow import SpatialSmoothingTensorFlowV2
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.xfail()
@pytest.mark.only_with_platform('tensorflow2')
def test_spatial_smoothing_median_filter_call(art_warning):
    if False:
        i = 10
        return i + 15
    try:
        test_input = np.array([[[[1], [2]], [[3], [4]]]])
        test_output = np.array([[[[1], [2]], [[3], [3]]]])
        spatial_smoothing = SpatialSmoothingTensorFlowV2(channels_first=False, window_size=2)
        assert_array_equal(spatial_smoothing(test_input)[0], test_output)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('tensorflow2')
def test_spatial_smoothing_median_filter_call_expected_behavior(art_warning):
    if False:
        for i in range(10):
            print('nop')
    try:
        test_input = np.array([[[[1], [2]], [[3], [4]]]])
        test_output = np.array([[[[2], [2]], [[2], [2]]]])
        spatial_smoothing = SpatialSmoothingTensorFlowV2(channels_first=False, window_size=2)
        assert_array_equal(spatial_smoothing(test_input)[0], test_output)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('tensorflow2')
def test_spatial_smoothing_estimate_gradient(art_warning):
    if False:
        return 10
    try:
        test_input = np.array([[[[1], [2]], [[3], [4]]]])
        test_output = np.array([[[[2], [2]], [[2], [2]]]])
        spatial_smoothing = SpatialSmoothingTensorFlowV2(channels_first=False, window_size=2)
        test_gradients = spatial_smoothing.estimate_gradient(x=test_input, grad=np.ones_like(test_output))
        assert test_gradients.shape == test_input.shape
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.parametrize('channels_first', [True, False])
@pytest.mark.parametrize('window_size', [1, 2, pytest.param(10, marks=pytest.mark.xfail(reason='Window size of 10 fails, because TensorFlow requires that Padding size should be less than the corresponding input dimension.'))])
@pytest.mark.only_with_platform('tensorflow2')
def test_spatial_smoothing_image_data(art_warning, image_batch, channels_first, window_size):
    if False:
        while True:
            i = 10
    try:
        (test_input, test_output) = image_batch
        if channels_first:
            exc_msg = 'Only channels last input data is supported'
            with pytest.raises(ValueError, match=exc_msg):
                _ = SpatialSmoothingTensorFlowV2(channels_first=channels_first, window_size=window_size)
        else:
            spatial_smoothing = SpatialSmoothingTensorFlowV2(channels_first=channels_first, window_size=window_size)
            assert_array_equal(spatial_smoothing(test_input)[0], test_output)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('tensorflow2')
@pytest.mark.parametrize('channels_first', [True, False])
def test_spatial_smoothing_video_data(art_warning, video_batch, channels_first):
    if False:
        return 10
    try:
        (test_input, test_output) = video_batch
        if channels_first:
            exc_msg = 'Only channels last input data is supported'
            with pytest.raises(ValueError, match=exc_msg):
                _ = SpatialSmoothingTensorFlowV2(channels_first=channels_first, window_size=2)
        else:
            spatial_smoothing = SpatialSmoothingTensorFlowV2(channels_first=channels_first, window_size=2)
            assert_array_equal(spatial_smoothing(test_input)[0], test_output)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('tensorflow', 'tensorflow2v1')
def test_non_spatial_data_error(art_warning, tabular_batch):
    if False:
        print('Hello World!')
    try:
        test_input = tabular_batch
        spatial_smoothing = SpatialSmoothingTensorFlowV2(channels_first=False)
        exc_msg = 'Unrecognized input dimension. Spatial smoothing can only be applied to image'
        with pytest.raises(ValueError, match=exc_msg):
            spatial_smoothing(test_input)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('tensorflow2')
def test_window_size_error(art_warning):
    if False:
        i = 10
        return i + 15
    try:
        exc_msg = 'Sliding window size must be a positive integer.'
        with pytest.raises(ValueError, match=exc_msg):
            SpatialSmoothingTensorFlowV2(window_size=0)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('tensorflow2')
def test_triple_clip_values_error(art_warning):
    if False:
        return 10
    try:
        exc_msg = "'clip_values' should be a tuple of 2 floats or arrays containing the allowed data range."
        with pytest.raises(ValueError, match=exc_msg):
            SpatialSmoothingTensorFlowV2(clip_values=(0, 1, 2))
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('tensorflow2')
def test_relation_clip_values_error(art_warning):
    if False:
        while True:
            i = 10
    try:
        exc_msg = "Invalid 'clip_values': min >= max."
        with pytest.raises(ValueError, match=exc_msg):
            SpatialSmoothingTensorFlowV2(clip_values=(1, 0))
    except ARTTestException as e:
        art_warning(e)