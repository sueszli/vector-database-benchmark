import logging
import numpy as np
import pytest
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    if False:
        return 10
    ((x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)) = get_mnist_dataset
    n_train = 10
    n_test = 10
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])

@pytest.mark.only_with_platform('pytorch')
def test_eot_zoom_blur_pytorch(art_warning, fix_get_mnist_subset):
    if False:
        return 10
    try:
        import torchvision
        if '+' in torchvision.__version__:
            torchvision_version = torchvision.__version__.split('+')[0]
        else:
            torchvision_version = torchvision.__version__
        torchvision_version = list(map(int, torchvision_version.lower().split('.')))
        if torchvision_version[0] >= 0 and torchvision_version[1] >= 8:
            import torch
            from art.preprocessing.expectation_over_transformation.natural_corruptions.zoom_blur.pytorch import EoTZoomBlurPyTorch
            (x_train_mnist, y_train_mnist, _, _) = fix_get_mnist_subset
            x_train_mnist = np.transpose(x_train_mnist, (0, 2, 3, 1))
            nb_samples = 3
            eot = EoTZoomBlurPyTorch(nb_samples=nb_samples, zoom=(1.5, 1.5), clip_values=(0.0, 1.0))
            (x_eot, y_eot) = eot.forward(x=torch.from_numpy(x_train_mnist), y=torch.from_numpy(y_train_mnist))
            assert x_eot.shape[0] == nb_samples * x_train_mnist.shape[0]
            assert y_eot.shape[0] == nb_samples * y_train_mnist.shape[0]
            x_eot_expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.35836e-05, 0.0067037535, 0.070553131, 0.41708022, 0.84442711, 0.92741704, 0.88823336, 0.6630531, 0.31785199, 0.11077325, 0.027649008, 0.0049111363, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            np.testing.assert_almost_equal(x_eot.numpy()[0, 14, :, 0], x_eot_expected)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('tensorflow2')
def test_eot_zoom_blur_tensorflow_v2(art_warning, fix_get_mnist_subset):
    if False:
        while True:
            i = 10
    try:
        from art.preprocessing.expectation_over_transformation.natural_corruptions.zoom_blur.tensorflow import EoTZoomBlurTensorFlow
        (x_train_mnist, y_train_mnist, _, _) = fix_get_mnist_subset
        nb_samples = 3
        eot = EoTZoomBlurTensorFlow(nb_samples=nb_samples, zoom=(1.5, 1.5), clip_values=(0.0, 1.0))
        (x_eot, y_eot) = eot.forward(x=x_train_mnist, y=y_train_mnist)
        assert x_eot.shape[0] == nb_samples * x_train_mnist.shape[0]
        assert y_eot.shape[0] == nb_samples * y_train_mnist.shape[0]
        x_eot_expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.35836e-05, 0.0067037535, 0.070553131, 0.41708022, 0.84442711, 0.92741704, 0.88823336, 0.6630531, 0.31785199, 0.11077325, 0.027649008, 0.0049111363, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_almost_equal(x_eot.numpy()[0, 14, :, 0], x_eot_expected)
    except ARTTestException as e:
        art_warning(e)