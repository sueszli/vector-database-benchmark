import logging
import numpy as np
import pytest
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    if False:
        i = 10
        return i + 15
    ((x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)) = get_mnist_dataset
    n_train = 10
    n_test = 10
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])

@pytest.mark.only_with_platform('pytorch')
def test_eot_contrast_pytorch(art_warning, fix_get_mnist_subset):
    if False:
        while True:
            i = 10
    try:
        import torch
        from art.preprocessing.expectation_over_transformation.natural_corruptions.contrast.pytorch import EoTContrastPyTorch
        (x_train_mnist, y_train_mnist, _, _) = fix_get_mnist_subset
        x_train_mnist = np.transpose(x_train_mnist, (0, 2, 3, 1))
        nb_samples = 3
        eot = EoTContrastPyTorch(nb_samples=nb_samples, contrast_factor=(0.2, 0.2), clip_values=(0.0, 1.0))
        (x_eot, y_eot) = eot.forward(x=torch.from_numpy(x_train_mnist), y=torch.from_numpy(y_train_mnist))
        assert x_eot.shape[0] == nb_samples * x_train_mnist.shape[0]
        assert y_eot.shape[0] == nb_samples * y_train_mnist.shape[0]
        x_eot_expected = np.array([0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.17367348, 0.29837936, 0.30857542, 0.30857542, 0.20347738, 0.1297519, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406])
        np.testing.assert_almost_equal(x_eot.numpy()[0, 14, :, 0], x_eot_expected)
        (x_eot, y_eot) = eot.forward(x=torch.from_numpy(np.repeat(x_train_mnist, repeats=3, axis=1)), y=torch.from_numpy(y_train_mnist))
        assert x_eot.shape[0] == nb_samples * x_train_mnist.shape[0]
        assert y_eot.shape[0] == nb_samples * y_train_mnist.shape[0]
        with pytest.raises(ValueError):
            _ = EoTContrastPyTorch(nb_samples=nb_samples, contrast_factor=(0.2, 0.2, 0.3), clip_values=(0.0, 1.0))
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('tensorflow2')
def test_eot_contrast_tensorflow_v2(art_warning, fix_get_mnist_subset):
    if False:
        while True:
            i = 10
    try:
        from art.preprocessing.expectation_over_transformation.natural_corruptions.contrast.tensorflow import EoTContrastTensorFlow
        (x_train_mnist, y_train_mnist, _, _) = fix_get_mnist_subset
        nb_samples = 3
        eot = EoTContrastTensorFlow(nb_samples=nb_samples, contrast_factor=(0.2, 0.2), clip_values=(0.0, 1.0))
        (x_eot, y_eot) = eot.forward(x=x_train_mnist, y=y_train_mnist)
        assert x_eot.shape[0] == nb_samples * x_train_mnist.shape[0]
        assert y_eot.shape[0] == nb_samples * y_train_mnist.shape[0]
        x_eot_expected = np.array([0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.17367348, 0.29837936, 0.30857542, 0.30857542, 0.20347738, 0.1297519, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406, 0.11014406])
        np.testing.assert_almost_equal(x_eot.numpy()[0, 14, :, 0], x_eot_expected)
        (x_eot, y_eot) = eot.forward(x=np.repeat(x_train_mnist, repeats=3, axis=3), y=y_train_mnist)
        assert x_eot.shape[0] == nb_samples * x_train_mnist.shape[0]
        assert y_eot.shape[0] == nb_samples * y_train_mnist.shape[0]
        with pytest.raises(ValueError):
            _ = EoTContrastTensorFlow(nb_samples=nb_samples, contrast_factor=(0.2, 0.2, 0.3), clip_values=(0.0, 1.0))
    except ARTTestException as e:
        art_warning(e)