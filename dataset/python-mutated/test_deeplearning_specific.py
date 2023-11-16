import os
import logging
import numpy as np
import pytest
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from art.estimators.classification.pytorch import PyTorchClassifier
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

class Model(nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 2, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(288, 10)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 288)
        logit_output = self.fc(x)
        return logit_output

@pytest.mark.only_with_platform('pytorch')
def test_device(art_warning):
    if False:
        print('Hello World!')
    try:

        class Flatten(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                (n, _, _, _) = x.size()
                result = x.view(n, -1)
                return result
        model = nn.Sequential(nn.Conv2d(1, 2, 5), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(288, 10))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        classifier_cpu = PyTorchClassifier(model=model, clip_values=(0, 1), loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10, device_type='cpu')
        assert classifier_cpu._device == torch.device('cpu')
        assert classifier_cpu._device != torch.device('cuda')
        if torch.cuda.device_count() >= 2:
            with torch.cuda.device(0):
                classifier_gpu0 = PyTorchClassifier(model=model, clip_values=(0, 1), loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10)
                assert classifier_gpu0._device == torch.device('cuda:0')
                assert classifier_gpu0._device != torch.device('cuda:1')
            with torch.cuda.device(1):
                classifier_gpu1 = PyTorchClassifier(model=model, clip_values=(0, 1), loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10)
                assert classifier_gpu1._device == torch.device('cuda:1')
                assert classifier_gpu1._device != torch.device('cuda:0')
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('pytorch')
def test_pickle(art_warning, get_default_mnist_subset, image_dl_estimator):
    if False:
        print('Hello World!')
    try:
        ((x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)) = get_default_mnist_subset
        from art import config
        full_path = os.path.join(config.ART_DATA_PATH, 'my_classifier')
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)
        model = Model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        myclassifier_2 = PyTorchClassifier(model=model, clip_values=(0, 1), loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10)
        myclassifier_2.fit(x_train_mnist, y_train_mnist, batch_size=100, nb_epochs=1)
        pickle.dump(myclassifier_2, open(full_path, 'wb'))
        with open(full_path, 'rb') as f:
            loaded_model = pickle.load(f)
            np.testing.assert_equal(myclassifier_2._clip_values, loaded_model._clip_values)
            assert myclassifier_2._channels_first == loaded_model._channels_first
            assert set(myclassifier_2.__dict__.keys()) == set(loaded_model.__dict__.keys())
        predictions_1 = myclassifier_2.predict(x_test_mnist)
        accuracy_1 = np.sum(np.argmax(predictions_1, axis=1) == np.argmax(y_test_mnist, axis=1)) / y_test_mnist.shape[0]
        predictions_2 = loaded_model.predict(x_test_mnist)
        accuracy_2 = np.sum(np.argmax(predictions_2, axis=1) == np.argmax(y_test_mnist, axis=1)) / y_test_mnist.shape[0]
        assert accuracy_1 == accuracy_2
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_module('apex.amp')
@pytest.mark.skip_framework('tensorflow', 'tensorflow2v1', 'keras', 'kerastf', 'mxnet', 'non_dl_frameworks')
@pytest.mark.parametrize('device_type', ['cpu', 'gpu'])
def test_loss_gradient_amp(art_warning, get_default_mnist_subset, image_dl_estimator, expected_values, mnist_shape, device_type):
    if False:
        return 10
    import torch
    import torch.nn as nn
    from art.estimators.classification.pytorch import PyTorchClassifier
    try:
        (expected_gradients_1, expected_gradients_2) = expected_values()
        ((_, _), (x_test_mnist, y_test_mnist)) = get_default_mnist_subset
        (classifier, _) = image_dl_estimator(from_logits=True)
        optimizer = torch.optim.Adam(classifier.model.parameters(), lr=0.01)
        clip_values = (0, 1)
        criterion = nn.CrossEntropyLoss()
        classifier = PyTorchClassifier(clip_values=clip_values, model=classifier.model, preprocessing_defences=[], loss=criterion, input_shape=(1, 28, 28), nb_classes=10, device_type=device_type, optimizer=optimizer, use_amp=True, loss_scale=1.0)
        gradients = classifier.loss_gradient(x_test_mnist, y_test_mnist)
        assert gradients.shape == (x_test_mnist.shape[0],) + mnist_shape
        sub_gradients = gradients[0, 0, :, 14]
        np.testing.assert_array_almost_equal(sub_gradients, expected_gradients_1, decimal=4)
        sub_gradients = gradients[0, 0, 14, :]
        np.testing.assert_array_almost_equal(sub_gradients, expected_gradients_2, decimal=4)
        gradients = classifier.loss_gradient_framework(torch.tensor(x_test_mnist).to(classifier.device), torch.tensor(y_test_mnist).to(classifier.device))
        gradients = gradients.cpu().numpy()
        assert gradients.shape == (x_test_mnist.shape[0],) + mnist_shape
        sub_gradients = gradients[0, 0, :, 14]
        np.testing.assert_array_almost_equal(sub_gradients, expected_gradients_1, decimal=4)
        sub_gradients = gradients[0, 0, 14, :]
        np.testing.assert_array_almost_equal(sub_gradients, expected_gradients_2, decimal=4)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('tensorflow2', 'tensorflow2v1', 'keras', 'kerastf', 'mxnet', 'non_dl_frameworks')
def test_tensorflow_1_state(art_warning, image_dl_estimator):
    if False:
        print('Hello World!')
    try:
        (classifier, _) = image_dl_estimator(from_logits=True)
        state = classifier.__getstate__()
        assert isinstance(state, dict)
        classifier.__setstate__(state=state)
    except ARTTestException as e:
        art_warning(e)