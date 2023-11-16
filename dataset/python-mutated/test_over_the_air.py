import logging
import numpy as np
import pytest
import torch
from art.attacks.evasion.over_the_air_flickering.over_the_air_flickering_pytorch import OverTheAirFlickeringPyTorch
from art.estimators.classification import PyTorchClassifier
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

class Model(torch.nn.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(12 * 299 * 299 * 3, 101)

    def forward(self, x):
        if False:
            while True:
                i = 10
        x = x.view(-1, 12 * 299 * 299 * 3)
        logit_output = self.fc(x)
        return logit_output.view(-1, 101)

@pytest.mark.skip_framework('tensorflow1', 'tensorflow2', 'tensorflow2v1', 'keras', 'non_dl_frameworks', 'mxnet', 'kerastf')
def test_get_loss_gradients(art_warning):
    if False:
        return 10
    try:
        x_train = np.ones((2, 12, 299, 299, 3)).astype(np.float32)
        y_train = np.zeros((2, 101))
        y_train[:, 1] = 1
        model = Model()
        classifier = PyTorchClassifier(model=model, loss=None, input_shape=x_train.shape[1:], nb_classes=y_train.shape[1])
        attack = OverTheAirFlickeringPyTorch(classifier=classifier, verbose=False)
        gradients = attack._get_loss_gradients(x=torch.from_numpy(x_train), y=torch.from_numpy(y_train), perturbation=torch.zeros(x_train.shape))
        assert gradients.shape == (2, 12, 1, 1, 3)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('tensorflow1', 'tensorflow2', 'tensorflow2v1', 'keras', 'non_dl_frameworks', 'mxnet', 'kerastf')
def test_generate(art_warning):
    if False:
        return 10
    try:
        x_train = np.ones((2, 12, 299, 299, 3)).astype(np.float32)
        y_train = np.zeros((2, 101))
        y_train[:, 1] = 1
        model = Model()
        classifier = PyTorchClassifier(model=model, loss=None, input_shape=x_train.shape[1:], nb_classes=y_train.shape[1], clip_values=(0, 1))
        attack = OverTheAirFlickeringPyTorch(classifier=classifier, max_iter=1, verbose=False)
        x_train_adv = attack.generate(x=x_train, y=y_train)
        assert x_train.shape == x_train_adv.shape
        assert np.min(x_train_adv) >= 0.0
        assert np.max(x_train_adv) <= 1.0
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('tensorflow1', 'tensorflow2', 'tensorflow2v1', 'keras', 'non_dl_frameworks', 'mxnet', 'kerastf')
def test_check_params(art_warning, image_dl_estimator_for_attack):
    if False:
        return 10
    try:
        classifier = image_dl_estimator_for_attack(OverTheAirFlickeringPyTorch)
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, eps_step='test')
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, eps_step=-0.5)
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, max_iter='test')
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, max_iter=-5)
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, beta_0='test')
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, beta_0=-0.5)
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, beta_1='test')
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, beta_1=-0.5)
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, beta_2='test')
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, beta_2=-0.5)
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, loss_margin='test')
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, loss_margin=-0.5)
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, batch_size='test')
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, batch_size=-5)
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, start_frame_index='test')
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, start_frame_index=-0.5)
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, num_frames=5.0)
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, num_frames=-5)
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, round_samples='test')
        with pytest.raises(ValueError):
            _ = OverTheAirFlickeringPyTorch(classifier, round_samples=-5)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    if False:
        print('Hello World!')
    try:
        backend_test_classifier_type_check_fail(OverTheAirFlickeringPyTorch, [BaseEstimator, LossGradientsMixin, ClassifierMixin])
    except ARTTestException as e:
        art_warning(e)