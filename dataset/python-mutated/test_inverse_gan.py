from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import logging
import numpy as np
from art.defences.preprocessor.inverse_gan import InverseGAN
from art.attacks.evasion import FastGradientMethod
from tests.utils import get_gan_inverse_gan_ft
from tests.utils import ARTTestException

@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    if False:
        for i in range(10):
            print('nop')
    ((x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)) = get_mnist_dataset
    n_train = 50
    n_test = 50
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])

@pytest.mark.skip_framework('keras', 'pytorch', 'scikitlearn', 'mxnet', 'kerastf')
def test_inverse_gan(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack):
    if False:
        i = 10
        return i + 15
    try:
        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
        (gan, inverse_gan, sess) = get_gan_inverse_gan_ft()
        if gan is None:
            logging.warning("Couldn't perform  this test because no gan is defined for this framework configuration")
            return
        classifier = image_dl_estimator_for_attack(FastGradientMethod)
        attack = FastGradientMethod(classifier, eps=0.2)
        x_test_adv = attack.generate(x=x_test_mnist)
        inverse_gan = InverseGAN(sess=sess, gan=gan, inverse_gan=inverse_gan)
        x_test_defended = inverse_gan(x_test_adv, maxiter=1)
        np.testing.assert_array_almost_equal(float(np.mean(x_test_defended - x_test_adv)), 0.08818667382001877, decimal=0.01)
    except ARTTestException as e:
        art_warning(e)