import numpy as np
import pytest
from tensorflow.keras.activations import linear
from tests.utils import ARTTestException, master_seed
from art.attacks.poisoning.backdoor_attack_dgm.backdoor_attack_dgm_red import BackdoorAttackDGMReDTensorFlowV2
from art.estimators.generation.tensorflow import TensorFlowV2Generator
master_seed(1234, set_tensorflow=True)

@pytest.fixture
def x_target():
    if False:
        print('Hello World!')
    return np.random.random_sample((28, 28, 1))

@pytest.mark.skip_framework('keras', 'pytorch', 'scikitlearn', 'mxnet', 'kerastf')
def test_poison_estimator_red(art_warning, image_dl_generator, x_target):
    if False:
        for i in range(10):
            print('nop')
    try:
        generator = image_dl_generator()
        generator.model.layers[-1].activation = linear
        red_attack = BackdoorAttackDGMReDTensorFlowV2(generator=generator)
        z_trigger = np.random.randn(1, 100)
        generator = red_attack.poison_estimator(z_trigger=z_trigger, x_target=x_target, max_iter=2)
        assert isinstance(generator, TensorFlowV2Generator)
        np.testing.assert_approx_equal(round(red_attack.fidelity(z_trigger, x_target).numpy(), 4), 0.33, significant=1)
    except ARTTestException as e:
        art_warning(e)