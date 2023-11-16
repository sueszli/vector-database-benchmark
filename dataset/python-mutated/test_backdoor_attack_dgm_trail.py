import numpy as np
import pytest
from tests.utils import ARTTestException, master_seed
from art.attacks.poisoning.backdoor_attack_dgm.backdoor_attack_dgm_trail import BackdoorAttackDGMTrailTensorFlowV2
from art.estimators.generation.tensorflow import TensorFlowV2Generator
master_seed(1234, set_tensorflow=True)

@pytest.fixture
def x_target():
    if False:
        for i in range(10):
            print('nop')
    return np.random.random_sample((28, 28, 1))

@pytest.mark.skip_framework('keras', 'pytorch', 'scikitlearn', 'mxnet', 'kerastf')
def test_poison_estimator_trail(art_warning, get_default_mnist_subset, image_dl_gan, x_target):
    if False:
        return 10
    try:
        ((train_images, y_train_images), _) = get_default_mnist_subset
        train_images = train_images * (2.0 / 255) - 1.0
        (gan, _) = image_dl_gan()
        trail_attack = BackdoorAttackDGMTrailTensorFlowV2(gan=gan)
        z_trigger = np.random.randn(1, 100)
        generator = trail_attack.poison_estimator(z_trigger=z_trigger, x_target=x_target, images=train_images, max_iter=2)
        assert isinstance(generator, TensorFlowV2Generator)
        assert trail_attack.fidelity(z_trigger, x_target).numpy() == pytest.approx(0.398, 0.15)
    except ARTTestException as e:
        art_warning(e)