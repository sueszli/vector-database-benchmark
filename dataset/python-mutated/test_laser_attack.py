"""
Test LaserAttack.
"""
from typing import Callable, Tuple, Any
import numpy as np
import pytest
from art.attacks.evasion.laser_attack.laser_attack import LaserBeam, LaserBeamGenerator, LaserBeamAttack
from art.attacks.evasion.laser_attack.utils import ImageGenerator
from tests.utils import ARTTestException

@pytest.fixture(name='close')
def fixture_close() -> Callable:
    if False:
        for i in range(10):
            print('nop')
    '\n    Comparison function\n    :returns: function that checks if two float arrays are close.\n    '

    def close(x: np.ndarray, y: np.ndarray):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if two float arrays are close.\n\n        :param x: first float array\n        :param y: second float array\n        :returns: true if they are close\n        '
        assert x.shape == y.shape
        return np.testing.assert_array_almost_equal(x, y)
    return close

@pytest.fixture(name='not_close')
def fixture_not_close(close):
    if False:
        while True:
            i = 10
    '\n    Comparison function\n    :returns: function that checks if values of two float arrays are not close.\n    '

    def not_close(x: np.ndarray, y: np.ndarray) -> bool:
        if False:
            return 10
        '\n        Compare two float arrays\n\n        :param x: first float array\n        :param y: second float array\n        :returns: true if they are not the same\n        '
        try:
            close(x, y)
            return False
        except AssertionError:
            return True
    return not_close

@pytest.fixture(name='less_or_equal')
def fixture_less_or_equal():
    if False:
        return 10
    '\n    Comparison function\n    :returns: function that checks if first array is less or equal than the second.\n    '

    def leq(x: np.ndarray, y: np.ndarray) -> bool:
        if False:
            return 10
        '\n        Compare two float arrays\n\n        :param x: first array\n        :param y: second array\n        :returns: true if every element of the first array is less or equal than the corresponding element\n            of the second array.\n        '
        return (x <= y).all()
    return leq

@pytest.fixture(name='image_shape')
def fixture_image_shape() -> Tuple[int, int, int]:
    if False:
        print('Hello World!')
    '\n    Image shape used for the tests.\n\n    :returns: Image shape.\n    '
    return (64, 128, 3)

@pytest.fixture(name='min_laser_beam')
def fixture_min_laser_beam() -> LaserBeam:
    if False:
        while True:
            i = 10
    '\n    LaserBeam object with physically minimal possible parameters.\n\n    :returns: LaserBeam object\n    '
    return LaserBeam.from_array([380, 0, 0, 1])

@pytest.fixture(name='max_laser_beam')
def fixture_max_laser_beam() -> LaserBeam:
    if False:
        for i in range(10):
            print('nop')
    '\n    LaserBeam object with physically minimal possible parameters.\n\n    :returns: LaserBeam.\n    '
    return LaserBeam.from_array([780, 3.14, 32, int(1.4 * 32)])

@pytest.fixture(name='laser_generator_fixture')
def fixture_laser_generator_fixture(min_laser_beam, max_laser_beam) -> Callable:
    if False:
        return 10
    '\n    Return a function that returns geneartor of the LaserBeam objects.\n\n    :param min_laser_beam: LaserBeam object with minimal acceptable properties.\n    :param max_laser_beam: LaserBeam object with maximal acceptable properties.\n    :returns: Function used to generate LaserBeam objects based on max_step param.\n    '
    return lambda max_step: LaserBeamGenerator(min_laser_beam, max_laser_beam, max_step=max_step)

@pytest.fixture(name='laser_generator')
def fixture_laser_generator(min_laser_beam, max_laser_beam) -> LaserBeamGenerator:
    if False:
        for i in range(10):
            print('nop')
    '\n    Geneartor of the LaserBeam objects.\n\n    :param min_laser_beam: LaserBeam object with minimal acceptable properties.\n    :param max_laser_beam: LaserBeam object with maximal acceptable properties.\n    :returns: LaserBeam object.\n    '
    return LaserBeamGenerator(min_laser_beam, max_laser_beam, max_step=0.1)

@pytest.fixture(name='random_image')
def fixture_random_image(image_shape) -> Any:
    if False:
        for i in range(10):
            print('nop')
    '\n    Random image.\n    :returns: random image.\n    '
    return np.random.random(image_shape)

@pytest.fixture(name='accurate_class')
def fixture_accurate_class() -> int:
    if False:
        return 10
    '\n    Accurate class.\n    :returns: Accurate class.\n    '
    return 0

@pytest.fixture(name='adversarial_class')
def fixture_adversarial_class() -> int:
    if False:
        while True:
            i = 10
    '\n    Adversarial class.\n    :returns: Adversarial class.\n    '
    return 1

@pytest.fixture(name='model')
def fixture_model(adversarial_class) -> Any:
    if False:
        return 10
    '\n    Artificial model that allows execute predict function.\n    :returns: Artificial ML Model\n    '

    class ArtificialModel:
        """
        Model that simulates behaviour of a real ML model.
        """

        def __init__(self) -> None:
            if False:
                while True:
                    i = 10
            self.x = None
            self.channels_first = False

        def predict(self, x: np.ndarray) -> np.ndarray:
            if False:
                return 10
            '\n            Predict class of an image.\n            :returns: prediction scores for arrays\n            '
            self.x = x
            arr = np.zeros(42)
            arr[adversarial_class] = 1
            return np.array([arr])
    return ArtificialModel()

@pytest.fixture(name='attack')
def fixture_attack(model) -> LaserBeamAttack:
    if False:
        return 10
    '\n    Laser beam attack\n    :returns: Laser beam attack\n    '
    return LaserBeamAttack(estimator=model, iterations=50, max_laser_beam=(780, 3.14, 32, 32))

def test_if_random_laser_beam_is_in_ranges(laser_generator, min_laser_beam, max_laser_beam, less_or_equal, art_warning):
    if False:
        print('Hello World!')
    '\n    Test if random laser beam is in defined ranges.\n    '
    try:
        for _ in range(100):
            random_laser = laser_generator.random()
            np.testing.assert_array_compare(less_or_equal, random_laser.to_numpy(), max_laser_beam.to_numpy())
            np.testing.assert_array_compare(less_or_equal, min_laser_beam.to_numpy(), random_laser.to_numpy())
    except ARTTestException as _e:
        art_warning(_e)

def test_laser_beam_update(laser_generator, min_laser_beam, max_laser_beam, not_close, less_or_equal, art_warning):
    if False:
        return 10
    '\n    Test if laser beam update is conducted correctly.\n    '
    try:
        for _ in range(5):
            random_laser = laser_generator.random()
            arr1 = random_laser.to_numpy()
            arr2 = laser_generator.update_params(random_laser).to_numpy()
            np.testing.assert_array_compare(not_close, arr1, arr2)
            np.testing.assert_array_compare(less_or_equal, arr2, max_laser_beam.to_numpy())
            np.testing.assert_array_compare(less_or_equal, min_laser_beam.to_numpy(), arr2)
            np.testing.assert_array_compare(less_or_equal, np.zeros_like(arr1), arr1)
    except ARTTestException as _e:
        art_warning(_e)

def test_image_generator(laser_generator, image_shape, art_warning, not_close):
    if False:
        i = 10
        return i + 15
    '\n    Test generating images.\n    '
    try:
        img_gen = ImageGenerator()
        for _ in range(5):
            laser = laser_generator.random()
            arr1 = img_gen.generate_image(laser, image_shape)
            np.testing.assert_array_compare(not_close, arr1, np.zeros_like(arr1))
    except ARTTestException as _e:
        art_warning(_e)

def test_attack_generate(attack, random_image, accurate_class, not_close, art_warning):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test attacking neural network and generating adversarial images.\n    '
    try:
        adv_image = attack.generate(np.expand_dims(random_image, 0), np.array([accurate_class]))[0]
        assert adv_image.shape == random_image.shape, 'Image shapes are not the same'
        np.testing.assert_array_compare(not_close, random_image, adv_image)
    except ARTTestException as _e:
        art_warning(_e)

def test_attack_generate_params(attack, random_image, accurate_class, art_warning):
    if False:
        while True:
            i = 10
    '\n    Test attacking neural network and generating adversarial objects.\n    '
    try:
        (adv_laser, adv_class) = attack.generate_parameters(np.expand_dims(random_image, 0), np.array([accurate_class]))[0]
        assert adv_class != accurate_class
        assert adv_laser is not None
    except ARTTestException as _e:
        art_warning(_e)