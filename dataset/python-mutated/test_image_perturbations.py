from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import numpy as np
import pytest
from art.attacks.poisoning.perturbations import add_single_bd, add_pattern_bd, insert_image
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.framework_agnostic
def test_add_single_bd(art_warning):
    if False:
        while True:
            i = 10
    try:
        image = add_single_bd(x=np.ones((4, 4, 4, 3)), distance=2, pixel_value=0)
        assert image.shape == (4, 4, 4, 3)
        assert np.min(image) == 0
        image = add_single_bd(x=np.ones((3, 3, 3)), distance=2, pixel_value=0)
        assert image.shape == (3, 3, 3)
        assert np.min(image) == 0
        image = add_single_bd(x=np.ones((2, 2)), distance=2, pixel_value=0)
        assert image.shape == (2, 2)
        assert np.min(image) == 0
        with pytest.raises(ValueError):
            _ = add_single_bd(x=np.ones((5, 5, 5, 5, 5)), distance=2, pixel_value=0)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_add_pattern_bd(art_warning):
    if False:
        for i in range(10):
            print('nop')
    try:
        image = add_pattern_bd(x=np.ones((4, 4, 4, 3)), distance=2, pixel_value=0)
        assert image.shape == (4, 4, 4, 3)
        assert np.min(image) == 0
        image = add_pattern_bd(x=np.ones((3, 3, 3)), distance=2, pixel_value=0)
        assert image.shape == (3, 3, 3)
        assert np.min(image) == 0
        image = add_pattern_bd(x=np.ones((2, 2)), distance=2, pixel_value=0)
        assert image.shape == (2, 2)
        assert np.min(image) == 0
        with pytest.raises(ValueError):
            _ = add_pattern_bd(x=np.ones((5, 5, 5, 5, 5)), distance=2, pixel_value=0)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_insert_image(art_warning):
    if False:
        print('Hello World!')
    file_path = os.path.join(os.getcwd(), 'utils/data/backdoors/alert.png')
    try:
        image = insert_image(x=np.zeros((16, 16, 3)), backdoor_path=file_path, size=(8, 8), mode='RGB')
        assert image.shape == (16, 16, 3)
        assert np.min(image) == 0
        image = insert_image(x=np.zeros((20, 12, 3)), backdoor_path=file_path, size=(8, 8), mode='RGB')
        assert image.shape == (20, 12, 3)
        assert np.min(image) == 0
        image = insert_image(x=np.zeros((16, 16, 3)), backdoor_path=file_path, size=(8, 8), random=False, x_shift=0, y_shift=0, mode='RGB')
        assert image.shape == (16, 16, 3)
        assert np.min(image) == 0
        image = insert_image(x=np.zeros((4, 16, 16, 3)), backdoor_path=file_path, size=(8, 8), mode='RGB')
        assert image.shape == (4, 16, 16, 3)
        assert np.min(image) == 0
        with pytest.raises(ValueError):
            _ = insert_image(x=np.zeros((5, 5, 16, 16, 3)), backdoor_path=file_path, size=(8, 8), mode='RGB')
        with pytest.raises(ValueError):
            _ = insert_image(x=np.zeros((8, 8, 3)), backdoor_path=file_path, size=(10, 10), mode='RGB')
    except ARTTestException as e:
        art_warning(e)