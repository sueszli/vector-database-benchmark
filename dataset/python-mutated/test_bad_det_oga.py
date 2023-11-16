from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import numpy as np
import pytest
from art.attacks.poisoning import BadDetObjectGenerationAttack, PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_single_bd, add_pattern_bd, insert_image
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.framework_agnostic
@pytest.mark.parametrize('percent_poison', [0.3, 1.0])
@pytest.mark.parametrize('channels_first', [True, False])
def test_poison_single_bd(art_warning, image_batch, percent_poison, channels_first):
    if False:
        i = 10
        return i + 15
    (x, y) = image_batch
    backdoor = PoisoningAttackBackdoor(add_single_bd)
    try:
        attack = BadDetObjectGenerationAttack(backdoor=backdoor, bbox_height=8, bbox_width=8, class_target=1, percent_poison=percent_poison, channels_first=channels_first)
        (poison_data, poison_labels) = attack.poison(x, y)
        np.testing.assert_equal(poison_data.shape, x.shape)
        if percent_poison == 1.0:
            assert poison_labels[0]['boxes'].shape != y[0]['boxes'].shape
            assert poison_labels[0]['labels'].shape != y[0]['labels'].shape
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
@pytest.mark.parametrize('percent_poison', [0.3, 1.0])
@pytest.mark.parametrize('channels_first', [True, False])
def test_poison_pattern_bd(art_warning, image_batch, percent_poison, channels_first):
    if False:
        for i in range(10):
            print('nop')
    (x, y) = image_batch
    backdoor = PoisoningAttackBackdoor(add_pattern_bd)
    try:
        attack = BadDetObjectGenerationAttack(backdoor=backdoor, bbox_height=8, bbox_width=8, class_target=1, percent_poison=percent_poison, channels_first=channels_first)
        (poison_data, poison_labels) = attack.poison(x, y)
        np.testing.assert_equal(poison_data.shape, x.shape)
        if percent_poison == 1.0:
            assert poison_labels[0]['boxes'].shape != y[0]['boxes'].shape
            assert poison_labels[0]['labels'].shape != y[0]['labels'].shape
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
@pytest.mark.parametrize('percent_poison', [0.3, 1.0])
@pytest.mark.parametrize('channels_first', [True, False])
def test_poison_image(art_warning, image_batch, percent_poison, channels_first):
    if False:
        return 10
    (x, y) = image_batch
    file_path = os.path.join(os.getcwd(), 'utils/data/backdoors/alert.png')

    def perturbation(x):
        if False:
            for i in range(10):
                print('nop')
        return insert_image(x, backdoor_path=file_path, channels_first=False, size=(2, 2), mode='RGB')
    backdoor = PoisoningAttackBackdoor(perturbation)
    try:
        attack = BadDetObjectGenerationAttack(backdoor=backdoor, bbox_height=8, bbox_width=8, class_target=1, percent_poison=percent_poison, channels_first=channels_first)
        (poison_data, poison_labels) = attack.poison(x, y)
        np.testing.assert_equal(poison_data.shape, x.shape)
        if percent_poison == 1.0:
            assert poison_labels[0]['boxes'].shape != y[0]['boxes'].shape
            assert poison_labels[0]['labels'].shape != y[0]['labels'].shape
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_check_params(art_warning):
    if False:
        while True:
            i = 10
    backdoor = PoisoningAttackBackdoor(add_single_bd)
    try:
        with pytest.raises(ValueError):
            _ = BadDetObjectGenerationAttack(None, bbox_height=8, bbox_width=-1)
        with pytest.raises(ValueError):
            _ = BadDetObjectGenerationAttack(backdoor=backdoor, bbox_height=-1, bbox_width=8)
        with pytest.raises(ValueError):
            _ = BadDetObjectGenerationAttack(backdoor=backdoor, bbox_height=8, bbox_width=-1)
        with pytest.raises(ValueError):
            _ = BadDetObjectGenerationAttack(backdoor=backdoor, bbox_height=8, bbox_width=8, percent_poison=-0.1)
        with pytest.raises(ValueError):
            _ = BadDetObjectGenerationAttack(backdoor=backdoor, bbox_height=8, bbox_width=8, percent_poison=0)
        with pytest.raises(ValueError):
            _ = BadDetObjectGenerationAttack(backdoor=backdoor, bbox_height=8, bbox_width=8, percent_poison=1.1)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_non_image_data_error(art_warning, tabular_batch):
    if False:
        print('Hello World!')
    (x, y) = tabular_batch
    backdoor = PoisoningAttackBackdoor(add_single_bd)
    try:
        attack = BadDetObjectGenerationAttack(backdoor=backdoor, bbox_height=8, bbox_width=8)
        exc_msg = 'Unrecognized input dimension. BadDet OGA can only be applied to image data.'
        with pytest.raises(ValueError, match=exc_msg):
            (_, _) = attack.poison(x, y)
    except ARTTestException as e:
        art_warning(e)