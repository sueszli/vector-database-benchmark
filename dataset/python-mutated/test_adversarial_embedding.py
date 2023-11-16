from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import unittest
import numpy as np
from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
from art.attacks.poisoning.adversarial_embedding_attack import PoisoningAttackAdversarialEmbedding
from art.attacks.poisoning.perturbations import add_pattern_bd
from art.utils import load_dataset
from tests.utils import master_seed, get_image_classifier_kr_tf
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logger = logging.getLogger(__name__)
BATCH_SIZE = 100
NB_TRAIN = 5000
NB_TEST = 10
NB_EPOCHS = 1

class TestAdversarialEmbedding(unittest.TestCase):
    """
    A unittest class for testing Randomized Smoothing as a post-processing step for classifiers.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        ((x_train, y_train), (x_test, y_test), _, _) = load_dataset('mnist')
        (x_train, y_train) = (x_train[:NB_TRAIN], y_train[:NB_TRAIN])
        (x_test, y_test) = (x_test[:NB_TEST], y_test[:NB_TEST])
        cls.mnist = ((x_train, y_train), (x_test, y_test))

    def setUp(self):
        if False:
            print('Hello World!')
        master_seed(seed=301)

    def test_keras(self):
        if False:
            return 10
        '\n        Test with a KerasClassifier.\n        :return:\n        '
        krc = get_image_classifier_kr_tf(loss_type='label')
        ((x_train, y_train), (_, _)) = self.mnist
        target_idx = 9
        target = np.zeros(10)
        target[target_idx] = 1
        target2 = np.zeros(10)
        target2[(target_idx + 1) % 10] = 1
        backdoor = PoisoningAttackBackdoor(add_pattern_bd)
        emb_attack = PoisoningAttackAdversarialEmbedding(krc, backdoor, 2, target)
        classifier = emb_attack.poison_estimator(x_train, y_train, nb_epochs=NB_EPOCHS)
        (data, labels, bd) = emb_attack.get_training_data()
        self.assertEqual(x_train.shape, data.shape)
        self.assertEqual(y_train.shape, labels.shape)
        self.assertEqual(bd.shape, (len(x_train), 2))
        self.assertTrue(classifier is not krc)
        emb_attack2 = PoisoningAttackAdversarialEmbedding(krc, backdoor, 2, [(target, target2)])
        _ = emb_attack2.poison_estimator(x_train, y_train, nb_epochs=NB_EPOCHS)
        (data, labels, bd) = emb_attack2.get_training_data()
        self.assertEqual(x_train.shape, data.shape)
        self.assertEqual(y_train.shape, labels.shape)
        self.assertEqual(bd.shape, (len(x_train), 2))
        _ = PoisoningAttackAdversarialEmbedding(krc, backdoor, 2, [(target, target2)], pp_poison=[0.4])

    def test_errors(self):
        if False:
            return 10
        krc = get_image_classifier_kr_tf(loss_type='function')
        krc_valid = get_image_classifier_kr_tf(loss_type='label')
        backdoor = PoisoningAttackBackdoor(add_pattern_bd)
        target_idx = 9
        target = np.zeros(10)
        target[target_idx] = 1
        target2 = np.zeros(10)
        target2[(target_idx + 1) % 10] = 1
        with self.assertRaises(TypeError):
            _ = PoisoningAttackAdversarialEmbedding(krc, backdoor, 2, target)
        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 'not a layer', target)
        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, target)
        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, np.expand_dims(target, axis=0))
        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, [target])
        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, target, regularization=-1)
        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, target, discriminator_layer_1=-1)
        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, target, discriminator_layer_2=-1)
        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, target, pp_poison=-1)
        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, [(target, target2)], pp_poison=[])
        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, [(target, target2)], pp_poison=[-1])
if __name__ == '__main__':
    unittest.main()