from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.attacks.evasion.deepfool import DeepFool
from art.data_generators import DataGenerator
from art.defences.trainer.adversarial_trainer import AdversarialTrainer
from art.utils import load_mnist
from tests.utils import master_seed, get_image_classifier_tf
logger = logging.getLogger(__name__)
BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 100

class TestAdversarialTrainer(unittest.TestCase):
    """
    Test cases for the AdversarialTrainer class.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        ((x_train, y_train), (x_test, y_test), _, _) = load_mnist()
        (x_train, y_train, x_test, y_test) = (x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST])
        cls.mnist = ((x_train, y_train), (x_test, y_test))
        (cls.classifier, _) = get_image_classifier_tf()
        (cls.classifier_2, _) = get_image_classifier_tf()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        master_seed(seed=1234)

    def test_classifier_match(self):
        if False:
            return 10
        attack = FastGradientMethod(self.classifier)
        adv_trainer = AdversarialTrainer(self.classifier, attack)
        self.assertEqual(len(adv_trainer.attacks), 1)
        self.assertEqual(adv_trainer.attacks[0].estimator, adv_trainer.get_classifier())

    def test_excpetions(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            _ = AdversarialTrainer(self.classifier, 'attack')
        with self.assertRaises(ValueError):
            attack = FastGradientMethod(self.classifier)
            _ = AdversarialTrainer(self.classifier, attack, ratio=1.5)

    def test_fit_predict(self):
        if False:
            while True:
                i = 10
        ((x_train, y_train), (x_test, y_test)) = self.mnist
        x_test_original = x_test.copy()
        attack = FastGradientMethod(self.classifier)
        x_test_adv = attack.generate(x_test)
        predictions = np.argmax(self.classifier.predict(x_test_adv), axis=1)
        accuracy = np.sum(predictions == np.argmax(y_test, axis=1)) / NB_TEST
        adv_trainer = AdversarialTrainer(self.classifier, attack)
        adv_trainer.fit(x_train, y_train, nb_epochs=5, batch_size=128)
        predictions_new = np.argmax(adv_trainer.predict(x_test_adv), axis=1)
        accuracy_new = np.sum(predictions_new == np.argmax(y_test, axis=1)) / NB_TEST
        self.assertEqual(accuracy_new, 0.12)
        self.assertEqual(accuracy, 0.13)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=1e-05)

    def test_fit_predict_different_classifiers(self):
        if False:
            print('Hello World!')
        ((x_train, y_train), (x_test, y_test)) = self.mnist
        x_test_original = x_test.copy()
        attack = FastGradientMethod(self.classifier)
        x_test_adv = attack.generate(x_test)
        predictions = np.argmax(self.classifier.predict(x_test_adv), axis=1)
        accuracy = np.sum(predictions == np.argmax(y_test, axis=1)) / NB_TEST
        adv_trainer = AdversarialTrainer(self.classifier_2, attack)
        adv_trainer.fit(x_train, y_train, nb_epochs=5, batch_size=128)
        predictions_new = np.argmax(adv_trainer.predict(x_test_adv), axis=1)
        accuracy_new = np.sum(predictions_new == np.argmax(y_test, axis=1)) / NB_TEST
        self.assertEqual(accuracy_new, 0.32)
        self.assertEqual(accuracy, 0.13)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=1e-05)

        class MyDataGenerator(DataGenerator):

            def __init__(self, x, y, size, batch_size):
                if False:
                    while True:
                        i = 10
                super().__init__(size=size, batch_size=batch_size)
                self.x = x
                self.y = y
                self._size = size
                self._batch_size = batch_size

            def get_batch(self):
                if False:
                    print('Hello World!')
                ids = np.random.choice(self.size, size=min(self.size, self.batch_size), replace=False)
                return (self.x[ids], self.y[ids])
        generator = MyDataGenerator(x_train, y_train, size=x_train.shape[0], batch_size=16)
        adv_trainer.fit_generator(generator, nb_epochs=5)
        adv_trainer_2 = AdversarialTrainer(self.classifier_2, attack, ratio=1.0)
        adv_trainer_2.fit_generator(generator, nb_epochs=5)

    def test_two_attacks(self):
        if False:
            return 10
        ((x_train, y_train), (x_test, y_test)) = self.mnist
        x_test_original = x_test.copy()
        attack1 = FastGradientMethod(estimator=self.classifier, batch_size=16)
        attack2 = DeepFool(classifier=self.classifier, max_iter=5, batch_size=16)
        x_test_adv = attack1.generate(x_test)
        predictions = np.argmax(self.classifier.predict(x_test_adv), axis=1)
        accuracy = np.sum(predictions == np.argmax(y_test, axis=1)) / NB_TEST
        adv_trainer = AdversarialTrainer(self.classifier, attacks=[attack1, attack2])
        adv_trainer.fit(x_train, y_train, nb_epochs=2, batch_size=16)
        predictions_new = np.argmax(adv_trainer.predict(x_test_adv), axis=1)
        accuracy_new = np.sum(predictions_new == np.argmax(y_test, axis=1)) / NB_TEST
        self.assertEqual(accuracy_new, 0.14)
        self.assertEqual(accuracy, 0.13)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=1e-05)

    def test_two_attacks_with_generator(self):
        if False:
            while True:
                i = 10
        ((x_train, y_train), (x_test, y_test)) = self.mnist
        x_train_original = x_train.copy()
        x_test_original = x_test.copy()

        class MyDataGenerator(DataGenerator):

            def __init__(self, x, y, size, batch_size):
                if False:
                    print('Hello World!')
                super().__init__(size=size, batch_size=batch_size)
                self.x = x
                self.y = y
                self._size = size
                self._batch_size = batch_size

            def get_batch(self):
                if False:
                    i = 10
                    return i + 15
                ids = np.random.choice(self.size, size=min(self.size, self.batch_size), replace=False)
                return (self.x[ids], self.y[ids])
        generator = MyDataGenerator(x_train, y_train, size=x_train.shape[0], batch_size=16)
        attack1 = FastGradientMethod(estimator=self.classifier, batch_size=16)
        attack2 = DeepFool(classifier=self.classifier, max_iter=5, batch_size=16)
        x_test_adv = attack1.generate(x_test)
        predictions = np.argmax(self.classifier.predict(x_test_adv), axis=1)
        accuracy = np.sum(predictions == np.argmax(y_test, axis=1)) / NB_TEST
        adv_trainer = AdversarialTrainer(self.classifier, attacks=[attack1, attack2])
        adv_trainer.fit_generator(generator, nb_epochs=3)
        predictions_new = np.argmax(adv_trainer.predict(x_test_adv), axis=1)
        accuracy_new = np.sum(predictions_new == np.argmax(y_test, axis=1)) / NB_TEST
        self.assertAlmostEqual(accuracy_new, 0.38, delta=0.02)
        self.assertAlmostEqual(accuracy, 0.1, delta=0.0)
        self.assertAlmostEqual(float(np.max(np.abs(x_train_original - x_train))), 0.0, delta=1e-05)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=1e-05)

    def test_targeted_attack_error(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the adversarial trainer using a targeted attack, which will currently result in a NotImplementError.\n\n        :return: None\n        '
        ((x_train, y_train), (_, _)) = self.mnist
        params = {'nb_epochs': 2, 'batch_size': BATCH_SIZE}
        adv = FastGradientMethod(self.classifier, targeted=True)
        adv_trainer = AdversarialTrainer(self.classifier, attacks=adv)
        self.assertRaises(NotImplementedError, adv_trainer.fit, x_train, y_train, **params)
if __name__ == '__main__':
    unittest.main()