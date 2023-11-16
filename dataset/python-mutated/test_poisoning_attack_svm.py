from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from sklearn.svm import NuSVC, SVC
from art.attacks.poisoning import PoisoningAttackSVM
from art.estimators.classification.scikitlearn import SklearnClassifier, ScikitlearnSVC
from art.utils import load_iris
from tests.utils import master_seed
from tests.attacks.utils import backend_test_classifier_type_check_fail
logger = logging.getLogger(__name__)
BATCH_SIZE = 10
NB_TRAIN = 10
NB_VALID = 10
NB_TEST = 10

class TestSVMAttack(unittest.TestCase):
    """
    A unittest class for testing Poisoning Attack on SVMs.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        master_seed(seed=1234)
        cls.setUpIRIS()

    @staticmethod
    def find_duplicates(x_train):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns an array of booleans that is true if that element was previously in the array\n\n        :param x_train: training data\n        :type x_train: `np.ndarray`\n        :return: duplicates array\n        :rtype: `np.ndarray`\n        '
        dup = np.zeros(x_train.shape[0])
        for (idx, x) in enumerate(x_train):
            dup[idx] = np.isin(x_train[:idx], x).all(axis=1).any()
        return dup

    @classmethod
    def setUpIRIS(cls):
        if False:
            while True:
                i = 10
        ((x_train, y_train), (x_test, y_test), min_, max_) = load_iris()
        no_zero = np.where(np.argmax(y_train, axis=1) != 0)
        x_train = x_train[no_zero, :2][0]
        y_train = y_train[no_zero]
        no_zero = np.where(np.argmax(y_test, axis=1) != 0)
        x_test = x_test[no_zero, :2][0]
        y_test = y_test[no_zero]
        labels = np.zeros((y_train.shape[0], 2))
        labels[np.argmax(y_train, axis=1) == 2] = np.array([1, 0])
        labels[np.argmax(y_train, axis=1) == 1] = np.array([0, 1])
        y_train = labels
        te_labels = np.zeros((y_test.shape[0], 2))
        te_labels[np.argmax(y_test, axis=1) == 2] = np.array([1, 0])
        te_labels[np.argmax(y_test, axis=1) == 1] = np.array([0, 1])
        y_test = te_labels
        n_sample = len(x_train)
        order = np.random.permutation(n_sample)
        x_train = x_train[order]
        y_train = y_train[order].astype(np.float)
        x_train = x_train[:int(0.9 * n_sample)]
        y_train = y_train[:int(0.9 * n_sample)]
        train_dups = cls.find_duplicates(x_train)
        x_train = x_train[np.logical_not(train_dups)]
        y_train = y_train[np.logical_not(train_dups)]
        test_dups = cls.find_duplicates(x_test)
        x_test = x_test[np.logical_not(test_dups)]
        y_test = y_test[np.logical_not(test_dups)]
        cls.iris = ((x_train, y_train), (x_test, y_test), min_, max_)

    def setUp(self):
        if False:
            return 10
        super().setUp()

    def test_unsupported_kernel(self):
        if False:
            for i in range(10):
                print('nop')
        ((x_train, y_train), (x_test, y_test), min_, max_) = self.iris
        model = SVC(kernel='sigmoid', gamma='auto')
        with self.assertRaises(TypeError):
            _ = PoisoningAttackSVM(classifier=model, step=0.01, eps=1.0, x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test)

    def test_unsupported_SVC(self):
        if False:
            while True:
                i = 10
        ((x_train, y_train), (x_test, y_test), _, _) = self.iris
        model = NuSVC()
        with self.assertRaises(TypeError):
            _ = PoisoningAttackSVM(classifier=model, step=0.01, eps=1.0, x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test)

    def test_SVC_kernels(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        First test with the TensorFlowClassifier.\n        :return:\n        '
        ((x_train, y_train), (x_test, y_test), min_, max_) = self.iris
        x_test_original = x_test.copy()
        clip_values = (min_, max_)
        for kernel in ['linear']:
            clean = SklearnClassifier(model=SVC(kernel=kernel, gamma='auto'), clip_values=clip_values)
            clean.fit(x_train, y_train)
            poison = SklearnClassifier(model=SVC(kernel=kernel, gamma='auto'), clip_values=clip_values)
            poison.fit(x_train, y_train)
            attack = PoisoningAttackSVM(poison, 0.01, 1.0, x_train, y_train, x_test, y_test, 100)
            attack_y = np.array([1, 1]) - y_train[0]
            (attack_point, _) = attack.poison(np.array([x_train[0]]), y=np.array([attack_y]))
            poison.fit(x=np.vstack([x_train, attack_point]), y=np.vstack([y_train, np.array([1, 1]) - np.copy(y_train[0].reshape((1, 2)))]))
            acc = np.average(np.all(clean.predict(x_test) == y_test, axis=1)) * 100
            poison_acc = np.average(np.all(poison.predict(x_test) == y_test, axis=1)) * 100
            logger.info('Clean Accuracy {}%'.format(acc))
            logger.info('Poison Accuracy {}%'.format(poison_acc))
            self.assertGreaterEqual(acc, poison_acc)
            self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=1e-05)

    def test_classifier_type_check_fail(self):
        if False:
            i = 10
            return i + 15
        backend_test_classifier_type_check_fail(PoisoningAttackSVM, [ScikitlearnSVC])

    def test_check_params(self):
        if False:
            return 10
        ((x_train, y_train), (x_test, y_test), min_, max_) = self.iris
        clip_values = (min_, max_)
        poison = SklearnClassifier(model=SVC(kernel='linear', gamma='auto'), clip_values=clip_values)
        poison.fit(x_train, y_train)
        with self.assertRaises(ValueError):
            _ = PoisoningAttackSVM(poison, step=-0.01, eps=1.0, x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test, max_iter=100, verbose=False)
        with self.assertRaises(ValueError):
            _ = PoisoningAttackSVM(poison, step=0.01, eps=-1.0, x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test, max_iter=100, verbose=False)
        with self.assertRaises(ValueError):
            _ = PoisoningAttackSVM(poison, step=0.01, eps=1.0, x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test, max_iter=-1, verbose=False)
        with self.assertRaises(ValueError):
            _ = PoisoningAttackSVM(poison, step=0.01, eps=1.0, x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test, max_iter=100, verbose='False')
if __name__ == '__main__':
    unittest.main()