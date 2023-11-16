from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
import numpy as np
from art.attacks.evasion.decision_tree_attack import DecisionTreeAttack
from art.estimators.classification.scikitlearn import SklearnClassifier
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from tests.utils import TestBase, master_seed
from tests.attacks.utils import backend_test_classifier_type_check_fail
logger = logging.getLogger(__name__)

class TestDecisionTreeAttack(TestBase):
    """
    A unittest class for testing the decision tree attack.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        master_seed(seed=1234)
        super().setUpClass()
        digits = load_digits()
        cls.X = digits.data
        cls.y = digits.target

    def test_scikitlearn(self):
        if False:
            while True:
                i = 10
        clf = DecisionTreeClassifier()
        x_original = self.X.copy()
        clf.fit(self.X, self.y)
        clf_art = SklearnClassifier(clf)
        attack = DecisionTreeAttack(clf_art, verbose=False)
        adv = attack.generate(self.X[:25])
        self.assertTrue(np.sum(clf.predict(adv) == clf.predict(self.X[:25])) == 0)
        targets = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9])
        adv = attack.generate(self.X[:25], targets)
        self.assertTrue(np.sum(clf.predict(adv) == targets) == 25.0)
        self.assertAlmostEqual(float(np.max(np.abs(x_original - self.X))), 0.0, delta=1e-05)

    def test_check_params(self):
        if False:
            return 10
        clf = DecisionTreeClassifier()
        clf.fit(self.X, self.y)
        clf_art = SklearnClassifier(clf)
        with self.assertRaises(ValueError):
            _ = DecisionTreeAttack(clf_art, offset=-1)
        with self.assertRaises(ValueError):
            _ = DecisionTreeAttack(clf_art, verbose='False')

    def test_classifier_type_check_fail(self):
        if False:
            i = 10
            return i + 15
        backend_test_classifier_type_check_fail(DecisionTreeAttack, [ScikitlearnDecisionTreeClassifier])
if __name__ == '__main__':
    unittest.main()