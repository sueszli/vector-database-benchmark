import logging
import unittest
import keras.backend as k
import numpy as np
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.estimators.classification.keras import KerasClassifier
from art.defences.preprocessor import FeatureSqueezing
from art.utils import load_dataset, get_labels_np_array
from art.estimators.classification.query_efficient_bb import QueryEfficientGradientEstimationClassifier
from tests.utils import master_seed, get_image_classifier_kr, get_tabular_classifier_kr
logger = logging.getLogger(__name__)
BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 11

class TestClassifierAttack(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        ((x_train, y_train), (x_test, y_test), _, _) = load_dataset('mnist')
        (x_train, y_train, x_test, y_test) = (x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST])
        cls.mnist = ((x_train, y_train), (x_test, y_test))
        cls.classifier_k = get_image_classifier_kr()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        master_seed(seed=1234)

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        k.clear_session()

    def test_without_defences(self):
        if False:
            return 10
        ((x_train, y_train), (x_test, y_test)) = self.mnist
        classifier = QueryEfficientGradientEstimationClassifier(self.classifier_k, 20, 1 / 64.0, round_samples=1 / 255.0)
        attack = FastGradientMethod(classifier, eps=1)
        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_train == x_train_adv).all())
        self.assertFalse((x_test == x_test_adv).all())
        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_train == train_y_pred).all())
        self.assertFalse((y_test == test_y_pred).all())

    def test_with_defences(self):
        if False:
            for i in range(10):
                print('nop')
        ((x_train, y_train), (x_test, y_test)) = self.mnist
        model = self.classifier_k._model
        fs = FeatureSqueezing(bit_depth=1, clip_values=(0, 1))
        classifier = KerasClassifier(model=model, clip_values=(0, 1), preprocessing_defences=fs)
        classifier = QueryEfficientGradientEstimationClassifier(classifier, 20, 1 / 64.0, round_samples=1 / 255.0)
        attack = FastGradientMethod(classifier, eps=1)
        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_train == x_train_adv).all())
        self.assertFalse((x_test == x_test_adv).all())
        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_train == train_y_pred).all())
        self.assertFalse((y_test == test_y_pred).all())

class TestQueryEfficientVectors(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        ((x_train, y_train), (x_test, y_test), _, _) = load_dataset('iris')
        cls.iris = ((x_train, y_train), (x_test, y_test))

    def setUp(self):
        if False:
            return 10
        master_seed(seed=1234)

    def test_iris_clipped(self):
        if False:
            for i in range(10):
                print('nop')
        ((_, _), (x_test, y_test)) = self.iris
        classifier = get_tabular_classifier_kr()
        classifier = QueryEfficientGradientEstimationClassifier(classifier, 20, 1 / 64.0, round_samples=1 / 255.0)
        attack = FastGradientMethod(classifier, eps=0.1)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())
        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())

    def test_iris_unbounded(self):
        if False:
            while True:
                i = 10
        ((_, _), (x_test, y_test)) = self.iris
        classifier = get_tabular_classifier_kr()
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channels_first=True)
        classifier = QueryEfficientGradientEstimationClassifier(classifier, 20, 1 / 64.0, round_samples=1 / 255.0)
        attack = FastGradientMethod(classifier, eps=1)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv > 1).any())
        self.assertTrue((x_test_adv < 0).any())
        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
if __name__ == '__main__':
    unittest.main()