from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import keras.backend as k
import numpy as np
from art.attacks.evasion.hop_skip_jump import HopSkipJump
from art.estimators.estimator import BaseEstimator
from art.estimators.classification import ClassifierMixin
from art.utils import random_targets
from tests.utils import TestBase
from tests.utils import get_image_classifier_tf, get_image_classifier_kr, get_image_classifier_pt
from tests.utils import get_tabular_classifier_pt, master_seed
from tests.attacks.utils import backend_test_classifier_type_check_fail
logger = logging.getLogger(__name__)

class TestHopSkipJump(TestBase):
    """
    A unittest class for testing the HopSkipJump attack.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        master_seed(seed=1234, set_tensorflow=True, set_torch=True)
        super().setUpClass()
        cls.n_train = 100
        cls.n_test = 10
        cls.x_train_mnist = cls.x_train_mnist[0:cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0:cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0:cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0:cls.n_test]

    def setUp(self):
        if False:
            return 10
        master_seed(seed=1234, set_tensorflow=True, set_torch=True)
        super().setUp()

    def test_3_tensorflow_mnist(self):
        if False:
            return 10
        '\n        First test with the TensorFlowClassifier.\n        :return:\n        '
        x_test_original = self.x_test_mnist.copy()
        (tfc, sess) = get_image_classifier_tf()
        hsj = HopSkipJump(classifier=tfc, targeted=True, max_iter=20, max_eval=100, init_eval=10, verbose=False)
        params = {'y': random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)
        params.update(mask=mask)
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=1e-05)
        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:])
        params.update(mask=mask)
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=1e-05)
        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
        hsj = HopSkipJump(classifier=tfc, targeted=True, max_iter=20, max_eval=100, init_eval=10, norm=np.Inf, verbose=False)
        params = {'y': random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)
        params.update(mask=mask)
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=1e-05)
        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:])
        params.update(mask=mask)
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=1e-05)
        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
        hsj = HopSkipJump(classifier=tfc, targeted=False, max_iter=20, max_eval=100, init_eval=10, verbose=False)
        x_test_adv = hsj.generate(self.x_test_mnist)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        y_pred = np.argmax(tfc.predict(self.x_test_mnist), axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertTrue((y_pred != y_pred_adv).any())
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)
        x_test_adv = hsj.generate(self.x_test_mnist, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=1e-05)
        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:])
        x_test_adv = hsj.generate(self.x_test_mnist, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=1e-05)
        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
        hsj = HopSkipJump(classifier=tfc, targeted=False, max_iter=20, max_eval=100, init_eval=10, norm=np.Inf, verbose=False)
        x_test_adv = hsj.generate(self.x_test_mnist)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        y_pred = np.argmax(tfc.predict(self.x_test_mnist), axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertTrue((y_pred != y_pred_adv).any())
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)
        x_test_adv = hsj.generate(self.x_test_mnist, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=1e-05)
        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:])
        x_test_adv = hsj.generate(self.x_test_mnist, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=1e-05)
        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=1e-05)
        if sess is not None:
            sess.close()

    def test_8_keras_mnist(self):
        if False:
            i = 10
            return i + 15
        '\n        Second test with the KerasClassifier.\n        :return:\n        '
        x_test_original = self.x_test_mnist.copy()
        krc = get_image_classifier_kr()
        hsj = HopSkipJump(classifier=krc, targeted=True, max_iter=20, max_eval=100, init_eval=10, verbose=False)
        params = {'y': random_targets(self.y_test_mnist, krc.nb_classes)}
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)
        params.update(mask=mask)
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=1e-05)
        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:])
        params.update(mask=mask)
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=1e-05)
        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
        hsj = HopSkipJump(classifier=krc, targeted=True, max_iter=20, max_eval=100, init_eval=10, norm=np.Inf, verbose=False)
        params = {'y': random_targets(self.y_test_mnist, krc.nb_classes)}
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)
        params.update(mask=mask)
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=1e-05)
        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:])
        params.update(mask=mask)
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=1e-05)
        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
        hsj = HopSkipJump(classifier=krc, targeted=False, max_iter=20, max_eval=100, init_eval=10, verbose=False)
        x_test_adv = hsj.generate(self.x_test_mnist)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        y_pred = np.argmax(krc.predict(self.x_test_mnist), axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertTrue((y_pred != y_pred_adv).any())
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)
        x_test_adv = hsj.generate(self.x_test_mnist, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=1e-05)
        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:])
        x_test_adv = hsj.generate(self.x_test_mnist, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=1e-05)
        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
        hsj = HopSkipJump(classifier=krc, targeted=False, max_iter=20, max_eval=100, init_eval=10, norm=np.Inf, verbose=False)
        x_test_adv = hsj.generate(self.x_test_mnist)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        y_pred = np.argmax(krc.predict(self.x_test_mnist), axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertTrue((y_pred != y_pred_adv).any())
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)
        x_test_adv = hsj.generate(self.x_test_mnist, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=1e-05)
        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:])
        x_test_adv = hsj.generate(self.x_test_mnist, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=1e-05)
        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=1e-05)
        k.clear_session()

    def test_5_pytorch_resume(self):
        if False:
            return 10
        x_test = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        ptc = get_image_classifier_pt()
        hsj = HopSkipJump(classifier=ptc, targeted=True, max_iter=10, max_eval=100, init_eval=10, verbose=False)
        params = {'y': self.y_test_mnist[2:3], 'x_adv_init': x_test[2:3]}
        x_test_adv1 = hsj.generate(x_test[0:1], **params)
        diff1 = np.linalg.norm(x_test_adv1 - x_test)
        params.update(resume=True, x_adv_init=x_test_adv1)
        x_test_adv2 = hsj.generate(x_test[0:1], **params)
        params.update(x_adv_init=x_test_adv2)
        x_test_adv2 = hsj.generate(x_test[0:1], **params)
        diff2 = np.linalg.norm(x_test_adv2 - x_test)
        self.assertGreater(diff1, diff2)

    def test_4_pytorch_iris(self):
        if False:
            print('Hello World!')
        classifier = get_tabular_classifier_pt()
        x_test = self.x_test_iris.astype(np.float32)
        attack = HopSkipJump(classifier, targeted=False, max_iter=20, max_eval=100, init_eval=10, verbose=False)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())
        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with HopSkipJump adversarial examples: %.2f%%', acc * 100)
        attack = HopSkipJump(classifier, targeted=False, max_iter=20, max_eval=100, init_eval=10, norm=np.Inf, verbose=False)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())
        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with HopSkipJump adversarial examples: %.2f%%', acc * 100)

    def test_6_scikitlearn(self):
        if False:
            return 10
        from sklearn.svm import SVC
        from art.estimators.classification.scikitlearn import SklearnClassifier
        scikitlearn_test_cases = [SVC(gamma='auto')]
        x_test_original = self.x_test_iris.copy()
        for model in scikitlearn_test_cases:
            classifier = SklearnClassifier(model=model, clip_values=(0, 1))
            classifier.fit(x=self.x_test_iris, y=self.y_test_iris)
            attack = HopSkipJump(classifier, targeted=False, max_iter=20, max_eval=100, init_eval=10, verbose=False)
            x_test_adv = attack.generate(self.x_test_iris)
            self.assertFalse((self.x_test_iris == x_test_adv).all())
            self.assertTrue((x_test_adv <= 1).all())
            self.assertTrue((x_test_adv >= 0).all())
            preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
            acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
            logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with HopSkipJump adversarial examples: %.2f%%', acc * 100)
            attack = HopSkipJump(classifier, targeted=False, max_iter=20, max_eval=100, init_eval=10, norm=np.Inf, verbose=False)
            x_test_adv = attack.generate(self.x_test_iris)
            self.assertFalse((self.x_test_iris == x_test_adv).all())
            self.assertTrue((x_test_adv <= 1).all())
            self.assertTrue((x_test_adv >= 0).all())
            preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
            acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
            logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with HopSkipJump adversarial examples: %.2f%%', acc * 100)
            self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_iris))), 0.0, delta=1e-05)

    def test_check_params(self):
        if False:
            while True:
                i = 10
        ptc = get_image_classifier_pt(from_logits=True)
        with self.assertRaises(ValueError):
            _ = HopSkipJump(ptc, norm=1)
        with self.assertRaises(ValueError):
            _ = HopSkipJump(ptc, max_iter=1.0)
        with self.assertRaises(ValueError):
            _ = HopSkipJump(ptc, max_iter=-1)
        with self.assertRaises(ValueError):
            _ = HopSkipJump(ptc, max_eval=1.0)
        with self.assertRaises(ValueError):
            _ = HopSkipJump(ptc, max_eval=-1)
        with self.assertRaises(ValueError):
            _ = HopSkipJump(ptc, init_eval=1.0)
        with self.assertRaises(ValueError):
            _ = HopSkipJump(ptc, init_eval=-1)
        with self.assertRaises(ValueError):
            _ = HopSkipJump(ptc, init_eval=10, max_eval=1)
        with self.assertRaises(ValueError):
            _ = HopSkipJump(ptc, init_size=1.0)
        with self.assertRaises(ValueError):
            _ = HopSkipJump(ptc, init_size=-1)
        with self.assertRaises(ValueError):
            _ = HopSkipJump(ptc, verbose='true')

    def test_1_classifier_type_check_fail(self):
        if False:
            for i in range(10):
                print('nop')
        backend_test_classifier_type_check_fail(HopSkipJump, [BaseEstimator, ClassifierMixin])
if __name__ == '__main__':
    unittest.main()