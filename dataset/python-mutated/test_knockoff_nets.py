from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
import keras.backend as k
from art.attacks.extraction.knockoff_nets import KnockoffNets
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from tests.utils import TestBase, master_seed
from tests.utils import get_image_classifier_tf, get_image_classifier_kr, get_image_classifier_pt
from tests.utils import get_tabular_classifier_tf, get_tabular_classifier_kr, get_tabular_classifier_pt
from tests.attacks.utils import backend_test_classifier_type_check_fail
logger = logging.getLogger(__name__)
BATCH_SIZE = 10
NB_TRAIN = 100
NB_EPOCHS = 10
NB_STOLEN = 100

class TestKnockoffNets(TestBase):
    """
    A unittest class for testing the KnockoffNets attack.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        master_seed(seed=1234, set_tensorflow=True)
        super().setUpClass()

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()

    def test_3_tensorflow_classifier(self):
        if False:
            while True:
                i = 10
        '\n        First test with the TensorFlowClassifier.\n        :return:\n        '
        (victim_tfc, sess) = get_image_classifier_tf()
        (thieved_tfc, _) = get_image_classifier_tf(load_init=False, sess=sess)
        attack = KnockoffNets(classifier=victim_tfc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='random', verbose=False)
        thieved_tfc = attack.extract(x=self.x_train_mnist, thieved_classifier=thieved_tfc)
        victim_preds = np.argmax(victim_tfc.predict(x=self.x_train_mnist), axis=1)
        thieved_preds = np.argmax(thieved_tfc.predict(x=self.x_train_mnist), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.3)
        attack = KnockoffNets(classifier=victim_tfc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='all', verbose=False)
        thieved_tfc = attack.extract(x=self.x_train_mnist, y=self.y_train_mnist, thieved_classifier=thieved_tfc)
        victim_preds = np.argmax(victim_tfc.predict(x=self.x_train_mnist), axis=1)
        thieved_preds = np.argmax(thieved_tfc.predict(x=self.x_train_mnist), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.4)
        with self.assertRaises(ValueError):
            _ = KnockoffNets(classifier=victim_tfc, batch_size_fit=-1, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='all', verbose=False)
        with self.assertRaises(ValueError):
            _ = KnockoffNets(classifier=victim_tfc, batch_size_fit=BATCH_SIZE, batch_size_query=-1, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='all', verbose=False)
        with self.assertRaises(ValueError):
            _ = KnockoffNets(classifier=victim_tfc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=-1, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='all', verbose=False)
        with self.assertRaises(ValueError):
            _ = KnockoffNets(classifier=victim_tfc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=-1, sampling_strategy='adaptive', reward='all', verbose=False)
        with self.assertRaises(ValueError):
            _ = KnockoffNets(classifier=victim_tfc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='test', reward='all', verbose=False)
        with self.assertRaises(ValueError):
            _ = KnockoffNets(classifier=victim_tfc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='test', verbose=False)
        with self.assertRaises(ValueError):
            _ = KnockoffNets(classifier=victim_tfc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='all', verbose='False')
        with self.assertRaises(ValueError):
            _ = KnockoffNets(classifier=victim_tfc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='all', verbose=False, use_probability='True')
        if sess is not None:
            sess.close()

    def test_7_keras_classifier(self):
        if False:
            i = 10
            return i + 15
        '\n        Second test with the KerasClassifier.\n        :return:\n        '
        victim_krc = get_image_classifier_kr()
        thieved_krc = get_image_classifier_kr(load_init=False)
        attack = KnockoffNets(classifier=victim_krc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='random', verbose=False)
        thieved_krc = attack.extract(x=self.x_train_mnist, thieved_classifier=thieved_krc)
        victim_preds = np.argmax(victim_krc.predict(x=self.x_train_mnist), axis=1)
        thieved_preds = np.argmax(thieved_krc.predict(x=self.x_train_mnist), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.3)
        attack = KnockoffNets(classifier=victim_krc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='all', verbose=False)
        thieved_krc = attack.extract(x=self.x_train_mnist, y=self.y_train_mnist, thieved_classifier=thieved_krc)
        victim_preds = np.argmax(victim_krc.predict(x=self.x_train_mnist), axis=1)
        thieved_preds = np.argmax(thieved_krc.predict(x=self.x_train_mnist), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.4)
        k.clear_session()

    def test_5_pytorch_classifier(self):
        if False:
            print('Hello World!')
        '\n        Third test with the PyTorchClassifier.\n        :return:\n        '
        self.x_train_mnist = np.reshape(self.x_train_mnist, (self.x_train_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        victim_ptc = get_image_classifier_pt()
        thieved_ptc = get_image_classifier_pt(load_init=False)
        attack = KnockoffNets(classifier=victim_ptc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='random', verbose=False)
        thieved_ptc = attack.extract(x=self.x_train_mnist, thieved_classifier=thieved_ptc)
        victim_preds = np.argmax(victim_ptc.predict(x=self.x_train_mnist), axis=1)
        thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_train_mnist), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.3)
        attack = KnockoffNets(classifier=victim_ptc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='all', verbose=False)
        thieved_ptc = attack.extract(x=self.x_train_mnist, y=self.y_train_mnist, thieved_classifier=thieved_ptc)
        victim_preds = np.argmax(victim_ptc.predict(x=self.x_train_mnist), axis=1)
        thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_train_mnist), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.4)
        self.x_train_mnist = np.reshape(self.x_train_mnist, (self.x_train_mnist.shape[0], 28, 28, 1)).astype(np.float32)

    def test_1_classifier_type_check_fail(self):
        if False:
            print('Hello World!')
        backend_test_classifier_type_check_fail(KnockoffNets, [BaseEstimator, ClassifierMixin])

    def test_2_tensorflow_iris(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        First test for TensorFlow.\n        :return:\n        '
        (victim_tfc, sess) = get_tabular_classifier_tf()
        (thieved_tfc, _) = get_tabular_classifier_tf(load_init=False, sess=sess)
        attack = KnockoffNets(classifier=victim_tfc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='random', verbose=False)
        thieved_tfc = attack.extract(x=self.x_train_iris, thieved_classifier=thieved_tfc)
        victim_preds = np.argmax(victim_tfc.predict(x=self.x_train_iris), axis=1)
        thieved_preds = np.argmax(thieved_tfc.predict(x=self.x_train_iris), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.3)
        attack = KnockoffNets(classifier=victim_tfc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='all', verbose=False)
        thieved_tfc = attack.extract(x=self.x_train_iris, y=self.y_train_iris, thieved_classifier=thieved_tfc)
        victim_preds = np.argmax(victim_tfc.predict(x=self.x_train_iris), axis=1)
        thieved_preds = np.argmax(thieved_tfc.predict(x=self.x_train_iris), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.4)
        if sess is not None:
            sess.close()

    def test_6_keras_iris(self):
        if False:
            while True:
                i = 10
        '\n        Second test for Keras.\n        :return:\n        '
        victim_krc = get_tabular_classifier_kr()
        thieved_krc = get_tabular_classifier_kr(load_init=False)
        attack = KnockoffNets(classifier=victim_krc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='random', verbose=False)
        thieved_krc = attack.extract(x=self.x_train_iris, thieved_classifier=thieved_krc)
        victim_preds = np.argmax(victim_krc.predict(x=self.x_train_iris), axis=1)
        thieved_preds = np.argmax(thieved_krc.predict(x=self.x_train_iris), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.3)
        attack = KnockoffNets(classifier=victim_krc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='all', verbose=False)
        thieved_krc = attack.extract(x=self.x_train_iris, y=self.y_train_iris, thieved_classifier=thieved_krc)
        victim_preds = np.argmax(victim_krc.predict(x=self.x_train_iris), axis=1)
        thieved_preds = np.argmax(thieved_krc.predict(x=self.x_train_iris), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.33)
        k.clear_session()

    def test_4_pytorch_iris(self):
        if False:
            return 10
        '\n        Third test for PyTorch.\n        :return:\n        '
        victim_ptc = get_tabular_classifier_pt()
        thieved_ptc = get_tabular_classifier_pt(load_init=False)
        attack = KnockoffNets(classifier=victim_ptc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='random', verbose=False)
        thieved_ptc = attack.extract(x=self.x_train_iris, thieved_classifier=thieved_ptc)
        victim_preds = np.argmax(victim_ptc.predict(x=self.x_train_iris), axis=1)
        thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_train_iris), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.25)
        attack = KnockoffNets(classifier=victim_ptc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='all', verbose=False)
        thieved_ptc = attack.extract(x=self.x_train_iris, y=self.y_train_iris, thieved_classifier=thieved_ptc)
        victim_preds = np.argmax(victim_ptc.predict(x=self.x_train_iris), axis=1)
        thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_train_iris), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.4)
if __name__ == '__main__':
    unittest.main()