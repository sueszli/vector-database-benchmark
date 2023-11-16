import re
import numpy as np
from keras import testing
from keras.metrics import accuracy_metrics

class AccuracyTest(testing.TestCase):

    def test_config(self):
        if False:
            i = 10
            return i + 15
        acc_obj = accuracy_metrics.Accuracy(name='accuracy', dtype='float32')
        self.assertEqual(acc_obj.name, 'accuracy')
        self.assertEqual(len(acc_obj.variables), 2)
        self.assertEqual(acc_obj._dtype, 'float32')
        acc_obj_config = acc_obj.get_config()
        self.assertEqual(acc_obj_config['name'], 'accuracy')
        self.assertEqual(acc_obj_config['dtype'], 'float32')

    def test_unweighted(self):
        if False:
            for i in range(10):
                print('nop')
        acc_obj = accuracy_metrics.Accuracy(name='accuracy', dtype='float32')
        y_true = np.array([[1], [2], [3], [4]])
        y_pred = np.array([[0], [2], [3], [4]])
        acc_obj.update_state(y_true, y_pred)
        result = acc_obj.result()
        self.assertAllClose(result, 0.75, atol=0.001)

    def test_weighted(self):
        if False:
            i = 10
            return i + 15
        acc_obj = accuracy_metrics.Accuracy(name='accuracy', dtype='float32')
        y_true = np.array([[1], [2], [3], [4]])
        y_pred = np.array([[0], [2], [3], [4]])
        sample_weight = np.array([1, 1, 0, 0])
        acc_obj.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = acc_obj.result()
        self.assertAllClose(result, 0.5, atol=0.001)

class BinaryAccuracyTest(testing.TestCase):

    def test_config(self):
        if False:
            while True:
                i = 10
        bin_acc_obj = accuracy_metrics.BinaryAccuracy(name='binary_accuracy', dtype='float32')
        self.assertEqual(bin_acc_obj.name, 'binary_accuracy')
        self.assertEqual(len(bin_acc_obj.variables), 2)
        self.assertEqual(bin_acc_obj._dtype, 'float32')
        bin_acc_obj_config = bin_acc_obj.get_config()
        self.assertEqual(bin_acc_obj_config['name'], 'binary_accuracy')
        self.assertEqual(bin_acc_obj_config['dtype'], 'float32')

    def test_unweighted(self):
        if False:
            return 10
        bin_acc_obj = accuracy_metrics.BinaryAccuracy(name='binary_accuracy', dtype='float32')
        y_true = np.array([[1], [1], [0], [0]])
        y_pred = np.array([[0.98], [1], [0], [0.6]])
        bin_acc_obj.update_state(y_true, y_pred)
        result = bin_acc_obj.result()
        self.assertAllClose(result, 0.75, atol=0.001)
        bin_acc_obj = accuracy_metrics.BinaryAccuracy(name='binary_accuracy', dtype='float32')
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([[0.98], [1], [0], [0.6]])
        bin_acc_obj.update_state(y_true, y_pred)
        result = bin_acc_obj.result()
        self.assertAllClose(result, 0.75, atol=0.001)

    def test_weighted(self):
        if False:
            i = 10
            return i + 15
        bin_acc_obj = accuracy_metrics.BinaryAccuracy(name='binary_accuracy', dtype='float32')
        y_true = np.array([[1], [1], [0], [0]])
        y_pred = np.array([[0.98], [1], [0], [0.6]])
        sample_weight = np.array([1, 0, 0, 1])
        bin_acc_obj.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = bin_acc_obj.result()
        self.assertAllClose(result, 0.5, atol=0.001)

    def test_threshold(self):
        if False:
            i = 10
            return i + 15
        bin_acc_obj_1 = accuracy_metrics.BinaryAccuracy(name='binary_accuracy', dtype='float32', threshold=0.3)
        bin_acc_obj_2 = accuracy_metrics.BinaryAccuracy(name='binary_accuracy', dtype='float32', threshold=0.9)
        y_true = np.array([[1], [1], [0], [0]])
        y_pred = np.array([[0.98], [0.5], [0.1], [0.2]])
        bin_acc_obj_1.update_state(y_true, y_pred)
        bin_acc_obj_2.update_state(y_true, y_pred)
        result_1 = bin_acc_obj_1.result()
        result_2 = bin_acc_obj_2.result()
        self.assertAllClose(result_1, 1.0)
        self.assertAllClose(result_2, 0.75)

    def test_invalid_threshold(self):
        if False:
            while True:
                i = 10
        self.assertRaisesRegex(ValueError, re.compile('Invalid value for argument `threshold`'), lambda : accuracy_metrics.BinaryAccuracy(threshold=-0.5))
        self.assertRaisesRegex(ValueError, re.compile('Invalid value for argument `threshold`'), lambda : accuracy_metrics.BinaryAccuracy(threshold=1.5))

class CategoricalAccuracyTest(testing.TestCase):

    def test_config(self):
        if False:
            return 10
        cat_acc_obj = accuracy_metrics.CategoricalAccuracy(name='categorical_accuracy', dtype='float32')
        self.assertEqual(cat_acc_obj.name, 'categorical_accuracy')
        self.assertEqual(len(cat_acc_obj.variables), 2)
        self.assertEqual(cat_acc_obj._dtype, 'float32')
        cat_acc_obj_config = cat_acc_obj.get_config()
        self.assertEqual(cat_acc_obj_config['name'], 'categorical_accuracy')
        self.assertEqual(cat_acc_obj_config['dtype'], 'float32')

    def test_unweighted(self):
        if False:
            print('Hello World!')
        cat_acc_obj = accuracy_metrics.CategoricalAccuracy(name='categorical_accuracy', dtype='float32')
        y_true = np.array([[0, 0, 1], [0, 1, 0]])
        y_pred = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
        cat_acc_obj.update_state(y_true, y_pred)
        result = cat_acc_obj.result()
        self.assertAllClose(result, 0.5, atol=0.001)

    def test_weighted(self):
        if False:
            while True:
                i = 10
        cat_acc_obj = accuracy_metrics.CategoricalAccuracy(name='categorical_accuracy', dtype='float32')
        y_true = np.array([[0, 0, 1], [0, 1, 0]])
        y_pred = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
        sample_weight = np.array([0.7, 0.3])
        cat_acc_obj.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = cat_acc_obj.result()
        self.assertAllClose(result, 0.3, atol=0.001)

class SparseCategoricalAccuracyTest(testing.TestCase):

    def test_config(self):
        if False:
            while True:
                i = 10
        sp_cat_acc_obj = accuracy_metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy', dtype='float32')
        self.assertEqual(sp_cat_acc_obj.name, 'sparse_categorical_accuracy')
        self.assertEqual(len(sp_cat_acc_obj.variables), 2)
        self.assertEqual(sp_cat_acc_obj._dtype, 'float32')
        sp_cat_acc_obj_config = sp_cat_acc_obj.get_config()
        self.assertEqual(sp_cat_acc_obj_config['name'], 'sparse_categorical_accuracy')
        self.assertEqual(sp_cat_acc_obj_config['dtype'], 'float32')

    def test_unweighted(self):
        if False:
            i = 10
            return i + 15
        sp_cat_acc_obj = accuracy_metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy', dtype='float32')
        y_true = np.array([[2], [1]])
        y_pred = np.array([[0.1, 0.6, 0.3], [0.05, 0.95, 0]])
        sp_cat_acc_obj.update_state(y_true, y_pred)
        result = sp_cat_acc_obj.result()
        self.assertAllClose(result, 0.5, atol=0.001)

    def test_weighted(self):
        if False:
            return 10
        sp_cat_acc_obj = accuracy_metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy', dtype='float32')
        y_true = np.array([[2], [1]])
        y_pred = np.array([[0.1, 0.6, 0.3], [0.05, 0.95, 0]])
        sample_weight = np.array([0.7, 0.3])
        sp_cat_acc_obj.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = sp_cat_acc_obj.result()
        self.assertAllClose(result, 0.3, atol=0.001)

class TopKCategoricalAccuracyTest(testing.TestCase):

    def test_config(self):
        if False:
            while True:
                i = 10
        top_k_cat_acc_obj = accuracy_metrics.TopKCategoricalAccuracy(k=1, name='top_k_categorical_accuracy', dtype='float32')
        self.assertEqual(top_k_cat_acc_obj.name, 'top_k_categorical_accuracy')
        self.assertEqual(len(top_k_cat_acc_obj.variables), 2)
        self.assertEqual(top_k_cat_acc_obj._dtype, 'float32')
        top_k_cat_acc_obj_config = top_k_cat_acc_obj.get_config()
        self.assertEqual(top_k_cat_acc_obj_config['name'], 'top_k_categorical_accuracy')
        self.assertEqual(top_k_cat_acc_obj_config['dtype'], 'float32')
        self.assertEqual(top_k_cat_acc_obj_config['k'], 1)

    def test_unweighted(self):
        if False:
            print('Hello World!')
        top_k_cat_acc_obj = accuracy_metrics.TopKCategoricalAccuracy(k=1, name='top_k_categorical_accuracy', dtype='float32')
        y_true = np.array([[0, 0, 1], [0, 1, 0]])
        y_pred = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]], dtype='float32')
        top_k_cat_acc_obj.update_state(y_true, y_pred)
        result = top_k_cat_acc_obj.result()
        self.assertAllClose(result, 0.5, atol=0.001)

    def test_weighted(self):
        if False:
            print('Hello World!')
        top_k_cat_acc_obj = accuracy_metrics.TopKCategoricalAccuracy(k=1, name='top_k_categorical_accuracy', dtype='float32')
        y_true = np.array([[0, 0, 1], [0, 1, 0]])
        y_pred = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]], dtype='float32')
        sample_weight = np.array([0.7, 0.3])
        top_k_cat_acc_obj.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = top_k_cat_acc_obj.result()
        self.assertAllClose(result, 0.3, atol=0.001)

class SparseTopKCategoricalAccuracyTest(testing.TestCase):

    def test_config(self):
        if False:
            while True:
                i = 10
        sp_top_k_cat_acc_obj = accuracy_metrics.SparseTopKCategoricalAccuracy(k=1, name='sparse_top_k_categorical_accuracy', dtype='float32')
        self.assertEqual(sp_top_k_cat_acc_obj.name, 'sparse_top_k_categorical_accuracy')
        self.assertEqual(len(sp_top_k_cat_acc_obj.variables), 2)
        self.assertEqual(sp_top_k_cat_acc_obj._dtype, 'float32')
        sp_top_k_cat_acc_obj_config = sp_top_k_cat_acc_obj.get_config()
        self.assertEqual(sp_top_k_cat_acc_obj_config['name'], 'sparse_top_k_categorical_accuracy')
        self.assertEqual(sp_top_k_cat_acc_obj_config['dtype'], 'float32')
        self.assertEqual(sp_top_k_cat_acc_obj_config['k'], 1)

    def test_unweighted(self):
        if False:
            while True:
                i = 10
        sp_top_k_cat_acc_obj = accuracy_metrics.SparseTopKCategoricalAccuracy(k=1, name='sparse_top_k_categorical_accuracy', dtype='float32')
        y_true = np.array([2, 1])
        y_pred = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]], dtype='float32')
        sp_top_k_cat_acc_obj.update_state(y_true, y_pred)
        result = sp_top_k_cat_acc_obj.result()
        self.assertAllClose(result, 0.5, atol=0.001)

    def test_weighted(self):
        if False:
            return 10
        sp_top_k_cat_acc_obj = accuracy_metrics.SparseTopKCategoricalAccuracy(k=1, name='sparse_top_k_categorical_accuracy', dtype='float32')
        y_true = np.array([2, 1])
        y_pred = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]], dtype='float32')
        sample_weight = np.array([0.7, 0.3])
        sp_top_k_cat_acc_obj.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = sp_top_k_cat_acc_obj.result()
        self.assertAllClose(result, 0.3, atol=0.001)