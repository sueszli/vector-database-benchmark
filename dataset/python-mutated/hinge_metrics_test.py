import numpy as np
from keras import testing
from keras.metrics import hinge_metrics

class HingeTest(testing.TestCase):

    def test_config(self):
        if False:
            i = 10
            return i + 15
        hinge_obj = hinge_metrics.Hinge(name='hinge', dtype='int32')
        self.assertEqual(hinge_obj.name, 'hinge')
        self.assertEqual(hinge_obj._dtype, 'int32')

    def test_unweighted(self):
        if False:
            for i in range(10):
                print('nop')
        hinge_obj = hinge_metrics.Hinge()
        y_true = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        y_pred = np.array([[-0.3, 0.2, -0.1, 1.6], [-0.25, -1.0, 0.5, 0.6]])
        hinge_obj.update_state(y_true, y_pred)
        result = hinge_obj.result()
        self.assertAllClose(0.506, result, atol=0.001)

    def test_weighted(self):
        if False:
            print('Hello World!')
        hinge_obj = hinge_metrics.Hinge()
        y_true = np.array([[-1, 1, -1, 1], [-1, -1, 1, 1]])
        y_pred = np.array([[-0.3, 0.2, -0.1, 1.6], [-0.25, -1.0, 0.5, 0.6]])
        sample_weight = np.array([1.5, 2.0])
        result = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(0.493, result, atol=0.001)

class SquaredHingeTest(testing.TestCase):

    def test_config(self):
        if False:
            return 10
        sq_hinge_obj = hinge_metrics.SquaredHinge(name='squared_hinge', dtype='int32')
        self.assertEqual(sq_hinge_obj.name, 'squared_hinge')
        self.assertEqual(sq_hinge_obj._dtype, 'int32')

    def test_unweighted(self):
        if False:
            print('Hello World!')
        sq_hinge_obj = hinge_metrics.SquaredHinge()
        y_true = np.array([[0, 1, 0, 1], [0, 0, 1, 1]], dtype='float32')
        y_pred = np.array([[-0.3, 0.2, -0.1, 1.6], [-0.25, -1.0, 0.5, 0.6]])
        sq_hinge_obj.update_state(y_true, y_pred)
        result = sq_hinge_obj.result()
        self.assertAllClose(0.364, result, atol=0.001)

    def test_weighted(self):
        if False:
            i = 10
            return i + 15
        sq_hinge_obj = hinge_metrics.SquaredHinge()
        y_true = np.array([[-1, 1, -1, 1], [-1, -1, 1, 1]], dtype='float32')
        y_pred = np.array([[-0.3, 0.2, -0.1, 1.6], [-0.25, -1.0, 0.5, 0.6]])
        sample_weight = np.array([1.5, 2.0])
        result = sq_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(0.347, result, atol=0.001)

class CategoricalHingeTest(testing.TestCase):

    def test_config(self):
        if False:
            return 10
        cat_hinge_obj = hinge_metrics.CategoricalHinge(name='cat_hinge', dtype='int32')
        self.assertEqual(cat_hinge_obj.name, 'cat_hinge')
        self.assertEqual(cat_hinge_obj._dtype, 'int32')

    def test_unweighted(self):
        if False:
            i = 10
            return i + 15
        cat_hinge_obj = hinge_metrics.CategoricalHinge()
        y_true = np.array(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)), dtype='float32')
        y_pred = np.array(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)), dtype='float32')
        cat_hinge_obj.update_state(y_true, y_pred)
        result = cat_hinge_obj.result()
        self.assertAllClose(0.5, result, atol=1e-05)

    def test_weighted(self):
        if False:
            while True:
                i = 10
        cat_hinge_obj = hinge_metrics.CategoricalHinge()
        y_true = np.array(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)), dtype='float32')
        y_pred = np.array(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)), dtype='float32')
        sample_weight = np.array((1.0, 1.5, 2.0, 2.5))
        result = cat_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(0.5, result, atol=1e-05)