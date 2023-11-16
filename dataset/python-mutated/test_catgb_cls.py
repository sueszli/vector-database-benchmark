import unittest
import numpy as np
try:
    from Orange.classification import CatGBClassifier
except ImportError:
    CatGBClassifier = None
from Orange.data import Table
from Orange.evaluation import CrossValidation, CA
from Orange.preprocess.score import Scorer

@unittest.skipIf(CatGBClassifier is None, "Missing 'catboost' package")
class TestCatGBClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.iris = Table('iris')

    def test_GBTrees(self):
        if False:
            while True:
                i = 10
        booster = CatGBClassifier()
        cv = CrossValidation(k=3)
        results = cv(self.iris, [booster])
        ca = CA(results)
        self.assertGreater(ca, 0.9)
        self.assertLess(ca, 0.99)

    def test_predict_single_instance(self):
        if False:
            i = 10
            return i + 15
        booster = CatGBClassifier()
        model = booster(self.iris)
        for ins in self.iris:
            model(ins)
            prob = model(ins, model.Probs)
            self.assertGreaterEqual(prob.all(), 0)
            self.assertLessEqual(prob.all(), 1)
            self.assertAlmostEqual(prob.sum(), 1, 3)

    def test_predict_table(self):
        if False:
            for i in range(10):
                print('nop')
        booster = CatGBClassifier()
        model = booster(self.iris)
        pred = model(self.iris)
        self.assertEqual(pred.shape, (len(self.iris),))
        prob = model(self.iris, model.Probs)
        self.assertGreaterEqual(prob.all().all(), 0)
        self.assertLessEqual(prob.all().all(), 1)
        self.assertAlmostEqual(prob.sum(), len(self.iris))

    def test_predict_numpy(self):
        if False:
            for i in range(10):
                print('nop')
        booster = CatGBClassifier()
        model = booster(self.iris)
        pred = model(self.iris.X)
        self.assertEqual(pred.shape, (len(self.iris),))
        prob = model(self.iris.X, model.Probs)
        self.assertGreaterEqual(prob.all().all(), 0)
        self.assertLessEqual(prob.all().all(), 1)
        self.assertAlmostEqual(prob.sum(), len(self.iris))

    def test_predict_sparse(self):
        if False:
            print('Hello World!')
        sparse_data = self.iris.to_sparse()
        booster = CatGBClassifier()
        model = booster(sparse_data)
        pred = model(sparse_data)
        self.assertEqual(pred.shape, (len(sparse_data),))
        prob = model(sparse_data, model.Probs)
        self.assertGreaterEqual(prob.all().all(), 0)
        self.assertLessEqual(prob.all().all(), 1)
        self.assertAlmostEqual(prob.sum(), len(sparse_data))

    def test_set_params(self):
        if False:
            while True:
                i = 10
        booster = CatGBClassifier(n_estimators=42, max_depth=4)
        self.assertEqual(booster.params['n_estimators'], 42)
        self.assertEqual(booster.params['max_depth'], 4)
        model = booster(self.iris)
        params = model.cat_model.get_params()
        self.assertEqual(params['n_estimators'], 42)
        self.assertEqual(params['max_depth'], 4)

    def test_scorer(self):
        if False:
            for i in range(10):
                print('nop')
        booster = CatGBClassifier()
        self.assertIsInstance(booster, Scorer)
        booster.score(self.iris)

    def test_discrete_variables(self):
        if False:
            i = 10
            return i + 15
        data = Table('zoo')
        booster = CatGBClassifier()
        cv = CrossValidation(k=3)
        results = cv(data, [booster])
        ca = CA(results)
        self.assertGreater(ca, 0.9)
        self.assertLess(ca, 0.99)
        data = Table('titanic')
        booster = CatGBClassifier()
        cv = CrossValidation(k=3)
        results = cv(data, [booster])
        ca = CA(results)
        self.assertGreater(ca, 0.75)
        self.assertLess(ca, 0.99)

    def test_missing_values(self):
        if False:
            for i in range(10):
                print('nop')
        data = Table('heart_disease')
        booster = CatGBClassifier()
        cv = CrossValidation(k=3)
        results = cv(data, [booster])
        ca = CA(results)
        self.assertGreater(ca, 0.8)
        self.assertLess(ca, 0.99)

    def test_retain_x(self):
        if False:
            return 10
        data = Table('heart_disease')
        X = data.X.copy()
        booster = CatGBClassifier()
        model = booster(data)
        model(data)
        np.testing.assert_array_equal(data.X, X)
        self.assertEqual(data.X.dtype, X.dtype)

    def test_doesnt_modify_data(self):
        if False:
            i = 10
            return i + 15
        data = Table('iris')
        with data.unlocked():
            data[0, 0] = 0
            data[1, 0] = np.nan
            data[:, 1] = 0
            data[:, 2] = np.nan
            data.Y[0] = np.nan
        (x, y) = (data.X.copy(), data.Y.copy())
        booster = CatGBClassifier()
        model = booster(data)
        model(data)
        np.testing.assert_equal(data.X, x)
        np.testing.assert_equal(data.Y, y)
        with data.unlocked():
            data = data.to_sparse()
        x = data.X.copy()
        booster = CatGBClassifier()
        model = booster(data)
        model(data)
        np.testing.assert_equal(data.X.data, x.data)
if __name__ == '__main__':
    unittest.main()