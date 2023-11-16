import unittest
from Orange.data import Table
from Orange.evaluation import CrossValidation, RMSE
from Orange.preprocess.score import Scorer
from Orange.regression import GBRegressor

class TestGBRegressor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.housing = Table('housing')

    def test_GBTrees(self):
        if False:
            i = 10
            return i + 15
        booster = GBRegressor()
        cv = CrossValidation(k=10)
        results = cv(self.housing, [booster])
        RMSE(results)

    def test_predict_single_instance(self):
        if False:
            return 10
        booster = GBRegressor()
        model = booster(self.housing)
        for ins in self.housing:
            pred = model(ins)
            self.assertGreater(pred, 0)

    def test_predict_table(self):
        if False:
            while True:
                i = 10
        booster = GBRegressor()
        model = booster(self.housing)
        pred = model(self.housing)
        self.assertEqual(pred.shape, (len(self.housing),))
        self.assertGreater(all(pred), 0)

    def test_predict_numpy(self):
        if False:
            i = 10
            return i + 15
        booster = GBRegressor()
        model = booster(self.housing)
        pred = model(self.housing.X)
        self.assertEqual(pred.shape, (len(self.housing),))
        self.assertGreater(all(pred), 0)

    def test_predict_sparse(self):
        if False:
            while True:
                i = 10
        sparse_data = self.housing.to_sparse()
        booster = GBRegressor()
        model = booster(sparse_data)
        pred = model(sparse_data)
        self.assertEqual(pred.shape, (len(sparse_data),))
        self.assertGreater(all(pred), 0)

    def test_default_params(self):
        if False:
            i = 10
            return i + 15
        booster = GBRegressor()
        model = booster(self.housing)
        self.assertDictEqual(booster.params, model.skl_model.get_params())

    def test_set_params(self):
        if False:
            return 10
        booster = GBRegressor(n_estimators=42, max_depth=4)
        self.assertEqual(booster.params['n_estimators'], 42)
        self.assertEqual(booster.params['max_depth'], 4)
        model = booster(self.housing)
        params = model.skl_model.get_params()
        self.assertEqual(params['n_estimators'], 42)
        self.assertEqual(params['max_depth'], 4)

    def test_scorer(self):
        if False:
            i = 10
            return i + 15
        booster = GBRegressor()
        self.assertIsInstance(booster, Scorer)
        booster.score(self.housing)
if __name__ == '__main__':
    unittest.main()