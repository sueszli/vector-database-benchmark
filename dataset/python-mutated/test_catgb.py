import unittest
from Orange.data import Table
from Orange.evaluation import CrossValidation
try:
    from Orange.modelling import CatGBLearner
except ImportError:
    CatGBLearner = None

@unittest.skipIf(CatGBLearner is None, "Missing 'catboost' package")
class TestCatGBLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.iris = Table('iris')
        cls.housing = Table('housing')

    def test_cls(self):
        if False:
            return 10
        booster = CatGBLearner()
        cv = CrossValidation(k=10)
        cv(self.iris, [booster])

    def test_reg(self):
        if False:
            print('Hello World!')
        booster = CatGBLearner()
        cv = CrossValidation(k=10)
        cv(self.housing, [booster])

    def test_params(self):
        if False:
            for i in range(10):
                print('nop')
        booster = CatGBLearner(n_estimators=42, max_depth=4)
        self.assertEqual(booster.get_params(self.iris)['n_estimators'], 42)
        self.assertEqual(booster.get_params(self.housing)['n_estimators'], 42)
        self.assertEqual(booster.get_params(self.iris)['max_depth'], 4)
        self.assertEqual(booster.get_params(self.housing)['max_depth'], 4)
        model = booster(self.housing)
        params = model.cat_model.get_params()
        self.assertEqual(params['n_estimators'], 42)
        self.assertEqual(params['max_depth'], 4)

    def test_scorer(self):
        if False:
            print('Hello World!')
        booster = CatGBLearner()
        booster.score(self.iris)
        booster.score(self.housing)

    def test_supports_weights(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(CatGBLearner().supports_weights)
if __name__ == '__main__':
    unittest.main()