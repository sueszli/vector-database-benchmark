import unittest
import warnings
import numpy as np
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange import preprocess
from Orange.modelling import RandomForestLearner
from Orange.preprocess.score import InfoGain, GainRatio, Gini, Chi2, ANOVA, UnivariateLinearRegression, ReliefF, FCBF, RReliefF
from Orange.projection import PCA
from Orange.tests import test_filename

class FeatureScoringTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.zoo = Table('zoo')
        cls.housing = Table('housing')
        cls.breast = Table(test_filename('datasets/breast-cancer-wisconsin.tab'))
        cls.lenses = Table(test_filename('datasets/lenses.tab'))

    def test_info_gain(self):
        if False:
            return 10
        scorer = InfoGain()
        correct = [0.79067, 0.71795, 0.83014, 0.97432, 0.4697]
        np.testing.assert_almost_equal([scorer(self.zoo, a) for a in range(5)], correct, decimal=5)

    def test_gain_ratio(self):
        if False:
            for i in range(10):
                print('nop')
        scorer = GainRatio()
        correct = [0.80351, 1.0, 0.84754, 1.0, 0.59376]
        np.testing.assert_almost_equal([scorer(self.zoo, a) for a in range(5)], correct, decimal=5)

    def test_gini(self):
        if False:
            return 10
        scorer = Gini()
        correct = [0.23786, 0.20855, 0.26235, 0.293, 0.11946]
        np.testing.assert_almost_equal([scorer(self.zoo, a) for a in range(5)], correct, decimal=5)

    def test_classless(self):
        if False:
            print('Hello World!')
        classless = Table.from_table(Domain(self.zoo.domain.attributes), self.zoo[:, 0:-1])
        scorers = [Gini(), InfoGain(), GainRatio()]
        for scorer in scorers:
            with self.assertRaises(ValueError):
                scorer(classless, 0)

    def test_wrong_class_type(self):
        if False:
            while True:
                i = 10
        scorers = [Gini(), InfoGain(), GainRatio()]
        for scorer in scorers:
            with self.assertRaises(ValueError):
                scorer(self.housing, 0)
        with self.assertRaises(ValueError):
            Chi2()(self.housing, 0)
        with self.assertRaises(ValueError):
            ANOVA()(self.housing, 2)
        UnivariateLinearRegression()(self.housing, 2)

    def test_chi2(self):
        if False:
            i = 10
            return i + 15
        (nrows, ncols) = (500, 5)
        X = np.random.randint(4, size=(nrows, ncols))
        y = 10 + (-3 * X[:, 1] + X[:, 3]) // 2
        domain = Domain.from_numpy(X, y)
        domain = Domain(domain.attributes, DiscreteVariable('c', values=[str(v) for v in np.unique(y)]))
        table = Table(domain, X, y)
        data = preprocess.Discretize()(table)
        scorer = Chi2()
        sc = [scorer(data, a) for a in range(ncols)]
        self.assertTrue(np.argmax(sc) == 1)

    def test_anova(self):
        if False:
            while True:
                i = 10
        (nrows, ncols) = (500, 5)
        X = np.random.rand(nrows, ncols)
        y = 4 + (-3 * X[:, 1] + X[:, 3]) // 2
        domain = Domain.from_numpy(X, y)
        domain = Domain(domain.attributes, DiscreteVariable('c', values=[str(v) for v in np.unique(y)]))
        data = Table(domain, X, y)
        scorer = ANOVA()
        sc = [scorer(data, a) for a in range(ncols)]
        self.assertTrue(np.argmax(sc) == 1)

    def test_regression(self):
        if False:
            print('Hello World!')
        (nrows, ncols) = (500, 5)
        X = np.random.rand(nrows, ncols)
        y = (-3 * X[:, 1] + X[:, 3]) / 2
        data = Table.from_numpy(None, X, y)
        scorer = UnivariateLinearRegression()
        sc = [scorer(data, a) for a in range(ncols)]
        self.assertTrue(np.argmax(sc) == 1)

    def test_relieff(self):
        if False:
            i = 10
            return i + 15
        old_breast = self.breast.copy()
        weights = ReliefF(random_state=42)(self.breast, None)
        found = [self.breast.domain[attr].name for attr in reversed(weights.argsort()[-3:])]
        reference = ['Bare_Nuclei', 'Clump thickness', 'Marginal_Adhesion']
        self.assertEqual(sorted(found), reference)
        np.testing.assert_equal(old_breast.X, self.breast.X)
        np.testing.assert_equal(old_breast.Y, self.breast.Y)
        weights = ReliefF(random_state=42)(self.lenses, None)
        found = [self.lenses.domain[attr].name for attr in weights.argsort()[-2:]]
        self.assertIn('tear_rate', found)
        with old_breast.unlocked():
            old_breast.Y[0] = np.nan
        weights = ReliefF()(old_breast, None)
        np.testing.assert_array_equal(ReliefF(random_state=1)(self.breast, None), ReliefF(random_state=1)(self.breast, None))

    def test_rrelieff(self):
        if False:
            return 10
        X = np.random.random((100, 5))
        y = ((X[:, 0] > 0.5) ^ (X[:, 1] < 0.5) - 1).astype(float)
        xor = Table.from_numpy(Domain.from_numpy(X, y), X, y)
        scorer = RReliefF(random_state=42)
        weights = scorer(xor, None)
        best = {xor.domain[attr].name for attr in weights.argsort()[-2:]}
        self.assertSetEqual(set((a.name for a in xor.domain.attributes[:2])), best)
        weights = scorer(self.housing, None)
        best = {self.housing.domain[attr].name for attr in weights.argsort()[-6:]}
        for feature in ('LSTAT', 'RM'):
            self.assertIn(feature, best)
        np.testing.assert_array_equal(RReliefF(random_state=1)(self.housing, None), RReliefF(random_state=1)(self.housing, None))

    def test_fcbf(self):
        if False:
            i = 10
            return i + 15
        scorer = FCBF()
        weights = scorer(self.zoo, None)
        found = [self.zoo.domain[attr].name for attr in reversed(weights.argsort()[-5:])]
        reference = ['legs', 'milk', 'toothed', 'feathers', 'backbone']
        self.assertEqual(found, reference)
        data = Table(Domain([ContinuousVariable('1'), ContinuousVariable('2')], DiscreteVariable('target')), np.full((2, 2), np.nan), np.r_[0.0, 1])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value.*double_scalars')
            warnings.filterwarnings('ignore', 'invalid value.*true_divide')
            weights = scorer(data, None)
            np.testing.assert_equal(weights, np.nan)

    def test_learner_with_transformation(self):
        if False:
            i = 10
            return i + 15
        learner = RandomForestLearner(random_state=0)
        iris = Table('iris')
        data = PCA(n_components=2)(iris)(iris)
        scores = learner.score_data(data)
        np.testing.assert_almost_equal(scores, [[0.7760495, 0.2239505]])

    def test_learner_transform_without_variable(self):
        if False:
            return 10
        data = self.housing

        def preprocessor_random_column(data):
            if False:
                while True:
                    i = 10

            def random_column(d):
                if False:
                    i = 10
                    return i + 15
                return np.random.RandomState(42).rand(len(d))
            nat = ContinuousVariable('nat', compute_value=random_column)
            ndom = Domain(data.domain.attributes + (nat,), data.domain.class_vars)
            return data.transform(ndom)
        learner = RandomForestLearner(random_state=42, preprocessors=[])
        scores1 = learner.score_data(preprocessor_random_column(data))
        learner = RandomForestLearner(random_state=42, preprocessors=[preprocessor_random_column])
        scores2 = learner.score_data(data)
        np.testing.assert_equal(scores1[0][:-1], scores2[0])
if __name__ == '__main__':
    unittest.main()