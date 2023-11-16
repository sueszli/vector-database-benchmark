import unittest
import warnings
from sklearn.exceptions import ConvergenceWarning
from Orange.data import Table
from Orange.classification import NNClassificationLearner
from Orange.modelling import NNLearner, ConstantLearner
from Orange.regression import NNRegressionLearner
from Orange.evaluation import CA, CrossValidation, MSE

class TestNNLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.iris = Table('iris')
        cls.housing = Table('housing')
        cls.learner = NNLearner()

    def setUp(self):
        if False:
            print('Hello World!')
        warnings.filterwarnings('ignore', '.*', ConvergenceWarning)
        super().setUp()

    def test_NN_classification(self):
        if False:
            print('Hello World!')
        cv = CrossValidation(k=3)
        results = cv(self.iris, [NNClassificationLearner()])
        ca = CA(results)
        self.assertGreater(ca, 0.8)
        self.assertLess(ca, 0.99)

    def test_NN_regression(self):
        if False:
            while True:
                i = 10
        const = ConstantLearner()
        cv = CrossValidation(k=3)
        results = cv(self.housing, [NNRegressionLearner(), const])
        mse = MSE()
        res = mse(results)
        self.assertLess(res[0], 35)
        self.assertLess(res[0], res[1])

    def test_NN_model(self):
        if False:
            i = 10
            return i + 15
        cv = CrossValidation(k=3)
        results = cv(self.iris, [self.learner])
        self.assertGreater(CA(results), 0.9)
        cv = CrossValidation(k=3)
        results = cv(self.housing, [self.learner])
        mse = MSE()
        res = mse(results)
        self.assertLess(res[0], 35)

    def test_NN_classification_predict_single_instance(self):
        if False:
            while True:
                i = 10
        lrn = NNClassificationLearner()
        clf = lrn(self.iris)
        for ins in self.iris[::20]:
            clf(ins)
            (_, _) = clf(ins, clf.ValueProbs)

    def test_NN_regression_predict_single_instance(self):
        if False:
            while True:
                i = 10
        lrn = NNRegressionLearner()
        clf = lrn(self.housing)
        for ins in self.housing[::20]:
            clf(ins)

    def test_supports_weights(self):
        if False:
            while True:
                i = 10
        self.assertFalse(NNRegressionLearner().supports_weights)
        self.assertFalse(NNClassificationLearner().supports_weights)