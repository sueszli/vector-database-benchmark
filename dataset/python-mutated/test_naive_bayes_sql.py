import unittest
from numpy import array
import Orange.classification.naive_bayes as nb
from Orange import preprocess
from Orange.data.sql.table import SqlTable
from Orange.data import Domain
from Orange.data.variable import DiscreteVariable
from Orange.tests.sql.base import DataBaseTest as dbt

class NaiveBayesTest(unittest.TestCase, dbt):

    def setUpDB(self):
        if False:
            return 10
        (self.conn, self.iris) = self.create_iris_sql_table()

    def tearDownDB(self):
        if False:
            print('Hello World!')
        self.drop_iris_sql_table()

    @dbt.run_on(['postgres'])
    def test_NaiveBayes(self):
        if False:
            return 10
        iris_v = ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor']
        table = SqlTable(self.conn, self.iris, type_hints=Domain([], DiscreteVariable('iris', values=iris_v)))
        disc = preprocess.Discretize()
        table = disc(table)
        bayes = nb.NaiveBayesLearner()
        clf = bayes(table)
        self.assertEqual(clf(table[0]), table[0].get_class())
        pred = clf(table)
        actual = array([ins.get_class() for ins in table])
        ca = pred == actual
        ca = ca.sum() / len(ca)
        self.assertGreater(ca, 0.95)
        self.assertLess(ca, 1.0)