import warnings
import numpy as np
from scipy.sparse import csr_matrix, SparseEfficiencyWarning
from Orange import data
from Orange.data import Table
from Orange.tests import test_table as tabletests

class InterfaceTest(tabletests.InterfaceTest):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.table = data.Table.from_numpy(self.domain, csr_matrix(self.table.X), csr_matrix(self.table.Y))

    def test_row_assignment(self):
        if False:
            return 10
        warnings.filterwarnings('ignore', '.*', SparseEfficiencyWarning)
        super().test_row_assignment()

    def test_value_assignment(self):
        if False:
            for i in range(10):
                print('nop')
        warnings.filterwarnings('ignore', '.*', SparseEfficiencyWarning)
        super().test_value_assignment()

    def test_str(self):
        if False:
            return 10
        iris = Table('iris')
        with iris.unlocked():
            (iris.X, iris.Y) = (csr_matrix(iris.X), csr_matrix(iris.Y))
        str(iris)

    def test_Y_setter_1d(self):
        if False:
            i = 10
            return i + 15
        iris = Table('iris')
        assert iris.Y.shape == (150,)
        with iris.unlocked():
            iris.Y = csr_matrix(iris.Y)
        self.assertEqual(iris.Y.shape, (150,))

    def test_Y_setter_2d(self):
        if False:
            while True:
                i = 10
        iris = Table('iris')
        assert iris.Y.shape == (150,)
        new_y = iris.Y[:, np.newaxis]
        with iris.unlocked():
            iris.Y = np.hstack((new_y, new_y))
            iris.Y = csr_matrix(iris.Y)
        self.assertEqual(iris.Y.shape, (150, 2))

    def test_Y_setter_2d_single_instance(self):
        if False:
            return 10
        iris = Table('iris')[:1]
        new_y = iris.Y[:, np.newaxis]
        with iris.unlocked_reference():
            iris.Y = np.hstack((new_y, new_y))
            iris.Y = csr_matrix(iris.Y)
        self.assertEqual(iris.Y.shape, (1, 2))