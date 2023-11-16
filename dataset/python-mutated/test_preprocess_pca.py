import unittest
from Orange.data import Table
from Orange.preprocess import ProjectPCA
from Orange.tests import test_filename

class TestPCAProjector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.ionosphere = Table(test_filename('datasets/ionosphere.tab'))

    def test_project_pca_default(self):
        if False:
            for i in range(10):
                print('nop')
        data = self.ionosphere
        projector = ProjectPCA()
        data_pc = projector(data)
        self.assertEqual(data_pc.X.shape[1], data.X.shape[1])
        self.assertTrue((data.metas == data_pc.metas).all())
        self.assertTrue((data.Y == data_pc.Y).any())

    def test_project_pca(self):
        if False:
            return 10
        data = self.ionosphere
        projector = ProjectPCA(n_components=5)
        data_pc = projector(data)
        self.assertEqual(data_pc.X.shape[1], 5)
        self.assertTrue((data.metas == data_pc.metas).all())
        self.assertTrue((data.Y == data_pc.Y).any())