import unittest
import numpy as np
import scipy.sparse as sp
from Orange.data import Table
from Orange.preprocess import Randomize

class TestRandomizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        np.random.seed(42)
        cls.zoo = Table('zoo')

    def test_randomize_default(self):
        if False:
            print('Hello World!')
        data = self.zoo
        randomizer = Randomize()
        data_rand = randomizer(data)
        self.assertTrue((data.X == data_rand.X).all())
        self.assertTrue((data.metas == data_rand.metas).all())
        self.assertTrue((data.Y != data_rand.Y).any())
        self.assertTrue((np.sort(data.Y, axis=0) == np.sort(data_rand.Y, axis=0)).all())

    def test_randomize_classes(self):
        if False:
            return 10
        data = self.zoo
        randomizer = Randomize(rand_type=Randomize.RandomizeClasses)
        data_rand = randomizer(data)
        self.assertTrue((data.X == data_rand.X).all())
        self.assertTrue((data.metas == data_rand.metas).all())
        self.assertTrue((data.Y != data_rand.Y).any())
        self.assertTrue((np.sort(data.Y, axis=0) == np.sort(data_rand.Y, axis=0)).all())

    def test_randomize_attributes(self):
        if False:
            i = 10
            return i + 15
        data = self.zoo
        randomizer = Randomize(rand_type=Randomize.RandomizeAttributes)
        data_rand = randomizer(data)
        self.assertTrue((data.Y == data_rand.Y).all())
        self.assertTrue((data.metas == data_rand.metas).all())
        self.assertTrue((data.X != data_rand.X).any())
        self.assertTrue((np.sort(data.X, axis=0) == np.sort(data_rand.X, axis=0)).all())

    def test_randomize_metas(self):
        if False:
            for i in range(10):
                print('nop')
        data = self.zoo
        randomizer = Randomize(rand_type=Randomize.RandomizeMetas)
        data_rand = randomizer(data)
        self.assertTrue((data.X == data_rand.X).all())
        self.assertTrue((data.Y == data_rand.Y).all())
        self.assertTrue((data.metas != data_rand.metas).any())
        self.assertTrue((np.sort(data.metas, axis=0) == np.sort(data_rand.metas, axis=0)).all())

    def test_randomize_all(self):
        if False:
            while True:
                i = 10
        data = self.zoo
        rand_type = Randomize.RandomizeClasses | Randomize.RandomizeAttributes | Randomize.RandomizeMetas
        randomizer = Randomize(rand_type=rand_type)
        data_rand = randomizer(data)
        self.assertTrue((data.Y != data_rand.Y).any())
        self.assertTrue((np.sort(data.Y, axis=0) == np.sort(data_rand.Y, axis=0)).all())
        self.assertTrue((data.X != data_rand.X).any())
        self.assertTrue((np.sort(data.X, axis=0) == np.sort(data_rand.X, axis=0)).all())
        self.assertTrue((data.metas != data_rand.metas).any())
        self.assertTrue((np.sort(data.metas, axis=0) == np.sort(data_rand.metas, axis=0)).all())

    def test_randomize_keep_original_data(self):
        if False:
            while True:
                i = 10
        data_orig = self.zoo
        data = Table('zoo')
        _ = Randomize(rand_type=Randomize.RandomizeClasses)(data)
        _ = Randomize(rand_type=Randomize.RandomizeAttributes)(data)
        _ = Randomize(rand_type=Randomize.RandomizeMetas)(data)
        self.assertTrue((data.X == data_orig.X).all())
        self.assertTrue((data.metas == data_orig.metas).all())
        self.assertTrue((data.Y == data_orig.Y).all())

    def test_randomize_replicate(self):
        if False:
            while True:
                i = 10
        randomizer1 = Randomize(rand_seed=1)
        rand_data11 = randomizer1(self.zoo)
        rand_data12 = randomizer1(self.zoo)
        randomizer2 = Randomize(rand_seed=1)
        rand_data2 = randomizer2(self.zoo)
        np.testing.assert_array_equal(rand_data11.Y, rand_data12.Y)
        np.testing.assert_array_equal(rand_data11.Y, rand_data2.Y)

    def test_randomize(self):
        if False:
            print('Hello World!')
        x = np.arange(10000, dtype=int).reshape((100, 100))
        randomized = Randomize().randomize(x.copy())
        np.testing.assert_equal(randomized % 100, x % 100)
        randomized = np.array(sorted(list(map(list, randomized))), dtype=int)
        self.assertFalse(np.all(randomized == x))

    def test_randomize_sparse(self):
        if False:
            return 10
        x = np.array([[0, 0, 3, 0], [1, 0, 2, 0], [4, 5, 6, 7]])
        randomize = Randomize().randomize
        randomized = randomize(sp.csr_matrix(x), rand_state=1)
        randomized = randomized.toarray()
        self.assertFalse(np.all(x == randomized))
        self.assertTrue(all((sorted(x[:, i]) == sorted(randomized[:, i]) for i in range(4))))
        randomized = np.array(sorted(list(map(list, randomized))), dtype=int)
        self.assertFalse(np.all(randomized == x))
        x = np.array([[1, 2, 3, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        randomized = randomize(sp.csr_matrix(x), rand_state=14655)
        self.assertFalse(np.all(x == randomized.todense()))
        r_once = randomize(sp.csr_matrix(x), rand_state=1)
        r_twice = randomize(r_once.copy(), rand_state=1)
        self.assertFalse(np.all(r_once.todense() == r_twice.todense()))