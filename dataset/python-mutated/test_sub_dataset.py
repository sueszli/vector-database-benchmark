import unittest
from chainer import datasets
from chainer import testing

class TestSubDataset(unittest.TestCase):

    def test_sub_dataset(self):
        if False:
            for i in range(10):
                print('nop')
        original = [1, 2, 3, 4, 5]
        subset = datasets.SubDataset(original, 1, 4)
        self.assertEqual(len(subset), 3)
        self.assertEqual(subset[0], 2)
        self.assertEqual(subset[1], 3)
        self.assertEqual(subset[2], 4)

    def test_sub_dataset_overrun(self):
        if False:
            while True:
                i = 10
        original = [1, 2, 3, 4, 5]
        subset = datasets.SubDataset(original, 1, 4)
        with self.assertRaises(IndexError):
            subset[len(subset)]

    def test_permuted_sub_dataset(self):
        if False:
            return 10
        original = [1, 2, 3, 4, 5]
        subset = datasets.SubDataset(original, 1, 4, [2, 0, 3, 1, 4])
        self.assertEqual(len(subset), 3)
        self.assertEqual(subset[0], 1)
        self.assertEqual(subset[1], 4)
        self.assertEqual(subset[2], 2)

    def test_permuted_sub_dataset_len_mismatch(self):
        if False:
            while True:
                i = 10
        original = [1, 2, 3, 4, 5]
        with self.assertRaises(ValueError):
            datasets.SubDataset(original, 1, 4, [2, 0, 3, 1])

class TestSplitDataset(unittest.TestCase):

    def test_split_dataset(self):
        if False:
            print('Hello World!')
        original = [1, 2, 3, 4, 5]
        (subset1, subset2) = datasets.split_dataset(original, 2)
        self.assertEqual(len(subset1), 2)
        self.assertEqual(subset1[0], 1)
        self.assertEqual(subset1[1], 2)
        self.assertEqual(len(subset2), 3)
        self.assertEqual(subset2[0], 3)
        self.assertEqual(subset2[1], 4)
        self.assertEqual(subset2[2], 5)

    def test_split_dataset_head(self):
        if False:
            while True:
                i = 10
        original = [1, 2, 3, 4, 5]
        (subset1, subset2) = datasets.split_dataset(original, 0)
        self.assertEqual(len(subset1), 0)
        self.assertEqual(len(subset2), 5)

    def test_split_dataset_tail(self):
        if False:
            for i in range(10):
                print('nop')
        original = [1, 2, 3, 4, 5]
        (subset1, subset2) = datasets.split_dataset(original, 5)
        self.assertEqual(len(subset1), 5)
        self.assertEqual(len(subset2), 0)

    def test_split_dataset_invalid_position(self):
        if False:
            for i in range(10):
                print('nop')
        original = [1, 2, 3, 4, 5]
        with self.assertRaises(ValueError):
            datasets.split_dataset(original, -1)
        with self.assertRaises(ValueError):
            datasets.split_dataset(original, 6)

    def test_split_dataset_invalid_type(self):
        if False:
            while True:
                i = 10
        original = [1, 2, 3, 4, 5]
        with self.assertRaises(TypeError):
            datasets.split_dataset(original, 3.5)

    def test_permuted_split_dataset(self):
        if False:
            while True:
                i = 10
        original = [1, 2, 3, 4, 5]
        (subset1, subset2) = datasets.split_dataset(original, 2, [2, 0, 3, 1, 4])
        self.assertEqual(len(subset1), 2)
        self.assertEqual(subset1[0], 3)
        self.assertEqual(subset1[1], 1)
        self.assertEqual(len(subset2), 3)
        self.assertEqual(subset2[0], 4)
        self.assertEqual(subset2[1], 2)
        self.assertEqual(subset2[2], 5)

    def test_split_dataset_with_invalid_length_permutation(self):
        if False:
            return 10
        original = [1, 2, 3, 4, 5]
        with self.assertRaises(ValueError):
            datasets.split_dataset(original, 2, [2, 0, 3, 1])
        with self.assertRaises(ValueError):
            datasets.split_dataset(original, 2, [2, 0, 3, 1, 4, 5])

    def test_split_dataset_random(self):
        if False:
            i = 10
            return i + 15
        original = [1, 2, 3, 4, 5]
        (subset1, subset2) = datasets.split_dataset_random(original, 2)
        reconst = sorted(set(subset1).union(subset2))
        self.assertEqual(reconst, original)
        (subset1a, subset2a) = datasets.split_dataset_random(original, 2, seed=3)
        reconst = sorted(set(subset1a).union(subset2a))
        self.assertEqual(reconst, original)
        (subset1b, subset2b) = datasets.split_dataset_random(original, 2, seed=3)
        self.assertEqual(set(subset1a), set(subset1b))
        self.assertEqual(set(subset2a), set(subset2b))
        reconst = sorted(set(subset1a).union(subset2a))
        self.assertEqual(reconst, original)

    def test_split_dataset_n(self):
        if False:
            print('Hello World!')
        original = list(range(7))
        subsets = datasets.split_dataset_n(original, 3)
        self.assertEqual(len(subsets), 3)
        self.assertEqual(list(subsets[0]), original[:2])
        self.assertEqual(list(subsets[1]), original[2:4])
        self.assertEqual(list(subsets[2]), original[4:6])
        order = list(range(6, -1, -1))
        subsets = datasets.split_dataset_n(original, 2, order)
        self.assertEqual(len(subsets), 2)
        self.assertEqual(list(subsets[0]), [6, 5, 4])
        self.assertEqual(list(subsets[1]), [3, 2, 1])
        original = list(range(6))
        subsets = datasets.split_dataset_n(original, 3)
        self.assertEqual(len(subsets), 3)
        self.assertEqual(list(subsets[0]), original[:2])
        self.assertEqual(list(subsets[1]), original[2:4])
        self.assertEqual(list(subsets[2]), original[4:6])

    def test_split_dataset_n_random(self):
        if False:
            return 10
        original = list(range(6))
        subsets = datasets.split_dataset_n_random(original, 2)
        reconst = sorted(set(subsets[0]).union(subsets[1]))
        self.assertEqual(reconst, original)
        subsets1 = datasets.split_dataset_n_random(original, 2, seed=3)
        reconst = sorted(set(subsets1[0]).union(subsets1[1]))
        self.assertEqual(reconst, original)
        subsets2 = datasets.split_dataset_n_random(original, 2, seed=3)
        self.assertEqual(set(subsets1[0]), set(subsets2[0]))
        self.assertEqual(set(subsets1[1]), set(subsets2[1]))
        original = list(range(7))
        subsets = datasets.split_dataset_n_random(original, 3)
        self.assertEqual(len(subsets), 3)
        for subset in subsets:
            self.assertEqual(len(subset), 2)
        reconst = set(subsets[0]).union(subsets[1]).union(subsets[2])
        self.assertEqual(len(reconst), 6)

class TestGetCrossValidationDatasets(unittest.TestCase):

    def test_get_cross_validation_datasets(self):
        if False:
            i = 10
            return i + 15
        original = [1, 2, 3, 4, 5, 6]
        (cv1, cv2, cv3) = datasets.get_cross_validation_datasets(original, 3)
        (tr1, te1) = cv1
        self.assertEqual(len(tr1), 4)
        self.assertEqual(tr1[0], 1)
        self.assertEqual(tr1[1], 2)
        self.assertEqual(tr1[2], 3)
        self.assertEqual(tr1[3], 4)
        self.assertEqual(len(te1), 2)
        self.assertEqual(te1[0], 5)
        self.assertEqual(te1[1], 6)
        (tr2, te2) = cv2
        self.assertEqual(len(tr2), 4)
        self.assertEqual(tr2[0], 5)
        self.assertEqual(tr2[1], 6)
        self.assertEqual(tr2[2], 1)
        self.assertEqual(tr2[3], 2)
        self.assertEqual(len(te2), 2)
        self.assertEqual(te2[0], 3)
        self.assertEqual(te2[1], 4)
        (tr3, te3) = cv3
        self.assertEqual(len(tr3), 4)
        self.assertEqual(tr3[0], 3)
        self.assertEqual(tr3[1], 4)
        self.assertEqual(tr3[2], 5)
        self.assertEqual(tr3[3], 6)
        self.assertEqual(len(te3), 2)
        self.assertEqual(te3[0], 1)
        self.assertEqual(te3[1], 2)

    def test_get_cross_validation_datasets_2(self):
        if False:
            print('Hello World!')
        original = [1, 2, 3, 4, 5, 6, 7]
        (cv1, cv2, cv3) = datasets.get_cross_validation_datasets(original, 3)
        (tr1, te1) = cv1
        self.assertEqual(len(tr1), 4)
        self.assertEqual(tr1[0], 1)
        self.assertEqual(tr1[1], 2)
        self.assertEqual(tr1[2], 3)
        self.assertEqual(tr1[3], 4)
        self.assertEqual(len(te1), 3)
        self.assertEqual(te1[0], 5)
        self.assertEqual(te1[1], 6)
        self.assertEqual(te1[2], 7)
        (tr2, te2) = cv2
        self.assertEqual(len(tr2), 5)
        self.assertEqual(tr2[0], 5)
        self.assertEqual(tr2[1], 6)
        self.assertEqual(tr2[2], 7)
        self.assertEqual(tr2[3], 1)
        self.assertEqual(tr2[4], 2)
        self.assertEqual(len(te2), 2)
        self.assertEqual(te2[0], 3)
        self.assertEqual(te2[1], 4)
        (tr3, te3) = cv3
        self.assertEqual(len(tr3), 5)
        self.assertEqual(tr3[0], 3)
        self.assertEqual(tr3[1], 4)
        self.assertEqual(tr3[2], 5)
        self.assertEqual(tr3[3], 6)
        self.assertEqual(tr3[4], 7)
        self.assertEqual(len(te3), 2)
        self.assertEqual(te3[0], 1)
        self.assertEqual(te3[1], 2)

    def test_get_cross_validation_datasets_random(self):
        if False:
            while True:
                i = 10
        original = [1, 2, 3, 4, 5, 6]
        cvs = datasets.get_cross_validation_datasets_random(original, 3)
        for (tr, te) in cvs:
            reconst = sorted(set(tr).union(te))
            self.assertEqual(reconst, original)
            self.assertEqual(len(tr) + len(te), len(original))
        validation_union = sorted(list(cvs[0][1]) + list(cvs[1][1]) + list(cvs[2][1]))
        self.assertEqual(validation_union, original)
        cvs_a = datasets.get_cross_validation_datasets_random(original, 3, seed=5)
        cvs_b = datasets.get_cross_validation_datasets_random(original, 3, seed=5)
        for ((tr_a, te_a), (tr_b, te_b)) in zip(cvs_a, cvs_b):
            self.assertEqual(set(tr_a), set(tr_b))
            self.assertEqual(set(te_a), set(te_b))
testing.run_module(__name__, __file__)