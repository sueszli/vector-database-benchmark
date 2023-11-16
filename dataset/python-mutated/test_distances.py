from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import copy
import unittest
import numpy as np
import turicreate as tc
from turicreate.toolkits._main import ToolkitError
from collections import Counter
import sys
if sys.version_info.major > 2:
    unittest.TestCase.assertItemsEqual = unittest.TestCase.assertCountEqual

class StandardDistancesTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            for i in range(10):
                print('nop')
        self.a = {'a': 0.5, 'b': 0.7}
        self.b = {'b': 1.0, 'c': 0.1, 'd': 0.5}
        self.av = [3, 4, 1]
        self.bv = [1, 2, 3]
        self.al = ['a', 'b', 'b', 'c']
        self.bl = ['a', 'b']

    def test_euclidean(self):
        if False:
            print('Hello World!')
        self.assertAlmostEqual(euclidean(self.a, self.b), tc.distances.euclidean(self.a, self.b))
        self.assertAlmostEqual((2 * 2 + 2 * 2 + 2 * 2) ** 0.5, tc.distances.euclidean(self.av, self.bv))

    def test_squared_euclidean(self):
        if False:
            return 10
        self.assertAlmostEqual(euclidean(self.a, self.b) ** 2, tc.distances.squared_euclidean(self.a, self.b))

    def test_manhattan(self):
        if False:
            print('Hello World!')
        self.assertAlmostEqual(manhattan(self.a, self.b), tc.distances.manhattan(self.a, self.b))

    def test_cosine(self):
        if False:
            i = 10
            return i + 15
        self.assertAlmostEqual(cosine(self.a, self.b), tc.distances.cosine(self.a, self.b))

    def test_transformed_dot_product(self):
        if False:
            print('Hello World!')
        self.assertAlmostEqual(transformed_dot_product(self.a, self.b), tc.distances.transformed_dot_product(self.a, self.b))

    def test_jaccard(self):
        if False:
            return 10
        self.assertAlmostEqual(jaccard(self.a, self.b), tc.distances.jaccard(self.a, self.b))
        self.assertAlmostEqual(jaccard(self.al, self.bl), tc.distances.jaccard(self.al, self.bl))

    def test_weighted_jaccard(self):
        if False:
            print('Hello World!')
        self.assertAlmostEqual(weighted_jaccard(self.a, self.b), tc.distances.weighted_jaccard(self.a, self.b))
        self.assertAlmostEqual(weighted_jaccard(self.al, self.bl), tc.distances.weighted_jaccard(self.al, self.bl))

    def test_edge_cases(self):
        if False:
            while True:
                i = 10
        self.assertAlmostEqual(tc.distances.euclidean({}, {}), 0.0)
        self.assertAlmostEqual(tc.distances.euclidean({}, {'a': 1.0}), 1.0)
        self.assertAlmostEqual(tc.distances.jaccard({}, {}), 0.0)
        dists = ['euclidean', 'squared_euclidean', 'manhattan', 'cosine', 'jaccard', 'weighted_jaccard', 'levenshtein']
        for d in dists:
            dist_fn = tc.distances.__dict__[d]
            with self.assertRaises(ToolkitError):
                dist_fn([1.0], {'a': 1.0})
            with self.assertRaises(ToolkitError):
                dist_fn(5.0, 7.0)

class DistanceUtilsTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            while True:
                i = 10
        self.x = {'a': 1.0, 'b': 1.0, 'c': 1, 'd': [1.0, 2.0, 3.0], 'e': {'cat': 10, 'dog': 11, 'fossa': 12}, 'f': 'what on earth is a fossa?'}
        self.y = {'a': 2.0, 'b': 3.0, 'c': 4.0, 'd': [4.0, 5.0, 6.0], 'e': {'eel': 5, 'dog': 12, 'fossa': 10}, 'f': 'a fossa is the best animal on earth'}
        self.dist = [[('a', 'b', 'c'), 'euclidean', 1], [('d',), 'manhattan', 2], [('e',), 'jaccard', 1.5], [('f',), 'levenshtein', 0.3]]

    def test_composite_dist_validation(self):
        if False:
            while True:
                i = 10
        '\n        Make sure the composite distance validation utility allows good\n        distances through and catches bad distances.\n        '
        try:
            tc.distances._util._validate_composite_distance(self.dist)
        except:
            assert False, 'Composite distance validation failed for a good distance.'
        with self.assertRaises(ValueError):
            tc.distances._util._validate_composite_distance([])
        with self.assertRaises(TypeError):
            tc.distances._util._validate_composite_distance(tuple(self.dist))
        dist = copy.deepcopy(self.dist)
        dist.append([[], 'euclidean', 13])
        with self.assertRaises(ValueError):
            tc.distances._util._validate_composite_distance(dist)
        dist = copy.deepcopy(self.dist)
        dist.append([['test', 17], 'manhattan', 13])
        with self.assertRaises(TypeError):
            tc.distances._util._validate_composite_distance(dist)
        dist = copy.deepcopy(self.dist)
        dist.append([['d'], 17, 13])
        with self.assertRaises(ValueError):
            tc.distances._util._validate_composite_distance(dist)
        dist = copy.deepcopy(self.dist)
        dist.append([['d'], 'haversine', 13])
        with self.assertRaises(ValueError):
            tc.distances._util._validate_composite_distance(dist)
        dist = copy.deepcopy(self.dist)
        dist.append([['d'], 'euclidean', 'a lot'])
        with self.assertRaises(ValueError):
            tc.distances._util._validate_composite_distance(dist)

    def test_composite_feature_scrub(self):
        if False:
            return 10
        '\n        Make sure excluded features are properly removed from a composite\n        distance specification.\n        '
        dist = [[('a', 'b', 'c', 'goat'), 'euclidean', 1], [('d', 'horse', 'goat'), 'manhattan', 2], [('e', 'ibex', 'ibex'), 'jaccard', 1.5], [('f',), 'levenshtein', 0.3]]
        feature_denylist = ['goat', 'horse', 'ibex']
        ans = tc.distances._util._scrub_composite_distance_features(dist, feature_denylist)
        for (d, d_ans) in zip(self.dist, ans):
            self.assertSequenceEqual(d[0], d_ans[0])
        feature_denylist.append('f')
        ans = tc.distances._util._scrub_composite_distance_features(dist, feature_denylist)
        self.assertEqual(len(ans), 3)
        self.assertItemsEqual(tc.distances._util._get_composite_distance_features(ans), ['a', 'b', 'c', 'd', 'e'])

    def test_composite_dist_type_convert(self):
        if False:
            while True:
                i = 10
        '\n        Make sure the utility to convert distance names to function handles\n        works properly.\n        '
        converted_dist = tc.distances._util._convert_distance_names_to_functions(self.dist)
        ans = [tc.distances.euclidean, tc.distances.manhattan, tc.distances.jaccard, tc.distances.levenshtein]
        self.assertSequenceEqual(ans, [x[1] for x in converted_dist])

    def test_composite_dist_compute(self):
        if False:
            print('Hello World!')
        '\n        Check the correctness of the composite distance computation utility.\n        '
        d = tc.distances.compute_composite_distance(self.dist, self.x, self.x)
        self.assertAlmostEqual(d, 0.0)
        d = tc.distances.compute_composite_distance(self.dist, self.y, self.y)
        self.assertAlmostEqual(d, 0.0)
        d = tc.distances.compute_composite_distance(self.dist, self.x, self.y)
        self.assertAlmostEqual(d, 30.29165739, places=5)
        sf = tc.SFrame([self.x, self.y]).unpack('X1', column_name_prefix='')
        m = tc.nearest_neighbors.create(sf, distance=self.dist, verbose=False)
        knn = m.query(sf[:1], k=2, verbose=False)
        self.assertAlmostEqual(d, knn['distance'][1], places=5)

    def test_composite_features_extract(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the utility that returns the union of features in a composite\n        distance.\n        '
        dist = copy.deepcopy(self.dist)
        dist.append([['a', 'b', 'a'], 'cosine', 13])
        ans = ['a', 'b', 'c', 'd', 'e', 'f']
        self.assertItemsEqual(ans, tc.distances._util._get_composite_distance_features(dist))

class LocalDistancesTest(unittest.TestCase):
    """
    Unit test for the distances computed in this script.
    """

    @classmethod
    def setUpClass(self):
        if False:
            return 10
        self.a = {'a': 0.5, 'b': 0.7}
        self.b = {'b': 1.0, 'c': 0.1, 'd': 0.5}
        self.S = 'fossa'
        self.T = 'fossil'

    def test_local_jaccard(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertAlmostEqual(jaccard(self.a, self.b), 1 - 1.0 / 4)
        self.assertAlmostEqual(jaccard(self.a, {}), 1)
        self.assertAlmostEqual(jaccard(self.a, self.a), 0)

    def test_local_weighted_jaccard(self):
        if False:
            for i in range(10):
                print('nop')
        ans = 1 - (0.0 + 0.7 + 0.0 + 0.0) / (0.5 + 1.0 + 0.1 + 0.5)
        self.assertAlmostEqual(weighted_jaccard(self.a, self.b), ans)
        self.assertAlmostEqual(weighted_jaccard(self.a, {}), 1)
        self.assertAlmostEqual(weighted_jaccard(self.a, self.a), 0)

    def test_local_cosine(self):
        if False:
            i = 10
            return i + 15
        ans = 1 - 0.7 / ((0.5 ** 2 + 0.7 ** 2) ** 0.5 * (1 ** 2 + 0.1 ** 2 + 0.5 ** 2) ** 0.5)
        self.assertAlmostEqual(cosine(self.a, self.b), ans)
        self.assertAlmostEqual(cosine(self.a, {}), 1)
        self.assertAlmostEqual(cosine(self.a, self.a), 0)

    def test_local_transformed_dot_product(self):
        if False:
            print('Hello World!')
        ans = np.log(1.0 + np.exp(-0.7))
        self.assertAlmostEqual(transformed_dot_product(self.a, self.b), ans)
        ans = np.log(1 + np.exp(-1 * (0.5 ** 2 + 0.7 ** 2)))
        self.assertAlmostEqual(transformed_dot_product(self.a, self.a), ans)

    def test_local_euclidean(self):
        if False:
            while True:
                i = 10
        self.assertAlmostEqual(euclidean(self.a, self.a), 0)
        ans = (0.5 ** 2 + (1.0 - 0.7) ** 2 + 0.1 ** 2 + 0.5 ** 2) ** 0.5
        self.assertAlmostEqual(euclidean(self.a, self.b), ans)
        ans = (0.5 ** 2 + 0.7 ** 2) ** 0.5
        self.assertAlmostEqual(euclidean(self.a, {}), ans)

    def test_local_squared_euclidean(self):
        if False:
            i = 10
            return i + 15
        self.assertAlmostEqual(squared_euclidean(self.a, self.a), 0)
        ans = 0.5 ** 2 + (1.0 - 0.7) ** 2 + 0.1 ** 2 + 0.5 ** 2
        self.assertAlmostEqual(squared_euclidean(self.a, self.b), ans)
        ans = 0.5 ** 2 + 0.7 ** 2
        self.assertAlmostEqual(squared_euclidean(self.a, {}), ans)

    def test_local_manhattan(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertAlmostEqual(manhattan(self.a, self.a), 0)
        ans = 0.5 + (1.0 - 0.7) + 0.1 + 0.5
        self.assertAlmostEqual(manhattan(self.a, self.b), ans)
        ans = 0.5 + 0.7
        self.assertAlmostEqual(manhattan(self.a, {}), ans)

    def test_local_levenshtein(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(levenshtein(self.S, self.T), 2)
        self.assertEqual(levenshtein(self.S, self.S), 0)
        self.assertEqual(levenshtein(self.T, self.T), 0)
        self.assertEqual(levenshtein(self.S, ''), len(self.S))
        self.assertEqual(levenshtein(self.T, ''), len(self.T))

def jaccard(a, b):
    if False:
        return 10
    if isinstance(a, dict) and isinstance(b, dict):
        a = a.keys()
        b = b.keys()
    a = set(a)
    b = set(b)
    ans = 1.0 - float(len(a.intersection(b))) / len(a.union(b))
    return ans

def weighted_jaccard(a, b):
    if False:
        while True:
            i = 10
    if isinstance(a, list) and isinstance(b, list):
        a = dict(Counter(a))
        b = dict(Counter(b))
    a2 = a.copy()
    b2 = b.copy()
    numer = 0
    denom = 0
    keys = set(list(a.keys()) + list(b.keys()))
    for k in keys:
        a2.setdefault(k, 0)
        b2.setdefault(k, 0)
        numer += min(a2[k], b2[k])
        denom += max(a2[k], b2[k])
    return 1.0 - float(numer) / denom

def cosine(a, b):
    if False:
        while True:
            i = 10
    ks = set(a.keys()).intersection(set(b.keys()))
    num = sum([a[k] * b[k] for k in ks])
    den = sum([v ** 2 for (k, v) in a.items()]) * sum([v ** 2 for (k, v) in b.items()])
    den = den ** 0.5
    if den == 0:
        den = 0.0001
    return 1 - num / den

def transformed_dot_product(a, b):
    if False:
        for i in range(10):
            print('nop')
    ks = set(a.keys()).intersection(set(b.keys()))
    dotprod = sum([a[k] * b[k] for k in ks])
    return np.log(1 + np.exp(-1 * dotprod))

def euclidean(a, b):
    if False:
        print('Hello World!')
    return squared_euclidean(a, b) ** 0.5

def squared_euclidean(a, b):
    if False:
        print('Hello World!')
    a2 = a.copy()
    b2 = b.copy()
    ans = 0
    keys = set(a.keys()).union(set(b.keys()))
    for k in keys:
        a2.setdefault(k, 0)
        b2.setdefault(k, 0)
        ans += (a2[k] - b2[k]) ** 2
    return ans

def manhattan(a, b):
    if False:
        i = 10
        return i + 15
    a2 = a.copy()
    b2 = b.copy()
    ans = 0
    keys = set(a.keys()).union(set(b.keys()))
    for k in keys:
        a2.setdefault(k, 0)
        b2.setdefault(k, 0)
        ans += abs(a2[k] - b2[k])
    return ans

def levenshtein(a, b):
    if False:
        i = 10
        return i + 15
    m = len(a)
    n = len(b)
    D = np.zeros((m + 1, n + 1), dtype=int)
    D[:, 0] = np.arange(m + 1)
    D[0, :] = np.arange(n + 1)
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)
    return D[m, n]