"""Unit tests for those parts of the Bio.PDB module using Bio.PDB.kdtrees."""
import unittest
try:
    from numpy import array, dot, sqrt, argsort
    from numpy.random import random
except ImportError:
    from Bio import MissingExternalDependencyError
    raise MissingExternalDependencyError('Install NumPy if you want to use Bio.PDB.') from None
try:
    from Bio.PDB import kdtrees
except ImportError:
    from Bio import MissingExternalDependencyError
    raise MissingExternalDependencyError('C module Bio.PDB.kdtrees not compiled') from None
from Bio.PDB.NeighborSearch import NeighborSearch

class NeighborTest(unittest.TestCase):

    def test_neighbor_search(self):
        if False:
            return 10
        'NeighborSearch: Find nearby randomly generated coordinates.\n\n        Based on the self test in Bio.PDB.NeighborSearch.\n        '

        class RandomAtom:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.coord = 100 * random(3)

            def get_coord(self):
                if False:
                    print('Hello World!')
                return self.coord
        for i in range(20):
            atoms = [RandomAtom() for j in range(100)]
            ns = NeighborSearch(atoms)
            hits = ns.search_all(5.0)
            self.assertIsInstance(hits, list)
            self.assertGreaterEqual(len(hits), 0)
        x = array([250, 250, 250])
        self.assertEqual([], ns.search(x, 5.0, 'A'))
        self.assertEqual([], ns.search(x, 5.0, 'R'))
        self.assertEqual([], ns.search(x, 5.0, 'C'))
        self.assertEqual([], ns.search(x, 5.0, 'M'))
        self.assertEqual([], ns.search(x, 5.0, 'S'))

class KDTreeTest(unittest.TestCase):
    nr_points = 5000
    bucket_size = 5
    radius = 0.05
    query_radius = 10

    def test_KDTree_exceptions(self):
        if False:
            return 10
        bucket_size = self.bucket_size
        nr_points = self.nr_points
        radius = self.radius
        coords = random((nr_points, 3)) * 100000000000000
        with self.assertRaises(Exception) as context:
            kdt = kdtrees.KDTree(coords, bucket_size)
        self.assertIn('coordinate values should lie between -1e6 and 1e6', str(context.exception))
        with self.assertRaises(Exception) as context:
            kdt = kdtrees.KDTree(random((nr_points, 3 - 2)), bucket_size)
        self.assertIn('expected a Nx3 numpy array', str(context.exception))

    def test_KDTree_point_search(self):
        if False:
            while True:
                i = 10
        'Test searching all points within a certain radius of center.\n\n        Using the kdtrees C module, search all point pairs that are\n        within radius, and compare the results to a manual search.\n        '
        bucket_size = self.bucket_size
        nr_points = self.nr_points
        for radius in (self.radius, 100 * self.radius):
            for i in range(10):
                coords = random((nr_points, 3))
                center = random(3)
                kdt = kdtrees.KDTree(coords, bucket_size)
                points1 = kdt.search(center, radius)
                points1.sort(key=lambda point: point.index)
                points2 = []
                for i in range(nr_points):
                    p = coords[i]
                    v = p - center
                    r = sqrt(dot(v, v))
                    if r <= radius:
                        point2 = kdtrees.Point(i, r)
                        points2.append(point2)
                self.assertEqual(len(points1), len(points2))
                for (point1, point2) in zip(points1, points2):
                    self.assertEqual(point1.index, point2.index)
                    self.assertAlmostEqual(point1.radius, point2.radius)

    def test_KDTree_neighbor_search_simple(self):
        if False:
            print('Hello World!')
        'Test all fixed radius neighbor search.\n\n        Test all fixed radius neighbor search using the KD tree C\n        module, and compare the results to those of a simple but\n        slow algorithm.\n        '
        bucket_size = self.bucket_size
        nr_points = self.nr_points
        radius = self.radius
        for i in range(10):
            coords = random((nr_points, 3))
            kdt = kdtrees.KDTree(coords, bucket_size)
            neighbors1 = kdt.neighbor_search(radius)
            neighbors2 = kdt.neighbor_simple_search(radius)
            self.assertEqual(len(neighbors1), len(neighbors2))
            key = lambda neighbor: (neighbor.index1, neighbor.index2)
            neighbors1.sort(key=key)
            neighbors2.sort(key=key)
            for (neighbor1, neighbor2) in zip(neighbors1, neighbors2):
                self.assertEqual(neighbor1.index1, neighbor2.index1)
                self.assertEqual(neighbor1.index2, neighbor2.index2)
                self.assertAlmostEqual(neighbor1.radius, neighbor2.radius)

    def test_KDTree_neighbor_search_manual(self):
        if False:
            i = 10
            return i + 15
        'Test all fixed radius neighbor search.\n\n        Test all fixed radius neighbor search using the KD tree C\n        module, and compare the results to those of a manual search.\n        '
        bucket_size = self.bucket_size
        nr_points = self.nr_points // 10
        for radius in (self.radius, 3 * self.radius):
            for i in range(5):
                coords = random((nr_points, 3))
                kdt = kdtrees.KDTree(coords, bucket_size)
                neighbors1 = kdt.neighbor_search(radius)
                neighbors2 = []
                indices = argsort(coords[:, 0])
                for j1 in range(nr_points):
                    index1 = indices[j1]
                    p1 = coords[index1]
                    for j2 in range(j1 + 1, nr_points):
                        index2 = indices[j2]
                        p2 = coords[index2]
                        if p2[0] - p1[0] > radius:
                            break
                        v = p1 - p2
                        r = sqrt(dot(v, v))
                        if r <= radius:
                            if index1 < index2:
                                (i1, i2) = (index1, index2)
                            else:
                                (i1, i2) = (index2, index1)
                            neighbor = kdtrees.Neighbor(i1, i2, r)
                            neighbors2.append(neighbor)
                self.assertEqual(len(neighbors1), len(neighbors2))
                key = lambda neighbor: (neighbor.index1, neighbor.index2)
                neighbors1.sort(key=key)
                neighbors2.sort(key=key)
                for (neighbor1, neighbor2) in zip(neighbors1, neighbors2):
                    self.assertEqual(neighbor1.index1, neighbor2.index1)
                    self.assertEqual(neighbor1.index2, neighbor2.index2)
                    self.assertAlmostEqual(neighbor1.radius, neighbor2.radius)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)