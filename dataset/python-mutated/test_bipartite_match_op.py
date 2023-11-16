import unittest
import numpy as np
from op_test import OpTest

def bipartite_match(distance, match_indices, match_dist):
    if False:
        return 10
    'Bipartite Matching algorithm.\n    Arg:\n        distance (numpy.array) : The distance of two entries with shape [M, N].\n        match_indices (numpy.array): the matched indices from column to row\n            with shape [1, N], it must be initialized to -1.\n        match_dist (numpy.array): The matched distance from column to row\n            with shape [1, N], it must be initialized to 0.\n    '
    match_pair = []
    (row, col) = distance.shape
    for i in range(row):
        for j in range(col):
            match_pair.append((i, j, distance[i][j]))
    match_sorted = sorted(match_pair, key=lambda tup: tup[2], reverse=True)
    row_indices = -1 * np.ones((row,), dtype=np.int_)
    idx = 0
    for (i, j, dist) in match_sorted:
        if idx >= row:
            break
        if match_indices[j] == -1 and row_indices[i] == -1 and (dist > 0):
            match_indices[j] = i
            row_indices[i] = j
            match_dist[j] = dist
            idx += 1

def argmax_match(distance, match_indices, match_dist, threshold):
    if False:
        i = 10
        return i + 15
    (r, c) = distance.shape
    for j in range(c):
        if match_indices[j] != -1:
            continue
        col_dist = distance[:, j]
        indices = np.argwhere(col_dist >= threshold).flatten()
        if len(indices) < 1:
            continue
        match_indices[j] = indices[np.argmax(col_dist[indices])]
        match_dist[j] = col_dist[match_indices[j]]

def batch_bipartite_match(distance, lod, match_type=None, dist_threshold=None):
    if False:
        for i in range(10):
            print('nop')
    'Bipartite Matching algorithm for batch input.\n    Arg:\n        distance (numpy.array) : The distance of two entries with shape [M, N].\n        lod (list of int): The length of each input in this batch.\n    '
    n = len(lod)
    m = distance.shape[1]
    match_indices = -1 * np.ones((n, m), dtype=np.int_)
    match_dist = np.zeros((n, m), dtype=np.float32)
    cur_offset = 0
    for i in range(n):
        if lod[i] == 0:
            continue
        bipartite_match(distance[cur_offset:cur_offset + lod[i], :], match_indices[i, :], match_dist[i, :])
        if match_type == 'per_prediction':
            argmax_match(distance[cur_offset:cur_offset + lod[i], :], match_indices[i, :], match_dist[i, :], dist_threshold)
        cur_offset += lod[i]
    return (match_indices, match_dist)

class TestBipartiteMatchOpWithLoD(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'bipartite_match'
        lod = [[5, 6, 12]]
        dist = np.random.random((23, 217)).astype('float32')
        (match_indices, match_dist) = batch_bipartite_match(dist, lod[0])
        self.inputs = {'DistMat': (dist, lod)}
        self.outputs = {'ColToRowMatchIndices': match_indices, 'ColToRowMatchDist': match_dist}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_dygraph=False)

class TestBipartiteMatchOpWithoutLoD(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'bipartite_match'
        lod = [[8]]
        dist = np.random.random((8, 17)).astype('float32')
        (match_indices, match_dist) = batch_bipartite_match(dist, lod[0])
        self.inputs = {'DistMat': dist}
        self.outputs = {'ColToRowMatchIndices': match_indices, 'ColToRowMatchDist': match_dist}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_dygraph=False)

class TestBipartiteMatchOpWithoutLoDLargeScaleInput(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'bipartite_match'
        lod = [[300]]
        dist = np.random.random((300, 17)).astype('float32')
        (match_indices, match_dist) = batch_bipartite_match(dist, lod[0])
        self.inputs = {'DistMat': dist}
        self.outputs = {'ColToRowMatchIndices': match_indices, 'ColToRowMatchDist': match_dist}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_dygraph=False)

class TestBipartiteMatchOpWithPerPredictionType(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'bipartite_match'
        lod = [[5, 6, 12]]
        dist = np.random.random((23, 237)).astype('float32')
        (match_indices, match_dist) = batch_bipartite_match(dist, lod[0], 'per_prediction', 0.5)
        self.inputs = {'DistMat': (dist, lod)}
        self.outputs = {'ColToRowMatchIndices': match_indices, 'ColToRowMatchDist': match_dist}
        self.attrs = {'match_type': 'per_prediction', 'dist_threshold': 0.5}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_dygraph=False)

class TestBipartiteMatchOpWithEmptyLoD(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'bipartite_match'
        lod = [[5, 6, 0, 12]]
        dist = np.random.random((23, 217)).astype('float32')
        (match_indices, match_dist) = batch_bipartite_match(dist, lod[0])
        self.inputs = {'DistMat': (dist, lod)}
        self.outputs = {'ColToRowMatchIndices': match_indices, 'ColToRowMatchDist': match_dist}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_dygraph=False)
if __name__ == '__main__':
    unittest.main()