import hashlib
import numpy as np
from .constants import INPUT, LABEL2ID, OUTPUT

def _labeling_from_architecture(architecture, vertices):
    if False:
        i = 10
        return i + 15
    return [INPUT] + [architecture['op{}'.format(i)] for i in range(1, vertices - 1)] + [OUTPUT]

def _adjancency_matrix_from_architecture(architecture, vertices):
    if False:
        return 10
    matrix = np.zeros((vertices, vertices), dtype=bool)
    for i in range(1, vertices):
        for k in architecture['input{}'.format(i)]:
            matrix[k, i] = 1
    return matrix

def nasbench_format_to_architecture_repr(adjacency_matrix, labeling):
    if False:
        for i in range(10):
            print('nop')
    '\n    Computes a graph-invariance MD5 hash of the matrix and label pair.\n    Imported from NAS-Bench-101 repo.\n\n    Parameters\n    ----------\n    adjacency_matrix : np.ndarray\n        A 2D array of shape NxN, where N is the number of vertices.\n        ``matrix[u][v]`` is 1 if there is a direct edge from `u` to `v`,\n        otherwise it will be 0.\n    labeling : list of str\n        A list of str that starts with input and ends with output. The intermediate\n        nodes are chosen from candidate operators.\n\n    Returns\n    -------\n    tuple and int and dict\n        Converted number of vertices and architecture.\n    '
    num_vertices = adjacency_matrix.shape[0]
    assert len(labeling) == num_vertices
    architecture = {}
    for i in range(1, num_vertices - 1):
        architecture['op{}'.format(i)] = labeling[i]
        assert labeling[i] not in [INPUT, OUTPUT]
    for i in range(1, num_vertices):
        architecture['input{}'.format(i)] = [k for k in range(i) if adjacency_matrix[k, i]]
    return (num_vertices, architecture)

def infer_num_vertices(architecture):
    if False:
        i = 10
        return i + 15
    '\n    Infer number of vertices from an architecture dict.\n\n    Parameters\n    ----------\n    architecture : dict\n        Architecture in NNI format.\n\n    Returns\n    -------\n    int\n        Number of vertices.\n    '
    op_keys = set([k for k in architecture.keys() if k.startswith('op')])
    intermediate_vertices = len(op_keys)
    assert op_keys == {'op{}'.format(i) for i in range(1, intermediate_vertices + 1)}
    return intermediate_vertices + 2

def hash_module(architecture, vertices):
    if False:
        return 10
    '\n    Computes a graph-invariance MD5 hash of the matrix and label pair.\n    This snippet is modified from code in NAS-Bench-101 repo.\n\n    Parameters\n    ----------\n    matrix : np.ndarray\n        Square upper-triangular adjacency matrix.\n    labeling : list of int\n        Labels of length equal to both dimensions of matrix.\n\n    Returns\n    -------\n    str\n        MD5 hash of the matrix and labeling.\n    '
    labeling = _labeling_from_architecture(architecture, vertices)
    labeling = [LABEL2ID[t] for t in labeling]
    matrix = _adjancency_matrix_from_architecture(architecture, vertices)
    in_edges = np.sum(matrix, axis=0).tolist()
    out_edges = np.sum(matrix, axis=1).tolist()
    assert len(in_edges) == len(out_edges) == len(labeling)
    hashes = list(zip(out_edges, in_edges, labeling))
    hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
    for _ in range(vertices):
        new_hashes = []
        for v in range(vertices):
            in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
            out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
            new_hashes.append(hashlib.md5((''.join(sorted(in_neighbors)) + '|' + ''.join(sorted(out_neighbors)) + '|' + hashes[v]).encode('utf-8')).hexdigest())
        hashes = new_hashes
    fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()
    return fingerprint