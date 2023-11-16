"""Trying out tree structure for nested logit

sum is standing for likelihood calculations

should collect and aggregate likelihood contributions bottom up

"""
from statsmodels.compat.python import lrange
import numpy as np
tree = [[0, 1], [[2, 3], [4, 5, 6]], [7]]
xb = 2 * np.arange(8)
testxb = 1

def branch(tree):
    if False:
        print('Hello World!')
    'walking a tree bottom-up\n    '
    if not isinstance(tree[0], int):
        branchsum = 0
        for b in tree:
            branchsum += branch(b)
    else:
        print(tree)
        print('final branch with', tree, sum(tree))
        if testxb:
            return sum(xb[tree])
        else:
            return sum(tree)
    print('working on branch', tree, branchsum)
    return branchsum
print(branch(tree))
testxb = 0

def branch2(tree):
    if False:
        for i in range(10):
            print('nop')
    'walking a tree bottom-up based on dictionary\n    '
    if isinstance(tree, tuple):
        (name, subtree) = tree
        print(name, data2[name])
        print('subtree', subtree)
        if testxb:
            branchsum = data2[name]
        else:
            branchsum = name
        for b in subtree:
            branchsum = branchsum + branch2(b)
    else:
        leavessum = sum((data2[bi] for bi in tree))
        print('final branch with', tree, ''.join(tree), leavessum)
        if testxb:
            return leavessum
        else:
            return ''.join(tree)
    print('working on branch', tree, branchsum)
    return branchsum
tree = [[0, 1], [[2, 3], [4, 5, 6]], [7]]
tree2 = ('top', [('B1', ['a', 'b']), ('B2', [('B21', ['c', 'd']), ('B22', ['e', 'f', 'g'])]), ('B3', ['h'])])
data2 = dict([i for i in zip('abcdefgh', lrange(8))])
data2.update({'top': 1000, 'B1': 100, 'B2': 200, 'B21': 21, 'B22': 22, 'B3': 300})
print('\n tree with dictionary data')
print(branch2(tree2))
paramsind = {'B1': [], 'a': ['consta', 'p'], 'b': ['constb', 'p'], 'B2': ['const2', 'x2'], 'B21': [], 'c': ['consta', 'p', 'time'], 'd': ['consta', 'p', 'time'], 'B22': ['x22'], 'e': ['conste', 'p', 'hince'], 'f': ['constt', 'p', 'hincf'], 'g': ['p', 'hincg'], 'B3': [], 'h': ['consth', 'p', 'h'], 'top': []}
paramsnames = sorted(set([i for j in paramsind.values() for i in j]))
paramsidx = dict(((name, idx) for (idx, name) in enumerate(paramsnames)))
inddict = dict(((k, [paramsidx[j] for j in v]) for (k, v) in paramsind.items()))
"\n>>> paramsnames\n['const2', 'consta', 'constb', 'conste', 'consth', 'constt', 'h', 'hince',\n 'hincf', 'hincg', 'p', 'time', 'x2', 'x22']\n>>> parmasidx\n{'conste': 3, 'consta': 1, 'constb': 2, 'h': 6, 'time': 11, 'consth': 4,\n 'p': 10, 'constt': 5, 'const2': 0, 'x2': 12, 'x22': 13, 'hince': 7,\n 'hincg': 9, 'hincf': 8}\n>>> inddict\n{'a': [1, 10], 'c': [1, 10, 11], 'b': [2, 10], 'e': [3, 10, 7],\n 'd': [1, 10, 11], 'g': [10, 9], 'f': [5, 10, 8], 'h': [4, 10, 6],\n 'top': [], 'B22': [13], 'B21': [], 'B1': [], 'B2': [0, 12], 'B3': []}\n>>> paramsind\n{'a': ['consta', 'p'], 'c': ['consta', 'p', 'time'], 'b': ['constb', 'p'],\n 'e': ['conste', 'p', 'hince'], 'd': ['consta', 'p', 'time'],\n 'g': ['p', 'hincg'], 'f': ['constt', 'p', 'hincf'], 'h': ['consth', 'p', 'h'],\n 'top': [], 'B22': ['x22'], 'B21': [], 'B1': [], 'B2': ['const2', 'x2'],\n 'B3': []}\n"