import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.constants import golden as phi
from scipy.spatial import cKDTree
TOL = 1e-12
NS = range(1, 13)
NAMES = ['I', 'O', 'T'] + ['C%d' % n for n in NS] + ['D%d' % n for n in NS]
SIZES = [60, 24, 12] + list(NS) + [2 * n for n in NS]

def _calculate_rmsd(P, Q):
    if False:
        while True:
            i = 10
    'Calculates the root-mean-square distance between the points of P and Q.\n    The distance is taken as the minimum over all possible matchings. It is\n    zero if P and Q are identical and non-zero if not.\n    '
    distance_matrix = cdist(P, Q, metric='sqeuclidean')
    matching = linear_sum_assignment(distance_matrix)
    return np.sqrt(distance_matrix[matching].sum())

def _generate_pyramid(n, axis):
    if False:
        print('Hello World!')
    thetas = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    P = np.vstack([np.zeros(n), np.cos(thetas), np.sin(thetas)]).T
    P = np.concatenate((P, [[1, 0, 0]]))
    return np.roll(P, axis, axis=1)

def _generate_prism(n, axis):
    if False:
        print('Hello World!')
    thetas = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    bottom = np.vstack([-np.ones(n), np.cos(thetas), np.sin(thetas)]).T
    top = np.vstack([+np.ones(n), np.cos(thetas), np.sin(thetas)]).T
    P = np.concatenate((bottom, top))
    return np.roll(P, axis, axis=1)

def _generate_icosahedron():
    if False:
        print('Hello World!')
    x = np.array([[0, -1, -phi], [0, -1, +phi], [0, +1, -phi], [0, +1, +phi]])
    return np.concatenate([np.roll(x, i, axis=1) for i in range(3)])

def _generate_octahedron():
    if False:
        i = 10
        return i + 15
    return np.array([[-1, 0, 0], [+1, 0, 0], [0, -1, 0], [0, +1, 0], [0, 0, -1], [0, 0, +1]])

def _generate_tetrahedron():
    if False:
        return 10
    return np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])

@pytest.mark.parametrize('name', [-1, None, True, np.array(['C3'])])
def test_group_type(name):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError, match='must be a string'):
        Rotation.create_group(name)

@pytest.mark.parametrize('name', ['Q', ' ', 'CA', 'C ', 'DA', 'D ', 'I2', ''])
def test_group_name(name):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError, match="must be one of 'I', 'O', 'T', 'Dn', 'Cn'"):
        Rotation.create_group(name)

@pytest.mark.parametrize('name', ['C0', 'D0'])
def test_group_order_positive(name):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError, match='Group order must be positive'):
        Rotation.create_group(name)

@pytest.mark.parametrize('axis', ['A', 'b', 0, 1, 2, 4, False, None])
def test_axis_valid(axis):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError, match='`axis` must be one of'):
        Rotation.create_group('C1', axis)

def test_icosahedral():
    if False:
        i = 10
        return i + 15
    'The icosahedral group fixes the rotations of an icosahedron. Here we\n    test that the icosahedron is invariant after application of the elements\n    of the rotation group.'
    P = _generate_icosahedron()
    for g in Rotation.create_group('I'):
        g = Rotation.from_quat(g.as_quat())
        assert _calculate_rmsd(P, g.apply(P)) < TOL

def test_octahedral():
    if False:
        i = 10
        return i + 15
    'Test that the octahedral group correctly fixes the rotations of an\n    octahedron.'
    P = _generate_octahedron()
    for g in Rotation.create_group('O'):
        assert _calculate_rmsd(P, g.apply(P)) < TOL

def test_tetrahedral():
    if False:
        print('Hello World!')
    'Test that the tetrahedral group correctly fixes the rotations of a\n    tetrahedron.'
    P = _generate_tetrahedron()
    for g in Rotation.create_group('T'):
        assert _calculate_rmsd(P, g.apply(P)) < TOL

@pytest.mark.parametrize('n', NS)
@pytest.mark.parametrize('axis', 'XYZ')
def test_dicyclic(n, axis):
    if False:
        for i in range(10):
            print('nop')
    'Test that the dicyclic group correctly fixes the rotations of a\n    prism.'
    P = _generate_prism(n, axis='XYZ'.index(axis))
    for g in Rotation.create_group('D%d' % n, axis=axis):
        assert _calculate_rmsd(P, g.apply(P)) < TOL

@pytest.mark.parametrize('n', NS)
@pytest.mark.parametrize('axis', 'XYZ')
def test_cyclic(n, axis):
    if False:
        i = 10
        return i + 15
    'Test that the cyclic group correctly fixes the rotations of a\n    pyramid.'
    P = _generate_pyramid(n, axis='XYZ'.index(axis))
    for g in Rotation.create_group('C%d' % n, axis=axis):
        assert _calculate_rmsd(P, g.apply(P)) < TOL

@pytest.mark.parametrize('name, size', zip(NAMES, SIZES))
def test_group_sizes(name, size):
    if False:
        for i in range(10):
            print('nop')
    assert len(Rotation.create_group(name)) == size

@pytest.mark.parametrize('name, size', zip(NAMES, SIZES))
def test_group_no_duplicates(name, size):
    if False:
        return 10
    g = Rotation.create_group(name)
    kdtree = cKDTree(g.as_quat())
    assert len(kdtree.query_pairs(0.001)) == 0

@pytest.mark.parametrize('name, size', zip(NAMES, SIZES))
def test_group_symmetry(name, size):
    if False:
        print('Hello World!')
    g = Rotation.create_group(name)
    q = np.concatenate((-g.as_quat(), g.as_quat()))
    distance = np.sort(cdist(q, q))
    deltas = np.max(distance, axis=0) - np.min(distance, axis=0)
    assert (deltas < TOL).all()

@pytest.mark.parametrize('name', NAMES)
def test_reduction(name):
    if False:
        for i in range(10):
            print('nop')
    'Test that the elements of the rotation group are correctly\n    mapped onto the identity rotation.'
    g = Rotation.create_group(name)
    f = g.reduce(g)
    assert_array_almost_equal(f.magnitude(), np.zeros(len(g)))

@pytest.mark.parametrize('name', NAMES)
def test_single_reduction(name):
    if False:
        for i in range(10):
            print('nop')
    g = Rotation.create_group(name)
    f = g[-1].reduce(g)
    assert_array_almost_equal(f.magnitude(), 0)
    assert f.as_quat().shape == (4,)