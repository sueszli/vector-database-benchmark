"""Tests for the vector code in Bio.PDB."""
import unittest
try:
    import numpy
    from numpy.random import random
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Install NumPy if you want to use Bio.PDB.') from None
from Bio.PDB.vectors import Vector
from Bio.PDB import rotmat, refmat, calc_angle, calc_dihedral, rotaxis, m2rotaxis
from Bio.PDB.vectors import get_spherical_coordinates, coord_space, homog_trans_mtx
from Bio.PDB.vectors import multi_coord_space

class VectorTests(unittest.TestCase):
    """Tests for the Vector class."""

    def test_division(self):
        if False:
            print('Hello World!')
        'Confirm division works.'
        v = Vector(1, 1, 1) / 2
        self.assertEqual(repr(v), '<Vector 0.50, 0.50, 0.50>')

    def test_Vector(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Vector object.'
        v1 = Vector(0, 0, 1)
        v2 = Vector(0, 0, 0)
        v3 = Vector(0, 1, 0)
        v4 = Vector(1, 1, 0)
        self.assertEqual(calc_angle(v1, v2, v3), 1.5707963267948966)
        self.assertEqual(calc_dihedral(v1, v2, v3, v4), 1.5707963267948966)
        self.assertTrue(numpy.array_equal((v1 - v2).get_array(), numpy.array([0.0, 0.0, 1.0])))
        self.assertTrue(numpy.array_equal((v1 - 1).get_array(), numpy.array([-1.0, -1.0, 0.0])))
        self.assertTrue(numpy.array_equal((v1 - (1, 2, 3)).get_array(), numpy.array([-1.0, -2.0, -2.0])))
        self.assertTrue(numpy.array_equal((v1 + v2).get_array(), numpy.array([0.0, 0.0, 1.0])))
        self.assertTrue(numpy.array_equal((v1 + 3).get_array(), numpy.array([3.0, 3.0, 4.0])))
        self.assertTrue(numpy.array_equal((v1 + (1, 2, 3)).get_array(), numpy.array([1.0, 2.0, 4.0])))
        self.assertTrue(numpy.array_equal(v1.get_array() / 2, numpy.array([0, 0, 0.5])))
        self.assertTrue(numpy.array_equal(v1.get_array() / 2, numpy.array([0, 0, 0.5])))
        self.assertEqual(v1 * v2, 0.0)
        self.assertTrue(numpy.array_equal((v1 ** v2).get_array(), numpy.array([0.0, -0.0, 0.0])))
        self.assertTrue(numpy.array_equal((v1 ** 2).get_array(), numpy.array([0.0, 0.0, 2.0])))
        self.assertTrue(numpy.array_equal((v1 ** (1, 2, 3)).get_array(), numpy.array([0.0, 0.0, 3.0])))
        self.assertEqual(v1.norm(), 1.0)
        self.assertEqual(v1.normsq(), 1.0)
        v1[2] = 10
        self.assertEqual(v1.__getitem__(2), 10)

    def test_normalization(self):
        if False:
            i = 10
            return i + 15
        'Test Vector normalization.'
        v1 = Vector([2, 0, 0])
        self.assertTrue(numpy.array_equal(v1.normalized().get_array(), numpy.array([1, 0, 0])))
        self.assertTrue(numpy.array_equal(v1.get_array(), numpy.array([2, 0, 0])))
        v1.normalize()
        self.assertTrue(numpy.array_equal(v1.get_array(), numpy.array([1, 0, 0])))

    def test_refmat(self):
        if False:
            return 10
        'Test refmat can mirror one matrix to another.'
        v1 = Vector(0, 0, 1)
        v2 = Vector(0, 1, 0)
        ref = refmat(v1, v2)
        self.assertTrue(numpy.allclose(ref[0], [1.0, 0.0, 0.0]))
        self.assertTrue(numpy.allclose(ref[1], [0.0, 0.0, 1.0]))
        self.assertTrue(numpy.allclose(ref[2], [0.0, 1.0, 0.0]))
        self.assertTrue(numpy.allclose(v1.left_multiply(ref).get_array(), [0.0, 1.0, 0.0]))

    def test_rotmat_90(self):
        if False:
            i = 10
            return i + 15
        'Test regular 90 deg rotation.'
        v1 = Vector(0, 0, 1)
        v2 = Vector(0, 1, 0)
        rot = rotmat(v1, v2)
        self.assertTrue(numpy.allclose(rot[0], numpy.array([1.0, 0.0, 0.0])))
        self.assertTrue(numpy.allclose(rot[1], numpy.array([0.0, 0.0, 1.0])))
        self.assertTrue(numpy.allclose(rot[2], numpy.array([0.0, -1.0, 0.0])))
        self.assertTrue(numpy.allclose(v1.left_multiply(rot).get_array(), [0.0, 1.0, 0.0]))
        self.assertTrue(numpy.allclose(v1.right_multiply(numpy.transpose(rot)).get_array(), [0.0, 1.0, 0.0]))

    def test_rotmat_180(self):
        if False:
            for i in range(10):
                print('nop')
        'Test rotmat when the rotation is 180 deg (singularity).'
        v1 = Vector([1.0, 0.8, 0])
        v2 = Vector([-1.0, -0.8, 0])
        rot = rotmat(v1, v2)
        v3 = v1.left_multiply(rot)
        self.assertTrue(numpy.allclose(v2.get_array(), v3.get_array()))

    def test_rotmat_0(self):
        if False:
            print('Hello World!')
        'Test rotmat when the rotation is 0 deg (singularity).'
        v1 = Vector([1.0, 0.8, 0])
        v2 = Vector([1.0, 0.8, 0])
        rot = rotmat(v1, v2)
        v3 = v1.left_multiply(rot)
        self.assertTrue(numpy.allclose(v1.get_array(), v3.get_array()))

    def test_m2rotaxis_90(self):
        if False:
            i = 10
            return i + 15
        'Test 90 deg rotation.'
        v1 = Vector(0, 0, 1)
        v2 = Vector(0, 1, 0)
        rot = rotmat(v1, v2)
        (angle, axis) = m2rotaxis(rot)
        self.assertTrue(numpy.allclose(axis.get_array(), [-1.0, 0.0, 0.0]))
        self.assertLess(abs(angle - numpy.pi / 2), 1e-05)

    def test_m2rotaxis_180(self):
        if False:
            for i in range(10):
                print('nop')
        'Test 180 deg rotation.'
        v1 = Vector([1.0, 0.8, 0])
        v2 = Vector([-1.0, -0.8, 0])
        rot = rotmat(v1, v2)
        (angle, axis) = m2rotaxis(rot)
        self.assertLess(abs(axis * v1), 1e-05)
        self.assertLess(abs(angle - numpy.pi), 1e-05)

    def test_m2rotaxis_0(self):
        if False:
            return 10
        'Test 0 deg rotation. Axis must be [1, 0, 0] as per Vector docs.'
        v1 = Vector([1.0, 0.8, 0])
        v2 = Vector([1.0, 0.8, 0])
        rot = rotmat(v1, v2)
        (angle, axis) = m2rotaxis(rot)
        self.assertTrue(numpy.allclose(axis.get_array(), [1, 0, 0]))
        self.assertLess(abs(angle), 1e-05)

    def test_Vector_angles(self):
        if False:
            i = 10
            return i + 15
        'Test Vector angles.'
        angle = random() * numpy.pi
        axis = Vector(random(3) - random(3))
        axis.normalize()
        m = rotaxis(angle, axis)
        (cangle, caxis) = m2rotaxis(m)
        self.assertAlmostEqual(angle, cangle, places=3)
        self.assertTrue(numpy.allclose(list(map(int, (axis - caxis).get_array())), [0, 0, 0]), f'Want {axis.get_array()!r} and {caxis.get_array()!r} to be almost equal')

    def test_get_spherical_coordinates(self):
        if False:
            print('Hello World!')
        'Test spherical coordinates.'
        srt22 = numpy.sqrt(2.0) / 2
        r45 = numpy.radians(45)
        r135 = numpy.radians(135)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    sc = get_spherical_coordinates([0.5 if i else -0.5, 0.5 if j else -0.5, (1 if k else -1) * srt22])
                    self.assertEqual(1.0, sc[0])
                    self.assertEqual((1 if j else -1) * (r45 if i else r135), sc[1])
                    self.assertEqual(r45 if k else r135, sc[2])

    def test_coord_space(self):
        if False:
            for i in range(10):
                print('nop')
        'Confirm can generate coordinate space transform for 3 points.'
        point_set = (numpy.array([[2.0], [0.0], [2.0], [1.0]]), numpy.array([[0.0], [0.0], [0.0], [1.0]]), numpy.array([[0.0], [0.0], [2.0], [1.0]]))
        homog_id = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        mtxs = coord_space(point_set[0], point_set[1], point_set[2], True)
        for i in range(2):
            self.assertTrue(numpy.array_equal(mtxs[i], homog_id))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    tm = homog_trans_mtx(3 if i else -3, 3 if j else -3, 3 if k else -3)
                    ps2 = [1, 2, 3]
                    for i in range(3):
                        ps2[i] = tm.dot(point_set[i])
                    mtxs = coord_space(ps2[0], ps2[1], ps2[2], True)
                    rslt = [1, 2, 3]
                    for i in range(3):
                        rslt[i] = mtxs[0].dot(ps2[i])
                    self.assertTrue(numpy.array_equal(rslt, point_set))
                    for i in range(3):
                        rslt[i] = mtxs[1].dot(rslt[i])
                    self.assertTrue(numpy.array_equal(rslt, ps2))

    def test_multi_coord_space(self):
        if False:
            print('Hello World!')
        'Confirm multi_coord_space computes forward, reverse transforms.'
        point_set = numpy.array([[[2.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 2.0, 1.0]]])
        homog_id = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        mtxs = multi_coord_space(point_set, 1, True)
        for i in range(2):
            self.assertTrue(numpy.array_equal(mtxs[i][0], homog_id))
        test_set = numpy.empty([8, 3, 4], dtype=numpy.float64)
        m = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    tm = homog_trans_mtx(3 if i else -3, 3 if j else -3, 3 if k else -3)
                    for i in range(3):
                        test_set[m, i] = tm.dot(point_set[0][i])
                    m += 1
        mtxs = multi_coord_space(test_set, 8, True)
        for m in range(8):
            rslt = [1, 2, 3]
            for i in range(3):
                rslt[i] = mtxs[0][m].dot(test_set[m][i])
            self.assertTrue(numpy.array_equal(rslt, point_set[0]))
            for i in range(3):
                rslt[i] = mtxs[1][m].dot(rslt[i])
            self.assertTrue(numpy.array_equal(rslt, test_set[m]))
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)