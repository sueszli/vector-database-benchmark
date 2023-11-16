"""Tests for PTM quantum channel representation class."""
import copy
import unittest
import numpy as np
from numpy.testing import assert_allclose
from qiskit import QiskitError
from qiskit.quantum_info.states import DensityMatrix
from qiskit.quantum_info.operators.channel import PTM
from .channel_test_case import ChannelTestCase

class TestPTM(ChannelTestCase):
    """Tests for PTM channel representation."""

    def test_init(self):
        if False:
            return 10
        'Test initialization'
        mat4 = np.eye(4) / 2.0
        chan = PTM(mat4)
        assert_allclose(chan.data, mat4)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)
        mat16 = np.eye(16) / 4
        chan = PTM(mat16)
        assert_allclose(chan.data, mat16)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(chan.num_qubits, 2)
        self.assertRaises(QiskitError, PTM, mat16, input_dims=2, output_dims=4)
        self.assertRaises(QiskitError, PTM, np.eye(6) / 2, input_dims=3, output_dims=2)

    def test_circuit_init(self):
        if False:
            return 10
        'Test initialization from a circuit.'
        (circuit, target) = self.simple_circuit_no_measure()
        op = PTM(circuit)
        target = PTM(target)
        self.assertEqual(op, target)

    def test_circuit_init_except(self):
        if False:
            return 10
        'Test initialization from circuit with measure raises exception.'
        circuit = self.simple_circuit_with_measure()
        self.assertRaises(QiskitError, PTM, circuit)

    def test_equal(self):
        if False:
            print('Hello World!')
        'Test __eq__ method'
        mat = self.rand_matrix(4, 4, real=True)
        self.assertEqual(PTM(mat), PTM(mat))

    def test_copy(self):
        if False:
            i = 10
            return i + 15
        'Test copy method'
        mat = np.eye(4)
        with self.subTest('Deep copy'):
            orig = PTM(mat)
            cpy = orig.copy()
            cpy._data[0, 0] = 0.0
            self.assertFalse(cpy == orig)
        with self.subTest('Shallow copy'):
            orig = PTM(mat)
            clone = copy.copy(orig)
            clone._data[0, 0] = 0.0
            self.assertTrue(clone == orig)

    def test_clone(self):
        if False:
            i = 10
            return i + 15
        'Test clone method'
        mat = np.eye(4)
        orig = PTM(mat)
        clone = copy.copy(orig)
        clone._data[0, 0] = 0.0
        self.assertTrue(clone == orig)

    def test_is_cptp(self):
        if False:
            for i in range(10):
                print('nop')
        'Test is_cptp method.'
        self.assertTrue(PTM(self.depol_ptm(0.25)).is_cptp())
        self.assertFalse(PTM(1.25 * self.ptmI - 0.25 * self.depol_ptm(1)).is_cptp())

    def test_compose_except(self):
        if False:
            return 10
        'Test compose different dimension exception'
        self.assertRaises(QiskitError, PTM(np.eye(4)).compose, PTM(np.eye(16)))
        self.assertRaises(QiskitError, PTM(np.eye(4)).compose, 2)

    def test_compose(self):
        if False:
            i = 10
            return i + 15
        'Test compose method.'
        rho = DensityMatrix(self.rand_rho(2))
        chan1 = PTM(self.ptmX)
        chan2 = PTM(self.ptmY)
        chan = chan1.compose(chan2)
        rho_targ = rho.evolve(PTM(self.ptmZ))
        self.assertEqual(rho.evolve(chan), rho_targ)
        chan1 = PTM(self.depol_ptm(0.5))
        chan = chan1.compose(chan1)
        rho_targ = rho.evolve(PTM(self.depol_ptm(0.75)))
        self.assertEqual(rho.evolve(chan), rho_targ)
        ptm1 = self.rand_matrix(4, 4, real=True)
        ptm2 = self.rand_matrix(4, 4, real=True)
        chan1 = PTM(ptm1, input_dims=2, output_dims=2)
        chan2 = PTM(ptm2, input_dims=2, output_dims=2)
        rho_targ = rho.evolve(chan1).evolve(chan2)
        chan = chan1.compose(chan2)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho.evolve(chan), rho_targ)
        chan = chan1 & chan2
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho.evolve(chan), rho_targ)

    def test_dot(self):
        if False:
            while True:
                i = 10
        'Test dot method.'
        rho = DensityMatrix(self.rand_rho(2))
        chan1 = PTM(self.ptmX)
        chan2 = PTM(self.ptmY)
        rho_targ = rho.evolve(PTM(self.ptmZ))
        self.assertEqual(rho.evolve(chan2.dot(chan1)), rho_targ)
        self.assertEqual(rho.evolve(chan2 @ chan1), rho_targ)
        ptm1 = self.rand_matrix(4, 4, real=True)
        ptm2 = self.rand_matrix(4, 4, real=True)
        chan1 = PTM(ptm1, input_dims=2, output_dims=2)
        chan2 = PTM(ptm2, input_dims=2, output_dims=2)
        rho_targ = rho.evolve(chan1).evolve(chan2)
        self.assertEqual(rho.evolve(chan2.dot(chan1)), rho_targ)
        self.assertEqual(rho.evolve(chan2 @ chan1), rho_targ)

    def test_compose_front(self):
        if False:
            return 10
        'Test deprecated front compose method.'
        rho = DensityMatrix(self.rand_rho(2))
        chan1 = PTM(self.ptmX)
        chan2 = PTM(self.ptmY)
        chan = chan2.compose(chan1, front=True)
        rho_targ = rho.evolve(PTM(self.ptmZ))
        self.assertEqual(rho.evolve(chan), rho_targ)
        ptm1 = self.rand_matrix(4, 4, real=True)
        ptm2 = self.rand_matrix(4, 4, real=True)
        chan1 = PTM(ptm1, input_dims=2, output_dims=2)
        chan2 = PTM(ptm2, input_dims=2, output_dims=2)
        rho_targ = rho.evolve(chan1).evolve(chan2)
        chan = chan2.compose(chan1, front=True)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho.evolve(chan), rho_targ)

    def test_expand(self):
        if False:
            i = 10
            return i + 15
        'Test expand method.'
        (rho0, rho1) = (np.diag([1, 0]), np.diag([0, 1]))
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = PTM(self.ptmI)
        chan2 = PTM(self.ptmX)
        chan = chan1.expand(chan2)
        rho_targ = DensityMatrix(np.kron(rho1, rho0))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = chan2.expand(chan1)
        rho_targ = DensityMatrix(np.kron(rho0, rho1))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan_dep = PTM(self.depol_ptm(1))
        chan = chan_dep.expand(chan_dep)
        rho_targ = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_tensor(self):
        if False:
            print('Hello World!')
        'Test tensor method.'
        (rho0, rho1) = (np.diag([1, 0]), np.diag([0, 1]))
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = PTM(self.ptmI)
        chan2 = PTM(self.ptmX)
        chan = chan2.tensor(chan1)
        rho_targ = DensityMatrix(np.kron(rho1, rho0))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = chan1.tensor(chan2)
        rho_targ = DensityMatrix(np.kron(rho0, rho1))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan_dep = PTM(self.depol_ptm(1))
        chan = chan_dep.tensor(chan_dep)
        rho_targ = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_power(self):
        if False:
            return 10
        'Test power method.'
        p_id = 0.9
        depol = PTM(self.depol_ptm(1 - p_id))
        p_id3 = p_id ** 3
        chan3 = depol.power(3)
        targ3 = PTM(self.depol_ptm(1 - p_id3))
        self.assertEqual(chan3, targ3)

    def test_add(self):
        if False:
            while True:
                i = 10
        'Test add method.'
        mat1 = 0.5 * self.ptmI
        mat2 = 0.5 * self.depol_ptm(1)
        chan1 = PTM(mat1)
        chan2 = PTM(mat2)
        targ = PTM(mat1 + mat2)
        self.assertEqual(chan1._add(chan2), targ)
        self.assertEqual(chan1 + chan2, targ)
        targ = PTM(mat1 - mat2)
        self.assertEqual(chan1 - chan2, targ)

    def test_add_qargs(self):
        if False:
            return 10
        'Test add method with qargs.'
        mat = self.rand_matrix(8 ** 2, 8 ** 2)
        mat0 = self.rand_matrix(4, 4)
        mat1 = self.rand_matrix(4, 4)
        op = PTM(mat)
        op0 = PTM(mat0)
        op1 = PTM(mat1)
        op01 = op1.tensor(op0)
        eye = PTM(self.ptmI)
        with self.subTest(msg='qargs=[0]'):
            value = op + op0([0])
            target = op + eye.tensor(eye).tensor(op0)
            self.assertEqual(value, target)
        with self.subTest(msg='qargs=[1]'):
            value = op + op0([1])
            target = op + eye.tensor(op0).tensor(eye)
            self.assertEqual(value, target)
        with self.subTest(msg='qargs=[2]'):
            value = op + op0([2])
            target = op + op0.tensor(eye).tensor(eye)
            self.assertEqual(value, target)
        with self.subTest(msg='qargs=[0, 1]'):
            value = op + op01([0, 1])
            target = op + eye.tensor(op1).tensor(op0)
            self.assertEqual(value, target)
        with self.subTest(msg='qargs=[1, 0]'):
            value = op + op01([1, 0])
            target = op + eye.tensor(op0).tensor(op1)
            self.assertEqual(value, target)
        with self.subTest(msg='qargs=[0, 2]'):
            value = op + op01([0, 2])
            target = op + op1.tensor(eye).tensor(op0)
            self.assertEqual(value, target)
        with self.subTest(msg='qargs=[2, 0]'):
            value = op + op01([2, 0])
            target = op + op0.tensor(eye).tensor(op1)
            self.assertEqual(value, target)

    def test_sub_qargs(self):
        if False:
            for i in range(10):
                print('nop')
        'Test subtract method with qargs.'
        mat = self.rand_matrix(8 ** 2, 8 ** 2)
        mat0 = self.rand_matrix(4, 4)
        mat1 = self.rand_matrix(4, 4)
        op = PTM(mat)
        op0 = PTM(mat0)
        op1 = PTM(mat1)
        op01 = op1.tensor(op0)
        eye = PTM(self.ptmI)
        with self.subTest(msg='qargs=[0]'):
            value = op - op0([0])
            target = op - eye.tensor(eye).tensor(op0)
            self.assertEqual(value, target)
        with self.subTest(msg='qargs=[1]'):
            value = op - op0([1])
            target = op - eye.tensor(op0).tensor(eye)
            self.assertEqual(value, target)
        with self.subTest(msg='qargs=[2]'):
            value = op - op0([2])
            target = op - op0.tensor(eye).tensor(eye)
            self.assertEqual(value, target)
        with self.subTest(msg='qargs=[0, 1]'):
            value = op - op01([0, 1])
            target = op - eye.tensor(op1).tensor(op0)
            self.assertEqual(value, target)
        with self.subTest(msg='qargs=[1, 0]'):
            value = op - op01([1, 0])
            target = op - eye.tensor(op0).tensor(op1)
            self.assertEqual(value, target)
        with self.subTest(msg='qargs=[0, 2]'):
            value = op - op01([0, 2])
            target = op - op1.tensor(eye).tensor(op0)
            self.assertEqual(value, target)
        with self.subTest(msg='qargs=[2, 0]'):
            value = op - op01([2, 0])
            target = op - op0.tensor(eye).tensor(op1)
            self.assertEqual(value, target)

    def test_add_except(self):
        if False:
            while True:
                i = 10
        'Test add method raises exceptions.'
        chan1 = PTM(self.ptmI)
        chan2 = PTM(np.eye(16))
        self.assertRaises(QiskitError, chan1._add, chan2)
        self.assertRaises(QiskitError, chan1._add, 5)

    def test_multiply(self):
        if False:
            while True:
                i = 10
        'Test multiply method.'
        chan = PTM(self.ptmI)
        val = 0.5
        targ = PTM(val * self.ptmI)
        self.assertEqual(chan._multiply(val), targ)
        self.assertEqual(val * chan, targ)
        targ = PTM(self.ptmI * val)
        self.assertEqual(chan * val, targ)

    def test_multiply_except(self):
        if False:
            print('Hello World!')
        'Test multiply method raises exceptions.'
        chan = PTM(self.ptmI)
        self.assertRaises(QiskitError, chan._multiply, 's')
        self.assertRaises(QiskitError, chan.__rmul__, 's')
        self.assertRaises(QiskitError, chan._multiply, chan)
        self.assertRaises(QiskitError, chan.__rmul__, chan)

    def test_negate(self):
        if False:
            return 10
        'Test negate method'
        chan = PTM(self.ptmI)
        targ = PTM(-self.ptmI)
        self.assertEqual(-chan, targ)
if __name__ == '__main__':
    unittest.main()