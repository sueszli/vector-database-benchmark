"""Tests for Stinespring quantum channel representation class."""
import copy
import unittest
import numpy as np
from numpy.testing import assert_allclose
from qiskit import QiskitError
from qiskit.quantum_info.states import DensityMatrix
from qiskit.quantum_info import Stinespring
from .channel_test_case import ChannelTestCase

class TestStinespring(ChannelTestCase):
    """Tests for Stinespring channel representation."""

    def test_init(self):
        if False:
            while True:
                i = 10
        'Test initialization'
        chan = Stinespring(self.UI)
        assert_allclose(chan.data, self.UI)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)
        chan = Stinespring(self.depol_stine(0.5))
        assert_allclose(chan.data, self.depol_stine(0.5))
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)
        (stine_l, stine_r) = (self.rand_matrix(4, 2), self.rand_matrix(4, 2))
        chan = Stinespring((stine_l, stine_r))
        assert_allclose(chan.data, (stine_l, stine_r))
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)
        chan = Stinespring((stine_l, stine_l))
        assert_allclose(chan.data, stine_l)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)
        self.assertRaises(QiskitError, Stinespring, stine_l, input_dims=4, output_dims=4)

    def test_circuit_init(self):
        if False:
            for i in range(10):
                print('nop')
        'Test initialization from a circuit.'
        (circuit, target) = self.simple_circuit_no_measure()
        op = Stinespring(circuit)
        target = Stinespring(target)
        self.assertEqual(op, target)

    def test_circuit_init_except(self):
        if False:
            return 10
        'Test initialization from circuit with measure raises exception.'
        circuit = self.simple_circuit_with_measure()
        self.assertRaises(QiskitError, Stinespring, circuit)

    def test_equal(self):
        if False:
            print('Hello World!')
        'Test __eq__ method'
        stine = tuple((self.rand_matrix(4, 2) for _ in range(2)))
        self.assertEqual(Stinespring(stine), Stinespring(stine))

    def test_copy(self):
        if False:
            for i in range(10):
                print('nop')
        'Test copy method'
        mat = np.eye(4)
        with self.subTest('Deep copy'):
            orig = Stinespring(mat)
            cpy = orig.copy()
            cpy._data[0][0, 0] = 0.0
            self.assertFalse(cpy == orig)
        with self.subTest('Shallow copy'):
            orig = Stinespring(mat)
            clone = copy.copy(orig)
            clone._data[0][0, 0] = 0.0
            self.assertTrue(clone == orig)

    def test_clone(self):
        if False:
            i = 10
            return i + 15
        'Test clone method'
        mat = np.eye(4)
        orig = Stinespring(mat)
        clone = copy.copy(orig)
        clone._data[0][0, 0] = 0.0
        self.assertTrue(clone == orig)

    def test_is_cptp(self):
        if False:
            while True:
                i = 10
        'Test is_cptp method.'
        self.assertTrue(Stinespring(self.depol_stine(0.5)).is_cptp())
        self.assertTrue(Stinespring(self.UX).is_cptp())
        (stine_l, stine_r) = (self.rand_matrix(4, 2), self.rand_matrix(4, 2))
        self.assertFalse(Stinespring((stine_l, stine_r)).is_cptp())
        self.assertFalse(Stinespring(self.UI + self.UX).is_cptp())

    def test_conjugate(self):
        if False:
            print('Hello World!')
        'Test conjugate method.'
        (stine_l, stine_r) = (self.rand_matrix(16, 2), self.rand_matrix(16, 2))
        targ = Stinespring(stine_l.conj(), output_dims=4)
        chan1 = Stinespring(stine_l, output_dims=4)
        chan = chan1.conjugate()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (2, 4))
        targ = Stinespring((stine_l.conj(), stine_r.conj()), output_dims=4)
        chan1 = Stinespring((stine_l, stine_r), output_dims=4)
        chan = chan1.conjugate()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (2, 4))

    def test_transpose(self):
        if False:
            for i in range(10):
                print('nop')
        'Test transpose method.'
        (stine_l, stine_r) = (self.rand_matrix(4, 2), self.rand_matrix(4, 2))
        targ = Stinespring(stine_l.T, 4, 2)
        chan1 = Stinespring(stine_l, 2, 4)
        chan = chan1.transpose()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))
        targ = Stinespring((stine_l.T, stine_r.T), 4, 2)
        chan1 = Stinespring((stine_l, stine_r), 2, 4)
        chan = chan1.transpose()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))

    def test_adjoint(self):
        if False:
            print('Hello World!')
        'Test adjoint method.'
        (stine_l, stine_r) = (self.rand_matrix(4, 2), self.rand_matrix(4, 2))
        targ = Stinespring(stine_l.T.conj(), 4, 2)
        chan1 = Stinespring(stine_l, 2, 4)
        chan = chan1.adjoint()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))
        targ = Stinespring((stine_l.T.conj(), stine_r.T.conj()), 4, 2)
        chan1 = Stinespring((stine_l, stine_r), 2, 4)
        chan = chan1.adjoint()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))

    def test_compose_except(self):
        if False:
            while True:
                i = 10
        'Test compose different dimension exception'
        self.assertRaises(QiskitError, Stinespring(np.eye(2)).compose, Stinespring(np.eye(4)))
        self.assertRaises(QiskitError, Stinespring(np.eye(2)).compose, 2)

    def test_compose(self):
        if False:
            while True:
                i = 10
        'Test compose method.'
        rho_init = DensityMatrix(self.rand_rho(2))
        chan1 = Stinespring(self.UX)
        chan2 = Stinespring(self.UY)
        chan = chan1.compose(chan2)
        rho_targ = rho_init & Stinespring(self.UZ)
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan1 = Stinespring(self.depol_stine(0.5))
        chan = chan1.compose(chan1)
        rho_targ = rho_init & Stinespring(self.depol_stine(0.75))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        (stine1, stine2) = (self.rand_matrix(16, 2), self.rand_matrix(8, 4))
        chan1 = Stinespring(stine1, input_dims=2, output_dims=4)
        chan2 = Stinespring(stine2, input_dims=4, output_dims=2)
        rho_targ = rho_init & chan1 & chan2
        chan = chan1.compose(chan2)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = chan1 & chan2
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_dot(self):
        if False:
            print('Hello World!')
        'Test deprecated front compose method.'
        rho_init = DensityMatrix(self.rand_rho(2))
        chan1 = Stinespring(self.UX)
        chan2 = Stinespring(self.UY)
        rho_targ = rho_init.evolve(Stinespring(self.UZ))
        self.assertEqual(rho_init.evolve(chan1.dot(chan2)), rho_targ)
        self.assertEqual(rho_init.evolve(chan1 @ chan2), rho_targ)
        chan1 = Stinespring(self.depol_stine(0.5))
        rho_targ = rho_init & Stinespring(self.depol_stine(0.75))
        self.assertEqual(rho_init.evolve(chan1.dot(chan1)), rho_targ)
        self.assertEqual(rho_init.evolve(chan1 @ chan1), rho_targ)
        (stine1, stine2) = (self.rand_matrix(16, 2), self.rand_matrix(8, 4))
        chan1 = Stinespring(stine1, input_dims=2, output_dims=4)
        chan2 = Stinespring(stine2, input_dims=4, output_dims=2)
        rho_targ = rho_init & chan1 & chan2
        self.assertEqual(rho_init.evolve(chan2.dot(chan1)), rho_targ)
        self.assertEqual(rho_init.evolve(chan2 @ chan1), rho_targ)

    def test_compose_front(self):
        if False:
            while True:
                i = 10
        'Test deprecated front compose method.'
        rho_init = DensityMatrix(self.rand_rho(2))
        chan1 = Stinespring(self.UX)
        chan2 = Stinespring(self.UY)
        chan = chan1.compose(chan2, front=True)
        rho_targ = rho_init & Stinespring(self.UZ)
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan1 = Stinespring(self.depol_stine(0.5))
        chan = chan1.compose(chan1, front=True)
        rho_targ = rho_init & Stinespring(self.depol_stine(0.75))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        (stine1, stine2) = (self.rand_matrix(16, 2), self.rand_matrix(8, 4))
        chan1 = Stinespring(stine1, input_dims=2, output_dims=4)
        chan2 = Stinespring(stine2, input_dims=4, output_dims=2)
        rho_targ = rho_init & chan1 & chan2
        chan = chan2.compose(chan1, front=True)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_expand(self):
        if False:
            i = 10
            return i + 15
        'Test expand method.'
        (rho0, rho1) = (np.diag([1, 0]), np.diag([0, 1]))
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = Stinespring(self.UI)
        chan2 = Stinespring(self.UX)
        chan = chan1.expand(chan2)
        rho_targ = DensityMatrix(np.kron(rho1, rho0))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = chan2.expand(chan1)
        rho_targ = DensityMatrix(np.kron(rho0, rho1))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan_dep = Stinespring(self.depol_stine(1))
        chan = chan_dep.expand(chan_dep)
        rho_targ = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_tensor(self):
        if False:
            for i in range(10):
                print('nop')
        'Test tensor method.'
        (rho0, rho1) = (np.diag([1, 0]), np.diag([0, 1]))
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = Stinespring(self.UI)
        chan2 = Stinespring(self.UX)
        chan = chan2.tensor(chan1)
        rho_targ = DensityMatrix(np.kron(rho1, rho0))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = chan1.tensor(chan2)
        rho_targ = DensityMatrix(np.kron(rho0, rho1))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan_dep = Stinespring(self.depol_stine(1))
        chan = chan_dep.tensor(chan_dep)
        rho_targ = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_power(self):
        if False:
            for i in range(10):
                print('nop')
        'Test power method.'
        rho_init = DensityMatrix(np.diag([1, 0]))
        p_id = 0.9
        chan1 = Stinespring(self.depol_stine(1 - p_id))
        p_id3 = p_id ** 3
        chan = chan1.power(3)
        rho_targ = rho_init & chan1 & chan1 & chan1
        self.assertEqual(rho_init & chan, rho_targ)
        rho_targ = rho_init & Stinespring(self.depol_stine(1 - p_id3))
        self.assertEqual(rho_init & chan, rho_targ)

    def test_add(self):
        if False:
            i = 10
            return i + 15
        'Test add method.'
        rho_init = DensityMatrix(self.rand_rho(2))
        (stine1, stine2) = (self.rand_matrix(16, 2), self.rand_matrix(16, 2))
        chan1 = Stinespring(stine1, input_dims=2, output_dims=4)
        chan2 = Stinespring(stine2, input_dims=2, output_dims=4)
        rho_targ = (rho_init & chan1) + (rho_init & chan2)
        chan = chan1._add(chan2)
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = chan1 + chan2
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = Stinespring((stine1, stine2))
        rho_targ = 2 * (rho_init & chan)
        chan = chan._add(chan)
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_subtract(self):
        if False:
            while True:
                i = 10
        'Test subtract method.'
        rho_init = DensityMatrix(self.rand_rho(2))
        (stine1, stine2) = (self.rand_matrix(16, 2), self.rand_matrix(16, 2))
        chan1 = Stinespring(stine1, input_dims=2, output_dims=4)
        chan2 = Stinespring(stine2, input_dims=2, output_dims=4)
        rho_targ = (rho_init & chan1) - (rho_init & chan2)
        chan = chan1 - chan2
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = Stinespring((stine1, stine2))
        rho_targ = 0 * (rho_init & chan)
        chan = chan - chan
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_add_qargs(self):
        if False:
            for i in range(10):
                print('nop')
        'Test add method with qargs.'
        rho = DensityMatrix(self.rand_rho(8))
        stine = self.rand_matrix(32, 8)
        stine0 = self.rand_matrix(8, 2)
        op = Stinespring(stine)
        op0 = Stinespring(stine0)
        eye = Stinespring(self.UI)
        with self.subTest(msg='qargs=[0]'):
            value = op + op0([0])
            target = op + eye.tensor(eye).tensor(op0)
            self.assertEqual(rho & value, rho & target)
        with self.subTest(msg='qargs=[1]'):
            value = op + op0([1])
            target = op + eye.tensor(op0).tensor(eye)
            self.assertEqual(rho & value, rho & target)
        with self.subTest(msg='qargs=[2]'):
            value = op + op0([2])
            target = op + op0.tensor(eye).tensor(eye)
            self.assertEqual(rho & value, rho & target)

    def test_sub_qargs(self):
        if False:
            for i in range(10):
                print('nop')
        'Test sub method with qargs.'
        rho = DensityMatrix(self.rand_rho(8))
        stine = self.rand_matrix(32, 8)
        stine0 = self.rand_matrix(8, 2)
        op = Stinespring(stine)
        op0 = Stinespring(stine0)
        eye = Stinespring(self.UI)
        with self.subTest(msg='qargs=[0]'):
            value = op - op0([0])
            target = op - eye.tensor(eye).tensor(op0)
            self.assertEqual(rho & value, rho & target)
        with self.subTest(msg='qargs=[1]'):
            value = op - op0([1])
            target = op - eye.tensor(op0).tensor(eye)
            self.assertEqual(rho & value, rho & target)
        with self.subTest(msg='qargs=[2]'):
            value = op - op0([2])
            target = op - op0.tensor(eye).tensor(eye)
            self.assertEqual(rho & value, rho & target)

    def test_multiply(self):
        if False:
            i = 10
            return i + 15
        'Test multiply method.'
        rho_init = DensityMatrix(self.rand_rho(2))
        val = 0.5
        (stine1, stine2) = (self.rand_matrix(16, 2), self.rand_matrix(16, 2))
        chan1 = Stinespring(stine1, input_dims=2, output_dims=4)
        rho_targ = val * (rho_init & chan1)
        chan = chan1._multiply(val)
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = val * chan1
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        rho_targ = (rho_init & chan1) * val
        chan = chan1 * val
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan2 = Stinespring((stine1, stine2), input_dims=2, output_dims=4)
        rho_targ = val * (rho_init & chan2)
        chan = chan2._multiply(val)
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = val * chan2
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_multiply_except(self):
        if False:
            i = 10
            return i + 15
        'Test multiply method raises exceptions.'
        chan = Stinespring(self.depol_stine(1))
        self.assertRaises(QiskitError, chan._multiply, 's')
        self.assertRaises(QiskitError, chan.__rmul__, 's')
        self.assertRaises(QiskitError, chan._multiply, chan)
        self.assertRaises(QiskitError, chan.__rmul__, chan)

    def test_negate(self):
        if False:
            i = 10
            return i + 15
        'Test negate method'
        rho_init = DensityMatrix(np.diag([1, 0]))
        rho_targ = DensityMatrix(np.diag([-0.5, -0.5]))
        chan = -Stinespring(self.depol_stine(1))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
if __name__ == '__main__':
    unittest.main()