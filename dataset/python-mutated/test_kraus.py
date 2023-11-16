"""Tests for Kraus quantum channel representation class."""
import copy
import unittest
import numpy as np
from numpy.testing import assert_allclose
from qiskit import QiskitError
from qiskit.quantum_info.states import DensityMatrix
from qiskit.quantum_info import Kraus
from .channel_test_case import ChannelTestCase

class TestKraus(ChannelTestCase):
    """Tests for Kraus channel representation."""

    def test_init(self):
        if False:
            i = 10
            return i + 15
        'Test initialization'
        chan = Kraus(self.UI)
        assert_allclose(chan.data, [self.UI])
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)
        chan = Kraus(self.depol_kraus(0.5))
        assert_allclose(chan.data, self.depol_kraus(0.5))
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)
        (kraus_l, kraus_r) = ([self.UI, self.UX], [self.UY, self.UZ])
        chan = Kraus((kraus_l, kraus_r))
        assert_allclose(chan.data, (kraus_l, kraus_r))
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)
        chan = Kraus((kraus_l, kraus_l))
        assert_allclose(chan.data, kraus_l)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)
        kraus = [np.zeros((4, 2))]
        chan = Kraus(kraus)
        assert_allclose(chan.data, kraus)
        self.assertEqual(chan.dim, (2, 4))
        self.assertIsNone(chan.num_qubits)
        self.assertRaises(QiskitError, Kraus, kraus, input_dims=4, output_dims=4)

    def test_circuit_init(self):
        if False:
            print('Hello World!')
        'Test initialization from a circuit.'
        (circuit, target) = self.simple_circuit_no_measure()
        op = Kraus(circuit)
        target = Kraus(target)
        self.assertEqual(op, target)

    def test_circuit_init_except(self):
        if False:
            for i in range(10):
                print('nop')
        'Test initialization from circuit with measure raises exception.'
        circuit = self.simple_circuit_with_measure()
        self.assertRaises(QiskitError, Kraus, circuit)

    def test_equal(self):
        if False:
            for i in range(10):
                print('nop')
        'Test __eq__ method'
        kraus = [self.rand_matrix(2, 2) for _ in range(2)]
        self.assertEqual(Kraus(kraus), Kraus(kraus))

    def test_copy(self):
        if False:
            print('Hello World!')
        'Test copy method'
        mat = np.eye(2)
        with self.subTest('Deep copy'):
            orig = Kraus(mat)
            cpy = orig.copy()
            cpy._data[0][0][0, 0] = 0.0
            self.assertFalse(cpy == orig)
        with self.subTest('Shallow copy'):
            orig = Kraus(mat)
            clone = copy.copy(orig)
            clone._data[0][0][0, 0] = 0.0
            self.assertTrue(clone == orig)

    def test_clone(self):
        if False:
            i = 10
            return i + 15
        'Test clone method'
        mat = np.eye(4)
        orig = Kraus(mat)
        clone = copy.copy(orig)
        clone._data[0][0][0, 0] = 0.0
        self.assertTrue(clone == orig)

    def test_is_cptp(self):
        if False:
            print('Hello World!')
        'Test is_cptp method.'
        self.assertTrue(Kraus(self.depol_kraus(0.5)).is_cptp())
        self.assertTrue(Kraus(self.UX).is_cptp())
        self.assertFalse(Kraus(([self.UI], [self.UX])).is_cptp())
        self.assertFalse(Kraus([self.UI, self.UX]).is_cptp())

    def test_conjugate(self):
        if False:
            for i in range(10):
                print('nop')
        'Test conjugate method.'
        (kraus_l, kraus_r) = (self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4))
        targ = Kraus([np.conjugate(k) for k in kraus_l])
        chan1 = Kraus(kraus_l)
        chan = chan1.conjugate()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (2, 4))
        targ = Kraus(([np.conjugate(k) for k in kraus_l], [np.conjugate(k) for k in kraus_r]))
        chan1 = Kraus((kraus_l, kraus_r))
        chan = chan1.conjugate()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (2, 4))

    def test_transpose(self):
        if False:
            i = 10
            return i + 15
        'Test transpose method.'
        (kraus_l, kraus_r) = (self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4))
        targ = Kraus([np.transpose(k) for k in kraus_l])
        chan1 = Kraus(kraus_l)
        chan = chan1.transpose()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))
        targ = Kraus(([np.transpose(k) for k in kraus_l], [np.transpose(k) for k in kraus_r]))
        chan1 = Kraus((kraus_l, kraus_r))
        chan = chan1.transpose()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))

    def test_adjoint(self):
        if False:
            i = 10
            return i + 15
        'Test adjoint method.'
        (kraus_l, kraus_r) = (self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4))
        targ = Kraus([np.transpose(k).conj() for k in kraus_l])
        chan1 = Kraus(kraus_l)
        chan = chan1.adjoint()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))
        targ = Kraus(([np.transpose(k).conj() for k in kraus_l], [np.transpose(k).conj() for k in kraus_r]))
        chan1 = Kraus((kraus_l, kraus_r))
        chan = chan1.adjoint()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))

    def test_compose_except(self):
        if False:
            while True:
                i = 10
        'Test compose different dimension exception'
        self.assertRaises(QiskitError, Kraus(np.eye(2)).compose, Kraus(np.eye(4)))
        self.assertRaises(QiskitError, Kraus(np.eye(2)).compose, 2)

    def test_compose(self):
        if False:
            while True:
                i = 10
        'Test compose method.'
        rho = DensityMatrix(self.rand_rho(2))
        chan1 = Kraus(self.UX)
        chan2 = Kraus(self.UY)
        chan = chan1.compose(chan2)
        targ = rho & Kraus(self.UZ)
        self.assertEqual(rho & chan, targ)
        chan1 = Kraus(self.depol_kraus(0.5))
        chan = chan1.compose(chan1)
        targ = rho & Kraus(self.depol_kraus(0.75))
        self.assertEqual(rho & chan, targ)
        (kraus1, kraus2) = (self.rand_kraus(2, 4, 4), self.rand_kraus(4, 2, 4))
        chan1 = Kraus(kraus1)
        chan2 = Kraus(kraus2)
        targ = rho & chan1 & chan2
        chan = chan1.compose(chan2)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho & chan, targ)
        chan = chan1 & chan2
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho & chan, targ)

    def test_dot(self):
        if False:
            while True:
                i = 10
        'Test dot method.'
        rho = DensityMatrix(self.rand_rho(2))
        chan1 = Kraus(self.UX)
        chan2 = Kraus(self.UY)
        targ = rho.evolve(Kraus(self.UZ))
        self.assertEqual(rho.evolve(chan1.dot(chan2)), targ)
        self.assertEqual(rho.evolve(chan1 @ chan2), targ)
        chan1 = Kraus(self.depol_kraus(0.5))
        targ = rho & Kraus(self.depol_kraus(0.75))
        self.assertEqual(rho.evolve(chan1.dot(chan1)), targ)
        self.assertEqual(rho.evolve(chan1 @ chan1), targ)
        (kraus1, kraus2) = (self.rand_kraus(2, 4, 4), self.rand_kraus(4, 2, 4))
        chan1 = Kraus(kraus1)
        chan2 = Kraus(kraus2)
        targ = rho & chan1 & chan2
        self.assertEqual(rho.evolve(chan2.dot(chan1)), targ)
        self.assertEqual(rho.evolve(chan2 @ chan1), targ)

    def test_compose_front(self):
        if False:
            print('Hello World!')
        'Test deprecated front compose method.'
        rho = DensityMatrix(self.rand_rho(2))
        chan1 = Kraus(self.UX)
        chan2 = Kraus(self.UY)
        chan = chan1.compose(chan2, front=True)
        targ = rho & Kraus(self.UZ)
        self.assertEqual(rho & chan, targ)
        chan1 = Kraus(self.depol_kraus(0.5))
        chan = chan1.compose(chan1, front=True)
        targ = rho & Kraus(self.depol_kraus(0.75))
        self.assertEqual(rho & chan, targ)
        (kraus1, kraus2) = (self.rand_kraus(2, 4, 4), self.rand_kraus(4, 2, 4))
        chan1 = Kraus(kraus1)
        chan2 = Kraus(kraus2)
        targ = rho & chan1 & chan2
        chan = chan2.compose(chan1, front=True)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho & chan, targ)

    def test_expand(self):
        if False:
            print('Hello World!')
        'Test expand method.'
        (rho0, rho1) = (np.diag([1, 0]), np.diag([0, 1]))
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = Kraus(self.UI)
        chan2 = Kraus(self.UX)
        chan = chan1.expand(chan2)
        rho_targ = DensityMatrix(np.kron(rho1, rho0))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init & chan, rho_targ)
        chan = chan2.expand(chan1)
        rho_targ = DensityMatrix(np.kron(rho0, rho1))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init & chan, rho_targ)
        chan_dep = Kraus(self.depol_kraus(1))
        chan = chan_dep.expand(chan_dep)
        rho_targ = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init & chan, rho_targ)

    def test_tensor(self):
        if False:
            print('Hello World!')
        'Test tensor method.'
        (rho0, rho1) = (np.diag([1, 0]), np.diag([0, 1]))
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = Kraus(self.UI)
        chan2 = Kraus(self.UX)
        chan = chan2.tensor(chan1)
        rho_targ = DensityMatrix(np.kron(rho1, rho0))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init & chan, rho_targ)
        chan = chan1.tensor(chan2)
        rho_targ = DensityMatrix(np.kron(rho0, rho1))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init & chan, rho_targ)
        chan_dep = Kraus(self.depol_kraus(1))
        chan = chan_dep.tensor(chan_dep)
        rho_targ = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init & chan, rho_targ)

    def test_power(self):
        if False:
            i = 10
            return i + 15
        'Test power method.'
        rho = DensityMatrix(np.diag([1, 0]))
        p_id = 0.9
        chan = Kraus(self.depol_kraus(1 - p_id))
        p_id3 = p_id ** 3
        chan3 = chan.power(3)
        targ3a = rho & chan & chan & chan
        self.assertEqual(rho & chan3, targ3a)
        targ3b = rho & Kraus(self.depol_kraus(1 - p_id3))
        self.assertEqual(rho & chan3, targ3b)

    def test_add(self):
        if False:
            return 10
        'Test add method.'
        rho = DensityMatrix(self.rand_rho(2))
        (kraus1, kraus2) = (self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4))
        chan1 = Kraus(kraus1)
        chan2 = Kraus(kraus2)
        targ = (rho & chan1) + (rho & chan2)
        chan = chan1._add(chan2)
        self.assertEqual(rho & chan, targ)
        chan = chan1 + chan2
        self.assertEqual(rho & chan, targ)
        chan = Kraus((kraus1, kraus2))
        targ = 2 * (rho & chan)
        chan = chan._add(chan)
        self.assertEqual(rho & chan, targ)

    def test_add_qargs(self):
        if False:
            print('Hello World!')
        'Test add method with qargs.'
        rho = DensityMatrix(self.rand_rho(8))
        kraus = self.rand_kraus(8, 8, 4)
        kraus0 = self.rand_kraus(2, 2, 4)
        op = Kraus(kraus)
        op0 = Kraus(kraus0)
        eye = Kraus(self.UI)
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
        kraus = self.rand_kraus(8, 8, 4)
        kraus0 = self.rand_kraus(2, 2, 4)
        op = Kraus(kraus)
        op0 = Kraus(kraus0)
        eye = Kraus(self.UI)
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

    def test_subtract(self):
        if False:
            i = 10
            return i + 15
        'Test subtract method.'
        rho = DensityMatrix(self.rand_rho(2))
        (kraus1, kraus2) = (self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4))
        chan1 = Kraus(kraus1)
        chan2 = Kraus(kraus2)
        targ = (rho & chan1) - (rho & chan2)
        chan = chan1 - chan2
        self.assertEqual(rho & chan, targ)
        chan = Kraus((kraus1, kraus2))
        targ = 0 * (rho & chan)
        chan = chan - chan
        self.assertEqual(rho & chan, targ)

    def test_multiply(self):
        if False:
            while True:
                i = 10
        'Test multiply method.'
        rho = DensityMatrix(self.rand_rho(2))
        val = 0.5
        (kraus1, kraus2) = (self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4))
        chan1 = Kraus(kraus1)
        targ = val * (rho & chan1)
        chan = chan1._multiply(val)
        self.assertEqual(rho & chan, targ)
        chan = val * chan1
        self.assertEqual(rho & chan, targ)
        targ = (rho & chan1) * val
        chan = chan1 * val
        self.assertEqual(rho & chan, targ)
        chan2 = Kraus((kraus1, kraus2))
        targ = val * (rho & chan2)
        chan = chan2._multiply(val)
        self.assertEqual(rho & chan, targ)
        chan = val * chan2
        self.assertEqual(rho & chan, targ)

    def test_multiply_except(self):
        if False:
            i = 10
            return i + 15
        'Test multiply method raises exceptions.'
        chan = Kraus(self.depol_kraus(1))
        self.assertRaises(QiskitError, chan._multiply, 's')
        self.assertRaises(QiskitError, chan.__rmul__, 's')
        self.assertRaises(QiskitError, chan._multiply, chan)
        self.assertRaises(QiskitError, chan.__rmul__, chan)

    def test_negate(self):
        if False:
            print('Hello World!')
        'Test negate method'
        rho = DensityMatrix(np.diag([1, 0]))
        targ = DensityMatrix(np.diag([-0.5, -0.5]))
        chan = -Kraus(self.depol_kraus(1))
        self.assertEqual(rho & chan, targ)
if __name__ == '__main__':
    unittest.main()