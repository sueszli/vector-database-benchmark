"""Test Operator construction, including OpPrimitives and singletons."""
import unittest
from test.python.opflow import QiskitOpflowTestCase
import numpy as np
from qiskit import QuantumCircuit, BasicAer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.opflow import StateFn, Zero, One, Plus, Minus, PrimitiveOp, CircuitOp, SummedOp, H, I, Z, X, Y, CX, CircuitStateFn, DictToCircuitSum

class TestStateConstruction(QiskitOpflowTestCase):
    """State Construction tests."""

    def test_state_singletons(self):
        if False:
            for i in range(10):
                print('nop')
        'state singletons test'
        self.assertEqual(Zero.primitive, {'0': 1})
        self.assertEqual(One.primitive, {'1': 1})
        self.assertEqual((Zero ^ 5).primitive, {'00000': 1})
        self.assertEqual((One ^ 5).primitive, {'11111': 1})
        self.assertEqual((Zero ^ One ^ 3).primitive, {'010101': 1})

    def test_zero_broadcast(self):
        if False:
            for i in range(10):
                print('nop')
        'zero broadcast test'
        np.testing.assert_array_almost_equal(((H ^ 5) @ Zero).to_matrix(), (Plus ^ 5).to_matrix())

    def test_state_to_matrix(self):
        if False:
            for i in range(10):
                print('nop')
        'state to matrix test'
        np.testing.assert_array_equal(Zero.to_matrix(), np.array([1, 0]))
        np.testing.assert_array_equal(One.to_matrix(), np.array([0, 1]))
        np.testing.assert_array_almost_equal(Plus.to_matrix(), (Zero.to_matrix() + One.to_matrix()) / np.sqrt(2))
        np.testing.assert_array_almost_equal(Minus.to_matrix(), (Zero.to_matrix() - One.to_matrix()) / np.sqrt(2))
        gnarly_state = (One ^ Plus ^ Zero ^ Minus * 0.3) @ StateFn(Statevector.from_label('r0+l')) + StateFn(X ^ Z ^ Y ^ I) * 0.1j
        gnarly_mat = gnarly_state.to_matrix()
        gnarly_mat_separate = (One ^ Plus ^ Zero ^ Minus * 0.3).to_matrix()
        gnarly_mat_separate = np.dot(gnarly_mat_separate, StateFn(Statevector.from_label('r0+l')).to_matrix())
        gnarly_mat_separate = gnarly_mat_separate + (StateFn(X ^ Z ^ Y ^ I) * 0.1j).to_matrix()
        np.testing.assert_array_almost_equal(gnarly_mat, gnarly_mat_separate)

    def test_qiskit_result_instantiation(self):
        if False:
            while True:
                i = 10
        'qiskit result instantiation test'
        qc = QuantumCircuit(3)
        qc.h(0)
        sv_res = execute(qc, BasicAer.get_backend('statevector_simulator')).result()
        sv_vector = sv_res.get_statevector()
        qc_op = PrimitiveOp(qc) @ Zero
        qasm_res = execute(qc_op.to_circuit(meas=True), BasicAer.get_backend('qasm_simulator')).result()
        np.testing.assert_array_almost_equal(StateFn(sv_res).to_matrix(), [0.5 ** 0.5, 0.5 ** 0.5, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_almost_equal(StateFn(sv_vector).to_matrix(), [0.5 ** 0.5, 0.5 ** 0.5, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_almost_equal(StateFn(qasm_res).to_matrix(), [0.5 ** 0.5, 0.5 ** 0.5, 0, 0, 0, 0, 0, 0], decimal=1)
        np.testing.assert_array_almost_equal(((I ^ I ^ H) @ Zero).to_matrix(), [0.5 ** 0.5, 0.5 ** 0.5, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_almost_equal(qc_op.to_matrix(), [0.5 ** 0.5, 0.5 ** 0.5, 0, 0, 0, 0, 0, 0])

    def test_state_meas_composition(self):
        if False:
            i = 10
            return i + 15
        'state meas composition test'
        pass

    def test_add_direct(self):
        if False:
            i = 10
            return i + 15
        'add direct test'
        wf = StateFn({'101010': 0.5, '111111': 0.3}) + (Zero ^ 6)
        self.assertEqual(wf.primitive, {'101010': 0.5, '111111': 0.3, '000000': 1.0})
        wf = 4 * StateFn({'101010': 0.5, '111111': 0.3}) + (3 + 0.1j) * (Zero ^ 6)
        self.assertEqual(wf.primitive, {'000000': 3 + 0.1j, '101010': 2 + 0j, '111111': 1.2 + 0j})

    def test_circuit_state_fn_from_dict_as_sum(self):
        if False:
            i = 10
            return i + 15
        'state fn circuit from dict as sum test'
        statedict = {'1010101': 0.5, '1000000': 0.1, '0000000': 0.2j, '1111111': 0.5j}
        sfc_sum = CircuitStateFn.from_dict(statedict)
        self.assertIsInstance(sfc_sum, SummedOp)
        for sfc_op in sfc_sum.oplist:
            self.assertIsInstance(sfc_op, CircuitStateFn)
            samples = sfc_op.sample()
            self.assertIn(list(samples.keys())[0], statedict)
            self.assertEqual(sfc_op.coeff, statedict[list(samples.keys())[0]])
        np.testing.assert_array_almost_equal(StateFn(statedict).to_matrix(), sfc_sum.to_matrix())

    def test_circuit_state_fn_from_dict_initialize(self):
        if False:
            print('Hello World!')
        'state fn circuit from dict initialize test'
        statedict = {'101': 0.5, '100': 0.1, '000': 0.2, '111': 0.5}
        sfc = CircuitStateFn.from_dict(statedict)
        self.assertIsInstance(sfc, CircuitStateFn)
        samples = sfc.sample()
        np.testing.assert_array_almost_equal(StateFn(statedict).to_matrix(), np.round(sfc.to_matrix(), decimals=1))
        for (k, v) in samples.items():
            self.assertIn(k, statedict)
            self.assertAlmostEqual(v, np.abs(statedict[k]) ** 0.5, delta=0.5)
        sfc_vector = CircuitStateFn.from_vector(StateFn(statedict).to_matrix())
        np.testing.assert_array_almost_equal(StateFn(statedict).to_matrix(), sfc_vector.to_matrix())

    def test_circuit_state_fn_from_complex_vector_initialize(self):
        if False:
            for i in range(10):
                print('nop')
        'state fn circuit from complex vector initialize test'
        sfc = CircuitStateFn.from_vector(np.array([1j / np.sqrt(2), 0, 1j / np.sqrt(2), 0]))
        self.assertIsInstance(sfc, CircuitStateFn)

    def test_sampling(self):
        if False:
            for i in range(10):
                print('nop')
        'state fn circuit from dict initialize test'
        statedict = {'101': 0.5 + 1j, '100': 0.1 + 2j, '000': 0.2 + 0j, '111': 0.5 + 1j}
        sfc = CircuitStateFn.from_dict(statedict)
        circ_samples = sfc.sample()
        dict_samples = StateFn(statedict).sample()
        vec_samples = StateFn(statedict).to_matrix_op().sample()
        for (k, v) in circ_samples.items():
            self.assertIn(k, dict_samples)
            self.assertIn(k, vec_samples)
            self.assertAlmostEqual(v, np.abs(dict_samples[k]) ** 0.5, delta=0.5)
            self.assertAlmostEqual(v, np.abs(vec_samples[k]) ** 0.5, delta=0.5)

    def test_dict_to_circuit_sum(self):
        if False:
            while True:
                i = 10
        'Test DictToCircuitSum converter.'
        dict_state_3q = StateFn({'101': 0.5, '100': 0.1, '000': 0.2, '111': 0.5})
        circuit_state_3q = DictToCircuitSum().convert(dict_state_3q)
        self.assertIsInstance(circuit_state_3q, CircuitStateFn)
        np.testing.assert_array_almost_equal(dict_state_3q.to_matrix(), circuit_state_3q.to_matrix())
        dict_state_4q = dict_state_3q ^ Zero
        circuit_state_4q = DictToCircuitSum().convert(dict_state_4q)
        self.assertIsInstance(circuit_state_4q, SummedOp)
        np.testing.assert_array_almost_equal(dict_state_4q.to_matrix(), circuit_state_4q.to_matrix())
        vect_state_3q = dict_state_3q.to_matrix_op()
        circuit_state_3q_vect = DictToCircuitSum().convert(vect_state_3q)
        self.assertIsInstance(circuit_state_3q_vect, CircuitStateFn)
        np.testing.assert_array_almost_equal(vect_state_3q.to_matrix(), circuit_state_3q_vect.to_matrix())

    def test_circuit_permute(self):
        if False:
            for i in range(10):
                print('nop')
        "Test the CircuitStateFn's .permute method"
        perm = range(7)[::-1]
        c_op = (CX ^ 3 ^ X) @ (H ^ 7) @ (X ^ Y ^ Z ^ I ^ X ^ X ^ X) @ (Y ^ (CX ^ 3)) @ (X ^ Y ^ Z ^ I ^ X ^ X ^ X) @ Zero
        c_op_perm = c_op.permute(perm)
        self.assertNotEqual(c_op, c_op_perm)
        c_op_id = c_op_perm.permute(perm)
        self.assertEqual(c_op, c_op_id)

    def test_primitive_param_binding(self):
        if False:
            print('Hello World!')
        'Test that assign_parameters binds parameters of both the underlying primitive and coeffs.'
        theta = ParameterVector('theta', 2)
        op = StateFn(theta[0] * X) * theta[1]
        bound = op.assign_parameters(dict(zip(theta, [0.2, 0.3])))
        self.assertEqual(bound.coeff, 0.3)
        self.assertEqual(bound.primitive.coeff, 0.2)

    def test_flatten_statefn_composed_with_composed_op(self):
        if False:
            return 10
        'Test that composing a StateFn with a ComposedOp constructs a single ComposedOp'
        circuit = QuantumCircuit(1)
        vector = [1, 0]
        ex = ~StateFn(I) @ (CircuitOp(circuit) @ StateFn(vector))
        self.assertEqual(len(ex), 3)
        self.assertEqual(ex.eval(), 1)

    def test_tensorstate_to_matrix(self):
        if False:
            return 10
        'Test tensored states to matrix works correctly with a global coefficient.\n\n        Regression test of Qiskit/qiskit-terra#9398.\n        '
        state = 0.5 * (Plus ^ Zero)
        expected = 1 / (2 * np.sqrt(2)) * np.array([1, 0, 1, 0])
        np.testing.assert_almost_equal(state.to_matrix(), expected)
if __name__ == '__main__':
    unittest.main()