"""Test the NormalizeRXAngle pass"""
import unittest
import numpy as np
from ddt import ddt, named_data
from qiskit import QuantumCircuit
from qiskit.transpiler.passes.optimization.normalize_rx_angle import NormalizeRXAngle
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import FakeBelemV2
from qiskit.transpiler import Target
from qiskit.circuit.library.standard_gates import SXGate

@ddt
class TestNormalizeRXAngle(QiskitTestCase):
    """Tests the NormalizeRXAngle pass."""

    def test_not_convert_to_x_if_no_calib_in_target(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that RX(pi) is NOT converted to X,\n        if X calibration is not present in the target'
        empty_target = Target()
        tp = NormalizeRXAngle(target=empty_target)
        qc = QuantumCircuit(1)
        qc.rx(90, 0)
        transpiled_circ = tp(qc)
        self.assertEqual(transpiled_circ.count_ops().get('x', 0), 0)

    def test_sx_conversion_works(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that RX(pi/2) is converted to SX,\n        if SX calibration is present in the target'
        target = Target()
        target.add_instruction(SXGate(), properties={(0,): None})
        tp = NormalizeRXAngle(target=target)
        qc = QuantumCircuit(1)
        qc.rx(np.pi / 2, 0)
        transpiled_circ = tp(qc)
        self.assertEqual(transpiled_circ.count_ops().get('sx', 0), 1)

    def test_rz_added_for_negative_rotation_angles(self):
        if False:
            print('Hello World!')
        'Check that RZ is added before and after RX,\n        if RX rotation angle is negative'
        backend = FakeBelemV2()
        tp = NormalizeRXAngle(target=backend.target)
        qc = QuantumCircuit(1)
        qc.rx(-1 / 3 * np.pi, 0)
        transpiled_circ = tp(qc)
        qc_ref = QuantumCircuit(1)
        qc_ref.rz(np.pi, 0)
        qc_ref.rx(np.pi / 3, 0)
        qc_ref.rz(-np.pi, 0)
        self.assertQuantumCircuitEqual(transpiled_circ, qc_ref)

    @named_data({'name': '-0.3pi', 'raw_theta': -0.3 * np.pi, 'correct_wrapped_theta': 0.3 * np.pi}, {'name': '1.7pi', 'raw_theta': 1.7 * np.pi, 'correct_wrapped_theta': 0.3 * np.pi}, {'name': '2.2pi', 'raw_theta': 2.2 * np.pi, 'correct_wrapped_theta': 0.2 * np.pi})
    def test_angle_wrapping_works(self, raw_theta, correct_wrapped_theta):
        if False:
            for i in range(10):
                print('nop')
        'Check that RX rotation angles are correctly wrapped to [0, pi]'
        backend = FakeBelemV2()
        tp = NormalizeRXAngle(target=backend.target)
        qc = QuantumCircuit(1)
        qc.rx(raw_theta, 0)
        transpiled_circuit = tp(qc)
        wrapped_theta = transpiled_circuit.get_instructions('rx')[0].operation.params[0]
        self.assertAlmostEqual(wrapped_theta, correct_wrapped_theta)

    @named_data({'name': 'angles are within resolution', 'resolution': 0.1, 'rx_angles': [0.3, 0.303], 'correct_num_of_cals': 1}, {'name': 'angles are not within resolution', 'resolution': 0.1, 'rx_angles': [0.2, 0.4], 'correct_num_of_cals': 2}, {'name': 'same angle three times', 'resolution': 0.1, 'rx_angles': [0.2, 0.2, 0.2], 'correct_num_of_cals': 1})
    def test_quantize_angles(self, resolution, rx_angles, correct_num_of_cals):
        if False:
            while True:
                i = 10
        'Test that quantize_angles() adds a new calibration only if\n        the requested angle is not in the vicinity of the already generated angles.\n        '
        backend = FakeBelemV2()
        tp = NormalizeRXAngle(backend.target, resolution_in_radian=resolution)
        qc = QuantumCircuit(1)
        for rx_angle in rx_angles:
            qc.rx(rx_angle, 0)
        transpiled_circuit = tp(qc)
        angles = [inst.operation.params[0] for inst in transpiled_circuit.data if inst.operation.name == 'rx']
        angles_without_duplicate = list(dict.fromkeys(angles))
        self.assertEqual(len(angles_without_duplicate), correct_num_of_cals)
if __name__ == '__main__':
    unittest.main()