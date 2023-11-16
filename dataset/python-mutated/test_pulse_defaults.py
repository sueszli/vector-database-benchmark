"""Test the PulseDefaults part of the backend."""
import copy
import warnings
import numpy as np
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import FakeOpenPulse2Q

class TestPulseDefaults(QiskitTestCase):
    """Test the PulseDefaults creation and method usage."""

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.defs = FakeOpenPulse2Q().defaults()
        self.inst_map = self.defs.instruction_schedule_map

    def test_buffer(self):
        if False:
            while True:
                i = 10
        'Test getting the buffer value.'
        self.assertEqual(self.defs.buffer, 10)

    def test_freq_est(self):
        if False:
            for i in range(10):
                print('nop')
        'Test extracting qubit frequencies.'
        warnings.simplefilter('ignore')
        self.assertEqual(self.defs.qubit_freq_est[1], 5.0 * 1000000000.0)
        self.assertEqual(self.defs.meas_freq_est[0], 6.5 * 1000000000.0)
        warnings.simplefilter('default')

    def test_default_building(self):
        if False:
            print('Hello World!')
        'Test building of ops definition is properly built from backend.'
        self.assertTrue(self.inst_map.has('u1', (0,)))
        self.assertTrue(self.inst_map.has('u3', (0,)))
        self.assertTrue(self.inst_map.has('u3', 1))
        self.assertTrue(self.inst_map.has('cx', (0, 1)))
        self.assertEqual(self.inst_map.get_parameters('u1', 0), ('P0',))
        u1_minus_pi = self.inst_map.get('u1', 0, P0=np.pi)
        fc_cmd = u1_minus_pi.instructions[0][-1]
        self.assertAlmostEqual(fc_cmd.phase, -np.pi)

    def test_str(self):
        if False:
            return 10
        'Test that __str__ method works.'
        self.assertEqual('<PulseDefaults(<InstructionScheduleMap(1Q instructions:\n  q0:', str(self.defs)[:61])
        self.assertTrue('Multi qubit instructions:\n  (0, 1): ' in str(self.defs)[70:])
        self.assertTrue('Qubit Frequencies [GHz]\n[4.9, 5.0]\nMeasurement Frequencies [GHz]\n[6.5, 6.6] )>' in str(self.defs)[100:])

    def test_deepcopy(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that deepcopy creates an identical object.'
        copy_defs = copy.deepcopy(self.defs)
        self.assertEqual(list(copy_defs.to_dict().keys()), list(self.defs.to_dict().keys()))