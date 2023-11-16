"""MinimumPoint pass testing"""
import math
from qiskit.transpiler.passes import MinimumPoint
from qiskit.dagcircuit import DAGCircuit
from qiskit.test import QiskitTestCase

class TestMinimumPointtPass(QiskitTestCase):
    """Tests for MinimumPoint pass."""

    def test_minimum_point_reached_fixed_point_single_field(self):
        if False:
            for i in range(10):
                print('nop')
        'Test a fixed point is reached with a single field.'
        min_pass = MinimumPoint(['depth'], prefix='test')
        dag = DAGCircuit()
        min_pass.property_set['depth'] = 42
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 0)
        self.assertEqual((math.inf,), state.score)
        self.assertIsNone(state.dag)
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 1)
        self.assertEqual(state.score, (42,))
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        out_dag = min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 1)
        self.assertEqual((42,), state.score)
        self.assertTrue(min_pass.property_set['test_minimum_point'])
        self.assertEqual(out_dag, state.dag)

    def test_minimum_point_reached_fixed_point_multiple_fields(self):
        if False:
            i = 10
            return i + 15
        'Test a fixed point is reached with a multiple fields.'
        min_pass = MinimumPoint(['fidelity', 'depth', 'size'], prefix='test')
        dag = DAGCircuit()
        min_pass.property_set['fidelity'] = 0.875
        min_pass.property_set['depth'] = 15
        min_pass.property_set['size'] = 20
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 0)
        self.assertEqual((math.inf, math.inf, math.inf), state.score)
        self.assertIsNone(state.dag)
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 1)
        self.assertEqual(state.score, (0.875, 15, 20))
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        out_dag = min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 1)
        self.assertEqual(state.score, (0.875, 15, 20))
        self.assertTrue(min_pass.property_set['test_minimum_point'])
        self.assertEqual(out_dag, state.dag)

    def test_min_over_backtrack_range(self):
        if False:
            while True:
                i = 10
        'Test minimum returned over backtrack depth.'
        min_pass = MinimumPoint(['fidelity', 'depth', 'size'], prefix='test')
        dag = DAGCircuit()
        min_pass.property_set['fidelity'] = 0.875
        min_pass.property_set['depth'] = 15
        min_pass.property_set['size'] = 20
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 0)
        self.assertEqual((math.inf, math.inf, math.inf), state.score)
        self.assertIsNone(state.dag)
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        min_pass.property_set['fidelity'] = 0.775
        min_pass.property_set['depth'] = 25
        min_pass.property_set['size'] = 35
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 1)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        min_pass.property_set['fidelity'] = 0.775
        min_pass.property_set['depth'] = 45
        min_pass.property_set['size'] = 35
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 2)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        min_pass.property_set['fidelity'] = 0.775
        min_pass.property_set['depth'] = 36
        min_pass.property_set['size'] = 40
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 3)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        min_pass.property_set['fidelity'] = 0.775
        min_pass.property_set['depth'] = 36
        min_pass.property_set['size'] = 40
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 4)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        min_pass.property_set['fidelity'] = 0.775
        min_pass.property_set['depth'] = 36
        min_pass.property_set['size'] = 40
        out_dag = min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 5)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertTrue(min_pass.property_set['test_minimum_point'])
        self.assertIs(out_dag, state.dag)

    def test_min_reset_backtrack_range(self):
        if False:
            print('Hello World!')
        'Test minimum resets backtrack depth.'
        min_pass = MinimumPoint(['fidelity', 'depth', 'size'], prefix='test')
        dag = DAGCircuit()
        min_pass.property_set['fidelity'] = 0.875
        min_pass.property_set['depth'] = 15
        min_pass.property_set['size'] = 20
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 0)
        self.assertEqual((math.inf, math.inf, math.inf), state.score)
        self.assertIsNone(state.dag)
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        min_pass.property_set['fidelity'] = 0.775
        min_pass.property_set['depth'] = 25
        min_pass.property_set['size'] = 35
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 1)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        min_pass.property_set['fidelity'] = 0.775
        min_pass.property_set['depth'] = 45
        min_pass.property_set['size'] = 35
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 2)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        min_pass.property_set['fidelity'] = 0.775
        min_pass.property_set['depth'] = 36
        min_pass.property_set['size'] = 40
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 3)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        min_pass.property_set['fidelity'] = 0.775
        min_pass.property_set['depth'] = 36
        min_pass.property_set['size'] = 40
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 4)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        min_pass.property_set['fidelity'] = 0.775
        min_pass.property_set['depth'] = 10
        min_pass.property_set['size'] = 10
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 1)
        self.assertEqual((0.775, 10, 10), state.score)
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        min_pass.property_set['fidelity'] = 0.775
        min_pass.property_set['depth'] = 25
        min_pass.property_set['size'] = 35
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 2)
        self.assertEqual((0.775, 10, 10), state.score)
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        min_pass.property_set['fidelity'] = 0.775
        min_pass.property_set['depth'] = 45
        min_pass.property_set['size'] = 35
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 3)
        self.assertEqual((0.775, 10, 10), state.score)
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        min_pass.property_set['fidelity'] = 0.775
        min_pass.property_set['depth'] = 36
        min_pass.property_set['size'] = 40
        min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 4)
        self.assertEqual((0.775, 10, 10), state.score)
        self.assertIsNone(min_pass.property_set['test_minimum_point'])
        min_pass.property_set['fidelity'] = 0.775
        min_pass.property_set['depth'] = 36
        min_pass.property_set['size'] = 40
        out_dag = min_pass.run(dag)
        state = min_pass.property_set['test_minimum_point_state']
        self.assertEqual(state.since, 5)
        self.assertEqual((0.775, 10, 10), state.score)
        self.assertTrue(min_pass.property_set['test_minimum_point'])
        self.assertIs(out_dag, state.dag)