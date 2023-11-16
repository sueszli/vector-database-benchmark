"""Test the SabrePreLayout pass"""
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import TranspilerError, CouplingMap, PassManager
from qiskit.transpiler.passes.layout.sabre_pre_layout import SabrePreLayout
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase

class TestSabrePreLayout(QiskitTestCase):
    """Tests the SabrePreLayout pass."""

    def test_no_constraints(self):
        if False:
            return 10
        'Test we raise at runtime if no target or coupling graph are provided.'
        qc = QuantumCircuit(2)
        empty_pass = SabrePreLayout(coupling_map=None)
        with self.assertRaises(TranspilerError):
            empty_pass.run(circuit_to_dag(qc))

    def test_starting_layout_created(self):
        if False:
            print('Hello World!')
        'Test the case that no perfect layout exists and SabrePreLayout can find a\n        starting layout.'
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 0)
        coupling_map = CouplingMap.from_ring(5)
        pm = PassManager([SabrePreLayout(coupling_map=coupling_map)])
        pm.run(qc)
        self.assertIn('sabre_starting_layouts', pm.property_set)
        layouts = pm.property_set['sabre_starting_layouts']
        self.assertEqual(len(layouts), 1)
        layout = layouts[0]
        self.assertEqual([layout[q] for q in qc.qubits], [2, 1, 0, 4])

    def test_perfect_layout_exists(self):
        if False:
            return 10
        'Test the case that a perfect layout exists.'
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 0)
        coupling_map = CouplingMap.from_ring(4)
        pm = PassManager([SabrePreLayout(coupling_map=coupling_map)])
        pm.run(qc)
        self.assertNotIn('sabre_starting_layouts', pm.property_set)

    def test_max_distance(self):
        if False:
            while True:
                i = 10
        'Test the ``max_distance`` option to SabrePreLayout.'
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        coupling_map = CouplingMap.from_ring(6)
        pm = PassManager([SabrePreLayout(coupling_map=coupling_map, max_distance=2)])
        pm.run(qc)
        self.assertNotIn('sabre_starting_layouts', pm.property_set)
        pm = PassManager([SabrePreLayout(coupling_map=coupling_map, max_distance=3)])
        pm.run(qc)
        self.assertIn('sabre_starting_layouts', pm.property_set)

    def test_call_limit_vf2(self):
        if False:
            while True:
                i = 10
        'Test the ``call_limit_vf2`` option to SabrePreLayout.'
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 0)
        coupling_map = CouplingMap.from_ring(5)
        pm = PassManager([SabrePreLayout(coupling_map=coupling_map, call_limit_vf2=1, max_distance=3)])
        pm.run(qc)
        self.assertNotIn('sabre_starting_layouts', pm.property_set)