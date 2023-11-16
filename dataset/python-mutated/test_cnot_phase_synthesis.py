"""Test cnot circuit and cnot-phase circuit synthesis algorithms"""
import unittest
import ddt
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis.linear import synth_cnot_count_full_pmh
from qiskit.synthesis.linear_phase import synth_cnot_phase_aam
from qiskit.transpiler.synthesis.graysynth import cnot_synth, graysynth
from qiskit.test import QiskitTestCase

@ddt.ddt
class TestGraySynth(QiskitTestCase):
    """Test the Gray-Synth algorithm."""

    @ddt.data(synth_cnot_phase_aam, graysynth)
    def test_gray_synth(self, synth_func):
        if False:
            for i in range(10):
                print('nop')
        "Test synthesis of a small parity network via gray_synth.\n\n        The algorithm should take the following matrix as an input:\n        S =\n        [[0, 1, 1, 0, 1, 1],\n         [0, 1, 1, 0, 1, 0],\n         [0, 0, 0, 1, 1, 0],\n         [1, 0, 0, 1, 1, 1],\n         [0, 1, 0, 0, 1, 0],\n         [0, 1, 0, 0, 1, 0]]\n\n        Along with some rotation angles:\n        ['s', 't', 'z', 's', 't', 't'])\n\n        which together specify the Fourier expansion in the sum-over-paths representation\n        of a quantum circuit.\n\n        And should return the following circuit (or an equivalent one):\n                          ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐\n        q_0: |0>──────────┤ X ├┤ X ├┤ T ├┤ X ├┤ X ├┤ X ├┤ X ├┤ T ├┤ X ├┤ T ├┤ X ├┤ X ├┤ Z ├┤ X ├\n                          └─┬─┘└─┬─┘└───┘└─┬─┘└─┬─┘└─┬─┘└─┬─┘└───┘└─┬─┘└───┘└─┬─┘└─┬─┘└───┘└─┬─┘\n        q_1: |0>────────────┼────┼─────────■────┼────┼────┼─────────┼─────────┼────┼─────────■──\n                            │    │              │    │    │         │         │    │\n        q_2: |0>───────■────■────┼──────────────■────┼────┼─────────┼────■────┼────┼────────────\n                ┌───┐┌─┴─┐┌───┐  │                   │    │         │  ┌─┴─┐  │    │\n        q_3: |0>┤ S ├┤ X ├┤ S ├──■───────────────────┼────┼─────────■──┤ X ├──┼────┼────────────\n                └───┘└───┘└───┘                      │    │            └───┘  │    │\n        q_4: |0>─────────────────────────────────────■────┼───────────────────■────┼────────────\n                                                          │                        │\n        q_5: |0>──────────────────────────────────────────■────────────────────────■────────────\n\n        "
        cnots = [[0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 1], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0]]
        angles = ['s', 't', 'z', 's', 't', 't']
        c_gray = synth_func(cnots, angles)
        unitary_gray = UnitaryGate(Operator(c_gray))
        q = QuantumRegister(6, 'q')
        c_compare = QuantumCircuit(q)
        c_compare.s(q[3])
        c_compare.cx(q[2], q[3])
        c_compare.s(q[3])
        c_compare.cx(q[2], q[0])
        c_compare.cx(q[3], q[0])
        c_compare.t(q[0])
        c_compare.cx(q[1], q[0])
        c_compare.cx(q[2], q[0])
        c_compare.cx(q[4], q[0])
        c_compare.cx(q[5], q[0])
        c_compare.t(q[0])
        c_compare.cx(q[3], q[0])
        c_compare.t(q[0])
        c_compare.cx(q[2], q[3])
        c_compare.cx(q[4], q[0])
        c_compare.cx(q[5], q[0])
        c_compare.z(q[0])
        c_compare.cx(q[1], q[0])
        unitary_compare = UnitaryGate(Operator(c_compare))
        self.assertEqual(unitary_gray, unitary_compare)

    @ddt.data(synth_cnot_phase_aam, graysynth)
    def test_paper_example(self, synth_func):
        if False:
            while True:
                i = 10
        'Test synthesis of a diagonal operator from the paper.\n\n        The diagonal operator in Example 4.2\n            U|x> = e^(2.pi.i.f(x))|x>,\n        where\n            f(x) = 1/8*(x1^x2 + x0 + x0^x3 + x0^x1^x2 + x0^x1^x3 + x0^x1)\n\n        The algorithm should take the following matrix as an input:\n        S = [[0, 1, 1, 1, 1, 1],\n             [1, 0, 0, 1, 1, 1],\n             [1, 0, 0, 1, 0, 0],\n             [0, 0, 1, 0, 1, 0]]\n\n        and only T gates as phase rotations,\n\n        And should return the following circuit (or an equivalent one):\n                ┌───┐┌───┐     ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐     ┌───┐┌───┐\n        q_0: |0>┤ T ├┤ X ├─────┤ T ├┤ X ├┤ X ├┤ T ├┤ X ├┤ T ├┤ X ├┤ T ├─────┤ X ├┤ X ├\n                ├───┤└─┬─┘┌───┐└───┘└─┬─┘└─┬─┘└───┘└─┬─┘└───┘└─┬─┘└───┘┌───┐└─┬─┘└─┬─┘\n        q_1: |0>┤ X ├──┼──┤ T ├───────■────┼─────────┼─────────┼───────┤ X ├──■────┼──\n                └─┬─┘  │  └───┘            │         │         │       └─┬─┘       │\n        q_2: |0>──■────┼───────────────────┼─────────■─────────┼─────────■─────────┼──\n                       │                   │                   │                   │\n        q_3: |0>───────■───────────────────■───────────────────■───────────────────■──\n        '
        cnots = [[0, 1, 1, 1, 1, 1], [1, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0]]
        angles = ['t'] * 6
        c_gray = synth_func(cnots, angles)
        unitary_gray = UnitaryGate(Operator(c_gray))
        q = QuantumRegister(4, 'q')
        c_compare = QuantumCircuit(q)
        c_compare.t(q[0])
        c_compare.cx(q[2], q[1])
        c_compare.cx(q[3], q[0])
        c_compare.t(q[0])
        c_compare.t(q[1])
        c_compare.cx(q[1], q[0])
        c_compare.cx(q[3], q[0])
        c_compare.t(q[0])
        c_compare.cx(q[2], q[0])
        c_compare.t(q[0])
        c_compare.cx(q[3], q[0])
        c_compare.t(q[0])
        c_compare.cx(q[2], q[1])
        c_compare.cx(q[1], q[0])
        c_compare.cx(q[3], q[0])
        unitary_compare = UnitaryGate(Operator(c_compare))
        self.assertEqual(unitary_gray, unitary_compare)

    @ddt.data(synth_cnot_phase_aam, graysynth)
    def test_ccz(self, synth_func):
        if False:
            for i in range(10):
                print('nop')
        'Test synthesis of the doubly-controlled Z gate.\n\n        The diagonal operator in Example 4.3\n            U|x> = e^(2.pi.i.f(x))|x>,\n        where\n            f(x) = 1/8*(x0 + x1 + x2 - x0^x1 - x0^x2 - x1^x2 + x0^x1^x2)\n\n        The algorithm should take the following matrix as an input:\n        S = [[1, 0, 0, 1, 1, 0, 1],\n             [0, 1, 0, 1, 0, 1, 1],\n             [0, 0, 1, 0, 1, 1, 1]]\n\n        and only T and T* gates as phase rotations,\n\n        And should return the following circuit (or an equivalent one):\n                ┌───┐\n        q_0: |0>┤ T ├───────■──────────────■───────────────────■──────────────■──\n                └───┘┌───┐┌─┴─┐┌───┐       │                   │            ┌─┴─┐\n        q_1: |0>─────┤ T ├┤ X ├┤ T*├───────┼─────────■─────────┼─────────■──┤ X ├\n                     └───┘└───┘└───┘┌───┐┌─┴─┐┌───┐┌─┴─┐┌───┐┌─┴─┐┌───┐┌─┴─┐└───┘\n        q_2: |0>────────────────────┤ T ├┤ X ├┤ T*├┤ X ├┤ T*├┤ X ├┤ T ├┤ X ├─────\n                                    └───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘\n        '
        cnots = [[1, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 1, 1]]
        angles = ['t', 't', 't', 'tdg', 'tdg', 'tdg', 't']
        c_gray = synth_func(cnots, angles)
        unitary_gray = UnitaryGate(Operator(c_gray))
        q = QuantumRegister(3, 'q')
        c_compare = QuantumCircuit(q)
        c_compare.t(q[0])
        c_compare.t(q[1])
        c_compare.cx(q[0], q[1])
        c_compare.tdg(q[1])
        c_compare.t(q[2])
        c_compare.cx(q[0], q[2])
        c_compare.tdg(q[2])
        c_compare.cx(q[1], q[2])
        c_compare.tdg(q[2])
        c_compare.cx(q[0], q[2])
        c_compare.t(q[2])
        c_compare.cx(q[1], q[2])
        c_compare.cx(q[0], q[1])
        unitary_compare = UnitaryGate(Operator(c_compare))
        self.assertEqual(unitary_gray, unitary_compare)

@ddt.ddt
class TestPatelMarkovHayes(QiskitTestCase):
    """Test the Patel-Markov-Hayes algorithm for synthesizing linear
    CNOT-only circuits."""

    @ddt.data(synth_cnot_count_full_pmh, cnot_synth)
    def test_patel_markov_hayes(self, synth_func):
        if False:
            while True:
                i = 10
        'Test synthesis of a small linear circuit\n        (example from paper, Figure 3).\n\n        The algorithm should take the following matrix as an input:\n        S = [[1, 1, 0, 0, 0, 0],\n             [1, 0, 0, 1, 1, 0],\n             [0, 1, 0, 0, 1, 0],\n             [1, 1, 1, 1, 1, 1],\n             [1, 1, 0, 1, 1, 1],\n             [0, 0, 1, 1, 1, 0]]\n\n        And should return the following circuit (or an equivalent one):\n                          ┌───┐\n        q_0: |0>──────────┤ X ├──────────────────────────────────────────■────■────■──\n                          └─┬─┘┌───┐                                   ┌─┴─┐  │    │\n        q_1: |0>────────────■──┤ X ├────────────────────────────────■──┤ X ├──┼────┼──\n                     ┌───┐     └─┬─┘┌───┐          ┌───┐          ┌─┴─┐└───┘  │    │\n        q_2: |0>─────┤ X ├───────┼──┤ X ├───────■──┤ X ├───────■──┤ X ├───────┼────┼──\n                ┌───┐└─┬─┘       │  └─┬─┘┌───┐┌─┴─┐└─┬─┘       │  └───┘       │  ┌─┴─┐\n        q_3: |0>┤ X ├──┼─────────■────┼──┤ X ├┤ X ├──■────■────┼──────────────┼──┤ X ├\n                └─┬─┘  │              │  └─┬─┘├───┤       │  ┌─┴─┐          ┌─┴─┐└───┘\n        q_4: |0>──■────┼──────────────■────■──┤ X ├───────┼──┤ X ├──────────┤ X ├─────\n                       │                      └─┬─┘     ┌─┴─┐└───┘          └───┘\n        q_5: |0>───────■────────────────────────■───────┤ X ├─────────────────────────\n                                                        └───┘\n        '
        state = [[1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0], [1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0]]
        c_patel = synth_func(state)
        unitary_patel = UnitaryGate(Operator(c_patel))
        q = QuantumRegister(6, 'q')
        c_compare = QuantumCircuit(q)
        c_compare.cx(q[4], q[3])
        c_compare.cx(q[5], q[2])
        c_compare.cx(q[1], q[0])
        c_compare.cx(q[3], q[1])
        c_compare.cx(q[4], q[2])
        c_compare.cx(q[4], q[3])
        c_compare.cx(q[5], q[4])
        c_compare.cx(q[2], q[3])
        c_compare.cx(q[3], q[2])
        c_compare.cx(q[3], q[5])
        c_compare.cx(q[2], q[4])
        c_compare.cx(q[1], q[2])
        c_compare.cx(q[0], q[1])
        c_compare.cx(q[0], q[4])
        c_compare.cx(q[0], q[3])
        unitary_compare = UnitaryGate(Operator(c_compare))
        self.assertEqual(unitary_patel, unitary_compare)
if __name__ == '__main__':
    unittest.main()