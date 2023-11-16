"""
Tests for the Collect2qBlocks transpiler pass.
"""
import unittest
from math import pi
from ddt import ddt, data, unpack
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Collect2qBlocks
from qiskit.test import QiskitTestCase
from qiskit.circuit.library import CXGate, U1Gate, U2Gate, RXXGate, RXGate, RZGate

@ddt
class TestCollect2qBlocks(QiskitTestCase):
    """
    Tests to verify that blocks of 2q interactions are found correctly.
    """

    def test_blocks_in_topological_order(self):
        if False:
            for i in range(10):
                print('nop')
        'the pass returns blocks in correct topological order\n                                                    ______\n        q0:--[p]-------.----      q0:-------------|      |--\n                       |                 ______   |  U2  |\n        q1:--[u]--(+)-(+)---   =  q1:---|      |--|______|--\n                   |                    |  U1  |\n        q2:--------.--------      q2:---|______|------------\n        '
        qr = QuantumRegister(3, 'qr')
        qc = QuantumCircuit(qr)
        qc.p(0.5, qr[0])
        qc.u(0.0, 0.2, 0.6, qr[1])
        qc.cx(qr[2], qr[1])
        qc.cx(qr[0], qr[1])
        dag = circuit_to_dag(qc)
        topo_ops = list(dag.topological_op_nodes())
        block_1 = [topo_ops[1], topo_ops[2]]
        block_2 = [topo_ops[0], topo_ops[3]]
        pass_ = Collect2qBlocks()
        pass_.run(dag)
        self.assertTrue(pass_.property_set['block_list'], [block_1, block_2])

    def test_block_interrupted_by_gate(self):
        if False:
            print('Hello World!')
        "Test that blocks interrupted by a gate that can't be added\n        to the block can be collected correctly\n\n        This was raised in #2775 where a measure in the middle of a block\n        stopped the block collection from working properly. This was because\n        the pass didn't expect to have measures in the middle of the circuit.\n\n        blocks : [['cx', 'id', 'id', 'id'], ['id', 'cx']]\n\n                ┌───┐┌───┐┌─┐     ┌───┐┌───┐\n        q_0: |0>┤ X ├┤ I ├┤M├─────┤ I ├┤ X ├\n                └─┬─┘├───┤└╥┘┌───┐└───┘└─┬─┘\n        q_1: |0>──■──┤ I ├─╫─┤ I ├───────■──\n                     └───┘ ║ └───┘\n         c_0: 0 ═══════════╩════════════════\n\n        "
        qc = QuantumCircuit(2, 1)
        qc.cx(1, 0)
        qc.id(0)
        qc.id(1)
        qc.measure(0, 0)
        qc.id(0)
        qc.id(1)
        qc.cx(1, 0)
        dag = circuit_to_dag(qc)
        pass_ = Collect2qBlocks()
        pass_.run(dag)
        good_names = ['cx', 'u1', 'u2', 'u3', 'id']
        dag_nodes = [node for node in dag.topological_op_nodes() if node.name in good_names]
        dag_nodes = [set(dag_nodes[:4]), set(dag_nodes[4:])]
        pass_nodes = [set(bl) for bl in pass_.property_set['block_list']]
        self.assertEqual(dag_nodes, pass_nodes)

    def test_do_not_merge_conditioned_gates(self):
        if False:
            while True:
                i = 10
        "Validate that classically conditioned gates are never considered for\n        inclusion in a block. Note that there are cases where gates conditioned\n        on the same (register, value) pair could be correctly merged, but this\n        is not yet implemented.\n\n                 ┌────────┐┌────────┐┌────────┐      ┌───┐\n        qr_0: |0>┤ P(0.1) ├┤ P(0.2) ├┤ P(0.3) ├──■───┤ X ├────■───\n                 └────────┘└───┬────┘└───┬────┘┌─┴─┐ └─┬─┘  ┌─┴─┐\n        qr_1: |0>──────────────┼─────────┼─────┤ X ├───■────┤ X ├─\n                               │         │     └───┘   │    └─┬─┘\n        qr_2: |0>──────────────┼─────────┼─────────────┼──────┼───\n                            ┌──┴──┐   ┌──┴──┐       ┌──┴──┐┌──┴──┐\n         cr_0: 0 ═══════════╡     ╞═══╡     ╞═══════╡     ╞╡     ╞\n                            │ = 0 │   │ = 0 │       │ = 0 ││ = 1 │\n         cr_1: 0 ═══════════╡     ╞═══╡     ╞═══════╡     ╞╡     ╞\n                            └─────┘   └─────┘       └─────┘└─────┘\n\n        Blocks collected: [['cx']]\n        "
        qr = QuantumRegister(3, 'qr')
        cr = ClassicalRegister(2, 'cr')
        qc = QuantumCircuit(qr, cr)
        qc.p(0.1, 0)
        qc.p(0.2, 0).c_if(cr, 0)
        qc.p(0.3, 0).c_if(cr, 0)
        qc.cx(0, 1)
        qc.cx(1, 0).c_if(cr, 0)
        qc.cx(0, 1).c_if(cr, 1)
        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.run(qc)
        self.assertEqual([['cx']], [[n.name for n in block] for block in pass_manager.property_set['block_list']])

    @unpack
    @data((CXGate(), U1Gate(0.1), U2Gate(0.2, 0.3)), (RXXGate(pi / 2), RZGate(0.1), RXGate(pi / 2)), (Gate('custom2qgate', 2, []), Gate('custom1qgate1', 1, []), Gate('custom1qgate2', 1, [])))
    def test_collect_arbitrary_gates(self, twoq_gate, oneq_gate1, oneq_gate2):
        if False:
            return 10
        'Validate we can collect blocks irrespective of gate types in the circuit.'
        qc = QuantumCircuit(3)
        qc.append(oneq_gate1, [0])
        qc.append(oneq_gate2, [1])
        qc.append(twoq_gate, [0, 1])
        qc.append(oneq_gate1, [0])
        qc.append(oneq_gate2, [1])
        qc.append(oneq_gate1, [1])
        qc.append(oneq_gate2, [2])
        qc.append(twoq_gate, [1, 2])
        qc.append(oneq_gate1, [1])
        qc.append(oneq_gate2, [2])
        qc.append(oneq_gate1, [0])
        qc.append(oneq_gate2, [1])
        qc.append(twoq_gate, [0, 1])
        qc.append(oneq_gate1, [0])
        qc.append(oneq_gate2, [1])
        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.run(qc)
        self.assertEqual(len(pass_manager.property_set['block_list']), 3)
if __name__ == '__main__':
    unittest.main()