"""
Tests for the Collect2qBlocks transpiler pass.
"""
import math
import unittest
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CollectMultiQBlocks
from qiskit.test import QiskitTestCase

class TestCollect2qBlocks(QiskitTestCase):
    """
    Tests to verify that blocks of 2q interactions are found correctly.
    """

    def test_blocks_in_topological_order(self):
        if False:
            while True:
                i = 10
        'the pass returns blocks in correct topological order\n                                                    ______\n        q0:--[u1]-------.----      q0:-------------|      |--\n                        |                 ______   |  U2  |\n        q1:--[u2]--(+)-(+)---   =  q1:---|      |--|______|--\n                    |                    |  U1  |\n        q2:---------.--------      q2:---|______|------------\n        '
        qr = QuantumRegister(3, 'qr')
        qc = QuantumCircuit(qr)
        qc.p(0.5, qr[0])
        qc.u(math.pi / 2, 0.2, 0.6, qr[1])
        qc.cx(qr[2], qr[1])
        qc.cx(qr[0], qr[1])
        dag = circuit_to_dag(qc)
        topo_ops = list(dag.topological_op_nodes())
        block_1 = [topo_ops[1], topo_ops[2]]
        block_2 = [topo_ops[0], topo_ops[3]]
        pass_ = CollectMultiQBlocks()
        pass_.run(dag)
        self.assertTrue(pass_.property_set['block_list'], [block_1, block_2])

    def test_block_interrupted_by_gate(self):
        if False:
            for i in range(10):
                print('nop')
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
        pass_ = CollectMultiQBlocks()
        pass_.run(dag)
        good_names = ['cx', 'u1', 'u2', 'u3', 'id']
        dag_nodes = [node for node in dag.topological_op_nodes() if node.name in good_names]
        dag_nodes = [set(dag_nodes[:4]), set(dag_nodes[4:])]
        pass_nodes = [set(bl) for bl in pass_.property_set['block_list']]
        self.assertEqual(dag_nodes, pass_nodes)

    def test_block_with_classical_register(self):
        if False:
            print('Hello World!')
        "Test that only blocks that share quantum wires are added to the block.\n        It was the case that gates which shared a classical wire could be added to\n        the same block, despite not sharing the same qubits. This was fixed in #2956.\n\n                                    ┌─────────────────────┐\n        q_0: |0>────────────────────┤ U2(0.25*pi,0.25*pi) ├\n                     ┌─────────────┐└──────────┬──────────┘\n        q_1: |0>──■──┤ U1(0.25*pi) ├───────────┼───────────\n                ┌─┴─┐└──────┬──────┘           │\n        q_2: |0>┤ X ├───────┼──────────────────┼───────────\n                └───┘    ┌──┴──┐            ┌──┴──┐\n        c0_0: 0 ═════════╡ = 0 ╞════════════╡ = 0 ╞════════\n                         └─────┘            └─────┘\n\n        Previously the blocks collected were : [['cx', 'u1', 'u2']]\n        This is now corrected to : [['cx', 'u1']]\n        "
        qasmstr = '\n        OPENQASM 2.0;\n        include "qelib1.inc";\n        qreg q[3];\n        creg c0[1];\n\n        cx q[1],q[2];\n        if(c0==0) u1(0.25*pi) q[1];\n        if(c0==0) u2(0.25*pi, 0.25*pi) q[0];\n        '
        qc = QuantumCircuit.from_qasm_str(qasmstr)
        pass_manager = PassManager()
        pass_manager.append(CollectMultiQBlocks())
        pass_manager.run(qc)
        self.assertEqual([['cx']], [[n.name for n in block] for block in pass_manager.property_set['block_list']])

    def test_do_not_merge_conditioned_gates(self):
        if False:
            for i in range(10):
                print('nop')
        "Validate that classically conditioned gates are never considered for\n        inclusion in a block. Note that there are cases where gates conditioned\n        on the same (register, value) pair could be correctly merged, but this is\n        not yet implemented.\n\n                 ┌─────────┐┌─────────┐┌─────────┐      ┌───┐\n        qr_0: |0>┤ U1(0.1) ├┤ U1(0.2) ├┤ U1(0.3) ├──■───┤ X ├────■───\n                 └─────────┘└────┬────┘└────┬────┘┌─┴─┐ └─┬─┘  ┌─┴─┐\n        qr_1: |0>────────────────┼──────────┼─────┤ X ├───■────┤ X ├─\n                                 │          │     └───┘   │    └─┬─┘\n        qr_2: |0>────────────────┼──────────┼─────────────┼──────┼───\n                              ┌──┴──┐    ┌──┴──┐       ┌──┴──┐┌──┴──┐\n         cr_0: 0 ═════════════╡     ╞════╡     ╞═══════╡     ╞╡     ╞\n                              │ = 0 │    │ = 0 │       │ = 0 ││ = 1 │\n         cr_1: 0 ═════════════╡     ╞════╡     ╞═══════╡     ╞╡     ╞\n                              └─────┘    └─────┘       └─────┘└─────┘\n\n        Previously the blocks collected were : [['u1', 'u1', 'u1', 'cx', 'cx', 'cx']]\n        This is now corrected to : [['cx']]\n        "
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
        pass_manager.append(CollectMultiQBlocks())
        pass_manager.run(qc)
        for block in pass_manager.property_set['block_list']:
            self.assertTrue(len(block) <= 1)

    def test_do_not_go_across_barrier(self):
        if False:
            print('Hello World!')
        'Validate that blocks are not collected across barriers\n                   ░\n        q_0: ──■───░───■──\n             ┌─┴─┐ ░ ┌─┴─┐\n        q_1: ┤ X ├─░─┤ X ├\n             └───┘ ░ └───┘\n        q_2: ──────░──────\n                   ░\n        '
        qr = QuantumRegister(3, 'qr')
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)
        qc.barrier()
        qc.cx(0, 1)
        pass_manager = PassManager()
        pass_manager.append(CollectMultiQBlocks())
        pass_manager.run(qc)
        for block in pass_manager.property_set['block_list']:
            self.assertTrue(len(block) <= 1)

    def test_optimal_blocking(self):
        if False:
            while True:
                i = 10
        'Test that blocks are created optimally in at least the two quibit case.\n        Here, if the topological ordering of nodes is wrong then we might create\n        an extra block\n              ┌───┐\n        qr_0: ┤ X ├───────■────■───────\n              └───┘┌───┐┌─┴─┐┌─┴─┐┌───┐\n        qr_1: ──■──┤ H ├┤ X ├┤ X ├┤ H ├\n              ┌─┴─┐├───┤└───┘└───┘└───┘\n        qr_2: ┤ X ├┤ X ├───────────────\n              └───┘└───┘\n        '
        qr = QuantumRegister(3, 'qr')
        qc = QuantumCircuit(qr)
        qc.x(0)
        qc.cx(1, 2)
        qc.h(1)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.h(1)
        qc.x(2)
        pass_manager = PassManager()
        pass_manager.append(CollectMultiQBlocks())
        pass_manager.run(qc)
        self.assertTrue(len(pass_manager.property_set['block_list']) == 2)

    def test_ignore_measurement(self):
        if False:
            i = 10
            return i + 15
        'Test that doing a measurement on one qubit will not prevent\n        gates from being added to the block that do not act on the qubit\n        that was measured\n                       ┌─┐\n        q_0: ──■───────┤M├──────────\n             ┌─┴─┐     └╥┘     ┌───┐\n        q_1: ┤ X ├──■───╫───■──┤ X ├\n             └───┘┌─┴─┐ ║ ┌─┴─┐├───┤\n        q_2: ─────┤ X ├─╫─┤ X ├┤ H ├\n                  └───┘ ║ └───┘└───┘\n        c_0: ═══════════╩═══════════\n        '
        qc = QuantumCircuit(3, 1)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure(0, 0)
        qc.cx(1, 2)
        qc.x(1)
        qc.h(2)
        pass_manager = PassManager()
        pass_manager.append(CollectMultiQBlocks(max_block_size=3))
        pass_manager.run(qc)
        self.assertTrue(len(pass_manager.property_set['block_list']) == 1)

    def test_larger_blocks(self):
        if False:
            return 10
        'Test that a max block size of 4 is still being processed\n        reasonably. Currently, this test just makes sure that the circuit can be run.\n        This is because the current multiqubit block collector is not optimal for this case\n        q_0: ──■──────────────■───────\n             ┌─┴─┐            │\n        q_1: ┤ X ├──■─────────■───────\n             └───┘┌─┴─┐     ┌─┴─┐\n        q_2: ─────┤ X ├──■──┤ X ├─────\n                  └───┘┌─┴─┐└───┘\n        q_3: ──────────┤ X ├──■────■──\n                       └───┘┌─┴─┐┌─┴─┐\n        q_4: ───────────────┤ X ├┤ X ├\n                            └───┘└───┘\n        '
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.ccx(0, 1, 2)
        qc.cx(3, 4)
        pass_manager = PassManager()
        pass_manager.append(CollectMultiQBlocks(max_block_size=4))
        pass_manager.run(qc)
if __name__ == '__main__':
    unittest.main()