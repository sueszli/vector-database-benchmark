"""Test functionality to collect, split and consolidate blocks from DAGCircuits."""
import unittest
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.converters import circuit_to_dag, circuit_to_dagdependency, circuit_to_instruction, dag_to_circuit, dagdependency_to_circuit
from qiskit.test import QiskitTestCase
from qiskit.circuit import QuantumCircuit, Measure, Clbit
from qiskit.dagcircuit.collect_blocks import BlockCollector, BlockSplitter, BlockCollapser

class TestCollectBlocks(QiskitTestCase):
    """Tests to verify correctness of collecting, splitting, and consolidating blocks
    from DAGCircuit and DAGDependency. Additional tests appear as a part of
    CollectLinearFunctions and CollectCliffords passes.
    """

    def test_collect_gates_from_dagcircuit_1(self):
        if False:
            for i in range(10):
                print('nop')
        'Test collecting CX gates from DAGCircuits.'
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.z(0)
        qc.cx(0, 3)
        qc.cx(0, 4)
        block_collector = BlockCollector(circuit_to_dag(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name == 'cx', split_blocks=False, min_block_size=2)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 2)
        self.assertEqual(len(blocks[1]), 2)

    def test_collect_gates_from_dagcircuit_2(self):
        if False:
            print('Hello World!')
        'Test collecting both CX and Z gates from DAGCircuits.'
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.z(0)
        qc.cx(0, 3)
        qc.cx(0, 4)
        block_collector = BlockCollector(circuit_to_dag(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['cx', 'z'], split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 5)

    def test_collect_gates_from_dagcircuit_3(self):
        if False:
            return 10
        'Test collecting CX gates from DAGCircuits.'
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.z(0)
        qc.cx(1, 3)
        qc.cx(0, 3)
        qc.cx(0, 4)
        block_collector = BlockCollector(circuit_to_dag(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['cx'], split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 2)

    def test_collect_gates_from_dagdependency_1(self):
        if False:
            return 10
        'Test collecting CX gates from DAGDependency.'
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.z(0)
        qc.cx(0, 3)
        qc.cx(0, 4)
        block_collector = BlockCollector(circuit_to_dagdependency(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name == 'cx', split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 4)

    def test_collect_gates_from_dagdependency_2(self):
        if False:
            print('Hello World!')
        'Test collecting both CX and Z gates from DAGDependency.'
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.z(0)
        qc.cx(0, 3)
        qc.cx(0, 4)
        block_collector = BlockCollector(circuit_to_dagdependency(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['cx', 'z'], split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 5)

    def test_collect_and_split_gates_from_dagcircuit(self):
        if False:
            for i in range(10):
                print('nop')
        'Test collecting and splitting blocks from DAGCircuit.'
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(3, 5)
        qc.cx(2, 4)
        qc.swap(1, 0)
        qc.cz(5, 3)
        block_collector = BlockCollector(circuit_to_dag(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: True, split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 5)
        split_blocks = BlockSplitter().run(blocks[0])
        self.assertEqual(len(split_blocks), 3)

    def test_collect_and_split_gates_from_dagdependency(self):
        if False:
            print('Hello World!')
        'Test collecting and splitting blocks from DAGDependecy.'
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(3, 5)
        qc.cx(2, 4)
        qc.swap(1, 0)
        qc.cz(5, 3)
        block_collector = BlockCollector(circuit_to_dagdependency(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: True, split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 5)
        split_blocks = BlockSplitter().run(blocks[0])
        self.assertEqual(len(split_blocks), 3)

    def test_circuit_has_measure(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that block collection works properly when there is a measure in the\n        middle of the circuit.'
        qc = QuantumCircuit(2, 1)
        qc.cx(1, 0)
        qc.x(0)
        qc.x(1)
        qc.measure(0, 0)
        qc.x(0)
        qc.cx(1, 0)
        block_collector = BlockCollector(circuit_to_dag(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['x', 'cx'], split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 3)
        self.assertEqual(len(blocks[1]), 2)

    def test_circuit_has_measure_dagdependency(self):
        if False:
            while True:
                i = 10
        'Test that block collection works properly when there is a measure in the\n        middle of the circuit.'
        qc = QuantumCircuit(2, 1)
        qc.cx(1, 0)
        qc.x(0)
        qc.x(1)
        qc.measure(0, 0)
        qc.x(0)
        qc.cx(1, 0)
        block_collector = BlockCollector(circuit_to_dagdependency(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['x', 'cx'], split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 3)
        self.assertEqual(len(blocks[1]), 2)

    def test_circuit_has_conditional_gates(self):
        if False:
            return 10
        'Test that block collection works properly when there the circuit\n        contains conditional gates.'
        qc = QuantumCircuit(2, 1)
        qc.x(0)
        qc.x(1)
        qc.cx(1, 0)
        qc.x(1).c_if(0, 1)
        qc.x(0)
        qc.x(1)
        qc.cx(0, 1)
        block_collector = BlockCollector(circuit_to_dag(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['x', 'cx'], split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 7)
        block_collector = BlockCollector(circuit_to_dag(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['x', 'cx'] and (not getattr(node.op, 'condition', None)), split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 4)
        self.assertEqual(len(blocks[1]), 2)

    def test_circuit_has_conditional_gates_dagdependency(self):
        if False:
            return 10
        'Test that block collection works properly when there the circuit\n        contains conditional gates.'
        qc = QuantumCircuit(2, 1)
        qc.x(0)
        qc.x(1)
        qc.cx(1, 0)
        qc.x(1).c_if(0, 1)
        qc.x(0)
        qc.x(1)
        qc.cx(0, 1)
        block_collector = BlockCollector(circuit_to_dagdependency(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['x', 'cx'], split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 7)
        block_collector = BlockCollector(circuit_to_dag(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['x', 'cx'] and (not getattr(node.op, 'condition', None)), split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 4)
        self.assertEqual(len(blocks[1]), 2)

    def test_multiple_collection_methods(self):
        if False:
            i = 10
            return i + 15
        'Test that block collection allows to collect blocks using several different\n        filter functions.'
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.swap(1, 4)
        qc.swap(4, 3)
        qc.z(0)
        qc.z(1)
        qc.z(2)
        qc.z(3)
        qc.z(4)
        qc.swap(3, 4)
        qc.cx(0, 3)
        qc.cx(0, 4)
        block_collector = BlockCollector(circuit_to_dag(qc))
        linear_blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['cx', 'swap'], split_blocks=False, min_block_size=1)
        cx_blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['cx'], split_blocks=False, min_block_size=1)
        swapz_blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['swap', 'z'], split_blocks=False, min_block_size=1)
        self.assertEqual(len(linear_blocks), 2)
        self.assertEqual(len(linear_blocks[0]), 4)
        self.assertEqual(len(linear_blocks[1]), 3)
        self.assertEqual(len(cx_blocks), 2)
        self.assertEqual(len(cx_blocks[0]), 2)
        self.assertEqual(len(cx_blocks[1]), 2)
        self.assertEqual(len(swapz_blocks), 1)
        self.assertEqual(len(swapz_blocks[0]), 8)

    def test_min_block_size(self):
        if False:
            print('Hello World!')
        'Test that the option min_block_size for collecting blocks works correctly.'
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.cx(0, 1)
        block_collector = BlockCollector(circuit_to_dag(circuit))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['cx', 'swap'], split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 3)
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['cx', 'swap'], split_blocks=False, min_block_size=2)
        self.assertEqual(len(blocks), 2)
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['cx', 'swap'], split_blocks=False, min_block_size=3)
        self.assertEqual(len(blocks), 1)
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['cx', 'swap'], split_blocks=False, min_block_size=4)
        self.assertEqual(len(blocks), 0)

    def test_split_blocks(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that splitting blocks of nodes into sub-blocks works correctly.'
        circuit = QuantumCircuit(5)
        circuit.cx(0, 2)
        circuit.cx(1, 4)
        circuit.cx(2, 0)
        circuit.cx(0, 3)
        circuit.swap(3, 2)
        circuit.swap(4, 1)
        block_collector = BlockCollector(circuit_to_dag(circuit))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['cx', 'swap'], split_blocks=False, min_block_size=2)
        self.assertEqual(len(blocks), 1)
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['cx', 'swap'], split_blocks=True, min_block_size=2)
        self.assertEqual(len(blocks), 2)

    def test_do_not_split_blocks(self):
        if False:
            while True:
                i = 10
        'Test that splitting blocks of nodes into sub-blocks works correctly.'
        circuit = QuantumCircuit(5)
        circuit.cx(0, 3)
        circuit.cx(0, 2)
        circuit.cx(1, 4)
        circuit.swap(4, 2)
        block_collector = BlockCollector(circuit_to_dagdependency(circuit))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['cx', 'swap'], split_blocks=True, min_block_size=1)
        self.assertEqual(len(blocks), 1)

    def test_collect_blocks_with_cargs(self):
        if False:
            return 10
        'Test collecting and collapsing blocks with classical bits appearing as cargs.'
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.measure_all()
        dag = circuit_to_dag(qc)
        blocks = BlockCollector(dag).collect_all_matching_blocks(lambda node: isinstance(node.op, Measure), split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 3)
        self.assertEqual(blocks[0][0].op, Measure())
        self.assertEqual(blocks[0][1].op, Measure())
        self.assertEqual(blocks[0][2].op, Measure())

        def _collapse_fn(circuit):
            if False:
                print('Hello World!')
            op = circuit_to_instruction(circuit)
            op.name = 'COLLAPSED'
            return op
        dag = BlockCollapser(dag).collapse_to_operation(blocks, _collapse_fn)
        collapsed_qc = dag_to_circuit(dag)
        self.assertEqual(len(collapsed_qc.data), 5)
        self.assertEqual(collapsed_qc.data[0].operation.name, 'h')
        self.assertEqual(collapsed_qc.data[1].operation.name, 'h')
        self.assertEqual(collapsed_qc.data[2].operation.name, 'h')
        self.assertEqual(collapsed_qc.data[3].operation.name, 'barrier')
        self.assertEqual(collapsed_qc.data[4].operation.name, 'COLLAPSED')
        self.assertEqual(collapsed_qc.data[4].operation.definition.num_qubits, 3)
        self.assertEqual(collapsed_qc.data[4].operation.definition.num_clbits, 3)

    def test_collect_blocks_with_cargs_dagdependency(self):
        if False:
            while True:
                i = 10
        'Test collecting and collapsing blocks with classical bits appearing as cargs,\n        using DAGDependency.'
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.measure_all()
        dag = circuit_to_dagdependency(qc)
        blocks = BlockCollector(dag).collect_all_matching_blocks(lambda node: isinstance(node.op, Measure), split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 3)
        self.assertEqual(blocks[0][0].op, Measure())
        self.assertEqual(blocks[0][1].op, Measure())
        self.assertEqual(blocks[0][2].op, Measure())

        def _collapse_fn(circuit):
            if False:
                while True:
                    i = 10
            op = circuit_to_instruction(circuit)
            op.name = 'COLLAPSED'
            return op
        dag = BlockCollapser(dag).collapse_to_operation(blocks, _collapse_fn)
        collapsed_qc = dagdependency_to_circuit(dag)
        self.assertEqual(len(collapsed_qc.data), 5)
        self.assertEqual(collapsed_qc.data[0].operation.name, 'h')
        self.assertEqual(collapsed_qc.data[1].operation.name, 'h')
        self.assertEqual(collapsed_qc.data[2].operation.name, 'h')
        self.assertEqual(collapsed_qc.data[3].operation.name, 'barrier')
        self.assertEqual(collapsed_qc.data[4].operation.name, 'COLLAPSED')
        self.assertEqual(collapsed_qc.data[4].operation.definition.num_qubits, 3)
        self.assertEqual(collapsed_qc.data[4].operation.definition.num_clbits, 3)

    def test_collect_blocks_with_clbits(self):
        if False:
            while True:
                i = 10
        'Test collecting and collapsing blocks with classical bits appearing under\n        condition.'
        qc = QuantumCircuit(4, 3)
        qc.cx(0, 1).c_if(0, 1)
        qc.cx(2, 3)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.cx(2, 3).c_if(1, 0)
        dag = circuit_to_dag(qc)
        blocks = BlockCollector(dag).collect_all_matching_blocks(lambda node: node.op.name == 'cx', split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 5)

        def _collapse_fn(circuit):
            if False:
                i = 10
                return i + 15
            op = circuit_to_instruction(circuit)
            op.name = 'COLLAPSED'
            return op
        dag = BlockCollapser(dag).collapse_to_operation(blocks, _collapse_fn)
        collapsed_qc = dag_to_circuit(dag)
        self.assertEqual(len(collapsed_qc.data), 1)
        self.assertEqual(collapsed_qc.data[0].operation.name, 'COLLAPSED')
        self.assertEqual(collapsed_qc.data[0].operation.definition.num_qubits, 4)
        self.assertEqual(collapsed_qc.data[0].operation.definition.num_clbits, 2)

    def test_collect_blocks_with_clbits_dagdependency(self):
        if False:
            print('Hello World!')
        'Test collecting and collapsing blocks with classical bits appearing\n        under conditions, using DAGDependency.'
        qc = QuantumCircuit(4, 3)
        qc.cx(0, 1).c_if(0, 1)
        qc.cx(2, 3)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.cx(2, 3).c_if(1, 0)
        dag = circuit_to_dagdependency(qc)
        blocks = BlockCollector(dag).collect_all_matching_blocks(lambda node: node.op.name == 'cx', split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 5)

        def _collapse_fn(circuit):
            if False:
                i = 10
                return i + 15
            op = circuit_to_instruction(circuit)
            op.name = 'COLLAPSED'
            return op
        dag = BlockCollapser(dag).collapse_to_operation(blocks, _collapse_fn)
        collapsed_qc = dagdependency_to_circuit(dag)
        self.assertEqual(len(collapsed_qc.data), 1)
        self.assertEqual(collapsed_qc.data[0].operation.name, 'COLLAPSED')
        self.assertEqual(collapsed_qc.data[0].operation.definition.num_qubits, 4)
        self.assertEqual(collapsed_qc.data[0].operation.definition.num_clbits, 2)

    def test_collect_blocks_with_clbits2(self):
        if False:
            while True:
                i = 10
        'Test collecting and collapsing blocks with classical bits appearing under\n        condition.'
        qreg = QuantumRegister(4, 'qr')
        creg = ClassicalRegister(3, 'cr')
        cbit = Clbit()
        qc = QuantumCircuit(qreg, creg, [cbit])
        qc.cx(0, 1).c_if(creg[1], 1)
        qc.cx(2, 3).c_if(cbit, 0)
        qc.cx(1, 2)
        qc.cx(0, 1).c_if(creg[2], 1)
        dag = circuit_to_dag(qc)
        blocks = BlockCollector(dag).collect_all_matching_blocks(lambda node: node.op.name == 'cx', split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 4)

        def _collapse_fn(circuit):
            if False:
                print('Hello World!')
            op = circuit_to_instruction(circuit)
            op.name = 'COLLAPSED'
            return op
        dag = BlockCollapser(dag).collapse_to_operation(blocks, _collapse_fn)
        collapsed_qc = dag_to_circuit(dag)
        self.assertEqual(len(collapsed_qc.data), 1)
        self.assertEqual(collapsed_qc.data[0].operation.name, 'COLLAPSED')
        self.assertEqual(collapsed_qc.data[0].operation.definition.num_qubits, 4)
        self.assertEqual(collapsed_qc.data[0].operation.definition.num_clbits, 3)

    def test_collect_blocks_with_clbits2_dagdependency(self):
        if False:
            print('Hello World!')
        'Test collecting and collapsing blocks with classical bits appearing under\n        condition, using DAGDependency.'
        qreg = QuantumRegister(4, 'qr')
        creg = ClassicalRegister(3, 'cr')
        cbit = Clbit()
        qc = QuantumCircuit(qreg, creg, [cbit])
        qc.cx(0, 1).c_if(creg[1], 1)
        qc.cx(2, 3).c_if(cbit, 0)
        qc.cx(1, 2)
        qc.cx(0, 1).c_if(creg[2], 1)
        dag = circuit_to_dag(qc)
        blocks = BlockCollector(dag).collect_all_matching_blocks(lambda node: node.op.name == 'cx', split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 4)

        def _collapse_fn(circuit):
            if False:
                for i in range(10):
                    print('nop')
            op = circuit_to_instruction(circuit)
            op.name = 'COLLAPSED'
            return op
        dag = BlockCollapser(dag).collapse_to_operation(blocks, _collapse_fn)
        collapsed_qc = dag_to_circuit(dag)
        self.assertEqual(len(collapsed_qc.data), 1)
        self.assertEqual(collapsed_qc.data[0].operation.name, 'COLLAPSED')
        self.assertEqual(collapsed_qc.data[0].operation.definition.num_qubits, 4)
        self.assertEqual(collapsed_qc.data[0].operation.definition.num_clbits, 3)

    def test_collect_blocks_with_cregs(self):
        if False:
            while True:
                i = 10
        'Test collecting and collapsing blocks with classical registers appearing under\n        condition.'
        qreg = QuantumRegister(4, 'qr')
        creg = ClassicalRegister(3, 'cr')
        creg2 = ClassicalRegister(2, 'cr2')
        qc = QuantumCircuit(qreg, creg, creg2)
        qc.cx(0, 1).c_if(creg, 3)
        qc.cx(1, 2)
        qc.cx(0, 1).c_if(creg[2], 1)
        dag = circuit_to_dag(qc)
        blocks = BlockCollector(dag).collect_all_matching_blocks(lambda node: node.op.name == 'cx', split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 3)

        def _collapse_fn(circuit):
            if False:
                while True:
                    i = 10
            op = circuit_to_instruction(circuit)
            op.name = 'COLLAPSED'
            return op
        dag = BlockCollapser(dag).collapse_to_operation(blocks, _collapse_fn)
        collapsed_qc = dag_to_circuit(dag)
        self.assertEqual(len(collapsed_qc.data), 1)
        self.assertEqual(collapsed_qc.data[0].operation.name, 'COLLAPSED')
        self.assertEqual(len(collapsed_qc.data[0].operation.definition.cregs), 1)
        self.assertEqual(collapsed_qc.data[0].operation.definition.num_qubits, 3)
        self.assertEqual(collapsed_qc.data[0].operation.definition.num_clbits, 3)

    def test_collect_blocks_with_cregs_dagdependency(self):
        if False:
            return 10
        'Test collecting and collapsing blocks with classical registers appearing under\n        condition, using DAGDependency.'
        qreg = QuantumRegister(4, 'qr')
        creg = ClassicalRegister(3, 'cr')
        creg2 = ClassicalRegister(2, 'cr2')
        qc = QuantumCircuit(qreg, creg, creg2)
        qc.cx(0, 1).c_if(creg, 3)
        qc.cx(1, 2)
        qc.cx(0, 1).c_if(creg[2], 1)
        dag = circuit_to_dagdependency(qc)
        blocks = BlockCollector(dag).collect_all_matching_blocks(lambda node: node.op.name == 'cx', split_blocks=False, min_block_size=1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 3)

        def _collapse_fn(circuit):
            if False:
                return 10
            op = circuit_to_instruction(circuit)
            op.name = 'COLLAPSED'
            return op
        dag = BlockCollapser(dag).collapse_to_operation(blocks, _collapse_fn)
        collapsed_qc = dagdependency_to_circuit(dag)
        self.assertEqual(len(collapsed_qc.data), 1)
        self.assertEqual(collapsed_qc.data[0].operation.name, 'COLLAPSED')
        self.assertEqual(len(collapsed_qc.data[0].operation.definition.cregs), 1)
        self.assertEqual(collapsed_qc.data[0].operation.definition.num_qubits, 3)
        self.assertEqual(collapsed_qc.data[0].operation.definition.num_clbits, 3)

    def test_collect_blocks_backwards_dagcircuit(self):
        if False:
            for i in range(10):
                print('nop')
        'Test collecting H gates from DAGCircuit in the forward vs. the reverse\n        directions.'
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.h(3)
        qc.cx(1, 2)
        qc.z(0)
        qc.z(1)
        qc.z(2)
        qc.z(3)
        block_collector = BlockCollector(circuit_to_dag(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['h', 'z'], split_blocks=False, min_block_size=1, collect_from_back=False)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 6)
        self.assertEqual(len(blocks[1]), 2)
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['h', 'z'], split_blocks=False, min_block_size=1, collect_from_back=True)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 2)
        self.assertEqual(len(blocks[1]), 6)

    def test_collect_blocks_backwards_dagdependency(self):
        if False:
            return 10
        'Test collecting H gates from DAGDependency in the forward vs. the reverse\n        directions.'
        qc = QuantumCircuit(4)
        qc.z(0)
        qc.z(1)
        qc.z(2)
        qc.z(3)
        qc.cx(1, 2)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.h(3)
        block_collector = BlockCollector(circuit_to_dagdependency(qc))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['h', 'z'], split_blocks=False, min_block_size=1, collect_from_back=False)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 6)
        self.assertEqual(len(blocks[1]), 2)
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['h', 'z'], split_blocks=False, min_block_size=1, collect_from_back=True)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 1)
        self.assertEqual(len(blocks[1]), 7)

    def test_split_layers_dagcircuit(self):
        if False:
            i = 10
            return i + 15
        'Test that splitting blocks of nodes into layers works correctly.'
        circuit = QuantumCircuit(5)
        circuit.cx(0, 2)
        circuit.cx(1, 4)
        circuit.cx(2, 0)
        circuit.cx(0, 3)
        circuit.swap(3, 2)
        circuit.swap(4, 1)
        block_collector = BlockCollector(circuit_to_dag(circuit))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['cx', 'swap'], split_blocks=False, min_block_size=1, split_layers=True)
        self.assertEqual(len(blocks), 4)
        self.assertEqual(len(blocks[0]), 2)
        self.assertEqual(len(blocks[1]), 2)
        self.assertEqual(len(blocks[2]), 1)
        self.assertEqual(len(blocks[3]), 1)

    def test_split_layers_dagdependency(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that splitting blocks of nodes into layers works correctly.'
        circuit = QuantumCircuit(5)
        circuit.cx(0, 2)
        circuit.cx(1, 4)
        circuit.cx(2, 0)
        circuit.cx(0, 3)
        circuit.swap(3, 2)
        circuit.swap(4, 1)
        block_collector = BlockCollector(circuit_to_dagdependency(circuit))
        blocks = block_collector.collect_all_matching_blocks(lambda node: node.op.name in ['cx', 'swap'], split_blocks=False, min_block_size=1, split_layers=True)
        self.assertEqual(len(blocks), 4)
        self.assertEqual(len(blocks[0]), 2)
        self.assertEqual(len(blocks[1]), 2)
        self.assertEqual(len(blocks[2]), 1)
        self.assertEqual(len(blocks[3]), 1)

    def test_block_collapser_register_condition(self):
        if False:
            i = 10
            return i + 15
        'Test that BlockCollapser can handle a register being used more than once.'
        qc = QuantumCircuit(1, 2)
        qc.x(0).c_if(qc.cregs[0], 0)
        qc.y(0).c_if(qc.cregs[0], 1)
        dag = circuit_to_dag(qc)
        blocks = BlockCollector(dag).collect_all_matching_blocks(lambda _: True, split_blocks=False, min_block_size=1)
        dag = BlockCollapser(dag).collapse_to_operation(blocks, lambda circ: circ.to_instruction())
        collapsed_qc = dag_to_circuit(dag)
        self.assertEqual(len(collapsed_qc.data), 1)
        self.assertEqual(collapsed_qc.data[0].operation.definition.num_qubits, 1)
        self.assertEqual(collapsed_qc.data[0].operation.definition.num_clbits, 2)
if __name__ == '__main__':
    unittest.main()