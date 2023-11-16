"""Converts any block of 2 qubit gates into a matrix."""
from qiskit.quantum_info import Operator
from qiskit.exceptions import QiskitError
from qiskit._accelerate.convert_2q_block_matrix import blocks_to_matrix

def _block_to_matrix(block, block_index_map):
    if False:
        for i in range(10):
            print('nop')
    '\n    The function converts any sequence of operations between two qubits into a matrix\n    that can be utilized to create a gate or a unitary.\n\n    Args:\n        block (List(DAGOpNode)): A block of operations on two qubits.\n        block_index_map (dict(Qubit, int)): The mapping of the qubit indices in the main circuit.\n\n    Returns:\n        NDArray: Matrix representation of the block of operations.\n    '
    op_list = []
    block_index_length = len(block_index_map)
    if block_index_length != 2:
        raise QiskitError('This function can only operate with blocks of 2 qubits.' + f'This block had {block_index_length}')
    for node in block:
        try:
            current = node.op.to_matrix()
        except QiskitError:
            current = Operator(node.op).data
        q_list = [block_index_map[qubit] for qubit in node.qargs]
        op_list.append((current, q_list))
    matrix = blocks_to_matrix(op_list)
    return matrix