"""Circuit utility functions"""
import numpy as np

def summarize_circuits(circuits):
    if False:
        return 10
    'Summarize circuits based on QuantumCircuit, and five metrics are summarized.\n        - Number of qubits\n        - Number of classical bits\n        - Number of operations\n        - Depth of circuits\n        - Counts of different gate operations\n\n    The average statistic of the first four is provided if multiple circuits are provided.\n\n    Args:\n        circuits (QuantumCircuit or [QuantumCircuit]): the to-be-summarized circuits\n\n    Returns:\n        str: a formatted string records the summary\n    '
    if not isinstance(circuits, list):
        circuits = [circuits]
    ret = ''
    ret += f'Submitting {len(circuits)} circuits.\n'
    ret += '============================================================================\n'
    stats = np.zeros(4)
    for (i, circuit) in enumerate(circuits):
        depth = circuit.depth()
        size = circuit.size()
        num_qubits = sum((reg.size for reg in circuit.qregs))
        num_clbits = sum((reg.size for reg in circuit.cregs))
        op_counts = circuit.count_ops()
        stats[0] += num_qubits
        stats[1] += num_clbits
        stats[2] += size
        stats[3] += depth
        ret = ''.join([ret, '{}-th circuit: {} qubits, {} classical bits and {} operations with depth {}\nop_counts: {}\n'.format(i, num_qubits, num_clbits, size, depth, op_counts)])
    if len(circuits) > 1:
        stats /= len(circuits)
        ret = ''.join([ret, 'Average: {:.2f} qubits, {:.2f} classical bits and {:.2f} operations with depth {:.2f}\n'.format(stats[0], stats[1], stats[2], stats[3])])
    ret += '============================================================================\n'
    return ret