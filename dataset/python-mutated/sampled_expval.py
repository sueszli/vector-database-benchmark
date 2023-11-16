"""Routines for computing expectation values from sampled distributions"""
import numpy as np
from qiskit._accelerate.sampled_exp_val import sampled_expval_float, sampled_expval_complex
from qiskit.exceptions import QiskitError
from .distributions import QuasiDistribution, ProbDistribution
OPERS = {'Z', 'I', '0', '1'}

def sampled_expectation_value(dist, oper):
    if False:
        while True:
            i = 10
    'Computes expectation value from a sampled distribution\n\n    Note that passing a raw dict requires bit-string keys.\n\n    Parameters:\n        dist (Counts or QuasiDistribution or ProbDistribution or dict): Input sampled distribution\n        oper (str or Pauli or PauliOp or PauliSumOp or SparsePauliOp): The operator for\n                                                                       the observable\n\n    Returns:\n        float: The expectation value\n    Raises:\n        QiskitError: if the input distribution or operator is an invalid type\n    '
    from .counts import Counts
    from qiskit.quantum_info import Pauli, SparsePauliOp
    from qiskit.opflow import PauliOp, PauliSumOp
    if isinstance(dist, (QuasiDistribution, ProbDistribution)):
        dist = dist.binary_probabilities()
    if not isinstance(dist, (Counts, dict)):
        raise QiskitError('Invalid input distribution type')
    if isinstance(oper, str):
        oper_strs = [oper.upper()]
        coeffs = np.asarray([1.0])
    elif isinstance(oper, Pauli):
        oper_strs = [oper.to_label()]
        coeffs = np.asarray([1.0])
    elif isinstance(oper, PauliOp):
        oper_strs = [oper.primitive.to_label()]
        coeffs = np.asarray([1.0])
    elif isinstance(oper, PauliSumOp):
        spo = oper.primitive
        oper_strs = spo.paulis.to_labels()
        coeffs = np.asarray(spo.coeffs) * oper.coeff
    elif isinstance(oper, SparsePauliOp):
        oper_strs = oper.paulis.to_labels()
        coeffs = np.asarray(oper.coeffs)
    else:
        raise QiskitError('Invalid operator type')
    bitstring_len = len(next(iter(dist)))
    if any((len(op) != bitstring_len for op in oper_strs)):
        raise QiskitError(f'One or more operators not same length ({bitstring_len}) as input bitstrings')
    for op in oper_strs:
        if set(op).difference(OPERS):
            raise QiskitError(f'Input operator {op} is not diagonal')
    if coeffs.dtype == np.dtype(complex).type:
        return sampled_expval_complex(oper_strs, coeffs, dist)
    else:
        return sampled_expval_float(oper_strs, coeffs, dist)