"""Code from commutative_analysis pass that checks commutation relations between DAG nodes."""
from functools import lru_cache
from typing import List
import numpy as np
from qiskit.circuit.operation import Operation
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.quantum_info.operators import Operator

@lru_cache(maxsize=None)
def _identity_op(num_qubits):
    if False:
        i = 10
        return i + 15
    'Cached identity matrix'
    return Operator(np.eye(2 ** num_qubits), input_dims=(2,) * num_qubits, output_dims=(2,) * num_qubits)

class CommutationChecker:
    """This code is essentially copy-pasted from commutative_analysis.py.
    This code cleverly hashes commutativity and non-commutativity results between DAG nodes and seems
    quite efficient for large Clifford circuits.
    They may be other possible efficiency improvements: using rule-based commutativity analysis,
    evicting from the cache less useful entries, etc.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.cache = {}

    def _hashable_parameters(self, params):
        if False:
            while True:
                i = 10
        'Convert the parameters of a gate into a hashable format for lookup in a dictionary.\n\n        This aims to be fast in common cases, and is not intended to work outside of the lifetime of a\n        single commutation pass; it does not handle mutable state correctly if the state is actually\n        changed.'
        try:
            hash(params)
            return params
        except TypeError:
            pass
        if isinstance(params, (list, tuple)):
            return tuple((self._hashable_parameters(x) for x in params))
        if isinstance(params, np.ndarray):
            return (np.ndarray, id(params))
        return ('fallback', str(params))

    def commute(self, op1: Operation, qargs1: List, cargs1: List, op2: Operation, qargs2: List, cargs2: List, max_num_qubits: int=3) -> bool:
        if False:
            i = 10
            return i + 15
        "\n        Checks if two Operations commute. The return value of `True` means that the operations\n        truly commute, and the return value of `False` means that either the operations do not\n        commute or that the commutation check was skipped (for example, when the operations\n        have conditions or have too many qubits).\n\n        Args:\n            op1: first operation.\n            qargs1: first operation's qubits.\n            cargs1: first operation's clbits.\n            op2: second operation.\n            qargs2: second operation's qubits.\n            cargs2: second operation's clbits.\n            max_num_qubits: the maximum number of qubits to consider, the check may be skipped if\n                the number of qubits for either operation exceeds this amount.\n\n        Returns:\n            bool: whether two operations commute.\n        "
        if getattr(op1, 'condition', None) is not None or getattr(op2, 'condition', None) is not None:
            return False
        if isinstance(op1, ControlFlowOp) or isinstance(op2, ControlFlowOp):
            return False
        intersection_q = set(qargs1).intersection(set(qargs2))
        intersection_c = set(cargs1).intersection(set(cargs2))
        if not (intersection_q or intersection_c):
            return True
        if len(qargs1) > max_num_qubits or len(qargs2) > max_num_qubits:
            return False
        for op in [op1, op2]:
            if getattr(op, '_directive', False) or op.name in {'measure', 'reset', 'delay'} or (getattr(op, 'is_parameterized', False) and op.is_parameterized()):
                return False
        qarg = {q: i for (i, q) in enumerate(qargs1)}
        num_qubits = len(qarg)
        for q in qargs2:
            if q not in qarg:
                qarg[q] = num_qubits
                num_qubits += 1
        qarg1 = tuple((qarg[q] for q in qargs1))
        qarg2 = tuple((qarg[q] for q in qargs2))
        node1_key = (op1.name, self._hashable_parameters(op1.params), qarg1)
        node2_key = (op2.name, self._hashable_parameters(op2.params), qarg2)
        try:
            return self.cache[node1_key, node2_key]
        except KeyError:
            pass
        operator_1 = Operator(op1, input_dims=(2,) * len(qarg1), output_dims=(2,) * len(qarg1))
        operator_2 = Operator(op2, input_dims=(2,) * len(qarg2), output_dims=(2,) * len(qarg2))
        if qarg1 == qarg2:
            op12 = operator_1.compose(operator_2)
            op21 = operator_2.compose(operator_1)
        else:
            extra_qarg2 = num_qubits - len(qarg1)
            if extra_qarg2:
                id_op = _identity_op(extra_qarg2)
                operator_1 = id_op.tensor(operator_1)
            op12 = operator_1.compose(operator_2, qargs=qarg2, front=False)
            op21 = operator_1.compose(operator_2, qargs=qarg2, front=True)
        self.cache[node1_key, node2_key] = self.cache[node2_key, node1_key] = ret = op12 == op21
        return ret