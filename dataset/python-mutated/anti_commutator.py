"""Anti commutator function."""
from __future__ import annotations
from typing import TypeVar
from qiskit.quantum_info.operators.linear_op import LinearOp
OperatorTypeT = TypeVar('OperatorTypeT', bound=LinearOp)

def anti_commutator(a: OperatorTypeT, b: OperatorTypeT) -> OperatorTypeT:
    if False:
        for i in range(10):
            print('nop')
    'Compute anti-commutator of a and b.\n\n    .. math::\n\n        ab + ba.\n\n    Args:\n        a: Operator a.\n        b: Operator b.\n    Returns:\n        The anti-commutator\n    '
    return a @ b + b @ a