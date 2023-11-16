"""Double commutator function."""
from __future__ import annotations
from typing import TypeVar
from qiskit.quantum_info.operators.linear_op import LinearOp
OperatorTypeT = TypeVar('OperatorTypeT', bound=LinearOp)

def double_commutator(a: OperatorTypeT, b: OperatorTypeT, c: OperatorTypeT, *, commutator: bool=True) -> OperatorTypeT:
    if False:
        i = 10
        return i + 15
    'Compute symmetric double commutator of a, b and c.\n\n    See also Equation (13.6.18) in [1].\n\n    If `commutator` is `True`, it returns\n\n    .. math::\n\n         [[A, B], C]/2 + [A, [B, C]]/2\n         = (2ABC + 2CBA - BAC - CAB - ACB - BCA)/2.\n\n    If `commutator` is `False`, it returns\n\n    .. math::\n         \\lbrace[A, B], C\\rbrace/2 + \\lbrace A, [B, C]\\rbrace/2\n         = (2ABC - 2CBA - BAC + CAB - ACB + BCA)/2.\n\n    Args:\n        a: Operator a.\n        b: Operator b.\n        c: Operator c.\n        commutator: If ``True`` compute the double commutator,\n            if ``False`` the double anti-commutator.\n\n    Returns:\n        The double commutator\n\n    References:\n\n        [1]: R. McWeeny.\n            Methods of Molecular Quantum Mechanics.\n            2nd Edition, Academic Press, 1992.\n            ISBN 0-12-486552-6.\n    '
    sign_num = -1 if commutator else 1
    ab = a @ b
    ba = b @ a
    ac = a @ c
    ca = c @ a
    abc = ab @ c
    cba = c @ ba
    bac = ba @ c
    cab = c @ ab
    acb = ac @ b
    bca = b @ ca
    res = abc - sign_num * cba + 0.5 * (-bac + sign_num * cab - acb + sign_num * bca)
    return res