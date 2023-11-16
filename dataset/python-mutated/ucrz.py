"""Uniformly controlled Pauli-Z rotations."""
from __future__ import annotations
from .uc_pauli_rot import UCPauliRotGate

class UCRZGate(UCPauliRotGate):
    """Uniformly controlled Pauli-Z rotations.

    Implements the :class:`.UCGate` for the special case that all unitaries are Pauli-Z rotations,
    :math:`U_i = R_Z(a_i)` where :math:`a_i \\in \\mathbb{R}` is the rotation angle.
    """

    def __init__(self, angle_list: list[float]):
        if False:
            print('Hello World!')
        '\n        Args:\n            angle_list: List of rotation angles :math:`[a_0, ..., a_{2^{k-1}}]`.\n        '
        super().__init__(angle_list, 'Z')