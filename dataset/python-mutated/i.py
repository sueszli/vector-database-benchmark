"""Identity gate."""
from typing import Optional
from qiskit.circuit.singleton import SingletonGate, stdlib_singleton_key
from qiskit.circuit._utils import with_gate_array

@with_gate_array([[1, 0], [0, 1]])
class IGate(SingletonGate):
    """Identity gate.

    Identity gate corresponds to a single-qubit gate wait cycle,
    and should not be optimized or unrolled (it is an opaque gate).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.i` and
    :meth:`~qiskit.circuit.QuantumCircuit.id` methods.

    **Matrix Representation:**

    .. math::

        I = \\begin{pmatrix}
                1 & 0 \\\\
                0 & 1
            \\end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::
             ┌───┐
        q_0: ┤ I ├
             └───┘
    """

    def __init__(self, label: Optional[str]=None, *, duration=None, unit='dt'):
        if False:
            print('Hello World!')
        'Create new Identity gate.'
        super().__init__('id', 1, [], label=label, duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key()

    def inverse(self):
        if False:
            while True:
                i = 10
        'Invert this gate.'
        return IGate()

    def power(self, exponent: float):
        if False:
            while True:
                i = 10
        'Raise gate to a power.'
        return IGate()