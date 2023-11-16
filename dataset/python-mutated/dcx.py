"""Double-CNOT gate."""
from qiskit.circuit.singleton import SingletonGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array

@with_gate_array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
class DCXGate(SingletonGate):
    """Double-CNOT gate.

    A 2-qubit Clifford gate consisting of two back-to-back
    CNOTs with alternate controls.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.dcx` method.

    .. parsed-literal::
                  ┌───┐
        q_0: ──■──┤ X ├
             ┌─┴─┐└─┬─┘
        q_1: ┤ X ├──■──
             └───┘

    This is a classical logic gate, equivalent to a CNOT-SWAP (CNS) sequence,
    and locally equivalent to an iSWAP.

    .. math::

        DCX\\ q_0, q_1 =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 1 \\\\
                0 & 1 & 0 & 0 \\\\
                0 & 0 & 1 & 0
            \\end{pmatrix}
    """

    def __init__(self, label=None, *, duration=None, unit='dt'):
        if False:
            for i in range(10):
                print('nop')
        'Create new DCX gate.'
        super().__init__('dcx', 2, [], label=label, duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        if False:
            i = 10
            return i + 15
        '\n        gate dcx a, b { cx a, b; cx b, a; }\n        '
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(CXGate(), [q[0], q[1]], []), (CXGate(), [q[1], q[0]], [])]
        for (instr, qargs, cargs) in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc