"""Global Phase Gate"""
from typing import Optional
import numpy
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType

class GlobalPhaseGate(Gate):
    """The global phase gate (:math:`e^{i\\theta}`).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`

    **Mathematical Representation:**

    .. math::
        \\text{GlobalPhaseGate}\\ =
            \\begin{pmatrix}
                e^{i\\theta}
            \\end{pmatrix}
    """

    def __init__(self, phase: ParameterValueType, label: Optional[str]=None, *, duration=None, unit='dt'):
        if False:
            print('Hello World!')
        '\n        Args:\n            phase: The value of phase it takes.\n            label: An optional label for the gate.\n        '
        super().__init__('global_phase', 0, [phase], label=label, duration=duration, unit=unit)

    def _define(self):
        if False:
            while True:
                i = 10
        q = QuantumRegister(0, 'q')
        qc = QuantumCircuit(q, name=self.name, global_phase=self.params[0])
        self.definition = qc

    def inverse(self):
        if False:
            return 10
        'Return inverted GLobalPhaseGate gate.\n\n        :math:`\\text{GlobalPhaseGate}(\\lambda)^{\\dagger} = \\text{GlobalPhaseGate}(-\\lambda)`\n        '
        return GlobalPhaseGate(-self.params[0])

    def __array__(self, dtype=complex):
        if False:
            return 10
        'Return a numpy.array for the global_phase gate.'
        theta = self.params[0]
        return numpy.array([[numpy.exp(1j * theta)]], dtype=dtype)