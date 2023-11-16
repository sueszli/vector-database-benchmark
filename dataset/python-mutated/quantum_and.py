"""Implementations of boolean logic quantum circuits."""
from __future__ import annotations
from qiskit.circuit import QuantumRegister, QuantumCircuit, AncillaRegister
from qiskit.circuit.library.standard_gates import MCXGate

class AND(QuantumCircuit):
    """A circuit implementing the logical AND operation on a number of qubits.

    For the AND operation the state :math:`|1\\rangle` is interpreted as ``True``. The result
    qubit is flipped, if the state of all variable qubits is ``True``. In this format, the AND
    operation equals a multi-controlled X gate, which is controlled on all variable qubits.
    Using a list of flags however, qubits can be skipped or negated. Practically, the flags
    allow to skip controls or to apply pre- and post-X gates to the negated qubits.

    The AND gate without special flags equals the multi-controlled-X gate:

    .. plot::

       from qiskit.circuit.library import AND
       from qiskit.tools.jupyter.library import _generate_circuit_library_visualization
       circuit = AND(5)
       _generate_circuit_library_visualization(circuit)

    Using flags we can negate qubits or skip them. For instance, if we have 5 qubits and want to
    return ``True`` if the first qubit is ``False`` and the last two are ``True`` we use the flags
    ``[-1, 0, 0, 1, 1]``.

    .. plot::

       from qiskit.circuit.library import AND
       from qiskit.tools.jupyter.library import _generate_circuit_library_visualization
       circuit = AND(5, flags=[-1, 0, 0, 1, 1])
       _generate_circuit_library_visualization(circuit)

    """

    def __init__(self, num_variable_qubits: int, flags: list[int] | None=None, mcx_mode: str='noancilla') -> None:
        if False:
            print('Hello World!')
        'Create a new logical AND circuit.\n\n        Args:\n            num_variable_qubits: The qubits of which the OR is computed. The result will be written\n                into an additional result qubit.\n            flags: A list of +1/0/-1 marking negations or omissions of qubits.\n            mcx_mode: The mode to be used to implement the multi-controlled X gate.\n        '
        self.num_variable_qubits = num_variable_qubits
        self.flags = flags
        qr_variable = QuantumRegister(num_variable_qubits, name='variable')
        qr_result = QuantumRegister(1, name='result')
        circuit = QuantumCircuit(qr_variable, qr_result, name='and')
        flags = flags or [1] * num_variable_qubits
        control_qubits = [q for (q, flag) in zip(qr_variable, flags) if flag != 0]
        flip_qubits = [q for (q, flag) in zip(qr_variable, flags) if flag < 0]
        num_ancillas = MCXGate.get_num_ancilla_qubits(len(control_qubits), mode=mcx_mode)
        if num_ancillas > 0:
            qr_ancilla = AncillaRegister(num_ancillas, 'ancilla')
            circuit.add_register(qr_ancilla)
        else:
            qr_ancilla = AncillaRegister(0)
        if len(flip_qubits) > 0:
            circuit.x(flip_qubits)
        circuit.mcx(control_qubits, qr_result[:], qr_ancilla[:], mode=mcx_mode)
        if len(flip_qubits) > 0:
            circuit.x(flip_qubits)
        super().__init__(*circuit.qregs, name='and')
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)