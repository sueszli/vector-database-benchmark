"""Implementations of boolean logic quantum circuits."""
from __future__ import annotations
from typing import List, Optional
from qiskit.circuit import QuantumRegister, QuantumCircuit, AncillaRegister
from qiskit.circuit.library.standard_gates import MCXGate

class OR(QuantumCircuit):
    """A circuit implementing the logical OR operation on a number of qubits.

    For the OR operation the state :math:`|1\\rangle` is interpreted as ``True``. The result
    qubit is flipped, if the state of any variable qubit is ``True``. The OR is implemented using
    a multi-open-controlled X gate (i.e. flips if the state is :math:`|0\\rangle`) and
    applying an X gate on the result qubit.
    Using a list of flags, qubits can be skipped or negated.

    The OR gate without special flags:

    .. plot::

       from qiskit.circuit.library import OR
       from qiskit.tools.jupyter.library import _generate_circuit_library_visualization
       circuit = OR(5)
       _generate_circuit_library_visualization(circuit)

    Using flags we can negate qubits or skip them. For instance, if we have 5 qubits and want to
    return ``True`` if the first qubit is ``False`` or one of the last two are ``True`` we use the
    flags ``[-1, 0, 0, 1, 1]``.

    .. plot::

       from qiskit.circuit.library import OR
       from qiskit.tools.jupyter.library import _generate_circuit_library_visualization
       circuit = OR(5, flags=[-1, 0, 0, 1, 1])
       _generate_circuit_library_visualization(circuit)

    """

    def __init__(self, num_variable_qubits: int, flags: Optional[List[int]]=None, mcx_mode: str='noancilla') -> None:
        if False:
            print('Hello World!')
        'Create a new logical OR circuit.\n\n        Args:\n            num_variable_qubits: The qubits of which the OR is computed. The result will be written\n                into an additional result qubit.\n            flags: A list of +1/0/-1 marking negations or omissions of qubits.\n            mcx_mode: The mode to be used to implement the multi-controlled X gate.\n        '
        self.num_variable_qubits = num_variable_qubits
        self.flags = flags
        qr_variable = QuantumRegister(num_variable_qubits, name='variable')
        qr_result = QuantumRegister(1, name='result')
        circuit = QuantumCircuit(qr_variable, qr_result, name='or')
        flags = flags or [1] * num_variable_qubits
        control_qubits = [q for (q, flag) in zip(qr_variable, flags) if flag != 0]
        flip_qubits = [q for (q, flag) in zip(qr_variable, flags) if flag > 0]
        num_ancillas = MCXGate.get_num_ancilla_qubits(len(control_qubits), mode=mcx_mode)
        if num_ancillas > 0:
            qr_ancilla = AncillaRegister(num_ancillas, 'ancilla')
            circuit.add_register(qr_ancilla)
        else:
            qr_ancilla = AncillaRegister(0)
        circuit.x(qr_result)
        if len(flip_qubits) > 0:
            circuit.x(flip_qubits)
        circuit.mcx(control_qubits, qr_result[:], qr_ancilla[:], mode=mcx_mode)
        if len(flip_qubits) > 0:
            circuit.x(flip_qubits)
        super().__init__(*circuit.qregs, name='or')
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)