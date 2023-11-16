"""Multiple-Control, Multiple-Target Gate."""
from __future__ import annotations
from collections.abc import Callable
from qiskit import circuit
from qiskit.circuit import ControlledGate, Gate, QuantumRegister, QuantumCircuit
from qiskit.exceptions import QiskitError
from ..standard_gates import XGate, YGate, ZGate, HGate, TGate, TdgGate, SGate, SdgGate

class MCMT(QuantumCircuit):
    """The multi-controlled multi-target gate, for an arbitrary singly controlled target gate.

    For example, the H gate controlled on 3 qubits and acting on 2 target qubit is represented as:

    .. parsed-literal::

        ───■────
           │
        ───■────
           │
        ───■────
        ┌──┴───┐
        ┤0     ├
        │  2-H │
        ┤1     ├
        └──────┘

    This default implementations requires no ancilla qubits, by broadcasting the target gate
    to the number of target qubits and using Qiskit's generic control routine to control the
    broadcasted target on the control qubits. If ancilla qubits are available, a more efficient
    variant using the so-called V-chain decomposition can be used. This is implemented in
    :class:`~qiskit.circuit.library.MCMTVChain`.
    """

    def __init__(self, gate: Gate | Callable[[QuantumCircuit, circuit.Qubit, circuit.Qubit], circuit.Instruction], num_ctrl_qubits: int, num_target_qubits: int) -> None:
        if False:
            i = 10
            return i + 15
        'Create a new multi-control multi-target gate.\n\n        Args:\n            gate: The gate to be applied controlled on the control qubits and applied to the target\n                qubits. Can be either a Gate or a circuit method.\n                If it is a callable, it will be casted to a Gate.\n            num_ctrl_qubits: The number of control qubits.\n            num_target_qubits: The number of target qubits.\n\n        Raises:\n            AttributeError: If the gate cannot be casted to a controlled gate.\n            AttributeError: If the number of controls or targets is 0.\n        '
        if num_ctrl_qubits == 0 or num_target_qubits == 0:
            raise AttributeError('Need at least one control and one target qubit.')
        self.gate = self._identify_gate(gate)
        self.num_ctrl_qubits = num_ctrl_qubits
        self.num_target_qubits = num_target_qubits
        num_qubits = num_ctrl_qubits + num_target_qubits + self.num_ancilla_qubits
        super().__init__(num_qubits, name='mcmt')
        self._label = f'{num_target_qubits}-{self.gate.name.capitalize()}'
        self._build()

    def _build(self):
        if False:
            while True:
                i = 10
        'Define the MCMT gate without ancillas.'
        if self.num_target_qubits == 1:
            broadcasted_gate = self.gate
        else:
            broadcasted = QuantumCircuit(self.num_target_qubits, name=self._label)
            for target in list(range(self.num_target_qubits)):
                broadcasted.append(self.gate, [target], [])
            broadcasted_gate = broadcasted.to_gate()
        mcmt_gate = broadcasted_gate.control(self.num_ctrl_qubits)
        self.append(mcmt_gate, self.qubits, [])

    @property
    def num_ancilla_qubits(self):
        if False:
            print('Hello World!')
        'Return the number of ancillas.'
        return 0

    def _identify_gate(self, gate):
        if False:
            while True:
                i = 10
        'Case the gate input to a gate.'
        valid_gates = {'ch': HGate(), 'cx': XGate(), 'cy': YGate(), 'cz': ZGate(), 'h': HGate(), 's': SGate(), 'sdg': SdgGate(), 'x': XGate(), 'y': YGate(), 'z': ZGate(), 't': TGate(), 'tdg': TdgGate()}
        if isinstance(gate, ControlledGate):
            base_gate = gate.base_gate
        elif isinstance(gate, Gate):
            if gate.num_qubits != 1:
                raise AttributeError('Base gate must act on one qubit only.')
            base_gate = gate
        elif isinstance(gate, QuantumCircuit):
            if gate.num_qubits != 1:
                raise AttributeError('The circuit you specified as control gate can only have one qubit!')
            base_gate = gate.to_gate()
        else:
            if callable(gate):
                name = gate.__name__
            elif isinstance(gate, str):
                name = gate
            else:
                raise AttributeError(f'Invalid gate specified: {gate}')
            base_gate = valid_gates[name]
        return base_gate

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        if False:
            print('Hello World!')
        'Return the controlled version of the MCMT circuit.'
        if ctrl_state is None:
            return MCMT(self.gate, self.num_ctrl_qubits + num_ctrl_qubits, self.num_target_qubits)
        return super().control(num_ctrl_qubits, label, ctrl_state)

    def inverse(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the inverse MCMT circuit, which is itself.'
        return MCMT(self.gate, self.num_ctrl_qubits, self.num_target_qubits)

class MCMTVChain(MCMT):
    """The MCMT implementation using the CCX V-chain.

    This implementation requires ancillas but is decomposed into a much shallower circuit
    than the default implementation in :class:`~qiskit.circuit.library.MCMT`.

    **Expanded Circuit:**

    .. plot::

       from qiskit.circuit.library import MCMTVChain, ZGate
       from qiskit.tools.jupyter.library import _generate_circuit_library_visualization
       circuit = MCMTVChain(ZGate(), 2, 2)
       _generate_circuit_library_visualization(circuit.decompose())

    **Examples:**

        >>> from qiskit.circuit.library import HGate
        >>> MCMTVChain(HGate(), 3, 2).draw()

        q_0: ──■────────────────────────■──
               │                        │
        q_1: ──■────────────────────────■──
               │                        │
        q_2: ──┼────■──────────────■────┼──
               │    │  ┌───┐       │    │
        q_3: ──┼────┼──┤ H ├───────┼────┼──
               │    │  └─┬─┘┌───┐  │    │
        q_4: ──┼────┼────┼──┤ H ├──┼────┼──
             ┌─┴─┐  │    │  └─┬─┘  │  ┌─┴─┐
        q_5: ┤ X ├──■────┼────┼────■──┤ X ├
             └───┘┌─┴─┐  │    │  ┌─┴─┐└───┘
        q_6: ─────┤ X ├──■────■──┤ X ├─────
                  └───┘          └───┘
    """

    def _build(self):
        if False:
            for i in range(10):
                print('nop')
        'Define the MCMT gate.'
        control_qubits = self.qubits[:self.num_ctrl_qubits]
        target_qubits = self.qubits[self.num_ctrl_qubits:self.num_ctrl_qubits + self.num_target_qubits]
        ancilla_qubits = self.qubits[self.num_ctrl_qubits + self.num_target_qubits:]
        if len(ancilla_qubits) > 0:
            master_control = ancilla_qubits[-1]
        else:
            master_control = control_qubits[0]
        self._ccx_v_chain_rule(control_qubits, ancilla_qubits, reverse=False)
        for qubit in target_qubits:
            self.append(self.gate.control(), [master_control, qubit], [])
        self._ccx_v_chain_rule(control_qubits, ancilla_qubits, reverse=True)

    @property
    def num_ancilla_qubits(self):
        if False:
            print('Hello World!')
        'Return the number of ancilla qubits required.'
        return max(0, self.num_ctrl_qubits - 1)

    def _ccx_v_chain_rule(self, control_qubits: QuantumRegister | list[circuit.Qubit], ancilla_qubits: QuantumRegister | list[circuit.Qubit], reverse: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Get the rule for the CCX V-chain.\n\n        The CCX V-chain progressively computes the CCX of the control qubits and puts the final\n        result in the last ancillary qubit.\n\n        Args:\n            control_qubits: The control qubits.\n            ancilla_qubits: The ancilla qubits.\n            reverse: If True, compute the chain down to the qubit. If False, compute upwards.\n\n        Returns:\n            The rule for the (reversed) CCX V-chain.\n\n        Raises:\n            QiskitError: If an insufficient number of ancilla qubits was provided.\n        '
        if len(ancilla_qubits) == 0:
            return
        if len(ancilla_qubits) < len(control_qubits) - 1:
            raise QiskitError('Insufficient number of ancilla qubits.')
        iterations = list(enumerate(range(2, len(control_qubits))))
        if not reverse:
            self.ccx(control_qubits[0], control_qubits[1], ancilla_qubits[0])
            for (i, j) in iterations:
                self.ccx(control_qubits[j], ancilla_qubits[i], ancilla_qubits[i + 1])
        else:
            for (i, j) in reversed(iterations):
                self.ccx(control_qubits[j], ancilla_qubits[i], ancilla_qubits[i + 1])
            self.ccx(control_qubits[0], control_qubits[1], ancilla_qubits[0])

    def inverse(self):
        if False:
            return 10
        return MCMTVChain(self.gate, self.num_ctrl_qubits, self.num_target_qubits)