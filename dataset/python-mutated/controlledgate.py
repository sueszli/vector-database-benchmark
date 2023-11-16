"""Controlled unitary gate."""
from __future__ import annotations
import copy
from typing import Optional, Union
from qiskit.circuit.exceptions import CircuitError
from .quantumcircuit import QuantumCircuit
from .gate import Gate
from .quantumregister import QuantumRegister
from ._utils import _ctrl_state_to_int

class ControlledGate(Gate):
    """Controlled unitary gate."""

    def __init__(self, name: str, num_qubits: int, params: list, label: Optional[str]=None, num_ctrl_qubits: Optional[int]=1, definition: Optional['QuantumCircuit']=None, ctrl_state: Optional[Union[int, str]]=None, base_gate: Optional[Gate]=None, duration=None, unit=None, *, _base_label=None):
        if False:
            i = 10
            return i + 15
        "Create a new ControlledGate. In the new gate the first ``num_ctrl_qubits``\n        of the gate are the controls.\n\n        Args:\n            name: The name of the gate.\n            num_qubits: The number of qubits the gate acts on.\n            params: A list of parameters for the gate.\n            label: An optional label for the gate.\n            num_ctrl_qubits: Number of control qubits.\n            definition: A list of gate rules for implementing this gate. The\n                elements of the list are tuples of (:meth:`~qiskit.circuit.Gate`, [qubit_list],\n                [clbit_list]).\n            ctrl_state: The control state in decimal or as\n                a bitstring (e.g. '111'). If specified as a bitstring the length\n                must equal num_ctrl_qubits, MSB on left. If None, use\n                2**num_ctrl_qubits-1.\n            base_gate: Gate object to be controlled.\n\n        Raises:\n            CircuitError: If ``num_ctrl_qubits`` >= ``num_qubits``.\n            CircuitError: ctrl_state < 0 or ctrl_state > 2**num_ctrl_qubits.\n\n        Examples:\n\n        Create a controlled standard gate and apply it to a circuit.\n\n        .. plot::\n           :include-source:\n\n           from qiskit import QuantumCircuit, QuantumRegister\n           from qiskit.circuit.library.standard_gates import HGate\n\n           qr = QuantumRegister(3)\n           qc = QuantumCircuit(qr)\n           c3h_gate = HGate().control(2)\n           qc.append(c3h_gate, qr)\n           qc.draw('mpl')\n\n        Create a controlled custom gate and apply it to a circuit.\n\n        .. plot::\n           :include-source:\n\n           from qiskit import QuantumCircuit, QuantumRegister\n           from qiskit.circuit.library.standard_gates import HGate\n\n           qc1 = QuantumCircuit(2)\n           qc1.x(0)\n           qc1.h(1)\n           custom = qc1.to_gate().control(2)\n\n           qc2 = QuantumCircuit(4)\n           qc2.append(custom, [0, 3, 1, 2])\n           qc2.draw('mpl')\n        "
        self.base_gate = None if base_gate is None else base_gate.copy()
        super().__init__(name, num_qubits, params, label=label, duration=duration, unit=unit)
        self._num_ctrl_qubits = 1
        self.num_ctrl_qubits = num_ctrl_qubits
        self.definition = copy.deepcopy(definition)
        self._ctrl_state = None
        self.ctrl_state = ctrl_state
        self._name = name

    @property
    def definition(self) -> QuantumCircuit:
        if False:
            print('Hello World!')
        'Return definition in terms of other basic gates. If the gate has\n        open controls, as determined from `self.ctrl_state`, the returned\n        definition is conjugated with X without changing the internal\n        `_definition`.\n        '
        if self._open_ctrl:
            closed_gate = self.to_mutable()
            closed_gate.ctrl_state = None
            bit_ctrl_state = bin(self.ctrl_state)[2:].zfill(self.num_ctrl_qubits)
            qreg = QuantumRegister(self.num_qubits, 'q')
            qc_open_ctrl = QuantumCircuit(qreg)
            for (qind, val) in enumerate(bit_ctrl_state[::-1]):
                if val == '0':
                    qc_open_ctrl.x(qind)
            qc_open_ctrl.append(closed_gate, qargs=qreg[:])
            for (qind, val) in enumerate(bit_ctrl_state[::-1]):
                if val == '0':
                    qc_open_ctrl.x(qind)
            return qc_open_ctrl
        else:
            return super().definition

    @definition.setter
    def definition(self, excited_def: 'QuantumCircuit'):
        if False:
            return 10
        'Set controlled gate definition with closed controls.\n\n        Args:\n            excited_def: The circuit with all closed controls.\n        '
        self._definition = excited_def

    @property
    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Get name of gate. If the gate has open controls the gate name\n        will become:\n\n           <original_name_o<ctrl_state>\n\n        where <original_name> is the gate name for the default case of\n        closed control qubits and <ctrl_state> is the integer value of\n        the control state for the gate.\n        '
        if self._open_ctrl:
            return f'{self._name}_o{self.ctrl_state}'
        else:
            return self._name

    @name.setter
    def name(self, name_str):
        if False:
            i = 10
            return i + 15
        'Set the name of the gate. Note the reported name may differ\n        from the set name if the gate has open controls.\n        '
        self._name = name_str

    @property
    def num_ctrl_qubits(self):
        if False:
            while True:
                i = 10
        'Get number of control qubits.\n\n        Returns:\n            int: The number of control qubits for the gate.\n        '
        return self._num_ctrl_qubits

    @num_ctrl_qubits.setter
    def num_ctrl_qubits(self, num_ctrl_qubits):
        if False:
            return 10
        'Set the number of control qubits.\n\n        Args:\n            num_ctrl_qubits (int): The number of control qubits.\n\n        Raises:\n            CircuitError: ``num_ctrl_qubits`` is not an integer in ``[1, num_qubits]``.\n        '
        if num_ctrl_qubits != int(num_ctrl_qubits):
            raise CircuitError('The number of control qubits must be an integer.')
        num_ctrl_qubits = int(num_ctrl_qubits)
        upper_limit = self.num_qubits - getattr(self.base_gate, 'num_qubits', 0)
        if num_ctrl_qubits < 1 or num_ctrl_qubits > upper_limit:
            limit = 'num_qubits' if self.base_gate is None else 'num_qubits - base_gate.num_qubits'
            raise CircuitError(f'The number of control qubits must be in `[1, {limit}]`.')
        self._num_ctrl_qubits = num_ctrl_qubits

    @property
    def ctrl_state(self) -> int:
        if False:
            while True:
                i = 10
        'Return the control state of the gate as a decimal integer.'
        return self._ctrl_state

    @ctrl_state.setter
    def ctrl_state(self, ctrl_state: Union[int, str, None]):
        if False:
            return 10
        'Set the control state of this gate.\n\n        Args:\n            ctrl_state: The control state of the gate.\n\n        Raises:\n            CircuitError: ctrl_state is invalid.\n        '
        self._ctrl_state = _ctrl_state_to_int(ctrl_state, self.num_ctrl_qubits)

    @property
    def params(self):
        if False:
            print('Hello World!')
        'Get parameters from base_gate.\n\n        Returns:\n            list: List of gate parameters.\n\n        Raises:\n            CircuitError: Controlled gate does not define a base gate\n        '
        if self.base_gate:
            return self.base_gate.params
        else:
            raise CircuitError('Controlled gate does not define base gate for extracting params')

    @params.setter
    def params(self, parameters):
        if False:
            return 10
        'Set base gate parameters.\n\n        Args:\n            parameters (list): The list of parameters to set.\n\n        Raises:\n            CircuitError: If controlled gate does not define a base gate.\n        '
        if self.base_gate:
            if self.base_gate.mutable:
                self.base_gate.params = parameters
            elif parameters:
                raise CircuitError('cannot set parameters on immutable base gate')
        else:
            raise CircuitError('Controlled gate does not define base gate for extracting params')

    def __deepcopy__(self, memo=None):
        if False:
            while True:
                i = 10
        cpy = copy.copy(self)
        cpy.base_gate = self.base_gate.copy()
        if self._definition:
            cpy._definition = copy.deepcopy(self._definition, memo)
        return cpy

    @property
    def _open_ctrl(self) -> bool:
        if False:
            while True:
                i = 10
        'Return whether gate has any open controls'
        return self.ctrl_state < 2 ** self.num_ctrl_qubits - 1

    def __eq__(self, other) -> bool:
        if False:
            while True:
                i = 10
        return isinstance(other, ControlledGate) and self.num_ctrl_qubits == other.num_ctrl_qubits and (self.ctrl_state == other.ctrl_state) and (self.base_gate == other.base_gate) and (self.num_qubits == other.num_qubits) and (self.num_clbits == other.num_clbits) and (self.definition == other.definition)

    def inverse(self) -> 'ControlledGate':
        if False:
            print('Hello World!')
        'Invert this gate by calling inverse on the base gate.'
        return self.base_gate.inverse().control(self.num_ctrl_qubits, ctrl_state=self.ctrl_state)