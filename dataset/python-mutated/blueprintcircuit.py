"""Blueprint circuit object."""
from __future__ import annotations
from abc import ABC, abstractmethod
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.parametertable import ParameterTable, ParameterView

class BlueprintCircuit(QuantumCircuit, ABC):
    """Blueprint circuit object.

    In many applications it is necessary to pass around the structure a circuit will have without
    explicitly knowing e.g. its number of qubits, or other missing information. This can be solved
    by having a circuit that knows how to construct itself, once all information is available.

    This class provides an interface for such circuits. Before internal data of the circuit is
    accessed, the ``_build`` method is called. There the configuration of the circuit is checked.
    """

    def __init__(self, *regs, name: str | None=None) -> None:
        if False:
            i = 10
            return i + 15
        'Create a new blueprint circuit.'
        super().__init__(*regs, name=name)
        self._qregs: list[QuantumRegister] = []
        self._cregs: list[ClassicalRegister] = []
        self._qubits = []
        self._qubit_indices = {}
        self._is_built = False

    @abstractmethod
    def _check_configuration(self, raise_on_failure: bool=True) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Check if the current configuration allows the circuit to be built.\n\n        Args:\n            raise_on_failure: If True, raise if the configuration is invalid. If False, return\n                False if the configuration is invalid.\n\n        Returns:\n            True, if the configuration is valid. Otherwise, depending on the value of\n            ``raise_on_failure`` an error is raised or False is returned.\n        '
        raise NotImplementedError

    @abstractmethod
    def _build(self) -> None:
        if False:
            while True:
                i = 10
        'Build the circuit.'
        if self._is_built:
            return
        self._check_configuration()
        self._is_built = True

    def _invalidate(self) -> None:
        if False:
            while True:
                i = 10
        'Invalidate the current circuit build.'
        self._data = []
        self._parameter_table = ParameterTable()
        self.global_phase = 0
        self._is_built = False

    @property
    def qregs(self):
        if False:
            print('Hello World!')
        'A list of the quantum registers associated with the circuit.'
        return self._qregs

    @qregs.setter
    def qregs(self, qregs):
        if False:
            i = 10
            return i + 15
        'Set the quantum registers associated with the circuit.'
        self._qregs = []
        self._qubits = []
        self._ancillas = []
        self._qubit_indices = {}
        self.add_register(*qregs)
        self._invalidate()

    @property
    def data(self):
        if False:
            print('Hello World!')
        if not self._is_built:
            self._build()
        return super().data

    def decompose(self, gates_to_decompose=None, reps=1):
        if False:
            i = 10
            return i + 15
        if not self._is_built:
            self._build()
        return super().decompose(gates_to_decompose, reps)

    def draw(self, *args, **kwargs):
        if False:
            print('Hello World!')
        if not self._is_built:
            self._build()
        return super().draw(*args, **kwargs)

    @property
    def num_parameters(self) -> int:
        if False:
            return 10
        if not self._is_built:
            self._build()
        return super().num_parameters

    @property
    def parameters(self) -> ParameterView:
        if False:
            i = 10
            return i + 15
        if not self._is_built:
            self._build()
        return super().parameters

    def qasm(self, formatted=False, filename=None, encoding=None):
        if False:
            for i in range(10):
                print('nop')
        if not self._is_built:
            self._build()
        return super().qasm(formatted, filename, encoding)

    def _append(self, instruction, _qargs=None, _cargs=None):
        if False:
            i = 10
            return i + 15
        if not self._is_built:
            self._build()
        return super()._append(instruction, _qargs, _cargs)

    def compose(self, other, qubits=None, clbits=None, front=False, inplace=False, wrap=False):
        if False:
            return 10
        if not self._is_built:
            self._build()
        return super().compose(other, qubits, clbits, front, inplace, wrap)

    def inverse(self):
        if False:
            while True:
                i = 10
        if not self._is_built:
            self._build()
        return super().inverse()

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.data)

    def __getitem__(self, item):
        if False:
            print('Hello World!')
        return self.data[item]

    def size(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if not self._is_built:
            self._build()
        return super().size(*args, **kwargs)

    def to_instruction(self, parameter_map=None, label=None):
        if False:
            for i in range(10):
                print('nop')
        if not self._is_built:
            self._build()
        return super().to_instruction(parameter_map, label=label)

    def to_gate(self, parameter_map=None, label=None):
        if False:
            return 10
        if not self._is_built:
            self._build()
        return super().to_gate(parameter_map, label=label)

    def depth(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if not self._is_built:
            self._build()
        return super().depth(*args, **kwargs)

    def count_ops(self):
        if False:
            return 10
        if not self._is_built:
            self._build()
        return super().count_ops()

    def num_nonlocal_gates(self):
        if False:
            while True:
                i = 10
        if not self._is_built:
            self._build()
        return super().num_nonlocal_gates()

    def num_connected_components(self, unitary_only=False):
        if False:
            return 10
        if not self._is_built:
            self._build()
        return super().num_connected_components(unitary_only=unitary_only)

    def copy(self, name=None):
        if False:
            return 10
        if not self._is_built:
            self._build()
        circuit_copy = super().copy(name=name)
        circuit_copy._is_built = self._is_built
        return circuit_copy