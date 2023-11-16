"""Quantum circuit object."""
from __future__ import annotations
import copy
import multiprocessing as mp
import warnings
import typing
import math
from collections import OrderedDict, defaultdict, namedtuple
from typing import Union, Optional, Tuple, Type, TypeVar, Sequence, Callable, Mapping, Iterable, Any, DefaultDict, Literal, overload
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.utils.multiprocessing import is_main_process
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils import optionals as _optionals
from qiskit.utils.deprecation import deprecate_func
from . import _classical_resource_map
from ._utils import sort_parameters
from .classical import expr
from .parameterexpression import ParameterExpression, ParameterValueType
from .quantumregister import QuantumRegister, Qubit, AncillaRegister, AncillaQubit
from .classicalregister import ClassicalRegister, Clbit
from .parametertable import ParameterReferences, ParameterTable, ParameterView
from .parametervector import ParameterVector
from .instructionset import InstructionSet
from .operation import Operation
from .register import Register
from .bit import Bit
from .quantumcircuitdata import QuantumCircuitData, CircuitInstruction
from .delay import Delay
if typing.TYPE_CHECKING:
    import qiskit
    from qiskit.transpiler.layout import TranspileLayout
    from qiskit.quantum_info.operators.base_operator import BaseOperator
BitLocations = namedtuple('BitLocations', ('index', 'registers'))
S = TypeVar('S')
T = TypeVar('T')
QubitSpecifier = Union[Qubit, QuantumRegister, int, slice, Sequence[Union[Qubit, int]]]
ClbitSpecifier = Union[Clbit, ClassicalRegister, int, slice, Sequence[Union[Clbit, int]]]
BitType = TypeVar('BitType', Qubit, Clbit)

class QuantumCircuit:
    """Create a new circuit.

    A circuit is a list of instructions bound to some registers.

    Args:
        regs (list(:class:`~.Register`) or list(``int``) or list(list(:class:`~.Bit`))): The
            registers to be included in the circuit.

            * If a list of :class:`~.Register` objects, represents the :class:`.QuantumRegister`
              and/or :class:`.ClassicalRegister` objects to include in the circuit.

              For example:

                * ``QuantumCircuit(QuantumRegister(4))``
                * ``QuantumCircuit(QuantumRegister(4), ClassicalRegister(3))``
                * ``QuantumCircuit(QuantumRegister(4, 'qr0'), QuantumRegister(2, 'qr1'))``

            * If a list of ``int``, the amount of qubits and/or classical bits to include in
              the circuit. It can either be a single int for just the number of quantum bits,
              or 2 ints for the number of quantum bits and classical bits, respectively.

              For example:

                * ``QuantumCircuit(4) # A QuantumCircuit with 4 qubits``
                * ``QuantumCircuit(4, 3) # A QuantumCircuit with 4 qubits and 3 classical bits``

            * If a list of python lists containing :class:`.Bit` objects, a collection of
              :class:`.Bit` s to be added to the circuit.


        name (str): the name of the quantum circuit. If not set, an
            automatically generated string will be assigned.
        global_phase (float or ParameterExpression): The global phase of the circuit in radians.
        metadata (dict): Arbitrary key value metadata to associate with the
            circuit. This gets stored as free-form data in a dict in the
            :attr:`~qiskit.circuit.QuantumCircuit.metadata` attribute. It will
            not be directly used in the circuit.

    Raises:
        CircuitError: if the circuit name, if given, is not valid.

    Examples:

        Construct a simple Bell state circuit.

        .. plot::
           :include-source:

           from qiskit import QuantumCircuit

           qc = QuantumCircuit(2, 2)
           qc.h(0)
           qc.cx(0, 1)
           qc.measure([0, 1], [0, 1])
           qc.draw('mpl')

        Construct a 5-qubit GHZ circuit.

        .. code-block::

           from qiskit import QuantumCircuit

           qc = QuantumCircuit(5)
           qc.h(0)
           qc.cx(0, range(1, 5))
           qc.measure_all()

        Construct a 4-qubit Bernstein-Vazirani circuit using registers.

        .. plot::
           :include-source:

           from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

           qr = QuantumRegister(3, 'q')
           anc = QuantumRegister(1, 'ancilla')
           cr = ClassicalRegister(3, 'c')
           qc = QuantumCircuit(qr, anc, cr)

           qc.x(anc[0])
           qc.h(anc[0])
           qc.h(qr[0:3])
           qc.cx(qr[0:3], anc[0])
           qc.h(qr[0:3])
           qc.barrier(qr)
           qc.measure(qr, cr)

           qc.draw('mpl')
    """
    instances = 0
    prefix = 'circuit'

    def __init__(self, *regs: Register | int | Sequence[Bit], name: str | None=None, global_phase: ParameterValueType=0, metadata: dict | None=None):
        if False:
            for i in range(10):
                print('nop')
        if any((not isinstance(reg, (list, QuantumRegister, ClassicalRegister)) for reg in regs)):
            try:
                valid_reg_size = all((reg == int(reg) for reg in regs))
            except (ValueError, TypeError):
                valid_reg_size = False
            if not valid_reg_size:
                raise CircuitError("Circuit args must be Registers or integers. (%s '%s' was provided)" % ([type(reg).__name__ for reg in regs], regs))
            regs = tuple((int(reg) for reg in regs))
        self._base_name = None
        if name is None:
            self._base_name = self.cls_prefix()
            self._name_update()
        elif not isinstance(name, str):
            raise CircuitError('The circuit name should be a string (or None to auto-generate a name).')
        else:
            self._base_name = name
            self.name = name
        self._increment_instances()
        self._data: list[CircuitInstruction] = []
        self._op_start_times = None
        self._control_flow_scopes: list['qiskit.circuit.controlflow.builder.ControlFlowBuilderBlock'] = []
        self.qregs: list[QuantumRegister] = []
        self.cregs: list[ClassicalRegister] = []
        self._qubits: list[Qubit] = []
        self._clbits: list[Clbit] = []
        self._qubit_indices: dict[Qubit, BitLocations] = {}
        self._clbit_indices: dict[Clbit, BitLocations] = {}
        self._ancillas: list[AncillaQubit] = []
        self._calibrations: DefaultDict[str, dict[tuple, Any]] = defaultdict(dict)
        self.add_register(*regs)
        self._parameter_table = ParameterTable()
        self._parameters = None
        self._layout = None
        self._global_phase: ParameterValueType = 0
        self.global_phase = global_phase
        self.duration = None
        self.unit = 'dt'
        self.metadata = {} if metadata is None else metadata

    @staticmethod
    def from_instructions(instructions: Iterable[CircuitInstruction | tuple[qiskit.circuit.Instruction] | tuple[qiskit.circuit.Instruction, Iterable[Qubit]] | tuple[qiskit.circuit.Instruction, Iterable[Qubit], Iterable[Clbit]]], *, qubits: Iterable[Qubit]=(), clbits: Iterable[Clbit]=(), name: str | None=None, global_phase: ParameterValueType=0, metadata: dict | None=None) -> 'QuantumCircuit':
        if False:
            print('Hello World!')
        'Construct a circuit from an iterable of CircuitInstructions.\n\n        Args:\n            instructions: The instructions to add to the circuit.\n            qubits: Any qubits to add to the circuit. This argument can be used,\n                for example, to enforce a particular ordering of qubits.\n            clbits: Any classical bits to add to the circuit. This argument can be used,\n                for example, to enforce a particular ordering of classical bits.\n            name: The name of the circuit.\n            global_phase: The global phase of the circuit in radians.\n            metadata: Arbitrary key value metadata to associate with the circuit.\n\n        Returns:\n            The quantum circuit.\n        '
        circuit = QuantumCircuit(name=name, global_phase=global_phase, metadata=metadata)
        added_qubits = set()
        added_clbits = set()
        if qubits:
            qubits = list(qubits)
            circuit.add_bits(qubits)
            added_qubits.update(qubits)
        if clbits:
            clbits = list(clbits)
            circuit.add_bits(clbits)
            added_clbits.update(clbits)
        for instruction in instructions:
            if not isinstance(instruction, CircuitInstruction):
                instruction = CircuitInstruction(*instruction)
            qubits = [qubit for qubit in instruction.qubits if qubit not in added_qubits]
            clbits = [clbit for clbit in instruction.clbits if clbit not in added_clbits]
            circuit.add_bits(qubits)
            circuit.add_bits(clbits)
            added_qubits.update(qubits)
            added_clbits.update(clbits)
            circuit._append(instruction)
        return circuit

    @property
    def layout(self) -> Optional[TranspileLayout]:
        if False:
            return 10
        'Return any associated layout information about the circuit\n\n        This attribute contains an optional :class:`~.TranspileLayout`\n        object. This is typically set on the output from :func:`~.transpile`\n        or :meth:`.PassManager.run` to retain information about the\n        permutations caused on the input circuit by transpilation.\n\n        There are two types of permutations caused by the :func:`~.transpile`\n        function, an initial layout which permutes the qubits based on the\n        selected physical qubits on the :class:`~.Target`, and a final layout\n        which is an output permutation caused by :class:`~.SwapGate`\\s\n        inserted during routing.\n        '
        return self._layout

    @classmethod
    @property
    @deprecate_func(since='0.45.0', additional_msg='No alternative will be provided.', is_property=True)
    def header(cls) -> str:
        if False:
            i = 10
            return i + 15
        'The OpenQASM 2.0 header statement.'
        return 'OPENQASM 2.0;'

    @classmethod
    @property
    @deprecate_func(since='0.45.0', additional_msg='No alternative will be provided.', is_property=True)
    def extension_lib(cls) -> str:
        if False:
            print('Hello World!')
        'The standard OpenQASM 2 import statement.'
        return 'include "qelib1.inc";'

    @property
    def data(self) -> QuantumCircuitData:
        if False:
            while True:
                i = 10
        'Return the circuit data (instructions and context).\n\n        Returns:\n            QuantumCircuitData: a list-like object containing the :class:`.CircuitInstruction`\\ s\n            for each instruction.\n        '
        return QuantumCircuitData(self)

    @data.setter
    def data(self, data_input: Iterable):
        if False:
            while True:
                i = 10
        'Sets the circuit data from a list of instructions and context.\n\n        Args:\n            data_input (Iterable): A sequence of instructions with their execution contexts.  The\n                elements must either be instances of :class:`.CircuitInstruction` (preferred), or a\n                3-tuple of ``(instruction, qargs, cargs)`` (legacy).  In the legacy format,\n                ``instruction`` must be an :class:`~.circuit.Instruction`, while ``qargs`` and\n                ``cargs`` must be iterables of :class:`~.circuit.Qubit` or :class:`.Clbit`\n                specifiers (similar to the allowed forms in calls to :meth:`append`).\n        '
        data_input = list(data_input)
        self._data = []
        self._parameter_table = ParameterTable()
        if not data_input:
            return
        if isinstance(data_input[0], CircuitInstruction):
            for instruction in data_input:
                self.append(instruction)
        else:
            for (instruction, qargs, cargs) in data_input:
                self.append(instruction, qargs, cargs)

    @property
    def op_start_times(self) -> list[int]:
        if False:
            while True:
                i = 10
        'Return a list of operation start times.\n\n        This attribute is enabled once one of scheduling analysis passes\n        runs on the quantum circuit.\n\n        Returns:\n            List of integers representing instruction start times.\n            The index corresponds to the index of instruction in :attr:`QuantumCircuit.data`.\n\n        Raises:\n            AttributeError: When circuit is not scheduled.\n        '
        if self._op_start_times is None:
            raise AttributeError('This circuit is not scheduled. To schedule it run the circuit through one of the transpiler scheduling passes.')
        return self._op_start_times

    @property
    def calibrations(self) -> dict:
        if False:
            print('Hello World!')
        "Return calibration dictionary.\n\n        The custom pulse definition of a given gate is of the form\n        ``{'gate_name': {(qubits, params): schedule}}``\n        "
        return dict(self._calibrations)

    @calibrations.setter
    def calibrations(self, calibrations: dict):
        if False:
            for i in range(10):
                print('nop')
        "Set the circuit calibration data from a dictionary of calibration definition.\n\n        Args:\n            calibrations (dict): A dictionary of input in the format\n               ``{'gate_name': {(qubits, gate_params): schedule}}``\n        "
        self._calibrations = defaultdict(dict, calibrations)

    def has_calibration_for(self, instruction: CircuitInstruction | tuple):
        if False:
            while True:
                i = 10
        'Return True if the circuit has a calibration defined for the instruction context. In this\n        case, the operation does not need to be translated to the device basis.\n        '
        if isinstance(instruction, CircuitInstruction):
            operation = instruction.operation
            qubits = instruction.qubits
        else:
            (operation, qubits, _) = instruction
        if not self.calibrations or operation.name not in self.calibrations:
            return False
        qubits = tuple((self.qubits.index(qubit) for qubit in qubits))
        params = []
        for p in operation.params:
            if isinstance(p, ParameterExpression) and (not p.parameters):
                params.append(float(p))
            else:
                params.append(p)
        params = tuple(params)
        return (qubits, params) in self.calibrations[operation.name]

    @property
    def metadata(self) -> dict:
        if False:
            i = 10
            return i + 15
        'The user provided metadata associated with the circuit.\n\n        The metadata for the circuit is a user provided ``dict`` of metadata\n        for the circuit. It will not be used to influence the execution or\n        operation of the circuit, but it is expected to be passed between\n        all transforms of the circuit (ie transpilation) and that providers will\n        associate any circuit metadata with the results it returns from\n        execution of that circuit.\n        '
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: dict | None):
        if False:
            print('Hello World!')
        'Update the circuit metadata'
        if metadata is None:
            metadata = {}
            warnings.warn('Setting metadata to None was deprecated in Terra 0.24.0 and this ability will be removed in a future release. Instead, set metadata to an empty dictionary.', DeprecationWarning, stacklevel=2)
        elif not isinstance(metadata, dict):
            raise TypeError('Only a dictionary is accepted for circuit metadata')
        self._metadata = metadata

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return str(self.draw(output='text'))

    def __eq__(self, other) -> bool:
        if False:
            while True:
                i = 10
        if not isinstance(other, QuantumCircuit):
            return False
        from qiskit.converters import circuit_to_dag
        return circuit_to_dag(self, copy_operations=False) == circuit_to_dag(other, copy_operations=False)

    @classmethod
    def _increment_instances(cls):
        if False:
            while True:
                i = 10
        cls.instances += 1

    @classmethod
    def cls_instances(cls) -> int:
        if False:
            print('Hello World!')
        'Return the current number of instances of this class,\n        useful for auto naming.'
        return cls.instances

    @classmethod
    def cls_prefix(cls) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return the prefix to use for auto naming.'
        return cls.prefix

    def _name_update(self) -> None:
        if False:
            print('Hello World!')
        'update name of instance using instance number'
        if not is_main_process():
            pid_name = f'-{mp.current_process().pid}'
        else:
            pid_name = ''
        self.name = f'{self._base_name}-{self.cls_instances()}{pid_name}'

    def has_register(self, register: Register) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if this circuit has the register r.\n\n        Args:\n            register (Register): a quantum or classical register.\n\n        Returns:\n            bool: True if the register is contained in this circuit.\n        '
        has_reg = False
        if isinstance(register, QuantumRegister) and register in self.qregs:
            has_reg = True
        elif isinstance(register, ClassicalRegister) and register in self.cregs:
            has_reg = True
        return has_reg

    def reverse_ops(self) -> 'QuantumCircuit':
        if False:
            while True:
                i = 10
        'Reverse the circuit by reversing the order of instructions.\n\n        This is done by recursively reversing all instructions.\n        It does not invert (adjoint) any gate.\n\n        Returns:\n            QuantumCircuit: the reversed circuit.\n\n        Examples:\n\n            input:\n\n            .. parsed-literal::\n\n                     ┌───┐\n                q_0: ┤ H ├─────■──────\n                     └───┘┌────┴─────┐\n                q_1: ─────┤ RX(1.57) ├\n                          └──────────┘\n\n            output:\n\n            .. parsed-literal::\n\n                                 ┌───┐\n                q_0: ─────■──────┤ H ├\n                     ┌────┴─────┐└───┘\n                q_1: ┤ RX(1.57) ├─────\n                     └──────────┘\n        '
        reverse_circ = QuantumCircuit(self.qubits, self.clbits, *self.qregs, *self.cregs, name=self.name + '_reverse')
        for instruction in reversed(self.data):
            reverse_circ._append(instruction.replace(operation=instruction.operation.reverse_ops()))
        reverse_circ.duration = self.duration
        reverse_circ.unit = self.unit
        return reverse_circ

    def reverse_bits(self) -> 'QuantumCircuit':
        if False:
            i = 10
            return i + 15
        'Return a circuit with the opposite order of wires.\n\n        The circuit is "vertically" flipped. If a circuit is\n        defined over multiple registers, the resulting circuit will have\n        the same registers but with their order flipped.\n\n        This method is useful for converting a circuit written in little-endian\n        convention to the big-endian equivalent, and vice versa.\n\n        Returns:\n            QuantumCircuit: the circuit with reversed bit order.\n\n        Examples:\n\n            input:\n\n            .. parsed-literal::\n\n                     ┌───┐\n                a_0: ┤ H ├──■─────────────────\n                     └───┘┌─┴─┐\n                a_1: ─────┤ X ├──■────────────\n                          └───┘┌─┴─┐\n                a_2: ──────────┤ X ├──■───────\n                               └───┘┌─┴─┐\n                b_0: ───────────────┤ X ├──■──\n                                    └───┘┌─┴─┐\n                b_1: ────────────────────┤ X ├\n                                         └───┘\n\n            output:\n\n            .. parsed-literal::\n\n                                         ┌───┐\n                b_0: ────────────────────┤ X ├\n                                    ┌───┐└─┬─┘\n                b_1: ───────────────┤ X ├──■──\n                               ┌───┐└─┬─┘\n                a_0: ──────────┤ X ├──■───────\n                          ┌───┐└─┬─┘\n                a_1: ─────┤ X ├──■────────────\n                     ┌───┐└─┬─┘\n                a_2: ┤ H ├──■─────────────────\n                     └───┘\n        '
        circ = QuantumCircuit(list(reversed(self.qubits)), list(reversed(self.clbits)), name=self.name, global_phase=self.global_phase)
        new_qubit_map = circ.qubits[::-1]
        new_clbit_map = circ.clbits[::-1]
        for reg in reversed(self.qregs):
            bits = [new_qubit_map[self.find_bit(qubit).index] for qubit in reversed(reg)]
            circ.add_register(QuantumRegister(bits=bits, name=reg.name))
        for reg in reversed(self.cregs):
            bits = [new_clbit_map[self.find_bit(clbit).index] for clbit in reversed(reg)]
            circ.add_register(ClassicalRegister(bits=bits, name=reg.name))
        for instruction in self.data:
            qubits = [new_qubit_map[self.find_bit(qubit).index] for qubit in instruction.qubits]
            clbits = [new_clbit_map[self.find_bit(clbit).index] for clbit in instruction.clbits]
            circ._append(instruction.replace(qubits=qubits, clbits=clbits))
        return circ

    def inverse(self) -> 'QuantumCircuit':
        if False:
            print('Hello World!')
        'Invert (take adjoint of) this circuit.\n\n        This is done by recursively inverting all gates.\n\n        Returns:\n            QuantumCircuit: the inverted circuit\n\n        Raises:\n            CircuitError: if the circuit cannot be inverted.\n\n        Examples:\n\n            input:\n\n            .. parsed-literal::\n\n                     ┌───┐\n                q_0: ┤ H ├─────■──────\n                     └───┘┌────┴─────┐\n                q_1: ─────┤ RX(1.57) ├\n                          └──────────┘\n\n            output:\n\n            .. parsed-literal::\n\n                                  ┌───┐\n                q_0: ──────■──────┤ H ├\n                     ┌─────┴─────┐└───┘\n                q_1: ┤ RX(-1.57) ├─────\n                     └───────────┘\n        '
        inverse_circ = QuantumCircuit(self.qubits, self.clbits, *self.qregs, *self.cregs, name=self.name + '_dg', global_phase=-self.global_phase)
        for instruction in reversed(self._data):
            inverse_circ._append(instruction.replace(operation=instruction.operation.inverse()))
        return inverse_circ

    def repeat(self, reps: int) -> 'QuantumCircuit':
        if False:
            for i in range(10):
                print('nop')
        'Repeat this circuit ``reps`` times.\n\n        Args:\n            reps (int): How often this circuit should be repeated.\n\n        Returns:\n            QuantumCircuit: A circuit containing ``reps`` repetitions of this circuit.\n        '
        repeated_circ = QuantumCircuit(self.qubits, self.clbits, *self.qregs, *self.cregs, name=self.name + f'**{reps}')
        if reps > 0:
            try:
                inst: Instruction = self.to_gate()
            except QiskitError:
                inst = self.to_instruction()
            for _ in range(reps):
                repeated_circ._append(inst, self.qubits, self.clbits)
        return repeated_circ

    def power(self, power: float, matrix_power: bool=False) -> 'QuantumCircuit':
        if False:
            while True:
                i = 10
        'Raise this circuit to the power of ``power``.\n\n        If ``power`` is a positive integer and ``matrix_power`` is ``False``, this implementation\n        defaults to calling ``repeat``. Otherwise, if the circuit is unitary, the matrix is\n        computed to calculate the matrix power.\n\n        Args:\n            power (float): The power to raise this circuit to.\n            matrix_power (bool): If True, the circuit is converted to a matrix and then the\n                matrix power is computed. If False, and ``power`` is a positive integer,\n                the implementation defaults to ``repeat``.\n\n        Raises:\n            CircuitError: If the circuit needs to be converted to a gate but it is not unitary.\n\n        Returns:\n            QuantumCircuit: A circuit implementing this circuit raised to the power of ``power``.\n        '
        if power >= 0 and isinstance(power, (int, np.integer)) and (not matrix_power):
            return self.repeat(power)
        if self.num_parameters > 0:
            raise CircuitError('Cannot raise a parameterized circuit to a non-positive power or matrix-power, please bind the free parameters: {}'.format(self.parameters))
        try:
            gate = self.to_gate()
        except QiskitError as ex:
            raise CircuitError('The circuit contains non-unitary operations and cannot be controlled. Note that no qiskit.circuit.Instruction objects may be in the circuit for this operation.') from ex
        power_circuit = QuantumCircuit(self.qubits, self.clbits, *self.qregs, *self.cregs)
        power_circuit.append(gate.power(power), list(range(gate.num_qubits)))
        return power_circuit

    def control(self, num_ctrl_qubits: int=1, label: str | None=None, ctrl_state: str | int | None=None) -> 'QuantumCircuit':
        if False:
            while True:
                i = 10
        "Control this circuit on ``num_ctrl_qubits`` qubits.\n\n        Args:\n            num_ctrl_qubits (int): The number of control qubits.\n            label (str): An optional label to give the controlled operation for visualization.\n            ctrl_state (str or int): The control state in decimal or as a bitstring\n                (e.g. '111'). If None, use ``2**num_ctrl_qubits - 1``.\n\n        Returns:\n            QuantumCircuit: The controlled version of this circuit.\n\n        Raises:\n            CircuitError: If the circuit contains a non-unitary operation and cannot be controlled.\n        "
        try:
            gate = self.to_gate()
        except QiskitError as ex:
            raise CircuitError('The circuit contains non-unitary operations and cannot be controlled. Note that no qiskit.circuit.Instruction objects may be in the circuit for this operation.') from ex
        controlled_gate = gate.control(num_ctrl_qubits, label, ctrl_state)
        control_qreg = QuantumRegister(num_ctrl_qubits)
        controlled_circ = QuantumCircuit(control_qreg, self.qubits, *self.qregs, name=f'c_{self.name}')
        controlled_circ.append(controlled_gate, controlled_circ.qubits)
        return controlled_circ

    def compose(self, other: Union['QuantumCircuit', Instruction], qubits: QubitSpecifier | Sequence[QubitSpecifier] | None=None, clbits: ClbitSpecifier | Sequence[ClbitSpecifier] | None=None, front: bool=False, inplace: bool=False, wrap: bool=False) -> Optional['QuantumCircuit']:
        if False:
            i = 10
            return i + 15
        'Compose circuit with ``other`` circuit or instruction, optionally permuting wires.\n\n        ``other`` can be narrower or of equal width to ``self``.\n\n        Args:\n            other (qiskit.circuit.Instruction or QuantumCircuit):\n                (sub)circuit or instruction to compose onto self.  If not a :obj:`.QuantumCircuit`,\n                this can be anything that :obj:`.append` will accept.\n            qubits (list[Qubit|int]): qubits of self to compose onto.\n            clbits (list[Clbit|int]): clbits of self to compose onto.\n            front (bool): If True, front composition will be performed.  This is not possible within\n                control-flow builder context managers.\n            inplace (bool): If True, modify the object. Otherwise return composed circuit.\n            wrap (bool): If True, wraps the other circuit into a gate (or instruction, depending on\n                whether it contains only unitary instructions) before composing it onto self.\n\n        Returns:\n            QuantumCircuit: the composed circuit (returns None if inplace==True).\n\n        Raises:\n            CircuitError: if no correct wire mapping can be made between the two circuits, such as\n                if ``other`` is wider than ``self``.\n            CircuitError: if trying to emit a new circuit while ``self`` has a partially built\n                control-flow context active, such as the context-manager forms of :meth:`if_test`,\n                :meth:`for_loop` and :meth:`while_loop`.\n            CircuitError: if trying to compose to the front of a circuit when a control-flow builder\n                block is active; there is no clear meaning to this action.\n\n        Examples:\n            .. code-block:: python\n\n                >>> lhs.compose(rhs, qubits=[3, 2], inplace=True)\n\n            .. parsed-literal::\n\n                            ┌───┐                   ┌─────┐                ┌───┐\n                lqr_1_0: ───┤ H ├───    rqr_0: ──■──┤ Tdg ├    lqr_1_0: ───┤ H ├───────────────\n                            ├───┤              ┌─┴─┐└─────┘                ├───┤\n                lqr_1_1: ───┤ X ├───    rqr_1: ┤ X ├───────    lqr_1_1: ───┤ X ├───────────────\n                         ┌──┴───┴──┐           └───┘                    ┌──┴───┴──┐┌───┐\n                lqr_1_2: ┤ U1(0.1) ├  +                     =  lqr_1_2: ┤ U1(0.1) ├┤ X ├───────\n                         └─────────┘                                    └─────────┘└─┬─┘┌─────┐\n                lqr_2_0: ─────■─────                           lqr_2_0: ─────■───────■──┤ Tdg ├\n                            ┌─┴─┐                                          ┌─┴─┐        └─────┘\n                lqr_2_1: ───┤ X ├───                           lqr_2_1: ───┤ X ├───────────────\n                            └───┘                                          └───┘\n                lcr_0: 0 ═══════════                           lcr_0: 0 ═══════════════════════\n\n                lcr_1: 0 ═══════════                           lcr_1: 0 ═══════════════════════\n\n        '
        from qiskit.circuit.controlflow.switch_case import SwitchCaseOp
        if inplace and front and self._control_flow_scopes:
            raise CircuitError('Cannot compose to the front of a circuit while a control-flow context is active.')
        if not inplace and self._control_flow_scopes:
            raise CircuitError('Cannot emit a new composed circuit while a control-flow context is active.')
        dest = self if inplace else self.copy()
        if isinstance(other, QuantumCircuit):
            if not self.clbits and other.clbits:
                dest.add_bits(other.clbits)
                for reg in other.cregs:
                    dest.add_register(reg)
        if wrap and isinstance(other, QuantumCircuit):
            other = other.to_gate() if all((isinstance(ins.operation, Gate) for ins in other.data)) else other.to_instruction()
        if not isinstance(other, QuantumCircuit):
            if qubits is None:
                qubits = self.qubits[:other.num_qubits]
            if clbits is None:
                clbits = self.clbits[:other.num_clbits]
            if front:
                old_data = list(dest.data)
                dest.clear()
                dest.append(other, qubits, clbits)
                for instruction in old_data:
                    dest._append(instruction)
            else:
                dest.append(other, qargs=qubits, cargs=clbits)
            if inplace:
                return None
            return dest
        if other.num_qubits > dest.num_qubits or other.num_clbits > dest.num_clbits:
            raise CircuitError("Trying to compose with another QuantumCircuit which has more 'in' edges.")
        edge_map: dict[Qubit | Clbit, Qubit | Clbit] = {}
        if qubits is None:
            edge_map.update(zip(other.qubits, dest.qubits))
        else:
            mapped_qubits = dest.qbit_argument_conversion(qubits)
            if len(mapped_qubits) != len(other.qubits):
                raise CircuitError(f'Number of items in qubits parameter ({len(mapped_qubits)}) does not match number of qubits in the circuit ({len(other.qubits)}).')
            edge_map.update(zip(other.qubits, mapped_qubits))
        if clbits is None:
            edge_map.update(zip(other.clbits, dest.clbits))
        else:
            mapped_clbits = dest.cbit_argument_conversion(clbits)
            if len(mapped_clbits) != len(other.clbits):
                raise CircuitError(f'Number of items in clbits parameter ({len(mapped_clbits)}) does not match number of clbits in the circuit ({len(other.clbits)}).')
            edge_map.update(zip(other.clbits, dest.cbit_argument_conversion(clbits)))
        variable_mapper = _classical_resource_map.VariableMapper(dest.cregs, edge_map, dest.add_register)
        mapped_instrs: list[CircuitInstruction] = []
        for instr in other.data:
            n_qargs: list[Qubit] = [edge_map[qarg] for qarg in instr.qubits]
            n_cargs: list[Clbit] = [edge_map[carg] for carg in instr.clbits]
            n_op = instr.operation.copy()
            if (condition := getattr(n_op, 'condition', None)) is not None:
                n_op.condition = variable_mapper.map_condition(condition)
            if isinstance(n_op, SwitchCaseOp):
                n_op.target = variable_mapper.map_target(n_op.target)
            mapped_instrs.append(CircuitInstruction(n_op, n_qargs, n_cargs))
        if front:
            mapped_instrs += dest.data
            dest.clear()
        append = dest._control_flow_scopes[-1].append if dest._control_flow_scopes else dest._append
        for instr in mapped_instrs:
            append(instr)
        for (gate, cals) in other.calibrations.items():
            dest._calibrations[gate].update(cals)
        dest.global_phase += other.global_phase
        if inplace:
            return None
        return dest

    def tensor(self, other: 'QuantumCircuit', inplace: bool=False) -> Optional['QuantumCircuit']:
        if False:
            print('Hello World!')
        "Tensor ``self`` with ``other``.\n\n        Remember that in the little-endian convention the leftmost operation will be at the bottom\n        of the circuit. See also\n        `the docs <qiskit.org/documentation/tutorials/circuits/3_summary_of_quantum_operations.html>`__\n        for more information.\n\n        .. parsed-literal::\n\n                 ┌────────┐        ┌─────┐          ┌─────┐\n            q_0: ┤ bottom ├ ⊗ q_0: ┤ top ├  = q_0: ─┤ top ├──\n                 └────────┘        └─────┘         ┌┴─────┴─┐\n                                              q_1: ┤ bottom ├\n                                                   └────────┘\n\n        Args:\n            other (QuantumCircuit): The other circuit to tensor this circuit with.\n            inplace (bool): If True, modify the object. Otherwise return composed circuit.\n\n        Examples:\n\n            .. plot::\n               :include-source:\n\n               from qiskit import QuantumCircuit\n               top = QuantumCircuit(1)\n               top.x(0);\n               bottom = QuantumCircuit(2)\n               bottom.cry(0.2, 0, 1);\n               tensored = bottom.tensor(top)\n               tensored.draw('mpl')\n\n        Returns:\n            QuantumCircuit: The tensored circuit (returns None if inplace==True).\n        "
        num_qubits = self.num_qubits + other.num_qubits
        num_clbits = self.num_clbits + other.num_clbits
        if len(self.qregs) == len(other.qregs) == 1 and self.qregs[0].name == other.qregs[0].name == 'q':
            if num_clbits > 0:
                dest = QuantumCircuit(num_qubits, num_clbits)
            else:
                dest = QuantumCircuit(num_qubits)
        elif len(self.cregs) == len(other.cregs) == 1 and self.cregs[0].name == other.cregs[0].name == 'meas':
            cr = ClassicalRegister(self.num_clbits + other.num_clbits, 'meas')
            dest = QuantumCircuit(*other.qregs, *self.qregs, cr)
        else:
            dest = QuantumCircuit(other.qubits, self.qubits, other.clbits, self.clbits, *other.qregs, *self.qregs, *other.cregs, *self.cregs)
        dest.compose(other, range(other.num_qubits), range(other.num_clbits), inplace=True)
        dest.compose(self, range(other.num_qubits, num_qubits), range(other.num_clbits, num_clbits), inplace=True)
        if inplace:
            self.__dict__.update(dest.__dict__)
            return None
        return dest

    @property
    def qubits(self) -> list[Qubit]:
        if False:
            print('Hello World!')
        '\n        Returns a list of quantum bits in the order that the registers were added.\n        '
        return self._qubits

    @property
    def clbits(self) -> list[Clbit]:
        if False:
            while True:
                i = 10
        '\n        Returns a list of classical bits in the order that the registers were added.\n        '
        return self._clbits

    @property
    def ancillas(self) -> list[AncillaQubit]:
        if False:
            i = 10
            return i + 15
        '\n        Returns a list of ancilla bits in the order that the registers were added.\n        '
        return self._ancillas

    def __and__(self, rhs: 'QuantumCircuit') -> 'QuantumCircuit':
        if False:
            return 10
        'Overload & to implement self.compose.'
        return self.compose(rhs)

    def __iand__(self, rhs: 'QuantumCircuit') -> 'QuantumCircuit':
        if False:
            for i in range(10):
                print('nop')
        'Overload &= to implement self.compose in place.'
        self.compose(rhs, inplace=True)
        return self

    def __xor__(self, top: 'QuantumCircuit') -> 'QuantumCircuit':
        if False:
            i = 10
            return i + 15
        'Overload ^ to implement self.tensor.'
        return self.tensor(top)

    def __ixor__(self, top: 'QuantumCircuit') -> 'QuantumCircuit':
        if False:
            i = 10
            return i + 15
        'Overload ^= to implement self.tensor in place.'
        self.tensor(top, inplace=True)
        return self

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        'Return number of operations in circuit.'
        return len(self._data)

    @typing.overload
    def __getitem__(self, item: int) -> CircuitInstruction:
        if False:
            while True:
                i = 10
        ...

    @typing.overload
    def __getitem__(self, item: slice) -> list[CircuitInstruction]:
        if False:
            print('Hello World!')
        ...

    def __getitem__(self, item):
        if False:
            return 10
        'Return indexed operation.'
        return self._data[item]

    @staticmethod
    def cast(value: S, type_: Callable[..., T]) -> Union[S, T]:
        if False:
            for i in range(10):
                print('nop')
        'Best effort to cast value to type. Otherwise, returns the value.'
        try:
            return type_(value)
        except (ValueError, TypeError):
            return value

    def qbit_argument_conversion(self, qubit_representation: QubitSpecifier) -> list[Qubit]:
        if False:
            while True:
                i = 10
        '\n        Converts several qubit representations (such as indexes, range, etc.)\n        into a list of qubits.\n\n        Args:\n            qubit_representation (Object): representation to expand\n\n        Returns:\n            List(Qubit): the resolved instances of the qubits.\n        '
        return _bit_argument_conversion(qubit_representation, self.qubits, self._qubit_indices, Qubit)

    def cbit_argument_conversion(self, clbit_representation: ClbitSpecifier) -> list[Clbit]:
        if False:
            print('Hello World!')
        '\n        Converts several classical bit representations (such as indexes, range, etc.)\n        into a list of classical bits.\n\n        Args:\n            clbit_representation (Object): representation to expand\n\n        Returns:\n            List(tuple): Where each tuple is a classical bit.\n        '
        return _bit_argument_conversion(clbit_representation, self.clbits, self._clbit_indices, Clbit)

    def _resolve_classical_resource(self, specifier):
        if False:
            return 10
        'Resolve a single classical resource specifier into a concrete resource, raising an error\n        if the specifier is invalid.\n\n        This is slightly different to :meth:`.cbit_argument_conversion`, because it should not\n        unwrap :obj:`.ClassicalRegister` instances into lists, and in general it should not allow\n        iterables or broadcasting.  It is expected to be used as a callback for things like\n        :meth:`.InstructionSet.c_if` to check the validity of their arguments.\n\n        Args:\n            specifier (Union[Clbit, ClassicalRegister, int]): a specifier of a classical resource\n                present in this circuit.  An ``int`` will be resolved into a :obj:`.Clbit` using the\n                same conventions as measurement operations on this circuit use.\n\n        Returns:\n            Union[Clbit, ClassicalRegister]: the resolved resource.\n\n        Raises:\n            CircuitError: if the resource is not present in this circuit, or if the integer index\n                passed is out-of-bounds.\n        '
        if isinstance(specifier, Clbit):
            if specifier not in self._clbit_indices:
                raise CircuitError(f'Clbit {specifier} is not present in this circuit.')
            return specifier
        if isinstance(specifier, ClassicalRegister):
            if specifier not in self.cregs:
                raise CircuitError(f'Register {specifier} is not present in this circuit.')
            return specifier
        if isinstance(specifier, int):
            try:
                return self._clbits[specifier]
            except IndexError:
                raise CircuitError(f'Classical bit index {specifier} is out-of-range.') from None
        raise CircuitError(f"Unknown classical resource specifier: '{specifier}'.")

    def _validate_expr(self, node: expr.Expr) -> expr.Expr:
        if False:
            return 10
        for var in expr.iter_vars(node):
            if isinstance(var.var, Clbit):
                if var.var not in self._clbit_indices:
                    raise CircuitError(f'Clbit {var.var} is not present in this circuit.')
            elif isinstance(var.var, ClassicalRegister):
                if var.var not in self.cregs:
                    raise CircuitError(f'Register {var.var} is not present in this circuit.')
        return node

    def append(self, instruction: Operation | CircuitInstruction, qargs: Sequence[QubitSpecifier] | None=None, cargs: Sequence[ClbitSpecifier] | None=None) -> InstructionSet:
        if False:
            i = 10
            return i + 15
        'Append one or more instructions to the end of the circuit, modifying the circuit in\n        place.\n\n        The ``qargs`` and ``cargs`` will be expanded and broadcast according to the rules of the\n        given :class:`~.circuit.Instruction`, and any non-:class:`.Bit` specifiers (such as\n        integer indices) will be resolved into the relevant instances.\n\n        If a :class:`.CircuitInstruction` is given, it will be unwrapped, verified in the context of\n        this circuit, and a new object will be appended to the circuit.  In this case, you may not\n        pass ``qargs`` or ``cargs`` separately.\n\n        Args:\n            instruction: :class:`~.circuit.Instruction` instance to append, or a\n                :class:`.CircuitInstruction` with all its context.\n            qargs: specifiers of the :class:`~.circuit.Qubit`\\ s to attach instruction to.\n            cargs: specifiers of the :class:`.Clbit`\\ s to attach instruction to.\n\n        Returns:\n            qiskit.circuit.InstructionSet: a handle to the :class:`.CircuitInstruction`\\ s that\n            were actually added to the circuit.\n\n        Raises:\n            CircuitError: if the operation passed is not an instance of :class:`~.circuit.Instruction` .\n        '
        if isinstance(instruction, CircuitInstruction):
            operation = instruction.operation
            qargs = instruction.qubits
            cargs = instruction.clbits
        else:
            operation = instruction
        if not isinstance(operation, Operation):
            if hasattr(operation, 'to_instruction'):
                operation = operation.to_instruction()
                if not isinstance(operation, Operation):
                    raise CircuitError('operation.to_instruction() is not an Operation.')
            else:
                if issubclass(operation, Operation):
                    raise CircuitError('Object is a subclass of Operation, please add () to pass an instance of this object.')
                raise CircuitError('Object to append must be an Operation or have a to_instruction() method.')
        if hasattr(operation, 'params'):
            is_parameter = any((isinstance(param, Parameter) for param in operation.params))
            if is_parameter:
                operation = copy.deepcopy(operation)
        expanded_qargs = [self.qbit_argument_conversion(qarg) for qarg in qargs or []]
        expanded_cargs = [self.cbit_argument_conversion(carg) for carg in cargs or []]
        if self._control_flow_scopes:
            appender = self._control_flow_scopes[-1].append
            requester = self._control_flow_scopes[-1].request_classical_resource
        else:
            appender = self._append
            requester = self._resolve_classical_resource
        instructions = InstructionSet(resource_requester=requester)
        if isinstance(operation, Instruction):
            for (qarg, carg) in operation.broadcast_arguments(expanded_qargs, expanded_cargs):
                self._check_dups(qarg)
                instruction = CircuitInstruction(operation, qarg, carg)
                appender(instruction)
                instructions.add(instruction)
        else:
            for (qarg, carg) in Instruction.broadcast_arguments(operation, expanded_qargs, expanded_cargs):
                self._check_dups(qarg)
                instruction = CircuitInstruction(operation, qarg, carg)
                appender(instruction)
                instructions.add(instruction)
        return instructions

    @typing.overload
    def _append(self, instruction: CircuitInstruction, _qargs: None=None, _cargs: None=None) -> CircuitInstruction:
        if False:
            i = 10
            return i + 15
        ...

    @typing.overload
    def _append(self, operation: Operation, qargs: Sequence[Qubit], cargs: Sequence[Clbit]) -> Operation:
        if False:
            print('Hello World!')
        ...

    def _append(self, instruction: CircuitInstruction | Instruction, qargs: Sequence[Qubit] | None=None, cargs: Sequence[Clbit] | None=None):
        if False:
            i = 10
            return i + 15
        'Append an instruction to the end of the circuit, modifying the circuit in place.\n\n        .. warning::\n\n            This is an internal fast-path function, and it is the responsibility of the caller to\n            ensure that all the arguments are valid; there is no error checking here.  In\n            particular, all the qubits and clbits must already exist in the circuit and there can be\n            no duplicates in the list.\n\n        .. note::\n\n            This function may be used by callers other than :obj:`.QuantumCircuit` when the caller\n            is sure that all error-checking, broadcasting and scoping has already been performed,\n            and the only reference to the circuit the instructions are being appended to is within\n            that same function.  In particular, it is not safe to call\n            :meth:`QuantumCircuit._append` on a circuit that is received by a function argument.\n            This is because :meth:`.QuantumCircuit._append` will not recognise the scoping\n            constructs of the control-flow builder interface.\n\n        Args:\n            instruction: Operation instance to append\n            qargs: Qubits to attach the instruction to.\n            cargs: Clbits to attach the instruction to.\n\n        Returns:\n            Operation: a handle to the instruction that was just added\n\n        :meta public:\n        '
        old_style = not isinstance(instruction, CircuitInstruction)
        if old_style:
            instruction = CircuitInstruction(instruction, qargs, cargs)
        self._data.append(instruction)
        if isinstance(instruction.operation, Instruction):
            self._update_parameter_table(instruction)
        self.duration = None
        self.unit = 'dt'
        return instruction.operation if old_style else instruction

    def _update_parameter_table(self, instruction: CircuitInstruction):
        if False:
            i = 10
            return i + 15
        for (param_index, param) in enumerate(instruction.operation.params):
            if isinstance(param, (ParameterExpression, QuantumCircuit)):
                atomic_parameters = set(param.parameters)
            else:
                atomic_parameters = set()
            for parameter in atomic_parameters:
                if parameter in self._parameter_table:
                    self._parameter_table[parameter].add((instruction.operation, param_index))
                else:
                    if parameter.name in self._parameter_table.get_names():
                        raise CircuitError(f'Name conflict on adding parameter: {parameter.name}')
                    self._parameter_table[parameter] = ParameterReferences(((instruction.operation, param_index),))
                    self._parameters = None

    def add_register(self, *regs: Register | int | Sequence[Bit]) -> None:
        if False:
            return 10
        'Add registers.'
        if not regs:
            return
        if any((isinstance(reg, int) for reg in regs)):
            if len(regs) == 1 and isinstance(regs[0], int):
                if regs[0] == 0:
                    regs = ()
                else:
                    regs = (QuantumRegister(regs[0], 'q'),)
            elif len(regs) == 2 and all((isinstance(reg, int) for reg in regs)):
                if regs[0] == 0:
                    qregs: tuple[QuantumRegister, ...] = ()
                else:
                    qregs = (QuantumRegister(regs[0], 'q'),)
                if regs[1] == 0:
                    cregs: tuple[ClassicalRegister, ...] = ()
                else:
                    cregs = (ClassicalRegister(regs[1], 'c'),)
                regs = qregs + cregs
            else:
                raise CircuitError('QuantumCircuit parameters can be Registers or Integers. If Integers, up to 2 arguments. QuantumCircuit was called with %s.' % (regs,))
        for register in regs:
            if isinstance(register, Register) and any((register.name == reg.name for reg in self.qregs + self.cregs)):
                raise CircuitError('register name "%s" already exists' % register.name)
            if isinstance(register, AncillaRegister):
                for bit in register:
                    if bit not in self._qubit_indices:
                        self._ancillas.append(bit)
            if isinstance(register, QuantumRegister):
                self.qregs.append(register)
                for (idx, bit) in enumerate(register):
                    if bit in self._qubit_indices:
                        self._qubit_indices[bit].registers.append((register, idx))
                    else:
                        self._qubits.append(bit)
                        self._qubit_indices[bit] = BitLocations(len(self._qubits) - 1, [(register, idx)])
            elif isinstance(register, ClassicalRegister):
                self.cregs.append(register)
                for (idx, bit) in enumerate(register):
                    if bit in self._clbit_indices:
                        self._clbit_indices[bit].registers.append((register, idx))
                    else:
                        self._clbits.append(bit)
                        self._clbit_indices[bit] = BitLocations(len(self._clbits) - 1, [(register, idx)])
            elif isinstance(register, list):
                self.add_bits(register)
            else:
                raise CircuitError('expected a register')

    def add_bits(self, bits: Iterable[Bit]) -> None:
        if False:
            return 10
        'Add Bits to the circuit.'
        duplicate_bits = set(self._qubit_indices).union(self._clbit_indices).intersection(bits)
        if duplicate_bits:
            raise CircuitError(f'Attempted to add bits found already in circuit: {duplicate_bits}')
        for bit in bits:
            if isinstance(bit, AncillaQubit):
                self._ancillas.append(bit)
            if isinstance(bit, Qubit):
                self._qubits.append(bit)
                self._qubit_indices[bit] = BitLocations(len(self._qubits) - 1, [])
            elif isinstance(bit, Clbit):
                self._clbits.append(bit)
                self._clbit_indices[bit] = BitLocations(len(self._clbits) - 1, [])
            else:
                raise CircuitError('Expected an instance of Qubit, Clbit, or AncillaQubit, but was passed {}'.format(bit))

    def find_bit(self, bit: Bit) -> BitLocations:
        if False:
            i = 10
            return i + 15
        'Find locations in the circuit which can be used to reference a given :obj:`~Bit`.\n\n        Args:\n            bit (Bit): The bit to locate.\n\n        Returns:\n            namedtuple(int, List[Tuple(Register, int)]): A 2-tuple. The first element (``index``)\n                contains the index at which the ``Bit`` can be found (in either\n                :obj:`~QuantumCircuit.qubits`, :obj:`~QuantumCircuit.clbits`, depending on its\n                type). The second element (``registers``) is a list of ``(register, index)``\n                pairs with an entry for each :obj:`~Register` in the circuit which contains the\n                :obj:`~Bit` (and the index in the :obj:`~Register` at which it can be found).\n\n        Notes:\n            The circuit index of an :obj:`~AncillaQubit` will be its index in\n            :obj:`~QuantumCircuit.qubits`, not :obj:`~QuantumCircuit.ancillas`.\n\n        Raises:\n            CircuitError: If the supplied :obj:`~Bit` was of an unknown type.\n            CircuitError: If the supplied :obj:`~Bit` could not be found on the circuit.\n        '
        try:
            if isinstance(bit, Qubit):
                return self._qubit_indices[bit]
            elif isinstance(bit, Clbit):
                return self._clbit_indices[bit]
            else:
                raise CircuitError(f'Could not locate bit of unknown type: {type(bit)}')
        except KeyError as err:
            raise CircuitError(f'Could not locate provided bit: {bit}. Has it been added to the QuantumCircuit?') from err

    def _check_dups(self, qubits: Sequence[Qubit]) -> None:
        if False:
            print('Hello World!')
        'Raise exception if list of qubits contains duplicates.'
        squbits = set(qubits)
        if len(squbits) != len(qubits):
            raise CircuitError('duplicate qubit arguments')

    def to_instruction(self, parameter_map: dict[Parameter, ParameterValueType] | None=None, label: str | None=None) -> Instruction:
        if False:
            print('Hello World!')
        'Create an Instruction out of this circuit.\n\n        Args:\n            parameter_map(dict): For parameterized circuits, a mapping from\n               parameters in the circuit to parameters to be used in the\n               instruction. If None, existing circuit parameters will also\n               parameterize the instruction.\n            label (str): Optional gate label.\n\n        Returns:\n            qiskit.circuit.Instruction: a composite instruction encapsulating this circuit\n            (can be decomposed back)\n        '
        from qiskit.converters.circuit_to_instruction import circuit_to_instruction
        return circuit_to_instruction(self, parameter_map, label=label)

    def to_gate(self, parameter_map: dict[Parameter, ParameterValueType] | None=None, label: str | None=None) -> Gate:
        if False:
            i = 10
            return i + 15
        'Create a Gate out of this circuit.\n\n        Args:\n            parameter_map(dict): For parameterized circuits, a mapping from\n               parameters in the circuit to parameters to be used in the\n               gate. If None, existing circuit parameters will also\n               parameterize the gate.\n            label (str): Optional gate label.\n\n        Returns:\n            Gate: a composite gate encapsulating this circuit\n            (can be decomposed back)\n        '
        from qiskit.converters.circuit_to_gate import circuit_to_gate
        return circuit_to_gate(self, parameter_map, label=label)

    def decompose(self, gates_to_decompose: Type[Gate] | Sequence[Type[Gate]] | Sequence[str] | str | None=None, reps: int=1) -> 'QuantumCircuit':
        if False:
            for i in range(10):
                print('nop')
        "Call a decomposition pass on this circuit,\n        to decompose one level (shallow decompose).\n\n        Args:\n            gates_to_decompose (type or str or list(type, str)): Optional subset of gates\n                to decompose. Can be a gate type, such as ``HGate``, or a gate name, such\n                as 'h', or a gate label, such as 'My H Gate', or a list of any combination\n                of these. If a gate name is entered, it will decompose all gates with that\n                name, whether the gates have labels or not. Defaults to all gates in circuit.\n            reps (int): Optional number of times the circuit should be decomposed.\n                For instance, ``reps=2`` equals calling ``circuit.decompose().decompose()``.\n                can decompose specific gates specific time\n\n        Returns:\n            QuantumCircuit: a circuit one level decomposed\n        "
        from qiskit.transpiler.passes.basis.decompose import Decompose
        from qiskit.transpiler.passes.synthesis import HighLevelSynthesis
        from qiskit.converters.circuit_to_dag import circuit_to_dag
        from qiskit.converters.dag_to_circuit import dag_to_circuit
        dag = circuit_to_dag(self, copy_operations=True)
        dag = HighLevelSynthesis().run(dag)
        pass_ = Decompose(gates_to_decompose)
        for _ in range(reps):
            dag = pass_.run(dag)
        return dag_to_circuit(dag, copy_operations=False)

    def qasm(self, formatted: bool=False, filename: str | None=None, encoding: str | None=None) -> str | None:
        if False:
            while True:
                i = 10
        "Return OpenQASM 2.0 string.\n\n        .. seealso::\n\n            :func:`.qasm2.dump` and :func:`.qasm2.dumps`\n                The preferred entry points to the OpenQASM 2 export capabilities.  These match the\n                interface for other serialisers in Qiskit.\n\n        Args:\n            formatted (bool): Return formatted OpenQASM 2.0 string.\n            filename (str): Save OpenQASM 2.0 to file with name 'filename'.\n            encoding (str): Optionally specify the encoding to use for the\n                output file if ``filename`` is specified. By default this is\n                set to the system's default encoding (ie whatever\n                ``locale.getpreferredencoding()`` returns) and can be set to\n                any valid codec or alias from stdlib's\n                `codec module <https://docs.python.org/3/library/codecs.html#standard-encodings>`__\n\n        Returns:\n            str: If formatted=False.\n\n        Raises:\n            MissingOptionalLibraryError: If pygments is not installed and ``formatted`` is\n                ``True``.\n            QASM2ExportError: If circuit has free parameters.\n            QASM2ExportError: If an operation that has no OpenQASM 2 representation is encountered.\n        "
        from qiskit import qasm2
        out = qasm2.dumps(self)
        if filename is not None:
            with open(filename, 'w+', encoding=encoding) as file:
                print(out, file=file)
        if formatted:
            _optionals.HAS_PYGMENTS.require_now('formatted OpenQASM 2.0 output')
            import pygments
            from pygments.formatters import Terminal256Formatter
            from qiskit.qasm.pygments import OpenQASMLexer
            from qiskit.qasm.pygments import QasmTerminalStyle
            code = pygments.highlight(out, OpenQASMLexer(), Terminal256Formatter(style=QasmTerminalStyle))
            print(code)
            return None
        return out + '\n'

    def draw(self, output: str | None=None, scale: float | None=None, filename: str | None=None, style: dict | str | None=None, interactive: bool=False, plot_barriers: bool=True, reverse_bits: bool=None, justify: str | None=None, vertical_compression: str | None='medium', idle_wires: bool=True, with_layout: bool=True, fold: int | None=None, ax: Any | None=None, initial_state: bool=False, cregbundle: bool=None, wire_order: list=None, expr_len: int=30):
        if False:
            i = 10
            return i + 15
        'Draw the quantum circuit. Use the output parameter to choose the drawing format:\n\n        **text**: ASCII art TextDrawing that can be printed in the console.\n\n        **mpl**: images with color rendered purely in Python using matplotlib.\n\n        **latex**: high-quality images compiled via latex.\n\n        **latex_source**: raw uncompiled latex output.\n\n        .. warning::\n\n            Support for :class:`~.expr.Expr` nodes in conditions and :attr:`.SwitchCaseOp.target`\n            fields is preliminary and incomplete.  The ``text`` and ``mpl`` drawers will make a\n            best-effort attempt to show data dependencies, but the LaTeX-based drawers will skip\n            these completely.\n\n        Args:\n            output (str): select the output method to use for drawing the circuit.\n                Valid choices are ``text``, ``mpl``, ``latex``, ``latex_source``.\n                By default the `text` drawer is used unless the user config file\n                (usually ``~/.qiskit/settings.conf``) has an alternative backend set\n                as the default. For example, ``circuit_drawer = latex``. If the output\n                kwarg is set, that backend will always be used over the default in\n                the user config file.\n            scale (float): scale of image to draw (shrink if < 1.0). Only used by\n                the `mpl`, `latex` and `latex_source` outputs. Defaults to 1.0.\n            filename (str): file path to save image to. Defaults to None.\n            style (dict or str): dictionary of style or file name of style json file.\n                This option is only used by the `mpl` or `latex` output type.\n                If `style` is a str, it is used as the path to a json file\n                which contains a style dict. The file will be opened, parsed, and\n                then any style elements in the dict will replace the default values\n                in the input dict. A file to be loaded must end in ``.json``, but\n                the name entered here can omit ``.json``. For example,\n                ``style=\'iqp.json\'`` or ``style=\'iqp\'``.\n                If `style` is a dict and the ``\'name\'`` key is set, that name\n                will be used to load a json file, followed by loading the other\n                items in the style dict. For example, ``style={\'name\': \'iqp\'}``.\n                If `style` is not a str and `name` is not a key in the style dict,\n                then the default value from the user config file (usually\n                ``~/.qiskit/settings.conf``) will be used, for example,\n                ``circuit_mpl_style = iqp``.\n                If none of these are set, the `clifford` style will be used.\n                The search path for style json files can be specified in the user\n                config, for example,\n                ``circuit_mpl_style_path = /home/user/styles:/home/user``.\n                See: :class:`~qiskit.visualization.qcstyle.DefaultStyle` for more\n                information on the contents.\n            interactive (bool): when set to true, show the circuit in a new window\n                (for `mpl` this depends on the matplotlib backend being used\n                supporting this). Note when used with either the `text` or the\n                `latex_source` output type this has no effect and will be silently\n                ignored. Defaults to False.\n            reverse_bits (bool): when set to True, reverse the bit order inside\n                registers for the output visualization. Defaults to False unless the\n                user config file (usually ``~/.qiskit/settings.conf``) has an\n                alternative value set. For example, ``circuit_reverse_bits = True``.\n            plot_barriers (bool): enable/disable drawing barriers in the output\n                circuit. Defaults to True.\n            justify (string): options are ``left``, ``right`` or ``none``. If\n                anything else is supplied, it defaults to left justified. It refers\n                to where gates should be placed in the output circuit if there is\n                an option. ``none`` results in each gate being placed in its own\n                column.\n            vertical_compression (string): ``high``, ``medium`` or ``low``. It\n                merges the lines generated by the `text` output so the drawing\n                will take less vertical room.  Default is ``medium``. Only used by\n                the `text` output, will be silently ignored otherwise.\n            idle_wires (bool): include idle wires (wires with no circuit elements)\n                in output visualization. Default is True.\n            with_layout (bool): include layout information, with labels on the\n                physical layout. Default is True.\n            fold (int): sets pagination. It can be disabled using -1. In `text`,\n                sets the length of the lines. This is useful when the drawing does\n                not fit in the console. If None (default), it will try to guess the\n                console width using ``shutil.get_terminal_size()``. However, if\n                running in jupyter, the default line length is set to 80 characters.\n                In `mpl`, it is the number of (visual) layers before folding.\n                Default is 25.\n            ax (matplotlib.axes.Axes): Only used by the `mpl` backend. An optional\n                Axes object to be used for the visualization output. If none is\n                specified, a new matplotlib Figure will be created and used.\n                Additionally, if specified there will be no returned Figure since\n                it is redundant.\n            initial_state (bool): Optional. Adds ``|0>`` in the beginning of the wire.\n                Default is False.\n            cregbundle (bool): Optional. If set True, bundle classical registers.\n                Default is True, except for when ``output`` is set to  ``"text"``.\n            wire_order (list): Optional. A list of integers used to reorder the display\n                of the bits. The list must have an entry for every bit with the bits\n                in the range 0 to (``num_qubits`` + ``num_clbits``).\n            expr_len (int): Optional. The number of characters to display if an :class:`~.expr.Expr`\n                is used for the condition in a :class:`.ControlFlowOp`. If this number is exceeded,\n                the string will be truncated at that number and \'...\' added to the end.\n\n        Returns:\n            :class:`.TextDrawing` or :class:`matplotlib.figure` or :class:`PIL.Image` or\n            :class:`str`:\n\n            * `TextDrawing` (output=\'text\')\n                A drawing that can be printed as ascii art.\n            * `matplotlib.figure.Figure` (output=\'mpl\')\n                A matplotlib figure object for the circuit diagram.\n            * `PIL.Image` (output=\'latex\')\n                An in-memory representation of the image of the circuit diagram.\n            * `str` (output=\'latex_source\')\n                The LaTeX source code for visualizing the circuit diagram.\n\n        Raises:\n            VisualizationError: when an invalid output method is selected\n            ImportError: when the output methods requires non-installed libraries.\n\n        Example:\n            .. plot::\n               :include-source:\n\n               from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit\n               q = QuantumRegister(1)\n               c = ClassicalRegister(1)\n               qc = QuantumCircuit(q, c)\n               qc.h(q)\n               qc.measure(q, c)\n               qc.draw(output=\'mpl\', style={\'backgroundcolor\': \'#EEEEEE\'})\n        '
        from qiskit.visualization import circuit_drawer
        return circuit_drawer(self, scale=scale, filename=filename, style=style, output=output, interactive=interactive, plot_barriers=plot_barriers, reverse_bits=reverse_bits, justify=justify, vertical_compression=vertical_compression, idle_wires=idle_wires, with_layout=with_layout, fold=fold, ax=ax, initial_state=initial_state, cregbundle=cregbundle, wire_order=wire_order, expr_len=expr_len)

    def size(self, filter_function: Callable[..., int]=lambda x: not getattr(x.operation, '_directive', False)) -> int:
        if False:
            i = 10
            return i + 15
        'Returns total number of instructions in circuit.\n\n        Args:\n            filter_function (callable): a function to filter out some instructions.\n                Should take as input a tuple of (Instruction, list(Qubit), list(Clbit)).\n                By default filters out "directives", such as barrier or snapshot.\n\n        Returns:\n            int: Total number of gate operations.\n        '
        return sum(map(filter_function, self._data))

    def depth(self, filter_function: Callable[..., int]=lambda x: not getattr(x.operation, '_directive', False)) -> int:
        if False:
            i = 10
            return i + 15
        'Return circuit depth (i.e., length of critical path).\n\n        Args:\n            filter_function (callable): A function to filter instructions.\n                Should take as input a tuple of (Instruction, list(Qubit), list(Clbit)).\n                Instructions for which the function returns False are ignored in the\n                computation of the circuit depth.\n                By default filters out "directives", such as barrier or snapshot.\n\n        Returns:\n            int: Depth of circuit.\n\n        Notes:\n            The circuit depth and the DAG depth need not be the\n            same.\n        '
        bit_indices: dict[Qubit | Clbit, int] = {bit: idx for (idx, bit) in enumerate(self.qubits + self.clbits)}
        if not bit_indices:
            return 0
        op_stack = [0] * len(bit_indices)
        for instruction in self._data:
            levels = []
            reg_ints = []
            for (ind, reg) in enumerate(instruction.qubits + instruction.clbits):
                reg_ints.append(bit_indices[reg])
                if filter_function(instruction):
                    levels.append(op_stack[reg_ints[ind]] + 1)
                else:
                    levels.append(op_stack[reg_ints[ind]])
            if getattr(instruction.operation, 'condition', None):
                if isinstance(instruction.operation.condition[0], Clbit):
                    condition_bits = [instruction.operation.condition[0]]
                else:
                    condition_bits = instruction.operation.condition[0]
                for cbit in condition_bits:
                    idx = bit_indices[cbit]
                    if idx not in reg_ints:
                        reg_ints.append(idx)
                        levels.append(op_stack[idx] + 1)
            max_level = max(levels)
            for ind in reg_ints:
                op_stack[ind] = max_level
        return max(op_stack)

    def width(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Return number of qubits plus clbits in circuit.\n\n        Returns:\n            int: Width of circuit.\n\n        '
        return len(self.qubits) + len(self.clbits)

    @property
    def num_qubits(self) -> int:
        if False:
            return 10
        'Return number of qubits.'
        return len(self.qubits)

    @property
    def num_ancillas(self) -> int:
        if False:
            while True:
                i = 10
        'Return the number of ancilla qubits.'
        return len(self.ancillas)

    @property
    def num_clbits(self) -> int:
        if False:
            return 10
        'Return number of classical bits.'
        return len(self.clbits)

    def count_ops(self) -> 'OrderedDict[Instruction, int]':
        if False:
            print('Hello World!')
        'Count each operation kind in the circuit.\n\n        Returns:\n            OrderedDict: a breakdown of how many operations of each kind, sorted by amount.\n        '
        count_ops: dict[Instruction, int] = {}
        for instruction in self._data:
            count_ops[instruction.operation.name] = count_ops.get(instruction.operation.name, 0) + 1
        return OrderedDict(sorted(count_ops.items(), key=lambda kv: kv[1], reverse=True))

    def num_nonlocal_gates(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Return number of non-local gates (i.e. involving 2+ qubits).\n\n        Conditional nonlocal gates are also included.\n        '
        multi_qubit_gates = 0
        for instruction in self._data:
            if instruction.operation.num_qubits > 1 and (not getattr(instruction.operation, '_directive', False)):
                multi_qubit_gates += 1
        return multi_qubit_gates

    def get_instructions(self, name: str) -> list[CircuitInstruction]:
        if False:
            return 10
        'Get instructions matching name.\n\n        Args:\n            name (str): The name of instruction to.\n\n        Returns:\n            list(tuple): list of (instruction, qargs, cargs).\n        '
        return [match for match in self._data if match.operation.name == name]

    def num_connected_components(self, unitary_only: bool=False) -> int:
        if False:
            i = 10
            return i + 15
        'How many non-entangled subcircuits can the circuit be factored to.\n\n        Args:\n            unitary_only (bool): Compute only unitary part of graph.\n\n        Returns:\n            int: Number of connected components in circuit.\n        '
        bits = self.qubits if unitary_only else self.qubits + self.clbits
        bit_indices: dict[Qubit | Clbit, int] = {bit: idx for (idx, bit) in enumerate(bits)}
        sub_graphs = [[bit] for bit in range(len(bit_indices))]
        num_sub_graphs = len(sub_graphs)
        for instruction in self._data:
            if unitary_only:
                args = instruction.qubits
                num_qargs = len(args)
            else:
                args = instruction.qubits + instruction.clbits
                num_qargs = len(args) + (1 if getattr(instruction.operation, 'condition', None) else 0)
            if num_qargs >= 2 and (not getattr(instruction.operation, '_directive', False)):
                graphs_touched = []
                num_touched = 0
                if not unitary_only:
                    for bit in instruction.operation.condition_bits:
                        idx = bit_indices[bit]
                        for k in range(num_sub_graphs):
                            if idx in sub_graphs[k]:
                                graphs_touched.append(k)
                                break
                for item in args:
                    reg_int = bit_indices[item]
                    for k in range(num_sub_graphs):
                        if reg_int in sub_graphs[k]:
                            if k not in graphs_touched:
                                graphs_touched.append(k)
                                break
                graphs_touched = list(set(graphs_touched))
                num_touched = len(graphs_touched)
                if num_touched > 1:
                    connections = []
                    for idx in graphs_touched:
                        connections.extend(sub_graphs[idx])
                    _sub_graphs = []
                    for idx in range(num_sub_graphs):
                        if idx not in graphs_touched:
                            _sub_graphs.append(sub_graphs[idx])
                    _sub_graphs.append(connections)
                    sub_graphs = _sub_graphs
                    num_sub_graphs -= num_touched - 1
            if num_sub_graphs == 1:
                break
        return num_sub_graphs

    def num_unitary_factors(self) -> int:
        if False:
            return 10
        'Computes the number of tensor factors in the unitary\n        (quantum) part of the circuit only.\n        '
        return self.num_connected_components(unitary_only=True)

    def num_tensor_factors(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Computes the number of tensor factors in the unitary\n        (quantum) part of the circuit only.\n\n        Notes:\n            This is here for backwards compatibility, and will be\n            removed in a future release of Qiskit. You should call\n            `num_unitary_factors` instead.\n        '
        return self.num_unitary_factors()

    def copy(self, name: str | None=None) -> 'QuantumCircuit':
        if False:
            i = 10
            return i + 15
        'Copy the circuit.\n\n        Args:\n          name (str): name to be given to the copied circuit. If None, then the name stays the same.\n\n        Returns:\n          QuantumCircuit: a deepcopy of the current circuit, with the specified name\n        '
        cpy = self.copy_empty_like(name)
        operation_copies = {id(instruction.operation): instruction.operation.copy() for instruction in self._data}
        cpy._parameter_table = ParameterTable({param: ParameterReferences(((operation_copies[id(operation)], param_index) for (operation, param_index) in self._parameter_table[param])) for param in self._parameter_table})
        cpy._data = [instruction.replace(operation=operation_copies[id(instruction.operation)]) for instruction in self._data]
        return cpy

    def copy_empty_like(self, name: str | None=None) -> 'QuantumCircuit':
        if False:
            for i in range(10):
                print('nop')
        'Return a copy of self with the same structure but empty.\n\n        That structure includes:\n            * name, calibrations and other metadata\n            * global phase\n            * all the qubits and clbits, including the registers\n\n        Args:\n            name (str): Name for the copied circuit. If None, then the name stays the same.\n\n        Returns:\n            QuantumCircuit: An empty copy of self.\n        '
        if not (name is None or isinstance(name, str)):
            raise TypeError(f"invalid name for a circuit: '{name}'. The name must be a string or 'None'.")
        cpy = copy.copy(self)
        cpy.qregs = self.qregs.copy()
        cpy.cregs = self.cregs.copy()
        cpy._qubits = self._qubits.copy()
        cpy._ancillas = self._ancillas.copy()
        cpy._clbits = self._clbits.copy()
        cpy._qubit_indices = self._qubit_indices.copy()
        cpy._clbit_indices = self._clbit_indices.copy()
        cpy._parameter_table = ParameterTable()
        cpy._data = []
        cpy._calibrations = copy.deepcopy(self._calibrations)
        cpy._metadata = copy.deepcopy(self._metadata)
        if name:
            cpy.name = name
        return cpy

    def clear(self) -> None:
        if False:
            while True:
                i = 10
        'Clear all instructions in self.\n\n        Clearing the circuits will keep the metadata and calibrations.\n        '
        self._data.clear()
        self._parameter_table.clear()

    def _create_creg(self, length: int, name: str) -> ClassicalRegister:
        if False:
            while True:
                i = 10
        'Creates a creg, checking if ClassicalRegister with same name exists'
        if name in [creg.name for creg in self.cregs]:
            save_prefix = ClassicalRegister.prefix
            ClassicalRegister.prefix = name
            new_creg = ClassicalRegister(length)
            ClassicalRegister.prefix = save_prefix
        else:
            new_creg = ClassicalRegister(length, name)
        return new_creg

    def _create_qreg(self, length: int, name: str) -> QuantumRegister:
        if False:
            return 10
        'Creates a qreg, checking if QuantumRegister with same name exists'
        if name in [qreg.name for qreg in self.qregs]:
            save_prefix = QuantumRegister.prefix
            QuantumRegister.prefix = name
            new_qreg = QuantumRegister(length)
            QuantumRegister.prefix = save_prefix
        else:
            new_qreg = QuantumRegister(length, name)
        return new_qreg

    def reset(self, qubit: QubitSpecifier) -> InstructionSet:
        if False:
            for i in range(10):
                print('nop')
        'Reset the quantum bit(s) to their default state.\n\n        Args:\n            qubit: qubit(s) to reset.\n\n        Returns:\n            qiskit.circuit.InstructionSet: handle to the added instruction.\n        '
        from .reset import Reset
        return self.append(Reset(), [qubit], [])

    def measure(self, qubit: QubitSpecifier, cbit: ClbitSpecifier) -> InstructionSet:
        if False:
            for i in range(10):
                print('nop')
        'Measure a quantum bit (``qubit``) in the Z basis into a classical bit (``cbit``).\n\n        When a quantum state is measured, a qubit is projected in the computational (Pauli Z) basis\n        to either :math:`\\lvert 0 \\rangle` or :math:`\\lvert 1 \\rangle`. The classical bit ``cbit``\n        indicates the result\n        of that projection as a ``0`` or a ``1`` respectively. This operation is non-reversible.\n\n        Args:\n            qubit: qubit(s) to measure.\n            cbit: classical bit(s) to place the measurement result(s) in.\n\n        Returns:\n            qiskit.circuit.InstructionSet: handle to the added instructions.\n\n        Raises:\n            CircuitError: if arguments have bad format.\n\n        Examples:\n            In this example, a qubit is measured and the result of that measurement is stored in the\n            classical bit (usually expressed in diagrams as a double line):\n\n            .. code-block::\n\n               from qiskit import QuantumCircuit\n               circuit = QuantumCircuit(1, 1)\n               circuit.h(0)\n               circuit.measure(0, 0)\n               circuit.draw()\n\n\n            .. parsed-literal::\n\n                      ┌───┐┌─┐\n                   q: ┤ H ├┤M├\n                      └───┘└╥┘\n                 c: 1/══════╩═\n                            0\n\n            It is possible to call ``measure`` with lists of ``qubits`` and ``cbits`` as a shortcut\n            for one-to-one measurement. These two forms produce identical results:\n\n            .. code-block::\n\n               circuit = QuantumCircuit(2, 2)\n               circuit.measure([0,1], [0,1])\n\n            .. code-block::\n\n               circuit = QuantumCircuit(2, 2)\n               circuit.measure(0, 0)\n               circuit.measure(1, 1)\n\n            Instead of lists, you can use :class:`~qiskit.circuit.QuantumRegister` and\n            :class:`~qiskit.circuit.ClassicalRegister` under the same logic.\n\n            .. code-block::\n\n                from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister\n                qreg = QuantumRegister(2, "qreg")\n                creg = ClassicalRegister(2, "creg")\n                circuit = QuantumCircuit(qreg, creg)\n                circuit.measure(qreg, creg)\n\n            This is equivalent to:\n\n            .. code-block::\n\n                circuit = QuantumCircuit(qreg, creg)\n                circuit.measure(qreg[0], creg[0])\n                circuit.measure(qreg[1], creg[1])\n\n        '
        from .measure import Measure
        return self.append(Measure(), [qubit], [cbit])

    def measure_active(self, inplace: bool=True) -> Optional['QuantumCircuit']:
        if False:
            i = 10
            return i + 15
        'Adds measurement to all non-idle qubits. Creates a new ClassicalRegister with\n        a size equal to the number of non-idle qubits being measured.\n\n        Returns a new circuit with measurements if `inplace=False`.\n\n        Args:\n            inplace (bool): All measurements inplace or return new circuit.\n\n        Returns:\n            QuantumCircuit: Returns circuit with measurements when `inplace = False`.\n        '
        from qiskit.converters.circuit_to_dag import circuit_to_dag
        if inplace:
            circ = self
        else:
            circ = self.copy()
        dag = circuit_to_dag(circ)
        qubits_to_measure = [qubit for qubit in circ.qubits if qubit not in dag.idle_wires()]
        new_creg = circ._create_creg(len(qubits_to_measure), 'measure')
        circ.add_register(new_creg)
        circ.barrier()
        circ.measure(qubits_to_measure, new_creg)
        if not inplace:
            return circ
        else:
            return None

    def measure_all(self, inplace: bool=True, add_bits: bool=True) -> Optional['QuantumCircuit']:
        if False:
            for i in range(10):
                print('nop')
        'Adds measurement to all qubits.\n\n        By default, adds new classical bits in a :obj:`.ClassicalRegister` to store these\n        measurements.  If ``add_bits=False``, the results of the measurements will instead be stored\n        in the already existing classical bits, with qubit ``n`` being measured into classical bit\n        ``n``.\n\n        Returns a new circuit with measurements if ``inplace=False``.\n\n        Args:\n            inplace (bool): All measurements inplace or return new circuit.\n            add_bits (bool): Whether to add new bits to store the results.\n\n        Returns:\n            QuantumCircuit: Returns circuit with measurements when ``inplace=False``.\n\n        Raises:\n            CircuitError: if ``add_bits=False`` but there are not enough classical bits.\n        '
        if inplace:
            circ = self
        else:
            circ = self.copy()
        if add_bits:
            new_creg = circ._create_creg(len(circ.qubits), 'meas')
            circ.add_register(new_creg)
            circ.barrier()
            circ.measure(circ.qubits, new_creg)
        else:
            if len(circ.clbits) < len(circ.qubits):
                raise CircuitError('The number of classical bits must be equal or greater than the number of qubits.')
            circ.barrier()
            circ.measure(circ.qubits, circ.clbits[0:len(circ.qubits)])
        if not inplace:
            return circ
        else:
            return None

    def remove_final_measurements(self, inplace: bool=True) -> Optional['QuantumCircuit']:
        if False:
            i = 10
            return i + 15
        "Removes final measurements and barriers on all qubits if they are present.\n        Deletes the classical registers that were used to store the values from these measurements\n        that become idle as a result of this operation, and deletes classical bits that are\n        referenced only by removed registers, or that aren't referenced at all but have\n        become idle as a result of this operation.\n\n        Measurements and barriers are considered final if they are\n        followed by no other operations (aside from other measurements or barriers.)\n\n        Args:\n            inplace (bool): All measurements removed inplace or return new circuit.\n\n        Returns:\n            QuantumCircuit: Returns the resulting circuit when ``inplace=False``, else None.\n        "
        from qiskit.transpiler.passes import RemoveFinalMeasurements
        from qiskit.converters import circuit_to_dag
        if inplace:
            circ = self
        else:
            circ = self.copy()
        dag = circuit_to_dag(circ)
        remove_final_meas = RemoveFinalMeasurements()
        new_dag = remove_final_meas.run(dag)
        kept_cregs = set(new_dag.cregs.values())
        kept_clbits = set(new_dag.clbits)
        cregs_to_add = [creg for creg in circ.cregs if creg in kept_cregs]
        clbits_to_add = [clbit for clbit in circ._clbits if clbit in kept_clbits]
        circ.cregs = []
        circ._clbits = []
        circ._clbit_indices = {}
        circ.add_bits(clbits_to_add)
        for creg in cregs_to_add:
            circ.add_register(creg)
        circ.data.clear()
        circ._parameter_table.clear()
        for node in new_dag.topological_op_nodes():
            inst = node.op.copy()
            circ.append(inst, node.qargs, node.cargs)
        if not inplace:
            return circ
        else:
            return None

    @staticmethod
    def from_qasm_file(path: str) -> 'QuantumCircuit':
        if False:
            for i in range(10):
                print('nop')
        'Read an OpenQASM 2.0 program from a file and convert to an instance of\n        :class:`.QuantumCircuit`.\n\n        Args:\n          path (str): Path to the file for an OpenQASM 2 program\n\n        Return:\n          QuantumCircuit: The QuantumCircuit object for the input OpenQASM 2.\n\n        See also:\n            :func:`.qasm2.load`: the complete interface to the OpenQASM 2 importer.\n        '
        from qiskit import qasm2
        return qasm2.load(path, include_path=qasm2.LEGACY_INCLUDE_PATH, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS, custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL, strict=False)

    @staticmethod
    def from_qasm_str(qasm_str: str) -> 'QuantumCircuit':
        if False:
            print('Hello World!')
        'Convert a string containing an OpenQASM 2.0 program to a :class:`.QuantumCircuit`.\n\n        Args:\n          qasm_str (str): A string containing an OpenQASM 2.0 program.\n        Return:\n          QuantumCircuit: The QuantumCircuit object for the input OpenQASM 2\n\n        See also:\n            :func:`.qasm2.loads`: the complete interface to the OpenQASM 2 importer.\n        '
        from qiskit import qasm2
        return qasm2.loads(qasm_str, include_path=qasm2.LEGACY_INCLUDE_PATH, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS, custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL, strict=False)

    @property
    def global_phase(self) -> ParameterValueType:
        if False:
            i = 10
            return i + 15
        'Return the global phase of the current circuit scope in radians.'
        if self._control_flow_scopes:
            return self._control_flow_scopes[-1].global_phase
        return self._global_phase

    @global_phase.setter
    def global_phase(self, angle: ParameterValueType):
        if False:
            print('Hello World!')
        'Set the phase of the current circuit scope.\n\n        Args:\n            angle (float, ParameterExpression): radians\n        '
        if not (isinstance(angle, ParameterExpression) and angle.parameters):
            angle = float(angle) % (2 * np.pi)
        if self._control_flow_scopes:
            self._control_flow_scopes[-1].global_phase = angle
        else:
            self._global_phase = angle

    @property
    def parameters(self) -> ParameterView:
        if False:
            i = 10
            return i + 15
        'The parameters defined in the circuit.\n\n        This attribute returns the :class:`.Parameter` objects in the circuit sorted\n        alphabetically. Note that parameters instantiated with a :class:`.ParameterVector`\n        are still sorted numerically.\n\n        Examples:\n\n            The snippet below shows that insertion order of parameters does not matter.\n\n            .. code-block:: python\n\n                >>> from qiskit.circuit import QuantumCircuit, Parameter\n                >>> a, b, elephant = Parameter("a"), Parameter("b"), Parameter("elephant")\n                >>> circuit = QuantumCircuit(1)\n                >>> circuit.rx(b, 0)\n                >>> circuit.rz(elephant, 0)\n                >>> circuit.ry(a, 0)\n                >>> circuit.parameters  # sorted alphabetically!\n                ParameterView([Parameter(a), Parameter(b), Parameter(elephant)])\n\n            Bear in mind that alphabetical sorting might be unintuitive when it comes to numbers.\n            The literal "10" comes before "2" in strict alphabetical sorting.\n\n            .. code-block:: python\n\n                >>> from qiskit.circuit import QuantumCircuit, Parameter\n                >>> angles = [Parameter("angle_1"), Parameter("angle_2"), Parameter("angle_10")]\n                >>> circuit = QuantumCircuit(1)\n                >>> circuit.u(*angles, 0)\n                >>> circuit.draw()\n                   ┌─────────────────────────────┐\n                q: ┤ U(angle_1,angle_2,angle_10) ├\n                   └─────────────────────────────┘\n                >>> circuit.parameters\n                ParameterView([Parameter(angle_1), Parameter(angle_10), Parameter(angle_2)])\n\n            To respect numerical sorting, a :class:`.ParameterVector` can be used.\n\n            .. code-block:: python\n\n            >>> from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector\n            >>> x = ParameterVector("x", 12)\n            >>> circuit = QuantumCircuit(1)\n            >>> for x_i in x:\n            ...     circuit.rx(x_i, 0)\n            >>> circuit.parameters\n            ParameterView([\n                ParameterVectorElement(x[0]), ParameterVectorElement(x[1]),\n                ParameterVectorElement(x[2]), ParameterVectorElement(x[3]),\n                ..., ParameterVectorElement(x[11])\n            ])\n\n\n        Returns:\n            The sorted :class:`.Parameter` objects in the circuit.\n        '
        if self._parameters is None:
            self._parameters = sort_parameters(self._unsorted_parameters())
        return ParameterView(self._parameters)

    @property
    def num_parameters(self) -> int:
        if False:
            print('Hello World!')
        'The number of parameter objects in the circuit.'
        if self._parameters is not None:
            return len(self._parameters)
        return len(self._unsorted_parameters())

    def _unsorted_parameters(self) -> set[Parameter]:
        if False:
            i = 10
            return i + 15
        'Efficiently get all parameters in the circuit, without any sorting overhead.\n\n        .. warning::\n\n            The returned object may directly view onto the ``ParameterTable`` internals, and so\n            should not be mutated.  This is an internal performance detail.  Code outside of this\n            package should not use this method.\n        '
        parameters = self._parameter_table.get_keys()
        if isinstance(self.global_phase, ParameterExpression):
            parameters = parameters | self.global_phase.parameters
        return parameters

    @overload
    def assign_parameters(self, parameters: Union[Mapping[Parameter, ParameterValueType], Sequence[ParameterValueType]], inplace: Literal[False]=..., *, flat_input: bool=..., strict: bool=...) -> 'QuantumCircuit':
        if False:
            return 10
        ...

    @overload
    def assign_parameters(self, parameters: Union[Mapping[Parameter, ParameterValueType], Sequence[ParameterValueType]], inplace: Literal[True]=..., *, flat_input: bool=..., strict: bool=...) -> None:
        if False:
            for i in range(10):
                print('nop')
        ...

    def assign_parameters(self, parameters: Union[Mapping[Parameter, ParameterValueType], Sequence[ParameterValueType]], inplace: bool=False, *, flat_input: bool=False, strict: bool=True) -> Optional['QuantumCircuit']:
        if False:
            return 10
        "Assign parameters to new parameters or values.\n\n        If ``parameters`` is passed as a dictionary, the keys must be :class:`.Parameter`\n        instances in the current circuit. The values of the dictionary can either be numeric values\n        or new parameter objects.\n\n        If ``parameters`` is passed as a list or array, the elements are assigned to the\n        current parameters in the order of :attr:`parameters` which is sorted\n        alphabetically (while respecting the ordering in :class:`.ParameterVector` objects).\n\n        The values can be assigned to the current circuit object or to a copy of it.\n\n        Args:\n            parameters: Either a dictionary or iterable specifying the new parameter values.\n            inplace: If False, a copy of the circuit with the bound parameters is returned.\n                If True the circuit instance itself is modified.\n            flat_input: If ``True`` and ``parameters`` is a mapping type, it is assumed to be\n                exactly a mapping of ``{parameter: value}``.  By default (``False``), the mapping\n                may also contain :class:`.ParameterVector` keys that point to a corresponding\n                sequence of values, and these will be unrolled during the mapping.\n            strict: If ``False``, any parameters given in the mapping that are not used in the\n                circuit will be ignored.  If ``True`` (the default), an error will be raised\n                indicating a logic error.\n\n        Raises:\n            CircuitError: If parameters is a dict and contains parameters not present in the\n                circuit.\n            ValueError: If parameters is a list/array and the length mismatches the number of free\n                parameters in the circuit.\n\n        Returns:\n            A copy of the circuit with bound parameters if ``inplace`` is False, otherwise None.\n\n        Examples:\n\n            Create a parameterized circuit and assign the parameters in-place.\n\n            .. plot::\n               :include-source:\n\n               from qiskit.circuit import QuantumCircuit, Parameter\n\n               circuit = QuantumCircuit(2)\n               params = [Parameter('A'), Parameter('B'), Parameter('C')]\n               circuit.ry(params[0], 0)\n               circuit.crx(params[1], 0, 1)\n               circuit.draw('mpl')\n               circuit.assign_parameters({params[0]: params[2]}, inplace=True)\n               circuit.draw('mpl')\n\n            Bind the values out-of-place by list and get a copy of the original circuit.\n\n            .. plot::\n               :include-source:\n\n               from qiskit.circuit import QuantumCircuit, ParameterVector\n\n               circuit = QuantumCircuit(2)\n               params = ParameterVector('P', 2)\n               circuit.ry(params[0], 0)\n               circuit.crx(params[1], 0, 1)\n\n               bound_circuit = circuit.assign_parameters([1, 2])\n               bound_circuit.draw('mpl')\n\n               circuit.draw('mpl')\n\n        "
        if inplace:
            target = self
        else:
            target = self.copy()
            target._increment_instances()
            target._name_update()
        if isinstance(parameters, dict):
            raw_mapping = parameters if flat_input else self._unroll_param_dict(parameters)
            our_parameters = self._unsorted_parameters()
            if strict and (extras := (raw_mapping.keys() - our_parameters)):
                raise CircuitError(f"Cannot bind parameters ({', '.join((str(x) for x in extras))}) not present in the circuit.")
            parameter_binds = _ParameterBindsDict(raw_mapping, our_parameters)
        else:
            our_parameters = self.parameters
            if len(parameters) != len(our_parameters):
                raise ValueError('Mismatching number of values and parameters. For partial binding please pass a dictionary of {parameter: value} pairs.')
            parameter_binds = _ParameterBindsSequence(our_parameters, parameters)
        target._parameters = None
        all_references = [(parameter, value, target._parameter_table.pop(parameter, ())) for (parameter, value) in parameter_binds.items()]
        seen_operations = {}
        for (to_bind, bound_value, references) in all_references:
            update_parameters = tuple(bound_value.parameters) if isinstance(bound_value, ParameterExpression) else ()
            for (operation, index) in references:
                seen_operations[id(operation)] = operation
                assignee = operation.params[index]
                if isinstance(assignee, ParameterExpression):
                    new_parameter = assignee.assign(to_bind, bound_value)
                    for parameter in update_parameters:
                        if parameter not in target._parameter_table:
                            target._parameter_table[parameter] = ParameterReferences(())
                        target._parameter_table[parameter].add((operation, index))
                    if not new_parameter.parameters:
                        if new_parameter.is_real():
                            new_parameter = int(new_parameter) if new_parameter._symbol_expr.is_integer else float(new_parameter)
                        else:
                            new_parameter = complex(new_parameter)
                        new_parameter = operation.validate_parameter(new_parameter)
                elif isinstance(assignee, QuantumCircuit):
                    new_parameter = assignee.assign_parameters({to_bind: bound_value}, inplace=False, flat_input=True)
                else:
                    raise RuntimeError(f'Saw an unknown type during symbolic binding: {assignee}. This may indicate an internal logic error in symbol tracking.')
                operation.params[index] = new_parameter
        for operation in seen_operations.values():
            if (definition := getattr(operation, '_definition', None)) is not None and definition.num_parameters:
                definition.assign_parameters(parameter_binds.mapping, inplace=True, flat_input=True, strict=False)
        if isinstance(target.global_phase, ParameterExpression):
            new_phase = target.global_phase
            for parameter in new_phase.parameters & parameter_binds.mapping.keys():
                new_phase = new_phase.assign(parameter, parameter_binds.mapping[parameter])
            target.global_phase = new_phase

        def map_calibration(qubits, parameters, schedule):
            if False:
                print('Hello World!')
            modified = False
            new_parameters = list(parameters)
            for (i, parameter) in enumerate(new_parameters):
                if not isinstance(parameter, ParameterExpression):
                    continue
                if not (contained := (parameter.parameters & parameter_binds.mapping.keys())):
                    continue
                for to_bind in contained:
                    parameter = parameter.assign(to_bind, parameter_binds.mapping[to_bind])
                if not parameter.parameters:
                    parameter = int(parameter) if parameter._symbol_expr.is_integer else float(parameter)
                new_parameters[i] = parameter
                modified = True
            if modified:
                schedule.assign_parameters(parameter_binds.mapping)
            return ((qubits, tuple(new_parameters)), schedule)
        target._calibrations = defaultdict(dict, ((gate, dict((map_calibration(qubits, parameters, schedule) for ((qubits, parameters), schedule) in calibrations.items()))) for (gate, calibrations) in target._calibrations.items()))
        return None if inplace else target

    @staticmethod
    def _unroll_param_dict(parameter_binds: Mapping[Parameter, ParameterValueType]) -> Mapping[Parameter, ParameterValueType]:
        if False:
            i = 10
            return i + 15
        out = {}
        for (parameter, value) in parameter_binds.items():
            if isinstance(parameter, ParameterVector):
                if len(parameter) != len(value):
                    raise CircuitError(f"Parameter vector '{parameter.name}' has length {len(parameter)}, but was assigned to {len(value)} values.")
                out.update(zip(parameter, value))
            else:
                out[parameter] = value
        return out

    @deprecate_func(additional_msg='Use assign_parameters() instead', since='0.45.0')
    def bind_parameters(self, values: Union[Mapping[Parameter, float], Sequence[float]]) -> 'QuantumCircuit':
        if False:
            while True:
                i = 10
        'Assign numeric parameters to values yielding a new circuit.\n\n        If the values are given as list or array they are bound to the circuit in the order\n        of :attr:`parameters` (see the docstring for more details).\n\n        To assign new Parameter objects or bind the values in-place, without yielding a new\n        circuit, use the :meth:`assign_parameters` method.\n\n        Args:\n            values: ``{parameter: value, ...}`` or ``[value1, value2, ...]``\n\n        Raises:\n            CircuitError: If values is a dict and contains parameters not present in the circuit.\n            TypeError: If values contains a ParameterExpression.\n\n        Returns:\n            Copy of self with assignment substitution.\n        '
        if isinstance(values, dict):
            if any((isinstance(value, ParameterExpression) for value in values.values())):
                raise TypeError('Found ParameterExpression in values; use assign_parameters() instead.')
            return self.assign_parameters(values)
        else:
            if any((isinstance(value, ParameterExpression) for value in values)):
                raise TypeError('Found ParameterExpression in values; use assign_parameters() instead.')
            return self.assign_parameters(values)

    def barrier(self, *qargs: QubitSpecifier, label=None) -> InstructionSet:
        if False:
            return 10
        'Apply :class:`~.library.Barrier`. If ``qargs`` is empty, applies to all qubits\n        in the circuit.\n\n        Args:\n            qargs (QubitSpecifier): Specification for one or more qubit arguments.\n            label (str): The string label of the barrier.\n\n        Returns:\n            qiskit.circuit.InstructionSet: handle to the added instructions.\n        '
        from .barrier import Barrier
        qubits: list[QubitSpecifier] = []
        if not qargs:
            qubits.extend(self.qubits)
        for qarg in qargs:
            if isinstance(qarg, QuantumRegister):
                qubits.extend([qarg[j] for j in range(qarg.size)])
            elif isinstance(qarg, list):
                qubits.extend(qarg)
            elif isinstance(qarg, range):
                qubits.extend(list(qarg))
            elif isinstance(qarg, slice):
                qubits.extend(self.qubits[qarg])
            else:
                qubits.append(qarg)
        return self.append(Barrier(len(qubits), label=label), qubits, [])

    def delay(self, duration: ParameterValueType, qarg: QubitSpecifier | None=None, unit: str='dt') -> InstructionSet:
        if False:
            i = 10
            return i + 15
        "Apply :class:`~.circuit.Delay`. If qarg is ``None``, applies to all qubits.\n        When applying to multiple qubits, delays with the same duration will be created.\n\n        Args:\n            duration (int or float or ParameterExpression): duration of the delay.\n            qarg (Object): qubit argument to apply this delay.\n            unit (str): unit of the duration. Supported units: ``'s'``, ``'ms'``, ``'us'``,\n                ``'ns'``, ``'ps'``, and ``'dt'``. Default is ``'dt'``, i.e. integer time unit\n                depending on the target backend.\n\n        Returns:\n            qiskit.circuit.InstructionSet: handle to the added instructions.\n\n        Raises:\n            CircuitError: if arguments have bad format.\n        "
        qubits: list[QubitSpecifier] = []
        if qarg is None:
            for q in self.qubits:
                qubits.append(q)
        elif isinstance(qarg, QuantumRegister):
            qubits.extend([qarg[j] for j in range(qarg.size)])
        elif isinstance(qarg, list):
            qubits.extend(qarg)
        elif isinstance(qarg, (range, tuple)):
            qubits.extend(list(qarg))
        elif isinstance(qarg, slice):
            qubits.extend(self.qubits[qarg])
        else:
            qubits.append(qarg)
        instructions = InstructionSet(resource_requester=self._resolve_classical_resource)
        for q in qubits:
            inst: tuple[Instruction, Sequence[QubitSpecifier] | None, Sequence[ClbitSpecifier] | None] = (Delay(duration, unit), [q], [])
            self.append(*inst)
            instructions.add(*inst)
        return instructions

    def h(self, qubit: QubitSpecifier) -> InstructionSet:
        if False:
            return 10
        'Apply :class:`~qiskit.circuit.library.HGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            qubit: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.h import HGate
        return self.append(HGate(), [qubit], [])

    def ch(self, control_qubit: QubitSpecifier, target_qubit: QubitSpecifier, label: str | None=None, ctrl_state: str | int | None=None) -> InstructionSet:
        if False:
            for i in range(10):
                print('nop')
        "Apply :class:`~qiskit.circuit.library.CHGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            control_qubit: The qubit(s) used as the control.\n            target_qubit: The qubit(s) targeted by the gate.\n            label: The string label of the gate in the circuit.\n            ctrl_state:\n                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling\n                on the '1' state.\n\n        Returns:\n            A handle to the instructions created.\n        "
        from .library.standard_gates.h import CHGate
        return self.append(CHGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], [])

    @deprecate_func(since='0.45.0', additional_msg='Use QuantumCircuit.id as direct replacement.')
    def i(self, qubit: QubitSpecifier) -> InstructionSet:
        if False:
            i = 10
            return i + 15
        'Apply :class:`~qiskit.circuit.library.IGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            qubit: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        return self.id(qubit)

    def id(self, qubit: QubitSpecifier) -> InstructionSet:
        if False:
            while True:
                i = 10
        'Apply :class:`~qiskit.circuit.library.IGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            qubit: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.i import IGate
        return self.append(IGate(), [qubit], [])

    def ms(self, theta: ParameterValueType, qubits: Sequence[QubitSpecifier]) -> InstructionSet:
        if False:
            while True:
                i = 10
        'Apply :class:`~qiskit.circuit.library.MSGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            theta: The angle of the rotation.\n            qubits: The qubits to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.generalized_gates.gms import MSGate
        return self.append(MSGate(len(qubits), theta), qubits)

    def p(self, theta: ParameterValueType, qubit: QubitSpecifier) -> InstructionSet:
        if False:
            print('Hello World!')
        'Apply :class:`~qiskit.circuit.library.PhaseGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            theta: THe angle of the rotation.\n            qubit: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.p import PhaseGate
        return self.append(PhaseGate(theta), [qubit], [])

    def cp(self, theta: ParameterValueType, control_qubit: QubitSpecifier, target_qubit: QubitSpecifier, label: str | None=None, ctrl_state: str | int | None=None) -> InstructionSet:
        if False:
            while True:
                i = 10
        "Apply :class:`~qiskit.circuit.library.CPhaseGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            theta: The angle of the rotation.\n            control_qubit: The qubit(s) used as the control.\n            target_qubit: The qubit(s) targeted by the gate.\n            label: The string label of the gate in the circuit.\n            ctrl_state:\n                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling\n                on the '1' state.\n\n        Returns:\n            A handle to the instructions created.\n        "
        from .library.standard_gates.p import CPhaseGate
        return self.append(CPhaseGate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], [])

    def mcp(self, lam: ParameterValueType, control_qubits: Sequence[QubitSpecifier], target_qubit: QubitSpecifier) -> InstructionSet:
        if False:
            while True:
                i = 10
        'Apply :class:`~qiskit.circuit.library.MCPhaseGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            lam: The angle of the rotation.\n            control_qubits: The qubits used as the controls.\n            target_qubit: The qubit(s) targeted by the gate.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.p import MCPhaseGate
        num_ctrl_qubits = len(control_qubits)
        return self.append(MCPhaseGate(lam, num_ctrl_qubits), control_qubits[:] + [target_qubit], [])

    def r(self, theta: ParameterValueType, phi: ParameterValueType, qubit: QubitSpecifier) -> InstructionSet:
        if False:
            while True:
                i = 10
        'Apply :class:`~qiskit.circuit.library.RGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            theta: The angle of the rotation.\n            phi: The angle of the axis of rotation in the x-y plane.\n            qubit: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.r import RGate
        return self.append(RGate(theta, phi), [qubit], [])

    def rv(self, vx: ParameterValueType, vy: ParameterValueType, vz: ParameterValueType, qubit: QubitSpecifier) -> InstructionSet:
        if False:
            return 10
        'Apply :class:`~qiskit.circuit.library.RVGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Rotation around an arbitrary rotation axis :math:`v`, where :math:`|v|` is the angle of\n        rotation in radians.\n\n        Args:\n            vx: x-component of the rotation axis.\n            vy: y-component of the rotation axis.\n            vz: z-component of the rotation axis.\n            qubit: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.generalized_gates.rv import RVGate
        return self.append(RVGate(vx, vy, vz), [qubit], [])

    def rccx(self, control_qubit1: QubitSpecifier, control_qubit2: QubitSpecifier, target_qubit: QubitSpecifier) -> InstructionSet:
        if False:
            while True:
                i = 10
        'Apply :class:`~qiskit.circuit.library.RCCXGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            control_qubit1: The qubit(s) used as the first control.\n            control_qubit2: The qubit(s) used as the second control.\n            target_qubit: The qubit(s) targeted by the gate.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.x import RCCXGate
        return self.append(RCCXGate(), [control_qubit1, control_qubit2, target_qubit], [])

    def rcccx(self, control_qubit1: QubitSpecifier, control_qubit2: QubitSpecifier, control_qubit3: QubitSpecifier, target_qubit: QubitSpecifier) -> InstructionSet:
        if False:
            print('Hello World!')
        'Apply :class:`~qiskit.circuit.library.RC3XGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            control_qubit1: The qubit(s) used as the first control.\n            control_qubit2: The qubit(s) used as the second control.\n            control_qubit3: The qubit(s) used as the third control.\n            target_qubit: The qubit(s) targeted by the gate.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.x import RC3XGate
        return self.append(RC3XGate(), [control_qubit1, control_qubit2, control_qubit3, target_qubit], [])

    def rx(self, theta: ParameterValueType, qubit: QubitSpecifier, label: str | None=None) -> InstructionSet:
        if False:
            i = 10
            return i + 15
        'Apply :class:`~qiskit.circuit.library.RXGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            theta: The rotation angle of the gate.\n            qubit: The qubit(s) to apply the gate to.\n            label: The string label of the gate in the circuit.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.rx import RXGate
        return self.append(RXGate(theta, label=label), [qubit], [])

    def crx(self, theta: ParameterValueType, control_qubit: QubitSpecifier, target_qubit: QubitSpecifier, label: str | None=None, ctrl_state: str | int | None=None) -> InstructionSet:
        if False:
            i = 10
            return i + 15
        "Apply :class:`~qiskit.circuit.library.CRXGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            theta: The angle of the rotation.\n            control_qubit: The qubit(s) used as the control.\n            target_qubit: The qubit(s) targeted by the gate.\n            label: The string label of the gate in the circuit.\n            ctrl_state:\n                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling\n                on the '1' state.\n\n        Returns:\n            A handle to the instructions created.\n        "
        from .library.standard_gates.rx import CRXGate
        return self.append(CRXGate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], [])

    def rxx(self, theta: ParameterValueType, qubit1: QubitSpecifier, qubit2: QubitSpecifier) -> InstructionSet:
        if False:
            return 10
        'Apply :class:`~qiskit.circuit.library.RXXGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            theta: The angle of the rotation.\n            qubit1: The qubit(s) to apply the gate to.\n            qubit2: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.rxx import RXXGate
        return self.append(RXXGate(theta), [qubit1, qubit2], [])

    def ry(self, theta: ParameterValueType, qubit: QubitSpecifier, label: str | None=None) -> InstructionSet:
        if False:
            print('Hello World!')
        'Apply :class:`~qiskit.circuit.library.RYGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            theta: The rotation angle of the gate.\n            qubit: The qubit(s) to apply the gate to.\n            label: The string label of the gate in the circuit.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.ry import RYGate
        return self.append(RYGate(theta, label=label), [qubit], [])

    def cry(self, theta: ParameterValueType, control_qubit: QubitSpecifier, target_qubit: QubitSpecifier, label: str | None=None, ctrl_state: str | int | None=None) -> InstructionSet:
        if False:
            print('Hello World!')
        "Apply :class:`~qiskit.circuit.library.CRYGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            theta: The angle of the rotation.\n            control_qubit: The qubit(s) used as the control.\n            target_qubit: The qubit(s) targeted by the gate.\n            label: The string label of the gate in the circuit.\n            ctrl_state:\n                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling\n                on the '1' state.\n\n        Returns:\n            A handle to the instructions created.\n        "
        from .library.standard_gates.ry import CRYGate
        return self.append(CRYGate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], [])

    def ryy(self, theta: ParameterValueType, qubit1: QubitSpecifier, qubit2: QubitSpecifier) -> InstructionSet:
        if False:
            for i in range(10):
                print('nop')
        'Apply :class:`~qiskit.circuit.library.RYYGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            theta: The rotation angle of the gate.\n            qubit1: The qubit(s) to apply the gate to.\n            qubit2: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.ryy import RYYGate
        return self.append(RYYGate(theta), [qubit1, qubit2], [])

    def rz(self, phi: ParameterValueType, qubit: QubitSpecifier) -> InstructionSet:
        if False:
            i = 10
            return i + 15
        'Apply :class:`~qiskit.circuit.library.RZGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            phi: The rotation angle of the gate.\n            qubit: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.rz import RZGate
        return self.append(RZGate(phi), [qubit], [])

    def crz(self, theta: ParameterValueType, control_qubit: QubitSpecifier, target_qubit: QubitSpecifier, label: str | None=None, ctrl_state: str | int | None=None) -> InstructionSet:
        if False:
            print('Hello World!')
        "Apply :class:`~qiskit.circuit.library.CRZGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            theta: The angle of the rotation.\n            control_qubit: The qubit(s) used as the control.\n            target_qubit: The qubit(s) targeted by the gate.\n            label: The string label of the gate in the circuit.\n            ctrl_state:\n                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling\n                on the '1' state.\n\n        Returns:\n            A handle to the instructions created.\n        "
        from .library.standard_gates.rz import CRZGate
        return self.append(CRZGate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], [])

    def rzx(self, theta: ParameterValueType, qubit1: QubitSpecifier, qubit2: QubitSpecifier) -> InstructionSet:
        if False:
            print('Hello World!')
        'Apply :class:`~qiskit.circuit.library.RZXGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            theta: The rotation angle of the gate.\n            qubit1: The qubit(s) to apply the gate to.\n            qubit2: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.rzx import RZXGate
        return self.append(RZXGate(theta), [qubit1, qubit2], [])

    def rzz(self, theta: ParameterValueType, qubit1: QubitSpecifier, qubit2: QubitSpecifier) -> InstructionSet:
        if False:
            i = 10
            return i + 15
        'Apply :class:`~qiskit.circuit.library.RZZGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            theta: The rotation angle of the gate.\n            qubit1: The qubit(s) to apply the gate to.\n            qubit2: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.rzz import RZZGate
        return self.append(RZZGate(theta), [qubit1, qubit2], [])

    def ecr(self, qubit1: QubitSpecifier, qubit2: QubitSpecifier) -> InstructionSet:
        if False:
            print('Hello World!')
        'Apply :class:`~qiskit.circuit.library.ECRGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            qubit1, qubit2: The qubits to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.ecr import ECRGate
        return self.append(ECRGate(), [qubit1, qubit2], [])

    def s(self, qubit: QubitSpecifier) -> InstructionSet:
        if False:
            i = 10
            return i + 15
        'Apply :class:`~qiskit.circuit.library.SGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            qubit: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.s import SGate
        return self.append(SGate(), [qubit], [])

    def sdg(self, qubit: QubitSpecifier) -> InstructionSet:
        if False:
            for i in range(10):
                print('nop')
        'Apply :class:`~qiskit.circuit.library.SdgGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            qubit: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.s import SdgGate
        return self.append(SdgGate(), [qubit], [])

    def cs(self, control_qubit: QubitSpecifier, target_qubit: QubitSpecifier, label: str | None=None, ctrl_state: str | int | None=None) -> InstructionSet:
        if False:
            return 10
        "Apply :class:`~qiskit.circuit.library.CSGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            control_qubit: The qubit(s) used as the control.\n            target_qubit: The qubit(s) targeted by the gate.\n            label: The string label of the gate in the circuit.\n            ctrl_state:\n                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling\n                on the '1' state.\n\n        Returns:\n            A handle to the instructions created.\n        "
        from .library.standard_gates.s import CSGate
        return self.append(CSGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], [])

    def csdg(self, control_qubit: QubitSpecifier, target_qubit: QubitSpecifier, label: str | None=None, ctrl_state: str | int | None=None) -> InstructionSet:
        if False:
            while True:
                i = 10
        "Apply :class:`~qiskit.circuit.library.CSdgGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            control_qubit: The qubit(s) used as the control.\n            target_qubit: The qubit(s) targeted by the gate.\n            label: The string label of the gate in the circuit.\n            ctrl_state:\n                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling\n                on the '1' state.\n\n        Returns:\n            A handle to the instructions created.\n        "
        from .library.standard_gates.s import CSdgGate
        return self.append(CSdgGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], [])

    def swap(self, qubit1: QubitSpecifier, qubit2: QubitSpecifier) -> InstructionSet:
        if False:
            for i in range(10):
                print('nop')
        'Apply :class:`~qiskit.circuit.library.SwapGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            qubit1, qubit2: The qubits to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.swap import SwapGate
        return self.append(SwapGate(), [qubit1, qubit2], [])

    def iswap(self, qubit1: QubitSpecifier, qubit2: QubitSpecifier) -> InstructionSet:
        if False:
            return 10
        'Apply :class:`~qiskit.circuit.library.iSwapGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            qubit1, qubit2: The qubits to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.iswap import iSwapGate
        return self.append(iSwapGate(), [qubit1, qubit2], [])

    def cswap(self, control_qubit: QubitSpecifier, target_qubit1: QubitSpecifier, target_qubit2: QubitSpecifier, label: str | None=None, ctrl_state: str | int | None=None) -> InstructionSet:
        if False:
            i = 10
            return i + 15
        "Apply :class:`~qiskit.circuit.library.CSwapGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            control_qubit: The qubit(s) used as the control.\n            target_qubit1: The qubit(s) targeted by the gate.\n            target_qubit2: The qubit(s) targeted by the gate.\n            label: The string label of the gate in the circuit.\n            ctrl_state:\n                The control state in decimal, or as a bitstring (e.g. ``'1'``).  Defaults to controlling\n                on the ``'1'`` state.\n\n        Returns:\n            A handle to the instructions created.\n        "
        from .library.standard_gates.swap import CSwapGate
        return self.append(CSwapGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit1, target_qubit2], [])

    @deprecate_func(since='0.45.0', additional_msg='Use QuantumCircuit.cswap as direct replacement.')
    def fredkin(self, control_qubit: QubitSpecifier, target_qubit1: QubitSpecifier, target_qubit2: QubitSpecifier) -> InstructionSet:
        if False:
            i = 10
            return i + 15
        'Apply :class:`~qiskit.circuit.library.CSwapGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            control_qubit: The qubit(s) used as the control.\n            target_qubit1: The qubit(s) targeted by the gate.\n            target_qubit2: The qubit(s) targeted by the gate.\n\n        Returns:\n            A handle to the instructions created.\n\n        See Also:\n            QuantumCircuit.cswap: the same function with a different name.\n        '
        return self.cswap(control_qubit, target_qubit1, target_qubit2)

    def sx(self, qubit: QubitSpecifier) -> InstructionSet:
        if False:
            while True:
                i = 10
        'Apply :class:`~qiskit.circuit.library.SXGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            qubit: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.sx import SXGate
        return self.append(SXGate(), [qubit], [])

    def sxdg(self, qubit: QubitSpecifier) -> InstructionSet:
        if False:
            return 10
        'Apply :class:`~qiskit.circuit.library.SXdgGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            qubit: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.sx import SXdgGate
        return self.append(SXdgGate(), [qubit], [])

    def csx(self, control_qubit: QubitSpecifier, target_qubit: QubitSpecifier, label: str | None=None, ctrl_state: str | int | None=None) -> InstructionSet:
        if False:
            print('Hello World!')
        "Apply :class:`~qiskit.circuit.library.CSXGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            control_qubit: The qubit(s) used as the control.\n            target_qubit: The qubit(s) targeted by the gate.\n            label: The string label of the gate in the circuit.\n            ctrl_state:\n                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling\n                on the '1' state.\n\n        Returns:\n            A handle to the instructions created.\n        "
        from .library.standard_gates.sx import CSXGate
        return self.append(CSXGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], [])

    def t(self, qubit: QubitSpecifier) -> InstructionSet:
        if False:
            i = 10
            return i + 15
        'Apply :class:`~qiskit.circuit.library.TGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            qubit: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.t import TGate
        return self.append(TGate(), [qubit], [])

    def tdg(self, qubit: QubitSpecifier) -> InstructionSet:
        if False:
            for i in range(10):
                print('nop')
        'Apply :class:`~qiskit.circuit.library.TdgGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            qubit: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.t import TdgGate
        return self.append(TdgGate(), [qubit], [])

    def u(self, theta: ParameterValueType, phi: ParameterValueType, lam: ParameterValueType, qubit: QubitSpecifier) -> InstructionSet:
        if False:
            i = 10
            return i + 15
        'Apply :class:`~qiskit.circuit.library.UGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            theta: The :math:`\\theta` rotation angle of the gate.\n            phi: The :math:`\\phi` rotation angle of the gate.\n            lam: The :math:`\\lambda` rotation angle of the gate.\n            qubit: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.u import UGate
        return self.append(UGate(theta, phi, lam), [qubit], [])

    def cu(self, theta: ParameterValueType, phi: ParameterValueType, lam: ParameterValueType, gamma: ParameterValueType, control_qubit: QubitSpecifier, target_qubit: QubitSpecifier, label: str | None=None, ctrl_state: str | int | None=None) -> InstructionSet:
        if False:
            return 10
        "Apply :class:`~qiskit.circuit.library.CUGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            theta: The :math:`\\theta` rotation angle of the gate.\n            phi: The :math:`\\phi` rotation angle of the gate.\n            lam: The :math:`\\lambda` rotation angle of the gate.\n            gamma: The global phase applied of the U gate, if applied.\n            control_qubit: The qubit(s) used as the control.\n            target_qubit: The qubit(s) targeted by the gate.\n            label: The string label of the gate in the circuit.\n            ctrl_state:\n                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling\n                on the '1' state.\n\n        Returns:\n            A handle to the instructions created.\n        "
        from .library.standard_gates.u import CUGate
        return self.append(CUGate(theta, phi, lam, gamma, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], [])

    def x(self, qubit: QubitSpecifier, label: str | None=None) -> InstructionSet:
        if False:
            i = 10
            return i + 15
        'Apply :class:`~qiskit.circuit.library.XGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            qubit: The qubit(s) to apply the gate to.\n            label: The string label of the gate in the circuit.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.x import XGate
        return self.append(XGate(label=label), [qubit], [])

    def cx(self, control_qubit: QubitSpecifier, target_qubit: QubitSpecifier, label: str | None=None, ctrl_state: str | int | None=None) -> InstructionSet:
        if False:
            return 10
        "Apply :class:`~qiskit.circuit.library.CXGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            control_qubit: The qubit(s) used as the control.\n            target_qubit: The qubit(s) targeted by the gate.\n            label: The string label of the gate in the circuit.\n            ctrl_state:\n                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling\n                on the '1' state.\n\n        Returns:\n            A handle to the instructions created.\n        "
        from .library.standard_gates.x import CXGate
        return self.append(CXGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], [])

    @deprecate_func(since='0.45.0', additional_msg='Use QuantumCircuit.cx as direct replacement.')
    def cnot(self, control_qubit: QubitSpecifier, target_qubit: QubitSpecifier, label: str | None=None, ctrl_state: str | int | None=None) -> InstructionSet:
        if False:
            i = 10
            return i + 15
        "Apply :class:`~qiskit.circuit.library.CXGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            control_qubit: The qubit(s) used as the control.\n            target_qubit: The qubit(s) targeted by the gate.\n            label: The string label of the gate in the circuit.\n            ctrl_state:\n                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling\n                on the '1' state.\n\n        Returns:\n            A handle to the instructions created.\n\n        See Also:\n            QuantumCircuit.cx: the same function with a different name.\n        "
        return self.cx(control_qubit, target_qubit, label, ctrl_state)

    def dcx(self, qubit1: QubitSpecifier, qubit2: QubitSpecifier) -> InstructionSet:
        if False:
            for i in range(10):
                print('nop')
        'Apply :class:`~qiskit.circuit.library.DCXGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            qubit1: The qubit(s) to apply the gate to.\n            qubit2: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.dcx import DCXGate
        return self.append(DCXGate(), [qubit1, qubit2], [])

    def ccx(self, control_qubit1: QubitSpecifier, control_qubit2: QubitSpecifier, target_qubit: QubitSpecifier, ctrl_state: str | int | None=None) -> InstructionSet:
        if False:
            while True:
                i = 10
        "Apply :class:`~qiskit.circuit.library.CCXGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            control_qubit1: The qubit(s) used as the first control.\n            control_qubit2: The qubit(s) used as the second control.\n            target_qubit: The qubit(s) targeted by the gate.\n            ctrl_state:\n                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling\n                on the '1' state.\n\n        Returns:\n            A handle to the instructions created.\n        "
        from .library.standard_gates.x import CCXGate
        return self.append(CCXGate(ctrl_state=ctrl_state), [control_qubit1, control_qubit2, target_qubit], [])

    @deprecate_func(since='0.45.0', additional_msg='Use QuantumCircuit.ccx as direct replacement.')
    def toffoli(self, control_qubit1: QubitSpecifier, control_qubit2: QubitSpecifier, target_qubit: QubitSpecifier) -> InstructionSet:
        if False:
            print('Hello World!')
        'Apply :class:`~qiskit.circuit.library.CCXGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            control_qubit1: The qubit(s) used as the first control.\n            control_qubit2: The qubit(s) used as the second control.\n            target_qubit: The qubit(s) targeted by the gate.\n\n        Returns:\n            A handle to the instructions created.\n\n        See Also:\n            QuantumCircuit.ccx: the same gate with a different name.\n        '
        return self.ccx(control_qubit1, control_qubit2, target_qubit)

    def mcx(self, control_qubits: Sequence[QubitSpecifier], target_qubit: QubitSpecifier, ancilla_qubits: QubitSpecifier | Sequence[QubitSpecifier] | None=None, mode: str='noancilla') -> InstructionSet:
        if False:
            return 10
        "Apply :class:`~qiskit.circuit.library.MCXGate`.\n\n        The multi-cX gate can be implemented using different techniques, which use different numbers\n        of ancilla qubits and have varying circuit depth. These modes are:\n\n        - ``'noancilla'``: Requires 0 ancilla qubits.\n        - ``'recursion'``: Requires 1 ancilla qubit if more than 4 controls are used, otherwise 0.\n        - ``'v-chain'``: Requires 2 less ancillas than the number of control qubits.\n        - ``'v-chain-dirty'``: Same as for the clean ancillas (but the circuit will be longer).\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            control_qubits: The qubits used as the controls.\n            target_qubit: The qubit(s) targeted by the gate.\n            ancilla_qubits: The qubits used as the ancillae, if the mode requires them.\n            mode: The choice of mode, explained further above.\n\n        Returns:\n            A handle to the instructions created.\n\n        Raises:\n            ValueError: if the given mode is not known, or if too few ancilla qubits are passed.\n            AttributeError: if no ancilla qubits are passed, but some are needed.\n        "
        from .library.standard_gates.x import MCXGrayCode, MCXRecursive, MCXVChain
        num_ctrl_qubits = len(control_qubits)
        available_implementations = {'noancilla': MCXGrayCode(num_ctrl_qubits), 'recursion': MCXRecursive(num_ctrl_qubits), 'v-chain': MCXVChain(num_ctrl_qubits, False), 'v-chain-dirty': MCXVChain(num_ctrl_qubits, dirty_ancillas=True), 'advanced': MCXRecursive(num_ctrl_qubits), 'basic': MCXVChain(num_ctrl_qubits, dirty_ancillas=False), 'basic-dirty-ancilla': MCXVChain(num_ctrl_qubits, dirty_ancillas=True)}
        if ancilla_qubits:
            _ = self.qbit_argument_conversion(ancilla_qubits)
        try:
            gate = available_implementations[mode]
        except KeyError as ex:
            all_modes = list(available_implementations.keys())
            raise ValueError(f'Unsupported mode ({mode}) selected, choose one of {all_modes}') from ex
        if hasattr(gate, 'num_ancilla_qubits') and gate.num_ancilla_qubits > 0:
            required = gate.num_ancilla_qubits
            if ancilla_qubits is None:
                raise AttributeError(f'No ancillas provided, but {required} are needed!')
            if not hasattr(ancilla_qubits, '__len__'):
                ancilla_qubits = [ancilla_qubits]
            if len(ancilla_qubits) < required:
                actually = len(ancilla_qubits)
                raise ValueError(f'At least {required} ancillas required, but {actually} given.')
            ancilla_qubits = ancilla_qubits[:required]
        else:
            ancilla_qubits = []
        return self.append(gate, control_qubits[:] + [target_qubit] + ancilla_qubits[:], [])

    @deprecate_func(since='0.45.0', additional_msg='Use QuantumCircuit.mcx as direct replacement.')
    def mct(self, control_qubits: Sequence[QubitSpecifier], target_qubit: QubitSpecifier, ancilla_qubits: QubitSpecifier | Sequence[QubitSpecifier] | None=None, mode: str='noancilla') -> InstructionSet:
        if False:
            while True:
                i = 10
        "Apply :class:`~qiskit.circuit.library.MCXGate`.\n\n        The multi-cX gate can be implemented using different techniques, which use different numbers\n        of ancilla qubits and have varying circuit depth. These modes are:\n\n        - ``'noancilla'``: Requires 0 ancilla qubits.\n        - ``'recursion'``: Requires 1 ancilla qubit if more than 4 controls are used, otherwise 0.\n        - ``'v-chain'``: Requires 2 less ancillas than the number of control qubits.\n        - ``'v-chain-dirty'``: Same as for the clean ancillas (but the circuit will be longer).\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            control_qubits: The qubits used as the controls.\n            target_qubit: The qubit(s) targeted by the gate.\n            ancilla_qubits: The qubits used as the ancillae, if the mode requires them.\n            mode: The choice of mode, explained further above.\n\n        Returns:\n            A handle to the instructions created.\n\n        Raises:\n            ValueError: if the given mode is not known, or if too few ancilla qubits are passed.\n            AttributeError: if no ancilla qubits are passed, but some are needed.\n\n        See Also:\n            QuantumCircuit.mcx: the same gate with a different name.\n        "
        return self.mcx(control_qubits, target_qubit, ancilla_qubits, mode)

    def y(self, qubit: QubitSpecifier) -> InstructionSet:
        if False:
            print('Hello World!')
        'Apply :class:`~qiskit.circuit.library.YGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            qubit: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.y import YGate
        return self.append(YGate(), [qubit], [])

    def cy(self, control_qubit: QubitSpecifier, target_qubit: QubitSpecifier, label: str | None=None, ctrl_state: str | int | None=None) -> InstructionSet:
        if False:
            for i in range(10):
                print('nop')
        "Apply :class:`~qiskit.circuit.library.CYGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            control_qubit: The qubit(s) used as the controls.\n            target_qubit: The qubit(s) targeted by the gate.\n            label: The string label of the gate in the circuit.\n            ctrl_state:\n                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling\n                on the '1' state.\n\n        Returns:\n            A handle to the instructions created.\n        "
        from .library.standard_gates.y import CYGate
        return self.append(CYGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], [])

    def z(self, qubit: QubitSpecifier) -> InstructionSet:
        if False:
            return 10
        'Apply :class:`~qiskit.circuit.library.ZGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            qubit: The qubit(s) to apply the gate to.\n\n        Returns:\n            A handle to the instructions created.\n        '
        from .library.standard_gates.z import ZGate
        return self.append(ZGate(), [qubit], [])

    def cz(self, control_qubit: QubitSpecifier, target_qubit: QubitSpecifier, label: str | None=None, ctrl_state: str | int | None=None) -> InstructionSet:
        if False:
            for i in range(10):
                print('nop')
        "Apply :class:`~qiskit.circuit.library.CZGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            control_qubit: The qubit(s) used as the controls.\n            target_qubit: The qubit(s) targeted by the gate.\n            label: The string label of the gate in the circuit.\n            ctrl_state:\n                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling\n                on the '1' state.\n\n        Returns:\n            A handle to the instructions created.\n        "
        from .library.standard_gates.z import CZGate
        return self.append(CZGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], [])

    def ccz(self, control_qubit1: QubitSpecifier, control_qubit2: QubitSpecifier, target_qubit: QubitSpecifier, label: str | None=None, ctrl_state: str | int | None=None) -> InstructionSet:
        if False:
            for i in range(10):
                print('nop')
        "Apply :class:`~qiskit.circuit.library.CCZGate`.\n\n        For the full matrix form of this gate, see the underlying gate documentation.\n\n        Args:\n            control_qubit1: The qubit(s) used as the first control.\n            control_qubit2: The qubit(s) used as the second control.\n            target_qubit: The qubit(s) targeted by the gate.\n            label: The string label of the gate in the circuit.\n            ctrl_state:\n                The control state in decimal, or as a bitstring (e.g. '10').  Defaults to controlling\n                on the '11' state.\n\n        Returns:\n            A handle to the instructions created.\n        "
        from .library.standard_gates.z import CCZGate
        return self.append(CCZGate(label=label, ctrl_state=ctrl_state), [control_qubit1, control_qubit2, target_qubit], [])

    def pauli(self, pauli_string: str, qubits: Sequence[QubitSpecifier]) -> InstructionSet:
        if False:
            for i in range(10):
                print('nop')
        "Apply :class:`~qiskit.circuit.library.PauliGate`.\n\n        Args:\n            pauli_string: A string representing the Pauli operator to apply, e.g. 'XX'.\n            qubits: The qubits to apply this gate to.\n\n        Returns:\n            A handle to the instructions created.\n        "
        from qiskit.circuit.library.generalized_gates.pauli import PauliGate
        return self.append(PauliGate(pauli_string), qubits, [])

    def initialize(self, params: Sequence[complex] | str | int, qubits: Sequence[QubitSpecifier] | None=None, normalize: bool=False):
        if False:
            for i in range(10):
                print('nop')
        "Initialize qubits in a specific state.\n\n        Qubit initialization is done by first resetting the qubits to :math:`|0\\rangle`\n        followed by calling :class:`qiskit.extensions.StatePreparation`\n        class to prepare the qubits in a specified state.\n        Both these steps are included in the\n        :class:`qiskit.extensions.Initialize` instruction.\n\n        Args:\n            params: The state to initialize to, can be either of the following.\n\n                * Statevector or vector of complex amplitudes to initialize to.\n                * Labels of basis states of the Pauli eigenstates Z, X, Y. See\n                  :meth:`.Statevector.from_label`. Notice the order of the labels is reversed with\n                  respect to the qubit index to be applied to. Example label '01' initializes the\n                  qubit zero to :math:`|1\\rangle` and the qubit one to :math:`|0\\rangle`.\n                * An integer that is used as a bitmap indicating which qubits to initialize to\n                  :math:`|1\\rangle`. Example: setting params to 5 would initialize qubit 0 and qubit\n                  2 to :math:`|1\\rangle` and qubit 1 to :math:`|0\\rangle`.\n\n            qubits: Qubits to initialize. If ``None`` the initialization is applied to all qubits in\n                the circuit.\n            normalize: Whether to normalize an input array to a unit vector.\n\n        Returns:\n            A handle to the instructions created.\n\n        Examples:\n            Prepare a qubit in the state :math:`(|0\\rangle - |1\\rangle) / \\sqrt{2}`.\n\n            .. code-block::\n\n                import numpy as np\n                from qiskit import QuantumCircuit\n\n                circuit = QuantumCircuit(1)\n                circuit.initialize([1/np.sqrt(2), -1/np.sqrt(2)], 0)\n                circuit.draw()\n\n            output:\n\n            .. parsed-literal::\n\n                     ┌──────────────────────────────┐\n                q_0: ┤ Initialize(0.70711,-0.70711) ├\n                     └──────────────────────────────┘\n\n\n            Initialize from a string two qubits in the state :math:`|10\\rangle`.\n            The order of the labels is reversed with respect to qubit index.\n            More information about labels for basis states are in\n            :meth:`.Statevector.from_label`.\n\n            .. code-block::\n\n                import numpy as np\n                from qiskit import QuantumCircuit\n\n                circuit = QuantumCircuit(2)\n                circuit.initialize('01', circuit.qubits)\n                circuit.draw()\n\n            output:\n\n            .. parsed-literal::\n\n                     ┌──────────────────┐\n                q_0: ┤0                 ├\n                     │  Initialize(0,1) │\n                q_1: ┤1                 ├\n                     └──────────────────┘\n\n            Initialize two qubits from an array of complex amplitudes.\n\n            .. code-block::\n\n                import numpy as np\n                from qiskit import QuantumCircuit\n\n                circuit = QuantumCircuit(2)\n                circuit.initialize([0, 1/np.sqrt(2), -1.j/np.sqrt(2), 0], circuit.qubits)\n                circuit.draw()\n\n            output:\n\n            .. parsed-literal::\n\n                     ┌────────────────────────────────────┐\n                q_0: ┤0                                   ├\n                     │  Initialize(0,0.70711,-0.70711j,0) │\n                q_1: ┤1                                   ├\n                     └────────────────────────────────────┘\n        "
        from .library.data_preparation.initializer import Initialize
        if qubits is None:
            qubits = self.qubits
        elif isinstance(qubits, (int, np.integer, slice, Qubit)):
            qubits = [qubits]
        num_qubits = len(qubits) if isinstance(params, int) else None
        return self.append(Initialize(params, num_qubits, normalize), qubits)

    def unitary(self, obj: np.ndarray | Gate | BaseOperator, qubits: Sequence[QubitSpecifier], label: str | None=None):
        if False:
            while True:
                i = 10
        'Apply unitary gate specified by ``obj`` to ``qubits``.\n\n        Args:\n            obj: Unitary operator.\n            qubits: The circuit qubits to apply the transformation to.\n            label: Unitary name for backend [Default: None].\n\n        Returns:\n            QuantumCircuit: The quantum circuit.\n\n        Example:\n\n            Apply a gate specified by a unitary matrix to a quantum circuit\n\n            .. code-block:: python\n\n                from qiskit import QuantumCircuit\n                matrix = [[0, 0, 0, 1],\n                        [0, 0, 1, 0],\n                        [1, 0, 0, 0],\n                        [0, 1, 0, 0]]\n                circuit = QuantumCircuit(2)\n                circuit.unitary(matrix, [0, 1])\n        '
        from .library.generalized_gates.unitary import UnitaryGate
        gate = UnitaryGate(obj, label=label)
        if gate.num_qubits == 1:
            if isinstance(qubits, (int, Qubit)) or len(qubits) > 1:
                qubits = [qubits]
        return self.append(gate, qubits, [])

    @deprecate_func(since='0.45.0', additional_msg='Instead, compose the circuit with a qiskit.circuit.library.Diagonal circuit.', pending=True)
    def diagonal(self, diag, qubit):
        if False:
            print('Hello World!')
        'Attach a diagonal gate to a circuit.\n\n        The decomposition is based on Theorem 7 given in "Synthesis of Quantum Logic Circuits" by\n        Shende et al. (https://arxiv.org/pdf/quant-ph/0406176.pdf).\n\n        Args:\n            diag (list): list of the 2^k diagonal entries (for a diagonal gate on k qubits).\n                Must contain at least two entries\n            qubit (QuantumRegister | list): list of k qubits the diagonal is\n                acting on (the order of the qubits specifies the computational basis in which the\n                diagonal gate is provided: the first element in diag acts on the state where all\n                the qubits in q are in the state 0, the second entry acts on the state where all\n                the qubits q[1],...,q[k-1] are in the state zero and q[0] is in the state 1,\n                and so on)\n\n        Returns:\n            QuantumCircuit: the diagonal gate which was attached to the circuit.\n\n        Raises:\n            QiskitError: if the list of the diagonal entries or the qubit list is in bad format;\n                if the number of diagonal entries is not 2^k, where k denotes the number of qubits\n        '
        from .library.generalized_gates.diagonal import DiagonalGate
        if isinstance(qubit, QuantumRegister):
            qubit = qubit[:]
        if not isinstance(qubit, list):
            raise QiskitError('The qubits must be provided as a list (also if there is only one qubit).')
        if not isinstance(diag, list):
            raise QiskitError('The diagonal entries are not provided in a list.')
        num_action_qubits = math.log2(len(diag))
        if not len(qubit) == num_action_qubits:
            raise QiskitError('The number of diagonal entries does not correspond to the number of qubits.')
        return self.append(DiagonalGate(diag), qubit)

    @deprecate_func(since='0.45.0', additional_msg='Instead, append a qiskit.circuit.library.Isometry to the circuit.', pending=True)
    def iso(self, isometry, q_input, q_ancillas_for_output, q_ancillas_zero=None, q_ancillas_dirty=None, epsilon=1e-10):
        if False:
            while True:
                i = 10
        '\n        Attach an arbitrary isometry from m to n qubits to a circuit. In particular,\n        this allows to attach arbitrary unitaries on n qubits (m=n) or to prepare any state\n        on n qubits (m=0).\n        The decomposition used here was introduced by Iten et al. in https://arxiv.org/abs/1501.06911.\n\n        Args:\n            isometry (ndarray): an isometry from m to n qubits, i.e., a (complex) ndarray of\n                dimension 2^n×2^m with orthonormal columns (given in the computational basis\n                specified by the order of the ancillas and the input qubits, where the ancillas\n                are considered to be more significant than the input qubits.).\n            q_input (QuantumRegister | list[Qubit]): list of m qubits where the input\n                to the isometry is fed in (empty list for state preparation).\n            q_ancillas_for_output (QuantumRegister | list[Qubit]): list of n-m ancilla\n                qubits that are used for the output of the isometry and which are assumed to start\n                in the zero state. The qubits are listed with increasing significance.\n            q_ancillas_zero (QuantumRegister | list[Qubit]): list of ancilla qubits\n                which are assumed to start in the zero state. Default is q_ancillas_zero = None.\n            q_ancillas_dirty (QuantumRegister | list[Qubit]): list of ancilla qubits\n                which can start in an arbitrary state. Default is q_ancillas_dirty = None.\n            epsilon (float): error tolerance of calculations.\n                Default is epsilon = _EPS.\n\n        Returns:\n            QuantumCircuit: the isometry is attached to the quantum circuit.\n\n        Raises:\n            QiskitError: if the array is not an isometry of the correct size corresponding to\n                the provided number of qubits.\n        '
        from .library.generalized_gates.isometry import Isometry
        if q_input is None:
            q_input = []
        if q_ancillas_for_output is None:
            q_ancillas_for_output = []
        if q_ancillas_zero is None:
            q_ancillas_zero = []
        if q_ancillas_dirty is None:
            q_ancillas_dirty = []
        if isinstance(q_input, QuantumRegister):
            q_input = q_input[:]
        if isinstance(q_ancillas_for_output, QuantumRegister):
            q_ancillas_for_output = q_ancillas_for_output[:]
        if isinstance(q_ancillas_zero, QuantumRegister):
            q_ancillas_zero = q_ancillas_zero[:]
        if isinstance(q_ancillas_dirty, QuantumRegister):
            q_ancillas_dirty = q_ancillas_dirty[:]
        return self.append(Isometry(isometry, len(q_ancillas_zero), len(q_ancillas_dirty), epsilon=epsilon), q_input + q_ancillas_for_output + q_ancillas_zero + q_ancillas_dirty)

    @deprecate_func(since='0.45.0', additional_msg='Instead, append a qiskit.circuit.library.HamiltonianGate to the circuit.', pending=True)
    def hamiltonian(self, operator, time, qubits, label=None):
        if False:
            return 10
        'Apply hamiltonian evolution to qubits.\n\n        This gate resolves to a :class:`~.library.UnitaryGate` as :math:`U(t) = exp(-i t H)`,\n        which can be decomposed into basis gates if it is 2 qubits or less, or\n        simulated directly in Aer for more qubits.\n\n        Args:\n            operator (matrix or Operator): a hermitian operator.\n            time (float or ParameterExpression): time evolution parameter.\n            qubits (Union[int, Tuple[int]]): The circuit qubits to apply the\n                transformation to.\n            label (str): unitary name for backend [Default: None].\n\n        Returns:\n            QuantumCircuit: The quantum circuit.\n        '
        from .library.hamiltonian_gate import HamiltonianGate
        if not isinstance(qubits, list):
            qubits = [qubits]
        return self.append(HamiltonianGate(data=operator, time=time, label=label), qubits, [])

    @deprecate_func(since='0.45.0', additional_msg='Instead, append a qiskit.circuit.library.UCGate to the circuit.', pending=True)
    def uc(self, gate_list, q_controls, q_target, up_to_diagonal=False):
        if False:
            print('Hello World!')
        "Attach a uniformly controlled gates (also called multiplexed gates) to a circuit.\n\n        The decomposition was introduced by Bergholm et al. in\n        https://arxiv.org/pdf/quant-ph/0410066.pdf.\n\n        Args:\n            gate_list (list[ndarray]): list of two qubit unitaries [U_0,...,U_{2^k-1}],\n                where each single-qubit unitary U_i is a given as a 2*2 array\n            q_controls (QuantumRegister | list[(QuantumRegister,int)]): list of k control qubits.\n                The qubits are ordered according to their significance in the computational basis.\n                For example if q_controls=[q[1],q[2]] (with q = QuantumRegister(2)),\n                the unitary U_0 is performed\xa0if q[1] and q[2] are in the state zero, U_1 is\n                performed if q[2] is in the state zero and q[1] is in the state one, and so on\n            q_target (QuantumRegister | tuple(QuantumRegister, int)):  target qubit, where we act on with\n                the single-qubit gates.\n            up_to_diagonal (bool): If set to True, the uniformly controlled gate is decomposed up\n                to a diagonal gate, i.e. a unitary u' is implemented such that there exists a\n                diagonal gate d with u = d.dot(u'), where the unitary u describes the uniformly\n                controlled gate\n\n        Returns:\n            QuantumCircuit: the uniformly controlled gate is attached to the circuit.\n\n        Raises:\n            QiskitError: if the list number of control qubits does not correspond to the provided\n                number of single-qubit unitaries; if an input is of the wrong type\n        "
        from .library.generalized_gates.uc import UCGate
        if isinstance(q_controls, QuantumRegister):
            q_controls = q_controls[:]
        if isinstance(q_target, QuantumRegister):
            q_target = q_target[:]
            if len(q_target) == 1:
                q_target = q_target[0]
            else:
                raise QiskitError('The target qubit is a QuantumRegister containing more than one qubit.')
        if not isinstance(q_controls, list):
            raise QiskitError('The control qubits must be provided as a list (also if there is only one control qubit).')
        if not isinstance(gate_list, list):
            raise QiskitError('The single-qubit unitaries are not provided in a list.')
        num_contr = math.log2(len(gate_list))
        if num_contr < 0 or not num_contr.is_integer():
            raise QiskitError('The number of controlled single-qubit gates is not a non negative power of 2.')
        if num_contr != len(q_controls):
            raise QiskitError('Number of controlled gates does not correspond to the number of control qubits.')
        return self.append(UCGate(gate_list, up_to_diagonal), [q_target] + q_controls)

    @deprecate_func(since='0.45.0', additional_msg='Instead, append a qiskit.circuit.library.UCRXGate to the circuit.', pending=True)
    def ucrx(self, angle_list: list[float], q_controls: Sequence[QubitSpecifier], q_target: QubitSpecifier):
        if False:
            print('Hello World!')
        'Attach a uniformly controlled (also called multiplexed) Rx rotation gate to a circuit.\n\n        The decomposition is base on https://arxiv.org/pdf/quant-ph/0406176.pdf by Shende et al.\n\n        Args:\n            angle_list (list[float]): list of (real) rotation angles :math:`[a_0,...,a_{2^k-1}]`\n            q_controls (Sequence[QubitSpecifier]): list of k control qubits\n                (or empty list if no controls). The control qubits are ordered according to their\n                significance in increasing order: For example if ``q_controls=[q[0],q[1]]``\n                (with ``q = QuantumRegister(2)``), the rotation ``Rx(a_0)`` is performed if ``q[0]``\n                and ``q[1]`` are in the state zero, the rotation ``Rx(a_1)`` is performed if ``q[0]``\n                is in the state one and ``q[1]`` is in the state zero, and so on\n            q_target (QubitSpecifier): target qubit, where we act on with\n                the single-qubit rotation gates\n\n        Returns:\n            QuantumCircuit: the uniformly controlled rotation gate is attached to the circuit.\n\n        Raises:\n            QiskitError: if the list number of control qubits does not correspond to the provided\n                number of single-qubit unitaries; if an input is of the wrong type\n        '
        from .library.generalized_gates.ucrx import UCRXGate
        if isinstance(q_controls, QuantumRegister):
            q_controls = q_controls[:]
        if isinstance(q_target, QuantumRegister):
            q_target = q_target[:]
            if len(q_target) == 1:
                q_target = q_target[0]
            else:
                raise QiskitError('The target qubit is a QuantumRegister containing more than one qubit.')
        if not isinstance(angle_list, list):
            raise QiskitError('The angles must be provided as a list.')
        num_contr = math.log2(len(angle_list))
        if num_contr < 0 or not num_contr.is_integer():
            raise QiskitError('The number of controlled rotation gates is not a non-negative power of 2.')
        if num_contr != len(q_controls):
            raise QiskitError('Number of controlled rotations does not correspond to the number of control-qubits.')
        return self.append(UCRXGate(angle_list), [q_target] + q_controls, [])

    @deprecate_func(since='0.45.0', additional_msg='Instead, append a qiskit.circuit.library.UCRYGate to the circuit.', pending=True)
    def ucry(self, angle_list: list[float], q_controls: Sequence[QubitSpecifier], q_target: QubitSpecifier):
        if False:
            for i in range(10):
                print('nop')
        'Attach a uniformly controlled (also called multiplexed) Ry rotation gate to a circuit.\n\n        The decomposition is base on https://arxiv.org/pdf/quant-ph/0406176.pdf by Shende et al.\n\n        Args:\n            angle_list (list[float]): list of (real) rotation angles :math:`[a_0,...,a_{2^k-1}]`\n            q_controls (Sequence[QubitSpecifier]): list of k control qubits\n                (or empty list if no controls). The control qubits are ordered according to their\n                significance in increasing order: For example if ``q_controls=[q[0],q[1]]``\n                (with ``q = QuantumRegister(2)``), the rotation ``Ry(a_0)`` is performed if ``q[0]``\n                and ``q[1]`` are in the state zero, the rotation ``Ry(a_1)`` is performed if ``q[0]``\n                is in the state one and ``q[1]`` is in the state zero, and so on\n            q_target (QubitSpecifier): target qubit, where we act on with\n                the single-qubit rotation gates\n\n        Returns:\n            QuantumCircuit: the uniformly controlled rotation gate is attached to the circuit.\n\n        Raises:\n            QiskitError: if the list number of control qubits does not correspond to the provided\n                number of single-qubit unitaries; if an input is of the wrong type\n        '
        from .library.generalized_gates.ucry import UCRYGate
        if isinstance(q_controls, QuantumRegister):
            q_controls = q_controls[:]
        if isinstance(q_target, QuantumRegister):
            q_target = q_target[:]
            if len(q_target) == 1:
                q_target = q_target[0]
            else:
                raise QiskitError('The target qubit is a QuantumRegister containing more than one qubit.')
        if not isinstance(angle_list, list):
            raise QiskitError('The angles must be provided as a list.')
        num_contr = math.log2(len(angle_list))
        if num_contr < 0 or not num_contr.is_integer():
            raise QiskitError('The number of controlled rotation gates is not a non-negative power of 2.')
        if num_contr != len(q_controls):
            raise QiskitError('Number of controlled rotations does not correspond to the number of control-qubits.')
        return self.append(UCRYGate(angle_list), [q_target] + q_controls, [])

    @deprecate_func(since='0.45.0', additional_msg='Instead, append a qiskit.circuit.library.UCRZGate to the circuit.', pending=True)
    def ucrz(self, angle_list: list[float], q_controls: Sequence[QubitSpecifier], q_target: QubitSpecifier):
        if False:
            while True:
                i = 10
        'Attach a uniformly controlled (also called multiplexed) Rz rotation gate to a circuit.\n\n        The decomposition is base on https://arxiv.org/pdf/quant-ph/0406176.pdf by Shende et al.\n\n        Args:\n            angle_list (list[float]): list of (real) rotation angles :math:`[a_0,...,a_{2^k-1}]`\n            q_controls (Sequence[QubitSpecifier]): list of k control qubits\n                (or empty list if no controls). The control qubits are ordered according to their\n                significance in increasing order: For example if ``q_controls=[q[0],q[1]]``\n                (with ``q = QuantumRegister(2)``), the rotation ``Rz(a_0)`` is performed if ``q[0]``\n                and ``q[1]`` are in the state zero, the rotation ``Rz(a_1)`` is performed if ``q[0]``\n                is in the state one and ``q[1]`` is in the state zero, and so on\n            q_target (QubitSpecifier): target qubit, where we act on with\n                the single-qubit rotation gates\n\n        Returns:\n            QuantumCircuit: the uniformly controlled rotation gate is attached to the circuit.\n\n        Raises:\n            QiskitError: if the list number of control qubits does not correspond to the provided\n                number of single-qubit unitaries; if an input is of the wrong type\n        '
        from .library.generalized_gates.ucrz import UCRZGate
        if isinstance(q_controls, QuantumRegister):
            q_controls = q_controls[:]
        if isinstance(q_target, QuantumRegister):
            q_target = q_target[:]
            if len(q_target) == 1:
                q_target = q_target[0]
            else:
                raise QiskitError('The target qubit is a QuantumRegister containing more than one qubit.')
        if not isinstance(angle_list, list):
            raise QiskitError('The angles must be provided as a list.')
        num_contr = math.log2(len(angle_list))
        if num_contr < 0 or not num_contr.is_integer():
            raise QiskitError('The number of controlled rotation gates is not a non-negative power of 2.')
        if num_contr != len(q_controls):
            raise QiskitError('Number of controlled rotations does not correspond to the number of control-qubits.')
        return self.append(UCRZGate(angle_list), [q_target] + q_controls, [])

    @deprecate_func(since='0.45.0', additional_msg='Instead, use the QuantumCircuit.unitary method.')
    def squ(self, unitary_matrix, qubit, mode='ZYZ', up_to_diagonal=False):
        if False:
            for i in range(10):
                print('nop')
        'Decompose an arbitrary 2*2 unitary into three rotation gates.\n\n        Note that the decomposition is up to a global phase shift.\n        (This is a well known decomposition which can be found for example in Nielsen and Chuang\'s book\n        "Quantum computation and quantum information".)\n\n        Args:\n            unitary_matrix (ndarray): 2*2 unitary (given as a (complex) ndarray).\n            qubit (QuantumRegister or Qubit): The qubit which the gate is acting on.\n            mode (string): determines the used decomposition by providing the rotation axes.\n                The allowed modes are: "ZYZ" (default)\n            up_to_diagonal (bool):  if set to True, the single-qubit unitary is decomposed up to\n                a diagonal matrix, i.e. a unitary u\' is implemented such that there exists a 2*2\n                diagonal gate d with u = d.dot(u\')\n\n        Returns:\n            InstructionSet: The single-qubit unitary instruction attached to the circuit.\n\n        Raises:\n            QiskitError: if the format is wrong; if the array u is not unitary\n        '
        from qiskit.extensions.quantum_initializer.squ import SingleQubitUnitary
        if isinstance(qubit, QuantumRegister):
            qubit = qubit[:]
            if len(qubit) == 1:
                qubit = qubit[0]
            else:
                raise QiskitError('The target qubit is a QuantumRegister containing more than one qubit.')
        if not isinstance(qubit, Qubit):
            raise QiskitError('The target qubit is not a single qubit from a QuantumRegister.')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            squ = SingleQubitUnitary(unitary_matrix, mode, up_to_diagonal)
        return self.append(squ, [qubit], [])

    @deprecate_func(since='0.45.0', additional_msg="The Snapshot instruction has been superseded by Qiskit Aer's save instructions, see https://qiskit.org/ecosystem/aer/apidocs/aer_library.html#saving-simulator-data.")
    def snapshot(self, label, snapshot_type='statevector', qubits=None, params=None):
        if False:
            i = 10
            return i + 15
        'Take a statevector snapshot of the internal simulator representation.\n        Works on all qubits, and prevents reordering (like barrier).\n\n        For other types of snapshots use the Snapshot extension directly.\n\n        Args:\n            label (str): a snapshot label to report the result.\n            snapshot_type (str): the type of the snapshot.\n            qubits (list or None): the qubits to apply snapshot to [Default: None].\n            params (list or None): the parameters for snapshot_type [Default: None].\n\n        Returns:\n            QuantumCircuit: with attached command\n\n        Raises:\n            ExtensionError: malformed command\n        '
        from qiskit.extensions.simulator.snapshot import Snapshot
        from qiskit.extensions.exceptions import ExtensionError
        if isinstance(qubits, QuantumRegister):
            qubits = qubits[:]
        if not qubits:
            tuples = []
            if isinstance(self, QuantumCircuit):
                for register in self.qregs:
                    tuples.append(register)
            if not tuples:
                raise ExtensionError('no qubits for snapshot')
            qubits = []
            for tuple_element in tuples:
                if isinstance(tuple_element, QuantumRegister):
                    for j in range(tuple_element.size):
                        qubits.append(tuple_element[j])
                else:
                    qubits.append(tuple_element)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            snap = Snapshot(label, snapshot_type=snapshot_type, num_qubits=len(qubits), params=params)
        return self.append(snap, qubits)

    def _push_scope(self, qubits: Iterable[Qubit]=(), clbits: Iterable[Clbit]=(), registers: Iterable[Register]=(), allow_jumps: bool=True, forbidden_message: Optional[str]=None):
        if False:
            print('Hello World!')
        'Add a scope for collecting instructions into this circuit.\n\n        This should only be done by the control-flow context managers, which will handle cleaning up\n        after themselves at the end as well.\n\n        Args:\n            qubits: Any qubits that this scope should automatically use.\n            clbits: Any clbits that this scope should automatically use.\n            allow_jumps: Whether this scope allows jumps to be used within it.\n            forbidden_message: If given, all attempts to add instructions to this scope will raise a\n                :exc:`.CircuitError` with this message.\n        '
        from qiskit.circuit.controlflow.builder import ControlFlowBuilderBlock
        if self._control_flow_scopes:
            resource_requester = self._control_flow_scopes[-1].request_classical_resource
        else:
            resource_requester = self._resolve_classical_resource
        self._control_flow_scopes.append(ControlFlowBuilderBlock(qubits, clbits, resource_requester=resource_requester, registers=registers, allow_jumps=allow_jumps, forbidden_message=forbidden_message))

    def _pop_scope(self) -> 'qiskit.circuit.controlflow.builder.ControlFlowBuilderBlock':
        if False:
            for i in range(10):
                print('nop')
        'Finish a scope used in the control-flow builder interface, and return it to the caller.\n\n        This should only be done by the control-flow context managers, since they naturally\n        synchronise the creation and deletion of stack elements.'
        return self._control_flow_scopes.pop()

    def _peek_previous_instruction_in_scope(self) -> CircuitInstruction:
        if False:
            for i in range(10):
                print('nop')
        'Return the instruction 3-tuple of the most recent instruction in the current scope, even\n        if that scope is currently under construction.\n\n        This function is only intended for use by the control-flow ``if``-statement builders, which\n        may need to modify a previous instruction.'
        if self._control_flow_scopes:
            return self._control_flow_scopes[-1].peek()
        if not self._data:
            raise CircuitError('This circuit contains no instructions.')
        return self._data[-1]

    def _pop_previous_instruction_in_scope(self) -> CircuitInstruction:
        if False:
            while True:
                i = 10
        'Return the instruction 3-tuple of the most recent instruction in the current scope, even\n        if that scope is currently under construction, and remove it from that scope.\n\n        This function is only intended for use by the control-flow ``if``-statement builders, which\n        may need to replace a previous instruction with another.\n        '
        if self._control_flow_scopes:
            return self._control_flow_scopes[-1].pop()
        if not self._data:
            raise CircuitError('This circuit contains no instructions.')
        instruction = self._data.pop()
        if isinstance(instruction.operation, Instruction):
            self._update_parameter_table_on_instruction_removal(instruction)
        return instruction

    def _update_parameter_table_on_instruction_removal(self, instruction: CircuitInstruction):
        if False:
            return 10
        'Update the :obj:`.ParameterTable` of this circuit given that an instance of the given\n        ``instruction`` has just been removed from the circuit.\n\n        .. note::\n\n            This does not account for the possibility for the same instruction instance being added\n            more than once to the circuit.  At the time of writing (2021-11-17, main commit 271a82f)\n            there is a defensive ``deepcopy`` of parameterised instructions inside\n            :meth:`.QuantumCircuit.append`, so this should be safe.  Trying to account for it would\n            involve adding a potentially quadratic-scaling loop to check each entry in ``data``.\n        '
        atomic_parameters: list[tuple[Parameter, int]] = []
        for (index, parameter) in enumerate(instruction.operation.params):
            if isinstance(parameter, (ParameterExpression, QuantumCircuit)):
                atomic_parameters.extend(((p, index) for p in parameter.parameters))
        for (atomic_parameter, index) in atomic_parameters:
            new_entries = self._parameter_table[atomic_parameter].copy()
            new_entries.discard((instruction.operation, index))
            if not new_entries:
                del self._parameter_table[atomic_parameter]
                self._parameters = None
            else:
                self._parameter_table[atomic_parameter] = new_entries

    @typing.overload
    def while_loop(self, condition: tuple[ClassicalRegister | Clbit, int] | expr.Expr, body: None, qubits: None, clbits: None, *, label: str | None) -> 'qiskit.circuit.controlflow.while_loop.WhileLoopContext':
        if False:
            return 10
        ...

    @typing.overload
    def while_loop(self, condition: tuple[ClassicalRegister | Clbit, int] | expr.Expr, body: 'QuantumCircuit', qubits: Sequence[QubitSpecifier], clbits: Sequence[ClbitSpecifier], *, label: str | None) -> InstructionSet:
        if False:
            print('Hello World!')
        ...

    def while_loop(self, condition, body=None, qubits=None, clbits=None, *, label=None):
        if False:
            i = 10
            return i + 15
        'Create a ``while`` loop on this circuit.\n\n        There are two forms for calling this function.  If called with all its arguments (with the\n        possible exception of ``label``), it will create a\n        :obj:`~qiskit.circuit.controlflow.WhileLoopOp` with the given ``body``.  If ``body`` (and\n        ``qubits`` and ``clbits``) are *not* passed, then this acts as a context manager, which\n        will automatically build a :obj:`~qiskit.circuit.controlflow.WhileLoopOp` when the scope\n        finishes.  In this form, you do not need to keep track of the qubits or clbits you are\n        using, because the scope will handle it for you.\n\n        Example usage::\n\n            from qiskit.circuit import QuantumCircuit, Clbit, Qubit\n            bits = [Qubit(), Qubit(), Clbit()]\n            qc = QuantumCircuit(bits)\n\n            with qc.while_loop((bits[2], 0)):\n                qc.h(0)\n                qc.cx(0, 1)\n                qc.measure(0, 0)\n\n        Args:\n            condition (Tuple[Union[ClassicalRegister, Clbit], int]): An equality condition to be\n                checked prior to executing ``body``. The left-hand side of the condition must be a\n                :obj:`~ClassicalRegister` or a :obj:`~Clbit`, and the right-hand side must be an\n                integer or boolean.\n            body (Optional[QuantumCircuit]): The loop body to be repeatedly executed.  Omit this to\n                use the context-manager mode.\n            qubits (Optional[Sequence[Qubit]]): The circuit qubits over which the loop body should\n                be run.  Omit this to use the context-manager mode.\n            clbits (Optional[Sequence[Clbit]]): The circuit clbits over which the loop body should\n                be run.  Omit this to use the context-manager mode.\n            label (Optional[str]): The string label of the instruction in the circuit.\n\n        Returns:\n            InstructionSet or WhileLoopContext: If used in context-manager mode, then this should be\n            used as a ``with`` resource, which will infer the block content and operands on exit.\n            If the full form is used, then this returns a handle to the instructions created.\n\n        Raises:\n            CircuitError: if an incorrect calling convention is used.\n        '
        from qiskit.circuit.controlflow.while_loop import WhileLoopOp, WhileLoopContext
        if isinstance(condition, expr.Expr):
            condition = self._validate_expr(condition)
        else:
            condition = (self._resolve_classical_resource(condition[0]), condition[1])
        if body is None:
            if qubits is not None or clbits is not None:
                raise CircuitError("When using 'while_loop' as a context manager, you cannot pass qubits or clbits.")
            return WhileLoopContext(self, condition, label=label)
        elif qubits is None or clbits is None:
            raise CircuitError("When using 'while_loop' with a body, you must pass qubits and clbits.")
        return self.append(WhileLoopOp(condition, body, label), qubits, clbits)

    @typing.overload
    def for_loop(self, indexset: Iterable[int], loop_parameter: Parameter | None, body: None, qubits: None, clbits: None, *, label: str | None) -> 'qiskit.circuit.controlflow.for_loop.ForLoopContext':
        if False:
            while True:
                i = 10
        ...

    @typing.overload
    def for_loop(self, indexset: Iterable[int], loop_parameter: Union[Parameter, None], body: 'QuantumCircuit', qubits: Sequence[QubitSpecifier], clbits: Sequence[ClbitSpecifier], *, label: str | None) -> InstructionSet:
        if False:
            return 10
        ...

    def for_loop(self, indexset, loop_parameter=None, body=None, qubits=None, clbits=None, *, label=None):
        if False:
            print('Hello World!')
        'Create a ``for`` loop on this circuit.\n\n        There are two forms for calling this function.  If called with all its arguments (with the\n        possible exception of ``label``), it will create a\n        :class:`~qiskit.circuit.ForLoopOp` with the given ``body``.  If ``body`` (and\n        ``qubits`` and ``clbits``) are *not* passed, then this acts as a context manager, which,\n        when entered, provides a loop variable (unless one is given, in which case it will be\n        reused) and will automatically build a :class:`~qiskit.circuit.ForLoopOp` when the\n        scope finishes.  In this form, you do not need to keep track of the qubits or clbits you are\n        using, because the scope will handle it for you.\n\n        For example::\n\n            from qiskit import QuantumCircuit\n            qc = QuantumCircuit(2, 1)\n\n            with qc.for_loop(range(5)) as i:\n                qc.h(0)\n                qc.cx(0, 1)\n                qc.measure(0, 0)\n                qc.break_loop().c_if(0, True)\n\n        Args:\n            indexset (Iterable[int]): A collection of integers to loop over.  Always necessary.\n            loop_parameter (Optional[Parameter]): The parameter used within ``body`` to which\n                the values from ``indexset`` will be assigned.  In the context-manager form, if this\n                argument is not supplied, then a loop parameter will be allocated for you and\n                returned as the value of the ``with`` statement.  This will only be bound into the\n                circuit if it is used within the body.\n\n                If this argument is ``None`` in the manual form of this method, ``body`` will be\n                repeated once for each of the items in ``indexset`` but their values will be\n                ignored.\n            body (Optional[QuantumCircuit]): The loop body to be repeatedly executed.  Omit this to\n                use the context-manager mode.\n            qubits (Optional[Sequence[QubitSpecifier]]): The circuit qubits over which the loop body\n                should be run.  Omit this to use the context-manager mode.\n            clbits (Optional[Sequence[ClbitSpecifier]]): The circuit clbits over which the loop body\n                should be run.  Omit this to use the context-manager mode.\n            label (Optional[str]): The string label of the instruction in the circuit.\n\n        Returns:\n            InstructionSet or ForLoopContext: depending on the call signature, either a context\n            manager for creating the for loop (it will automatically be added to the circuit at the\n            end of the block), or an :obj:`~InstructionSet` handle to the appended loop operation.\n\n        Raises:\n            CircuitError: if an incorrect calling convention is used.\n        '
        from qiskit.circuit.controlflow.for_loop import ForLoopOp, ForLoopContext
        if body is None:
            if qubits is not None or clbits is not None:
                raise CircuitError("When using 'for_loop' as a context manager, you cannot pass qubits or clbits.")
            return ForLoopContext(self, indexset, loop_parameter, label=label)
        elif qubits is None or clbits is None:
            raise CircuitError("When using 'for_loop' with a body, you must pass qubits and clbits.")
        return self.append(ForLoopOp(indexset, loop_parameter, body, label), qubits, clbits)

    @typing.overload
    def if_test(self, condition: tuple[ClassicalRegister | Clbit, int], true_body: None, qubits: None, clbits: None, *, label: str | None) -> 'qiskit.circuit.controlflow.if_else.IfContext':
        if False:
            return 10
        ...

    @typing.overload
    def if_test(self, condition: tuple[ClassicalRegister | Clbit, int], true_body: 'QuantumCircuit', qubits: Sequence[QubitSpecifier], clbits: Sequence[ClbitSpecifier], *, label: str | None=None) -> InstructionSet:
        if False:
            for i in range(10):
                print('nop')
        ...

    def if_test(self, condition, true_body=None, qubits=None, clbits=None, *, label=None):
        if False:
            while True:
                i = 10
        'Create an ``if`` statement on this circuit.\n\n        There are two forms for calling this function.  If called with all its arguments (with the\n        possible exception of ``label``), it will create a\n        :obj:`~qiskit.circuit.IfElseOp` with the given ``true_body``, and there will be\n        no branch for the ``false`` condition (see also the :meth:`.if_else` method).  However, if\n        ``true_body`` (and ``qubits`` and ``clbits``) are *not* passed, then this acts as a context\n        manager, which can be used to build ``if`` statements.  The return value of the ``with``\n        statement is a chainable context manager, which can be used to create subsequent ``else``\n        blocks.  In this form, you do not need to keep track of the qubits or clbits you are using,\n        because the scope will handle it for you.\n\n        For example::\n\n            from qiskit.circuit import QuantumCircuit, Qubit, Clbit\n            bits = [Qubit(), Qubit(), Qubit(), Clbit(), Clbit()]\n            qc = QuantumCircuit(bits)\n\n            qc.h(0)\n            qc.cx(0, 1)\n            qc.measure(0, 0)\n            qc.h(0)\n            qc.cx(0, 1)\n            qc.measure(0, 1)\n\n            with qc.if_test((bits[3], 0)) as else_:\n                qc.x(2)\n            with else_:\n                qc.h(2)\n                qc.z(2)\n\n        Args:\n            condition (Tuple[Union[ClassicalRegister, Clbit], int]): A condition to be evaluated at\n                circuit runtime which, if true, will trigger the evaluation of ``true_body``. Can be\n                specified as either a tuple of a ``ClassicalRegister`` to be tested for equality\n                with a given ``int``, or as a tuple of a ``Clbit`` to be compared to either a\n                ``bool`` or an ``int``.\n            true_body (Optional[QuantumCircuit]): The circuit body to be run if ``condition`` is\n                true.\n            qubits (Optional[Sequence[QubitSpecifier]]): The circuit qubits over which the if/else\n                should be run.\n            clbits (Optional[Sequence[ClbitSpecifier]]): The circuit clbits over which the if/else\n                should be run.\n            label (Optional[str]): The string label of the instruction in the circuit.\n\n        Returns:\n            InstructionSet or IfContext: depending on the call signature, either a context\n            manager for creating the ``if`` block (it will automatically be added to the circuit at\n            the end of the block), or an :obj:`~InstructionSet` handle to the appended conditional\n            operation.\n\n        Raises:\n            CircuitError: If the provided condition references Clbits outside the\n                enclosing circuit.\n            CircuitError: if an incorrect calling convention is used.\n\n        Returns:\n            A handle to the instruction created.\n        '
        from qiskit.circuit.controlflow.if_else import IfElseOp, IfContext
        if isinstance(condition, expr.Expr):
            condition = self._validate_expr(condition)
        else:
            condition = (self._resolve_classical_resource(condition[0]), condition[1])
        if true_body is None:
            if qubits is not None or clbits is not None:
                raise CircuitError("When using 'if_test' as a context manager, you cannot pass qubits or clbits.")
            in_loop = bool(self._control_flow_scopes and self._control_flow_scopes[-1].allow_jumps)
            return IfContext(self, condition, in_loop=in_loop, label=label)
        elif qubits is None or clbits is None:
            raise CircuitError("When using 'if_test' with a body, you must pass qubits and clbits.")
        return self.append(IfElseOp(condition, true_body, None, label), qubits, clbits)

    def if_else(self, condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | tuple[Clbit, bool], true_body: 'QuantumCircuit', false_body: 'QuantumCircuit', qubits: Sequence[QubitSpecifier], clbits: Sequence[ClbitSpecifier], label: str | None=None) -> InstructionSet:
        if False:
            while True:
                i = 10
        'Apply :class:`~qiskit.circuit.IfElseOp`.\n\n        .. note::\n\n            This method does not have an associated context-manager form, because it is already\n            handled by the :meth:`.if_test` method.  You can use the ``else`` part of that with\n            something such as::\n\n                from qiskit.circuit import QuantumCircuit, Qubit, Clbit\n                bits = [Qubit(), Qubit(), Clbit()]\n                qc = QuantumCircuit(bits)\n                qc.h(0)\n                qc.cx(0, 1)\n                qc.measure(0, 0)\n                with qc.if_test((bits[2], 0)) as else_:\n                    qc.h(0)\n                with else_:\n                    qc.x(0)\n\n        Args:\n            condition: A condition to be evaluated at circuit runtime which,\n                if true, will trigger the evaluation of ``true_body``. Can be\n                specified as either a tuple of a ``ClassicalRegister`` to be\n                tested for equality with a given ``int``, or as a tuple of a\n                ``Clbit`` to be compared to either a ``bool`` or an ``int``.\n            true_body: The circuit body to be run if ``condition`` is true.\n            false_body: The circuit to be run if ``condition`` is false.\n            qubits: The circuit qubits over which the if/else should be run.\n            clbits: The circuit clbits over which the if/else should be run.\n            label: The string label of the instruction in the circuit.\n\n        Raises:\n            CircuitError: If the provided condition references Clbits outside the\n                enclosing circuit.\n\n        Returns:\n            A handle to the instruction created.\n        '
        from qiskit.circuit.controlflow.if_else import IfElseOp
        if isinstance(condition, expr.Expr):
            condition = self._validate_expr(condition)
        else:
            condition = (self._resolve_classical_resource(condition[0]), condition[1])
        return self.append(IfElseOp(condition, true_body, false_body, label), qubits, clbits)

    @typing.overload
    def switch(self, target: Union[ClbitSpecifier, ClassicalRegister], cases: None, qubits: None, clbits: None, *, label: Optional[str]) -> 'qiskit.circuit.controlflow.switch_case.SwitchContext':
        if False:
            return 10
        ...

    @typing.overload
    def switch(self, target: Union[ClbitSpecifier, ClassicalRegister], cases: Iterable[Tuple[typing.Any, QuantumCircuit]], qubits: Sequence[QubitSpecifier], clbits: Sequence[ClbitSpecifier], *, label: Optional[str]) -> InstructionSet:
        if False:
            i = 10
            return i + 15
        ...

    def switch(self, target, cases=None, qubits=None, clbits=None, *, label=None):
        if False:
            print('Hello World!')
        'Create a ``switch``/``case`` structure on this circuit.\n\n        There are two forms for calling this function.  If called with all its arguments (with the\n        possible exception of ``label``), it will create a :class:`.SwitchCaseOp` with the given\n        case structure.  If ``cases`` (and ``qubits`` and ``clbits``) are *not* passed, then this\n        acts as a context manager, which will automatically build a :class:`.SwitchCaseOp` when the\n        scope finishes.  In this form, you do not need to keep track of the qubits or clbits you are\n        using, because the scope will handle it for you.\n\n        Example usage::\n\n            from qiskit.circuit import QuantumCircuit, ClassicalRegister, QuantumRegister\n            qreg = QuantumRegister(3)\n            creg = ClassicalRegister(3)\n            qc = QuantumCircuit(qreg, creg)\n            qc.h([0, 1, 2])\n            qc.measure([0, 1, 2], [0, 1, 2])\n\n            with qc.switch(creg) as case:\n                with case(0):\n                    qc.x(0)\n                with case(1, 2):\n                    qc.z(1)\n                with case(case.DEFAULT):\n                    qc.cx(0, 1)\n\n        Args:\n            target (Union[ClassicalRegister, Clbit]): The classical value to switch one.  This must\n                be integer-like.\n            cases (Iterable[Tuple[typing.Any, QuantumCircuit]]): A sequence of case specifiers.\n                Each tuple defines one case body (the second item).  The first item of the tuple can\n                be either a single integer value, the special value :data:`.CASE_DEFAULT`, or a\n                tuple of several integer values.  Each of the integer values will be tried in turn;\n                control will then pass to the body corresponding to the first match.\n                :data:`.CASE_DEFAULT` matches all possible values.  Omit in context-manager form.\n            qubits (Sequence[Qubit]): The circuit qubits over which all case bodies execute. Omit in\n                context-manager form.\n            clbits (Sequence[Clbit]): The circuit clbits over which all case bodies execute. Omit in\n                context-manager form.\n            label (Optional[str]): The string label of the instruction in the circuit.\n\n        Returns:\n            InstructionSet or SwitchCaseContext: If used in context-manager mode, then this should\n            be used as a ``with`` resource, which will return an object that can be repeatedly\n            entered to produce cases for the switch statement.  If the full form is used, then this\n            returns a handle to the instructions created.\n\n        Raises:\n            CircuitError: if an incorrect calling convention is used.\n        '
        from qiskit.circuit.controlflow.switch_case import SwitchCaseOp, SwitchContext
        if isinstance(target, expr.Expr):
            target = self._validate_expr(target)
        else:
            target = self._resolve_classical_resource(target)
        if cases is None:
            if qubits is not None or clbits is not None:
                raise CircuitError("When using 'switch' as a context manager, you cannot pass qubits or clbits.")
            in_loop = bool(self._control_flow_scopes and self._control_flow_scopes[-1].allow_jumps)
            return SwitchContext(self, target, in_loop=in_loop, label=label)
        if qubits is None or clbits is None:
            raise CircuitError("When using 'switch' with cases, you must pass qubits and clbits.")
        return self.append(SwitchCaseOp(target, cases, label=label), qubits, clbits)

    def break_loop(self) -> InstructionSet:
        if False:
            print('Hello World!')
        'Apply :class:`~qiskit.circuit.BreakLoopOp`.\n\n        .. warning::\n\n            If you are using the context-manager "builder" forms of :meth:`.if_test`,\n            :meth:`.for_loop` or :meth:`.while_loop`, you can only call this method if you are\n            within a loop context, because otherwise the "resource width" of the operation cannot be\n            determined.  This would quickly lead to invalid circuits, and so if you are trying to\n            construct a reusable loop body (without the context managers), you must also use the\n            non-context-manager form of :meth:`.if_test` and :meth:`.if_else`.  Take care that the\n            :obj:`.BreakLoopOp` instruction must span all the resources of its containing loop, not\n            just the immediate scope.\n\n        Returns:\n            A handle to the instruction created.\n\n        Raises:\n            CircuitError: if this method was called within a builder context, but not contained\n                within a loop.\n        '
        from qiskit.circuit.controlflow.break_loop import BreakLoopOp, BreakLoopPlaceholder
        if self._control_flow_scopes:
            operation = BreakLoopPlaceholder()
            resources = operation.placeholder_resources()
            return self.append(operation, resources.qubits, resources.clbits)
        return self.append(BreakLoopOp(self.num_qubits, self.num_clbits), self.qubits, self.clbits)

    def continue_loop(self) -> InstructionSet:
        if False:
            i = 10
            return i + 15
        'Apply :class:`~qiskit.circuit.ContinueLoopOp`.\n\n        .. warning::\n\n            If you are using the context-manager "builder" forms of :meth:`.if_test`,\n            :meth:`.for_loop` or :meth:`.while_loop`, you can only call this method if you are\n            within a loop context, because otherwise the "resource width" of the operation cannot be\n            determined.  This would quickly lead to invalid circuits, and so if you are trying to\n            construct a reusable loop body (without the context managers), you must also use the\n            non-context-manager form of :meth:`.if_test` and :meth:`.if_else`.  Take care that the\n            :class:`~qiskit.circuit.ContinueLoopOp` instruction must span all the resources of its\n            containing loop, not just the immediate scope.\n\n        Returns:\n            A handle to the instruction created.\n\n        Raises:\n            CircuitError: if this method was called within a builder context, but not contained\n                within a loop.\n        '
        from qiskit.circuit.controlflow.continue_loop import ContinueLoopOp, ContinueLoopPlaceholder
        if self._control_flow_scopes:
            operation = ContinueLoopPlaceholder()
            resources = operation.placeholder_resources()
            return self.append(operation, resources.qubits, resources.clbits)
        return self.append(ContinueLoopOp(self.num_qubits, self.num_clbits), self.qubits, self.clbits)

    def add_calibration(self, gate: Union[Gate, str], qubits: Sequence[int], schedule, params: Sequence[ParameterValueType] | None=None) -> None:
        if False:
            while True:
                i = 10
        'Register a low-level, custom pulse definition for the given gate.\n\n        Args:\n            gate (Union[Gate, str]): Gate information.\n            qubits (Union[int, Tuple[int]]): List of qubits to be measured.\n            schedule (Schedule): Schedule information.\n            params (Optional[List[Union[float, Parameter]]]): A list of parameters.\n\n        Raises:\n            Exception: if the gate is of type string and params is None.\n        '

        def _format(operand):
            if False:
                print('Hello World!')
            try:
                evaluated = complex(operand)
                if np.isreal(evaluated):
                    evaluated = float(evaluated.real)
                    if evaluated.is_integer():
                        evaluated = int(evaluated)
                return evaluated
            except TypeError:
                return operand
        if isinstance(gate, Gate):
            params = gate.params
            gate = gate.name
        if params is not None:
            params = tuple(map(_format, params))
        else:
            params = ()
        self._calibrations[gate][tuple(qubits), params] = schedule

    def qubit_duration(self, *qubits: Union[Qubit, int]) -> float:
        if False:
            return 10
        'Return the duration between the start and stop time of the first and last instructions,\n        excluding delays, over the supplied qubits. Its time unit is ``self.unit``.\n\n        Args:\n            *qubits: Qubits within ``self`` to include.\n\n        Returns:\n            Return the duration between the first start and last stop time of non-delay instructions\n        '
        return self.qubit_stop_time(*qubits) - self.qubit_start_time(*qubits)

    def qubit_start_time(self, *qubits: Union[Qubit, int]) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Return the start time of the first instruction, excluding delays,\n        over the supplied qubits. Its time unit is ``self.unit``.\n\n        Return 0 if there are no instructions over qubits\n\n        Args:\n            *qubits: Qubits within ``self`` to include. Integers are allowed for qubits, indicating\n            indices of ``self.qubits``.\n\n        Returns:\n            Return the start time of the first instruction, excluding delays, over the qubits\n\n        Raises:\n            CircuitError: if ``self`` is a not-yet scheduled circuit.\n        '
        if self.duration is None:
            for instruction in self._data:
                if not isinstance(instruction.operation, Delay):
                    raise CircuitError('qubit_start_time undefined. Circuit must be scheduled first.')
            return 0
        qubits = [self.qubits[q] if isinstance(q, int) else q for q in qubits]
        starts = {q: 0 for q in qubits}
        dones = {q: False for q in qubits}
        for instruction in self._data:
            for q in qubits:
                if q in instruction.qubits:
                    if isinstance(instruction.operation, Delay):
                        if not dones[q]:
                            starts[q] += instruction.operation.duration
                    else:
                        dones[q] = True
            if len(qubits) == len([done for done in dones.values() if done]):
                return min((start for start in starts.values()))
        return 0

    def qubit_stop_time(self, *qubits: Union[Qubit, int]) -> float:
        if False:
            return 10
        'Return the stop time of the last instruction, excluding delays, over the supplied qubits.\n        Its time unit is ``self.unit``.\n\n        Return 0 if there are no instructions over qubits\n\n        Args:\n            *qubits: Qubits within ``self`` to include. Integers are allowed for qubits, indicating\n            indices of ``self.qubits``.\n\n        Returns:\n            Return the stop time of the last instruction, excluding delays, over the qubits\n\n        Raises:\n            CircuitError: if ``self`` is a not-yet scheduled circuit.\n        '
        if self.duration is None:
            for instruction in self._data:
                if not isinstance(instruction.operation, Delay):
                    raise CircuitError('qubit_stop_time undefined. Circuit must be scheduled first.')
            return 0
        qubits = [self.qubits[q] if isinstance(q, int) else q for q in qubits]
        stops = {q: self.duration for q in qubits}
        dones = {q: False for q in qubits}
        for instruction in reversed(self._data):
            for q in qubits:
                if q in instruction.qubits:
                    if isinstance(instruction.operation, Delay):
                        if not dones[q]:
                            stops[q] -= instruction.operation.duration
                    else:
                        dones[q] = True
            if len(qubits) == len([done for done in dones.values() if done]):
                return max((stop for stop in stops.values()))
        return 0
QuantumCircuit.isometry = QuantumCircuit.iso

class _ParameterBindsDict:
    __slots__ = ('mapping', 'allowed_keys')

    def __init__(self, mapping, allowed_keys):
        if False:
            while True:
                i = 10
        self.mapping = mapping
        self.allowed_keys = allowed_keys

    def items(self):
        if False:
            print('Hello World!')
        "Iterator through all the keys in the mapping that we care about.  Wrapping the main\n        mapping allows us to avoid reconstructing a new 'dict', but just use the given 'mapping'\n        without any copy / reconstruction."
        for (parameter, value) in self.mapping.items():
            if parameter in self.allowed_keys:
                yield (parameter, value)

class _ParameterBindsSequence:
    __slots__ = ('parameters', 'values', 'mapping_cache')

    def __init__(self, parameters, values):
        if False:
            while True:
                i = 10
        self.parameters = parameters
        self.values = values
        self.mapping_cache = None

    def items(self):
        if False:
            i = 10
            return i + 15
        'Iterator through all the keys in the mapping that we care about.'
        return zip(self.parameters, self.values)

    @property
    def mapping(self):
        if False:
            while True:
                i = 10
        'Cached version of a mapping.  This is only generated on demand.'
        if self.mapping_cache is None:
            self.mapping_cache = dict(zip(self.parameters, self.values))
        return self.mapping_cache

def _bit_argument_conversion(specifier, bit_sequence, bit_set, type_) -> list[Bit]:
    if False:
        while True:
            i = 10
    'Get the list of bits referred to by the specifier ``specifier``.\n\n    Valid types for ``specifier`` are integers, bits of the correct type (as given in ``type_``), or\n    iterables of one of those two scalar types.  Integers are interpreted as indices into the\n    sequence ``bit_sequence``.  All allowed bits must be in ``bit_set`` (which should implement\n    fast lookup), which is assumed to contain the same bits as ``bit_sequence``.\n\n    Returns:\n        List[Bit]: a list of the specified bits from ``bits``.\n\n    Raises:\n        CircuitError: if an incorrect type or index is encountered, if the same bit is specified\n            more than once, or if the specifier is to a bit not in the ``bit_set``.\n    '
    if isinstance(specifier, type_):
        if specifier in bit_set:
            return [specifier]
        raise CircuitError(f"Bit '{specifier}' is not in the circuit.")
    if isinstance(specifier, (int, np.integer)):
        try:
            return [bit_sequence[specifier]]
        except IndexError as ex:
            raise CircuitError(f'Index {specifier} out of range for size {len(bit_sequence)}.') from ex
    if isinstance(specifier, slice):
        return bit_sequence[specifier]
    try:
        return [_bit_argument_conversion_scalar(index, bit_sequence, bit_set, type_) for index in specifier]
    except TypeError as ex:
        message = f"Incorrect bit type: expected '{type_.__name__}' but got '{type(specifier).__name__}'" if isinstance(specifier, Bit) else f"Invalid bit index: '{specifier}' of type '{type(specifier)}'"
        raise CircuitError(message) from ex

def _bit_argument_conversion_scalar(specifier, bit_sequence, bit_set, type_):
    if False:
        i = 10
        return i + 15
    if isinstance(specifier, type_):
        if specifier in bit_set:
            return specifier
        raise CircuitError(f"Bit '{specifier}' is not in the circuit.")
    if isinstance(specifier, (int, np.integer)):
        try:
            return bit_sequence[specifier]
        except IndexError as ex:
            raise CircuitError(f'Index {specifier} out of range for size {len(bit_sequence)}.') from ex
    message = f"Incorrect bit type: expected '{type_.__name__}' but got '{type(specifier).__name__}'" if isinstance(specifier, Bit) else f"Invalid bit index: '{specifier}' of type '{type(specifier)}'"
    raise CircuitError(message)