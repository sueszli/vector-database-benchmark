"""The n-local circuit class."""
from __future__ import annotations
import typing
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
import numpy
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit import Instruction, Parameter, ParameterVector, ParameterExpression
from qiskit.exceptions import QiskitError
from ..blueprintcircuit import BlueprintCircuit
if typing.TYPE_CHECKING:
    import qiskit

class NLocal(BlueprintCircuit):
    """The n-local circuit class.

    The structure of the n-local circuit are alternating rotation and entanglement layers.
    In both layers, parameterized circuit-blocks act on the circuit in a defined way.
    In the rotation layer, the blocks are applied stacked on top of each other, while in the
    entanglement layer according to the ``entanglement`` strategy.
    The circuit blocks can have arbitrary sizes (smaller equal to the number of qubits in the
    circuit). Each layer is repeated ``reps`` times, and by default a final rotation layer is
    appended.

    For instance, a rotation block on 2 qubits and an entanglement block on 4 qubits using
    ``'linear'`` entanglement yields the following circuit.

    .. parsed-literal::

        ┌──────┐ ░ ┌──────┐                      ░ ┌──────┐
        ┤0     ├─░─┤0     ├──────────────── ... ─░─┤0     ├
        │  Rot │ ░ │      │┌──────┐              ░ │  Rot │
        ┤1     ├─░─┤1     ├┤0     ├──────── ... ─░─┤1     ├
        ├──────┤ ░ │  Ent ││      │┌──────┐      ░ ├──────┤
        ┤0     ├─░─┤2     ├┤1     ├┤0     ├ ... ─░─┤0     ├
        │  Rot │ ░ │      ││  Ent ││      │      ░ │  Rot │
        ┤1     ├─░─┤3     ├┤2     ├┤1     ├ ... ─░─┤1     ├
        ├──────┤ ░ └──────┘│      ││  Ent │      ░ ├──────┤
        ┤0     ├─░─────────┤3     ├┤2     ├ ... ─░─┤0     ├
        │  Rot │ ░         └──────┘│      │      ░ │  Rot │
        ┤1     ├─░─────────────────┤3     ├ ... ─░─┤1     ├
        └──────┘ ░                 └──────┘      ░ └──────┘

        |                                 |
        +---------------------------------+
               repeated reps times

    If specified, barriers can be inserted in between every block.
    If an initial state object is provided, it is added in front of the NLocal.
    """

    def __init__(self, num_qubits: int | None=None, rotation_blocks: QuantumCircuit | list[QuantumCircuit] | qiskit.circuit.Instruction | list[qiskit.circuit.Instruction] | None=None, entanglement_blocks: QuantumCircuit | list[QuantumCircuit] | qiskit.circuit.Instruction | list[qiskit.circuit.Instruction] | None=None, entanglement: list[int] | list[list[int]] | None=None, reps: int=1, insert_barriers: bool=False, parameter_prefix: str='θ', overwrite_block_parameters: bool | list[list[Parameter]]=True, skip_final_rotation_layer: bool=False, skip_unentangled_qubits: bool=False, initial_state: QuantumCircuit | None=None, name: str | None='nlocal', flatten: bool | None=None) -> None:
        if False:
            print('Hello World!')
        "\n        Args:\n            num_qubits: The number of qubits of the circuit.\n            rotation_blocks: The blocks used in the rotation layers. If multiple are passed,\n                these will be applied one after another (like new sub-layers).\n            entanglement_blocks: The blocks used in the entanglement layers. If multiple are passed,\n                these will be applied one after another. To use different entanglements for\n                the sub-layers, see :meth:`get_entangler_map`.\n            entanglement: The indices specifying on which qubits the input blocks act. If ``None``, the\n                entanglement blocks are applied at the top of the circuit.\n            reps: Specifies how often the rotation blocks and entanglement blocks are repeated.\n            insert_barriers: If ``True``, barriers are inserted in between each layer. If ``False``,\n                no barriers are inserted.\n            parameter_prefix: The prefix used if default parameters are generated.\n            overwrite_block_parameters: If the parameters in the added blocks should be overwritten.\n                If ``False``, the parameters in the blocks are not changed.\n            skip_final_rotation_layer: Whether a final rotation layer is added to the circuit.\n            skip_unentangled_qubits: If ``True``, the rotation gates act only on qubits that\n                are entangled. If ``False``, the rotation gates act on all qubits.\n            initial_state: A :class:`.QuantumCircuit` object which can be used to describe an initial\n                state prepended to the NLocal circuit.\n            name: The name of the circuit.\n            flatten: Set this to ``True`` to output a flat circuit instead of nesting it inside multiple\n                layers of gate objects. By default currently the contents of\n                the output circuit will be wrapped in nested objects for\n                cleaner visualization. However, if you're using this circuit\n                for anything besides visualization its **strongly** recommended\n                to set this flag to ``True`` to avoid a large performance\n                overhead for parameter binding.\n\n        Raises:\n            ValueError: If ``reps`` parameter is less than or equal to 0.\n            TypeError: If ``reps`` parameter is not an int value.\n        "
        super().__init__(name=name)
        self._num_qubits: int | None = None
        self._insert_barriers = insert_barriers
        self._reps = reps
        self._entanglement_blocks: list[QuantumCircuit] = []
        self._rotation_blocks: list[QuantumCircuit] = []
        self._prepended_blocks: list[QuantumCircuit] = []
        self._prepended_entanglement: list[list[list[int]] | str] = []
        self._appended_blocks: list[QuantumCircuit] = []
        self._appended_entanglement: list[list[list[int]] | str] = []
        self._entanglement = None
        self._entangler_maps = None
        self._ordered_parameters: ParameterVector | list[Parameter] = ParameterVector(name=parameter_prefix)
        self._overwrite_block_parameters = overwrite_block_parameters
        self._skip_final_rotation_layer = skip_final_rotation_layer
        self._skip_unentangled_qubits = skip_unentangled_qubits
        self._initial_state: QuantumCircuit | None = None
        self._initial_state_circuit: QuantumCircuit | None = None
        self._bounds: list[tuple[float | None, float | None]] | None = None
        self._flatten = flatten
        if int(reps) != reps:
            raise TypeError('The value of reps should be int')
        if reps < 0:
            raise ValueError('The value of reps should be larger than or equal to 0')
        if num_qubits is not None:
            self.num_qubits = num_qubits
        if entanglement_blocks is not None:
            self.entanglement_blocks = entanglement_blocks
        if rotation_blocks is not None:
            self.rotation_blocks = rotation_blocks
        if entanglement is not None:
            self.entanglement = entanglement
        if initial_state is not None:
            self.initial_state = initial_state

    @property
    def num_qubits(self) -> int:
        if False:
            return 10
        'Returns the number of qubits in this circuit.\n\n        Returns:\n            The number of qubits.\n        '
        return self._num_qubits if self._num_qubits is not None else 0

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        if False:
            i = 10
            return i + 15
        'Set the number of qubits for the n-local circuit.\n\n        Args:\n            The new number of qubits.\n        '
        if self._num_qubits != num_qubits:
            self._invalidate()
            self._num_qubits = num_qubits
            self.qregs = [QuantumRegister(num_qubits, name='q')]

    @property
    def flatten(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Returns whether the circuit is wrapped in nested gates/instructions or flattened.'
        return bool(self._flatten)

    @flatten.setter
    def flatten(self, flatten: bool) -> None:
        if False:
            print('Hello World!')
        self._invalidate()
        self._flatten = flatten

    def _convert_to_block(self, layer: typing.Any) -> QuantumCircuit:
        if False:
            i = 10
            return i + 15
        'Try to convert ``layer`` to a QuantumCircuit.\n\n        Args:\n            layer: The object to be converted to an NLocal block / Instruction.\n\n        Returns:\n            The layer converted to a circuit.\n\n        Raises:\n            TypeError: If the input cannot be converted to a circuit.\n        '
        if isinstance(layer, QuantumCircuit):
            return layer
        if isinstance(layer, Instruction):
            circuit = QuantumCircuit(layer.num_qubits)
            circuit.append(layer, list(range(layer.num_qubits)))
            return circuit
        try:
            circuit = QuantumCircuit(layer.num_qubits)
            circuit.append(layer.to_instruction(), list(range(layer.num_qubits)))
            return circuit
        except AttributeError:
            pass
        raise TypeError(f'Adding a {type(layer)} to an NLocal is not supported.')

    @property
    def rotation_blocks(self) -> list[QuantumCircuit]:
        if False:
            while True:
                i = 10
        'The blocks in the rotation layers.\n\n        Returns:\n            The blocks in the rotation layers.\n        '
        return self._rotation_blocks

    @rotation_blocks.setter
    def rotation_blocks(self, blocks: QuantumCircuit | list[QuantumCircuit] | Instruction | list[Instruction]) -> None:
        if False:
            while True:
                i = 10
        'Set the blocks in the rotation layers.\n\n        Args:\n            blocks: The new blocks for the rotation layers.\n        '
        if not isinstance(blocks, (list, numpy.ndarray)):
            blocks = [blocks]
        self._invalidate()
        self._rotation_blocks = [self._convert_to_block(block) for block in blocks]

    @property
    def entanglement_blocks(self) -> list[QuantumCircuit]:
        if False:
            print('Hello World!')
        'The blocks in the entanglement layers.\n\n        Returns:\n            The blocks in the entanglement layers.\n        '
        return self._entanglement_blocks

    @entanglement_blocks.setter
    def entanglement_blocks(self, blocks: QuantumCircuit | list[QuantumCircuit] | Instruction | list[Instruction]) -> None:
        if False:
            i = 10
            return i + 15
        'Set the blocks in the entanglement layers.\n\n        Args:\n            blocks: The new blocks for the entanglement layers.\n        '
        if not isinstance(blocks, (list, numpy.ndarray)):
            blocks = [blocks]
        self._invalidate()
        self._entanglement_blocks = [self._convert_to_block(block) for block in blocks]

    @property
    def entanglement(self) -> str | list[str] | list[list[str]] | list[int] | list[list[int]] | list[list[list[int]]] | list[list[list[list[int]]]] | Callable[[int], str] | Callable[[int], list[list[int]]]:
        if False:
            while True:
                i = 10
        'Get the entanglement strategy.\n\n        Returns:\n            The entanglement strategy, see :meth:`get_entangler_map` for more detail on how the\n            format is interpreted.\n        '
        return self._entanglement

    @entanglement.setter
    def entanglement(self, entanglement: str | list[str] | list[list[str]] | list[int] | list[list[int]] | list[list[list[int]]] | list[list[list[list[int]]]] | Callable[[int], str] | Callable[[int], list[list[int]]] | None) -> None:
        if False:
            while True:
                i = 10
        'Set the entanglement strategy.\n\n        Args:\n            entanglement: The entanglement strategy. See :meth:`get_entangler_map` for more detail\n                on the supported formats.\n        '
        self._invalidate()
        self._entanglement = entanglement

    @property
    def num_layers(self) -> int:
        if False:
            return 10
        'Return the number of layers in the n-local circuit.\n\n        Returns:\n            The number of layers in the circuit.\n        '
        return 2 * self._reps + int(not self._skip_final_rotation_layer)

    def _check_configuration(self, raise_on_failure: bool=True) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Check if the configuration of the NLocal class is valid.\n\n        Args:\n            raise_on_failure: Whether to raise on failure.\n\n        Returns:\n            True, if the configuration is valid and the circuit can be constructed. Otherwise\n            an ValueError is raised.\n\n        Raises:\n            ValueError: If the blocks are not set.\n            ValueError: If the number of repetitions is not set.\n            ValueError: If the qubit indices are not set.\n            ValueError: If the number of qubit indices does not match the number of blocks.\n            ValueError: If an index in the repetitions list exceeds the number of blocks.\n            ValueError: If the number of repetitions does not match the number of block-wise\n                parameters.\n            ValueError: If a specified qubit index is larger than the (manually set) number of\n                qubits.\n        '
        valid = True
        if self.num_qubits is None:
            valid = False
            if raise_on_failure:
                raise ValueError('No number of qubits specified.')
        if self.entanglement_blocks is None and self.rotation_blocks is None:
            valid = False
            if raise_on_failure:
                raise ValueError('The blocks are not set.')
        return valid

    @property
    def ordered_parameters(self) -> list[Parameter]:
        if False:
            i = 10
            return i + 15
        'The parameters used in the underlying circuit.\n\n        This includes float values and duplicates.\n\n        Examples:\n\n            >>> # prepare circuit ...\n            >>> print(nlocal)\n                 ┌───────┐┌──────────┐┌──────────┐┌──────────┐\n            q_0: ┤ Ry(1) ├┤ Ry(θ[1]) ├┤ Ry(θ[1]) ├┤ Ry(θ[3]) ├\n                 └───────┘└──────────┘└──────────┘└──────────┘\n            >>> nlocal.parameters\n            {Parameter(θ[1]), Parameter(θ[3])}\n            >>> nlocal.ordered_parameters\n            [1, Parameter(θ[1]), Parameter(θ[1]), Parameter(θ[3])]\n\n        Returns:\n            The parameters objects used in the circuit.\n        '
        if isinstance(self._ordered_parameters, ParameterVector):
            self._ordered_parameters.resize(self.num_parameters_settable)
            return list(self._ordered_parameters)
        return self._ordered_parameters

    @ordered_parameters.setter
    def ordered_parameters(self, parameters: ParameterVector | list[Parameter]) -> None:
        if False:
            i = 10
            return i + 15
        'Set the parameters used in the underlying circuit.\n\n        Args:\n            The parameters to be used in the underlying circuit.\n\n        Raises:\n            ValueError: If the length of ordered parameters does not match the number of\n                parameters in the circuit and they are not a ``ParameterVector`` (which could\n                be resized to fit the number of parameters).\n        '
        if not isinstance(parameters, ParameterVector) and len(parameters) != self.num_parameters_settable:
            raise ValueError('The length of ordered parameters must be equal to the number of settable parameters in the circuit ({}), but is {}'.format(self.num_parameters_settable, len(parameters)))
        self._ordered_parameters = parameters
        self._invalidate()

    @property
    def insert_barriers(self) -> bool:
        if False:
            i = 10
            return i + 15
        'If barriers are inserted in between the layers or not.\n\n        Returns:\n            ``True``, if barriers are inserted in between the layers, ``False`` if not.\n        '
        return self._insert_barriers

    @insert_barriers.setter
    def insert_barriers(self, insert_barriers: bool) -> None:
        if False:
            while True:
                i = 10
        'Specify whether barriers should be inserted in between the layers or not.\n\n        Args:\n            insert_barriers: If True, barriers are inserted, if False not.\n        '
        if insert_barriers is not self._insert_barriers:
            self._invalidate()
            self._insert_barriers = insert_barriers

    def get_unentangled_qubits(self) -> set[int]:
        if False:
            for i in range(10):
                print('nop')
        'Get the indices of unentangled qubits in a set.\n\n        Returns:\n            The unentangled qubits.\n        '
        entangled_qubits = set()
        for i in range(self._reps):
            for (j, block) in enumerate(self.entanglement_blocks):
                entangler_map = self.get_entangler_map(i, j, block.num_qubits)
                entangled_qubits.update([idx for indices in entangler_map for idx in indices])
        unentangled_qubits = set(range(self.num_qubits)) - entangled_qubits
        return unentangled_qubits

    @property
    def num_parameters_settable(self) -> int:
        if False:
            print('Hello World!')
        'The number of total parameters that can be set to distinct values.\n\n        This does not change when the parameters are bound or exchanged for same parameters,\n        and therefore is different from ``num_parameters`` which counts the number of unique\n        :class:`~qiskit.circuit.Parameter` objects currently in the circuit.\n\n        Returns:\n            The number of parameters originally available in the circuit.\n\n        Note:\n            This quantity does not require the circuit to be built yet.\n        '
        num = 0
        for i in range(self._reps):
            for (j, block) in enumerate(self.entanglement_blocks):
                entangler_map = self.get_entangler_map(i, j, block.num_qubits)
                num += len(entangler_map) * len(get_parameters(block))
        if self._skip_unentangled_qubits:
            unentangled_qubits = self.get_unentangled_qubits()
        num_rot = 0
        for block in self.rotation_blocks:
            block_indices = [list(range(j * block.num_qubits, (j + 1) * block.num_qubits)) for j in range(self.num_qubits // block.num_qubits)]
            if self._skip_unentangled_qubits:
                block_indices = [indices for indices in block_indices if set(indices).isdisjoint(unentangled_qubits)]
            num_rot += len(block_indices) * len(get_parameters(block))
        num += num_rot * (self._reps + int(not self._skip_final_rotation_layer))
        return num

    @property
    def reps(self) -> int:
        if False:
            while True:
                i = 10
        'The number of times rotation and entanglement block are repeated.\n\n        Returns:\n            The number of repetitions.\n        '
        return self._reps

    @reps.setter
    def reps(self, repetitions: int) -> None:
        if False:
            return 10
        'Set the repetitions.\n\n        If the repetitions are `0`, only one rotation layer with no entanglement\n        layers is applied (unless ``self.skip_final_rotation_layer`` is set to ``True``).\n\n        Args:\n            repetitions: The new repetitions.\n\n        Raises:\n            ValueError: If reps setter has parameter repetitions < 0.\n        '
        if repetitions < 0:
            raise ValueError('The repetitions should be larger than or equal to 0')
        if repetitions != self._reps:
            self._invalidate()
            self._reps = repetitions

    def print_settings(self) -> str:
        if False:
            return 10
        'Returns information about the setting.\n\n        Returns:\n            The class name and the attributes/parameters of the instance as ``str``.\n        '
        ret = f'NLocal: {self.__class__.__name__}\n'
        params = ''
        for (key, value) in self.__dict__.items():
            if key[0] == '_':
                params += f'-- {key[1:]}: {value}\n'
        ret += f'{params}'
        return ret

    @property
    def preferred_init_points(self) -> list[float] | None:
        if False:
            print('Hello World!')
        'The initial points for the parameters. Can be stored as initial guess in optimization.\n\n        Returns:\n            The initial values for the parameters, or None, if none have been set.\n        '
        return None

    def get_entangler_map(self, rep_num: int, block_num: int, num_block_qubits: int) -> Sequence[Sequence[int]]:
        if False:
            for i in range(10):
                print('nop')
        "Get the entangler map for in the repetition ``rep_num`` and the block ``block_num``.\n\n        The entangler map for the current block is derived from the value of ``self.entanglement``.\n        Below the different cases are listed, where ``i`` and ``j`` denote the repetition number\n        and the block number, respectively, and ``n`` the number of qubits in the block.\n\n        =================================== ========================================================\n        entanglement type                   entangler map\n        =================================== ========================================================\n        ``None``                            ``[[0, ..., n - 1]]``\n        ``str`` (e.g ``'full'``)            the specified connectivity on ``n`` qubits\n        ``List[int]``                       [``entanglement``]\n        ``List[List[int]]``                 ``entanglement``\n        ``List[List[List[int]]]``           ``entanglement[i]``\n        ``List[List[List[List[int]]]]``     ``entanglement[i][j]``\n        ``List[str]``                       the connectivity specified in ``entanglement[i]``\n        ``List[List[str]]``                 the connectivity specified in ``entanglement[i][j]``\n        ``Callable[int, str]``              same as ``List[str]``\n        ``Callable[int, List[List[int]]]``  same as ``List[List[List[int]]]``\n        =================================== ========================================================\n\n\n        Note that all indices are to be taken modulo the length of the array they act on, i.e.\n        no out-of-bounds index error will be raised but we re-iterate from the beginning of the\n        list.\n\n        Args:\n            rep_num: The current repetition we are in.\n            block_num: The block number within the entanglement layers.\n            num_block_qubits: The number of qubits in the block.\n\n        Returns:\n            The entangler map for the current block in the current repetition.\n\n        Raises:\n            ValueError: If the value of ``entanglement`` could not be cast to a corresponding\n                entangler map.\n        "
        (i, j, n) = (rep_num, block_num, num_block_qubits)
        entanglement = self._entanglement
        if entanglement is None:
            return [list(range(n))]
        if callable(entanglement):
            entanglement = entanglement(i)
        if isinstance(entanglement, str):
            return get_entangler_map(n, self.num_qubits, entanglement, offset=i)
        if not isinstance(entanglement, (tuple, list)):
            raise ValueError(f'Invalid value of entanglement: {entanglement}')
        num_i = len(entanglement)
        if all((isinstance(en, str) for en in entanglement)):
            return get_entangler_map(n, self.num_qubits, entanglement[i % num_i], offset=i)
        if all((isinstance(en, (int, numpy.integer)) for en in entanglement)):
            return [[int(en) for en in entanglement]]
        if not all((isinstance(en, (tuple, list)) for en in entanglement)):
            raise ValueError(f'Invalid value of entanglement: {entanglement}')
        num_j = len(entanglement[i % num_i])
        if all((isinstance(e2, str) for en in entanglement for e2 in en)):
            return get_entangler_map(n, self.num_qubits, entanglement[i % num_i][j % num_j], offset=i)
        if all((isinstance(e2, (int, numpy.int32, numpy.int64)) for en in entanglement for e2 in en)):
            for (ind, en) in enumerate(entanglement):
                entanglement[ind] = tuple(map(int, en))
            return entanglement
        if not all((isinstance(e2, (tuple, list)) for en in entanglement for e2 in en)):
            raise ValueError(f'Invalid value of entanglement: {entanglement}')
        if all((isinstance(e3, (int, numpy.int32, numpy.int64)) for en in entanglement for e2 in en for e3 in e2)):
            for en in entanglement:
                for (ind, e2) in enumerate(en):
                    en[ind] = tuple(map(int, e2))
            return entanglement[i % num_i]
        if not all((isinstance(e3, (tuple, list)) for en in entanglement for e2 in en for e3 in e2)):
            raise ValueError(f'Invalid value of entanglement: {entanglement}')
        if all((isinstance(e4, (int, numpy.int32, numpy.int64)) for en in entanglement for e2 in en for e3 in e2 for e4 in e3)):
            for en in entanglement:
                for e2 in en:
                    for (ind, e3) in enumerate(e2):
                        e2[ind] = tuple(map(int, e3))
            return entanglement[i % num_i][j % num_j]
        raise ValueError(f'Invalid value of entanglement: {entanglement}')

    @property
    def initial_state(self) -> QuantumCircuit:
        if False:
            i = 10
            return i + 15
        'Return the initial state that is added in front of the n-local circuit.\n\n        Returns:\n            The initial state.\n        '
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: QuantumCircuit) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the initial state.\n\n        Args:\n            initial_state: The new initial state.\n\n        Raises:\n            ValueError: If the number of qubits has been set before and the initial state\n                does not match the number of qubits.\n        '
        self._initial_state = initial_state
        self._invalidate()

    @property
    def parameter_bounds(self) -> list[tuple[float, float]] | None:
        if False:
            for i in range(10):
                print('nop')
        'The parameter bounds for the unbound parameters in the circuit.\n\n        Returns:\n            A list of pairs indicating the bounds, as (lower, upper). None indicates an unbounded\n            parameter in the corresponding direction. If ``None`` is returned, problem is fully\n            unbounded.\n        '
        if not self._is_built:
            self._build()
        return self._bounds

    @parameter_bounds.setter
    def parameter_bounds(self, bounds: list[tuple[float, float]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the parameter bounds.\n\n        Args:\n            bounds: The new parameter bounds.\n        '
        self._bounds = bounds

    def add_layer(self, other: QuantumCircuit | qiskit.circuit.Instruction, entanglement: list[int] | str | list[list[int]] | None=None, front: bool=False) -> 'NLocal':
        if False:
            print('Hello World!')
        'Append another layer to the NLocal.\n\n        Args:\n            other: The layer to compose, can be another NLocal, an Instruction or Gate,\n                or a QuantumCircuit.\n            entanglement: The entanglement or qubit indices.\n            front: If True, ``other`` is appended to the front, else to the back.\n\n        Returns:\n            self, such that chained composes are possible.\n\n        Raises:\n            TypeError: If `other` is not compatible, i.e. is no Instruction and does not have a\n                `to_instruction` method.\n        '
        block = self._convert_to_block(other)
        if entanglement is None:
            entanglement = [list(range(block.num_qubits))]
        elif isinstance(entanglement, list) and (not isinstance(entanglement[0], list)):
            entanglement = [entanglement]
        if front:
            self._prepended_blocks += [block]
            self._prepended_entanglement += [entanglement]
        else:
            self._appended_blocks += [block]
            self._appended_entanglement += [entanglement]
        if isinstance(entanglement, list):
            num_qubits = 1 + max((max(indices) for indices in entanglement))
            if num_qubits > self.num_qubits:
                self._invalidate()
                self.num_qubits = num_qubits
        if front is False and self._is_built:
            if self._insert_barriers and len(self.data) > 0:
                self.barrier()
            if isinstance(entanglement, str):
                entangler_map: Sequence[Sequence[int]] = get_entangler_map(block.num_qubits, self.num_qubits, entanglement)
            else:
                entangler_map = entanglement
            layer = QuantumCircuit(self.num_qubits)
            for i in entangler_map:
                params = self.ordered_parameters[-len(get_parameters(block)):]
                parameterized_block = self._parameterize_block(block, params=params)
                layer.compose(parameterized_block, i, inplace=True)
            self.compose(layer, inplace=True)
        else:
            self._invalidate()
        return self

    def assign_parameters(self, parameters: Mapping[Parameter, ParameterExpression | float] | Sequence[ParameterExpression | float], inplace: bool=False, **kwargs) -> QuantumCircuit | None:
        if False:
            print('Hello World!')
        'Assign parameters to the n-local circuit.\n\n        This method also supports passing a list instead of a dictionary. If a list\n        is passed, the list must have the same length as the number of unbound parameters in\n        the circuit. The parameters are assigned in the order of the parameters in\n        :meth:`ordered_parameters`.\n\n        Returns:\n            A copy of the NLocal circuit with the specified parameters.\n\n        Raises:\n            AttributeError: If the parameters are given as list and do not match the number\n                of parameters.\n        '
        if parameters is None or len(parameters) == 0:
            return self
        if not self._is_built:
            self._build()
        return super().assign_parameters(parameters, inplace=inplace, **kwargs)

    def _parameterize_block(self, block, param_iter=None, rep_num=None, block_num=None, indices=None, params=None):
        if False:
            i = 10
            return i + 15
        'Convert ``block`` to a circuit of correct width and parameterized using the iterator.'
        if self._overwrite_block_parameters:
            if params is None:
                params = self._parameter_generator(rep_num, block_num, indices)
            if params is None:
                params = [next(param_iter) for _ in range(len(get_parameters(block)))]
            update = dict(zip(block.parameters, params))
            return block.assign_parameters(update)
        return block.copy()

    def _build_rotation_layer(self, circuit, param_iter, i):
        if False:
            i = 10
            return i + 15
        'Build a rotation layer.'
        if self._skip_unentangled_qubits:
            unentangled_qubits = self.get_unentangled_qubits()
        for (j, block) in enumerate(self.rotation_blocks):
            layer = QuantumCircuit(*self.qregs)
            block_indices = [list(range(k * block.num_qubits, (k + 1) * block.num_qubits)) for k in range(self.num_qubits // block.num_qubits)]
            if self._skip_unentangled_qubits:
                block_indices = [indices for indices in block_indices if set(indices).isdisjoint(unentangled_qubits)]
            for indices in block_indices:
                parameterized_block = self._parameterize_block(block, param_iter, i, j, indices)
                layer.compose(parameterized_block, indices, inplace=True)
            circuit.compose(layer, inplace=True)

    def _build_entanglement_layer(self, circuit, param_iter, i):
        if False:
            return 10
        'Build an entanglement layer.'
        for (j, block) in enumerate(self.entanglement_blocks):
            layer = QuantumCircuit(*self.qregs)
            entangler_map = self.get_entangler_map(i, j, block.num_qubits)
            for indices in entangler_map:
                parameterized_block = self._parameterize_block(block, param_iter, i, j, indices)
                layer.compose(parameterized_block, indices, inplace=True)
            circuit.compose(layer, inplace=True)

    def _build_additional_layers(self, circuit, which):
        if False:
            while True:
                i = 10
        if which == 'appended':
            blocks = self._appended_blocks
            entanglements = self._appended_entanglement
        elif which == 'prepended':
            blocks = reversed(self._prepended_blocks)
            entanglements = reversed(self._prepended_entanglement)
        else:
            raise ValueError('`which` must be either `appended` or `prepended`.')
        for (block, ent) in zip(blocks, entanglements):
            layer = QuantumCircuit(*self.qregs)
            if isinstance(ent, str):
                ent = get_entangler_map(block.num_qubits, self.num_qubits, ent)
            for indices in ent:
                layer.compose(block, indices, inplace=True)
            circuit.compose(layer, inplace=True)

    def _build(self) -> None:
        if False:
            print('Hello World!')
        'If not already built, build the circuit.'
        if self._is_built:
            return
        super()._build()
        if self.num_qubits == 0:
            return
        if not self._flatten:
            circuit = QuantumCircuit(*self.qregs, name=self.name)
        else:
            circuit = self
        if self.initial_state:
            circuit.compose(self.initial_state.copy(), inplace=True)
        param_iter = iter(self.ordered_parameters)
        self._build_additional_layers(circuit, 'prepended')
        for i in range(self.reps):
            if self._insert_barriers and (i > 0 or len(self._prepended_blocks) > 0):
                circuit.barrier()
            self._build_rotation_layer(circuit, param_iter, i)
            if self._insert_barriers and len(self._rotation_blocks) > 0:
                circuit.barrier()
            self._build_entanglement_layer(circuit, param_iter, i)
        if not self._skip_final_rotation_layer:
            if self.insert_barriers and self.reps > 0:
                circuit.barrier()
            self._build_rotation_layer(circuit, param_iter, self.reps)
        self._build_additional_layers(circuit, 'appended')
        if isinstance(circuit.global_phase, ParameterExpression):
            try:
                circuit.global_phase = float(circuit.global_phase)
            except TypeError:
                pass
        if not self._flatten:
            try:
                block = circuit.to_gate()
            except QiskitError:
                block = circuit.to_instruction()
            self.append(block, self.qubits)

    def _parameter_generator(self, rep: int, block: int, indices: list[int]) -> Parameter | None:
        if False:
            for i in range(10):
                print('nop')
        'If certain blocks should use certain parameters this method can be overridden.'
        return None

def get_parameters(block: QuantumCircuit | Instruction) -> list[Parameter]:
    if False:
        i = 10
        return i + 15
    'Return the list of Parameters objects inside a circuit or instruction.\n\n    This is required since, in a standard gate the parameters are not necessarily Parameter\n    objects (e.g. U3Gate(0.1, 0.2, 0.3).params == [0.1, 0.2, 0.3]) and instructions and\n    circuits do not have the same interface for parameters.\n    '
    if isinstance(block, QuantumCircuit):
        return list(block.parameters)
    else:
        return [p for p in block.params if isinstance(p, ParameterExpression)]

def get_entangler_map(num_block_qubits: int, num_circuit_qubits: int, entanglement: str, offset: int=0) -> Sequence[tuple[int, ...]]:
    if False:
        print('Hello World!')
    'Get an entangler map for an arbitrary number of qubits.\n\n    Args:\n        num_block_qubits: The number of qubits of the entangling block.\n        num_circuit_qubits: The number of qubits of the circuit.\n        entanglement: The entanglement strategy.\n        offset: The block offset, can be used if the entanglements differ per block.\n            See mode ``sca`` for instance.\n\n    Returns:\n        The entangler map using mode ``entanglement`` to scatter a block of ``num_block_qubits``\n        qubits on ``num_circuit_qubits`` qubits.\n\n    Raises:\n        ValueError: If the entanglement mode ist not supported.\n    '
    (n, m) = (num_circuit_qubits, num_block_qubits)
    if m > n:
        raise ValueError('The number of block qubits must be smaller or equal to the number of qubits in the circuit.')
    if entanglement == 'pairwise' and num_block_qubits > 2:
        raise ValueError('Pairwise entanglement is not defined for blocks with more than 2 qubits.')
    if entanglement == 'full':
        return list(combinations(list(range(n)), m))
    elif entanglement == 'reverse_linear':
        reverse = [tuple(range(n - i - m, n - i)) for i in range(n - m + 1)]
        return reverse
    elif entanglement in ['linear', 'circular', 'sca', 'pairwise']:
        linear = [tuple(range(i, i + m)) for i in range(n - m + 1)]
        if entanglement == 'linear' or m == 1:
            return linear
        if entanglement == 'pairwise':
            return linear[::2] + linear[1::2]
        if n > m:
            circular = [tuple(range(n - m + 1, n)) + (0,)] + linear
        else:
            circular = linear
        if entanglement == 'circular':
            return circular
        shifted = circular[-offset:] + circular[:-offset]
        if offset % 2 == 1:
            sca = [ind[::-1] for ind in shifted]
        else:
            sca = shifted
        return sca
    else:
        raise ValueError(f'Unsupported entanglement type: {entanglement}')