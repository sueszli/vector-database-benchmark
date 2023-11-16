"""The two-local gate circuit."""
from __future__ import annotations
import typing
from collections.abc import Callable, Sequence
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, Parameter
from .n_local import NLocal
from ..standard_gates import IGate, XGate, YGate, ZGate, RXGate, RYGate, RZGate, HGate, SGate, SdgGate, TGate, TdgGate, RXXGate, RYYGate, RZXGate, RZZGate, SwapGate, CXGate, CYGate, CZGate, CRXGate, CRYGate, CRZGate, CHGate
if typing.TYPE_CHECKING:
    import qiskit

class TwoLocal(NLocal):
    """The two-local circuit.

    The two-local circuit is a parameterized circuit consisting of alternating rotation layers and
    entanglement layers. The rotation layers are single qubit gates applied on all qubits.
    The entanglement layer uses two-qubit gates to entangle the qubits according to a strategy set
    using ``entanglement``. Both the rotation and entanglement gates can be specified as
    string (e.g. ``'ry'`` or ``'cx'``), as gate-type (e.g. ``RYGate`` or ``CXGate``) or
    as QuantumCircuit (e.g. a 1-qubit circuit or 2-qubit circuit).

    A set of default entanglement strategies is provided:

    * ``'full'`` entanglement is each qubit is entangled with all the others.
    * ``'linear'`` entanglement is qubit :math:`i` entangled with qubit :math:`i + 1`,
      for all :math:`i \\in \\{0, 1, ... , n - 2\\}`, where :math:`n` is the total number of qubits.
    * ``'reverse_linear'`` entanglement is qubit :math:`i` entangled with qubit :math:`i + 1`,
      for all :math:`i \\in \\{n-2, n-3, ... , 1, 0\\}`, where :math:`n` is the total number of qubits.
      Note that if ``entanglement_blocks = 'cx'`` then this option provides the same unitary as
      ``'full'`` with fewer entangling gates.
    * ``'pairwise'`` entanglement is one layer where qubit :math:`i` is entangled with qubit
      :math:`i + 1`, for all even values of :math:`i`, and then a second layer where qubit :math:`i`
      is entangled with qubit :math:`i + 1`, for all odd values of :math:`i`.
    * ``'circular'`` entanglement is linear entanglement but with an additional entanglement of the
      first and last qubit before the linear part.
    * ``'sca'`` (shifted-circular-alternating) entanglement is a generalized and modified version
      of the proposed circuit 14 in `Sim et al. <https://arxiv.org/abs/1905.10876>`__.
      It consists of circular entanglement where the 'long' entanglement connecting the first with
      the last qubit is shifted by one each block.  Furthermore the role of control and target
      qubits are swapped every block (therefore alternating).

    The entanglement can further be specified using an entangler map, which is a list of index
    pairs, such as

    >>> entangler_map = [(0, 1), (1, 2), (2, 0)]

    If different entanglements per block should be used, provide a list of entangler maps.
    See the examples below on how this can be used.

    >>> entanglement = [entangler_map_layer_1, entangler_map_layer_2, ... ]

    Barriers can be inserted in between the different layers for better visualization using the
    ``insert_barriers`` attribute.

    For each parameterized gate a new parameter is generated using a
    :class:`~qiskit.circuit.library.ParameterVector`. The name of these parameters can be chosen
    using the ``parameter_prefix``.

    Examples:

        >>> two = TwoLocal(3, 'ry', 'cx', 'linear', reps=2, insert_barriers=True)
        >>> print(two)  # decompose the layers into standard gates
             ┌──────────┐ ░            ░ ┌──────────┐ ░            ░ ┌──────────┐
        q_0: ┤ Ry(θ[0]) ├─░───■────────░─┤ Ry(θ[3]) ├─░───■────────░─┤ Ry(θ[6]) ├
             ├──────────┤ ░ ┌─┴─┐      ░ ├──────────┤ ░ ┌─┴─┐      ░ ├──────────┤
        q_1: ┤ Ry(θ[1]) ├─░─┤ X ├──■───░─┤ Ry(θ[4]) ├─░─┤ X ├──■───░─┤ Ry(θ[7]) ├
             ├──────────┤ ░ └───┘┌─┴─┐ ░ ├──────────┤ ░ └───┘┌─┴─┐ ░ ├──────────┤
        q_2: ┤ Ry(θ[2]) ├─░──────┤ X ├─░─┤ Ry(θ[5]) ├─░──────┤ X ├─░─┤ Ry(θ[8]) ├
             └──────────┘ ░      └───┘ ░ └──────────┘ ░      └───┘ ░ └──────────┘

        >>> two = TwoLocal(3, ['ry','rz'], 'cz', 'full', reps=1, insert_barriers=True)
        >>> qc = QuantumCircuit(3)
        >>> qc += two
        >>> print(qc.decompose().draw())
             ┌──────────┐┌──────────┐ ░           ░ ┌──────────┐ ┌──────────┐
        q_0: ┤ Ry(θ[0]) ├┤ Rz(θ[3]) ├─░──■──■─────░─┤ Ry(θ[6]) ├─┤ Rz(θ[9]) ├
             ├──────────┤├──────────┤ ░  │  │     ░ ├──────────┤┌┴──────────┤
        q_1: ┤ Ry(θ[1]) ├┤ Rz(θ[4]) ├─░──■──┼──■──░─┤ Ry(θ[7]) ├┤ Rz(θ[10]) ├
             ├──────────┤├──────────┤ ░     │  │  ░ ├──────────┤├───────────┤
        q_2: ┤ Ry(θ[2]) ├┤ Rz(θ[5]) ├─░─────■──■──░─┤ Ry(θ[8]) ├┤ Rz(θ[11]) ├
             └──────────┘└──────────┘ ░           ░ └──────────┘└───────────┘

        >>> entangler_map = [[0, 1], [1, 2], [2, 0]]  # circular entanglement for 3 qubits
        >>> two = TwoLocal(3, 'x', 'crx', entangler_map, reps=1)
        >>> print(two)  # note: no barriers inserted this time!
                ┌───┐                             ┌──────────┐┌───┐
        q_0: |0>┤ X ├─────■───────────────────────┤ Rx(θ[2]) ├┤ X ├
                ├───┤┌────┴─────┐            ┌───┐└─────┬────┘└───┘
        q_1: |0>┤ X ├┤ Rx(θ[0]) ├─────■──────┤ X ├──────┼──────────
                ├───┤└──────────┘┌────┴─────┐└───┘      │     ┌───┐
        q_2: |0>┤ X ├────────────┤ Rx(θ[1]) ├───────────■─────┤ X ├
                └───┘            └──────────┘                 └───┘

        >>> entangler_map = [[0, 3], [0, 2]]  # entangle the first and last two-way
        >>> two = TwoLocal(4, [], 'cry', entangler_map, reps=1)
        >>> circuit = two + two
        >>> print(circuit.decompose().draw())  # note, that the parameters are the same!
        q_0: ─────■───────────■───────────■───────────■──────
                  │           │           │           │
        q_1: ─────┼───────────┼───────────┼───────────┼──────
                  │      ┌────┴─────┐     │      ┌────┴─────┐
        q_2: ─────┼──────┤ Ry(θ[1]) ├─────┼──────┤ Ry(θ[1]) ├
             ┌────┴─────┐└──────────┘┌────┴─────┐└──────────┘
        q_3: ┤ Ry(θ[0]) ├────────────┤ Ry(θ[0]) ├────────────
             └──────────┘            └──────────┘

        >>> layer_1 = [(0, 1), (0, 2)]
        >>> layer_2 = [(1, 2)]
        >>> two = TwoLocal(3, 'x', 'cx', [layer_1, layer_2], reps=2, insert_barriers=True)
        >>> print(two)
             ┌───┐ ░            ░ ┌───┐ ░       ░ ┌───┐
        q_0: ┤ X ├─░───■────■───░─┤ X ├─░───────░─┤ X ├
             ├───┤ ░ ┌─┴─┐  │   ░ ├───┤ ░       ░ ├───┤
        q_1: ┤ X ├─░─┤ X ├──┼───░─┤ X ├─░───■───░─┤ X ├
             ├───┤ ░ └───┘┌─┴─┐ ░ ├───┤ ░ ┌─┴─┐ ░ ├───┤
        q_2: ┤ X ├─░──────┤ X ├─░─┤ X ├─░─┤ X ├─░─┤ X ├
             └───┘ ░      └───┘ ░ └───┘ ░ └───┘ ░ └───┘

    """

    def __init__(self, num_qubits: int | None=None, rotation_blocks: str | type | qiskit.circuit.Instruction | QuantumCircuit | list[str | type | qiskit.circuit.Instruction | QuantumCircuit] | None=None, entanglement_blocks: str | type | qiskit.circuit.Instruction | QuantumCircuit | list[str | type | qiskit.circuit.Instruction | QuantumCircuit] | None=None, entanglement: str | list[list[int]] | Callable[[int], list[int]]='full', reps: int=3, skip_unentangled_qubits: bool=False, skip_final_rotation_layer: bool=False, parameter_prefix: str='θ', insert_barriers: bool=False, initial_state: QuantumCircuit | None=None, name: str='TwoLocal', flatten: bool | None=None) -> None:
        if False:
            return 10
        "\n        Args:\n            num_qubits: The number of qubits of the two-local circuit.\n            rotation_blocks: The gates used in the rotation layer. Can be specified via the name of\n                a gate (e.g. ``'ry'``) or the gate type itself (e.g. :class:`.RYGate`).\n                If only one gate is provided, the gate same gate is applied to each qubit.\n                If a list of gates is provided, all gates are applied to each qubit in the provided\n                order.\n                See the Examples section for more detail.\n            entanglement_blocks: The gates used in the entanglement layer. Can be specified in\n                the same format as ``rotation_blocks``.\n            entanglement: Specifies the entanglement structure. Can be a string (``'full'``,\n                ``'linear'``, ``'reverse_linear'``, ``'circular'`` or ``'sca'``),\n                a list of integer-pairs specifying the indices\n                of qubits entangled with one another, or a callable returning such a list provided with\n                the index of the entanglement layer.\n                Default to ``'full'`` entanglement.\n                Note that if ``entanglement_blocks = 'cx'``, then ``'full'`` entanglement provides the\n                same unitary as ``'reverse_linear'`` but the latter option has fewer entangling gates.\n                See the Examples section for more detail.\n            reps: Specifies how often a block consisting of a rotation layer and entanglement\n                layer is repeated.\n            skip_unentangled_qubits: If ``True``, the single qubit gates are only applied to qubits\n                that are entangled with another qubit. If ``False``, the single qubit gates are applied\n                to each qubit in the ansatz. Defaults to ``False``.\n            skip_final_rotation_layer: If ``False``, a rotation layer is added at the end of the\n                ansatz. If ``True``, no rotation layer is added.\n            parameter_prefix: The parameterized gates require a parameter to be defined, for which\n                we use instances of :class:`~qiskit.circuit.Parameter`. The name of each parameter will\n                be this specified prefix plus its index.\n            insert_barriers: If ``True``, barriers are inserted in between each layer. If ``False``,\n                no barriers are inserted. Defaults to ``False``.\n            initial_state: A :class:`.QuantumCircuit` object to prepend to the circuit.\n            flatten: Set this to ``True`` to output a flat circuit instead of nesting it inside multiple\n                layers of gate objects. By default currently the contents of\n                the output circuit will be wrapped in nested objects for\n                cleaner visualization. However, if you're using this circuit\n                for anything besides visualization its **strongly** recommended\n                to set this flag to ``True`` to avoid a large performance\n                overhead for parameter binding.\n\n        "
        super().__init__(num_qubits=num_qubits, rotation_blocks=rotation_blocks, entanglement_blocks=entanglement_blocks, entanglement=entanglement, reps=reps, skip_final_rotation_layer=skip_final_rotation_layer, skip_unentangled_qubits=skip_unentangled_qubits, insert_barriers=insert_barriers, initial_state=initial_state, parameter_prefix=parameter_prefix, name=name, flatten=flatten)

    def _convert_to_block(self, layer: str | type | Gate | QuantumCircuit) -> QuantumCircuit:
        if False:
            i = 10
            return i + 15
        "For a layer provided as str (e.g. ``'ry'``) or type (e.g. :class:`.RYGate`) this function\n         returns the\n         according layer type along with the number of parameters (e.g. ``(RYGate, 1)``).\n\n        Args:\n            layer: The qubit layer.\n\n        Returns:\n            The specified layer with the required number of parameters.\n\n        Raises:\n            TypeError: The type of ``layer`` is invalid.\n            ValueError: The type of ``layer`` is str but the name is unknown.\n            ValueError: The type of ``layer`` is type but the layer type is unknown.\n\n        Note:\n            Outlook: If layers knew their number of parameters as static property, we could also\n            allow custom layer types.\n        "
        if isinstance(layer, QuantumCircuit):
            return layer
        theta = Parameter('θ')
        valid_layers = {'ch': CHGate(), 'cx': CXGate(), 'cy': CYGate(), 'cz': CZGate(), 'crx': CRXGate(theta), 'cry': CRYGate(theta), 'crz': CRZGate(theta), 'h': HGate(), 'i': IGate(), 'id': IGate(), 'iden': IGate(), 'rx': RXGate(theta), 'rxx': RXXGate(theta), 'ry': RYGate(theta), 'ryy': RYYGate(theta), 'rz': RZGate(theta), 'rzx': RZXGate(theta), 'rzz': RZZGate(theta), 's': SGate(), 'sdg': SdgGate(), 'swap': SwapGate(), 'x': XGate(), 'y': YGate(), 'z': ZGate(), 't': TGate(), 'tdg': TdgGate()}
        if isinstance(layer, str):
            try:
                layer = valid_layers[layer]
            except KeyError as ex:
                raise ValueError(f'Unknown layer name `{layer}`.') from ex
        if isinstance(layer, type):
            instance = None
            for gate in valid_layers.values():
                if isinstance(gate, layer):
                    instance = gate
            if instance is None:
                raise ValueError(f'Unknown layer type`{layer}`.')
            layer = instance
        if isinstance(layer, Instruction):
            circuit = QuantumCircuit(layer.num_qubits)
            circuit.append(layer, list(range(layer.num_qubits)))
            return circuit
        raise TypeError(f'Invalid input type {type(layer)}. ' + '`layer` must be a type, str or QuantumCircuit.')

    def get_entangler_map(self, rep_num: int, block_num: int, num_block_qubits: int) -> Sequence[Sequence[int]]:
        if False:
            i = 10
            return i + 15
        'Overloading to handle the special case of 1 qubit where the entanglement are ignored.'
        if self.num_qubits <= 1:
            return []
        return super().get_entangler_map(rep_num, block_num, num_block_qubits)