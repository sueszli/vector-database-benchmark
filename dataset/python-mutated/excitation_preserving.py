"""The ExcitationPreserving 2-local circuit."""
from __future__ import annotations
from collections.abc import Callable
from numpy import pi
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library.standard_gates import RZGate
from .two_local import TwoLocal

class ExcitationPreserving(TwoLocal):
    """The heuristic excitation-preserving wave function ansatz.

    The ``ExcitationPreserving`` circuit preserves the ratio of :math:`|00\\rangle`,
    :math:`|01\\rangle + |10\\rangle` and :math:`|11\\rangle` states. To this end, this circuit
    uses two-qubit interactions of the form

    .. math::

        \\newcommand{\\th}{\\theta/2}

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & \\cos\\left(\\th\\right) & -i\\sin\\left(\\th\\right) & 0 \\\\
        0 & -i\\sin\\left(\\th\\right) & \\cos\\left(\\th\\right) & 0 \\\\
        0 & 0 & 0 & e^{-i\\phi}
        \\end{pmatrix}

    for the mode ``'fsim'`` or with :math:`e^{-i\\phi} = 1` for the mode ``'iswap'``.

    Note that other wave functions, such as UCC-ansatzes, are also excitation preserving.
    However these can become complex quickly, while this heuristically motivated circuit follows
    a simpler pattern.

    This trial wave function consists of layers of :math:`Z` rotations with 2-qubit entanglements.
    The entangling is creating using :math:`XX+YY` rotations and optionally a controlled-phase
    gate for the mode ``'fsim'``.

    See :class:`~qiskit.circuit.library.RealAmplitudes` for more detail on the possible arguments
    and options such as skipping unentanglement qubits, which apply here too.

    The rotations of the ExcitationPreserving ansatz can be written as

    Examples:

        >>> ansatz = ExcitationPreserving(3, reps=1, insert_barriers=True, entanglement='linear')
        >>> print(ansatz)  # show the circuit
             ┌──────────┐ ░ ┌────────────┐┌────────────┐                             ░ ┌──────────┐
        q_0: ┤ RZ(θ[0]) ├─░─┤0           ├┤0           ├─────────────────────────────░─┤ RZ(θ[5]) ├
             ├──────────┤ ░ │  RXX(θ[3]) ││  RYY(θ[3]) │┌────────────┐┌────────────┐ ░ ├──────────┤
        q_1: ┤ RZ(θ[1]) ├─░─┤1           ├┤1           ├┤0           ├┤0           ├─░─┤ RZ(θ[6]) ├
             ├──────────┤ ░ └────────────┘└────────────┘│  RXX(θ[4]) ││  RYY(θ[4]) │ ░ ├──────────┤
        q_2: ┤ RZ(θ[2]) ├─░─────────────────────────────┤1           ├┤1           ├─░─┤ RZ(θ[7]) ├
             └──────────┘ ░                             └────────────┘└────────────┘ ░ └──────────┘

        >>> ansatz = ExcitationPreserving(2, reps=1)
        >>> qc = QuantumCircuit(2)  # create a circuit and append the RY variational form
        >>> qc.cry(0.2, 0, 1)  # do some previous operation
        >>> qc.compose(ansatz, inplace=True)  # add the swaprz
        >>> qc.draw()
                        ┌──────────┐┌────────────┐┌────────────┐┌──────────┐
        q_0: ─────■─────┤ RZ(θ[0]) ├┤0           ├┤0           ├┤ RZ(θ[3]) ├
             ┌────┴────┐├──────────┤│  RXX(θ[2]) ││  RYY(θ[2]) │├──────────┤
        q_1: ┤ RY(0.2) ├┤ RZ(θ[1]) ├┤1           ├┤1           ├┤ RZ(θ[4]) ├
             └─────────┘└──────────┘└────────────┘└────────────┘└──────────┘

        >>> ansatz = ExcitationPreserving(3, reps=1, mode='fsim', entanglement=[[0,2]],
        ... insert_barriers=True)
        >>> print(ansatz)
             ┌──────────┐ ░ ┌────────────┐┌────────────┐        ░ ┌──────────┐
        q_0: ┤ RZ(θ[0]) ├─░─┤0           ├┤0           ├─■──────░─┤ RZ(θ[5]) ├
             ├──────────┤ ░ │            ││            │ │      ░ ├──────────┤
        q_1: ┤ RZ(θ[1]) ├─░─┤  RXX(θ[3]) ├┤  RYY(θ[3]) ├─┼──────░─┤ RZ(θ[6]) ├
             ├──────────┤ ░ │            ││            │ │θ[4]  ░ ├──────────┤
        q_2: ┤ RZ(θ[2]) ├─░─┤1           ├┤1           ├─■──────░─┤ RZ(θ[7]) ├
             └──────────┘ ░ └────────────┘└────────────┘        ░ └──────────┘
    """

    def __init__(self, num_qubits: int | None=None, mode: str='iswap', entanglement: str | list[list[int]] | Callable[[int], list[int]]='full', reps: int=3, skip_unentangled_qubits: bool=False, skip_final_rotation_layer: bool=False, parameter_prefix: str='θ', insert_barriers: bool=False, initial_state: QuantumCircuit | None=None, name: str='ExcitationPreserving', flatten: bool | None=None) -> None:
        if False:
            print('Hello World!')
        "\n        Args:\n            num_qubits: The number of qubits of the ExcitationPreserving circuit.\n            mode: Choose the entangler mode, can be `'iswap'` or `'fsim'`.\n            reps: Specifies how often the structure of a rotation layer followed by an entanglement\n                layer is repeated.\n            entanglement: Specifies the entanglement structure. Can be a string ('full', 'linear'\n                or 'sca'), a list of integer-pairs specifying the indices of qubits\n                entangled with one another, or a callable returning such a list provided with\n                the index of the entanglement layer.\n                See the Examples section of :class:`~qiskit.circuit.library.TwoLocal` for more\n                detail.\n            initial_state: A `QuantumCircuit` object to prepend to the circuit.\n            skip_unentangled_qubits: If True, the single qubit gates are only applied to qubits\n                that are entangled with another qubit. If False, the single qubit gates are applied\n                to each qubit in the Ansatz. Defaults to False.\n            skip_unentangled_qubits: If True, the single qubit gates are only applied to qubits\n                that are entangled with another qubit. If False, the single qubit gates are applied\n                to each qubit in the Ansatz. Defaults to False.\n            skip_final_rotation_layer: If True, a rotation layer is added at the end of the\n                ansatz. If False, no rotation layer is added. Defaults to True.\n            parameter_prefix: The parameterized gates require a parameter to be defined, for which\n                we use :class:`~qiskit.circuit.ParameterVector`.\n            insert_barriers: If True, barriers are inserted in between each layer. If False,\n                no barriers are inserted.\n            flatten: Set this to ``True`` to output a flat circuit instead of nesting it inside multiple\n                layers of gate objects. By default currently the contents of\n                the output circuit will be wrapped in nested objects for\n                cleaner visualization. However, if you're using this circuit\n                for anything besides visualization its **strongly** recommended\n                to set this flag to ``True`` to avoid a large performance\n                overhead for parameter binding.\n\n        Raises:\n            ValueError: If the selected mode is not supported.\n        "
        supported_modes = ['iswap', 'fsim']
        if mode not in supported_modes:
            raise ValueError(f'Unsupported mode {mode}, choose one of {supported_modes}')
        theta = Parameter('θ')
        swap = QuantumCircuit(2, name='Interaction')
        swap.rxx(theta, 0, 1)
        swap.ryy(theta, 0, 1)
        if mode == 'fsim':
            phi = Parameter('φ')
            swap.cp(phi, 0, 1)
        super().__init__(num_qubits=num_qubits, rotation_blocks=RZGate, entanglement_blocks=swap, entanglement=entanglement, reps=reps, skip_unentangled_qubits=skip_unentangled_qubits, skip_final_rotation_layer=skip_final_rotation_layer, parameter_prefix=parameter_prefix, insert_barriers=insert_barriers, initial_state=initial_state, name=name, flatten=flatten)

    @property
    def parameter_bounds(self) -> list[tuple[float, float]]:
        if False:
            return 10
        'Return the parameter bounds.\n\n        Returns:\n            The parameter bounds.\n        '
        return self.num_parameters * [(-pi, pi)]