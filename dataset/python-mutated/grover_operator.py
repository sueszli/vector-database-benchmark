"""The Grover operator."""
from __future__ import annotations
from typing import List, Optional, Union
import numpy
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Statevector, Operator, DensityMatrix
from .standard_gates import MCXGate

class GroverOperator(QuantumCircuit):
    """The Grover operator.

    Grover's search algorithm [1, 2] consists of repeated applications of the so-called
    Grover operator used to amplify the amplitudes of the desired output states.
    This operator, :math:`\\mathcal{Q}`, consists of the phase oracle, :math:`\\mathcal{S}_f`,
    zero phase-shift or zero reflection, :math:`\\mathcal{S}_0`, and an
    input state preparation :math:`\\mathcal{A}`:

    .. math::
        \\mathcal{Q} = \\mathcal{A} \\mathcal{S}_0 \\mathcal{A}^\\dagger \\mathcal{S}_f

    In the standard Grover search we have :math:`\\mathcal{A} = H^{\\otimes n}`:

    .. math::
        \\mathcal{Q} = H^{\\otimes n} \\mathcal{S}_0 H^{\\otimes n} \\mathcal{S}_f
                    = D \\mathcal{S_f}

    The operation :math:`D = H^{\\otimes n} \\mathcal{S}_0 H^{\\otimes n}` is also referred to as
    diffusion operator. In this formulation we can see that Grover's operator consists of two
    steps: first, the phase oracle multiplies the good states by -1 (with :math:`\\mathcal{S}_f`)
    and then the whole state is reflected around the mean (with :math:`D`).

    This class allows setting a different state preparation, as in quantum amplitude
    amplification (a generalization of Grover's algorithm), :math:`\\mathcal{A}` might not be
    a layer of Hardamard gates [3].

    The action of the phase oracle :math:`\\mathcal{S}_f` is defined as

    .. math::
        \\mathcal{S}_f: |x\\rangle \\mapsto (-1)^{f(x)}|x\\rangle

    where :math:`f(x) = 1` if :math:`x` is a good state and 0 otherwise. To highlight the fact
    that this oracle flips the phase of the good states and does not flip the state of a result
    qubit, we call :math:`\\mathcal{S}_f` a phase oracle.

    Note that you can easily construct a phase oracle from a bitflip oracle by sandwiching the
    controlled X gate on the result qubit by a X and H gate. For instance

    .. parsed-literal::

        Bitflip oracle     Phaseflip oracle
        q_0: ──■──         q_0: ────────────■────────────
             ┌─┴─┐              ┌───┐┌───┐┌─┴─┐┌───┐┌───┐
        out: ┤ X ├         out: ┤ X ├┤ H ├┤ X ├┤ H ├┤ X ├
             └───┘              └───┘└───┘└───┘└───┘└───┘

    There is some flexibility in defining the oracle and :math:`\\mathcal{A}` operator. Before the
    Grover operator is applied in Grover's algorithm, the qubits are first prepared with one
    application of the :math:`\\mathcal{A}` operator (or Hadamard gates in the standard formulation).
    Thus, we always have operation of the form
    :math:`\\mathcal{A} \\mathcal{S}_f \\mathcal{A}^\\dagger`. Therefore it is possible to move
    bitflip logic into :math:`\\mathcal{A}` and leaving the oracle only to do phaseflips via Z gates
    based on the bitflips. One possible use-case for this are oracles that do not uncompute the
    state qubits.

    The zero reflection :math:`\\mathcal{S}_0` is usually defined as

    .. math::
        \\mathcal{S}_0 = 2 |0\\rangle^{\\otimes n} \\langle 0|^{\\otimes n} - \\mathbb{I}_n

    where :math:`\\mathbb{I}_n` is the identity on :math:`n` qubits.
    By default, this class implements the negative version
    :math:`2 |0\\rangle^{\\otimes n} \\langle 0|^{\\otimes n} - \\mathbb{I}_n`, since this can simply
    be implemented with a multi-controlled Z sandwiched by X gates on the target qubit and the
    introduced global phase does not matter for Grover's algorithm.

    Examples:
        >>> from qiskit.circuit import QuantumCircuit
        >>> from qiskit.circuit.library import GroverOperator
        >>> oracle = QuantumCircuit(2)
        >>> oracle.z(0)  # good state = first qubit is |1>
        >>> grover_op = GroverOperator(oracle, insert_barriers=True)
        >>> grover_op.decompose().draw()
                 ┌───┐ ░ ┌───┐ ░ ┌───┐          ┌───┐      ░ ┌───┐
        state_0: ┤ Z ├─░─┤ H ├─░─┤ X ├───────■──┤ X ├──────░─┤ H ├
                 └───┘ ░ ├───┤ ░ ├───┤┌───┐┌─┴─┐├───┤┌───┐ ░ ├───┤
        state_1: ──────░─┤ H ├─░─┤ X ├┤ H ├┤ X ├┤ H ├┤ X ├─░─┤ H ├
                       ░ └───┘ ░ └───┘└───┘└───┘└───┘└───┘ ░ └───┘

        >>> oracle = QuantumCircuit(1)
        >>> oracle.z(0)  # the qubit state |1> is the good state
        >>> state_preparation = QuantumCircuit(1)
        >>> state_preparation.ry(0.2, 0)  # non-uniform state preparation
        >>> grover_op = GroverOperator(oracle, state_preparation)
        >>> grover_op.decompose().draw()
                 ┌───┐┌──────────┐┌───┐┌───┐┌───┐┌─────────┐
        state_0: ┤ Z ├┤ RY(-0.2) ├┤ X ├┤ Z ├┤ X ├┤ RY(0.2) ├
                 └───┘└──────────┘└───┘└───┘└───┘└─────────┘

        >>> oracle = QuantumCircuit(4)
        >>> oracle.z(3)
        >>> reflection_qubits = [0, 3]
        >>> state_preparation = QuantumCircuit(4)
        >>> state_preparation.cry(0.1, 0, 3)
        >>> state_preparation.ry(0.5, 3)
        >>> grover_op = GroverOperator(oracle, state_preparation,
        ... reflection_qubits=reflection_qubits)
        >>> grover_op.decompose().draw()
                                              ┌───┐          ┌───┐
        state_0: ──────────────────────■──────┤ X ├───────■──┤ X ├──────────■────────────────
                                       │      └───┘       │  └───┘          │
        state_1: ──────────────────────┼──────────────────┼─────────────────┼────────────────
                                       │                  │                 │
        state_2: ──────────────────────┼──────────────────┼─────────────────┼────────────────
                 ┌───┐┌──────────┐┌────┴─────┐┌───┐┌───┐┌─┴─┐┌───┐┌───┐┌────┴────┐┌─────────┐
        state_3: ┤ Z ├┤ RY(-0.5) ├┤ RY(-0.1) ├┤ X ├┤ H ├┤ X ├┤ H ├┤ X ├┤ RY(0.1) ├┤ RY(0.5) ├
                 └───┘└──────────┘└──────────┘└───┘└───┘└───┘└───┘└───┘└─────────┘└─────────┘

        >>> mark_state = Statevector.from_label('011')
        >>> diffuse_operator = 2 * DensityMatrix.from_label('000') - Operator.from_label('III')
        >>> grover_op = GroverOperator(oracle=mark_state, zero_reflection=diffuse_operator)
        >>> grover_op.decompose().draw(fold=70)
                 ┌─────────────────┐      ┌───┐                          »
        state_0: ┤0                ├──────┤ H ├──────────────────────────»
                 │                 │┌─────┴───┴─────┐     ┌───┐          »
        state_1: ┤1 UCRZ(0,pi,0,0) ├┤0              ├─────┤ H ├──────────»
                 │                 ││  UCRZ(pi/2,0) │┌────┴───┴────┐┌───┐»
        state_2: ┤2                ├┤1              ├┤ UCRZ(-pi/4) ├┤ H ├»
                 └─────────────────┘└───────────────┘└─────────────┘└───┘»
        «         ┌─────────────────┐      ┌───┐
        «state_0: ┤0                ├──────┤ H ├─────────────────────────
        «         │                 │┌─────┴───┴─────┐    ┌───┐
        «state_1: ┤1 UCRZ(pi,0,0,0) ├┤0              ├────┤ H ├──────────
        «         │                 ││  UCRZ(pi/2,0) │┌───┴───┴────┐┌───┐
        «state_2: ┤2                ├┤1              ├┤ UCRZ(pi/4) ├┤ H ├
        «         └─────────────────┘└───────────────┘└────────────┘└───┘

    References:
        [1]: L. K. Grover (1996), A fast quantum mechanical algorithm for database search,
            `arXiv:quant-ph/9605043 <https://arxiv.org/abs/quant-ph/9605043>`_.
        [2]: I. Chuang & M. Nielsen, Quantum Computation and Quantum Information,
            Cambridge: Cambridge University Press, 2000. Chapter 6.1.2.
        [3]: Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
            Quantum Amplitude Amplification and Estimation.
            `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_.
    """

    def __init__(self, oracle: Union[QuantumCircuit, Statevector], state_preparation: Optional[QuantumCircuit]=None, zero_reflection: Optional[Union[QuantumCircuit, DensityMatrix, Operator]]=None, reflection_qubits: Optional[List[int]]=None, insert_barriers: bool=False, mcx_mode: str='noancilla', name: str='Q') -> None:
        if False:
            return 10
        "\n        Args:\n            oracle: The phase oracle implementing a reflection about the bad state. Note that this\n                is not a bitflip oracle, see the docstring for more information.\n            state_preparation: The operator preparing the good and bad state.\n                For Grover's algorithm, this is a n-qubit Hadamard gate and for amplitude\n                amplification or estimation the operator :math:`\\mathcal{A}`.\n            zero_reflection: The reflection about the zero state, :math:`\\mathcal{S}_0`.\n            reflection_qubits: Qubits on which the zero reflection acts on.\n            insert_barriers: Whether barriers should be inserted between the reflections and A.\n            mcx_mode: The mode to use for building the default zero reflection.\n            name: The name of the circuit.\n        "
        super().__init__(name=name)
        if isinstance(oracle, Statevector):
            from qiskit.circuit.library import Diagonal
            oracle = Diagonal((-1) ** oracle.data)
        self._oracle = oracle
        if isinstance(zero_reflection, (Operator, DensityMatrix)):
            from qiskit.circuit.library import Diagonal
            zero_reflection = Diagonal(zero_reflection.data.diagonal())
        self._zero_reflection = zero_reflection
        self._reflection_qubits = reflection_qubits
        self._state_preparation = state_preparation
        self._insert_barriers = insert_barriers
        self._mcx_mode = mcx_mode
        self._build()

    @property
    def reflection_qubits(self):
        if False:
            while True:
                i = 10
        'Reflection qubits, on which S0 is applied (if S0 is not user-specified).'
        if self._reflection_qubits is not None:
            return self._reflection_qubits
        num_state_qubits = self.oracle.num_qubits - self.oracle.num_ancillas
        return list(range(num_state_qubits))

    @property
    def zero_reflection(self) -> QuantumCircuit:
        if False:
            print('Hello World!')
        'The subcircuit implementing the reflection about 0.'
        if self._zero_reflection is not None:
            return self._zero_reflection
        num_state_qubits = self.oracle.num_qubits - self.oracle.num_ancillas
        return _zero_reflection(num_state_qubits, self.reflection_qubits, self._mcx_mode)

    @property
    def state_preparation(self) -> QuantumCircuit:
        if False:
            while True:
                i = 10
        'The subcircuit implementing the A operator or Hadamards.'
        if self._state_preparation is not None:
            return self._state_preparation
        num_state_qubits = self.oracle.num_qubits - self.oracle.num_ancillas
        hadamards = QuantumCircuit(num_state_qubits, name='H')
        hadamards.h(self.reflection_qubits)
        return hadamards

    @property
    def oracle(self):
        if False:
            return 10
        'The oracle implementing a reflection about the bad state.'
        return self._oracle

    def _build(self):
        if False:
            while True:
                i = 10
        num_state_qubits = self.oracle.num_qubits - self.oracle.num_ancillas
        circuit = QuantumCircuit(QuantumRegister(num_state_qubits, name='state'), name='Q')
        num_ancillas = numpy.max([self.oracle.num_ancillas, self.zero_reflection.num_ancillas, self.state_preparation.num_ancillas])
        if num_ancillas > 0:
            circuit.add_register(AncillaRegister(num_ancillas, name='ancilla'))
        circuit.compose(self.oracle, list(range(self.oracle.num_qubits)), inplace=True)
        if self._insert_barriers:
            circuit.barrier()
        circuit.compose(self.state_preparation.inverse(), list(range(self.state_preparation.num_qubits)), inplace=True)
        if self._insert_barriers:
            circuit.barrier()
        circuit.compose(self.zero_reflection, list(range(self.zero_reflection.num_qubits)), inplace=True)
        if self._insert_barriers:
            circuit.barrier()
        circuit.compose(self.state_preparation, list(range(self.state_preparation.num_qubits)), inplace=True)
        circuit.global_phase = numpy.pi
        self.add_register(*circuit.qregs)
        try:
            circuit_wrapped = circuit.to_gate()
        except QiskitError:
            circuit_wrapped = circuit.to_instruction()
        self.compose(circuit_wrapped, qubits=self.qubits, inplace=True)

def _zero_reflection(num_state_qubits: int, qubits: List[int], mcx_mode: Optional[str]=None) -> QuantumCircuit:
    if False:
        while True:
            i = 10
    qr_state = QuantumRegister(num_state_qubits, 'state')
    reflection = QuantumCircuit(qr_state, name='S_0')
    num_ancillas = MCXGate.get_num_ancilla_qubits(len(qubits) - 1, mcx_mode)
    if num_ancillas > 0:
        qr_ancilla = AncillaRegister(num_ancillas, 'ancilla')
        reflection.add_register(qr_ancilla)
    else:
        qr_ancilla = AncillaRegister(0)
    reflection.x(qubits)
    if len(qubits) == 1:
        reflection.z(0)
    else:
        reflection.h(qubits[-1])
        reflection.mcx(qubits[:-1], qubits[-1], qr_ancilla[:], mode=mcx_mode)
        reflection.h(qubits[-1])
    reflection.x(qubits)
    return reflection