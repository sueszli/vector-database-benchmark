"""PauliTrotterEvolution Class"""
import logging
from typing import Optional, Union, cast
import numpy as np
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.opflow.converters.pauli_basis_change import PauliBasisChange
from qiskit.opflow.evolutions.evolution_base import EvolutionBase
from qiskit.opflow.evolutions.evolved_op import EvolvedOp
from qiskit.opflow.evolutions.trotterizations.trotterization_base import TrotterizationBase
from qiskit.opflow.evolutions.trotterizations.trotterization_factory import TrotterizationFactory
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.operator_globals import I, Z
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.opflow.primitive_ops.circuit_op import CircuitOp
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.utils.deprecation import deprecate_func
logger = logging.getLogger(__name__)

class PauliTrotterEvolution(EvolutionBase):
    """
    Deprecated: An Evolution algorithm replacing exponentiated sums of Paulis by changing
    them each to the Z basis, rotating with an rZ, changing back, and Trotterizing.

    More specifically, we compute basis change circuits for each Pauli into a single-qubit Z,
    evolve the Z by the desired evolution time with an rZ gate, and change the basis back using
    the adjoint of the original basis change circuit. For sums of Paulis, the individual Pauli
    evolution circuits are composed together by Trotterization scheme.
    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, trotter_mode: Optional[Union[str, TrotterizationBase]]='trotter', reps: Optional[int]=1) -> None:
        if False:
            return 10
        "\n        Args:\n            trotter_mode: A string ('trotter', 'suzuki', or 'qdrift') to pass to the\n                TrotterizationFactory, or a TrotterizationBase, indicating how to combine\n                individual Pauli evolution circuits to equal the exponentiation of the Pauli sum.\n            reps: How many Trotterization repetitions to make, to improve the approximation\n                accuracy.\n            # TODO uncomment when we implement Abelian grouped evolution.\n            # group_paulis: Whether to group Pauli sums into Abelian\n            #     sub-groups, so a single diagonalization circuit can be used for each group\n            #     rather than each Pauli.\n        "
        super().__init__()
        if isinstance(trotter_mode, TrotterizationBase):
            self._trotter = trotter_mode
        else:
            self._trotter = TrotterizationFactory.build(mode=trotter_mode, reps=reps)

    @property
    def trotter(self) -> TrotterizationBase:
        if False:
            for i in range(10):
                print('nop')
        'TrotterizationBase used to evolve SummedOps.'
        return self._trotter

    @trotter.setter
    def trotter(self, trotter: TrotterizationBase) -> None:
        if False:
            print('Hello World!')
        'Set TrotterizationBase used to evolve SummedOps.'
        self._trotter = trotter

    def convert(self, operator: OperatorBase) -> OperatorBase:
        if False:
            while True:
                i = 10
        '\n        Traverse the operator, replacing ``EvolvedOps`` with ``CircuitOps`` containing\n        Trotterized evolutions equalling the exponentiation of -i * operator.\n\n        Args:\n            operator: The Operator to convert.\n\n        Returns:\n            The converted operator.\n        '
        return self._recursive_convert(operator)

    def _get_evolution_synthesis(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the ``EvolutionSynthesis`` corresponding to this Trotterization.'
        if self.trotter.order == 1:
            return LieTrotter(reps=self.trotter.reps)
        return SuzukiTrotter(reps=self.trotter.reps, order=self.trotter.order)

    def _recursive_convert(self, operator: OperatorBase) -> OperatorBase:
        if False:
            print('Hello World!')
        if isinstance(operator, EvolvedOp):
            if isinstance(operator.primitive, (PauliOp, PauliSumOp)):
                pauli = operator.primitive.primitive
                time = operator.coeff * operator.primitive.coeff
                evo = PauliEvolutionGate(pauli, time=time, synthesis=self._get_evolution_synthesis())
                return CircuitOp(evo)
            if not {'Pauli'} == operator.primitive_strings():
                logger.warning('Evolved Hamiltonian is not composed of only Paulis, converting to Pauli representation, which can be expensive.')
                pauli_ham = operator.primitive.to_pauli_op(massive=False)
                operator = EvolvedOp(pauli_ham, coeff=operator.coeff)
            if isinstance(operator.primitive, SummedOp):
                oplist = [x for x in operator.primitive if not isinstance(x, PauliOp) or sum(x.primitive.x + x.primitive.z) != 0]
                identity_phases = [x.coeff for x in operator.primitive if isinstance(x, PauliOp) and sum(x.primitive.x + x.primitive.z) == 0]
                new_primitive = SummedOp(oplist, coeff=operator.primitive.coeff)
                trotterized = self.trotter.convert(new_primitive)
                circuit_no_identities = self._recursive_convert(trotterized)
                global_phase = -sum(identity_phases) * operator.primitive.coeff
                circuit_no_identities.primitive.global_phase = global_phase
                return circuit_no_identities
            elif isinstance(operator.primitive, ListOp):
                converted_ops = [self._recursive_convert(op) for op in operator.primitive.oplist]
                return operator.primitive.__class__(converted_ops, coeff=operator.coeff)
        elif isinstance(operator, ListOp):
            return operator.traverse(self.convert).reduce()
        return operator

    def evolution_for_pauli(self, pauli_op: PauliOp) -> PrimitiveOp:
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute evolution Operator for a single Pauli using a ``PauliBasisChange``.\n\n        Args:\n            pauli_op: The ``PauliOp`` to evolve.\n\n        Returns:\n            A ``PrimitiveOp``, either the evolution ``CircuitOp`` or a ``PauliOp`` equal to the\n            identity if pauli_op is the identity.\n        '

        def replacement_fn(cob_instr_op, dest_pauli_op):
            if False:
                i = 10
                return i + 15
            z_evolution = dest_pauli_op.exp_i()
            return cob_instr_op.adjoint().compose(z_evolution).compose(cob_instr_op)
        sig_bits = np.logical_or(pauli_op.primitive.z, pauli_op.primitive.x)
        a_sig_bit = int(max(np.extract(sig_bits, np.arange(pauli_op.num_qubits)[::-1])))
        destination = I.tensorpower(a_sig_bit) ^ Z * pauli_op.coeff
        cob = PauliBasisChange(destination_basis=destination, replacement_fn=replacement_fn)
        return cast(PrimitiveOp, cob.convert(pauli_op))

    def evolution_for_abelian_paulisum(self, op_sum: SummedOp) -> PrimitiveOp:
        if False:
            while True:
                i = 10
        'Evolution for abelian pauli sum'
        raise NotImplementedError