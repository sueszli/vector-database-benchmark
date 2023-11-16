"""PauliBasisChange Class"""
from functools import partial, reduce
from typing import Callable, List, Optional, Tuple, Union, cast
import numpy as np
from qiskit import QuantumCircuit
from qiskit.opflow.converters.converter_base import ConverterBase
from qiskit.opflow.list_ops.composed_op import ComposedOp
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.operator_globals import H, I, S
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.opflow.state_fns.operator_state_fn import OperatorStateFn
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.quantum_info import Pauli
from qiskit.utils.deprecation import deprecate_func

class PauliBasisChange(ConverterBase):
    """
    Deprecated: Converter for changing Paulis into other bases. By default, the diagonal basis
    composed only of Pauli {Z, I}^n is used as the destination basis to which to convert.
    Meaning, if a Pauli containing X or Y terms is passed in, which cannot be
    sampled or evolved natively on some Quantum hardware, the Pauli can be replaced by a
    composition of a change of basis circuit and a Pauli composed of only Z
    and I terms (diagonal), which can be evolved or sampled natively on the Quantum
    hardware.

    The replacement function determines how the ``PauliOps`` should be replaced by their computed
    change-of-basis ``CircuitOps`` and destination ``PauliOps``. Several convenient out-of-the-box
    replacement functions have been added as static methods, such as ``measurement_replacement_fn``.

    This class uses the typical basis change method found in most Quantum Computing textbooks
    (such as on page 210 of Nielsen and Chuang's, "Quantum Computation and Quantum Information",
    ISBN: 978-1-107-00217-3), which involves diagonalizing the single-qubit Paulis with H and S†
    gates, mapping the eigenvectors of the diagonalized origin Pauli to the diagonalized
    destination Pauli using CNOTS, and then de-diagonalizing any single qubit Paulis to their
    non-diagonal destination values. Many other methods are possible, as well as variations on
    this method, such as the placement of the CNOT chains.
    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, destination_basis: Optional[Union[Pauli, PauliOp]]=None, traverse: bool=True, replacement_fn: Optional[Callable]=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Args:\n            destination_basis: The Pauli into the basis of which the operators\n                will be converted. If None is specified, the destination basis will be the\n                diagonal ({I, Z}^n) basis requiring only single qubit rotations.\n            traverse: If true and the operator passed into convert contains sub-Operators,\n                such as ListOp, traverse the Operator and apply the conversion to every\n                applicable sub-operator within it.\n            replacement_fn: A function specifying what to do with the basis-change\n                ``CircuitOp`` and destination ``PauliOp`` when converting an Operator and\n                replacing converted values. By default, this will be\n\n                    1) For StateFns (or Measurements): replacing the StateFn with\n                       ComposedOp(StateFn(d), c) where c is the conversion circuit and d is the\n                       destination Pauli, so the overall beginning and ending operators are\n                       equivalent.\n\n                    2) For non-StateFn Operators: replacing the origin p with c·d·c†, where c\n                       is the conversion circuit and d is the destination, so the overall\n                       beginning and ending operators are equivalent.\n\n        '
        super().__init__()
        if destination_basis is not None:
            self.destination = destination_basis
        else:
            self._destination = None
        self._traverse = traverse
        self._replacement_fn = replacement_fn or PauliBasisChange.operator_replacement_fn

    @property
    def destination(self) -> Optional[PauliOp]:
        if False:
            i = 10
            return i + 15
        '\n        The destination ``PauliOp``, or ``None`` if using the default destination, the diagonal\n        basis.\n        '
        return self._destination

    @destination.setter
    def destination(self, dest: Union[Pauli, PauliOp]) -> None:
        if False:
            print('Hello World!')
        '\n        The destination ``PauliOp``, or ``None`` if using the default destination, the diagonal\n        basis.\n        '
        if isinstance(dest, Pauli):
            dest = PauliOp(dest)
        if not isinstance(dest, PauliOp):
            raise TypeError(f'PauliBasisChange can only convert into Pauli bases, not {type(dest)}.')
        self._destination = dest

    def convert(self, operator: OperatorBase) -> OperatorBase:
        if False:
            print('Hello World!')
        '\n        Given a ``PauliOp``, or an Operator containing ``PauliOps`` if ``_traverse`` is True,\n        converts each Pauli into the basis specified by self._destination and a\n        basis-change-circuit, calls ``replacement_fn`` with these two Operators, and replaces\n        the ``PauliOps`` with the output of ``replacement_fn``. For example, for the built-in\n        ``operator_replacement_fn`` below, each PauliOp p will be replaced by the composition\n        of the basis-change Clifford ``CircuitOp`` c with the destination PauliOp d and c†,\n        such that p = c·d·c†, up to global phase.\n\n        Args:\n            operator: The Operator to convert.\n\n        Returns:\n            The converted Operator.\n\n        '
        if isinstance(operator, OperatorStateFn) and isinstance(operator.primitive, PauliSumOp) and (operator.primitive.grouping_type == 'TPB'):
            primitive = operator.primitive.primitive.copy()
            origin_x = reduce(np.logical_or, primitive.paulis.x)
            origin_z = reduce(np.logical_or, primitive.paulis.z)
            origin_pauli = Pauli((origin_z, origin_x))
            (cob_instr_op, _) = self.get_cob_circuit(origin_pauli)
            primitive.paulis.z = np.logical_or(primitive.paulis.x, primitive.paulis.z)
            primitive.paulis.x = False
            primitive.paulis.phase = 0
            dest_pauli_sum_op = PauliSumOp(primitive, coeff=operator.coeff, grouping_type='TPB')
            return self._replacement_fn(cob_instr_op, dest_pauli_sum_op)
        if isinstance(operator, OperatorStateFn) and isinstance(operator.primitive, SummedOp) and all((isinstance(op, PauliSumOp) and op.grouping_type == 'TPB' for op in operator.primitive.oplist)):
            sf_list: List[OperatorBase] = [StateFn(op, is_measurement=operator.is_measurement) for op in operator.primitive.oplist]
            listop_of_statefns = SummedOp(oplist=sf_list, coeff=operator.coeff)
            return listop_of_statefns.traverse(self.convert)
        if isinstance(operator, OperatorStateFn) and isinstance(operator.primitive, PauliSumOp):
            operator = OperatorStateFn(operator.primitive.to_pauli_op(), coeff=operator.coeff, is_measurement=operator.is_measurement)
        if isinstance(operator, PauliSumOp):
            operator = operator.to_pauli_op()
        if isinstance(operator, (Pauli, PauliOp)):
            (cob_instr_op, dest_pauli_op) = self.get_cob_circuit(operator)
            return self._replacement_fn(cob_instr_op, dest_pauli_op)
        if isinstance(operator, StateFn) and 'Pauli' in operator.primitive_strings():
            if isinstance(operator.primitive, PauliOp):
                (cob_instr_op, dest_pauli_op) = self.get_cob_circuit(operator.primitive)
                return self._replacement_fn(cob_instr_op, dest_pauli_op * operator.coeff)
            elif operator.primitive.distributive:
                if operator.primitive.abelian:
                    origin_pauli = self.get_tpb_pauli(operator.primitive)
                    (cob_instr_op, _) = self.get_cob_circuit(origin_pauli)
                    diag_ops: List[OperatorBase] = [self.get_diagonal_pauli_op(op) for op in operator.primitive.oplist]
                    dest_pauli_op = operator.primitive.__class__(diag_ops, coeff=operator.coeff, abelian=True)
                    return self._replacement_fn(cob_instr_op, dest_pauli_op)
                else:
                    sf_list = [StateFn(op, is_measurement=operator.is_measurement) for op in operator.primitive.oplist]
                    listop_of_statefns = operator.primitive.__class__(oplist=sf_list, coeff=operator.coeff)
                    return listop_of_statefns.traverse(self.convert)
        elif isinstance(operator, ListOp) and self._traverse and ('Pauli' in operator.primitive_strings()):
            if operator.abelian:
                origin_pauli = self.get_tpb_pauli(operator)
                (cob_instr_op, _) = self.get_cob_circuit(origin_pauli)
                oplist = cast(List[PauliOp], operator.oplist)
                diag_ops = [self.get_diagonal_pauli_op(op) for op in oplist]
                dest_list_op = operator.__class__(diag_ops, coeff=operator.coeff, abelian=True)
                return self._replacement_fn(cob_instr_op, dest_list_op)
            else:
                return operator.traverse(self.convert)
        return operator

    @staticmethod
    def measurement_replacement_fn(cob_instr_op: PrimitiveOp, dest_pauli_op: Union[PauliOp, PauliSumOp, ListOp]) -> OperatorBase:
        if False:
            while True:
                i = 10
        '\n        A built-in convenience replacement function which produces measurements\n        isomorphic to an ``OperatorStateFn`` measurement holding the origin ``PauliOp``.\n\n        Args:\n            cob_instr_op: The basis-change ``CircuitOp``.\n            dest_pauli_op: The destination Pauli type operator.\n\n        Returns:\n            The ``~StateFn @ CircuitOp`` composition equivalent to a measurement by the original\n            ``PauliOp``.\n        '
        return ComposedOp([StateFn(dest_pauli_op, is_measurement=True), cob_instr_op])

    @staticmethod
    def statefn_replacement_fn(cob_instr_op: PrimitiveOp, dest_pauli_op: Union[PauliOp, PauliSumOp, ListOp]) -> OperatorBase:
        if False:
            print('Hello World!')
        '\n        A built-in convenience replacement function which produces state functions\n        isomorphic to an ``OperatorStateFn`` state function holding the origin ``PauliOp``.\n\n        Args:\n            cob_instr_op: The basis-change ``CircuitOp``.\n            dest_pauli_op: The destination Pauli type operator.\n\n        Returns:\n            The ``~CircuitOp @ StateFn`` composition equivalent to a state function defined by the\n            original ``PauliOp``.\n        '
        return ComposedOp([cob_instr_op.adjoint(), StateFn(dest_pauli_op)])

    @staticmethod
    def operator_replacement_fn(cob_instr_op: PrimitiveOp, dest_pauli_op: Union[PauliOp, PauliSumOp, ListOp]) -> OperatorBase:
        if False:
            for i in range(10):
                print('nop')
        '\n        A built-in convenience replacement function which produces Operators\n        isomorphic to the origin ``PauliOp``.\n\n        Args:\n            cob_instr_op: The basis-change ``CircuitOp``.\n            dest_pauli_op: The destination ``PauliOp``.\n\n        Returns:\n            The ``~CircuitOp @ PauliOp @ CircuitOp`` composition isomorphic to the\n            original ``PauliOp``.\n        '
        return ComposedOp([cob_instr_op.adjoint(), dest_pauli_op, cob_instr_op])

    def get_tpb_pauli(self, list_op: ListOp) -> Pauli:
        if False:
            return 10
        '\n        Gets the Pauli (not ``PauliOp``!) whose diagonalizing single-qubit rotations is a\n        superset of the diagonalizing single-qubit rotations for each of the Paulis in\n        ``list_op``. TPB stands for `Tensor Product Basis`.\n\n        Args:\n             list_op: the :class:`ListOp` whose TPB Pauli to return.\n\n        Returns:\n             The TBP Pauli.\n\n        '
        oplist = cast(List[PauliOp], list_op.oplist)
        origin_z = reduce(np.logical_or, [p_op.primitive.z for p_op in oplist])
        origin_x = reduce(np.logical_or, [p_op.primitive.x for p_op in oplist])
        return Pauli((origin_z, origin_x))

    def get_diagonal_pauli_op(self, pauli_op: PauliOp) -> PauliOp:
        if False:
            return 10
        'Get the diagonal ``PualiOp`` to which ``pauli_op`` could be rotated with only\n        single-qubit operations.\n\n        Args:\n            pauli_op: The ``PauliOp`` whose diagonal to compute.\n\n        Returns:\n            The diagonal ``PauliOp``.\n        '
        return PauliOp(Pauli((np.logical_or(pauli_op.primitive.z, pauli_op.primitive.x), [False] * pauli_op.num_qubits)), coeff=pauli_op.coeff)

    def get_diagonalizing_clifford(self, pauli: Union[Pauli, PauliOp]) -> OperatorBase:
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a ``CircuitOp`` with only single-qubit gates which takes the eigenvectors\n        of ``pauli`` to eigenvectors composed only of \\|0⟩ and \\|1⟩ tensor products. Equivalently,\n        finds the basis-change circuit to take ``pauli`` to a diagonal ``PauliOp`` composed only\n        of Z and I tensor products.\n\n        Note, underlying Pauli bits are in Qiskit endianness, so we need to reverse before we\n        begin composing with Operator flow.\n\n        Args:\n            pauli: the ``Pauli`` or ``PauliOp`` to whose diagonalizing circuit to compute.\n\n        Returns:\n            The diagonalizing ``CircuitOp``.\n\n        '
        if isinstance(pauli, PauliOp):
            pauli = pauli.primitive
        tensorall = cast(Callable[[List[PrimitiveOp]], PrimitiveOp], partial(reduce, lambda x, y: x.tensor(y)))
        y_to_x_origin = tensorall([S if has_y else I for has_y in reversed(np.logical_and(pauli.x, pauli.z))]).adjoint()
        x_to_z_origin = tensorall([H if has_x else I for has_x in reversed(pauli.x)])
        return x_to_z_origin.compose(y_to_x_origin)

    def pad_paulis_to_equal_length(self, pauli_op1: PauliOp, pauli_op2: PauliOp) -> Tuple[PauliOp, PauliOp]:
        if False:
            print('Hello World!')
        '\n        If ``pauli_op1`` and ``pauli_op2`` do not act over the same number of qubits, pad\n        identities to the end of the shorter of the two so they are of equal length. Padding is\n        applied to the end of the Paulis. Note that the Terra represents Paulis in big-endian\n        order, so this will appear as padding to the beginning of the Pauli x and z bit arrays.\n\n        Args:\n            pauli_op1: A pauli_op to possibly pad.\n            pauli_op2: A pauli_op to possibly pad.\n\n        Returns:\n            A tuple containing the padded PauliOps.\n\n        '
        num_qubits = max(pauli_op1.num_qubits, pauli_op2.num_qubits)
        (pauli_1, pauli_2) = (pauli_op1.primitive, pauli_op2.primitive)
        if not len(pauli_1.z) == num_qubits:
            missing_qubits = num_qubits - len(pauli_1.z)
            pauli_1 = Pauli(([False] * missing_qubits + pauli_1.z.tolist(), [False] * missing_qubits + pauli_1.x.tolist()))
        if not len(pauli_2.z) == num_qubits:
            missing_qubits = num_qubits - len(pauli_2.z)
            pauli_2 = Pauli(([False] * missing_qubits + pauli_2.z.tolist(), [False] * missing_qubits + pauli_2.x.tolist()))
        return (PauliOp(pauli_1, coeff=pauli_op1.coeff), PauliOp(pauli_2, coeff=pauli_op2.coeff))

    def construct_cnot_chain(self, diag_pauli_op1: PauliOp, diag_pauli_op2: PauliOp) -> PrimitiveOp:
        if False:
            print('Hello World!')
        "\n        Construct a ``CircuitOp`` (or ``PauliOp`` if equal to the identity) which takes the\n        eigenvectors of ``diag_pauli_op1`` to the eigenvectors of ``diag_pauli_op2``,\n        assuming both are diagonal (or performing this operation on their diagonalized Paulis\n        implicitly if not). This works by the insight that the eigenvalue of a diagonal Pauli's\n        eigenvector is equal to or -1 if the parity is 1 and 1 if the parity is 0, or\n        1 - (2 * parity). Therefore, using CNOTs, we can write the parity of diag_pauli_op1's\n        significant bits onto some qubit, and then write out that parity onto diag_pauli_op2's\n        significant bits.\n\n        Args:\n            diag_pauli_op1: The origin ``PauliOp``.\n            diag_pauli_op2: The destination ``PauliOp``.\n\n        Return:\n            The ``PrimitiveOp`` performs the mapping.\n        "
        pauli_1 = diag_pauli_op1.primitive if isinstance(diag_pauli_op1, PauliOp) else diag_pauli_op1
        pauli_2 = diag_pauli_op2.primitive if isinstance(diag_pauli_op2, PauliOp) else diag_pauli_op2
        origin_sig_bits = np.logical_or(pauli_1.z, pauli_1.x)
        destination_sig_bits = np.logical_or(pauli_2.z, pauli_2.x)
        num_qubits = max(len(pauli_1.z), len(pauli_2.z))
        sig_equal_sig_bits = np.logical_and(origin_sig_bits, destination_sig_bits)
        non_equal_sig_bits = np.logical_not(origin_sig_bits == destination_sig_bits)
        if not any(non_equal_sig_bits):
            return I ^ num_qubits
        sig_in_origin_only_indices = np.extract(np.logical_and(non_equal_sig_bits, origin_sig_bits), np.arange(num_qubits))
        sig_in_dest_only_indices = np.extract(np.logical_and(non_equal_sig_bits, destination_sig_bits), np.arange(num_qubits))
        if len(sig_in_origin_only_indices) > 0 and len(sig_in_dest_only_indices) > 0:
            origin_anchor_bit = min(sig_in_origin_only_indices)
            dest_anchor_bit = min(sig_in_dest_only_indices)
        else:
            origin_anchor_bit = min(np.extract(sig_equal_sig_bits, np.arange(num_qubits)))
            dest_anchor_bit = origin_anchor_bit
        cnots = QuantumCircuit(num_qubits)
        for i in sig_in_origin_only_indices:
            if not i == origin_anchor_bit:
                cnots.cx(i, origin_anchor_bit)
        if not origin_anchor_bit == dest_anchor_bit:
            cnots.swap(origin_anchor_bit, dest_anchor_bit)
        cnots.id(0)
        for i in sig_in_dest_only_indices:
            if not i == dest_anchor_bit:
                cnots.cx(i, dest_anchor_bit)
        return PrimitiveOp(cnots)

    def get_cob_circuit(self, origin: Union[Pauli, PauliOp]) -> Tuple[PrimitiveOp, PauliOp]:
        if False:
            print('Hello World!')
        '\n        Construct an Operator which maps the +1 and -1 eigenvectors\n        of the origin Pauli to the +1 and -1 eigenvectors of the destination Pauli. It does so by\n\n        1) converting any \\|i+⟩ or \\|i+⟩ eigenvector bits in the origin to\n           \\|+⟩ and \\|-⟩ with S†s, then\n\n        2) converting any \\|+⟩ or \\|+⟩ eigenvector bits in the converted origin to\n           \\|0⟩ and \\|1⟩ with Hs, then\n\n        3) writing the parity of the significant (Z-measured, rather than I)\n           bits in the origin to a single\n           "origin anchor bit," using cnots, which will hold the parity of these bits,\n\n        4) swapping the parity of the pauli anchor bit into a destination anchor bit using\n           a swap gate (only if they are different, if there are any bits which are significant\n           in both origin and dest, we set both anchors to one of these bits to avoid a swap).\n\n        5) writing the parity of the destination anchor bit into the other significant bits\n           of the destination,\n\n        6) converting the \\|0⟩ and \\|1⟩ significant eigenvector bits to \\|+⟩ and \\|-⟩ eigenvector\n           bits in the destination where the destination demands it\n           (e.g. pauli.x == true for a bit), using Hs 8) converting the \\|+⟩ and \\|-⟩\n           significant eigenvector bits to \\|i+⟩ and \\|i-⟩ eigenvector bits in the\n           destination where the destination demands it\n           (e.g. pauli.x == true and pauli.z == true for a bit), using Ss\n\n        Args:\n            origin: The ``Pauli`` or ``PauliOp`` to map.\n\n        Returns:\n            A tuple of a ``PrimitiveOp`` which equals the basis change mapping and a ``PauliOp``\n            which equals the destination basis.\n\n        Raises:\n            TypeError: Attempting to convert from non-Pauli origin.\n            ValueError: Attempting to change a non-identity Pauli to an identity Pauli, or vice\n                versa.\n\n        '
        if isinstance(origin, Pauli):
            origin = PauliOp(origin)
        if not isinstance(origin, PauliOp):
            raise TypeError(f'PauliBasisChange can only convert Pauli-based OpPrimitives, not {type(origin)}')
        destination = self.destination or self.get_diagonal_pauli_op(origin)
        (origin, destination) = self.pad_paulis_to_equal_length(origin, destination)
        origin_sig_bits = np.logical_or(origin.primitive.x, origin.primitive.z)
        destination_sig_bits = np.logical_or(destination.primitive.x, destination.primitive.z)
        if not any(origin_sig_bits) or not any(destination_sig_bits):
            if not (any(origin_sig_bits) or any(destination_sig_bits)):
                return (I ^ origin.num_qubits, destination)
            else:
                raise ValueError('Cannot change to or from a fully Identity Pauli.')
        cob_instruction = self.get_diagonalizing_clifford(origin)
        cob_instruction = self.construct_cnot_chain(origin, destination).compose(cob_instruction)
        dest_diagonlizing_clifford = self.get_diagonalizing_clifford(destination).adjoint()
        cob_instruction = dest_diagonlizing_clifford.compose(cob_instruction)
        return (cast(PrimitiveOp, cob_instruction), destination)