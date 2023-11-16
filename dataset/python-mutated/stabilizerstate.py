"""
Stabilizer state class.
"""
from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import Clifford, Pauli, PauliList
from qiskit.quantum_info.operators.symplectic.clifford_circuits import _append_x
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.circuit import QuantumCircuit, Instruction

class StabilizerState(QuantumState):
    """StabilizerState class.
    Stabilizer simulator using the convention from reference [1].
    Based on the internal class :class:`~qiskit.quantum_info.Clifford`.

    .. code-block::

        from qiskit import QuantumCircuit
        from qiskit.quantum_info import StabilizerState, Pauli

        # Bell state generation circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        stab = StabilizerState(qc)

        # Print the StabilizerState
        print(stab)

        # Calculate the StabilizerState measurement probabilities dictionary
        print (stab.probabilities_dict())

        # Calculate expectation value of the StabilizerState
        print (stab.expectation_value(Pauli('ZZ')))

    .. parsed-literal::

        StabilizerState(StabilizerTable: ['+XX', '+ZZ'])
        {'00': 0.5, '11': 0.5}
        1

    References:
        1. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
    """

    def __init__(self, data: StabilizerState | Clifford | Pauli | QuantumCircuit | Instruction, validate: bool=True):
        if False:
            i = 10
            return i + 15
        'Initialize a StabilizerState object.\n\n        Args:\n            data (StabilizerState or Clifford or Pauli or QuantumCircuit or\n                  qiskit.circuit.Instruction):\n                Data from which the stabilizer state can be constructed.\n            validate (boolean): validate that the stabilizer state data is\n                a valid Clifford.\n        '
        if isinstance(data, StabilizerState):
            self._data = data._data
        elif isinstance(data, Pauli):
            self._data = Clifford(data.to_instruction())
        else:
            self._data = Clifford(data, validate)
        super().__init__(op_shape=OpShape.auto(num_qubits_r=self._data.num_qubits, num_qubits_l=0))

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return (self._data.stab == other._data.stab).all()

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'StabilizerState({self._data.stabilizer})'

    @property
    def clifford(self):
        if False:
            return 10
        'Return StabilizerState Clifford data'
        return self._data

    def is_valid(self, atol=None, rtol=None):
        if False:
            return 10
        'Return True if a valid StabilizerState.'
        return self._data.is_unitary()

    def _add(self, other):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError(f'{type(self)} does not support addition')

    def _multiply(self, other):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError(f'{type(self)} does not support scalar multiplication')

    def trace(self) -> float:
        if False:
            return 10
        'Return the trace of the stabilizer state as a density matrix,\n        which equals to 1, since it is always a pure state.\n\n        Returns:\n            float: the trace (should equal 1).\n\n        Raises:\n            QiskitError: if input is not a StabilizerState.\n        '
        if not self.is_valid():
            raise QiskitError('StabilizerState is not a valid quantum state.')
        return 1.0

    def purity(self) -> float:
        if False:
            return 10
        'Return the purity of the quantum state,\n        which equals to 1, since it is always a pure state.\n\n        Returns:\n            float: the purity (should equal 1).\n\n        Raises:\n            QiskitError: if input is not a StabilizerState.\n        '
        if not self.is_valid():
            raise QiskitError('StabilizerState is not a valid quantum state.')
        return 1.0

    def to_operator(self) -> Operator:
        if False:
            while True:
                i = 10
        'Convert state to matrix operator class'
        return Clifford(self.clifford).to_operator()

    def conjugate(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the conjugate of the operator.'
        ret = self.copy()
        ret._data = ret._data.conjugate()
        return ret

    def tensor(self, other: StabilizerState) -> StabilizerState:
        if False:
            return 10
        'Return the tensor product stabilizer state self ⊗ other.\n\n        Args:\n            other (StabilizerState): a stabilizer state object.\n\n        Returns:\n            StabilizerState: the tensor product operator self ⊗ other.\n\n        Raises:\n            QiskitError: if other is not a StabilizerState.\n        '
        if not isinstance(other, StabilizerState):
            other = StabilizerState(other)
        ret = self.copy()
        ret._data = self.clifford.tensor(other.clifford)
        return ret

    def expand(self, other: StabilizerState) -> StabilizerState:
        if False:
            for i in range(10):
                print('nop')
        'Return the tensor product stabilizer state other ⊗ self.\n\n        Args:\n            other (StabilizerState): a stabilizer state object.\n\n        Returns:\n            StabilizerState: the tensor product operator other ⊗ self.\n\n        Raises:\n            QiskitError: if other is not a StabilizerState.\n        '
        if not isinstance(other, StabilizerState):
            other = StabilizerState(other)
        ret = self.copy()
        ret._data = self.clifford.expand(other.clifford)
        return ret

    def evolve(self, other: Clifford | QuantumCircuit | Instruction, qargs: list | None=None) -> StabilizerState:
        if False:
            while True:
                i = 10
        'Evolve a stabilizer state by a Clifford operator.\n\n        Args:\n            other (Clifford or QuantumCircuit or qiskit.circuit.Instruction):\n                The Clifford operator to evolve by.\n            qargs (list): a list of stabilizer subsystem positions to apply the operator on.\n\n        Returns:\n            StabilizerState: the output stabilizer state.\n\n        Raises:\n            QiskitError: if other is not a StabilizerState.\n            QiskitError: if the operator dimension does not match the\n                         specified StabilizerState subsystem dimensions.\n        '
        if not isinstance(other, StabilizerState):
            other = StabilizerState(other)
        ret = self.copy()
        ret._data = self.clifford.compose(other.clifford, qargs=qargs)
        return ret

    def expectation_value(self, oper: Pauli, qargs: None | list=None) -> complex:
        if False:
            i = 10
            return i + 15
        'Compute the expectation value of a Pauli operator.\n\n        Args:\n            oper (Pauli): a Pauli operator to evaluate expval.\n            qargs (None or list): subsystems to apply the operator on.\n\n        Returns:\n            complex: the expectation value (only 0 or 1 or -1 or i or -i).\n\n        Raises:\n            QiskitError: if oper is not a Pauli operator.\n        '
        if not isinstance(oper, Pauli):
            raise QiskitError('Operator for expectation value is not a Pauli operator.')
        num_qubits = self.clifford.num_qubits
        if qargs is None:
            qubits = range(num_qubits)
        else:
            qubits = qargs
        pauli = Pauli(num_qubits * 'I')
        phase = 0
        pauli_phase = (-1j) ** oper.phase if oper.phase else 1
        for (pos, qubit) in enumerate(qubits):
            pauli.x[qubit] = oper.x[pos]
            pauli.z[qubit] = oper.z[pos]
            phase += pauli.x[qubit] & pauli.z[qubit]
        for p in range(num_qubits):
            num_anti = 0
            num_anti += np.count_nonzero(pauli.z & self.clifford.stab_x[p])
            num_anti += np.count_nonzero(pauli.x & self.clifford.stab_z[p])
            if num_anti % 2 == 1:
                return 0
        pauli_z = pauli.z.copy()
        for p in range(num_qubits):
            num_anti = 0
            num_anti += np.count_nonzero(pauli.z & self.clifford.destab_x[p])
            num_anti += np.count_nonzero(pauli.x & self.clifford.destab_z[p])
            if num_anti % 2 == 0:
                continue
            phase += 2 * self.clifford.stab_phase[p]
            phase += np.count_nonzero(self.clifford.stab_z[p] & self.clifford.stab_x[p])
            phase += 2 * np.count_nonzero(pauli_z & self.clifford.stab_x[p])
            pauli_z = pauli_z ^ self.clifford.stab_z[p]
        if phase % 4 != 0:
            return -pauli_phase
        return pauli_phase

    def equiv(self, other: StabilizerState) -> bool:
        if False:
            print('Hello World!')
        'Return True if the two generating sets generate the same stabilizer group.\n\n        Args:\n            other (StabilizerState): another StabilizerState.\n\n        Returns:\n            bool: True if other has a generating set that generates the same StabilizerState.\n        '
        if not isinstance(other, StabilizerState):
            try:
                other = StabilizerState(other)
            except QiskitError:
                return False
        num_qubits = self.num_qubits
        if other.num_qubits != num_qubits:
            return False
        pauli_orig = PauliList.from_symplectic(self._data.stab_z, self._data.stab_x, 2 * self._data.stab_phase)
        pauli_other = PauliList.from_symplectic(other._data.stab_z, other._data.stab_x, 2 * other._data.stab_phase)
        if not np.all([pauli.commutes(pauli_other) for pauli in pauli_orig]):
            return False
        for i in range(num_qubits):
            exp_val = self.expectation_value(pauli_other[i])
            if exp_val != 1:
                return False
        return True

    def probabilities(self, qargs: None | list=None, decimals: None | int=None) -> np.ndarray:
        if False:
            print('Hello World!')
        'Return the subsystem measurement probability vector.\n\n        Measurement probabilities are with respect to measurement in the\n        computation (diagonal) basis.\n\n        Args:\n            qargs (None or list): subsystems to return probabilities for,\n                if None return for all subsystems (Default: None).\n            decimals (None or int): the number of decimal places to round\n                values. If None no rounding is done (Default: None).\n\n        Returns:\n            np.array: The Numpy vector array of probabilities.\n        '
        probs_dict = self.probabilities_dict(qargs, decimals)
        if qargs is None:
            qargs = range(self.clifford.num_qubits)
        probs = np.zeros(2 ** len(qargs))
        for (key, value) in probs_dict.items():
            place = int(key, 2)
            probs[place] = value
        return probs

    def probabilities_dict(self, qargs: None | list=None, decimals: None | int=None) -> dict:
        if False:
            print('Hello World!')
        'Return the subsystem measurement probability dictionary.\n\n        Measurement probabilities are with respect to measurement in the\n        computation (diagonal) basis.\n\n        This dictionary representation uses a Ket-like notation where the\n        dictionary keys are qudit strings for the subsystem basis vectors.\n        If any subsystem has a dimension greater than 10 comma delimiters are\n        inserted between integers so that subsystems can be distinguished.\n\n        Args:\n            qargs (None or list): subsystems to return probabilities for,\n                if None return for all subsystems (Default: None).\n            decimals (None or int): the number of decimal places to round\n                values. If None no rounding is done (Default: None).\n\n        Returns:\n            dict: The measurement probabilities in dict (ket) form.\n        '
        if qargs is None:
            qubits = range(self.clifford.num_qubits)
        else:
            qubits = qargs
        outcome = ['X'] * len(qubits)
        outcome_prob = 1.0
        probs = {}
        self._get_probablities(qubits, outcome, outcome_prob, probs)
        if decimals is not None:
            for (key, value) in probs.items():
                probs[key] = round(value, decimals)
        return probs

    def reset(self, qargs: list | None=None) -> StabilizerState:
        if False:
            return 10
        'Reset state or subsystems to the 0-state.\n\n        Args:\n            qargs (list or None): subsystems to reset, if None all\n                                  subsystems will be reset to their 0-state\n                                  (Default: None).\n\n        Returns:\n            StabilizerState: the reset state.\n\n        Additional Information:\n            If all subsystems are reset this will return the ground state\n            on all subsystems. If only some subsystems are reset this\n            function will perform a measurement on those subsystems and\n            evolve the subsystems so that the collapsed post-measurement\n            states are rotated to the 0-state. The RNG seed for this\n            sampling can be set using the :meth:`seed` method.\n        '
        if qargs is None:
            return StabilizerState(Clifford(np.eye(2 * self.clifford.num_qubits)))
        randbits = self._rng.integers(2, size=len(qargs))
        ret = self.copy()
        for (bit, qubit) in enumerate(qargs):
            outcome = ret._measure_and_update(qubit, randbits[bit])
            if outcome == 1:
                _append_x(ret.clifford, qubit)
        return ret

    def measure(self, qargs: list | None=None) -> tuple:
        if False:
            while True:
                i = 10
        'Measure subsystems and return outcome and post-measure state.\n\n        Note that this function uses the QuantumStates internal random\n        number generator for sampling the measurement outcome. The RNG\n        seed can be set using the :meth:`seed` method.\n\n        Args:\n            qargs (list or None): subsystems to sample measurements for,\n                                  if None sample measurement of all\n                                  subsystems (Default: None).\n\n        Returns:\n            tuple: the pair ``(outcome, state)`` where ``outcome`` is the\n                   measurement outcome string label, and ``state`` is the\n                   collapsed post-measurement stabilizer state for the\n                   corresponding outcome.\n        '
        if qargs is None:
            qargs = range(self.clifford.num_qubits)
        randbits = self._rng.integers(2, size=len(qargs))
        ret = self.copy()
        outcome = ''
        for (bit, qubit) in enumerate(qargs):
            outcome = str(ret._measure_and_update(qubit, randbits[bit])) + outcome
        return (outcome, ret)

    def sample_memory(self, shots: int, qargs: None | list=None) -> np.ndarray:
        if False:
            return 10
        'Sample a list of qubit measurement outcomes in the computational basis.\n\n        Args:\n            shots (int): number of samples to generate.\n            qargs (None or list): subsystems to sample measurements for,\n                                if None sample measurement of all\n                                subsystems (Default: None).\n\n        Returns:\n            np.array: list of sampled counts if the order sampled.\n\n        Additional Information:\n\n            This function implements the measurement :meth:`measure` method.\n\n            The seed for random number generator used for sampling can be\n            set to a fixed value by using the stats :meth:`seed` method.\n        '
        memory = []
        for _ in range(shots):
            stab = self.copy()
            memory.append(stab.measure(qargs)[0])
        return memory

    def _measure_and_update(self, qubit, randbit):
        if False:
            for i in range(10):
                print('nop')
        'Measure a single qubit and return outcome and post-measure state.\n\n        Note that this function uses the QuantumStates internal random\n        number generator for sampling the measurement outcome. The RNG\n        seed can be set using the :meth:`seed` method.\n\n        Note that stabilizer state measurements only have three probabilities:\n        (p0, p1) = (0.5, 0.5), (1, 0), or (0, 1)\n        The random case happens if there is a row anti-commuting with Z[qubit]\n        '
        num_qubits = self.clifford.num_qubits
        clifford = self.clifford
        stab_x = self.clifford.stab_x
        z_anticommuting = np.any(stab_x[:, qubit])
        if z_anticommuting == 0:
            aux_pauli = Pauli(num_qubits * 'I')
            for i in range(num_qubits):
                if clifford.x[i][qubit]:
                    aux_pauli = self._rowsum_deterministic(clifford, aux_pauli, i + num_qubits)
            outcome = aux_pauli.phase
            return outcome
        else:
            outcome = randbit
            p_qubit = np.min(np.nonzero(stab_x[:, qubit]))
            p_qubit += num_qubits
            for i in range(2 * num_qubits):
                if clifford.x[i][qubit] and i != p_qubit and (i != p_qubit - num_qubits):
                    self._rowsum_nondeterministic(clifford, i, p_qubit)
            clifford.destab[p_qubit - num_qubits] = clifford.stab[p_qubit - num_qubits].copy()
            clifford.x[p_qubit] = np.zeros(num_qubits)
            clifford.z[p_qubit] = np.zeros(num_qubits)
            clifford.z[p_qubit][qubit] = True
            clifford.phase[p_qubit] = outcome
            return outcome

    @staticmethod
    def _phase_exponent(x1, z1, x2, z2):
        if False:
            print('Hello World!')
        'Exponent g of i such that Pauli(x1,z1) * Pauli(x2,z2) = i^g Pauli(x1+x2,z1+z2)'
        phase = (x2 * z1 * (1 + 2 * z2 + 2 * x1) - x1 * z2 * (1 + 2 * z1 + 2 * x2)) % 4
        if phase < 0:
            phase += 4
        if phase == 2:
            raise QiskitError('Invalid rowsum phase exponent in measurement calculation.')
        return phase

    @staticmethod
    def _rowsum(accum_pauli, accum_phase, row_pauli, row_phase):
        if False:
            while True:
                i = 10
        'Aaronson-Gottesman rowsum helper function'
        newr = 2 * row_phase + 2 * accum_phase
        for qubit in range(row_pauli.num_qubits):
            newr += StabilizerState._phase_exponent(row_pauli.x[qubit], row_pauli.z[qubit], accum_pauli.x[qubit], accum_pauli.z[qubit])
        newr %= 4
        if (newr != 0) & (newr != 2):
            raise QiskitError('Invalid rowsum in measurement calculation.')
        accum_phase = int(newr == 2)
        accum_pauli.x ^= row_pauli.x
        accum_pauli.z ^= row_pauli.z
        return (accum_pauli, accum_phase)

    @staticmethod
    def _rowsum_nondeterministic(clifford, accum, row):
        if False:
            i = 10
            return i + 15
        'Updating StabilizerState Clifford in the\n        non-deterministic rowsum calculation.\n        row and accum are rows in the StabilizerState Clifford.'
        row_phase = clifford.phase[row]
        accum_phase = clifford.phase[accum]
        z = clifford.z
        x = clifford.x
        row_pauli = Pauli((z[row], x[row]))
        accum_pauli = Pauli((z[accum], x[accum]))
        (accum_pauli, accum_phase) = StabilizerState._rowsum(accum_pauli, accum_phase, row_pauli, row_phase)
        clifford.phase[accum] = accum_phase
        x[accum] = accum_pauli.x
        z[accum] = accum_pauli.z

    @staticmethod
    def _rowsum_deterministic(clifford, aux_pauli, row):
        if False:
            print('Hello World!')
        'Updating an auxilary Pauli aux_pauli in the\n        deterministic rowsum calculation.\n        The StabilizerState itself is not updated.'
        row_phase = clifford.phase[row]
        accum_phase = aux_pauli.phase
        accum_pauli = aux_pauli
        row_pauli = Pauli((clifford.z[row], clifford.x[row]))
        (accum_pauli, accum_phase) = StabilizerState._rowsum(accum_pauli, accum_phase, row_pauli, row_phase)
        aux_pauli = accum_pauli
        aux_pauli.phase = accum_phase
        return aux_pauli

    def _get_probablities(self, qubits, outcome, outcome_prob, probs):
        if False:
            while True:
                i = 10
        'Recursive helper function for calculating the probabilities'
        qubit_for_branching = -1
        ret = self.copy()
        for i in range(len(qubits)):
            qubit = qubits[len(qubits) - i - 1]
            if outcome[i] == 'X':
                is_deterministic = not any(ret.clifford.stab_x[:, qubit])
                if is_deterministic:
                    single_qubit_outcome = ret._measure_and_update(qubit, 0)
                    if single_qubit_outcome:
                        outcome[i] = '1'
                    else:
                        outcome[i] = '0'
                else:
                    qubit_for_branching = i
        if qubit_for_branching == -1:
            str_outcome = ''.join(outcome)
            probs[str_outcome] = outcome_prob
            return
        for single_qubit_outcome in range(0, 2):
            new_outcome = outcome.copy()
            if single_qubit_outcome:
                new_outcome[qubit_for_branching] = '1'
            else:
                new_outcome[qubit_for_branching] = '0'
            stab_cpy = ret.copy()
            stab_cpy._measure_and_update(qubits[len(qubits) - qubit_for_branching - 1], single_qubit_outcome)
            stab_cpy._get_probablities(qubits, new_outcome, 0.5 * outcome_prob, probs)