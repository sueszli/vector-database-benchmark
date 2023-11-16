"""Prepare a quantum state from the state where all qubits are 0."""
from typing import Union, Optional
import math
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates.x import CXGate, XGate
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.circuit.library.standard_gates.s import SGate, SdgGate
from qiskit.circuit.library.standard_gates.ry import RYGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info.states.statevector import Statevector
_EPS = 1e-10

class StatePreparation(Gate):
    """Complex amplitude state preparation.

    Class that implements the (complex amplitude) state preparation of some
    flexible collection of qubit registers.
    """

    def __init__(self, params: Union[str, list, int, Statevector], num_qubits: Optional[int]=None, inverse: bool=False, label: Optional[str]=None, normalize: bool=False):
        if False:
            while True:
                i = 10
        "\n        Args:\n            params:\n                * Statevector: Statevector to initialize to.\n                * list: vector of complex amplitudes to initialize to.\n                * string: labels of basis states of the Pauli eigenstates Z, X, Y. See\n                  :meth:`.Statevector.from_label`.\n                  Notice the order of the labels is reversed with respect to the qubit index to\n                  be applied to. Example label '01' initializes the qubit zero to :math:`|1\\rangle`\n                  and the qubit one to :math:`|0\\rangle`.\n                * int: an integer that is used as a bitmap indicating which qubits to initialize\n                  to :math:`|1\\rangle`. Example: setting params to 5 would initialize qubit 0 and qubit 2\n                  to :math:`|1\\rangle` and qubit 1 to :math:`|0\\rangle`.\n            num_qubits: This parameter is only used if params is an int. Indicates the total\n                number of qubits in the `initialize` call. Example: `initialize` covers 5 qubits\n                and params is 3. This allows qubits 0 and 1 to be initialized to :math:`|1\\rangle`\n                and the remaining 3 qubits to be initialized to :math:`|0\\rangle`.\n            inverse: if True, the inverse state is constructed.\n            label: An optional label for the gate\n            normalize (bool): Whether to normalize an input array to a unit vector.\n\n        Raises:\n            QiskitError: ``num_qubits`` parameter used when ``params`` is not an integer\n\n        When a Statevector argument is passed the state is prepared using a recursive\n        initialization algorithm, including optimizations, from [1], as well\n        as some additional optimizations including removing zero rotations and double cnots.\n\n        **References:**\n        [1] Shende, Bullock, Markov. Synthesis of Quantum Logic Circuits (2004)\n        [`https://arxiv.org/abs/quant-ph/0406176v5`]\n\n        "
        self._params_arg = params
        self._inverse = inverse
        self._name = 'state_preparation_dg' if self._inverse else 'state_preparation'
        if label is None:
            self._label = 'State Preparation Dg' if self._inverse else 'State Preparation'
        else:
            self._label = f'{label} Dg' if self._inverse else label
        if isinstance(params, Statevector):
            params = params.data
        if not isinstance(params, int) and num_qubits is not None:
            raise QiskitError('The num_qubits parameter to StatePreparation should only be used when params is an integer')
        self._from_label = isinstance(params, str)
        self._from_int = isinstance(params, int)
        if not self._from_label and (not self._from_int):
            norm = np.linalg.norm(params)
            if normalize:
                params = np.array(params, dtype=np.complex128) / norm
            elif not math.isclose(norm, 1.0, abs_tol=_EPS):
                raise QiskitError(f'Sum of amplitudes-squared is not 1, but {norm}.')
        num_qubits = self._get_num_qubits(num_qubits, params)
        params = [params] if isinstance(params, int) else params
        super().__init__(self._name, num_qubits, params, label=self._label)

    def _define(self):
        if False:
            for i in range(10):
                print('nop')
        if self._from_label:
            self.definition = self._define_from_label()
        elif self._from_int:
            self.definition = self._define_from_int()
        else:
            self.definition = self._define_synthesis()

    def _define_from_label(self):
        if False:
            print('Hello World!')
        q = QuantumRegister(self.num_qubits, 'q')
        initialize_circuit = QuantumCircuit(q, name='init_def')
        for (qubit, param) in enumerate(reversed(self.params)):
            if param == '1':
                initialize_circuit.append(XGate(), [q[qubit]])
            elif param == '+':
                initialize_circuit.append(HGate(), [q[qubit]])
            elif param == '-':
                initialize_circuit.append(XGate(), [q[qubit]])
                initialize_circuit.append(HGate(), [q[qubit]])
            elif param == 'r':
                initialize_circuit.append(HGate(), [q[qubit]])
                initialize_circuit.append(SGate(), [q[qubit]])
            elif param == 'l':
                initialize_circuit.append(HGate(), [q[qubit]])
                initialize_circuit.append(SdgGate(), [q[qubit]])
        if self._inverse:
            initialize_circuit = initialize_circuit.inverse()
        return initialize_circuit

    def _define_from_int(self):
        if False:
            print('Hello World!')
        q = QuantumRegister(self.num_qubits, 'q')
        initialize_circuit = QuantumCircuit(q, name='init_def')
        intstr = f'{int(np.real(self.params[0])):0{self.num_qubits}b}'[::-1]
        if len(intstr) > self.num_qubits:
            raise QiskitError('StatePreparation integer has %s bits, but this exceeds the number of qubits in the circuit, %s.' % (len(intstr), self.num_qubits))
        for (qubit, bit) in enumerate(intstr):
            if bit == '1':
                initialize_circuit.append(XGate(), [q[qubit]])
        return initialize_circuit

    def _define_synthesis(self):
        if False:
            while True:
                i = 10
        'Calculate a subcircuit that implements this initialization\n\n        Implements a recursive initialization algorithm, including optimizations,\n        from "Synthesis of Quantum Logic Circuits" Shende, Bullock, Markov\n        https://arxiv.org/abs/quant-ph/0406176v5\n\n        Additionally implements some extra optimizations: remove zero rotations and\n        double cnots.\n        '
        disentangling_circuit = self._gates_to_uncompute()
        if self._inverse is False:
            initialize_instr = disentangling_circuit.to_instruction().inverse()
        else:
            initialize_instr = disentangling_circuit.to_instruction()
        q = QuantumRegister(self.num_qubits, 'q')
        initialize_circuit = QuantumCircuit(q, name='init_def')
        initialize_circuit.append(initialize_instr, q[:])
        return initialize_circuit

    def _get_num_qubits(self, num_qubits, params):
        if False:
            for i in range(10):
                print('nop')
        'Get number of qubits needed for state preparation'
        if isinstance(params, str):
            num_qubits = len(params)
        elif isinstance(params, int):
            if num_qubits is None:
                num_qubits = int(math.log2(params)) + 1
        else:
            num_qubits = math.log2(len(params))
            if num_qubits == 0 or not num_qubits.is_integer():
                raise QiskitError('Desired statevector length not a positive power of 2.')
            num_qubits = int(num_qubits)
        return num_qubits

    def inverse(self):
        if False:
            while True:
                i = 10
        'Return inverted StatePreparation'
        label = None if self._label in ('State Preparation', 'State Preparation Dg') else self._label
        return StatePreparation(self._params_arg, inverse=not self._inverse, label=label)

    def broadcast_arguments(self, qargs, cargs):
        if False:
            while True:
                i = 10
        flat_qargs = [qarg for sublist in qargs for qarg in sublist]
        if self.num_qubits != len(flat_qargs):
            raise QiskitError('StatePreparation parameter vector has %d elements, therefore expects %s qubits. However, %s were provided.' % (2 ** self.num_qubits, self.num_qubits, len(flat_qargs)))
        yield (flat_qargs, [])

    def validate_parameter(self, parameter):
        if False:
            i = 10
            return i + 15
        'StatePreparation instruction parameter can be str, int, float, and complex.'
        if isinstance(parameter, str):
            if parameter in ['0', '1', '+', '-', 'l', 'r']:
                return parameter
            raise CircuitError('invalid param label {} for instruction {}. Label should be 0, 1, +, -, l, or r '.format(type(parameter), self.name))
        if isinstance(parameter, (int, float, complex)):
            return complex(parameter)
        elif isinstance(parameter, np.number):
            return complex(parameter.item())
        else:
            raise CircuitError(f'invalid param type {type(parameter)} for instruction  {self.name}')

    def _return_repeat(self, exponent: float) -> 'Gate':
        if False:
            for i in range(10):
                print('nop')
        return Gate(name=f'{self.name}*{exponent}', num_qubits=self.num_qubits, params=[])

    def _gates_to_uncompute(self):
        if False:
            while True:
                i = 10
        'Call to create a circuit with gates that take the desired vector to zero.\n\n        Returns:\n            QuantumCircuit: circuit to take self.params vector to :math:`|{00\\ldots0}\\rangle`\n        '
        q = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(q, name='disentangler')
        remaining_param = self.params
        for i in range(self.num_qubits):
            (remaining_param, thetas, phis) = StatePreparation._rotations_to_disentangle(remaining_param)
            add_last_cnot = True
            if np.linalg.norm(phis) != 0 and np.linalg.norm(thetas) != 0:
                add_last_cnot = False
            if np.linalg.norm(phis) != 0:
                rz_mult = self._multiplex(RZGate, phis, last_cnot=add_last_cnot)
                circuit.append(rz_mult.to_instruction(), q[i:self.num_qubits])
            if np.linalg.norm(thetas) != 0:
                ry_mult = self._multiplex(RYGate, thetas, last_cnot=add_last_cnot)
                circuit.append(ry_mult.to_instruction().reverse_ops(), q[i:self.num_qubits])
        circuit.global_phase -= np.angle(sum(remaining_param))
        return circuit

    @staticmethod
    def _rotations_to_disentangle(local_param):
        if False:
            return 10
        '\n        Static internal method to work out Ry and Rz rotation angles used\n        to disentangle the LSB qubit.\n        These rotations make up the block diagonal matrix U (i.e. multiplexor)\n        that disentangles the LSB.\n\n        [[Ry(theta_1).Rz(phi_1)  0   .   .   0],\n        [0         Ry(theta_2).Rz(phi_2) .  0],\n                                    .\n                                        .\n        0         0           Ry(theta_2^n).Rz(phi_2^n)]]\n        '
        remaining_vector = []
        thetas = []
        phis = []
        param_len = len(local_param)
        for i in range(param_len // 2):
            (remains, add_theta, add_phi) = StatePreparation._bloch_angles(local_param[2 * i:2 * (i + 1)])
            remaining_vector.append(remains)
            thetas.append(-add_theta)
            phis.append(-add_phi)
        return (remaining_vector, thetas, phis)

    @staticmethod
    def _bloch_angles(pair_of_complex):
        if False:
            while True:
                i = 10
        '\n        Static internal method to work out rotation to create the passed-in\n        qubit from the zero vector.\n        '
        [a_complex, b_complex] = pair_of_complex
        a_complex = complex(a_complex)
        b_complex = complex(b_complex)
        mag_a = abs(a_complex)
        final_r = np.sqrt(mag_a ** 2 + np.absolute(b_complex) ** 2)
        if final_r < _EPS:
            theta = 0
            phi = 0
            final_r = 0
            final_t = 0
        else:
            theta = 2 * np.arccos(mag_a / final_r)
            a_arg = np.angle(a_complex)
            b_arg = np.angle(b_complex)
            final_t = a_arg + b_arg
            phi = b_arg - a_arg
        return (final_r * np.exp(1j * final_t / 2), theta, phi)

    def _multiplex(self, target_gate, list_of_angles, last_cnot=True):
        if False:
            return 10
        '\n        Return a recursive implementation of a multiplexor circuit,\n        where each instruction itself has a decomposition based on\n        smaller multiplexors.\n\n        The LSB is the multiplexor "data" and the other bits are multiplexor "select".\n\n        Args:\n            target_gate (Gate): Ry or Rz gate to apply to target qubit, multiplexed\n                over all other "select" qubits\n            list_of_angles (list[float]): list of rotation angles to apply Ry and Rz\n            last_cnot (bool): add the last cnot if last_cnot = True\n\n        Returns:\n            DAGCircuit: the circuit implementing the multiplexor\'s action\n        '
        list_len = len(list_of_angles)
        local_num_qubits = int(math.log2(list_len)) + 1
        q = QuantumRegister(local_num_qubits)
        circuit = QuantumCircuit(q, name='multiplex' + str(local_num_qubits))
        lsb = q[0]
        msb = q[local_num_qubits - 1]
        if local_num_qubits == 1:
            circuit.append(target_gate(list_of_angles[0]), [q[0]])
            return circuit
        angle_weight = np.kron([[0.5, 0.5], [0.5, -0.5]], np.identity(2 ** (local_num_qubits - 2)))
        list_of_angles = angle_weight.dot(np.array(list_of_angles)).tolist()
        multiplex_1 = self._multiplex(target_gate, list_of_angles[0:list_len // 2], False)
        circuit.append(multiplex_1.to_instruction(), q[0:-1])
        circuit.append(CXGate(), [msb, lsb])
        multiplex_2 = self._multiplex(target_gate, list_of_angles[list_len // 2:], False)
        if list_len > 1:
            circuit.append(multiplex_2.to_instruction().reverse_ops(), q[0:-1])
        else:
            circuit.append(multiplex_2.to_instruction(), q[0:-1])
        if last_cnot:
            circuit.append(CXGate(), [msb, lsb])
        return circuit

def prepare_state(self, state, qubits=None, label=None, normalize=False):
    if False:
        while True:
            i = 10
    "Prepare qubits in a specific state.\n\n    This class implements a state preparing unitary. Unlike\n    :class:`qiskit.extensions.Initialize` it does not reset the qubits first.\n\n    Args:\n        state (str or list or int or Statevector):\n            * Statevector: Statevector to initialize to.\n            * str: labels of basis states of the Pauli eigenstates Z, X, Y. See\n              :meth:`.Statevector.from_label`. Notice the order of the labels is reversed with respect\n              to the qubit index to be applied to. Example label '01' initializes the qubit zero to\n              :math:`|1\\rangle` and the qubit one to :math:`|0\\rangle`.\n            * list: vector of complex amplitudes to initialize to.\n            * int: an integer that is used as a bitmap indicating which qubits to initialize\n              to :math:`|1\\rangle`. Example: setting params to 5 would initialize qubit 0 and qubit 2\n              to :math:`|1\\rangle` and qubit 1 to :math:`|0\\rangle`.\n\n        qubits (QuantumRegister or Qubit or int):\n            * QuantumRegister: A list of qubits to be initialized [Default: None].\n            * Qubit: Single qubit to be initialized [Default: None].\n            * int: Index of qubit to be initialized [Default: None].\n            * list: Indexes of qubits to be initialized [Default: None].\n        label (str): An optional label for the gate\n        normalize (bool): Whether to normalize an input array to a unit vector.\n\n    Returns:\n        qiskit.circuit.Instruction: a handle to the instruction that was just initialized\n\n    Examples:\n        Prepare a qubit in the state :math:`(|0\\rangle - |1\\rangle) / \\sqrt{2}`.\n\n        .. code-block::\n\n           import numpy as np\n           from qiskit import QuantumCircuit\n\n           circuit = QuantumCircuit(1)\n           circuit.prepare_state([1/np.sqrt(2), -1/np.sqrt(2)], 0)\n           circuit.draw()\n\n        output:\n\n        .. parsed-literal::\n\n                 ┌─────────────────────────────────────┐\n            q_0: ┤ State Preparation(0.70711,-0.70711) ├\n                 └─────────────────────────────────────┘\n\n\n        Prepare from a string two qubits in the state :math:`|10\\rangle`.\n        The order of the labels is reversed with respect to qubit index.\n        More information about labels for basis states are in\n        :meth:`.Statevector.from_label`.\n\n        .. code-block::\n\n            import numpy as np\n            from qiskit import QuantumCircuit\n\n            circuit = QuantumCircuit(2)\n            circuit.prepare_state('01', circuit.qubits)\n            circuit.draw()\n\n        output:\n\n        .. parsed-literal::\n\n                 ┌─────────────────────────┐\n            q_0: ┤0                        ├\n                 │  State Preparation(0,1) │\n            q_1: ┤1                        ├\n                 └─────────────────────────┘\n\n\n        Initialize two qubits from an array of complex amplitudes\n        .. code-block::\n\n            import numpy as np\n            from qiskit import QuantumCircuit\n\n            circuit = QuantumCircuit(2)\n            circuit.prepare_state([0, 1/np.sqrt(2), -1.j/np.sqrt(2), 0], circuit.qubits)\n            circuit.draw()\n\n        output:\n\n        .. parsed-literal::\n\n                 ┌───────────────────────────────────────────┐\n            q_0: ┤0                                          ├\n                 │  State Preparation(0,0.70711,-0.70711j,0) │\n            q_1: ┤1                                          ├\n                 └───────────────────────────────────────────┘\n    "
    if qubits is None:
        qubits = self.qubits
    elif isinstance(qubits, (int, np.integer, slice, Qubit)):
        qubits = [qubits]
    num_qubits = len(qubits) if isinstance(state, int) else None
    return self.append(StatePreparation(state, num_qubits, label=label, normalize=normalize), qubits)
QuantumCircuit.prepare_state = prepare_state