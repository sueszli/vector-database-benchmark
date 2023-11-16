"""
Multiple-Controlled U3 gate. Not using ancillary qubits.
"""
from math import pi
from typing import Optional, Union, Tuple, List
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.library.standard_gates.x import MCXGate
from qiskit.circuit.library.standard_gates.u3 import _generate_gray_code
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.exceptions import QiskitError

def _apply_cu(circuit, theta, phi, lam, control, target, use_basis_gates=True):
    if False:
        for i in range(10):
            print('nop')
    if use_basis_gates:
        circuit.p((lam + phi) / 2, [control])
        circuit.p((lam - phi) / 2, [target])
        circuit.cx(control, target)
        circuit.u(-theta / 2, 0, -(phi + lam) / 2, [target])
        circuit.cx(control, target)
        circuit.u(theta / 2, phi, 0, [target])
    else:
        circuit.cu(theta, phi, lam, 0, control, target)

def _apply_mcu_graycode(circuit, theta, phi, lam, ctls, tgt, use_basis_gates):
    if False:
        print('Hello World!')
    'Apply multi-controlled u gate from ctls to tgt using graycode\n    pattern with single-step angles theta, phi, lam.'
    n = len(ctls)
    gray_code = _generate_gray_code(n)
    last_pattern = None
    for pattern in gray_code:
        if '1' not in pattern:
            continue
        if last_pattern is None:
            last_pattern = pattern
        lm_pos = list(pattern).index('1')
        comp = [i != j for (i, j) in zip(pattern, last_pattern)]
        if True in comp:
            pos = comp.index(True)
        else:
            pos = None
        if pos is not None:
            if pos != lm_pos:
                circuit.cx(ctls[pos], ctls[lm_pos])
            else:
                indices = [i for (i, x) in enumerate(pattern) if x == '1']
                for idx in indices[1:]:
                    circuit.cx(ctls[idx], ctls[lm_pos])
        if pattern.count('1') % 2 == 0:
            _apply_cu(circuit, -theta, -lam, -phi, ctls[lm_pos], tgt, use_basis_gates=use_basis_gates)
        else:
            _apply_cu(circuit, theta, phi, lam, ctls[lm_pos], tgt, use_basis_gates=use_basis_gates)
        last_pattern = pattern

def _mcsu2_real_diagonal(unitary: np.ndarray, num_controls: int, ctrl_state: Optional[str]=None, use_basis_gates: bool=False) -> QuantumCircuit:
    if False:
        i = 10
        return i + 15
    '\n    Return a multi-controlled SU(2) gate [1]_ with a real main diagonal or secondary diagonal.\n\n    Args:\n        unitary: SU(2) unitary matrix with one real diagonal.\n        num_controls: The number of control qubits.\n        ctrl_state: The state on which the SU(2) operation is controlled. Defaults to all\n            control qubits being in state 1.\n        use_basis_gates: If ``True``, use ``[p, u, cx]`` gates to implement the decomposition.\n\n    Returns:\n        A :class:`.QuantumCircuit` implementing the multi-controlled SU(2) gate.\n\n    Raises:\n        QiskitError: If the input matrix is invalid.\n\n    References:\n\n        .. [1]: R. Vale et al. Decomposition of Multi-controlled Special Unitary Single-Qubit Gates\n            `arXiv:2302.06377 (2023) <https://arxiv.org/abs/2302.06377>`__\n\n    '
    from .x import MCXVChain
    from qiskit.circuit.library.generalized_gates import UnitaryGate
    from qiskit.quantum_info.operators.predicates import is_unitary_matrix
    from qiskit.compiler import transpile
    if unitary.shape != (2, 2):
        raise QiskitError(f'The unitary must be a 2x2 matrix, but has shape {unitary.shape}.')
    if not is_unitary_matrix(unitary):
        raise QiskitError(f'The unitary in must be an unitary matrix, but is {unitary}.')
    is_main_diag_real = np.isclose(unitary[0, 0].imag, 0.0) and np.isclose(unitary[1, 1].imag, 0.0)
    is_secondary_diag_real = np.isclose(unitary[0, 1].imag, 0.0) and np.isclose(unitary[1, 0].imag, 0.0)
    if not is_main_diag_real and (not is_secondary_diag_real):
        raise QiskitError('The unitary must have one real diagonal.')
    if is_secondary_diag_real:
        x = unitary[0, 1]
        z = unitary[1, 1]
    else:
        x = -unitary[0, 1].real
        z = unitary[1, 1] - unitary[0, 1].imag * 1j
    if np.isclose(z, -1):
        s_op = [[1.0, 0.0], [0.0, 1j]]
    else:
        alpha_r = np.sqrt((np.sqrt((z.real + 1.0) / 2.0) + 1.0) / 2.0)
        alpha_i = z.imag / (2.0 * np.sqrt((z.real + 1.0) * (np.sqrt((z.real + 1.0) / 2.0) + 1.0)))
        alpha = alpha_r + 1j * alpha_i
        beta = x / (2.0 * np.sqrt((z.real + 1.0) * (np.sqrt((z.real + 1.0) / 2.0) + 1.0)))
        s_op = np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]])
    s_gate = UnitaryGate(s_op)
    k_1 = int(np.ceil(num_controls / 2.0))
    k_2 = int(np.floor(num_controls / 2.0))
    ctrl_state_k_1 = None
    ctrl_state_k_2 = None
    if ctrl_state is not None:
        str_ctrl_state = f'{ctrl_state:0{num_controls}b}'
        ctrl_state_k_1 = str_ctrl_state[::-1][:k_1][::-1]
        ctrl_state_k_2 = str_ctrl_state[::-1][k_1:][::-1]
    circuit = QuantumCircuit(num_controls + 1, name='MCSU2')
    controls = list(range(num_controls))
    target = num_controls
    if not is_secondary_diag_real:
        circuit.h(target)
    mcx_1 = MCXVChain(num_ctrl_qubits=k_1, dirty_ancillas=True, ctrl_state=ctrl_state_k_1)
    circuit.append(mcx_1, controls[:k_1] + [target] + controls[k_1:2 * k_1 - 2])
    circuit.append(s_gate, [target])
    mcx_2 = MCXVChain(num_ctrl_qubits=k_2, dirty_ancillas=True, ctrl_state=ctrl_state_k_2)
    circuit.append(mcx_2.inverse(), controls[k_1:] + [target] + controls[k_1 - k_2 + 2:k_1])
    circuit.append(s_gate.inverse(), [target])
    mcx_3 = MCXVChain(num_ctrl_qubits=k_1, dirty_ancillas=True, ctrl_state=ctrl_state_k_1)
    circuit.append(mcx_3, controls[:k_1] + [target] + controls[k_1:2 * k_1 - 2])
    circuit.append(s_gate, [target])
    mcx_4 = MCXVChain(num_ctrl_qubits=k_2, dirty_ancillas=True, ctrl_state=ctrl_state_k_2)
    circuit.append(mcx_4, controls[k_1:] + [target] + controls[k_1 - k_2 + 2:k_1])
    circuit.append(s_gate.inverse(), [target])
    if not is_secondary_diag_real:
        circuit.h(target)
    if use_basis_gates:
        circuit = transpile(circuit, basis_gates=['p', 'u', 'cx'])
    return circuit

def mcrx(self, theta: ParameterValueType, q_controls: Union[QuantumRegister, List[Qubit]], q_target: Qubit, use_basis_gates: bool=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Apply Multiple-Controlled X rotation gate\n\n    Args:\n        self (QuantumCircuit): The QuantumCircuit object to apply the mcrx gate on.\n        theta (float): angle theta\n        q_controls (QuantumRegister or list(Qubit)): The list of control qubits\n        q_target (Qubit): The target qubit\n        use_basis_gates (bool): use p, u, cx\n\n    Raises:\n        QiskitError: parameter errors\n    '
    from .rx import RXGate
    control_qubits = self.qbit_argument_conversion(q_controls)
    target_qubit = self.qbit_argument_conversion(q_target)
    if len(target_qubit) != 1:
        raise QiskitError('The mcrz gate needs a single qubit as target.')
    all_qubits = control_qubits + target_qubit
    target_qubit = target_qubit[0]
    self._check_dups(all_qubits)
    n_c = len(control_qubits)
    if n_c == 1:
        _apply_cu(self, theta, -pi / 2, pi / 2, control_qubits[0], target_qubit, use_basis_gates=use_basis_gates)
    elif n_c < 4:
        theta_step = theta * (1 / 2 ** (n_c - 1))
        _apply_mcu_graycode(self, theta_step, -pi / 2, pi / 2, control_qubits, target_qubit, use_basis_gates=use_basis_gates)
    else:
        cgate = _mcsu2_real_diagonal(RXGate(theta).to_matrix(), num_controls=len(control_qubits), use_basis_gates=use_basis_gates)
        self.compose(cgate, control_qubits + [target_qubit], inplace=True)

def mcry(self, theta: ParameterValueType, q_controls: Union[QuantumRegister, List[Qubit]], q_target: Qubit, q_ancillae: Optional[Union[QuantumRegister, Tuple[QuantumRegister, int]]]=None, mode: str=None, use_basis_gates=False):
    if False:
        return 10
    '\n    Apply Multiple-Controlled Y rotation gate\n\n    Args:\n        self (QuantumCircuit): The QuantumCircuit object to apply the mcry gate on.\n        theta (float): angle theta\n        q_controls (list(Qubit)): The list of control qubits\n        q_target (Qubit): The target qubit\n        q_ancillae (QuantumRegister or tuple(QuantumRegister, int)): The list of ancillary qubits.\n        mode (string): The implementation mode to use\n        use_basis_gates (bool): use p, u, cx\n\n    Raises:\n        QiskitError: parameter errors\n    '
    from .ry import RYGate
    control_qubits = self.qbit_argument_conversion(q_controls)
    target_qubit = self.qbit_argument_conversion(q_target)
    if len(target_qubit) != 1:
        raise QiskitError('The mcrz gate needs a single qubit as target.')
    ancillary_qubits = [] if q_ancillae is None else self.qbit_argument_conversion(q_ancillae)
    all_qubits = control_qubits + target_qubit + ancillary_qubits
    target_qubit = target_qubit[0]
    self._check_dups(all_qubits)
    if mode is None:
        additional_vchain = MCXGate.get_num_ancilla_qubits(len(control_qubits), 'v-chain')
        if len(ancillary_qubits) >= additional_vchain:
            mode = 'basic'
        else:
            mode = 'noancilla'
    if mode == 'basic':
        self.ry(theta / 2, q_target)
        self.mcx(q_controls, q_target, q_ancillae, mode='v-chain')
        self.ry(-theta / 2, q_target)
        self.mcx(q_controls, q_target, q_ancillae, mode='v-chain')
    elif mode == 'noancilla':
        n_c = len(control_qubits)
        if n_c == 1:
            _apply_cu(self, theta, 0, 0, control_qubits[0], target_qubit, use_basis_gates=use_basis_gates)
        elif n_c < 4:
            theta_step = theta * (1 / 2 ** (n_c - 1))
            _apply_mcu_graycode(self, theta_step, 0, 0, control_qubits, target_qubit, use_basis_gates=use_basis_gates)
        else:
            cgate = _mcsu2_real_diagonal(RYGate(theta).to_matrix(), num_controls=len(control_qubits), use_basis_gates=use_basis_gates)
            self.compose(cgate, control_qubits + [target_qubit], inplace=True)
    else:
        raise QiskitError(f'Unrecognized mode for building MCRY circuit: {mode}.')

def mcrz(self, lam: ParameterValueType, q_controls: Union[QuantumRegister, List[Qubit]], q_target: Qubit, use_basis_gates: bool=False):
    if False:
        print('Hello World!')
    '\n    Apply Multiple-Controlled Z rotation gate\n\n    Args:\n        self (QuantumCircuit): The QuantumCircuit object to apply the mcrz gate on.\n        lam (float): angle lambda\n        q_controls (list(Qubit)): The list of control qubits\n        q_target (Qubit): The target qubit\n        use_basis_gates (bool): use p, u, cx\n\n    Raises:\n        QiskitError: parameter errors\n    '
    from .rz import CRZGate, RZGate
    control_qubits = self.qbit_argument_conversion(q_controls)
    target_qubit = self.qbit_argument_conversion(q_target)
    if len(target_qubit) != 1:
        raise QiskitError('The mcrz gate needs a single qubit as target.')
    all_qubits = control_qubits + target_qubit
    target_qubit = target_qubit[0]
    self._check_dups(all_qubits)
    n_c = len(control_qubits)
    if n_c == 1:
        if use_basis_gates:
            self.u(0, 0, lam / 2, target_qubit)
            self.cx(control_qubits[0], target_qubit)
            self.u(0, 0, -lam / 2, target_qubit)
            self.cx(control_qubits[0], target_qubit)
        else:
            self.append(CRZGate(lam), control_qubits + [target_qubit])
    else:
        cgate = _mcsu2_real_diagonal(RZGate(lam).to_matrix(), num_controls=len(control_qubits), use_basis_gates=use_basis_gates)
        self.compose(cgate, control_qubits + [target_qubit], inplace=True)
QuantumCircuit.mcrx = mcrx
QuantumCircuit.mcry = mcry
QuantumCircuit.mcrz = mcrz