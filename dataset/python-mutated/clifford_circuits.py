"""
Circuit simulation for the Clifford class.
"""
from __future__ import annotations
import numpy as np
from qiskit.circuit import Barrier, Delay, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError

def _append_circuit(clifford, circuit, qargs=None):
    if False:
        return 10
    'Update Clifford inplace by applying a Clifford circuit.\n\n    Args:\n        clifford (Clifford): The Clifford to update.\n        circuit (QuantumCircuit): The circuit to apply.\n        qargs (list or None): The qubits to apply circuit to.\n\n    Returns:\n        Clifford: the updated Clifford.\n\n    Raises:\n        QiskitError: if input circuit cannot be decomposed into Clifford operations.\n    '
    if qargs is None:
        qargs = list(range(clifford.num_qubits))
    for instruction in circuit:
        if instruction.clbits:
            raise QiskitError(f'Cannot apply Instruction with classical bits: {instruction.operation.name}')
        new_qubits = [qargs[circuit.find_bit(bit).index] for bit in instruction.qubits]
        clifford = _append_operation(clifford, instruction.operation, new_qubits)
    return clifford

def _append_operation(clifford, operation, qargs=None):
    if False:
        return 10
    'Update Clifford inplace by applying a Clifford operation.\n\n    Args:\n        clifford (Clifford): The Clifford to update.\n        operation (Instruction or Clifford or str): The operation or composite operation to apply.\n        qargs (list or None): The qubits to apply operation to.\n\n    Returns:\n        Clifford: the updated Clifford.\n\n    Raises:\n        QiskitError: if input operation cannot be converted into Clifford operations.\n    '
    if isinstance(operation, (Barrier, Delay)):
        return clifford
    if qargs is None:
        qargs = list(range(clifford.num_qubits))
    gate = operation
    if isinstance(gate, str):
        if gate not in _BASIS_1Q and gate not in _BASIS_2Q:
            raise QiskitError(f'Invalid Clifford gate name string {gate}')
        name = gate
    else:
        name = gate.name
        if getattr(gate, 'condition', None) is not None:
            raise QiskitError('Conditional gate is not a valid Clifford operation.')
    if name in _NON_CLIFFORD:
        raise QiskitError(f'Cannot update Clifford with non-Clifford gate {name}')
    if name in _BASIS_1Q:
        if len(qargs) != 1:
            raise QiskitError('Invalid qubits for 1-qubit gate.')
        return _BASIS_1Q[name](clifford, qargs[0])
    if name in _BASIS_2Q:
        if len(qargs) != 2:
            raise QiskitError('Invalid qubits for 2-qubit gate.')
        return _BASIS_2Q[name](clifford, qargs[0], qargs[1])
    if isinstance(gate, Gate) and name == 'u' and (len(qargs) == 1):
        try:
            (theta, phi, lambd) = tuple((_n_half_pis(par) for par in gate.params))
        except ValueError as err:
            raise QiskitError('U gate angles must be multiples of pi/2 to be a Clifford') from err
        if theta == 0:
            clifford = _append_rz(clifford, qargs[0], lambd + phi)
        elif theta == 1:
            clifford = _append_rz(clifford, qargs[0], lambd - 2)
            clifford = _append_h(clifford, qargs[0])
            clifford = _append_rz(clifford, qargs[0], phi)
        elif theta == 2:
            clifford = _append_rz(clifford, qargs[0], lambd - 1)
            clifford = _append_x(clifford, qargs[0])
            clifford = _append_rz(clifford, qargs[0], phi + 1)
        elif theta == 3:
            clifford = _append_rz(clifford, qargs[0], lambd)
            clifford = _append_h(clifford, qargs[0])
            clifford = _append_rz(clifford, qargs[0], phi + 2)
        return clifford
    from qiskit.quantum_info import Clifford
    if isinstance(gate, Clifford):
        composed_clifford = clifford.compose(gate, qargs=qargs, front=False)
        clifford.tableau = composed_clifford.tableau
        return clifford
    from qiskit.circuit.library import LinearFunction
    if isinstance(gate, LinearFunction):
        gate_as_clifford = Clifford.from_linear_function(gate)
        composed_clifford = clifford.compose(gate_as_clifford, qargs=qargs, front=False)
        clifford.tableau = composed_clifford.tableau
        return clifford
    from qiskit.circuit.library import PermutationGate
    if isinstance(gate, PermutationGate):
        gate_as_clifford = Clifford.from_permutation(gate)
        composed_clifford = clifford.compose(gate_as_clifford, qargs=qargs, front=False)
        clifford.tableau = composed_clifford.tableau
        return clifford
    if gate.definition is not None:
        try:
            return _append_circuit(clifford.copy(), gate.definition, qargs)
        except QiskitError:
            pass
    if isinstance(gate, Gate) and len(qargs) <= 3:
        try:
            matrix = gate.to_matrix()
            gate_cliff = Clifford.from_matrix(matrix)
            return _append_operation(clifford, gate_cliff, qargs=qargs)
        except TypeError as err:
            raise QiskitError(f'Cannot apply {gate.name} gate with unbounded parameters') from err
        except CircuitError as err:
            raise QiskitError(f'Cannot apply {gate.name} gate without to_matrix defined') from err
        except QiskitError as err:
            raise QiskitError(f'Cannot apply non-Clifford gate: {gate.name}') from err
    raise QiskitError(f'Cannot apply {gate}')

def _n_half_pis(param) -> int:
    if False:
        for i in range(10):
            print('nop')
    try:
        param = float(param)
        epsilon = (abs(param) + 0.5 * 1e-10) % (np.pi / 2)
        if epsilon > 1e-10:
            raise ValueError(f'{param} is not to a multiple of pi/2')
        multiple = int(np.round(param / (np.pi / 2)))
        return multiple % 4
    except TypeError as err:
        raise ValueError(f'{param} is not bounded') from err

def _append_rz(clifford, qubit, multiple):
    if False:
        for i in range(10):
            print('nop')
    'Apply an Rz gate to a Clifford.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        qubit (int): gate qubit index.\n        multiple (int): z-rotation angle in a multiple of pi/2\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    if multiple % 4 == 1:
        return _append_s(clifford, qubit)
    if multiple % 4 == 2:
        return _append_z(clifford, qubit)
    if multiple % 4 == 3:
        return _append_sdg(clifford, qubit)
    return clifford

def _append_i(clifford, qubit):
    if False:
        while True:
            i = 10
    'Apply an I gate to a Clifford.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        qubit (int): gate qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    return clifford

def _append_x(clifford, qubit):
    if False:
        for i in range(10):
            print('nop')
    'Apply an X gate to a Clifford.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        qubit (int): gate qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    clifford.phase ^= clifford.z[:, qubit]
    return clifford

def _append_y(clifford, qubit):
    if False:
        while True:
            i = 10
    'Apply a Y gate to a Clifford.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        qubit (int): gate qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= x ^ z
    return clifford

def _append_z(clifford, qubit):
    if False:
        for i in range(10):
            print('nop')
    'Apply an Z gate to a Clifford.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        qubit (int): gate qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    clifford.phase ^= clifford.x[:, qubit]
    return clifford

def _append_h(clifford, qubit):
    if False:
        for i in range(10):
            print('nop')
    'Apply a H gate to a Clifford.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        qubit (int): gate qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= x & z
    tmp = x.copy()
    x[:] = z
    z[:] = tmp
    return clifford

def _append_s(clifford, qubit):
    if False:
        while True:
            i = 10
    'Apply an S gate to a Clifford.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        qubit (int): gate qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= x & z
    z ^= x
    return clifford

def _append_sdg(clifford, qubit):
    if False:
        for i in range(10):
            print('nop')
    'Apply an Sdg gate to a Clifford.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        qubit (int): gate qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= x & ~z
    z ^= x
    return clifford

def _append_sx(clifford, qubit):
    if False:
        i = 10
        return i + 15
    'Apply an SX gate to a Clifford.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        qubit (int): gate qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= ~x & z
    x ^= z
    return clifford

def _append_sxdg(clifford, qubit):
    if False:
        print('Hello World!')
    'Apply an SXdg gate to a Clifford.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        qubit (int): gate qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= x & z
    x ^= z
    return clifford

def _append_v(clifford, qubit):
    if False:
        i = 10
        return i + 15
    'Apply a V gate to a Clifford.\n\n    This is equivalent to an Sdg gate followed by a H gate.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        qubit (int): gate qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    tmp = x.copy()
    x ^= z
    z[:] = tmp
    return clifford

def _append_w(clifford, qubit):
    if False:
        print('Hello World!')
    'Apply a W gate to a Clifford.\n\n    This is equivalent to two V gates.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        qubit (int): gate qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    tmp = z.copy()
    z ^= x
    x[:] = tmp
    return clifford

def _append_cx(clifford, control, target):
    if False:
        for i in range(10):
            print('nop')
    'Apply a CX gate to a Clifford.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        control (int): gate control qubit index.\n        target (int): gate target qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    x0 = clifford.x[:, control]
    z0 = clifford.z[:, control]
    x1 = clifford.x[:, target]
    z1 = clifford.z[:, target]
    clifford.phase ^= (x1 ^ z0 ^ True) & z1 & x0
    x1 ^= x0
    z0 ^= z1
    return clifford

def _append_cz(clifford, control, target):
    if False:
        i = 10
        return i + 15
    'Apply a CZ gate to a Clifford.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        control (int): gate control qubit index.\n        target (int): gate target qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    x0 = clifford.x[:, control]
    z0 = clifford.z[:, control]
    x1 = clifford.x[:, target]
    z1 = clifford.z[:, target]
    clifford.phase ^= x0 & x1 & (z0 ^ z1)
    z1 ^= x0
    z0 ^= x1
    return clifford

def _append_cy(clifford, control, target):
    if False:
        for i in range(10):
            print('nop')
    'Apply a CY gate to a Clifford.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        control (int): gate control qubit index.\n        target (int): gate target qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    clifford = _append_sdg(clifford, target)
    clifford = _append_cx(clifford, control, target)
    clifford = _append_s(clifford, target)
    return clifford

def _append_swap(clifford, qubit0, qubit1):
    if False:
        while True:
            i = 10
    'Apply a Swap gate to a Clifford.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        qubit0 (int): first qubit index.\n        qubit1 (int): second  qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    clifford.x[:, [qubit0, qubit1]] = clifford.x[:, [qubit1, qubit0]]
    clifford.z[:, [qubit0, qubit1]] = clifford.z[:, [qubit1, qubit0]]
    return clifford

def _append_iswap(clifford, qubit0, qubit1):
    if False:
        for i in range(10):
            print('nop')
    'Apply a iSwap gate to a Clifford.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        qubit0 (int): first qubit index.\n        qubit1 (int): second  qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    clifford = _append_s(clifford, qubit0)
    clifford = _append_h(clifford, qubit0)
    clifford = _append_s(clifford, qubit1)
    clifford = _append_cx(clifford, qubit0, qubit1)
    clifford = _append_cx(clifford, qubit1, qubit0)
    clifford = _append_h(clifford, qubit1)
    return clifford

def _append_dcx(clifford, qubit0, qubit1):
    if False:
        print('Hello World!')
    'Apply a DCX gate to a Clifford.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        qubit0 (int): first qubit index.\n        qubit1 (int): second  qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    clifford = _append_cx(clifford, qubit0, qubit1)
    clifford = _append_cx(clifford, qubit1, qubit0)
    return clifford

def _append_ecr(clifford, qubit0, qubit1):
    if False:
        while True:
            i = 10
    'Apply an ECR gate to a Clifford.\n\n    Args:\n        clifford (Clifford): a Clifford.\n        qubit0 (int): first qubit index.\n        qubit1 (int): second  qubit index.\n\n    Returns:\n        Clifford: the updated Clifford.\n    '
    clifford = _append_s(clifford, qubit0)
    clifford = _append_sx(clifford, qubit1)
    clifford = _append_cx(clifford, qubit0, qubit1)
    clifford = _append_x(clifford, qubit0)
    return clifford
_BASIS_1Q = {'i': _append_i, 'id': _append_i, 'iden': _append_i, 'x': _append_x, 'y': _append_y, 'z': _append_z, 'h': _append_h, 's': _append_s, 'sdg': _append_sdg, 'sinv': _append_sdg, 'sx': _append_sx, 'sxdg': _append_sxdg, 'v': _append_v, 'w': _append_w}
_BASIS_2Q = {'cx': _append_cx, 'cz': _append_cz, 'cy': _append_cy, 'swap': _append_swap, 'iswap': _append_iswap, 'ecr': _append_ecr, 'dcx': _append_dcx}
_NON_CLIFFORD = {'t', 'tdg', 'ccx', 'ccz'}