"""
These are the CNOT structure methods: anything that you need for creating CNOT structures.
"""
import logging
import numpy as np
_NETWORK_LAYOUTS = ['sequ', 'spin', 'cart', 'cyclic_spin', 'cyclic_line']
_CONNECTIVITY_TYPES = ['full', 'line', 'star']
logger = logging.getLogger(__name__)

def _lower_limit(num_qubits: int) -> int:
    if False:
        i = 10
        return i + 15
    '\n    Returns lower limit on the number of CNOT units that guarantees exact representation of\n    a unitary operator by quantum gates.\n\n    Args:\n        num_qubits: number of qubits.\n\n    Returns:\n        lower limit on the number of CNOT units.\n    '
    num_cnots = round(np.ceil((4 ** num_qubits - 3 * num_qubits - 1) / 4.0))
    return num_cnots

def make_cnot_network(num_qubits: int, network_layout: str='spin', connectivity_type: str='full', depth: int=0) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates a network consisting of building blocks each containing a CNOT gate and possibly some\n    single-qubit ones. This network models a quantum operator in question. Note, each building\n    block has 2 input and outputs corresponding to a pair of qubits. What we actually return here\n    is a chain of indices of qubit pairs shared by every building block in a row.\n\n    Args:\n        num_qubits: number of qubits.\n        network_layout: type of network geometry, ``{"sequ", "spin", "cart", "cyclic_spin",\n            "cyclic_line"}``.\n        connectivity_type: type of inter-qubit connectivity, ``{"full", "line", "star"}``.\n        depth: depth of the CNOT-network, i.e. the number of layers, where each layer consists of\n            a single CNOT-block; default value will be selected, if ``L <= 0``.\n\n    Returns:\n        A matrix of size ``(2, N)`` matrix that defines layers in cnot-network, where ``N``\n            is either equal ``L``, or defined by a concrete type of the network.\n\n    Raises:\n         ValueError: if unsupported type of CNOT-network layout or number of qubits or combination\n            of parameters are passed.\n    '
    if num_qubits < 2:
        raise ValueError('Number of qubits must be greater or equal to 2')
    if depth <= 0:
        new_depth = _lower_limit(num_qubits)
        logger.debug('Number of CNOT units chosen as the lower limit: %d, got a non-positive value: %d', new_depth, depth)
        depth = new_depth
    if network_layout == 'sequ':
        links = _get_connectivity(num_qubits=num_qubits, connectivity=connectivity_type)
        return _sequential_network(num_qubits=num_qubits, links=links, depth=depth)
    elif network_layout == 'spin':
        return _spin_network(num_qubits=num_qubits, depth=depth)
    elif network_layout == 'cart':
        cnots = _cartan_network(num_qubits=num_qubits)
        logger.debug('Optimal lower bound: %d; Cartan CNOTs: %d', _lower_limit(num_qubits), cnots.shape[1])
        return cnots
    elif network_layout == 'cyclic_spin':
        if connectivity_type != 'full':
            raise ValueError(f"'{network_layout}' layout expects 'full' connectivity")
        return _cyclic_spin_network(num_qubits, depth)
    elif network_layout == 'cyclic_line':
        if connectivity_type != 'line':
            raise ValueError(f"'{network_layout}' layout expects 'line' connectivity")
        return _cyclic_line_network(num_qubits, depth)
    else:
        raise ValueError(f'Unknown type of CNOT-network layout, expects one of {_NETWORK_LAYOUTS}, got {network_layout}')

def _get_connectivity(num_qubits: int, connectivity: str) -> dict:
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates connectivity structure between qubits.\n\n    Args:\n        num_qubits: number of qubits.\n        connectivity: type of connectivity structure, ``{"full", "line", "star"}``.\n\n    Returns:\n        dictionary of allowed links between qubits.\n\n    Raises:\n         ValueError: if unsupported type of CNOT-network layout is passed.\n    '
    if num_qubits == 1:
        links = {0: [0]}
    elif connectivity == 'full':
        links = {i: list(range(num_qubits)) for i in range(num_qubits)}
    elif connectivity == 'line':
        links = {i: [i - 1, i, i + 1] for i in range(1, num_qubits - 1)}
        links[0] = [0, 1]
        links[num_qubits - 1] = [num_qubits - 2, num_qubits - 1]
    elif connectivity == 'star':
        links = {i: [0, i] for i in range(1, num_qubits)}
        links[0] = list(range(num_qubits))
    else:
        raise ValueError(f'Unknown connectivity type, expects one of {_CONNECTIVITY_TYPES}, got {connectivity}')
    return links

def _sequential_network(num_qubits: int, links: dict, depth: int) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates a sequential network.\n\n    Args:\n        num_qubits: number of qubits.\n        links: dictionary of connectivity links.\n        depth: depth of the network (number of layers of building blocks).\n\n    Returns:\n        A matrix of ``(2, N)`` that defines layers in qubit network.\n    '
    layer = 0
    cnots = np.zeros((2, depth), dtype=int)
    while True:
        for i in range(0, num_qubits - 1):
            for j in range(i + 1, num_qubits):
                if j in links[i]:
                    cnots[0, layer] = i
                    cnots[1, layer] = j
                    layer += 1
                    if layer >= depth:
                        return cnots

def _spin_network(num_qubits: int, depth: int) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    '\n    Generates a spin-like network.\n\n    Args:\n        num_qubits: number of qubits.\n        depth: depth of the network (number of layers of building blocks).\n\n    Returns:\n        A matrix of size ``2 x L`` that defines layers in qubit network.\n    '
    layer = 0
    cnots = np.zeros((2, depth), dtype=int)
    while True:
        for i in range(0, num_qubits - 1, 2):
            cnots[0, layer] = i
            cnots[1, layer] = i + 1
            layer += 1
            if layer >= depth:
                return cnots
        for i in range(1, num_qubits - 1, 2):
            cnots[0, layer] = i
            cnots[1, layer] = i + 1
            layer += 1
            if layer >= depth:
                return cnots

def _cartan_network(num_qubits: int) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    '\n    Cartan decomposition in a recursive way, starting from n = 3.\n\n    Args:\n        num_qubits: number of qubits.\n\n    Returns:\n        2xN matrix that defines layers in qubit network, where N is the\n             depth of Cartan decomposition.\n\n    Raises:\n        ValueError: if number of qubits is less than 3.\n    '
    n = num_qubits
    if n > 3:
        cnots = np.asarray([[0, 0, 0], [1, 1, 1]])
        mult = np.asarray([[n - 2, n - 3, n - 2, n - 3], [n - 1, n - 1, n - 1, n - 1]])
        for _ in range(n - 2):
            cnots = np.hstack((np.tile(np.hstack((cnots, mult)), 3), cnots))
            mult[0, -1] -= 1
            mult = np.tile(mult, 2)
    elif n == 3:
        cnots = np.asarray([[0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1]])
    else:
        raise ValueError(f'The number of qubits must be >= 3, got {n}.')
    return cnots

def _cyclic_spin_network(num_qubits: int, depth: int) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    '\n    Same as in the spin-like network, but the first and the last qubits are also connected.\n\n    Args:\n        num_qubits: number of qubits.\n        depth: depth of the network (number of layers of building blocks).\n\n    Returns:\n        A matrix of size ``2 x L`` that defines layers in qubit network.\n    '
    cnots = np.zeros((2, depth), dtype=int)
    z = 0
    while True:
        for i in range(0, num_qubits, 2):
            if i + 1 <= num_qubits - 1:
                cnots[0, z] = i
                cnots[1, z] = i + 1
                z += 1
            if z >= depth:
                return cnots
        for i in range(1, num_qubits, 2):
            if i + 1 <= num_qubits - 1:
                cnots[0, z] = i
                cnots[1, z] = i + 1
                z += 1
            elif i == num_qubits - 1:
                cnots[0, z] = i
                cnots[1, z] = 0
                z += 1
            if z >= depth:
                return cnots

def _cyclic_line_network(num_qubits: int, depth: int) -> np.ndarray:
    if False:
        while True:
            i = 10
    '\n    Generates a line based CNOT structure.\n\n    Args:\n        num_qubits: number of qubits.\n        depth: depth of the network (number of layers of building blocks).\n\n    Returns:\n        A matrix of size ``2 x L`` that defines layers in qubit network.\n    '
    cnots = np.zeros((2, depth), dtype=int)
    for i in range(depth):
        cnots[0, i] = (i + 0) % num_qubits
        cnots[1, i] = (i + 1) % num_qubits
    return cnots