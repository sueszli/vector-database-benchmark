"""
This module contains the definition of creating and validating entangler map
based on the number of qubits.
"""

def get_entangler_map(map_type, num_qubits, offset=0):
    if False:
        print('Hello World!')
    "Utility method to get an entangler map among qubits.\n\n    Args:\n        map_type (str): 'full' entangles each qubit with all the subsequent ones\n                        'linear' entangles each qubit with the next\n                        'sca' (shifted circular alternating entanglement) is a\n                        circular entanglement where the 'long' entanglement is\n                        shifted by one position every block and every block the\n                        role or control/target qubits alternate\n        num_qubits (int): Number of qubits for which the map is needed\n        offset (int): Some map_types (e.g. 'sca') can shift the gates in\n                      the entangler map by the specified integer offset.\n\n    Returns:\n        list: A map of qubit index to an array of indexes to which this should be entangled\n\n    Raises:\n        ValueError: if map_type is not valid.\n    "
    ret = []
    if num_qubits > 1:
        if map_type == 'full':
            ret = [[i, j] for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        elif map_type == 'linear':
            ret = [[i, i + 1] for i in range(num_qubits - 1)]
        elif map_type == 'sca':
            offset_idx = offset % num_qubits
            if offset_idx % 2 == 0:
                for i in reversed(range(offset_idx)):
                    ret += [[i, i + 1]]
                ret += [[num_qubits - 1, 0]]
                for i in reversed(range(offset_idx + 1, num_qubits)):
                    ret += [[i - 1, i]]
            else:
                for i in range(num_qubits - offset_idx - 1, num_qubits - 1):
                    ret += [[i + 1, i]]
                ret += [[0, num_qubits - 1]]
                for i in range(num_qubits - offset_idx - 1):
                    ret += [[i + 1, i]]
        else:
            raise ValueError("map_type only supports 'full', 'linear' or 'sca' type.")
    return ret

def validate_entangler_map(entangler_map, num_qubits, allow_double_entanglement=False):
    if False:
        print('Hello World!')
    'Validate a user supplied entangler map and converts entries to ints.\n\n    Args:\n        entangler_map (list[list]) : An entangler map, keys are source qubit index (int),\n                                value is array\n                                of target qubit index(es) (int)\n        num_qubits (int) : Number of qubits\n        allow_double_entanglement (bool): If we allow in two qubits can be entangled each other\n\n    Returns:\n        list: Validated/converted map\n\n    Raises:\n        TypeError: entangler map is not list type or list of list\n        ValueError: the index of entangler map is out of range\n        ValueError: the qubits are cross-entangled.\n\n    '
    if isinstance(entangler_map, dict):
        raise TypeError('The type of entangler map is changed to list of list.')
    if not isinstance(entangler_map, list):
        raise TypeError("Entangler map type 'list' expected")
    for src_to_targ in entangler_map:
        if not isinstance(src_to_targ, list):
            raise TypeError(f'Entangle index list expected but got {type(src_to_targ)}')
    ret_map = []
    ret_map = [[int(src), int(targ)] for (src, targ) in entangler_map]
    for (src, targ) in ret_map:
        if src < 0 or src >= num_qubits:
            raise ValueError(f'Qubit entangle source value {src} invalid for {num_qubits} qubits')
        if targ < 0 or targ >= num_qubits:
            raise ValueError(f'Qubit entangle target value {targ} invalid for {num_qubits} qubits')
        if not allow_double_entanglement and [targ, src] in ret_map:
            raise ValueError(f'Qubit {src} and {targ} cross-entangled.')
    return ret_map