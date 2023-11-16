"""Primitive circuit operations on quantum circuits."""
from functools import reduce
from sympy.core.sorting import default_sort_key
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import Gate
__all__ = ['kmp_table', 'find_subcircuit', 'replace_subcircuit', 'convert_to_symbolic_indices', 'convert_to_real_indices', 'random_reduce', 'random_insert']

def kmp_table(word):
    if False:
        i = 10
        return i + 15
    "Build the 'partial match' table of the Knuth-Morris-Pratt algorithm.\n\n    Note: This is applicable to strings or\n    quantum circuits represented as tuples.\n    "
    pos = 2
    cnd = 0
    table = []
    table.append(-1)
    table.append(0)
    while pos < len(word):
        if word[pos - 1] == word[cnd]:
            cnd = cnd + 1
            table.append(cnd)
            pos = pos + 1
        elif cnd > 0:
            cnd = table[cnd]
        else:
            table.append(0)
            pos = pos + 1
    return table

def find_subcircuit(circuit, subcircuit, start=0, end=0):
    if False:
        return 10
    'Finds the subcircuit in circuit, if it exists.\n\n    Explanation\n    ===========\n\n    If the subcircuit exists, the index of the start of\n    the subcircuit in circuit is returned; otherwise,\n    -1 is returned.  The algorithm that is implemented\n    is the Knuth-Morris-Pratt algorithm.\n\n    Parameters\n    ==========\n\n    circuit : tuple, Gate or Mul\n        A tuple of Gates or Mul representing a quantum circuit\n    subcircuit : tuple, Gate or Mul\n        A tuple of Gates or Mul to find in circuit\n    start : int\n        The location to start looking for subcircuit.\n        If start is the same or past end, -1 is returned.\n    end : int\n        The last place to look for a subcircuit.  If end\n        is less than 1 (one), then the length of circuit\n        is taken to be end.\n\n    Examples\n    ========\n\n    Find the first instance of a subcircuit:\n\n    >>> from sympy.physics.quantum.circuitutils import find_subcircuit\n    >>> from sympy.physics.quantum.gate import X, Y, Z, H\n    >>> circuit = X(0)*Z(0)*Y(0)*H(0)\n    >>> subcircuit = Z(0)*Y(0)\n    >>> find_subcircuit(circuit, subcircuit)\n    1\n\n    Find the first instance starting at a specific position:\n\n    >>> find_subcircuit(circuit, subcircuit, start=1)\n    1\n\n    >>> find_subcircuit(circuit, subcircuit, start=2)\n    -1\n\n    >>> circuit = circuit*subcircuit\n    >>> find_subcircuit(circuit, subcircuit, start=2)\n    4\n\n    Find the subcircuit within some interval:\n\n    >>> find_subcircuit(circuit, subcircuit, start=2, end=2)\n    -1\n    '
    if isinstance(circuit, Mul):
        circuit = circuit.args
    if isinstance(subcircuit, Mul):
        subcircuit = subcircuit.args
    if len(subcircuit) == 0 or len(subcircuit) > len(circuit):
        return -1
    if end < 1:
        end = len(circuit)
    pos = start
    index = 0
    table = kmp_table(subcircuit)
    while pos + index < end:
        if subcircuit[index] == circuit[pos + index]:
            index = index + 1
        else:
            pos = pos + index - table[index]
            index = table[index] if table[index] > -1 else 0
        if index == len(subcircuit):
            return pos
    return -1

def replace_subcircuit(circuit, subcircuit, replace=None, pos=0):
    if False:
        for i in range(10):
            print('nop')
    'Replaces a subcircuit with another subcircuit in circuit,\n    if it exists.\n\n    Explanation\n    ===========\n\n    If multiple instances of subcircuit exists, the first instance is\n    replaced.  The position to being searching from (if different from\n    0) may be optionally given.  If subcircuit cannot be found, circuit\n    is returned.\n\n    Parameters\n    ==========\n\n    circuit : tuple, Gate or Mul\n        A quantum circuit.\n    subcircuit : tuple, Gate or Mul\n        The circuit to be replaced.\n    replace : tuple, Gate or Mul\n        The replacement circuit.\n    pos : int\n        The location to start search and replace\n        subcircuit, if it exists.  This may be used\n        if it is known beforehand that multiple\n        instances exist, and it is desirable to\n        replace a specific instance.  If a negative number\n        is given, pos will be defaulted to 0.\n\n    Examples\n    ========\n\n    Find and remove the subcircuit:\n\n    >>> from sympy.physics.quantum.circuitutils import replace_subcircuit\n    >>> from sympy.physics.quantum.gate import X, Y, Z, H\n    >>> circuit = X(0)*Z(0)*Y(0)*H(0)*X(0)*H(0)*Y(0)\n    >>> subcircuit = Z(0)*Y(0)\n    >>> replace_subcircuit(circuit, subcircuit)\n    (X(0), H(0), X(0), H(0), Y(0))\n\n    Remove the subcircuit given a starting search point:\n\n    >>> replace_subcircuit(circuit, subcircuit, pos=1)\n    (X(0), H(0), X(0), H(0), Y(0))\n\n    >>> replace_subcircuit(circuit, subcircuit, pos=2)\n    (X(0), Z(0), Y(0), H(0), X(0), H(0), Y(0))\n\n    Replace the subcircuit:\n\n    >>> replacement = H(0)*Z(0)\n    >>> replace_subcircuit(circuit, subcircuit, replace=replacement)\n    (X(0), H(0), Z(0), H(0), X(0), H(0), Y(0))\n    '
    if pos < 0:
        pos = 0
    if isinstance(circuit, Mul):
        circuit = circuit.args
    if isinstance(subcircuit, Mul):
        subcircuit = subcircuit.args
    if isinstance(replace, Mul):
        replace = replace.args
    elif replace is None:
        replace = ()
    loc = find_subcircuit(circuit, subcircuit, start=pos)
    if loc > -1:
        left = circuit[0:loc]
        right = circuit[loc + len(subcircuit):len(circuit)]
        circuit = left + replace + right
    return circuit

def _sympify_qubit_map(mapping):
    if False:
        i = 10
        return i + 15
    new_map = {}
    for key in mapping:
        new_map[key] = sympify(mapping[key])
    return new_map

def convert_to_symbolic_indices(seq, start=None, gen=None, qubit_map=None):
    if False:
        print('Hello World!')
    "Returns the circuit with symbolic indices and the\n    dictionary mapping symbolic indices to real indices.\n\n    The mapping is 1 to 1 and onto (bijective).\n\n    Parameters\n    ==========\n\n    seq : tuple, Gate/Integer/tuple or Mul\n        A tuple of Gate, Integer, or tuple objects, or a Mul\n    start : Symbol\n        An optional starting symbolic index\n    gen : object\n        An optional numbered symbol generator\n    qubit_map : dict\n        An existing mapping of symbolic indices to real indices\n\n    All symbolic indices have the format 'i#', where # is\n    some number >= 0.\n    "
    if isinstance(seq, Mul):
        seq = seq.args
    index_gen = numbered_symbols(prefix='i', start=-1)
    cur_ndx = next(index_gen)
    ndx_map = {}

    def create_inverse_map(symb_to_real_map):
        if False:
            while True:
                i = 10
        rev_items = lambda item: (item[1], item[0])
        return dict(map(rev_items, symb_to_real_map.items()))
    if start is not None:
        if not isinstance(start, Symbol):
            msg = 'Expected Symbol for starting index, got %r.' % start
            raise TypeError(msg)
        cur_ndx = start
    if gen is not None:
        if not isinstance(gen, numbered_symbols().__class__):
            msg = 'Expected a generator, got %r.' % gen
            raise TypeError(msg)
        index_gen = gen
    if qubit_map is not None:
        if not isinstance(qubit_map, dict):
            msg = 'Expected dict for existing map, got ' + '%r.' % qubit_map
            raise TypeError(msg)
        ndx_map = qubit_map
    ndx_map = _sympify_qubit_map(ndx_map)
    inv_map = create_inverse_map(ndx_map)
    sym_seq = ()
    for item in seq:
        if isinstance(item, Gate):
            result = convert_to_symbolic_indices(item.args, qubit_map=ndx_map, start=cur_ndx, gen=index_gen)
            (sym_item, new_map, cur_ndx, index_gen) = result
            ndx_map.update(new_map)
            inv_map = create_inverse_map(ndx_map)
        elif isinstance(item, (tuple, Tuple)):
            result = convert_to_symbolic_indices(item, qubit_map=ndx_map, start=cur_ndx, gen=index_gen)
            (sym_item, new_map, cur_ndx, index_gen) = result
            ndx_map.update(new_map)
            inv_map = create_inverse_map(ndx_map)
        elif item in inv_map:
            sym_item = inv_map[item]
        else:
            cur_ndx = next(gen)
            ndx_map[cur_ndx] = item
            inv_map[item] = cur_ndx
            sym_item = cur_ndx
        if isinstance(item, Gate):
            sym_item = item.__class__(*sym_item)
        sym_seq = sym_seq + (sym_item,)
    return (sym_seq, ndx_map, cur_ndx, index_gen)

def convert_to_real_indices(seq, qubit_map):
    if False:
        return 10
    "Returns the circuit with real indices.\n\n    Parameters\n    ==========\n\n    seq : tuple, Gate/Integer/tuple or Mul\n        A tuple of Gate, Integer, or tuple objects or a Mul\n    qubit_map : dict\n        A dictionary mapping symbolic indices to real indices.\n\n    Examples\n    ========\n\n    Change the symbolic indices to real integers:\n\n    >>> from sympy import symbols\n    >>> from sympy.physics.quantum.circuitutils import convert_to_real_indices\n    >>> from sympy.physics.quantum.gate import X, Y, H\n    >>> i0, i1 = symbols('i:2')\n    >>> index_map = {i0 : 0, i1 : 1}\n    >>> convert_to_real_indices(X(i0)*Y(i1)*H(i0)*X(i1), index_map)\n    (X(0), Y(1), H(0), X(1))\n    "
    if isinstance(seq, Mul):
        seq = seq.args
    if not isinstance(qubit_map, dict):
        msg = 'Expected dict for qubit_map, got %r.' % qubit_map
        raise TypeError(msg)
    qubit_map = _sympify_qubit_map(qubit_map)
    real_seq = ()
    for item in seq:
        if isinstance(item, Gate):
            real_item = convert_to_real_indices(item.args, qubit_map)
        elif isinstance(item, (tuple, Tuple)):
            real_item = convert_to_real_indices(item, qubit_map)
        else:
            real_item = qubit_map[item]
        if isinstance(item, Gate):
            real_item = item.__class__(*real_item)
        real_seq = real_seq + (real_item,)
    return real_seq

def random_reduce(circuit, gate_ids, seed=None):
    if False:
        while True:
            i = 10
    'Shorten the length of a quantum circuit.\n\n    Explanation\n    ===========\n\n    random_reduce looks for circuit identities in circuit, randomly chooses\n    one to remove, and returns a shorter yet equivalent circuit.  If no\n    identities are found, the same circuit is returned.\n\n    Parameters\n    ==========\n\n    circuit : Gate tuple of Mul\n        A tuple of Gates representing a quantum circuit\n    gate_ids : list, GateIdentity\n        List of gate identities to find in circuit\n    seed : int or list\n        seed used for _randrange; to override the random selection, provide a\n        list of integers: the elements of gate_ids will be tested in the order\n        given by the list\n\n    '
    from sympy.core.random import _randrange
    if not gate_ids:
        return circuit
    if isinstance(circuit, Mul):
        circuit = circuit.args
    ids = flatten_ids(gate_ids)
    randrange = _randrange(seed)
    while ids:
        i = randrange(len(ids))
        id = ids.pop(i)
        if find_subcircuit(circuit, id) != -1:
            break
    else:
        return circuit
    return replace_subcircuit(circuit, id)

def random_insert(circuit, choices, seed=None):
    if False:
        return 10
    'Insert a circuit into another quantum circuit.\n\n    Explanation\n    ===========\n\n    random_insert randomly chooses a location in the circuit to insert\n    a randomly selected circuit from amongst the given choices.\n\n    Parameters\n    ==========\n\n    circuit : Gate tuple or Mul\n        A tuple or Mul of Gates representing a quantum circuit\n    choices : list\n        Set of circuit choices\n    seed : int or list\n        seed used for _randrange; to override the random selections, give\n        a list two integers, [i, j] where i is the circuit location where\n        choice[j] will be inserted.\n\n    Notes\n    =====\n\n    Indices for insertion should be [0, n] if n is the length of the\n    circuit.\n    '
    from sympy.core.random import _randrange
    if not choices:
        return circuit
    if isinstance(circuit, Mul):
        circuit = circuit.args
    randrange = _randrange(seed)
    loc = randrange(len(circuit) + 1)
    choice = choices[randrange(len(choices))]
    circuit = list(circuit)
    circuit[loc:loc] = choice
    return tuple(circuit)

def flatten_ids(ids):
    if False:
        return 10
    collapse = lambda acc, an_id: acc + sorted(an_id.equivalent_ids, key=default_sort_key)
    ids = reduce(collapse, ids, [])
    ids.sort(key=default_sort_key)
    return ids