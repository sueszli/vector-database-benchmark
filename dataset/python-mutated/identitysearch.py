from collections import deque
from sympy.core.random import randint
from sympy.external import import_module
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.numbers import Number, equal_valued
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.dagger import Dagger
__all__ = ['generate_gate_rules', 'generate_equivalent_ids', 'GateIdentity', 'bfs_identity_search', 'random_identity_search', 'is_scalar_sparse_matrix', 'is_scalar_nonsparse_matrix', 'is_degenerate', 'is_reducible']
np = import_module('numpy')
scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})

def is_scalar_sparse_matrix(circuit, nqubits, identity_only, eps=1e-11):
    if False:
        for i in range(10):
            print('nop')
    "Checks if a given scipy.sparse matrix is a scalar matrix.\n\n    A scalar matrix is such that B = bI, where B is the scalar\n    matrix, b is some scalar multiple, and I is the identity\n    matrix.  A scalar matrix would have only the element b along\n    it's main diagonal and zeroes elsewhere.\n\n    Parameters\n    ==========\n\n    circuit : Gate tuple\n        Sequence of quantum gates representing a quantum circuit\n    nqubits : int\n        Number of qubits in the circuit\n    identity_only : bool\n        Check for only identity matrices\n    eps : number\n        The tolerance value for zeroing out elements in the matrix.\n        Values in the range [-eps, +eps] will be changed to a zero.\n    "
    if not np or not scipy:
        pass
    matrix = represent(Mul(*circuit), nqubits=nqubits, format='scipy.sparse')
    if isinstance(matrix, int):
        return matrix == 1 if identity_only else True
    else:
        dense_matrix = matrix.todense().getA()
        bool_real = np.logical_and(dense_matrix.real > -eps, dense_matrix.real < eps)
        bool_imag = np.logical_and(dense_matrix.imag > -eps, dense_matrix.imag < eps)
        corrected_real = np.where(bool_real, 0.0, dense_matrix.real)
        corrected_imag = np.where(bool_imag, 0.0, dense_matrix.imag)
        corrected_imag = corrected_imag * complex(1j)
        corrected_dense = corrected_real + corrected_imag
        row_indices = corrected_dense.nonzero()[0]
        col_indices = corrected_dense.nonzero()[1]
        bool_indices = row_indices == col_indices
        is_diagonal = bool_indices.all()
        first_element = corrected_dense[0][0]
        if first_element == 0.0 + 0j:
            return False
        trace_of_corrected = (corrected_dense / first_element).trace()
        expected_trace = pow(2, nqubits)
        has_correct_trace = trace_of_corrected == expected_trace
        real_is_one = abs(first_element.real - 1.0) < eps
        imag_is_zero = abs(first_element.imag) < eps
        is_one = real_is_one and imag_is_zero
        is_identity = is_one if identity_only else True
        return bool(is_diagonal and has_correct_trace and is_identity)

def is_scalar_nonsparse_matrix(circuit, nqubits, identity_only, eps=None):
    if False:
        print('Hello World!')
    'Checks if a given circuit, in matrix form, is equivalent to\n    a scalar value.\n\n    Parameters\n    ==========\n\n    circuit : Gate tuple\n        Sequence of quantum gates representing a quantum circuit\n    nqubits : int\n        Number of qubits in the circuit\n    identity_only : bool\n        Check for only identity matrices\n    eps : number\n        This argument is ignored. It is just for signature compatibility with\n        is_scalar_sparse_matrix.\n\n    Note: Used in situations when is_scalar_sparse_matrix has bugs\n    '
    matrix = represent(Mul(*circuit), nqubits=nqubits)
    if isinstance(matrix, Number):
        return matrix == 1 if identity_only else True
    else:
        matrix_trace = matrix.trace()
        adjusted_matrix_trace = matrix_trace / matrix[0] if not identity_only else matrix_trace
        is_identity = equal_valued(matrix[0], 1) if identity_only else True
        has_correct_trace = adjusted_matrix_trace == pow(2, nqubits)
        return bool(matrix.is_diagonal() and has_correct_trace and is_identity)
if np and scipy:
    is_scalar_matrix = is_scalar_sparse_matrix
else:
    is_scalar_matrix = is_scalar_nonsparse_matrix

def _get_min_qubits(a_gate):
    if False:
        while True:
            i = 10
    if isinstance(a_gate, Pow):
        return a_gate.base.min_qubits
    else:
        return a_gate.min_qubits

def ll_op(left, right):
    if False:
        for i in range(10):
            print('nop')
    "Perform a LL operation.\n\n    A LL operation multiplies both left and right circuits\n    with the dagger of the left circuit's leftmost gate, and\n    the dagger is multiplied on the left side of both circuits.\n\n    If a LL is possible, it returns the new gate rule as a\n    2-tuple (LHS, RHS), where LHS is the left circuit and\n    and RHS is the right circuit of the new rule.\n    If a LL is not possible, None is returned.\n\n    Parameters\n    ==========\n\n    left : Gate tuple\n        The left circuit of a gate rule expression.\n    right : Gate tuple\n        The right circuit of a gate rule expression.\n\n    Examples\n    ========\n\n    Generate a new gate rule using a LL operation:\n\n    >>> from sympy.physics.quantum.identitysearch import ll_op\n    >>> from sympy.physics.quantum.gate import X, Y, Z\n    >>> x = X(0); y = Y(0); z = Z(0)\n    >>> ll_op((x, y, z), ())\n    ((Y(0), Z(0)), (X(0),))\n\n    >>> ll_op((y, z), (x,))\n    ((Z(0),), (Y(0), X(0)))\n    "
    if len(left) > 0:
        ll_gate = left[0]
        ll_gate_is_unitary = is_scalar_matrix((Dagger(ll_gate), ll_gate), _get_min_qubits(ll_gate), True)
    if len(left) > 0 and ll_gate_is_unitary:
        new_left = left[1:len(left)]
        new_right = (Dagger(ll_gate),) + right
        return (new_left, new_right)
    return None

def lr_op(left, right):
    if False:
        for i in range(10):
            print('nop')
    "Perform a LR operation.\n\n    A LR operation multiplies both left and right circuits\n    with the dagger of the left circuit's rightmost gate, and\n    the dagger is multiplied on the right side of both circuits.\n\n    If a LR is possible, it returns the new gate rule as a\n    2-tuple (LHS, RHS), where LHS is the left circuit and\n    and RHS is the right circuit of the new rule.\n    If a LR is not possible, None is returned.\n\n    Parameters\n    ==========\n\n    left : Gate tuple\n        The left circuit of a gate rule expression.\n    right : Gate tuple\n        The right circuit of a gate rule expression.\n\n    Examples\n    ========\n\n    Generate a new gate rule using a LR operation:\n\n    >>> from sympy.physics.quantum.identitysearch import lr_op\n    >>> from sympy.physics.quantum.gate import X, Y, Z\n    >>> x = X(0); y = Y(0); z = Z(0)\n    >>> lr_op((x, y, z), ())\n    ((X(0), Y(0)), (Z(0),))\n\n    >>> lr_op((x, y), (z,))\n    ((X(0),), (Z(0), Y(0)))\n    "
    if len(left) > 0:
        lr_gate = left[len(left) - 1]
        lr_gate_is_unitary = is_scalar_matrix((Dagger(lr_gate), lr_gate), _get_min_qubits(lr_gate), True)
    if len(left) > 0 and lr_gate_is_unitary:
        new_left = left[0:len(left) - 1]
        new_right = right + (Dagger(lr_gate),)
        return (new_left, new_right)
    return None

def rl_op(left, right):
    if False:
        while True:
            i = 10
    "Perform a RL operation.\n\n    A RL operation multiplies both left and right circuits\n    with the dagger of the right circuit's leftmost gate, and\n    the dagger is multiplied on the left side of both circuits.\n\n    If a RL is possible, it returns the new gate rule as a\n    2-tuple (LHS, RHS), where LHS is the left circuit and\n    and RHS is the right circuit of the new rule.\n    If a RL is not possible, None is returned.\n\n    Parameters\n    ==========\n\n    left : Gate tuple\n        The left circuit of a gate rule expression.\n    right : Gate tuple\n        The right circuit of a gate rule expression.\n\n    Examples\n    ========\n\n    Generate a new gate rule using a RL operation:\n\n    >>> from sympy.physics.quantum.identitysearch import rl_op\n    >>> from sympy.physics.quantum.gate import X, Y, Z\n    >>> x = X(0); y = Y(0); z = Z(0)\n    >>> rl_op((x,), (y, z))\n    ((Y(0), X(0)), (Z(0),))\n\n    >>> rl_op((x, y), (z,))\n    ((Z(0), X(0), Y(0)), ())\n    "
    if len(right) > 0:
        rl_gate = right[0]
        rl_gate_is_unitary = is_scalar_matrix((Dagger(rl_gate), rl_gate), _get_min_qubits(rl_gate), True)
    if len(right) > 0 and rl_gate_is_unitary:
        new_right = right[1:len(right)]
        new_left = (Dagger(rl_gate),) + left
        return (new_left, new_right)
    return None

def rr_op(left, right):
    if False:
        return 10
    "Perform a RR operation.\n\n    A RR operation multiplies both left and right circuits\n    with the dagger of the right circuit's rightmost gate, and\n    the dagger is multiplied on the right side of both circuits.\n\n    If a RR is possible, it returns the new gate rule as a\n    2-tuple (LHS, RHS), where LHS is the left circuit and\n    and RHS is the right circuit of the new rule.\n    If a RR is not possible, None is returned.\n\n    Parameters\n    ==========\n\n    left : Gate tuple\n        The left circuit of a gate rule expression.\n    right : Gate tuple\n        The right circuit of a gate rule expression.\n\n    Examples\n    ========\n\n    Generate a new gate rule using a RR operation:\n\n    >>> from sympy.physics.quantum.identitysearch import rr_op\n    >>> from sympy.physics.quantum.gate import X, Y, Z\n    >>> x = X(0); y = Y(0); z = Z(0)\n    >>> rr_op((x, y), (z,))\n    ((X(0), Y(0), Z(0)), ())\n\n    >>> rr_op((x,), (y, z))\n    ((X(0), Z(0)), (Y(0),))\n    "
    if len(right) > 0:
        rr_gate = right[len(right) - 1]
        rr_gate_is_unitary = is_scalar_matrix((Dagger(rr_gate), rr_gate), _get_min_qubits(rr_gate), True)
    if len(right) > 0 and rr_gate_is_unitary:
        new_right = right[0:len(right) - 1]
        new_left = left + (Dagger(rr_gate),)
        return (new_left, new_right)
    return None

def generate_gate_rules(gate_seq, return_as_muls=False):
    if False:
        i = 10
        return i + 15
    'Returns a set of gate rules.  Each gate rules is represented\n    as a 2-tuple of tuples or Muls.  An empty tuple represents an arbitrary\n    scalar value.\n\n    This function uses the four operations (LL, LR, RL, RR)\n    to generate the gate rules.\n\n    A gate rule is an expression such as ABC = D or AB = CD, where\n    A, B, C, and D are gates.  Each value on either side of the\n    equal sign represents a circuit.  The four operations allow\n    one to find a set of equivalent circuits from a gate identity.\n    The letters denoting the operation tell the user what\n    activities to perform on each expression.  The first letter\n    indicates which side of the equal sign to focus on.  The\n    second letter indicates which gate to focus on given the\n    side.  Once this information is determined, the inverse\n    of the gate is multiplied on both circuits to create a new\n    gate rule.\n\n    For example, given the identity, ABCD = 1, a LL operation\n    means look at the left value and multiply both left sides by the\n    inverse of the leftmost gate A.  If A is Hermitian, the inverse\n    of A is still A.  The resulting new rule is BCD = A.\n\n    The following is a summary of the four operations.  Assume\n    that in the examples, all gates are Hermitian.\n\n        LL : left circuit, left multiply\n             ABCD = E -> AABCD = AE -> BCD = AE\n        LR : left circuit, right multiply\n             ABCD = E -> ABCDD = ED -> ABC = ED\n        RL : right circuit, left multiply\n             ABC = ED -> EABC = EED -> EABC = D\n        RR : right circuit, right multiply\n             AB = CD -> ABD = CDD -> ABD = C\n\n    The number of gate rules generated is n*(n+1), where n\n    is the number of gates in the sequence (unproven).\n\n    Parameters\n    ==========\n\n    gate_seq : Gate tuple, Mul, or Number\n        A variable length tuple or Mul of Gates whose product is equal to\n        a scalar matrix\n    return_as_muls : bool\n        True to return a set of Muls; False to return a set of tuples\n\n    Examples\n    ========\n\n    Find the gate rules of the current circuit using tuples:\n\n    >>> from sympy.physics.quantum.identitysearch import generate_gate_rules\n    >>> from sympy.physics.quantum.gate import X, Y, Z\n    >>> x = X(0); y = Y(0); z = Z(0)\n    >>> generate_gate_rules((x, x))\n    {((X(0),), (X(0),)), ((X(0), X(0)), ())}\n\n    >>> generate_gate_rules((x, y, z))\n    {((), (X(0), Z(0), Y(0))), ((), (Y(0), X(0), Z(0))),\n     ((), (Z(0), Y(0), X(0))), ((X(0),), (Z(0), Y(0))),\n     ((Y(0),), (X(0), Z(0))), ((Z(0),), (Y(0), X(0))),\n     ((X(0), Y(0)), (Z(0),)), ((Y(0), Z(0)), (X(0),)),\n     ((Z(0), X(0)), (Y(0),)), ((X(0), Y(0), Z(0)), ()),\n     ((Y(0), Z(0), X(0)), ()), ((Z(0), X(0), Y(0)), ())}\n\n    Find the gate rules of the current circuit using Muls:\n\n    >>> generate_gate_rules(x*x, return_as_muls=True)\n    {(1, 1)}\n\n    >>> generate_gate_rules(x*y*z, return_as_muls=True)\n    {(1, X(0)*Z(0)*Y(0)), (1, Y(0)*X(0)*Z(0)),\n     (1, Z(0)*Y(0)*X(0)), (X(0)*Y(0), Z(0)),\n     (Y(0)*Z(0), X(0)), (Z(0)*X(0), Y(0)),\n     (X(0)*Y(0)*Z(0), 1), (Y(0)*Z(0)*X(0), 1),\n     (Z(0)*X(0)*Y(0), 1), (X(0), Z(0)*Y(0)),\n     (Y(0), X(0)*Z(0)), (Z(0), Y(0)*X(0))}\n    '
    if isinstance(gate_seq, Number):
        if return_as_muls:
            return {(S.One, S.One)}
        else:
            return {((), ())}
    elif isinstance(gate_seq, Mul):
        gate_seq = gate_seq.args
    queue = deque()
    rules = set()
    max_ops = len(gate_seq)

    def process_new_rule(new_rule, ops):
        if False:
            print('Hello World!')
        if new_rule is not None:
            (new_left, new_right) = new_rule
            if new_rule not in rules and (new_right, new_left) not in rules:
                rules.add(new_rule)
            if ops + 1 < max_ops:
                queue.append(new_rule + (ops + 1,))
    queue.append((gate_seq, (), 0))
    rules.add((gate_seq, ()))
    while len(queue) > 0:
        (left, right, ops) = queue.popleft()
        new_rule = ll_op(left, right)
        process_new_rule(new_rule, ops)
        new_rule = lr_op(left, right)
        process_new_rule(new_rule, ops)
        new_rule = rl_op(left, right)
        process_new_rule(new_rule, ops)
        new_rule = rr_op(left, right)
        process_new_rule(new_rule, ops)
    if return_as_muls:
        mul_rules = set()
        for rule in rules:
            (left, right) = rule
            mul_rules.add((Mul(*left), Mul(*right)))
        rules = mul_rules
    return rules

def generate_equivalent_ids(gate_seq, return_as_muls=False):
    if False:
        i = 10
        return i + 15
    'Returns a set of equivalent gate identities.\n\n    A gate identity is a quantum circuit such that the product\n    of the gates in the circuit is equal to a scalar value.\n    For example, XYZ = i, where X, Y, Z are the Pauli gates and\n    i is the imaginary value, is considered a gate identity.\n\n    This function uses the four operations (LL, LR, RL, RR)\n    to generate the gate rules and, subsequently, to locate equivalent\n    gate identities.\n\n    Note that all equivalent identities are reachable in n operations\n    from the starting gate identity, where n is the number of gates\n    in the sequence.\n\n    The max number of gate identities is 2n, where n is the number\n    of gates in the sequence (unproven).\n\n    Parameters\n    ==========\n\n    gate_seq : Gate tuple, Mul, or Number\n        A variable length tuple or Mul of Gates whose product is equal to\n        a scalar matrix.\n    return_as_muls: bool\n        True to return as Muls; False to return as tuples\n\n    Examples\n    ========\n\n    Find equivalent gate identities from the current circuit with tuples:\n\n    >>> from sympy.physics.quantum.identitysearch import generate_equivalent_ids\n    >>> from sympy.physics.quantum.gate import X, Y, Z\n    >>> x = X(0); y = Y(0); z = Z(0)\n    >>> generate_equivalent_ids((x, x))\n    {(X(0), X(0))}\n\n    >>> generate_equivalent_ids((x, y, z))\n    {(X(0), Y(0), Z(0)), (X(0), Z(0), Y(0)), (Y(0), X(0), Z(0)),\n     (Y(0), Z(0), X(0)), (Z(0), X(0), Y(0)), (Z(0), Y(0), X(0))}\n\n    Find equivalent gate identities from the current circuit with Muls:\n\n    >>> generate_equivalent_ids(x*x, return_as_muls=True)\n    {1}\n\n    >>> generate_equivalent_ids(x*y*z, return_as_muls=True)\n    {X(0)*Y(0)*Z(0), X(0)*Z(0)*Y(0), Y(0)*X(0)*Z(0),\n     Y(0)*Z(0)*X(0), Z(0)*X(0)*Y(0), Z(0)*Y(0)*X(0)}\n    '
    if isinstance(gate_seq, Number):
        return {S.One}
    elif isinstance(gate_seq, Mul):
        gate_seq = gate_seq.args
    eq_ids = set()
    gate_rules = generate_gate_rules(gate_seq)
    for rule in gate_rules:
        (l, r) = rule
        if l == ():
            eq_ids.add(r)
        elif r == ():
            eq_ids.add(l)
    if return_as_muls:
        convert_to_mul = lambda id_seq: Mul(*id_seq)
        eq_ids = set(map(convert_to_mul, eq_ids))
    return eq_ids

class GateIdentity(Basic):
    """Wrapper class for circuits that reduce to a scalar value.

    A gate identity is a quantum circuit such that the product
    of the gates in the circuit is equal to a scalar value.
    For example, XYZ = i, where X, Y, Z are the Pauli gates and
    i is the imaginary value, is considered a gate identity.

    Parameters
    ==========

    args : Gate tuple
        A variable length tuple of Gates that form an identity.

    Examples
    ========

    Create a GateIdentity and look at its attributes:

    >>> from sympy.physics.quantum.identitysearch import GateIdentity
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> an_identity = GateIdentity(x, y, z)
    >>> an_identity.circuit
    X(0)*Y(0)*Z(0)

    >>> an_identity.equivalent_ids
    {(X(0), Y(0), Z(0)), (X(0), Z(0), Y(0)), (Y(0), X(0), Z(0)),
     (Y(0), Z(0), X(0)), (Z(0), X(0), Y(0)), (Z(0), Y(0), X(0))}
    """

    def __new__(cls, *args):
        if False:
            i = 10
            return i + 15
        obj = Basic.__new__(cls, *args)
        obj._circuit = Mul(*args)
        obj._rules = generate_gate_rules(args)
        obj._eq_ids = generate_equivalent_ids(args)
        return obj

    @property
    def circuit(self):
        if False:
            return 10
        return self._circuit

    @property
    def gate_rules(self):
        if False:
            return 10
        return self._rules

    @property
    def equivalent_ids(self):
        if False:
            while True:
                i = 10
        return self._eq_ids

    @property
    def sequence(self):
        if False:
            print('Hello World!')
        return self.args

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the string of gates in a tuple.'
        return str(self.circuit)

def is_degenerate(identity_set, gate_identity):
    if False:
        print('Hello World!')
    'Checks if a gate identity is a permutation of another identity.\n\n    Parameters\n    ==========\n\n    identity_set : set\n        A Python set with GateIdentity objects.\n    gate_identity : GateIdentity\n        The GateIdentity to check for existence in the set.\n\n    Examples\n    ========\n\n    Check if the identity is a permutation of another identity:\n\n    >>> from sympy.physics.quantum.identitysearch import (\n    ...     GateIdentity, is_degenerate)\n    >>> from sympy.physics.quantum.gate import X, Y, Z\n    >>> x = X(0); y = Y(0); z = Z(0)\n    >>> an_identity = GateIdentity(x, y, z)\n    >>> id_set = {an_identity}\n    >>> another_id = (y, z, x)\n    >>> is_degenerate(id_set, another_id)\n    True\n\n    >>> another_id = (x, x)\n    >>> is_degenerate(id_set, another_id)\n    False\n    '
    for an_id in identity_set:
        if gate_identity in an_id.equivalent_ids:
            return True
    return False

def is_reducible(circuit, nqubits, begin, end):
    if False:
        i = 10
        return i + 15
    'Determines if a circuit is reducible by checking\n    if its subcircuits are scalar values.\n\n    Parameters\n    ==========\n\n    circuit : Gate tuple\n        A tuple of Gates representing a circuit.  The circuit to check\n        if a gate identity is contained in a subcircuit.\n    nqubits : int\n        The number of qubits the circuit operates on.\n    begin : int\n        The leftmost gate in the circuit to include in a subcircuit.\n    end : int\n        The rightmost gate in the circuit to include in a subcircuit.\n\n    Examples\n    ========\n\n    Check if the circuit can be reduced:\n\n    >>> from sympy.physics.quantum.identitysearch import is_reducible\n    >>> from sympy.physics.quantum.gate import X, Y, Z\n    >>> x = X(0); y = Y(0); z = Z(0)\n    >>> is_reducible((x, y, z), 1, 0, 3)\n    True\n\n    Check if an interval in the circuit can be reduced:\n\n    >>> is_reducible((x, y, z), 1, 1, 3)\n    False\n\n    >>> is_reducible((x, y, y), 1, 1, 3)\n    True\n    '
    current_circuit = ()
    for ndx in reversed(range(begin, end)):
        next_gate = circuit[ndx]
        current_circuit = (next_gate,) + current_circuit
        if is_scalar_matrix(current_circuit, nqubits, False):
            return True
    return False

def bfs_identity_search(gate_list, nqubits, max_depth=None, identity_only=False):
    if False:
        i = 10
        return i + 15
    'Constructs a set of gate identities from the list of possible gates.\n\n    Performs a breadth first search over the space of gate identities.\n    This allows the finding of the shortest gate identities first.\n\n    Parameters\n    ==========\n\n    gate_list : list, Gate\n        A list of Gates from which to search for gate identities.\n    nqubits : int\n        The number of qubits the quantum circuit operates on.\n    max_depth : int\n        The longest quantum circuit to construct from gate_list.\n    identity_only : bool\n        True to search for gate identities that reduce to identity;\n        False to search for gate identities that reduce to a scalar.\n\n    Examples\n    ========\n\n    Find a list of gate identities:\n\n    >>> from sympy.physics.quantum.identitysearch import bfs_identity_search\n    >>> from sympy.physics.quantum.gate import X, Y, Z\n    >>> x = X(0); y = Y(0); z = Z(0)\n    >>> bfs_identity_search([x], 1, max_depth=2)\n    {GateIdentity(X(0), X(0))}\n\n    >>> bfs_identity_search([x, y, z], 1)\n    {GateIdentity(X(0), X(0)), GateIdentity(Y(0), Y(0)),\n     GateIdentity(Z(0), Z(0)), GateIdentity(X(0), Y(0), Z(0))}\n\n    Find a list of identities that only equal to 1:\n\n    >>> bfs_identity_search([x, y, z], 1, identity_only=True)\n    {GateIdentity(X(0), X(0)), GateIdentity(Y(0), Y(0)),\n     GateIdentity(Z(0), Z(0))}\n    '
    if max_depth is None or max_depth <= 0:
        max_depth = len(gate_list)
    id_only = identity_only
    queue = deque([()])
    ids = set()
    while len(queue) > 0:
        current_circuit = queue.popleft()
        for next_gate in gate_list:
            new_circuit = current_circuit + (next_gate,)
            circuit_reducible = is_reducible(new_circuit, nqubits, 1, len(new_circuit))
            if is_scalar_matrix(new_circuit, nqubits, id_only) and (not is_degenerate(ids, new_circuit)) and (not circuit_reducible):
                ids.add(GateIdentity(*new_circuit))
            elif len(new_circuit) < max_depth and (not circuit_reducible):
                queue.append(new_circuit)
    return ids

def random_identity_search(gate_list, numgates, nqubits):
    if False:
        return 10
    'Randomly selects numgates from gate_list and checks if it is\n    a gate identity.\n\n    If the circuit is a gate identity, the circuit is returned;\n    Otherwise, None is returned.\n    '
    gate_size = len(gate_list)
    circuit = ()
    for i in range(numgates):
        next_gate = gate_list[randint(0, gate_size - 1)]
        circuit = circuit + (next_gate,)
    is_scalar = is_scalar_matrix(circuit, nqubits, False)
    return circuit if is_scalar else None