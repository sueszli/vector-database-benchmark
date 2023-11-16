"""Grover's algorithm and helper functions.

Todo:

* W gate construction (or perhaps -W gate based on Mermin's book)
* Generalize the algorithm for an unknown function that returns 1 on multiple
  qubit states, not just one.
* Implement _represent_ZGate in OracleGate
"""
from sympy.core.numbers import pi
from sympy.core.sympify import sympify
from sympy.core.basic import Atom
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import eye
from sympy.core.numbers import NegativeOne
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.operator import UnitaryOperator
from sympy.physics.quantum.gate import Gate
from sympy.physics.quantum.qubit import IntQubit
__all__ = ['OracleGate', 'WGate', 'superposition_basis', 'grover_iteration', 'apply_grover']

def superposition_basis(nqubits):
    if False:
        return 10
    'Creates an equal superposition of the computational basis.\n\n    Parameters\n    ==========\n\n    nqubits : int\n        The number of qubits.\n\n    Returns\n    =======\n\n    state : Qubit\n        An equal superposition of the computational basis with nqubits.\n\n    Examples\n    ========\n\n    Create an equal superposition of 2 qubits::\n\n        >>> from sympy.physics.quantum.grover import superposition_basis\n        >>> superposition_basis(2)\n        |0>/2 + |1>/2 + |2>/2 + |3>/2\n    '
    amp = 1 / sqrt(2 ** nqubits)
    return sum([amp * IntQubit(n, nqubits=nqubits) for n in range(2 ** nqubits)])

class OracleGateFunction(Atom):
    """Wrapper for python functions used in `OracleGate`s"""

    def __new__(cls, function):
        if False:
            while True:
                i = 10
        if not callable(function):
            raise TypeError('Callable expected, got: %r' % function)
        obj = Atom.__new__(cls)
        obj.function = function
        return obj

    def _hashable_content(self):
        if False:
            return 10
        return (type(self), self.function)

    def __call__(self, *args):
        if False:
            while True:
                i = 10
        return self.function(*args)

class OracleGate(Gate):
    """A black box gate.

    The gate marks the desired qubits of an unknown function by flipping
    the sign of the qubits.  The unknown function returns true when it
    finds its desired qubits and false otherwise.

    Parameters
    ==========

    qubits : int
        Number of qubits.

    oracle : callable
        A callable function that returns a boolean on a computational basis.

    Examples
    ========

    Apply an Oracle gate that flips the sign of ``|2>`` on different qubits::

        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.qapply import qapply
        >>> from sympy.physics.quantum.grover import OracleGate
        >>> f = lambda qubits: qubits == IntQubit(2)
        >>> v = OracleGate(2, f)
        >>> qapply(v*IntQubit(2))
        -|2>
        >>> qapply(v*IntQubit(3))
        |3>
    """
    gate_name = 'V'
    gate_name_latex = 'V'

    @classmethod
    def _eval_args(cls, args):
        if False:
            for i in range(10):
                print('nop')
        if len(args) != 2:
            raise QuantumError('Insufficient/excessive arguments to Oracle.  Please ' + 'supply the number of qubits and an unknown function.')
        sub_args = (args[0],)
        sub_args = UnitaryOperator._eval_args(sub_args)
        if not sub_args[0].is_Integer:
            raise TypeError('Integer expected, got: %r' % sub_args[0])
        function = args[1]
        if not isinstance(function, OracleGateFunction):
            function = OracleGateFunction(function)
        return (sub_args[0], function)

    @classmethod
    def _eval_hilbert_space(cls, args):
        if False:
            i = 10
            return i + 15
        'This returns the smallest possible Hilbert space.'
        return ComplexSpace(2) ** args[0]

    @property
    def search_function(self):
        if False:
            return 10
        'The unknown function that helps find the sought after qubits.'
        return self.label[1]

    @property
    def targets(self):
        if False:
            print('Hello World!')
        'A tuple of target qubits.'
        return sympify(tuple(range(self.args[0])))

    def _apply_operator_Qubit(self, qubits, **options):
        if False:
            return 10
        'Apply this operator to a Qubit subclass.\n\n        Parameters\n        ==========\n\n        qubits : Qubit\n            The qubit subclass to apply this operator to.\n\n        Returns\n        =======\n\n        state : Expr\n            The resulting quantum state.\n        '
        if qubits.nqubits != self.nqubits:
            raise QuantumError('OracleGate operates on %r qubits, got: %r' % (self.nqubits, qubits.nqubits))
        if self.search_function(qubits):
            return -qubits
        else:
            return qubits

    def _represent_ZGate(self, basis, **options):
        if False:
            for i in range(10):
                print('nop')
        '\n        Represent the OracleGate in the computational basis.\n        '
        nbasis = 2 ** self.nqubits
        matrixOracle = eye(nbasis)
        for i in range(nbasis):
            if self.search_function(IntQubit(i, nqubits=self.nqubits)):
                matrixOracle[i, i] = NegativeOne()
        return matrixOracle

class WGate(Gate):
    """General n qubit W Gate in Grover's algorithm.

    The gate performs the operation ``2|phi><phi| - 1`` on some qubits.
    ``|phi> = (tensor product of n Hadamards)*(|0> with n qubits)``

    Parameters
    ==========

    nqubits : int
        The number of qubits to operate on

    """
    gate_name = 'W'
    gate_name_latex = 'W'

    @classmethod
    def _eval_args(cls, args):
        if False:
            for i in range(10):
                print('nop')
        if len(args) != 1:
            raise QuantumError('Insufficient/excessive arguments to W gate.  Please ' + 'supply the number of qubits to operate on.')
        args = UnitaryOperator._eval_args(args)
        if not args[0].is_Integer:
            raise TypeError('Integer expected, got: %r' % args[0])
        return args

    @property
    def targets(self):
        if False:
            for i in range(10):
                print('nop')
        return sympify(tuple(reversed(range(self.args[0]))))

    def _apply_operator_Qubit(self, qubits, **options):
        if False:
            print('Hello World!')
        '\n        qubits: a set of qubits (Qubit)\n        Returns: quantum object (quantum expression - QExpr)\n        '
        if qubits.nqubits != self.nqubits:
            raise QuantumError('WGate operates on %r qubits, got: %r' % (self.nqubits, qubits.nqubits))
        basis_states = superposition_basis(self.nqubits)
        change_to_basis = 2 / sqrt(2 ** self.nqubits) * basis_states
        return change_to_basis - qubits

def grover_iteration(qstate, oracle):
    if False:
        for i in range(10):
            print('nop')
    "Applies one application of the Oracle and W Gate, WV.\n\n    Parameters\n    ==========\n\n    qstate : Qubit\n        A superposition of qubits.\n    oracle : OracleGate\n        The black box operator that flips the sign of the desired basis qubits.\n\n    Returns\n    =======\n\n    Qubit : The qubits after applying the Oracle and W gate.\n\n    Examples\n    ========\n\n    Perform one iteration of grover's algorithm to see a phase change::\n\n        >>> from sympy.physics.quantum.qapply import qapply\n        >>> from sympy.physics.quantum.qubit import IntQubit\n        >>> from sympy.physics.quantum.grover import OracleGate\n        >>> from sympy.physics.quantum.grover import superposition_basis\n        >>> from sympy.physics.quantum.grover import grover_iteration\n        >>> numqubits = 2\n        >>> basis_states = superposition_basis(numqubits)\n        >>> f = lambda qubits: qubits == IntQubit(2)\n        >>> v = OracleGate(numqubits, f)\n        >>> qapply(grover_iteration(basis_states, v))\n        |2>\n\n    "
    wgate = WGate(oracle.nqubits)
    return wgate * oracle * qstate

def apply_grover(oracle, nqubits, iterations=None):
    if False:
        i = 10
        return i + 15
    "Applies grover's algorithm.\n\n    Parameters\n    ==========\n\n    oracle : callable\n        The unknown callable function that returns true when applied to the\n        desired qubits and false otherwise.\n\n    Returns\n    =======\n\n    state : Expr\n        The resulting state after Grover's algorithm has been iterated.\n\n    Examples\n    ========\n\n    Apply grover's algorithm to an even superposition of 2 qubits::\n\n        >>> from sympy.physics.quantum.qapply import qapply\n        >>> from sympy.physics.quantum.qubit import IntQubit\n        >>> from sympy.physics.quantum.grover import apply_grover\n        >>> f = lambda qubits: qubits == IntQubit(2)\n        >>> qapply(apply_grover(f, 2))\n        |2>\n\n    "
    if nqubits <= 0:
        raise QuantumError("Grover's algorithm needs nqubits > 0, received %r qubits" % nqubits)
    if iterations is None:
        iterations = floor(sqrt(2 ** nqubits) * (pi / 4))
    v = OracleGate(nqubits, oracle)
    iterated = superposition_basis(nqubits)
    for iter in range(iterations):
        iterated = grover_iteration(iterated, v)
        iterated = qapply(iterated)
    return iterated