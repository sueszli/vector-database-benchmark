from itertools import product
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.operator import HermitianOperator
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.matrixutils import numpy_ndarray, scipy_sparse_matrix, to_numpy
from sympy.physics.quantum.tensorproduct import TensorProduct, tensor_product_simp
from sympy.physics.quantum.trace import Tr

class Density(HermitianOperator):
    """Density operator for representing mixed states.

    TODO: Density operator support for Qubits

    Parameters
    ==========

    values : tuples/lists
    Each tuple/list should be of form (state, prob) or [state,prob]

    Examples
    ========

    Create a density operator with 2 states represented by Kets.

    >>> from sympy.physics.quantum.state import Ket
    >>> from sympy.physics.quantum.density import Density
    >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
    >>> d
    Density((|0>, 0.5),(|1>, 0.5))

    """

    @classmethod
    def _eval_args(cls, args):
        if False:
            i = 10
            return i + 15
        args = super()._eval_args(args)
        for arg in args:
            if not (isinstance(arg, Tuple) and len(arg) == 2):
                raise ValueError('Each argument should be of form [state,prob] or ( state, prob )')
        return args

    def states(self):
        if False:
            while True:
                i = 10
        'Return list of all states.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.quantum.state import Ket\n        >>> from sympy.physics.quantum.density import Density\n        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])\n        >>> d.states()\n        (|0>, |1>)\n\n        '
        return Tuple(*[arg[0] for arg in self.args])

    def probs(self):
        if False:
            print('Hello World!')
        'Return list of all probabilities.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.quantum.state import Ket\n        >>> from sympy.physics.quantum.density import Density\n        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])\n        >>> d.probs()\n        (0.5, 0.5)\n\n        '
        return Tuple(*[arg[1] for arg in self.args])

    def get_state(self, index):
        if False:
            i = 10
            return i + 15
        'Return specific state by index.\n\n        Parameters\n        ==========\n\n        index : index of state to be returned\n\n        Examples\n        ========\n\n        >>> from sympy.physics.quantum.state import Ket\n        >>> from sympy.physics.quantum.density import Density\n        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])\n        >>> d.states()[1]\n        |1>\n\n        '
        state = self.args[index][0]
        return state

    def get_prob(self, index):
        if False:
            i = 10
            return i + 15
        'Return probability of specific state by index.\n\n        Parameters\n        ===========\n\n        index : index of states whose probability is returned.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.quantum.state import Ket\n        >>> from sympy.physics.quantum.density import Density\n        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])\n        >>> d.probs()[1]\n        0.500000000000000\n\n        '
        prob = self.args[index][1]
        return prob

    def apply_op(self, op):
        if False:
            print('Hello World!')
        "op will operate on each individual state.\n\n        Parameters\n        ==========\n\n        op : Operator\n\n        Examples\n        ========\n\n        >>> from sympy.physics.quantum.state import Ket\n        >>> from sympy.physics.quantum.density import Density\n        >>> from sympy.physics.quantum.operator import Operator\n        >>> A = Operator('A')\n        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])\n        >>> d.apply_op(A)\n        Density((A*|0>, 0.5),(A*|1>, 0.5))\n\n        "
        new_args = [(op * state, prob) for (state, prob) in self.args]
        return Density(*new_args)

    def doit(self, **hints):
        if False:
            for i in range(10):
                print('nop')
        "Expand the density operator into an outer product format.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.quantum.state import Ket\n        >>> from sympy.physics.quantum.density import Density\n        >>> from sympy.physics.quantum.operator import Operator\n        >>> A = Operator('A')\n        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])\n        >>> d.doit()\n        0.5*|0><0| + 0.5*|1><1|\n\n        "
        terms = []
        for (state, prob) in self.args:
            state = state.expand()
            if isinstance(state, Add):
                for arg in product(state.args, repeat=2):
                    terms.append(prob * self._generate_outer_prod(arg[0], arg[1]))
            else:
                terms.append(prob * self._generate_outer_prod(state, state))
        return Add(*terms)

    def _generate_outer_prod(self, arg1, arg2):
        if False:
            i = 10
            return i + 15
        (c_part1, nc_part1) = arg1.args_cnc()
        (c_part2, nc_part2) = arg2.args_cnc()
        if len(nc_part1) == 0 or len(nc_part2) == 0:
            raise ValueError('Atleast one-pair of Non-commutative instance required for outer product.')
        if isinstance(nc_part1[0], TensorProduct) and len(nc_part1) == 1 and (len(nc_part2) == 1):
            op = tensor_product_simp(nc_part1[0] * Dagger(nc_part2[0]))
        else:
            op = Mul(*nc_part1) * Dagger(Mul(*nc_part2))
        return Mul(*c_part1) * Mul(*c_part2) * op

    def _represent(self, **options):
        if False:
            for i in range(10):
                print('nop')
        return represent(self.doit(), **options)

    def _print_operator_name_latex(self, printer, *args):
        if False:
            while True:
                i = 10
        return '\\rho'

    def _print_operator_name_pretty(self, printer, *args):
        if False:
            return 10
        return prettyForm('ρ')

    def _eval_trace(self, **kwargs):
        if False:
            i = 10
            return i + 15
        indices = kwargs.get('indices', [])
        return Tr(self.doit(), indices).doit()

    def entropy(self):
        if False:
            while True:
                i = 10
        ' Compute the entropy of a density matrix.\n\n        Refer to density.entropy() method  for examples.\n        '
        return entropy(self)

def entropy(density):
    if False:
        for i in range(10):
            print('nop')
    'Compute the entropy of a matrix/density object.\n\n    This computes -Tr(density*ln(density)) using the eigenvalue decomposition\n    of density, which is given as either a Density instance or a matrix\n    (numpy.ndarray, sympy.Matrix or scipy.sparse).\n\n    Parameters\n    ==========\n\n    density : density matrix of type Density, SymPy matrix,\n    scipy.sparse or numpy.ndarray\n\n    Examples\n    ========\n\n    >>> from sympy.physics.quantum.density import Density, entropy\n    >>> from sympy.physics.quantum.spin import JzKet\n    >>> from sympy import S\n    >>> up = JzKet(S(1)/2,S(1)/2)\n    >>> down = JzKet(S(1)/2,-S(1)/2)\n    >>> d = Density((up,S(1)/2),(down,S(1)/2))\n    >>> entropy(d)\n    log(2)/2\n\n    '
    if isinstance(density, Density):
        density = represent(density)
    if isinstance(density, scipy_sparse_matrix):
        density = to_numpy(density)
    if isinstance(density, Matrix):
        eigvals = density.eigenvals().keys()
        return expand(-sum((e * log(e) for e in eigvals)))
    elif isinstance(density, numpy_ndarray):
        import numpy as np
        eigvals = np.linalg.eigvals(density)
        return -np.sum(eigvals * np.log(eigvals))
    else:
        raise ValueError('numpy.ndarray, scipy.sparse or SymPy matrix expected')

def fidelity(state1, state2):
    if False:
        print('Hello World!')
    ' Computes the fidelity [1]_ between two quantum states\n\n    The arguments provided to this function should be a square matrix or a\n    Density object. If it is a square matrix, it is assumed to be diagonalizable.\n\n    Parameters\n    ==========\n\n    state1, state2 : a density matrix or Matrix\n\n\n    Examples\n    ========\n\n    >>> from sympy import S, sqrt\n    >>> from sympy.physics.quantum.dagger import Dagger\n    >>> from sympy.physics.quantum.spin import JzKet\n    >>> from sympy.physics.quantum.density import fidelity\n    >>> from sympy.physics.quantum.represent import represent\n    >>>\n    >>> up = JzKet(S(1)/2,S(1)/2)\n    >>> down = JzKet(S(1)/2,-S(1)/2)\n    >>> amp = 1/sqrt(2)\n    >>> updown = (amp*up) + (amp*down)\n    >>>\n    >>> # represent turns Kets into matrices\n    >>> up_dm = represent(up*Dagger(up))\n    >>> down_dm = represent(down*Dagger(down))\n    >>> updown_dm = represent(updown*Dagger(updown))\n    >>>\n    >>> fidelity(up_dm, up_dm)\n    1\n    >>> fidelity(up_dm, down_dm) #orthogonal states\n    0\n    >>> fidelity(up_dm, updown_dm).evalf().round(3)\n    0.707\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Fidelity_of_quantum_states\n\n    '
    state1 = represent(state1) if isinstance(state1, Density) else state1
    state2 = represent(state2) if isinstance(state2, Density) else state2
    if not isinstance(state1, Matrix) or not isinstance(state2, Matrix):
        raise ValueError('state1 and state2 must be of type Density or Matrix received type=%s for state1 and type=%s for state2' % (type(state1), type(state2)))
    if state1.shape != state2.shape and state1.is_square:
        raise ValueError('The dimensions of both args should be equal and the matrix obtained should be a square matrix')
    sqrt_state1 = state1 ** S.Half
    return Tr((sqrt_state1 * state2 * sqrt_state1) ** S.Half).doit()