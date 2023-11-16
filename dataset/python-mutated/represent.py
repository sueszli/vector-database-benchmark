"""Logic for representing operators in state in various bases.

TODO:

* Get represent working with continuous hilbert spaces.
* Document default basis functionality.
"""
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.integrals.integrals import integrate
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.matrixutils import flatten_scalar
from sympy.physics.quantum.state import KetBase, BraBase, StateBase
from sympy.physics.quantum.operator import Operator, OuterProduct
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.operatorset import operators_to_state, state_to_operators
__all__ = ['represent', 'rep_innerproduct', 'rep_expectation', 'integrate_result', 'get_basis', 'enumerate_states']

def _sympy_to_scalar(e):
    if False:
        return 10
    'Convert from a SymPy scalar to a Python scalar.'
    if isinstance(e, Expr):
        if e.is_Integer:
            return int(e)
        elif e.is_Float:
            return float(e)
        elif e.is_Rational:
            return float(e)
        elif e.is_Number or e.is_NumberSymbol or e == I:
            return complex(e)
    raise TypeError('Expected number, got: %r' % e)

def represent(expr, **options):
    if False:
        return 10
    "Represent the quantum expression in the given basis.\n\n    In quantum mechanics abstract states and operators can be represented in\n    various basis sets. Under this operation the follow transforms happen:\n\n    * Ket -> column vector or function\n    * Bra -> row vector of function\n    * Operator -> matrix or differential operator\n\n    This function is the top-level interface for this action.\n\n    This function walks the SymPy expression tree looking for ``QExpr``\n    instances that have a ``_represent`` method. This method is then called\n    and the object is replaced by the representation returned by this method.\n    By default, the ``_represent`` method will dispatch to other methods\n    that handle the representation logic for a particular basis set. The\n    naming convention for these methods is the following::\n\n        def _represent_FooBasis(self, e, basis, **options)\n\n    This function will have the logic for representing instances of its class\n    in the basis set having a class named ``FooBasis``.\n\n    Parameters\n    ==========\n\n    expr  : Expr\n        The expression to represent.\n    basis : Operator, basis set\n        An object that contains the information about the basis set. If an\n        operator is used, the basis is assumed to be the orthonormal\n        eigenvectors of that operator. In general though, the basis argument\n        can be any object that contains the basis set information.\n    options : dict\n        Key/value pairs of options that are passed to the underlying method\n        that finds the representation. These options can be used to\n        control how the representation is done. For example, this is where\n        the size of the basis set would be set.\n\n    Returns\n    =======\n\n    e : Expr\n        The SymPy expression of the represented quantum expression.\n\n    Examples\n    ========\n\n    Here we subclass ``Operator`` and ``Ket`` to create the z-spin operator\n    and its spin 1/2 up eigenstate. By defining the ``_represent_SzOp``\n    method, the ket can be represented in the z-spin basis.\n\n    >>> from sympy.physics.quantum import Operator, represent, Ket\n    >>> from sympy import Matrix\n\n    >>> class SzUpKet(Ket):\n    ...     def _represent_SzOp(self, basis, **options):\n    ...         return Matrix([1,0])\n    ...\n    >>> class SzOp(Operator):\n    ...     pass\n    ...\n    >>> sz = SzOp('Sz')\n    >>> up = SzUpKet('up')\n    >>> represent(up, basis=sz)\n    Matrix([\n    [1],\n    [0]])\n\n    Here we see an example of representations in a continuous\n    basis. We see that the result of representing various combinations\n    of cartesian position operators and kets give us continuous\n    expressions involving DiracDelta functions.\n\n    >>> from sympy.physics.quantum.cartesian import XOp, XKet, XBra\n    >>> X = XOp()\n    >>> x = XKet()\n    >>> y = XBra('y')\n    >>> represent(X*x)\n    x*DiracDelta(x - x_2)\n    >>> represent(X*x*y)\n    x*DiracDelta(x - x_3)*DiracDelta(x_1 - y)\n\n    "
    format = options.get('format', 'sympy')
    if format == 'numpy':
        import numpy as np
    if isinstance(expr, QExpr) and (not isinstance(expr, OuterProduct)):
        options['replace_none'] = False
        temp_basis = get_basis(expr, **options)
        if temp_basis is not None:
            options['basis'] = temp_basis
        try:
            return expr._represent(**options)
        except NotImplementedError as strerr:
            options['replace_none'] = True
            if isinstance(expr, (KetBase, BraBase)):
                try:
                    return rep_innerproduct(expr, **options)
                except NotImplementedError:
                    raise NotImplementedError(strerr)
            elif isinstance(expr, Operator):
                try:
                    return rep_expectation(expr, **options)
                except NotImplementedError:
                    raise NotImplementedError(strerr)
            else:
                raise NotImplementedError(strerr)
    elif isinstance(expr, Add):
        result = represent(expr.args[0], **options)
        for args in expr.args[1:]:
            result = result + represent(args, **options)
        return result
    elif isinstance(expr, Pow):
        (base, exp) = expr.as_base_exp()
        if format in ('numpy', 'scipy.sparse'):
            exp = _sympy_to_scalar(exp)
        base = represent(base, **options)
        if format == 'scipy.sparse' and exp < 0:
            from scipy.sparse.linalg import inv
            exp = -exp
            base = inv(base.tocsc()).tocsr()
        if format == 'numpy':
            return np.linalg.matrix_power(base, exp)
        return base ** exp
    elif isinstance(expr, TensorProduct):
        new_args = [represent(arg, **options) for arg in expr.args]
        return TensorProduct(*new_args)
    elif isinstance(expr, Dagger):
        return Dagger(represent(expr.args[0], **options))
    elif isinstance(expr, Commutator):
        A = expr.args[0]
        B = expr.args[1]
        return represent(Mul(A, B) - Mul(B, A), **options)
    elif isinstance(expr, AntiCommutator):
        A = expr.args[0]
        B = expr.args[1]
        return represent(Mul(A, B) + Mul(B, A), **options)
    elif isinstance(expr, InnerProduct):
        return represent(Mul(expr.bra, expr.ket), **options)
    elif not isinstance(expr, (Mul, OuterProduct)):
        if format in ('numpy', 'scipy.sparse'):
            return _sympy_to_scalar(expr)
        return expr
    if not isinstance(expr, (Mul, OuterProduct)):
        raise TypeError('Mul expected, got: %r' % expr)
    if 'index' in options:
        options['index'] += 1
    else:
        options['index'] = 1
    if 'unities' not in options:
        options['unities'] = []
    result = represent(expr.args[-1], **options)
    last_arg = expr.args[-1]
    for arg in reversed(expr.args[:-1]):
        if isinstance(last_arg, Operator):
            options['index'] += 1
            options['unities'].append(options['index'])
        elif isinstance(last_arg, BraBase) and isinstance(arg, KetBase):
            options['index'] += 1
        elif isinstance(last_arg, KetBase) and isinstance(arg, Operator):
            options['unities'].append(options['index'])
        elif isinstance(last_arg, KetBase) and isinstance(arg, BraBase):
            options['unities'].append(options['index'])
        next_arg = represent(arg, **options)
        if format == 'numpy' and isinstance(next_arg, np.ndarray):
            result = np.matmul(next_arg, result)
        else:
            result = next_arg * result
        last_arg = arg
    result = flatten_scalar(result)
    result = integrate_result(expr, result, **options)
    return result

def rep_innerproduct(expr, **options):
    if False:
        print('Hello World!')
    "\n    Returns an innerproduct like representation (e.g. ``<x'|x>``) for the\n    given state.\n\n    Attempts to calculate inner product with a bra from the specified\n    basis. Should only be passed an instance of KetBase or BraBase\n\n    Parameters\n    ==========\n\n    expr : KetBase or BraBase\n        The expression to be represented\n\n    Examples\n    ========\n\n    >>> from sympy.physics.quantum.represent import rep_innerproduct\n    >>> from sympy.physics.quantum.cartesian import XOp, XKet, PxOp, PxKet\n    >>> rep_innerproduct(XKet())\n    DiracDelta(x - x_1)\n    >>> rep_innerproduct(XKet(), basis=PxOp())\n    sqrt(2)*exp(-I*px_1*x/hbar)/(2*sqrt(hbar)*sqrt(pi))\n    >>> rep_innerproduct(PxKet(), basis=XOp())\n    sqrt(2)*exp(I*px*x_1/hbar)/(2*sqrt(hbar)*sqrt(pi))\n\n    "
    if not isinstance(expr, (KetBase, BraBase)):
        raise TypeError('expr passed is not a Bra or Ket')
    basis = get_basis(expr, **options)
    if not isinstance(basis, StateBase):
        raise NotImplementedError("Can't form this representation!")
    if 'index' not in options:
        options['index'] = 1
    basis_kets = enumerate_states(basis, options['index'], 2)
    if isinstance(expr, BraBase):
        bra = expr
        ket = basis_kets[1] if basis_kets[0].dual == expr else basis_kets[0]
    else:
        bra = basis_kets[1].dual if basis_kets[0] == expr else basis_kets[0].dual
        ket = expr
    prod = InnerProduct(bra, ket)
    result = prod.doit()
    format = options.get('format', 'sympy')
    return expr._format_represent(result, format)

def rep_expectation(expr, **options):
    if False:
        return 10
    "\n    Returns an ``<x'|A|x>`` type representation for the given operator.\n\n    Parameters\n    ==========\n\n    expr : Operator\n        Operator to be represented in the specified basis\n\n    Examples\n    ========\n\n    >>> from sympy.physics.quantum.cartesian import XOp, PxOp, PxKet\n    >>> from sympy.physics.quantum.represent import rep_expectation\n    >>> rep_expectation(XOp())\n    x_1*DiracDelta(x_1 - x_2)\n    >>> rep_expectation(XOp(), basis=PxOp())\n    <px_2|*X*|px_1>\n    >>> rep_expectation(XOp(), basis=PxKet())\n    <px_2|*X*|px_1>\n\n    "
    if 'index' not in options:
        options['index'] = 1
    if not isinstance(expr, Operator):
        raise TypeError('The passed expression is not an operator')
    basis_state = get_basis(expr, **options)
    if basis_state is None or not isinstance(basis_state, StateBase):
        raise NotImplementedError('Could not get basis kets for this operator')
    basis_kets = enumerate_states(basis_state, options['index'], 2)
    bra = basis_kets[1].dual
    ket = basis_kets[0]
    return qapply(bra * expr * ket)

def integrate_result(orig_expr, result, **options):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the result of integrating over any unities ``(|x><x|)`` in\n    the given expression. Intended for integrating over the result of\n    representations in continuous bases.\n\n    This function integrates over any unities that may have been\n    inserted into the quantum expression and returns the result.\n    It uses the interval of the Hilbert space of the basis state\n    passed to it in order to figure out the limits of integration.\n    The unities option must be\n    specified for this to work.\n\n    Note: This is mostly used internally by represent(). Examples are\n    given merely to show the use cases.\n\n    Parameters\n    ==========\n\n    orig_expr : quantum expression\n        The original expression which was to be represented\n\n    result: Expr\n        The resulting representation that we wish to integrate over\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, DiracDelta\n    >>> from sympy.physics.quantum.represent import integrate_result\n    >>> from sympy.physics.quantum.cartesian import XOp, XKet\n    >>> x_ket = XKet()\n    >>> X_op = XOp()\n    >>> x, x_1, x_2 = symbols('x, x_1, x_2')\n    >>> integrate_result(X_op*x_ket, x*DiracDelta(x-x_1)*DiracDelta(x_1-x_2))\n    x*DiracDelta(x - x_1)*DiracDelta(x_1 - x_2)\n    >>> integrate_result(X_op*x_ket, x*DiracDelta(x-x_1)*DiracDelta(x_1-x_2),\n    ...     unities=[1])\n    x*DiracDelta(x - x_2)\n\n    "
    if not isinstance(result, Expr):
        return result
    options['replace_none'] = True
    if 'basis' not in options:
        arg = orig_expr.args[-1]
        options['basis'] = get_basis(arg, **options)
    elif not isinstance(options['basis'], StateBase):
        options['basis'] = get_basis(orig_expr, **options)
    basis = options.pop('basis', None)
    if basis is None:
        return result
    unities = options.pop('unities', [])
    if len(unities) == 0:
        return result
    kets = enumerate_states(basis, unities)
    coords = [k.label[0] for k in kets]
    for coord in coords:
        if coord in result.free_symbols:
            basis_op = state_to_operators(basis)
            start = basis_op.hilbert_space.interval.start
            end = basis_op.hilbert_space.interval.end
            result = integrate(result, (coord, start, end))
    return result

def get_basis(expr, *, basis=None, replace_none=True, **options):
    if False:
        i = 10
        return i + 15
    "\n    Returns a basis state instance corresponding to the basis specified in\n    options=s. If no basis is specified, the function tries to form a default\n    basis state of the given expression.\n\n    There are three behaviors:\n\n    1. The basis specified in options is already an instance of StateBase. If\n       this is the case, it is simply returned. If the class is specified but\n       not an instance, a default instance is returned.\n\n    2. The basis specified is an operator or set of operators. If this\n       is the case, the operator_to_state mapping method is used.\n\n    3. No basis is specified. If expr is a state, then a default instance of\n       its class is returned.  If expr is an operator, then it is mapped to the\n       corresponding state.  If it is neither, then we cannot obtain the basis\n       state.\n\n    If the basis cannot be mapped, then it is not changed.\n\n    This will be called from within represent, and represent will\n    only pass QExpr's.\n\n    TODO (?): Support for Muls and other types of expressions?\n\n    Parameters\n    ==========\n\n    expr : Operator or StateBase\n        Expression whose basis is sought\n\n    Examples\n    ========\n\n    >>> from sympy.physics.quantum.represent import get_basis\n    >>> from sympy.physics.quantum.cartesian import XOp, XKet, PxOp, PxKet\n    >>> x = XKet()\n    >>> X = XOp()\n    >>> get_basis(x)\n    |x>\n    >>> get_basis(X)\n    |x>\n    >>> get_basis(x, basis=PxOp())\n    |px>\n    >>> get_basis(x, basis=PxKet)\n    |px>\n\n    "
    if basis is None and (not replace_none):
        return None
    if basis is None:
        if isinstance(expr, KetBase):
            return _make_default(expr.__class__)
        elif isinstance(expr, BraBase):
            return _make_default(expr.dual_class())
        elif isinstance(expr, Operator):
            state_inst = operators_to_state(expr)
            return state_inst if state_inst is not None else None
        else:
            return None
    elif isinstance(basis, Operator) or (not isinstance(basis, StateBase) and issubclass(basis, Operator)):
        state = operators_to_state(basis)
        if state is None:
            return None
        elif isinstance(state, StateBase):
            return state
        else:
            return _make_default(state)
    elif isinstance(basis, StateBase):
        return basis
    elif issubclass(basis, StateBase):
        return _make_default(basis)
    else:
        return None

def _make_default(expr):
    if False:
        for i in range(10):
            print('nop')
    try:
        expr = expr()
    except TypeError:
        return expr
    return expr

def enumerate_states(*args, **options):
    if False:
        print('Hello World!')
    "\n    Returns instances of the given state with dummy indices appended\n\n    Operates in two different modes:\n\n    1. Two arguments are passed to it. The first is the base state which is to\n       be indexed, and the second argument is a list of indices to append.\n\n    2. Three arguments are passed. The first is again the base state to be\n       indexed. The second is the start index for counting.  The final argument\n       is the number of kets you wish to receive.\n\n    Tries to call state._enumerate_state. If this fails, returns an empty list\n\n    Parameters\n    ==========\n\n    args : list\n        See list of operation modes above for explanation\n\n    Examples\n    ========\n\n    >>> from sympy.physics.quantum.cartesian import XBra, XKet\n    >>> from sympy.physics.quantum.represent import enumerate_states\n    >>> test = XKet('foo')\n    >>> enumerate_states(test, 1, 3)\n    [|foo_1>, |foo_2>, |foo_3>]\n    >>> test2 = XBra('bar')\n    >>> enumerate_states(test2, [4, 5, 10])\n    [<bar_4|, <bar_5|, <bar_10|]\n\n    "
    state = args[0]
    if len(args) not in (2, 3):
        raise NotImplementedError('Wrong number of arguments!')
    if not isinstance(state, StateBase):
        raise TypeError('First argument is not a state!')
    if len(args) == 3:
        num_states = args[2]
        options['start_index'] = args[1]
    else:
        num_states = len(args[1])
        options['index_list'] = args[1]
    try:
        ret = state._enumerate_state(num_states, **options)
    except NotImplementedError:
        ret = []
    return ret