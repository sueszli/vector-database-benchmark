"""Quantum mechanical operators.

TODO:

* Fix early 0 in apply_operators.
* Debug and test apply_operators.
* Get cse working with classes in this file.
* Doctests and documentation of special methods for InnerProduct, Commutator,
  AntiCommutator, represent, apply_operators.
"""
from typing import Optional
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Derivative, expand
from sympy.core.mul import Mul
from sympy.core.numbers import oo
from sympy.core.singleton import S
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qexpr import QExpr, dispatch_method
from sympy.matrices import eye
__all__ = ['Operator', 'HermitianOperator', 'UnitaryOperator', 'IdentityOperator', 'OuterProduct', 'DifferentialOperator']

class Operator(QExpr):
    """Base class for non-commuting quantum operators.

    An operator maps between quantum states [1]_. In quantum mechanics,
    observables (including, but not limited to, measured physical values) are
    represented as Hermitian operators [2]_.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator. For time-dependent operators, this will include the time.

    Examples
    ========

    Create an operator and examine its attributes::

        >>> from sympy.physics.quantum import Operator
        >>> from sympy import I
        >>> A = Operator('A')
        >>> A
        A
        >>> A.hilbert_space
        H
        >>> A.label
        (A,)
        >>> A.is_commutative
        False

    Create another operator and do some arithmetic operations::

        >>> B = Operator('B')
        >>> C = 2*A*A + I*B
        >>> C
        2*A**2 + I*B

    Operators do not commute::

        >>> A.is_commutative
        False
        >>> B.is_commutative
        False
        >>> A*B == B*A
        False

    Polymonials of operators respect the commutation properties::

        >>> e = (A+B)**3
        >>> e.expand()
        A*B*A + A*B**2 + A**2*B + A**3 + B*A*B + B*A**2 + B**2*A + B**3

    Operator inverses are handle symbolically::

        >>> A.inv()
        A**(-1)
        >>> A*A.inv()
        1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Operator_%28physics%29
    .. [2] https://en.wikipedia.org/wiki/Observable
    """
    is_hermitian: Optional[bool] = None
    is_unitary: Optional[bool] = None

    @classmethod
    def default_args(self):
        if False:
            i = 10
            return i + 15
        return ('O',)
    _label_separator = ','

    def _print_operator_name(self, printer, *args):
        if False:
            while True:
                i = 10
        return self.__class__.__name__
    _print_operator_name_latex = _print_operator_name

    def _print_operator_name_pretty(self, printer, *args):
        if False:
            i = 10
            return i + 15
        return prettyForm(self.__class__.__name__)

    def _print_contents(self, printer, *args):
        if False:
            while True:
                i = 10
        if len(self.label) == 1:
            return self._print_label(printer, *args)
        else:
            return '%s(%s)' % (self._print_operator_name(printer, *args), self._print_label(printer, *args))

    def _print_contents_pretty(self, printer, *args):
        if False:
            return 10
        if len(self.label) == 1:
            return self._print_label_pretty(printer, *args)
        else:
            pform = self._print_operator_name_pretty(printer, *args)
            label_pform = self._print_label_pretty(printer, *args)
            label_pform = prettyForm(*label_pform.parens(left='(', right=')'))
            pform = prettyForm(*pform.right(label_pform))
            return pform

    def _print_contents_latex(self, printer, *args):
        if False:
            i = 10
            return i + 15
        if len(self.label) == 1:
            return self._print_label_latex(printer, *args)
        else:
            return '%s\\left(%s\\right)' % (self._print_operator_name_latex(printer, *args), self._print_label_latex(printer, *args))

    def _eval_commutator(self, other, **options):
        if False:
            i = 10
            return i + 15
        'Evaluate [self, other] if known, return None if not known.'
        return dispatch_method(self, '_eval_commutator', other, **options)

    def _eval_anticommutator(self, other, **options):
        if False:
            while True:
                i = 10
        'Evaluate [self, other] if known.'
        return dispatch_method(self, '_eval_anticommutator', other, **options)

    def _apply_operator(self, ket, **options):
        if False:
            return 10
        return dispatch_method(self, '_apply_operator', ket, **options)

    def _apply_from_right_to(self, bra, **options):
        if False:
            i = 10
            return i + 15
        return None

    def matrix_element(self, *args):
        if False:
            while True:
                i = 10
        raise NotImplementedError('matrix_elements is not defined')

    def inverse(self):
        if False:
            while True:
                i = 10
        return self._eval_inverse()
    inv = inverse

    def _eval_inverse(self):
        if False:
            print('Hello World!')
        return self ** (-1)

    def __mul__(self, other):
        if False:
            return 10
        if isinstance(other, IdentityOperator):
            return self
        return Mul(self, other)

class HermitianOperator(Operator):
    """A Hermitian operator that satisfies H == Dagger(H).

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator. For time-dependent operators, this will include the time.

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, HermitianOperator
    >>> H = HermitianOperator('H')
    >>> Dagger(H)
    H
    """
    is_hermitian = True

    def _eval_inverse(self):
        if False:
            print('Hello World!')
        if isinstance(self, UnitaryOperator):
            return self
        else:
            return Operator._eval_inverse(self)

    def _eval_power(self, exp):
        if False:
            i = 10
            return i + 15
        if isinstance(self, UnitaryOperator):
            if exp.is_even:
                from sympy.core.singleton import S
                return S.One
            elif exp.is_odd:
                return self
        return Operator._eval_power(self, exp)

class UnitaryOperator(Operator):
    """A unitary operator that satisfies U*Dagger(U) == 1.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator. For time-dependent operators, this will include the time.

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, UnitaryOperator
    >>> U = UnitaryOperator('U')
    >>> U*Dagger(U)
    1
    """
    is_unitary = True

    def _eval_adjoint(self):
        if False:
            for i in range(10):
                print('nop')
        return self._eval_inverse()

class IdentityOperator(Operator):
    """An identity operator I that satisfies op * I == I * op == op for any
    operator op.

    Parameters
    ==========

    N : Integer
        Optional parameter that specifies the dimension of the Hilbert space
        of operator. This is used when generating a matrix representation.

    Examples
    ========

    >>> from sympy.physics.quantum import IdentityOperator
    >>> IdentityOperator()
    I
    """
    is_hermitian = True
    is_unitary = True

    @property
    def dimension(self):
        if False:
            return 10
        return self.N

    @classmethod
    def default_args(self):
        if False:
            for i in range(10):
                print('nop')
        return (oo,)

    def __init__(self, *args, **hints):
        if False:
            print('Hello World!')
        if not len(args) in (0, 1):
            raise ValueError('0 or 1 parameters expected, got %s' % args)
        self.N = args[0] if len(args) == 1 and args[0] else oo

    def _eval_commutator(self, other, **hints):
        if False:
            return 10
        return S.Zero

    def _eval_anticommutator(self, other, **hints):
        if False:
            i = 10
            return i + 15
        return 2 * other

    def _eval_inverse(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def _eval_adjoint(self):
        if False:
            print('Hello World!')
        return self

    def _apply_operator(self, ket, **options):
        if False:
            while True:
                i = 10
        return ket

    def _apply_from_right_to(self, bra, **options):
        if False:
            i = 10
            return i + 15
        return bra

    def _eval_power(self, exp):
        if False:
            for i in range(10):
                print('nop')
        return self

    def _print_contents(self, printer, *args):
        if False:
            i = 10
            return i + 15
        return 'I'

    def _print_contents_pretty(self, printer, *args):
        if False:
            i = 10
            return i + 15
        return prettyForm('I')

    def _print_contents_latex(self, printer, *args):
        if False:
            while True:
                i = 10
        return '{\\mathcal{I}}'

    def __mul__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, (Operator, Dagger)):
            return other
        return Mul(self, other)

    def _represent_default_basis(self, **options):
        if False:
            return 10
        if not self.N or self.N == oo:
            raise NotImplementedError('Cannot represent infinite dimensional' + ' identity operator as a matrix')
        format = options.get('format', 'sympy')
        if format != 'sympy':
            raise NotImplementedError('Representation in format ' + '%s not implemented.' % format)
        return eye(self.N)

class OuterProduct(Operator):
    """An unevaluated outer product between a ket and bra.

    This constructs an outer product between any subclass of ``KetBase`` and
    ``BraBase`` as ``|a><b|``. An ``OuterProduct`` inherits from Operator as they act as
    operators in quantum expressions.  For reference see [1]_.

    Parameters
    ==========

    ket : KetBase
        The ket on the left side of the outer product.
    bar : BraBase
        The bra on the right side of the outer product.

    Examples
    ========

    Create a simple outer product by hand and take its dagger::

        >>> from sympy.physics.quantum import Ket, Bra, OuterProduct, Dagger
        >>> from sympy.physics.quantum import Operator

        >>> k = Ket('k')
        >>> b = Bra('b')
        >>> op = OuterProduct(k, b)
        >>> op
        |k><b|
        >>> op.hilbert_space
        H
        >>> op.ket
        |k>
        >>> op.bra
        <b|
        >>> Dagger(op)
        |b><k|

    In simple products of kets and bras outer products will be automatically
    identified and created::

        >>> k*b
        |k><b|

    But in more complex expressions, outer products are not automatically
    created::

        >>> A = Operator('A')
        >>> A*k*b
        A*|k>*<b|

    A user can force the creation of an outer product in a complex expression
    by using parentheses to group the ket and bra::

        >>> A*(k*b)
        A*|k><b|

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Outer_product
    """
    is_commutative = False

    def __new__(cls, *args, **old_assumptions):
        if False:
            return 10
        from sympy.physics.quantum.state import KetBase, BraBase
        if len(args) != 2:
            raise ValueError('2 parameters expected, got %d' % len(args))
        ket_expr = expand(args[0])
        bra_expr = expand(args[1])
        if isinstance(ket_expr, (KetBase, Mul)) and isinstance(bra_expr, (BraBase, Mul)):
            (ket_c, kets) = ket_expr.args_cnc()
            (bra_c, bras) = bra_expr.args_cnc()
            if len(kets) != 1 or not isinstance(kets[0], KetBase):
                raise TypeError('KetBase subclass expected, got: %r' % Mul(*kets))
            if len(bras) != 1 or not isinstance(bras[0], BraBase):
                raise TypeError('BraBase subclass expected, got: %r' % Mul(*bras))
            if not kets[0].dual_class() == bras[0].__class__:
                raise TypeError('ket and bra are not dual classes: %r, %r' % (kets[0].__class__, bras[0].__class__))
            obj = Expr.__new__(cls, *(kets[0], bras[0]), **old_assumptions)
            obj.hilbert_space = kets[0].hilbert_space
            return Mul(*ket_c + bra_c) * obj
        op_terms = []
        if isinstance(ket_expr, Add) and isinstance(bra_expr, Add):
            for ket_term in ket_expr.args:
                for bra_term in bra_expr.args:
                    op_terms.append(OuterProduct(ket_term, bra_term, **old_assumptions))
        elif isinstance(ket_expr, Add):
            for ket_term in ket_expr.args:
                op_terms.append(OuterProduct(ket_term, bra_expr, **old_assumptions))
        elif isinstance(bra_expr, Add):
            for bra_term in bra_expr.args:
                op_terms.append(OuterProduct(ket_expr, bra_term, **old_assumptions))
        else:
            raise TypeError('Expected ket and bra expression, got: %r, %r' % (ket_expr, bra_expr))
        return Add(*op_terms)

    @property
    def ket(self):
        if False:
            while True:
                i = 10
        'Return the ket on the left side of the outer product.'
        return self.args[0]

    @property
    def bra(self):
        if False:
            return 10
        'Return the bra on the right side of the outer product.'
        return self.args[1]

    def _eval_adjoint(self):
        if False:
            return 10
        return OuterProduct(Dagger(self.bra), Dagger(self.ket))

    def _sympystr(self, printer, *args):
        if False:
            for i in range(10):
                print('nop')
        return printer._print(self.ket) + printer._print(self.bra)

    def _sympyrepr(self, printer, *args):
        if False:
            i = 10
            return i + 15
        return '%s(%s,%s)' % (self.__class__.__name__, printer._print(self.ket, *args), printer._print(self.bra, *args))

    def _pretty(self, printer, *args):
        if False:
            print('Hello World!')
        pform = self.ket._pretty(printer, *args)
        return prettyForm(*pform.right(self.bra._pretty(printer, *args)))

    def _latex(self, printer, *args):
        if False:
            return 10
        k = printer._print(self.ket, *args)
        b = printer._print(self.bra, *args)
        return k + b

    def _represent(self, **options):
        if False:
            print('Hello World!')
        k = self.ket._represent(**options)
        b = self.bra._represent(**options)
        return k * b

    def _eval_trace(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return self.ket._eval_trace(self.bra, **kwargs)

class DifferentialOperator(Operator):
    """An operator for representing the differential operator, i.e. d/dx

    It is initialized by passing two arguments. The first is an arbitrary
    expression that involves a function, such as ``Derivative(f(x), x)``. The
    second is the function (e.g. ``f(x)``) which we are to replace with the
    ``Wavefunction`` that this ``DifferentialOperator`` is applied to.

    Parameters
    ==========

    expr : Expr
           The arbitrary expression which the appropriate Wavefunction is to be
           substituted into

    func : Expr
           A function (e.g. f(x)) which is to be replaced with the appropriate
           Wavefunction when this DifferentialOperator is applied

    Examples
    ========

    You can define a completely arbitrary expression and specify where the
    Wavefunction is to be substituted

    >>> from sympy import Derivative, Function, Symbol
    >>> from sympy.physics.quantum.operator import DifferentialOperator
    >>> from sympy.physics.quantum.state import Wavefunction
    >>> from sympy.physics.quantum.qapply import qapply
    >>> f = Function('f')
    >>> x = Symbol('x')
    >>> d = DifferentialOperator(1/x*Derivative(f(x), x), f(x))
    >>> w = Wavefunction(x**2, x)
    >>> d.function
    f(x)
    >>> d.variables
    (x,)
    >>> qapply(d*w)
    Wavefunction(2, x)

    """

    @property
    def variables(self):
        if False:
            print('Hello World!')
        "\n        Returns the variables with which the function in the specified\n        arbitrary expression is evaluated\n\n        Examples\n        ========\n\n        >>> from sympy.physics.quantum.operator import DifferentialOperator\n        >>> from sympy import Symbol, Function, Derivative\n        >>> x = Symbol('x')\n        >>> f = Function('f')\n        >>> d = DifferentialOperator(1/x*Derivative(f(x), x), f(x))\n        >>> d.variables\n        (x,)\n        >>> y = Symbol('y')\n        >>> d = DifferentialOperator(Derivative(f(x, y), x) +\n        ...                          Derivative(f(x, y), y), f(x, y))\n        >>> d.variables\n        (x, y)\n        "
        return self.args[-1].args

    @property
    def function(self):
        if False:
            print('Hello World!')
        "\n        Returns the function which is to be replaced with the Wavefunction\n\n        Examples\n        ========\n\n        >>> from sympy.physics.quantum.operator import DifferentialOperator\n        >>> from sympy import Function, Symbol, Derivative\n        >>> x = Symbol('x')\n        >>> f = Function('f')\n        >>> d = DifferentialOperator(Derivative(f(x), x), f(x))\n        >>> d.function\n        f(x)\n        >>> y = Symbol('y')\n        >>> d = DifferentialOperator(Derivative(f(x, y), x) +\n        ...                          Derivative(f(x, y), y), f(x, y))\n        >>> d.function\n        f(x, y)\n        "
        return self.args[-1]

    @property
    def expr(self):
        if False:
            while True:
                i = 10
        "\n        Returns the arbitrary expression which is to have the Wavefunction\n        substituted into it\n\n        Examples\n        ========\n\n        >>> from sympy.physics.quantum.operator import DifferentialOperator\n        >>> from sympy import Function, Symbol, Derivative\n        >>> x = Symbol('x')\n        >>> f = Function('f')\n        >>> d = DifferentialOperator(Derivative(f(x), x), f(x))\n        >>> d.expr\n        Derivative(f(x), x)\n        >>> y = Symbol('y')\n        >>> d = DifferentialOperator(Derivative(f(x, y), x) +\n        ...                          Derivative(f(x, y), y), f(x, y))\n        >>> d.expr\n        Derivative(f(x, y), x) + Derivative(f(x, y), y)\n        "
        return self.args[0]

    @property
    def free_symbols(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the free symbols of the expression.\n        '
        return self.expr.free_symbols

    def _apply_operator_Wavefunction(self, func, **options):
        if False:
            while True:
                i = 10
        from sympy.physics.quantum.state import Wavefunction
        var = self.variables
        wf_vars = func.args[1:]
        f = self.function
        new_expr = self.expr.subs(f, func(*var))
        new_expr = new_expr.doit()
        return Wavefunction(new_expr, *wf_vars)

    def _eval_derivative(self, symbol):
        if False:
            while True:
                i = 10
        new_expr = Derivative(self.expr, symbol)
        return DifferentialOperator(new_expr, self.args[-1])

    def _print(self, printer, *args):
        if False:
            while True:
                i = 10
        return '%s(%s)' % (self._print_operator_name(printer, *args), self._print_label(printer, *args))

    def _print_pretty(self, printer, *args):
        if False:
            return 10
        pform = self._print_operator_name_pretty(printer, *args)
        label_pform = self._print_label_pretty(printer, *args)
        label_pform = prettyForm(*label_pform.parens(left='(', right=')'))
        pform = prettyForm(*pform.right(label_pform))
        return pform