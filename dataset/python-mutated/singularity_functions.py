from sympy.core import S, oo, diff
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.logic import fuzzy_not
from sympy.core.relational import Eq
from sympy.functions.elementary.complexes import im
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import Heaviside

class SingularityFunction(Function):
    """
    Singularity functions are a class of discontinuous functions.

    Explanation
    ===========

    Singularity functions take a variable, an offset, and an exponent as
    arguments. These functions are represented using Macaulay brackets as:

    SingularityFunction(x, a, n) := <x - a>^n

    The singularity function will automatically evaluate to
    ``Derivative(DiracDelta(x - a), x, -n - 1)`` if ``n < 0``
    and ``(x - a)**n*Heaviside(x - a, 1)`` if ``n >= 0``.

    Examples
    ========

    >>> from sympy import SingularityFunction, diff, Piecewise, DiracDelta, Heaviside, Symbol
    >>> from sympy.abc import x, a, n
    >>> SingularityFunction(x, a, n)
    SingularityFunction(x, a, n)
    >>> y = Symbol('y', positive=True)
    >>> n = Symbol('n', nonnegative=True)
    >>> SingularityFunction(y, -10, n)
    (y + 10)**n
    >>> y = Symbol('y', negative=True)
    >>> SingularityFunction(y, 10, n)
    0
    >>> SingularityFunction(x, 4, -1).subs(x, 4)
    oo
    >>> SingularityFunction(x, 10, -2).subs(x, 10)
    oo
    >>> SingularityFunction(4, 1, 5)
    243
    >>> diff(SingularityFunction(x, 1, 5) + SingularityFunction(x, 1, 4), x)
    4*SingularityFunction(x, 1, 3) + 5*SingularityFunction(x, 1, 4)
    >>> diff(SingularityFunction(x, 4, 0), x, 2)
    SingularityFunction(x, 4, -2)
    >>> SingularityFunction(x, 4, 5).rewrite(Piecewise)
    Piecewise(((x - 4)**5, x >= 4), (0, True))
    >>> expr = SingularityFunction(x, a, n)
    >>> y = Symbol('y', positive=True)
    >>> n = Symbol('n', nonnegative=True)
    >>> expr.subs({x: y, a: -10, n: n})
    (y + 10)**n

    The methods ``rewrite(DiracDelta)``, ``rewrite(Heaviside)``, and
    ``rewrite('HeavisideDiracDelta')`` returns the same output. One can use any
    of these methods according to their choice.

    >>> expr = SingularityFunction(x, 4, 5) + SingularityFunction(x, -3, -1) - SingularityFunction(x, 0, -2)
    >>> expr.rewrite(Heaviside)
    (x - 4)**5*Heaviside(x - 4, 1) + DiracDelta(x + 3) - DiracDelta(x, 1)
    >>> expr.rewrite(DiracDelta)
    (x - 4)**5*Heaviside(x - 4, 1) + DiracDelta(x + 3) - DiracDelta(x, 1)
    >>> expr.rewrite('HeavisideDiracDelta')
    (x - 4)**5*Heaviside(x - 4, 1) + DiracDelta(x + 3) - DiracDelta(x, 1)

    See Also
    ========

    DiracDelta, Heaviside

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Singularity_function

    """
    is_real = True

    def fdiff(self, argindex=1):
        if False:
            while True:
                i = 10
        '\n        Returns the first derivative of a DiracDelta Function.\n\n        Explanation\n        ===========\n\n        The difference between ``diff()`` and ``fdiff()`` is: ``diff()`` is the\n        user-level function and ``fdiff()`` is an object method. ``fdiff()`` is\n        a convenience method available in the ``Function`` class. It returns\n        the derivative of the function without considering the chain rule.\n        ``diff(function, x)`` calls ``Function._eval_derivative`` which in turn\n        calls ``fdiff()`` internally to compute the derivative of the function.\n\n        '
        if argindex == 1:
            (x, a, n) = self.args
            if n in (S.Zero, S.NegativeOne):
                return self.func(x, a, n - 1)
            elif n.is_positive:
                return n * self.func(x, a, n - 1)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, variable, offset, exponent):
        if False:
            return 10
        "\n        Returns a simplified form or a value of Singularity Function depending\n        on the argument passed by the object.\n\n        Explanation\n        ===========\n\n        The ``eval()`` method is automatically called when the\n        ``SingularityFunction`` class is about to be instantiated and it\n        returns either some simplified instance or the unevaluated instance\n        depending on the argument passed. In other words, ``eval()`` method is\n        not needed to be called explicitly, it is being called and evaluated\n        once the object is called.\n\n        Examples\n        ========\n\n        >>> from sympy import SingularityFunction, Symbol, nan\n        >>> from sympy.abc import x, a, n\n        >>> SingularityFunction(x, a, n)\n        SingularityFunction(x, a, n)\n        >>> SingularityFunction(5, 3, 2)\n        4\n        >>> SingularityFunction(x, a, nan)\n        nan\n        >>> SingularityFunction(x, 3, 0).subs(x, 3)\n        1\n        >>> SingularityFunction(4, 1, 5)\n        243\n        >>> x = Symbol('x', positive = True)\n        >>> a = Symbol('a', negative = True)\n        >>> n = Symbol('n', nonnegative = True)\n        >>> SingularityFunction(x, a, n)\n        (-a + x)**n\n        >>> x = Symbol('x', negative = True)\n        >>> a = Symbol('a', positive = True)\n        >>> SingularityFunction(x, a, n)\n        0\n\n        "
        x = variable
        a = offset
        n = exponent
        shift = x - a
        if fuzzy_not(im(shift).is_zero):
            raise ValueError('Singularity Functions are defined only for Real Numbers.')
        if fuzzy_not(im(n).is_zero):
            raise ValueError('Singularity Functions are not defined for imaginary exponents.')
        if shift is S.NaN or n is S.NaN:
            return S.NaN
        if (n + 2).is_negative:
            raise ValueError('Singularity Functions are not defined for exponents less than -2.')
        if shift.is_extended_negative:
            return S.Zero
        if n.is_nonnegative:
            if shift.is_zero:
                return S.Zero ** n
            if shift.is_extended_nonnegative:
                return shift ** n
        if n in (S.NegativeOne, -2):
            if shift.is_negative or shift.is_extended_positive:
                return S.Zero
            if shift.is_zero:
                return oo

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Converts a Singularity Function expression into its Piecewise form.\n\n        '
        (x, a, n) = self.args
        if n in (S.NegativeOne, S(-2)):
            return Piecewise((oo, Eq(x - a, 0)), (0, True))
        elif n.is_nonnegative:
            return Piecewise(((x - a) ** n, x - a >= 0), (0, True))

    def _eval_rewrite_as_Heaviside(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Rewrites a Singularity Function expression using Heavisides and DiracDeltas.\n\n        '
        (x, a, n) = self.args
        if n == -2:
            return diff(Heaviside(x - a), x.free_symbols.pop(), 2)
        if n == -1:
            return diff(Heaviside(x - a), x.free_symbols.pop(), 1)
        if n.is_nonnegative:
            return (x - a) ** n * Heaviside(x - a, 1)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        if False:
            i = 10
            return i + 15
        (z, a, n) = self.args
        shift = (z - a).subs(x, 0)
        if n < 0:
            return S.Zero
        elif n.is_zero and shift.is_zero:
            return S.Zero if cdir == -1 else S.One
        elif shift.is_positive:
            return shift ** n
        return S.Zero

    def _eval_nseries(self, x, n, logx=None, cdir=0):
        if False:
            print('Hello World!')
        (z, a, n) = self.args
        shift = (z - a).subs(x, 0)
        if n < 0:
            return S.Zero
        elif n.is_zero and shift.is_zero:
            return S.Zero if cdir == -1 else S.One
        elif shift.is_positive:
            return ((z - a) ** n)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return S.Zero
    _eval_rewrite_as_DiracDelta = _eval_rewrite_as_Heaviside
    _eval_rewrite_as_HeavisideDiracDelta = _eval_rewrite_as_Heaviside