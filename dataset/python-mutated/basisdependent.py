from __future__ import annotations
from typing import TYPE_CHECKING
from sympy.simplify import simplify as simp, trigsimp as tsimp
from sympy.core.decorators import call_highest_priority, _sympifyit
from sympy.core.assumptions import StdFactKB
from sympy.core.function import diff as df
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import factor as fctr
from sympy.core import S, Add, Mul
from sympy.core.expr import Expr
if TYPE_CHECKING:
    from sympy.vector.vector import BaseVector

class BasisDependent(Expr):
    """
    Super class containing functionality common to vectors and
    dyadics.
    Named so because the representation of these quantities in
    sympy.vector is dependent on the basis they are expressed in.
    """
    zero: BasisDependentZero

    @call_highest_priority('__radd__')
    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._add_func(self, other)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        if False:
            while True:
                i = 10
        return self._add_func(other, self)

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._add_func(self, -other)

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        if False:
            i = 10
            return i + 15
        return self._add_func(other, -self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        if False:
            return 10
        return self._mul_func(self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        if False:
            print('Hello World!')
        return self._mul_func(other, self)

    def __neg__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._mul_func(S.NegativeOne, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        if False:
            while True:
                i = 10
        return self._div_helper(other)

    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        if False:
            while True:
                i = 10
        return TypeError('Invalid divisor for division')

    def evalf(self, n=15, subs=None, maxn=100, chop=False, strict=False, quad=None, verbose=False):
        if False:
            i = 10
            return i + 15
        "\n        Implements the SymPy evalf routine for this quantity.\n\n        evalf's documentation\n        =====================\n\n        "
        options = {'subs': subs, 'maxn': maxn, 'chop': chop, 'strict': strict, 'quad': quad, 'verbose': verbose}
        vec = self.zero
        for (k, v) in self.components.items():
            vec += v.evalf(n, **options) * k
        return vec
    evalf.__doc__ += Expr.evalf.__doc__
    n = evalf

    def simplify(self, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Implements the SymPy simplify routine for this quantity.\n\n        simplify's documentation\n        ========================\n\n        "
        simp_components = [simp(v, **kwargs) * k for (k, v) in self.components.items()]
        return self._add_func(*simp_components)
    simplify.__doc__ += simp.__doc__

    def trigsimp(self, **opts):
        if False:
            i = 10
            return i + 15
        "\n        Implements the SymPy trigsimp routine, for this quantity.\n\n        trigsimp's documentation\n        ========================\n\n        "
        trig_components = [tsimp(v, **opts) * k for (k, v) in self.components.items()]
        return self._add_func(*trig_components)
    trigsimp.__doc__ += tsimp.__doc__

    def _eval_simplify(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.simplify(**kwargs)

    def _eval_trigsimp(self, **opts):
        if False:
            while True:
                i = 10
        return self.trigsimp(**opts)

    def _eval_derivative(self, wrt):
        if False:
            i = 10
            return i + 15
        return self.diff(wrt)

    def _eval_Integral(self, *symbols, **assumptions):
        if False:
            for i in range(10):
                print('nop')
        integral_components = [Integral(v, *symbols, **assumptions) * k for (k, v) in self.components.items()]
        return self._add_func(*integral_components)

    def as_numer_denom(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the expression as a tuple wrt the following\n        transformation -\n\n        expression -> a/b -> a, b\n\n        '
        return (self, S.One)

    def factor(self, *args, **kwargs):
        if False:
            return 10
        "\n        Implements the SymPy factor routine, on the scalar parts\n        of a basis-dependent expression.\n\n        factor's documentation\n        ========================\n\n        "
        fctr_components = [fctr(v, *args, **kwargs) * k for (k, v) in self.components.items()]
        return self._add_func(*fctr_components)
    factor.__doc__ += fctr.__doc__

    def as_coeff_Mul(self, rational=False):
        if False:
            print('Hello World!')
        'Efficiently extract the coefficient of a product.'
        return (S.One, self)

    def as_coeff_add(self, *deps):
        if False:
            i = 10
            return i + 15
        'Efficiently extract the coefficient of a summation.'
        l = [x * self.components[x] for x in self.components]
        return (0, tuple(l))

    def diff(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Implements the SymPy diff routine, for vectors.\n\n        diff's documentation\n        ========================\n\n        "
        for x in args:
            if isinstance(x, BasisDependent):
                raise TypeError('Invalid arg for differentiation')
        diff_components = [df(v, *args, **kwargs) * k for (k, v) in self.components.items()]
        return self._add_func(*diff_components)
    diff.__doc__ += df.__doc__

    def doit(self, **hints):
        if False:
            i = 10
            return i + 15
        'Calls .doit() on each term in the Dyadic'
        doit_components = [self.components[x].doit(**hints) * x for x in self.components]
        return self._add_func(*doit_components)

class BasisDependentAdd(BasisDependent, Add):
    """
    Denotes sum of basis dependent quantities such that they cannot
    be expressed as base or Mul instances.
    """

    def __new__(cls, *args, **options):
        if False:
            while True:
                i = 10
        components = {}
        for (i, arg) in enumerate(args):
            if not isinstance(arg, cls._expr_type):
                if isinstance(arg, Mul):
                    arg = cls._mul_func(*arg.args)
                elif isinstance(arg, Add):
                    arg = cls._add_func(*arg.args)
                else:
                    raise TypeError(str(arg) + ' cannot be interpreted correctly')
            if arg == cls.zero:
                continue
            if hasattr(arg, 'components'):
                for x in arg.components:
                    components[x] = components.get(x, 0) + arg.components[x]
        temp = list(components.keys())
        for x in temp:
            if components[x] == 0:
                del components[x]
        if len(components) == 0:
            return cls.zero
        newargs = [x * components[x] for x in components]
        obj = super().__new__(cls, *newargs, **options)
        if isinstance(obj, Mul):
            return cls._mul_func(*obj.args)
        assumptions = {'commutative': True}
        obj._assumptions = StdFactKB(assumptions)
        obj._components = components
        obj._sys = list(components.keys())[0]._sys
        return obj

class BasisDependentMul(BasisDependent, Mul):
    """
    Denotes product of base- basis dependent quantity with a scalar.
    """

    def __new__(cls, *args, **options):
        if False:
            i = 10
            return i + 15
        from sympy.vector import Cross, Dot, Curl, Gradient
        count = 0
        measure_number = S.One
        zeroflag = False
        extra_args = []
        for arg in args:
            if isinstance(arg, cls._zero_func):
                count += 1
                zeroflag = True
            elif arg == S.Zero:
                zeroflag = True
            elif isinstance(arg, (cls._base_func, cls._mul_func)):
                count += 1
                expr = arg._base_instance
                measure_number *= arg._measure_number
            elif isinstance(arg, cls._add_func):
                count += 1
                expr = arg
            elif isinstance(arg, (Cross, Dot, Curl, Gradient)):
                extra_args.append(arg)
            else:
                measure_number *= arg
        if count > 1:
            raise ValueError('Invalid multiplication')
        elif count == 0:
            return Mul(*args, **options)
        if zeroflag:
            return cls.zero
        if isinstance(expr, cls._add_func):
            newargs = [cls._mul_func(measure_number, x) for x in expr.args]
            return cls._add_func(*newargs)
        obj = super().__new__(cls, measure_number, expr._base_instance, *extra_args, **options)
        if isinstance(obj, Add):
            return cls._add_func(*obj.args)
        obj._base_instance = expr._base_instance
        obj._measure_number = measure_number
        assumptions = {'commutative': True}
        obj._assumptions = StdFactKB(assumptions)
        obj._components = {expr._base_instance: measure_number}
        obj._sys = expr._base_instance._sys
        return obj

    def _sympystr(self, printer):
        if False:
            print('Hello World!')
        measure_str = printer._print(self._measure_number)
        if '(' in measure_str or '-' in measure_str or '+' in measure_str:
            measure_str = '(' + measure_str + ')'
        return measure_str + '*' + printer._print(self._base_instance)

class BasisDependentZero(BasisDependent):
    """
    Class to denote a zero basis dependent instance.
    """
    components: dict['BaseVector', Expr] = {}
    _latex_form: str

    def __new__(cls):
        if False:
            print('Hello World!')
        obj = super().__new__(cls)
        obj._hash = (S.Zero, cls).__hash__()
        return obj

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return self._hash

    @call_highest_priority('__req__')
    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, self._zero_func)
    __req__ = __eq__

    @call_highest_priority('__radd__')
    def __add__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError('Invalid argument types for addition')

    @call_highest_priority('__add__')
    def __radd__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError('Invalid argument types for addition')

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, self._expr_type):
            return -other
        else:
            raise TypeError('Invalid argument types for subtraction')

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError('Invalid argument types for subtraction')

    def __neg__(self):
        if False:
            while True:
                i = 10
        return self

    def normalize(self):
        if False:
            return 10
        '\n        Returns the normalized version of this vector.\n        '
        return self

    def _sympystr(self, printer):
        if False:
            return 10
        return '0'