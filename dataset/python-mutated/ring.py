"""Implementation of :class:`Ring` class. """
from sympy.polys.domains.domain import Domain
from sympy.polys.polyerrors import ExactQuotientFailed, NotInvertible, NotReversible
from sympy.utilities import public

@public
class Ring(Domain):
    """Represents a ring domain. """
    is_Ring = True

    def get_ring(self):
        if False:
            while True:
                i = 10
        'Returns a ring associated with ``self``. '
        return self

    def exquo(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        'Exact quotient of ``a`` and ``b``, implies ``__floordiv__``.  '
        if a % b:
            raise ExactQuotientFailed(a, b, self)
        else:
            return a // b

    def quo(self, a, b):
        if False:
            print('Hello World!')
        'Quotient of ``a`` and ``b``, implies ``__floordiv__``. '
        return a // b

    def rem(self, a, b):
        if False:
            return 10
        'Remainder of ``a`` and ``b``, implies ``__mod__``.  '
        return a % b

    def div(self, a, b):
        if False:
            while True:
                i = 10
        'Division of ``a`` and ``b``, implies ``__divmod__``. '
        return divmod(a, b)

    def invert(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        'Returns inversion of ``a mod b``. '
        (s, t, h) = self.gcdex(a, b)
        if self.is_one(h):
            return s % b
        else:
            raise NotInvertible('zero divisor')

    def revert(self, a):
        if False:
            print('Hello World!')
        'Returns ``a**(-1)`` if possible. '
        if self.is_one(a) or self.is_one(-a):
            return a
        else:
            raise NotReversible('only units are reversible in a ring')

    def is_unit(self, a):
        if False:
            print('Hello World!')
        try:
            self.revert(a)
            return True
        except NotReversible:
            return False

    def numer(self, a):
        if False:
            for i in range(10):
                print('nop')
        'Returns numerator of ``a``. '
        return a

    def denom(self, a):
        if False:
            i = 10
            return i + 15
        'Returns denominator of `a`. '
        return self.one

    def free_module(self, rank):
        if False:
            i = 10
            return i + 15
        '\n        Generate a free module of rank ``rank`` over self.\n\n        >>> from sympy.abc import x\n        >>> from sympy import QQ\n        >>> QQ.old_poly_ring(x).free_module(2)\n        QQ[x]**2\n        '
        raise NotImplementedError

    def ideal(self, *gens):
        if False:
            i = 10
            return i + 15
        '\n        Generate an ideal of ``self``.\n\n        >>> from sympy.abc import x\n        >>> from sympy import QQ\n        >>> QQ.old_poly_ring(x).ideal(x**2)\n        <x**2>\n        '
        from sympy.polys.agca.ideals import ModuleImplementedIdeal
        return ModuleImplementedIdeal(self, self.free_module(1).submodule(*[[x] for x in gens]))

    def quotient_ring(self, e):
        if False:
            print('Hello World!')
        '\n        Form a quotient ring of ``self``.\n\n        Here ``e`` can be an ideal or an iterable.\n\n        >>> from sympy.abc import x\n        >>> from sympy import QQ\n        >>> QQ.old_poly_ring(x).quotient_ring(QQ.old_poly_ring(x).ideal(x**2))\n        QQ[x]/<x**2>\n        >>> QQ.old_poly_ring(x).quotient_ring([x**2])\n        QQ[x]/<x**2>\n\n        The division operator has been overloaded for this:\n\n        >>> QQ.old_poly_ring(x)/[x**2]\n        QQ[x]/<x**2>\n        '
        from sympy.polys.agca.ideals import Ideal
        from sympy.polys.domains.quotientring import QuotientRing
        if not isinstance(e, Ideal):
            e = self.ideal(*e)
        return QuotientRing(self, e)

    def __truediv__(self, e):
        if False:
            i = 10
            return i + 15
        return self.quotient_ring(e)