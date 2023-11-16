"""Computations with ideals of polynomial rings."""
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable

class Ideal(IntegerPowerable):
    """
    Abstract base class for ideals.

    Do not instantiate - use explicit constructors in the ring class instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> QQ.old_poly_ring(x).ideal(x+1)
    <x + 1>

    Attributes

    - ring - the ring this ideal belongs to

    Non-implemented methods:

    - _contains_elem
    - _contains_ideal
    - _quotient
    - _intersect
    - _union
    - _product
    - is_whole_ring
    - is_zero
    - is_prime, is_maximal, is_primary, is_radical
    - is_principal
    - height, depth
    - radical

    Methods that likely should be overridden in subclasses:

    - reduce_element
    """

    def _contains_elem(self, x):
        if False:
            i = 10
            return i + 15
        'Implementation of element containment.'
        raise NotImplementedError

    def _contains_ideal(self, I):
        if False:
            for i in range(10):
                print('nop')
        'Implementation of ideal containment.'
        raise NotImplementedError

    def _quotient(self, J):
        if False:
            print('Hello World!')
        'Implementation of ideal quotient.'
        raise NotImplementedError

    def _intersect(self, J):
        if False:
            i = 10
            return i + 15
        'Implementation of ideal intersection.'
        raise NotImplementedError

    def is_whole_ring(self):
        if False:
            print('Hello World!')
        'Return True if ``self`` is the whole ring.'
        raise NotImplementedError

    def is_zero(self):
        if False:
            while True:
                i = 10
        'Return True if ``self`` is the zero ideal.'
        raise NotImplementedError

    def _equals(self, J):
        if False:
            i = 10
            return i + 15
        'Implementation of ideal equality.'
        return self._contains_ideal(J) and J._contains_ideal(self)

    def is_prime(self):
        if False:
            i = 10
            return i + 15
        'Return True if ``self`` is a prime ideal.'
        raise NotImplementedError

    def is_maximal(self):
        if False:
            return 10
        'Return True if ``self`` is a maximal ideal.'
        raise NotImplementedError

    def is_radical(self):
        if False:
            return 10
        'Return True if ``self`` is a radical ideal.'
        raise NotImplementedError

    def is_primary(self):
        if False:
            return 10
        'Return True if ``self`` is a primary ideal.'
        raise NotImplementedError

    def is_principal(self):
        if False:
            return 10
        'Return True if ``self`` is a principal ideal.'
        raise NotImplementedError

    def radical(self):
        if False:
            return 10
        'Compute the radical of ``self``.'
        raise NotImplementedError

    def depth(self):
        if False:
            for i in range(10):
                print('nop')
        'Compute the depth of ``self``.'
        raise NotImplementedError

    def height(self):
        if False:
            return 10
        'Compute the height of ``self``.'
        raise NotImplementedError

    def __init__(self, ring):
        if False:
            i = 10
            return i + 15
        self.ring = ring

    def _check_ideal(self, J):
        if False:
            print('Hello World!')
        'Helper to check ``J`` is an ideal of our ring.'
        if not isinstance(J, Ideal) or J.ring != self.ring:
            raise ValueError('J must be an ideal of %s, got %s' % (self.ring, J))

    def contains(self, elem):
        if False:
            i = 10
            return i + 15
        '\n        Return True if ``elem`` is an element of this ideal.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import x\n        >>> from sympy import QQ\n        >>> QQ.old_poly_ring(x).ideal(x+1, x-1).contains(3)\n        True\n        >>> QQ.old_poly_ring(x).ideal(x**2, x**3).contains(x)\n        False\n        '
        return self._contains_elem(self.ring.convert(elem))

    def subset(self, other):
        if False:
            while True:
                i = 10
        '\n        Returns True if ``other`` is is a subset of ``self``.\n\n        Here ``other`` may be an ideal.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import x\n        >>> from sympy import QQ\n        >>> I = QQ.old_poly_ring(x).ideal(x+1)\n        >>> I.subset([x**2 - 1, x**2 + 2*x + 1])\n        True\n        >>> I.subset([x**2 + 1, x + 1])\n        False\n        >>> I.subset(QQ.old_poly_ring(x).ideal(x**2 - 1))\n        True\n        '
        if isinstance(other, Ideal):
            return self._contains_ideal(other)
        return all((self._contains_elem(x) for x in other))

    def quotient(self, J, **opts):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute the ideal quotient of ``self`` by ``J``.\n\n        That is, if ``self`` is the ideal `I`, compute the set\n        `I : J = \\{x \\in R | xJ \\subset I \\}`.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import x, y\n        >>> from sympy import QQ\n        >>> R = QQ.old_poly_ring(x, y)\n        >>> R.ideal(x*y).quotient(R.ideal(x))\n        <y>\n        '
        self._check_ideal(J)
        return self._quotient(J, **opts)

    def intersect(self, J):
        if False:
            while True:
                i = 10
        '\n        Compute the intersection of self with ideal J.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import x, y\n        >>> from sympy import QQ\n        >>> R = QQ.old_poly_ring(x, y)\n        >>> R.ideal(x).intersect(R.ideal(y))\n        <x*y>\n        '
        self._check_ideal(J)
        return self._intersect(J)

    def saturate(self, J):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute the ideal saturation of ``self`` by ``J``.\n\n        That is, if ``self`` is the ideal `I`, compute the set\n        `I : J^\\infty = \\{x \\in R | xJ^n \\subset I \\text{ for some } n\\}`.\n        '
        raise NotImplementedError

    def union(self, J):
        if False:
            while True:
                i = 10
        '\n        Compute the ideal generated by the union of ``self`` and ``J``.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import x\n        >>> from sympy import QQ\n        >>> QQ.old_poly_ring(x).ideal(x**2 - 1).union(QQ.old_poly_ring(x).ideal((x+1)**2)) == QQ.old_poly_ring(x).ideal(x+1)\n        True\n        '
        self._check_ideal(J)
        return self._union(J)

    def product(self, J):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute the ideal product of ``self`` and ``J``.\n\n        That is, compute the ideal generated by products `xy`, for `x` an element\n        of ``self`` and `y \\in J`.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import x, y\n        >>> from sympy import QQ\n        >>> QQ.old_poly_ring(x, y).ideal(x).product(QQ.old_poly_ring(x, y).ideal(y))\n        <x*y>\n        '
        self._check_ideal(J)
        return self._product(J)

    def reduce_element(self, x):
        if False:
            i = 10
            return i + 15
        '\n        Reduce the element ``x`` of our ring modulo the ideal ``self``.\n\n        Here "reduce" has no specific meaning: it could return a unique normal\n        form, simplify the expression a bit, or just do nothing.\n        '
        return x

    def __add__(self, e):
        if False:
            return 10
        if not isinstance(e, Ideal):
            R = self.ring.quotient_ring(self)
            if isinstance(e, R.dtype):
                return e
            if isinstance(e, R.ring.dtype):
                return R(e)
            return R.convert(e)
        self._check_ideal(e)
        return self.union(e)
    __radd__ = __add__

    def __mul__(self, e):
        if False:
            i = 10
            return i + 15
        if not isinstance(e, Ideal):
            try:
                e = self.ring.ideal(e)
            except CoercionFailed:
                return NotImplemented
        self._check_ideal(e)
        return self.product(e)
    __rmul__ = __mul__

    def _zeroth_power(self):
        if False:
            return 10
        return self.ring.ideal(1)

    def _first_power(self):
        if False:
            for i in range(10):
                print('nop')
        return self * 1

    def __eq__(self, e):
        if False:
            while True:
                i = 10
        if not isinstance(e, Ideal) or e.ring != self.ring:
            return False
        return self._equals(e)

    def __ne__(self, e):
        if False:
            while True:
                i = 10
        return not self == e

class ModuleImplementedIdeal(Ideal):
    """
    Ideal implementation relying on the modules code.

    Attributes:

    - _module - the underlying module
    """

    def __init__(self, ring, module):
        if False:
            return 10
        Ideal.__init__(self, ring)
        self._module = module

    def _contains_elem(self, x):
        if False:
            print('Hello World!')
        return self._module.contains([x])

    def _contains_ideal(self, J):
        if False:
            i = 10
            return i + 15
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self._module.is_submodule(J._module)

    def _intersect(self, J):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self.__class__(self.ring, self._module.intersect(J._module))

    def _quotient(self, J, **opts):
        if False:
            i = 10
            return i + 15
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self._module.module_quotient(J._module, **opts)

    def _union(self, J):
        if False:
            i = 10
            return i + 15
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self.__class__(self.ring, self._module.union(J._module))

    @property
    def gens(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return generators for ``self``.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.abc import x, y\n        >>> list(QQ.old_poly_ring(x, y).ideal(x, y, x**2 + y).gens)\n        [DMP_Python([[1], []], QQ), DMP_Python([[1, 0]], QQ), DMP_Python([[1], [], [1, 0]], QQ)]\n        '
        return (x[0] for x in self._module.gens)

    def is_zero(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return True if ``self`` is the zero ideal.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import x\n        >>> from sympy import QQ\n        >>> QQ.old_poly_ring(x).ideal(x).is_zero()\n        False\n        >>> QQ.old_poly_ring(x).ideal().is_zero()\n        True\n        '
        return self._module.is_zero()

    def is_whole_ring(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return True if ``self`` is the whole ring, i.e. one generator is a unit.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import x\n        >>> from sympy import QQ, ilex\n        >>> QQ.old_poly_ring(x).ideal(x).is_whole_ring()\n        False\n        >>> QQ.old_poly_ring(x).ideal(3).is_whole_ring()\n        True\n        >>> QQ.old_poly_ring(x, order=ilex).ideal(2 + x).is_whole_ring()\n        True\n        '
        return self._module.is_full_module()

    def __repr__(self):
        if False:
            print('Hello World!')
        from sympy.printing.str import sstr
        gens = [self.ring.to_sympy(x) for [x] in self._module.gens]
        return '<' + ','.join((sstr(g) for g in gens)) + '>'

    def _product(self, J):
        if False:
            while True:
                i = 10
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self.__class__(self.ring, self._module.submodule(*[[x * y] for [x] in self._module.gens for [y] in J._module.gens]))

    def in_terms_of_generators(self, e):
        if False:
            for i in range(10):
                print('nop')
        '\n        Express ``e`` in terms of the generators of ``self``.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import x\n        >>> from sympy import QQ\n        >>> I = QQ.old_poly_ring(x).ideal(x**2 + 1, x)\n        >>> I.in_terms_of_generators(1)  # doctest: +SKIP\n        [DMP_Python([1], QQ), DMP_Python([-1, 0], QQ)]\n        '
        return self._module.in_terms_of_generators([e])

    def reduce_element(self, x, **options):
        if False:
            i = 10
            return i + 15
        return self._module.reduce_element([x], **options)[0]