"""
Computations with homomorphisms of modules and rings.

This module implements classes for representing homomorphisms of rings and
their modules. Instead of instantiating the classes directly, you should use
the function ``homomorphism(from, to, matrix)`` to create homomorphism objects.
"""
from sympy.polys.agca.modules import Module, FreeModule, QuotientModule, SubModule, SubQuotientModule
from sympy.polys.polyerrors import CoercionFailed

class ModuleHomomorphism:
    """
    Abstract base class for module homomoprhisms. Do not instantiate.

    Instead, use the ``homomorphism`` function:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.agca import homomorphism

    >>> F = QQ.old_poly_ring(x).free_module(2)
    >>> homomorphism(F, F, [[1, 0], [0, 1]])
    Matrix([
    [1, 0], : QQ[x]**2 -> QQ[x]**2
    [0, 1]])

    Attributes:

    - ring - the ring over which we are considering modules
    - domain - the domain module
    - codomain - the codomain module
    - _ker - cached kernel
    - _img - cached image

    Non-implemented methods:

    - _kernel
    - _image
    - _restrict_domain
    - _restrict_codomain
    - _quotient_domain
    - _quotient_codomain
    - _apply
    - _mul_scalar
    - _compose
    - _add
    """

    def __init__(self, domain, codomain):
        if False:
            return 10
        if not isinstance(domain, Module):
            raise TypeError('Source must be a module, got %s' % domain)
        if not isinstance(codomain, Module):
            raise TypeError('Target must be a module, got %s' % codomain)
        if domain.ring != codomain.ring:
            raise ValueError('Source and codomain must be over same ring, got %s != %s' % (domain, codomain))
        self.domain = domain
        self.codomain = codomain
        self.ring = domain.ring
        self._ker = None
        self._img = None

    def kernel(self):
        if False:
            return 10
        '\n        Compute the kernel of ``self``.\n\n        That is, if ``self`` is the homomorphism `\\phi: M \\to N`, then compute\n        `ker(\\phi) = \\{x \\in M | \\phi(x) = 0\\}`.  This is a submodule of `M`.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.abc import x\n        >>> from sympy.polys.agca import homomorphism\n\n        >>> F = QQ.old_poly_ring(x).free_module(2)\n        >>> homomorphism(F, F, [[1, 0], [x, 0]]).kernel()\n        <[x, -1]>\n        '
        if self._ker is None:
            self._ker = self._kernel()
        return self._ker

    def image(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute the image of ``self``.\n\n        That is, if ``self`` is the homomorphism `\\phi: M \\to N`, then compute\n        `im(\\phi) = \\{\\phi(x) | x \\in M \\}`.  This is a submodule of `N`.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.abc import x\n        >>> from sympy.polys.agca import homomorphism\n\n        >>> F = QQ.old_poly_ring(x).free_module(2)\n        >>> homomorphism(F, F, [[1, 0], [x, 0]]).image() == F.submodule([1, 0])\n        True\n        '
        if self._img is None:
            self._img = self._image()
        return self._img

    def _kernel(self):
        if False:
            print('Hello World!')
        'Compute the kernel of ``self``.'
        raise NotImplementedError

    def _image(self):
        if False:
            return 10
        'Compute the image of ``self``.'
        raise NotImplementedError

    def _restrict_domain(self, sm):
        if False:
            print('Hello World!')
        'Implementation of domain restriction.'
        raise NotImplementedError

    def _restrict_codomain(self, sm):
        if False:
            print('Hello World!')
        'Implementation of codomain restriction.'
        raise NotImplementedError

    def _quotient_domain(self, sm):
        if False:
            i = 10
            return i + 15
        'Implementation of domain quotient.'
        raise NotImplementedError

    def _quotient_codomain(self, sm):
        if False:
            for i in range(10):
                print('nop')
        'Implementation of codomain quotient.'
        raise NotImplementedError

    def restrict_domain(self, sm):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return ``self``, with the domain restricted to ``sm``.\n\n        Here ``sm`` has to be a submodule of ``self.domain``.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.abc import x\n        >>> from sympy.polys.agca import homomorphism\n\n        >>> F = QQ.old_poly_ring(x).free_module(2)\n        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])\n        >>> h\n        Matrix([\n        [1, x], : QQ[x]**2 -> QQ[x]**2\n        [0, 0]])\n        >>> h.restrict_domain(F.submodule([1, 0]))\n        Matrix([\n        [1, x], : <[1, 0]> -> QQ[x]**2\n        [0, 0]])\n\n        This is the same as just composing on the right with the submodule\n        inclusion:\n\n        >>> h * F.submodule([1, 0]).inclusion_hom()\n        Matrix([\n        [1, x], : <[1, 0]> -> QQ[x]**2\n        [0, 0]])\n        '
        if not self.domain.is_submodule(sm):
            raise ValueError('sm must be a submodule of %s, got %s' % (self.domain, sm))
        if sm == self.domain:
            return self
        return self._restrict_domain(sm)

    def restrict_codomain(self, sm):
        if False:
            return 10
        '\n        Return ``self``, with codomain restricted to to ``sm``.\n\n        Here ``sm`` has to be a submodule of ``self.codomain`` containing the\n        image.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.abc import x\n        >>> from sympy.polys.agca import homomorphism\n\n        >>> F = QQ.old_poly_ring(x).free_module(2)\n        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])\n        >>> h\n        Matrix([\n        [1, x], : QQ[x]**2 -> QQ[x]**2\n        [0, 0]])\n        >>> h.restrict_codomain(F.submodule([1, 0]))\n        Matrix([\n        [1, x], : QQ[x]**2 -> <[1, 0]>\n        [0, 0]])\n        '
        if not sm.is_submodule(self.image()):
            raise ValueError('the image %s must contain sm, got %s' % (self.image(), sm))
        if sm == self.codomain:
            return self
        return self._restrict_codomain(sm)

    def quotient_domain(self, sm):
        if False:
            i = 10
            return i + 15
        '\n        Return ``self`` with domain replaced by ``domain/sm``.\n\n        Here ``sm`` must be a submodule of ``self.kernel()``.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.abc import x\n        >>> from sympy.polys.agca import homomorphism\n\n        >>> F = QQ.old_poly_ring(x).free_module(2)\n        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])\n        >>> h\n        Matrix([\n        [1, x], : QQ[x]**2 -> QQ[x]**2\n        [0, 0]])\n        >>> h.quotient_domain(F.submodule([-x, 1]))\n        Matrix([\n        [1, x], : QQ[x]**2/<[-x, 1]> -> QQ[x]**2\n        [0, 0]])\n        '
        if not self.kernel().is_submodule(sm):
            raise ValueError('kernel %s must contain sm, got %s' % (self.kernel(), sm))
        if sm.is_zero():
            return self
        return self._quotient_domain(sm)

    def quotient_codomain(self, sm):
        if False:
            i = 10
            return i + 15
        '\n        Return ``self`` with codomain replaced by ``codomain/sm``.\n\n        Here ``sm`` must be a submodule of ``self.codomain``.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.abc import x\n        >>> from sympy.polys.agca import homomorphism\n\n        >>> F = QQ.old_poly_ring(x).free_module(2)\n        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])\n        >>> h\n        Matrix([\n        [1, x], : QQ[x]**2 -> QQ[x]**2\n        [0, 0]])\n        >>> h.quotient_codomain(F.submodule([1, 1]))\n        Matrix([\n        [1, x], : QQ[x]**2 -> QQ[x]**2/<[1, 1]>\n        [0, 0]])\n\n        This is the same as composing with the quotient map on the left:\n\n        >>> (F/[(1, 1)]).quotient_hom() * h\n        Matrix([\n        [1, x], : QQ[x]**2 -> QQ[x]**2/<[1, 1]>\n        [0, 0]])\n        '
        if not self.codomain.is_submodule(sm):
            raise ValueError('sm must be a submodule of codomain %s, got %s' % (self.codomain, sm))
        if sm.is_zero():
            return self
        return self._quotient_codomain(sm)

    def _apply(self, elem):
        if False:
            while True:
                i = 10
        'Apply ``self`` to ``elem``.'
        raise NotImplementedError

    def __call__(self, elem):
        if False:
            for i in range(10):
                print('nop')
        return self.codomain.convert(self._apply(self.domain.convert(elem)))

    def _compose(self, oth):
        if False:
            return 10
        '\n        Compose ``self`` with ``oth``, that is, return the homomorphism\n        obtained by first applying then ``self``, then ``oth``.\n\n        (This method is private since in this syntax, it is non-obvious which\n        homomorphism is executed first.)\n        '
        raise NotImplementedError

    def _mul_scalar(self, c):
        if False:
            for i in range(10):
                print('nop')
        'Scalar multiplication. ``c`` is guaranteed in self.ring.'
        raise NotImplementedError

    def _add(self, oth):
        if False:
            return 10
        '\n        Homomorphism addition.\n        ``oth`` is guaranteed to be a homomorphism with same domain/codomain.\n        '
        raise NotImplementedError

    def _check_hom(self, oth):
        if False:
            for i in range(10):
                print('nop')
        'Helper to check that oth is a homomorphism with same domain/codomain.'
        if not isinstance(oth, ModuleHomomorphism):
            return False
        return oth.domain == self.domain and oth.codomain == self.codomain

    def __mul__(self, oth):
        if False:
            i = 10
            return i + 15
        if isinstance(oth, ModuleHomomorphism) and self.domain == oth.codomain:
            return oth._compose(self)
        try:
            return self._mul_scalar(self.ring.convert(oth))
        except CoercionFailed:
            return NotImplemented
    __rmul__ = __mul__

    def __truediv__(self, oth):
        if False:
            return 10
        try:
            return self._mul_scalar(1 / self.ring.convert(oth))
        except CoercionFailed:
            return NotImplemented

    def __add__(self, oth):
        if False:
            i = 10
            return i + 15
        if self._check_hom(oth):
            return self._add(oth)
        return NotImplemented

    def __sub__(self, oth):
        if False:
            while True:
                i = 10
        if self._check_hom(oth):
            return self._add(oth._mul_scalar(self.ring.convert(-1)))
        return NotImplemented

    def is_injective(self):
        if False:
            i = 10
            return i + 15
        '\n        Return True if ``self`` is injective.\n\n        That is, check if the elements of the domain are mapped to the same\n        codomain element.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.abc import x\n        >>> from sympy.polys.agca import homomorphism\n\n        >>> F = QQ.old_poly_ring(x).free_module(2)\n        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])\n        >>> h.is_injective()\n        False\n        >>> h.quotient_domain(h.kernel()).is_injective()\n        True\n        '
        return self.kernel().is_zero()

    def is_surjective(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return True if ``self`` is surjective.\n\n        That is, check if every element of the codomain has at least one\n        preimage.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.abc import x\n        >>> from sympy.polys.agca import homomorphism\n\n        >>> F = QQ.old_poly_ring(x).free_module(2)\n        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])\n        >>> h.is_surjective()\n        False\n        >>> h.restrict_codomain(h.image()).is_surjective()\n        True\n        '
        return self.image() == self.codomain

    def is_isomorphism(self):
        if False:
            i = 10
            return i + 15
        '\n        Return True if ``self`` is an isomorphism.\n\n        That is, check if every element of the codomain has precisely one\n        preimage. Equivalently, ``self`` is both injective and surjective.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.abc import x\n        >>> from sympy.polys.agca import homomorphism\n\n        >>> F = QQ.old_poly_ring(x).free_module(2)\n        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])\n        >>> h = h.restrict_codomain(h.image())\n        >>> h.is_isomorphism()\n        False\n        >>> h.quotient_domain(h.kernel()).is_isomorphism()\n        True\n        '
        return self.is_injective() and self.is_surjective()

    def is_zero(self):
        if False:
            return 10
        '\n        Return True if ``self`` is a zero morphism.\n\n        That is, check if every element of the domain is mapped to zero\n        under self.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.abc import x\n        >>> from sympy.polys.agca import homomorphism\n\n        >>> F = QQ.old_poly_ring(x).free_module(2)\n        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])\n        >>> h.is_zero()\n        False\n        >>> h.restrict_domain(F.submodule()).is_zero()\n        True\n        >>> h.quotient_codomain(h.image()).is_zero()\n        True\n        '
        return self.image().is_zero()

    def __eq__(self, oth):
        if False:
            i = 10
            return i + 15
        try:
            return (self - oth).is_zero()
        except TypeError:
            return False

    def __ne__(self, oth):
        if False:
            for i in range(10):
                print('nop')
        return not self == oth

class MatrixHomomorphism(ModuleHomomorphism):
    """
    Helper class for all homomoprhisms which are expressed via a matrix.

    That is, for such homomorphisms ``domain`` is contained in a module
    generated by finitely many elements `e_1, \\ldots, e_n`, so that the
    homomorphism is determined uniquely by its action on the `e_i`. It
    can thus be represented as a vector of elements of the codomain module,
    or potentially a supermodule of the codomain module
    (and hence conventionally as a matrix, if there is a similar interpretation
    for elements of the codomain module).

    Note that this class does *not* assume that the `e_i` freely generate a
    submodule, nor that ``domain`` is even all of this submodule. It exists
    only to unify the interface.

    Do not instantiate.

    Attributes:

    - matrix - the list of images determining the homomorphism.
    NOTE: the elements of matrix belong to either self.codomain or
          self.codomain.container

    Still non-implemented methods:

    - kernel
    - _apply
    """

    def __init__(self, domain, codomain, matrix):
        if False:
            print('Hello World!')
        ModuleHomomorphism.__init__(self, domain, codomain)
        if len(matrix) != domain.rank:
            raise ValueError('Need to provide %s elements, got %s' % (domain.rank, len(matrix)))
        converter = self.codomain.convert
        if isinstance(self.codomain, (SubModule, SubQuotientModule)):
            converter = self.codomain.container.convert
        self.matrix = tuple((converter(x) for x in matrix))

    def _sympy_matrix(self):
        if False:
            return 10
        'Helper function which returns a SymPy matrix ``self.matrix``.'
        from sympy.matrices import Matrix
        c = lambda x: x
        if isinstance(self.codomain, (QuotientModule, SubQuotientModule)):
            c = lambda x: x.data
        return Matrix([[self.ring.to_sympy(y) for y in c(x)] for x in self.matrix]).T

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        lines = repr(self._sympy_matrix()).split('\n')
        t = ' : %s -> %s' % (self.domain, self.codomain)
        s = ' ' * len(t)
        n = len(lines)
        for i in range(n // 2):
            lines[i] += s
        lines[n // 2] += t
        for i in range(n // 2 + 1, n):
            lines[i] += s
        return '\n'.join(lines)

    def _restrict_domain(self, sm):
        if False:
            for i in range(10):
                print('nop')
        'Implementation of domain restriction.'
        return SubModuleHomomorphism(sm, self.codomain, self.matrix)

    def _restrict_codomain(self, sm):
        if False:
            for i in range(10):
                print('nop')
        'Implementation of codomain restriction.'
        return self.__class__(self.domain, sm, self.matrix)

    def _quotient_domain(self, sm):
        if False:
            for i in range(10):
                print('nop')
        'Implementation of domain quotient.'
        return self.__class__(self.domain / sm, self.codomain, self.matrix)

    def _quotient_codomain(self, sm):
        if False:
            return 10
        'Implementation of codomain quotient.'
        Q = self.codomain / sm
        converter = Q.convert
        if isinstance(self.codomain, SubModule):
            converter = Q.container.convert
        return self.__class__(self.domain, self.codomain / sm, [converter(x) for x in self.matrix])

    def _add(self, oth):
        if False:
            i = 10
            return i + 15
        return self.__class__(self.domain, self.codomain, [x + y for (x, y) in zip(self.matrix, oth.matrix)])

    def _mul_scalar(self, c):
        if False:
            i = 10
            return i + 15
        return self.__class__(self.domain, self.codomain, [c * x for x in self.matrix])

    def _compose(self, oth):
        if False:
            print('Hello World!')
        return self.__class__(self.domain, oth.codomain, [oth(x) for x in self.matrix])

class FreeModuleHomomorphism(MatrixHomomorphism):
    """
    Concrete class for homomorphisms with domain a free module or a quotient
    thereof.

    Do not instantiate; the constructor does not check that your data is well
    defined. Use the ``homomorphism`` function instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.agca import homomorphism

    >>> F = QQ.old_poly_ring(x).free_module(2)
    >>> homomorphism(F, F, [[1, 0], [0, 1]])
    Matrix([
    [1, 0], : QQ[x]**2 -> QQ[x]**2
    [0, 1]])
    """

    def _apply(self, elem):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.domain, QuotientModule):
            elem = elem.data
        return sum((x * e for (x, e) in zip(elem, self.matrix)))

    def _image(self):
        if False:
            print('Hello World!')
        return self.codomain.submodule(*self.matrix)

    def _kernel(self):
        if False:
            for i in range(10):
                print('nop')
        syz = self.image().syzygy_module()
        return self.domain.submodule(*syz.gens)

class SubModuleHomomorphism(MatrixHomomorphism):
    """
    Concrete class for homomorphism with domain a submodule of a free module
    or a quotient thereof.

    Do not instantiate; the constructor does not check that your data is well
    defined. Use the ``homomorphism`` function instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.agca import homomorphism

    >>> M = QQ.old_poly_ring(x).free_module(2)*x
    >>> homomorphism(M, M, [[1, 0], [0, 1]])
    Matrix([
    [1, 0], : <[x, 0], [0, x]> -> <[x, 0], [0, x]>
    [0, 1]])
    """

    def _apply(self, elem):
        if False:
            return 10
        if isinstance(self.domain, SubQuotientModule):
            elem = elem.data
        return sum((x * e for (x, e) in zip(elem, self.matrix)))

    def _image(self):
        if False:
            return 10
        return self.codomain.submodule(*[self(x) for x in self.domain.gens])

    def _kernel(self):
        if False:
            while True:
                i = 10
        syz = self.image().syzygy_module()
        return self.domain.submodule(*[sum((xi * gi for (xi, gi) in zip(s, self.domain.gens))) for s in syz.gens])

def homomorphism(domain, codomain, matrix):
    if False:
        return 10
    '\n    Create a homomorphism object.\n\n    This function tries to build a homomorphism from ``domain`` to ``codomain``\n    via the matrix ``matrix``.\n\n    Examples\n    ========\n\n    >>> from sympy import QQ\n    >>> from sympy.abc import x\n    >>> from sympy.polys.agca import homomorphism\n\n    >>> R = QQ.old_poly_ring(x)\n    >>> T = R.free_module(2)\n\n    If ``domain`` is a free module generated by `e_1, \\ldots, e_n`, then\n    ``matrix`` should be an n-element iterable `(b_1, \\ldots, b_n)` where\n    the `b_i` are elements of ``codomain``. The constructed homomorphism is the\n    unique homomorphism sending `e_i` to `b_i`.\n\n    >>> F = R.free_module(2)\n    >>> h = homomorphism(F, T, [[1, x], [x**2, 0]])\n    >>> h\n    Matrix([\n    [1, x**2], : QQ[x]**2 -> QQ[x]**2\n    [x,    0]])\n    >>> h([1, 0])\n    [1, x]\n    >>> h([0, 1])\n    [x**2, 0]\n    >>> h([1, 1])\n    [x**2 + 1, x]\n\n    If ``domain`` is a submodule of a free module, them ``matrix`` determines\n    a homomoprhism from the containing free module to ``codomain``, and the\n    homomorphism returned is obtained by restriction to ``domain``.\n\n    >>> S = F.submodule([1, 0], [0, x])\n    >>> homomorphism(S, T, [[1, x], [x**2, 0]])\n    Matrix([\n    [1, x**2], : <[1, 0], [0, x]> -> QQ[x]**2\n    [x,    0]])\n\n    If ``domain`` is a (sub)quotient `N/K`, then ``matrix`` determines a\n    homomorphism from `N` to ``codomain``. If the kernel contains `K`, this\n    homomorphism descends to ``domain`` and is returned; otherwise an exception\n    is raised.\n\n    >>> homomorphism(S/[(1, 0)], T, [0, [x**2, 0]])\n    Matrix([\n    [0, x**2], : <[1, 0] + <[1, 0]>, [0, x] + <[1, 0]>, [1, 0] + <[1, 0]>> -> QQ[x]**2\n    [0,    0]])\n    >>> homomorphism(S/[(0, x)], T, [0, [x**2, 0]])\n    Traceback (most recent call last):\n    ...\n    ValueError: kernel <[1, 0], [0, 0]> must contain sm, got <[0,x]>\n\n    '

    def freepres(module):
        if False:
            return 10
        '\n        Return a tuple ``(F, S, Q, c)`` where ``F`` is a free module, ``S`` is a\n        submodule of ``F``, and ``Q`` a submodule of ``S``, such that\n        ``module = S/Q``, and ``c`` is a conversion function.\n        '
        if isinstance(module, FreeModule):
            return (module, module, module.submodule(), lambda x: module.convert(x))
        if isinstance(module, QuotientModule):
            return (module.base, module.base, module.killed_module, lambda x: module.convert(x).data)
        if isinstance(module, SubQuotientModule):
            return (module.base.container, module.base, module.killed_module, lambda x: module.container.convert(x).data)
        return (module.container, module, module.submodule(), lambda x: module.container.convert(x))
    (SF, SS, SQ, _) = freepres(domain)
    (TF, TS, TQ, c) = freepres(codomain)
    return FreeModuleHomomorphism(SF, TF, [c(x) for x in matrix]).restrict_domain(SS).restrict_codomain(TS).quotient_codomain(TQ).quotient_domain(SQ)