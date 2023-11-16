"""Trait for implementing domain elements. """
from sympy.utilities import public

@public
class DomainElement:
    """
    Represents an element of a domain.

    Mix in this trait into a class whose instances should be recognized as
    elements of a domain. Method ``parent()`` gives that domain.
    """
    __slots__ = ()

    def parent(self):
        if False:
            print('Hello World!')
        "Get the domain associated with ``self``\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ, symbols\n        >>> x, y = symbols('x, y')\n        >>> K = ZZ[x,y]\n        >>> p = K(x)**2 + K(y)**2\n        >>> p\n        x**2 + y**2\n        >>> p.parent()\n        ZZ[x,y]\n\n        Notes\n        =====\n\n        This is used by :py:meth:`~.Domain.convert` to identify the domain\n        associated with a domain element.\n        "
        raise NotImplementedError('abstract method')