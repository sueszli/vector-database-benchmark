from math import prod
from sympy.core import S, Integer
from sympy.core.function import Function
from sympy.core.logic import fuzzy_not
from sympy.core.relational import Ne
from sympy.core.sorting import default_sort_key
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.iterables import has_dups

def Eijk(*args, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Represent the Levi-Civita symbol.\n\n    This is a compatibility wrapper to ``LeviCivita()``.\n\n    See Also\n    ========\n\n    LeviCivita\n\n    '
    return LeviCivita(*args, **kwargs)

def eval_levicivita(*args):
    if False:
        while True:
            i = 10
    'Evaluate Levi-Civita symbol.'
    n = len(args)
    return prod((prod((args[j] - args[i] for j in range(i + 1, n))) / factorial(i) for i in range(n)))

class LeviCivita(Function):
    """
    Represent the Levi-Civita symbol.

    Explanation
    ===========

    For even permutations of indices it returns 1, for odd permutations -1, and
    for everything else (a repeated index) it returns 0.

    Thus it represents an alternating pseudotensor.

    Examples
    ========

    >>> from sympy import LeviCivita
    >>> from sympy.abc import i, j, k
    >>> LeviCivita(1, 2, 3)
    1
    >>> LeviCivita(1, 3, 2)
    -1
    >>> LeviCivita(1, 2, 2)
    0
    >>> LeviCivita(i, j, k)
    LeviCivita(i, j, k)
    >>> LeviCivita(i, j, i)
    0

    See Also
    ========

    Eijk

    """
    is_integer = True

    @classmethod
    def eval(cls, *args):
        if False:
            return 10
        if all((isinstance(a, (SYMPY_INTS, Integer)) for a in args)):
            return eval_levicivita(*args)
        if has_dups(args):
            return S.Zero

    def doit(self, **hints):
        if False:
            return 10
        return eval_levicivita(*self.args)

class KroneckerDelta(Function):
    """
    The discrete, or Kronecker, delta function.

    Explanation
    ===========

    A function that takes in two integers $i$ and $j$. It returns $0$ if $i$
    and $j$ are not equal, or it returns $1$ if $i$ and $j$ are equal.

    Examples
    ========

    An example with integer indices:

        >>> from sympy import KroneckerDelta
        >>> KroneckerDelta(1, 2)
        0
        >>> KroneckerDelta(3, 3)
        1

    Symbolic indices:

        >>> from sympy.abc import i, j, k
        >>> KroneckerDelta(i, j)
        KroneckerDelta(i, j)
        >>> KroneckerDelta(i, i)
        1
        >>> KroneckerDelta(i, i + 1)
        0
        >>> KroneckerDelta(i, i + 1 + k)
        KroneckerDelta(i, i + k + 1)

    Parameters
    ==========

    i : Number, Symbol
        The first index of the delta function.
    j : Number, Symbol
        The second index of the delta function.

    See Also
    ========

    eval
    DiracDelta

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Kronecker_delta

    """
    is_integer = True

    @classmethod
    def eval(cls, i, j, delta_range=None):
        if False:
            while True:
                i = 10
        '\n        Evaluates the discrete delta function.\n\n        Examples\n        ========\n\n        >>> from sympy import KroneckerDelta\n        >>> from sympy.abc import i, j, k\n\n        >>> KroneckerDelta(i, j)\n        KroneckerDelta(i, j)\n        >>> KroneckerDelta(i, i)\n        1\n        >>> KroneckerDelta(i, i + 1)\n        0\n        >>> KroneckerDelta(i, i + 1 + k)\n        KroneckerDelta(i, i + k + 1)\n\n        # indirect doctest\n\n        '
        if delta_range is not None:
            (dinf, dsup) = delta_range
            if (dinf - i > 0) == True:
                return S.Zero
            if (dinf - j > 0) == True:
                return S.Zero
            if (dsup - i < 0) == True:
                return S.Zero
            if (dsup - j < 0) == True:
                return S.Zero
        diff = i - j
        if diff.is_zero:
            return S.One
        elif fuzzy_not(diff.is_zero):
            return S.Zero
        if i.assumptions0.get('below_fermi') and j.assumptions0.get('above_fermi'):
            return S.Zero
        if j.assumptions0.get('below_fermi') and i.assumptions0.get('above_fermi'):
            return S.Zero
        if i != min(i, j, key=default_sort_key):
            if delta_range:
                return cls(j, i, delta_range)
            else:
                return cls(j, i)

    @property
    def delta_range(self):
        if False:
            while True:
                i = 10
        if len(self.args) > 2:
            return self.args[2]

    def _eval_power(self, expt):
        if False:
            i = 10
            return i + 15
        if expt.is_positive:
            return self
        if expt.is_negative and expt is not S.NegativeOne:
            return 1 / self

    @property
    def is_above_fermi(self):
        if False:
            return 10
        "\n        True if Delta can be non-zero above fermi.\n\n        Examples\n        ========\n\n        >>> from sympy import KroneckerDelta, Symbol\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n        >>> q = Symbol('q')\n        >>> KroneckerDelta(p, a).is_above_fermi\n        True\n        >>> KroneckerDelta(p, i).is_above_fermi\n        False\n        >>> KroneckerDelta(p, q).is_above_fermi\n        True\n\n        See Also\n        ========\n\n        is_below_fermi, is_only_below_fermi, is_only_above_fermi\n\n        "
        if self.args[0].assumptions0.get('below_fermi'):
            return False
        if self.args[1].assumptions0.get('below_fermi'):
            return False
        return True

    @property
    def is_below_fermi(self):
        if False:
            i = 10
            return i + 15
        "\n        True if Delta can be non-zero below fermi.\n\n        Examples\n        ========\n\n        >>> from sympy import KroneckerDelta, Symbol\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n        >>> q = Symbol('q')\n        >>> KroneckerDelta(p, a).is_below_fermi\n        False\n        >>> KroneckerDelta(p, i).is_below_fermi\n        True\n        >>> KroneckerDelta(p, q).is_below_fermi\n        True\n\n        See Also\n        ========\n\n        is_above_fermi, is_only_above_fermi, is_only_below_fermi\n\n        "
        if self.args[0].assumptions0.get('above_fermi'):
            return False
        if self.args[1].assumptions0.get('above_fermi'):
            return False
        return True

    @property
    def is_only_above_fermi(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        True if Delta is restricted to above fermi.\n\n        Examples\n        ========\n\n        >>> from sympy import KroneckerDelta, Symbol\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n        >>> q = Symbol('q')\n        >>> KroneckerDelta(p, a).is_only_above_fermi\n        True\n        >>> KroneckerDelta(p, q).is_only_above_fermi\n        False\n        >>> KroneckerDelta(p, i).is_only_above_fermi\n        False\n\n        See Also\n        ========\n\n        is_above_fermi, is_below_fermi, is_only_below_fermi\n\n        "
        return (self.args[0].assumptions0.get('above_fermi') or self.args[1].assumptions0.get('above_fermi')) or False

    @property
    def is_only_below_fermi(self):
        if False:
            while True:
                i = 10
        "\n        True if Delta is restricted to below fermi.\n\n        Examples\n        ========\n\n        >>> from sympy import KroneckerDelta, Symbol\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n        >>> q = Symbol('q')\n        >>> KroneckerDelta(p, i).is_only_below_fermi\n        True\n        >>> KroneckerDelta(p, q).is_only_below_fermi\n        False\n        >>> KroneckerDelta(p, a).is_only_below_fermi\n        False\n\n        See Also\n        ========\n\n        is_above_fermi, is_below_fermi, is_only_above_fermi\n\n        "
        return (self.args[0].assumptions0.get('below_fermi') or self.args[1].assumptions0.get('below_fermi')) or False

    @property
    def indices_contain_equal_information(self):
        if False:
            return 10
        "\n        Returns True if indices are either both above or below fermi.\n\n        Examples\n        ========\n\n        >>> from sympy import KroneckerDelta, Symbol\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n        >>> q = Symbol('q')\n        >>> KroneckerDelta(p, q).indices_contain_equal_information\n        True\n        >>> KroneckerDelta(p, q+1).indices_contain_equal_information\n        True\n        >>> KroneckerDelta(i, p).indices_contain_equal_information\n        False\n\n        "
        if self.args[0].assumptions0.get('below_fermi') and self.args[1].assumptions0.get('below_fermi'):
            return True
        if self.args[0].assumptions0.get('above_fermi') and self.args[1].assumptions0.get('above_fermi'):
            return True
        return self.is_below_fermi and self.is_above_fermi

    @property
    def preferred_index(self):
        if False:
            while True:
                i = 10
        "\n        Returns the index which is preferred to keep in the final expression.\n\n        Explanation\n        ===========\n\n        The preferred index is the index with more information regarding fermi\n        level. If indices contain the same information, 'a' is preferred before\n        'b'.\n\n        Examples\n        ========\n\n        >>> from sympy import KroneckerDelta, Symbol\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> j = Symbol('j', below_fermi=True)\n        >>> p = Symbol('p')\n        >>> KroneckerDelta(p, i).preferred_index\n        i\n        >>> KroneckerDelta(p, a).preferred_index\n        a\n        >>> KroneckerDelta(i, j).preferred_index\n        i\n\n        See Also\n        ========\n\n        killable_index\n\n        "
        if self._get_preferred_index():
            return self.args[1]
        else:
            return self.args[0]

    @property
    def killable_index(self):
        if False:
            print('Hello World!')
        "\n        Returns the index which is preferred to substitute in the final\n        expression.\n\n        Explanation\n        ===========\n\n        The index to substitute is the index with less information regarding\n        fermi level. If indices contain the same information, 'a' is preferred\n        before 'b'.\n\n        Examples\n        ========\n\n        >>> from sympy import KroneckerDelta, Symbol\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> j = Symbol('j', below_fermi=True)\n        >>> p = Symbol('p')\n        >>> KroneckerDelta(p, i).killable_index\n        p\n        >>> KroneckerDelta(p, a).killable_index\n        p\n        >>> KroneckerDelta(i, j).killable_index\n        j\n\n        See Also\n        ========\n\n        preferred_index\n\n        "
        if self._get_preferred_index():
            return self.args[0]
        else:
            return self.args[1]

    def _get_preferred_index(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the index which is preferred to keep in the final expression.\n\n        The preferred index is the index with more information regarding fermi\n        level. If indices contain the same information, index 0 is returned.\n\n        '
        if not self.is_above_fermi:
            if self.args[0].assumptions0.get('below_fermi'):
                return 0
            else:
                return 1
        elif not self.is_below_fermi:
            if self.args[0].assumptions0.get('above_fermi'):
                return 0
            else:
                return 1
        else:
            return 0

    @property
    def indices(self):
        if False:
            print('Hello World!')
        return self.args[0:2]

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        (i, j) = args
        return Piecewise((0, Ne(i, j)), (1, True))