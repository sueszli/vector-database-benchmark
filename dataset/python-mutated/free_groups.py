from __future__ import annotations
from sympy.core import S
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.sympify import CantSympify
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.magic import pollute
from sympy.utilities.misc import as_int

@public
def free_group(symbols):
    if False:
        print('Hello World!')
    'Construct a free group returning ``(FreeGroup, (f_0, f_1, ..., f_(n-1))``.\n\n    Parameters\n    ==========\n\n    symbols : str, Symbol/Expr or sequence of str, Symbol/Expr (may be empty)\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics import free_group\n    >>> F, x, y, z = free_group("x, y, z")\n    >>> F\n    <free group on the generators (x, y, z)>\n    >>> x**2*y**-1\n    x**2*y**-1\n    >>> type(_)\n    <class \'sympy.combinatorics.free_groups.FreeGroupElement\'>\n\n    '
    _free_group = FreeGroup(symbols)
    return (_free_group,) + tuple(_free_group.generators)

@public
def xfree_group(symbols):
    if False:
        for i in range(10):
            print('nop')
    'Construct a free group returning ``(FreeGroup, (f_0, f_1, ..., f_(n-1)))``.\n\n    Parameters\n    ==========\n\n    symbols : str, Symbol/Expr or sequence of str, Symbol/Expr (may be empty)\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.free_groups import xfree_group\n    >>> F, (x, y, z) = xfree_group("x, y, z")\n    >>> F\n    <free group on the generators (x, y, z)>\n    >>> y**2*x**-2*z**-1\n    y**2*x**-2*z**-1\n    >>> type(_)\n    <class \'sympy.combinatorics.free_groups.FreeGroupElement\'>\n\n    '
    _free_group = FreeGroup(symbols)
    return (_free_group, _free_group.generators)

@public
def vfree_group(symbols):
    if False:
        while True:
            i = 10
    'Construct a free group and inject ``f_0, f_1, ..., f_(n-1)`` as symbols\n    into the global namespace.\n\n    Parameters\n    ==========\n\n    symbols : str, Symbol/Expr or sequence of str, Symbol/Expr (may be empty)\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.free_groups import vfree_group\n    >>> vfree_group("x, y, z")\n    <free group on the generators (x, y, z)>\n    >>> x**2*y**-2*z # noqa: F821\n    x**2*y**-2*z\n    >>> type(_)\n    <class \'sympy.combinatorics.free_groups.FreeGroupElement\'>\n\n    '
    _free_group = FreeGroup(symbols)
    pollute([sym.name for sym in _free_group.symbols], _free_group.generators)
    return _free_group

def _parse_symbols(symbols):
    if False:
        return 10
    if not symbols:
        return ()
    if isinstance(symbols, str):
        return _symbols(symbols, seq=True)
    elif isinstance(symbols, (Expr, FreeGroupElement)):
        return (symbols,)
    elif is_sequence(symbols):
        if all((isinstance(s, str) for s in symbols)):
            return _symbols(symbols)
        elif all((isinstance(s, Expr) for s in symbols)):
            return symbols
    raise ValueError('The type of `symbols` must be one of the following: a str, Symbol/Expr or a sequence of one of these types')
_free_group_cache: dict[int, FreeGroup] = {}

class FreeGroup(DefaultPrinting):
    """
    Free group with finite or infinite number of generators. Its input API
    is that of a str, Symbol/Expr or a sequence of one of
    these types (which may be empty)

    See Also
    ========

    sympy.polys.rings.PolyRing

    References
    ==========

    .. [1] https://www.gap-system.org/Manuals/doc/ref/chap37.html

    .. [2] https://en.wikipedia.org/wiki/Free_group

    """
    is_associative = True
    is_group = True
    is_FreeGroup = True
    is_PermutationGroup = False
    relators: list[Expr] = []

    def __new__(cls, symbols):
        if False:
            return 10
        symbols = tuple(_parse_symbols(symbols))
        rank = len(symbols)
        _hash = hash((cls.__name__, symbols, rank))
        obj = _free_group_cache.get(_hash)
        if obj is None:
            obj = object.__new__(cls)
            obj._hash = _hash
            obj._rank = rank
            obj.dtype = type('FreeGroupElement', (FreeGroupElement,), {'group': obj})
            obj.symbols = symbols
            obj.generators = obj._generators()
            obj._gens_set = set(obj.generators)
            for (symbol, generator) in zip(obj.symbols, obj.generators):
                if isinstance(symbol, Symbol):
                    name = symbol.name
                    if hasattr(obj, name):
                        setattr(obj, name, generator)
            _free_group_cache[_hash] = obj
        return obj

    def _generators(group):
        if False:
            i = 10
            return i + 15
        'Returns the generators of the FreeGroup.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> F, x, y, z = free_group("x, y, z")\n        >>> F.generators\n        (x, y, z)\n\n        '
        gens = []
        for sym in group.symbols:
            elm = ((sym, 1),)
            gens.append(group.dtype(elm))
        return tuple(gens)

    def clone(self, symbols=None):
        if False:
            while True:
                i = 10
        return self.__class__(symbols or self.symbols)

    def __contains__(self, i):
        if False:
            return 10
        'Return True if ``i`` is contained in FreeGroup.'
        if not isinstance(i, FreeGroupElement):
            return False
        group = i.group
        return self == group

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._hash

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.rank

    def __str__(self):
        if False:
            while True:
                i = 10
        if self.rank > 30:
            str_form = '<free group with %s generators>' % self.rank
        else:
            str_form = '<free group on the generators '
            gens = self.generators
            str_form += str(gens) + '>'
        return str_form
    __repr__ = __str__

    def __getitem__(self, index):
        if False:
            return 10
        symbols = self.symbols[index]
        return self.clone(symbols=symbols)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        'No ``FreeGroup`` is equal to any "other" ``FreeGroup``.\n        '
        return self is other

    def index(self, gen):
        if False:
            while True:
                i = 10
        'Return the index of the generator `gen` from ``(f_0, ..., f_(n-1))``.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> F, x, y = free_group("x, y")\n        >>> F.index(y)\n        1\n        >>> F.index(x)\n        0\n\n        '
        if isinstance(gen, self.dtype):
            return self.generators.index(gen)
        else:
            raise ValueError('expected a generator of Free Group %s, got %s' % (self, gen))

    def order(self):
        if False:
            print('Hello World!')
        'Return the order of the free group.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> F, x, y = free_group("x, y")\n        >>> F.order()\n        oo\n\n        >>> free_group("")[0].order()\n        1\n\n        '
        if self.rank == 0:
            return S.One
        else:
            return S.Infinity

    @property
    def elements(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the elements of the free group.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> (z,) = free_group("")\n        >>> z.elements\n        {<identity>}\n\n        '
        if self.rank == 0:
            return {self.identity}
        else:
            raise ValueError('Group contains infinitely many elements, hence cannot be represented')

    @property
    def rank(self):
        if False:
            i = 10
            return i + 15
        '\n        In group theory, the `rank` of a group `G`, denoted `G.rank`,\n        can refer to the smallest cardinality of a generating set\n        for G, that is\n\n        \\operatorname{rank}(G)=\\min\\{ |X|: X\\subseteq G, \\left\\langle X\\right\\rangle =G\\}.\n\n        '
        return self._rank

    @property
    def is_abelian(self):
        if False:
            i = 10
            return i + 15
        'Returns if the group is Abelian.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, x, y, z = free_group("x y z")\n        >>> f.is_abelian\n        False\n\n        '
        return self.rank in (0, 1)

    @property
    def identity(self):
        if False:
            while True:
                i = 10
        'Returns the identity element of free group.'
        return self.dtype()

    def contains(self, g):
        if False:
            i = 10
            return i + 15
        'Tests if Free Group element ``g`` belong to self, ``G``.\n\n        In mathematical terms any linear combination of generators\n        of a Free Group is contained in it.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, x, y, z = free_group("x y z")\n        >>> f.contains(x**3*y**2)\n        True\n\n        '
        if not isinstance(g, FreeGroupElement):
            return False
        elif self != g.group:
            return False
        else:
            return True

    def center(self):
        if False:
            return 10
        'Returns the center of the free group `self`.'
        return {self.identity}

class FreeGroupElement(CantSympify, DefaultPrinting, tuple):
    """Used to create elements of FreeGroup. It cannot be used directly to
    create a free group element. It is called by the `dtype` method of the
    `FreeGroup` class.

    """
    is_assoc_word = True

    def new(self, init):
        if False:
            while True:
                i = 10
        return self.__class__(init)
    _hash = None

    def __hash__(self):
        if False:
            print('Hello World!')
        _hash = self._hash
        if _hash is None:
            self._hash = _hash = hash((self.group, frozenset(tuple(self))))
        return _hash

    def copy(self):
        if False:
            return 10
        return self.new(self)

    @property
    def is_identity(self):
        if False:
            for i in range(10):
                print('nop')
        if self.array_form == ():
            return True
        else:
            return False

    @property
    def array_form(self):
        if False:
            print('Hello World!')
        '\n        SymPy provides two different internal kinds of representation\n        of associative words. The first one is called the `array_form`\n        which is a tuple containing `tuples` as its elements, where the\n        size of each tuple is two. At the first position the tuple\n        contains the `symbol-generator`, while at the second position\n        of tuple contains the exponent of that generator at the position.\n        Since elements (i.e. words) do not commute, the indexing of tuple\n        makes that property to stay.\n\n        The structure in ``array_form`` of ``FreeGroupElement`` is of form:\n\n        ``( ( symbol_of_gen, exponent ), ( , ), ... ( , ) )``\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, x, y, z = free_group("x y z")\n        >>> (x*z).array_form\n        ((x, 1), (z, 1))\n        >>> (x**2*z*y*x**2).array_form\n        ((x, 2), (z, 1), (y, 1), (x, 2))\n\n        See Also\n        ========\n\n        letter_repr\n\n        '
        return tuple(self)

    @property
    def letter_form(self):
        if False:
            return 10
        '\n        The letter representation of a ``FreeGroupElement`` is a tuple\n        of generator symbols, with each entry corresponding to a group\n        generator. Inverses of the generators are represented by\n        negative generator symbols.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, a, b, c, d = free_group("a b c d")\n        >>> (a**3).letter_form\n        (a, a, a)\n        >>> (a**2*d**-2*a*b**-4).letter_form\n        (a, a, -d, -d, a, -b, -b, -b, -b)\n        >>> (a**-2*b**3*d).letter_form\n        (-a, -a, b, b, b, d)\n\n        See Also\n        ========\n\n        array_form\n\n        '
        return tuple(flatten([(i,) * j if j > 0 else (-i,) * -j for (i, j) in self.array_form]))

    def __getitem__(self, i):
        if False:
            i = 10
            return i + 15
        group = self.group
        r = self.letter_form[i]
        if r.is_Symbol:
            return group.dtype(((r, 1),))
        else:
            return group.dtype(((-r, -1),))

    def index(self, gen):
        if False:
            while True:
                i = 10
        if len(gen) != 1:
            raise ValueError()
        return self.letter_form.index(gen.letter_form[0])

    @property
    def letter_form_elm(self):
        if False:
            print('Hello World!')
        '\n        '
        group = self.group
        r = self.letter_form
        return [group.dtype(((elm, 1),)) if elm.is_Symbol else group.dtype(((-elm, -1),)) for elm in r]

    @property
    def ext_rep(self):
        if False:
            for i in range(10):
                print('nop')
        'This is called the External Representation of ``FreeGroupElement``\n        '
        return tuple(flatten(self.array_form))

    def __contains__(self, gen):
        if False:
            while True:
                i = 10
        return gen.array_form[0][0] in tuple([r[0] for r in self.array_form])

    def __str__(self):
        if False:
            return 10
        if self.is_identity:
            return '<identity>'
        str_form = ''
        array_form = self.array_form
        for i in range(len(array_form)):
            if i == len(array_form) - 1:
                if array_form[i][1] == 1:
                    str_form += str(array_form[i][0])
                else:
                    str_form += str(array_form[i][0]) + '**' + str(array_form[i][1])
            elif array_form[i][1] == 1:
                str_form += str(array_form[i][0]) + '*'
            else:
                str_form += str(array_form[i][0]) + '**' + str(array_form[i][1]) + '*'
        return str_form
    __repr__ = __str__

    def __pow__(self, n):
        if False:
            return 10
        n = as_int(n)
        group = self.group
        if n == 0:
            return group.identity
        if n < 0:
            n = -n
            return self.inverse() ** n
        result = self
        for i in range(n - 1):
            result = result * self
        return result

    def __mul__(self, other):
        if False:
            while True:
                i = 10
        'Returns the product of elements belonging to the same ``FreeGroup``.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, x, y, z = free_group("x y z")\n        >>> x*y**2*y**-4\n        x*y**-2\n        >>> z*y**-2\n        z*y**-2\n        >>> x**2*y*y**-1*x**-2\n        <identity>\n\n        '
        group = self.group
        if not isinstance(other, group.dtype):
            raise TypeError('only FreeGroup elements of same FreeGroup can be multiplied')
        if self.is_identity:
            return other
        if other.is_identity:
            return self
        r = list(self.array_form + other.array_form)
        zero_mul_simp(r, len(self.array_form) - 1)
        return group.dtype(tuple(r))

    def __truediv__(self, other):
        if False:
            print('Hello World!')
        group = self.group
        if not isinstance(other, group.dtype):
            raise TypeError('only FreeGroup elements of same FreeGroup can be multiplied')
        return self * other.inverse()

    def __rtruediv__(self, other):
        if False:
            while True:
                i = 10
        group = self.group
        if not isinstance(other, group.dtype):
            raise TypeError('only FreeGroup elements of same FreeGroup can be multiplied')
        return other * self.inverse()

    def __add__(self, other):
        if False:
            return 10
        return NotImplemented

    def inverse(self):
        if False:
            print('Hello World!')
        '\n        Returns the inverse of a ``FreeGroupElement`` element\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, x, y, z = free_group("x y z")\n        >>> x.inverse()\n        x**-1\n        >>> (x*y).inverse()\n        y**-1*x**-1\n\n        '
        group = self.group
        r = tuple([(i, -j) for (i, j) in self.array_form[::-1]])
        return group.dtype(r)

    def order(self):
        if False:
            for i in range(10):
                print('nop')
        'Find the order of a ``FreeGroupElement``.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, x, y = free_group("x y")\n        >>> (x**2*y*y**-1*x**-2).order()\n        1\n\n        '
        if self.is_identity:
            return S.One
        else:
            return S.Infinity

    def commutator(self, other):
        if False:
            return 10
        '\n        Return the commutator of `self` and `x`: ``~x*~self*x*self``\n\n        '
        group = self.group
        if not isinstance(other, group.dtype):
            raise ValueError('commutator of only FreeGroupElement of the same FreeGroup exists')
        else:
            return self.inverse() * other.inverse() * self * other

    def eliminate_words(self, words, _all=False, inverse=True):
        if False:
            i = 10
            return i + 15
        '\n        Replace each subword from the dictionary `words` by words[subword].\n        If words is a list, replace the words by the identity.\n\n        '
        again = True
        new = self
        if isinstance(words, dict):
            while again:
                again = False
                for sub in words:
                    prev = new
                    new = new.eliminate_word(sub, words[sub], _all=_all, inverse=inverse)
                    if new != prev:
                        again = True
        else:
            while again:
                again = False
                for sub in words:
                    prev = new
                    new = new.eliminate_word(sub, _all=_all, inverse=inverse)
                    if new != prev:
                        again = True
        return new

    def eliminate_word(self, gen, by=None, _all=False, inverse=True):
        if False:
            return 10
        '\n        For an associative word `self`, a subword `gen`, and an associative\n        word `by` (identity by default), return the associative word obtained by\n        replacing each occurrence of `gen` in `self` by `by`. If `_all = True`,\n        the occurrences of `gen` that may appear after the first substitution will\n        also be replaced and so on until no occurrences are found. This might not\n        always terminate (e.g. `(x).eliminate_word(x, x**2, _all=True)`).\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, x, y = free_group("x y")\n        >>> w = x**5*y*x**2*y**-4*x\n        >>> w.eliminate_word( x, x**2 )\n        x**10*y*x**4*y**-4*x**2\n        >>> w.eliminate_word( x, y**-1 )\n        y**-11\n        >>> w.eliminate_word(x**5)\n        y*x**2*y**-4*x\n        >>> w.eliminate_word(x*y, y)\n        x**4*y*x**2*y**-4*x\n\n        See Also\n        ========\n        substituted_word\n\n        '
        if by is None:
            by = self.group.identity
        if self.is_independent(gen) or gen == by:
            return self
        if gen == self:
            return by
        if gen ** (-1) == by:
            _all = False
        word = self
        l = len(gen)
        try:
            i = word.subword_index(gen)
            k = 1
        except ValueError:
            if not inverse:
                return word
            try:
                i = word.subword_index(gen ** (-1))
                k = -1
            except ValueError:
                return word
        word = word.subword(0, i) * by ** k * word.subword(i + l, len(word)).eliminate_word(gen, by)
        if _all:
            return word.eliminate_word(gen, by, _all=True, inverse=inverse)
        else:
            return word

    def __len__(self):
        if False:
            while True:
                i = 10
        '\n        For an associative word `self`, returns the number of letters in it.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, a, b = free_group("a b")\n        >>> w = a**5*b*a**2*b**-4*a\n        >>> len(w)\n        13\n        >>> len(a**17)\n        17\n        >>> len(w**0)\n        0\n\n        '
        return sum((abs(j) for (i, j) in self))

    def __eq__(self, other):
        if False:
            return 10
        '\n        Two  associative words are equal if they are words over the\n        same alphabet and if they are sequences of the same letters.\n        This is equivalent to saying that the external representations\n        of the words are equal.\n        There is no "universal" empty word, every alphabet has its own\n        empty word.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, swapnil0, swapnil1 = free_group("swapnil0 swapnil1")\n        >>> f\n        <free group on the generators (swapnil0, swapnil1)>\n        >>> g, swap0, swap1 = free_group("swap0 swap1")\n        >>> g\n        <free group on the generators (swap0, swap1)>\n\n        >>> swapnil0 == swapnil1\n        False\n        >>> swapnil0*swapnil1 == swapnil1/swapnil1*swapnil0*swapnil1\n        True\n        >>> swapnil0*swapnil1 == swapnil1*swapnil0\n        False\n        >>> swapnil1**0 == swap0**0\n        False\n\n        '
        group = self.group
        if not isinstance(other, group.dtype):
            return False
        return tuple.__eq__(self, other)

    def __lt__(self, other):
        if False:
            return 10
        '\n        The  ordering  of  associative  words is defined by length and\n        lexicography (this ordering is called short-lex ordering), that\n        is, shorter words are smaller than longer words, and words of the\n        same length are compared w.r.t. the lexicographical ordering induced\n        by the ordering of generators. Generators  are  sorted  according\n        to the order in which they were created. If the generators are\n        invertible then each generator `g` is larger than its inverse `g^{-1}`,\n        and `g^{-1}` is larger than every generator that is smaller than `g`.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, a, b = free_group("a b")\n        >>> b < a\n        False\n        >>> a < a.inverse()\n        False\n\n        '
        group = self.group
        if not isinstance(other, group.dtype):
            raise TypeError('only FreeGroup elements of same FreeGroup can be compared')
        l = len(self)
        m = len(other)
        if l < m:
            return True
        elif l > m:
            return False
        for i in range(l):
            a = self[i].array_form[0]
            b = other[i].array_form[0]
            p = group.symbols.index(a[0])
            q = group.symbols.index(b[0])
            if p < q:
                return True
            elif p > q:
                return False
            elif a[1] < b[1]:
                return True
            elif a[1] > b[1]:
                return False
        return False

    def __le__(self, other):
        if False:
            return 10
        return self == other or self < other

    def __gt__(self, other):
        if False:
            i = 10
            return i + 15
        '\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, x, y, z = free_group("x y z")\n        >>> y**2 > x**2\n        True\n        >>> y*z > z*y\n        False\n        >>> x > x.inverse()\n        True\n\n        '
        group = self.group
        if not isinstance(other, group.dtype):
            raise TypeError('only FreeGroup elements of same FreeGroup can be compared')
        return not self <= other

    def __ge__(self, other):
        if False:
            while True:
                i = 10
        return not self < other

    def exponent_sum(self, gen):
        if False:
            while True:
                i = 10
        '\n        For an associative word `self` and a generator or inverse of generator\n        `gen`, ``exponent_sum`` returns the number of times `gen` appears in\n        `self` minus the number of times its inverse appears in `self`. If\n        neither `gen` nor its inverse occur in `self` then 0 is returned.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> F, x, y = free_group("x, y")\n        >>> w = x**2*y**3\n        >>> w.exponent_sum(x)\n        2\n        >>> w.exponent_sum(x**-1)\n        -2\n        >>> w = x**2*y**4*x**-3\n        >>> w.exponent_sum(x)\n        -1\n\n        See Also\n        ========\n\n        generator_count\n\n        '
        if len(gen) != 1:
            raise ValueError('gen must be a generator or inverse of a generator')
        s = gen.array_form[0]
        return s[1] * sum([i[1] for i in self.array_form if i[0] == s[0]])

    def generator_count(self, gen):
        if False:
            for i in range(10):
                print('nop')
        '\n        For an associative word `self` and a generator `gen`,\n        ``generator_count`` returns the multiplicity of generator\n        `gen` in `self`.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> F, x, y = free_group("x, y")\n        >>> w = x**2*y**3\n        >>> w.generator_count(x)\n        2\n        >>> w = x**2*y**4*x**-3\n        >>> w.generator_count(x)\n        5\n\n        See Also\n        ========\n\n        exponent_sum\n\n        '
        if len(gen) != 1 or gen.array_form[0][1] < 0:
            raise ValueError('gen must be a generator')
        s = gen.array_form[0]
        return s[1] * sum([abs(i[1]) for i in self.array_form if i[0] == s[0]])

    def subword(self, from_i, to_j, strict=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        For an associative word `self` and two positive integers `from_i` and\n        `to_j`, `subword` returns the subword of `self` that begins at position\n        `from_i` and ends at `to_j - 1`, indexing is done with origin 0.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, a, b = free_group("a b")\n        >>> w = a**5*b*a**2*b**-4*a\n        >>> w.subword(2, 6)\n        a**3*b\n\n        '
        group = self.group
        if not strict:
            from_i = max(from_i, 0)
            to_j = min(len(self), to_j)
        if from_i < 0 or to_j > len(self):
            raise ValueError('`from_i`, `to_j` must be positive and no greater than the length of associative word')
        if to_j <= from_i:
            return group.identity
        else:
            letter_form = self.letter_form[from_i:to_j]
            array_form = letter_form_to_array_form(letter_form, group)
            return group.dtype(array_form)

    def subword_index(self, word, start=0):
        if False:
            while True:
                i = 10
        '\n        Find the index of `word` in `self`.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, a, b = free_group("a b")\n        >>> w = a**2*b*a*b**3\n        >>> w.subword_index(a*b*a*b)\n        1\n\n        '
        l = len(word)
        self_lf = self.letter_form
        word_lf = word.letter_form
        index = None
        for i in range(start, len(self_lf) - l + 1):
            if self_lf[i:i + l] == word_lf:
                index = i
                break
        if index is not None:
            return index
        else:
            raise ValueError('The given word is not a subword of self')

    def is_dependent(self, word):
        if False:
            return 10
        '\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> F, x, y = free_group("x, y")\n        >>> (x**4*y**-3).is_dependent(x**4*y**-2)\n        True\n        >>> (x**2*y**-1).is_dependent(x*y)\n        False\n        >>> (x*y**2*x*y**2).is_dependent(x*y**2)\n        True\n        >>> (x**12).is_dependent(x**-4)\n        True\n\n        See Also\n        ========\n\n        is_independent\n\n        '
        try:
            return self.subword_index(word) is not None
        except ValueError:
            pass
        try:
            return self.subword_index(word ** (-1)) is not None
        except ValueError:
            return False

    def is_independent(self, word):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        See Also\n        ========\n\n        is_dependent\n\n        '
        return not self.is_dependent(word)

    def contains_generators(self):
        if False:
            print('Hello World!')
        '\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> F, x, y, z = free_group("x, y, z")\n        >>> (x**2*y**-1).contains_generators()\n        {x, y}\n        >>> (x**3*z).contains_generators()\n        {x, z}\n\n        '
        group = self.group
        gens = set()
        for syllable in self.array_form:
            gens.add(group.dtype(((syllable[0], 1),)))
        return set(gens)

    def cyclic_subword(self, from_i, to_j):
        if False:
            while True:
                i = 10
        group = self.group
        l = len(self)
        letter_form = self.letter_form
        period1 = int(from_i / l)
        if from_i >= l:
            from_i -= l * period1
            to_j -= l * period1
        diff = to_j - from_i
        word = letter_form[from_i:to_j]
        period2 = int(to_j / l) - 1
        word += letter_form * period2 + letter_form[:diff - l + from_i - l * period2]
        word = letter_form_to_array_form(word, group)
        return group.dtype(word)

    def cyclic_conjugates(self):
        if False:
            i = 10
            return i + 15
        'Returns a words which are cyclic to the word `self`.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> F, x, y = free_group("x, y")\n        >>> w = x*y*x*y*x\n        >>> w.cyclic_conjugates()\n        {x*y*x**2*y, x**2*y*x*y, y*x*y*x**2, y*x**2*y*x, x*y*x*y*x}\n        >>> s = x*y*x**2*y*x\n        >>> s.cyclic_conjugates()\n        {x**2*y*x**2*y, y*x**2*y*x**2, x*y*x**2*y*x}\n\n        References\n        ==========\n\n        .. [1] https://planetmath.org/cyclicpermutation\n\n        '
        return {self.cyclic_subword(i, i + len(self)) for i in range(len(self))}

    def is_cyclic_conjugate(self, w):
        if False:
            print('Hello World!')
        '\n        Checks whether words ``self``, ``w`` are cyclic conjugates.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> F, x, y = free_group("x, y")\n        >>> w1 = x**2*y**5\n        >>> w2 = x*y**5*x\n        >>> w1.is_cyclic_conjugate(w2)\n        True\n        >>> w3 = x**-1*y**5*x**-1\n        >>> w3.is_cyclic_conjugate(w2)\n        False\n\n        '
        l1 = len(self)
        l2 = len(w)
        if l1 != l2:
            return False
        w1 = self.identity_cyclic_reduction()
        w2 = w.identity_cyclic_reduction()
        letter1 = w1.letter_form
        letter2 = w2.letter_form
        str1 = ' '.join(map(str, letter1))
        str2 = ' '.join(map(str, letter2))
        if len(str1) != len(str2):
            return False
        return str1 in str2 + ' ' + str2

    def number_syllables(self):
        if False:
            i = 10
            return i + 15
        'Returns the number of syllables of the associative word `self`.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, swapnil0, swapnil1 = free_group("swapnil0 swapnil1")\n        >>> (swapnil1**3*swapnil0*swapnil1**-1).number_syllables()\n        3\n\n        '
        return len(self.array_form)

    def exponent_syllable(self, i):
        if False:
            print('Hello World!')
        '\n        Returns the exponent of the `i`-th syllable of the associative word\n        `self`.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, a, b = free_group("a b")\n        >>> w = a**5*b*a**2*b**-4*a\n        >>> w.exponent_syllable( 2 )\n        2\n\n        '
        return self.array_form[i][1]

    def generator_syllable(self, i):
        if False:
            print('Hello World!')
        '\n        Returns the symbol of the generator that is involved in the\n        i-th syllable of the associative word `self`.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, a, b = free_group("a b")\n        >>> w = a**5*b*a**2*b**-4*a\n        >>> w.generator_syllable( 3 )\n        b\n\n        '
        return self.array_form[i][0]

    def sub_syllables(self, from_i, to_j):
        if False:
            print('Hello World!')
        '\n        `sub_syllables` returns the subword of the associative word `self` that\n        consists of syllables from positions `from_to` to `to_j`, where\n        `from_to` and `to_j` must be positive integers and indexing is done\n        with origin 0.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> f, a, b = free_group("a, b")\n        >>> w = a**5*b*a**2*b**-4*a\n        >>> w.sub_syllables(1, 2)\n        b\n        >>> w.sub_syllables(3, 3)\n        <identity>\n\n        '
        if not isinstance(from_i, int) or not isinstance(to_j, int):
            raise ValueError('both arguments should be integers')
        group = self.group
        if to_j <= from_i:
            return group.identity
        else:
            r = tuple(self.array_form[from_i:to_j])
            return group.dtype(r)

    def substituted_word(self, from_i, to_j, by):
        if False:
            return 10
        '\n        Returns the associative word obtained by replacing the subword of\n        `self` that begins at position `from_i` and ends at position `to_j - 1`\n        by the associative word `by`. `from_i` and `to_j` must be positive\n        integers, indexing is done with origin 0. In other words,\n        `w.substituted_word(w, from_i, to_j, by)` is the product of the three\n        words: `w.subword(0, from_i)`, `by`, and\n        `w.subword(to_j len(w))`.\n\n        See Also\n        ========\n\n        eliminate_word\n\n        '
        lw = len(self)
        if from_i >= to_j or from_i > lw or to_j > lw:
            raise ValueError('values should be within bounds')
        if from_i == 0 and to_j == lw:
            return by
        elif from_i == 0:
            return by * self.subword(to_j, lw)
        elif to_j == lw:
            return self.subword(0, from_i) * by
        else:
            return self.subword(0, from_i) * by * self.subword(to_j, lw)

    def is_cyclically_reduced(self):
        if False:
            print('Hello World!')
        'Returns whether the word is cyclically reduced or not.\n        A word is cyclically reduced if by forming the cycle of the\n        word, the word is not reduced, i.e a word w = `a_1 ... a_n`\n        is called cyclically reduced if `a_1 \\ne a_n^{-1}`.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> F, x, y = free_group("x, y")\n        >>> (x**2*y**-1*x**-1).is_cyclically_reduced()\n        False\n        >>> (y*x**2*y**2).is_cyclically_reduced()\n        True\n\n        '
        if not self:
            return True
        return self[0] != self[-1] ** (-1)

    def identity_cyclic_reduction(self):
        if False:
            return 10
        'Return a unique cyclically reduced version of the word.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> F, x, y = free_group("x, y")\n        >>> (x**2*y**2*x**-1).identity_cyclic_reduction()\n        x*y**2\n        >>> (x**-3*y**-1*x**5).identity_cyclic_reduction()\n        x**2*y**-1\n\n        References\n        ==========\n\n        .. [1] https://planetmath.org/cyclicallyreduced\n\n        '
        word = self.copy()
        group = self.group
        while not word.is_cyclically_reduced():
            exp1 = word.exponent_syllable(0)
            exp2 = word.exponent_syllable(-1)
            r = exp1 + exp2
            if r == 0:
                rep = word.array_form[1:word.number_syllables() - 1]
            else:
                rep = ((word.generator_syllable(0), exp1 + exp2),) + word.array_form[1:word.number_syllables() - 1]
            word = group.dtype(rep)
        return word

    def cyclic_reduction(self, removed=False):
        if False:
            i = 10
            return i + 15
        'Return a cyclically reduced version of the word. Unlike\n        `identity_cyclic_reduction`, this will not cyclically permute\n        the reduced word - just remove the "unreduced" bits on either\n        side of it. Compare the examples with those of\n        `identity_cyclic_reduction`.\n\n        When `removed` is `True`, return a tuple `(word, r)` where\n        self `r` is such that before the reduction the word was either\n        `r*word*r**-1`.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> F, x, y = free_group("x, y")\n        >>> (x**2*y**2*x**-1).cyclic_reduction()\n        x*y**2\n        >>> (x**-3*y**-1*x**5).cyclic_reduction()\n        y**-1*x**2\n        >>> (x**-3*y**-1*x**5).cyclic_reduction(removed=True)\n        (y**-1*x**2, x**-3)\n\n        '
        word = self.copy()
        g = self.group.identity
        while not word.is_cyclically_reduced():
            exp1 = abs(word.exponent_syllable(0))
            exp2 = abs(word.exponent_syllable(-1))
            exp = min(exp1, exp2)
            start = word[0] ** abs(exp)
            end = word[-1] ** abs(exp)
            word = start ** (-1) * word * end ** (-1)
            g = g * start
        if removed:
            return (word, g)
        return word

    def power_of(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if `self == other**n` for some integer n.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import free_group\n        >>> F, x, y = free_group("x, y")\n        >>> ((x*y)**2).power_of(x*y)\n        True\n        >>> (x**-3*y**-2*x**3).power_of(x**-3*y*x**3)\n        True\n\n        '
        if self.is_identity:
            return True
        l = len(other)
        if l == 1:
            gens = self.contains_generators()
            s = other in gens or other ** (-1) in gens
            return len(gens) == 1 and s
        (reduced, r1) = self.cyclic_reduction(removed=True)
        if not r1.is_identity:
            (other, r2) = other.cyclic_reduction(removed=True)
            if r1 == r2:
                return reduced.power_of(other)
            return False
        if len(self) < l or len(self) % l:
            return False
        prefix = self.subword(0, l)
        if prefix == other or prefix ** (-1) == other:
            rest = self.subword(l, len(self))
            return rest.power_of(other)
        return False

def letter_form_to_array_form(array_form, group):
    if False:
        print('Hello World!')
    '\n    This method converts a list given with possible repetitions of elements in\n    it. It returns a new list such that repetitions of consecutive elements is\n    removed and replace with a tuple element of size two such that the first\n    index contains `value` and the second index contains the number of\n    consecutive repetitions of `value`.\n\n    '
    a = list(array_form[:])
    new_array = []
    n = 1
    symbols = group.symbols
    for i in range(len(a)):
        if i == len(a) - 1:
            if a[i] == a[i - 1]:
                if -a[i] in symbols:
                    new_array.append((-a[i], -n))
                else:
                    new_array.append((a[i], n))
            elif -a[i] in symbols:
                new_array.append((-a[i], -1))
            else:
                new_array.append((a[i], 1))
            return new_array
        elif a[i] == a[i + 1]:
            n += 1
        else:
            if -a[i] in symbols:
                new_array.append((-a[i], -n))
            else:
                new_array.append((a[i], n))
            n = 1

def zero_mul_simp(l, index):
    if False:
        while True:
            i = 10
    'Used to combine two reduced words.'
    while index >= 0 and index < len(l) - 1 and (l[index][0] == l[index + 1][0]):
        exp = l[index][1] + l[index + 1][1]
        base = l[index][0]
        l[index] = (base, exp)
        del l[index + 1]
        if l[index][1] == 0:
            del l[index]
            index -= 1