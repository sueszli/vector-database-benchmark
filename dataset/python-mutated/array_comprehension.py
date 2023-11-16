import functools, itertools
from sympy.core.sympify import _sympify, sympify
from sympy.core.expr import Expr
from sympy.core import Basic, Tuple
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer

class ArrayComprehension(Basic):
    """
    Generate a list comprehension.

    Explanation
    ===========

    If there is a symbolic dimension, for example, say [i for i in range(1, N)] where
    N is a Symbol, then the expression will not be expanded to an array. Otherwise,
    calling the doit() function will launch the expansion.

    Examples
    ========

    >>> from sympy.tensor.array import ArrayComprehension
    >>> from sympy import symbols
    >>> i, j, k = symbols('i j k')
    >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
    >>> a
    ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
    >>> a.doit()
    [[11, 12, 13], [21, 22, 23], [31, 32, 33], [41, 42, 43]]
    >>> b = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k))
    >>> b.doit()
    ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k))
    """

    def __new__(cls, function, *symbols, **assumptions):
        if False:
            i = 10
            return i + 15
        if any((len(l) != 3 or None for l in symbols)):
            raise ValueError('ArrayComprehension requires values lower and upper bound for the expression')
        arglist = [sympify(function)]
        arglist.extend(cls._check_limits_validity(function, symbols))
        obj = Basic.__new__(cls, *arglist, **assumptions)
        obj._limits = obj._args[1:]
        obj._shape = cls._calculate_shape_from_limits(obj._limits)
        obj._rank = len(obj._shape)
        obj._loop_size = cls._calculate_loop_size(obj._shape)
        return obj

    @property
    def function(self):
        if False:
            while True:
                i = 10
        "The function applied across limits.\n\n        Examples\n        ========\n\n        >>> from sympy.tensor.array import ArrayComprehension\n        >>> from sympy import symbols\n        >>> i, j = symbols('i j')\n        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))\n        >>> a.function\n        10*i + j\n        "
        return self._args[0]

    @property
    def limits(self):
        if False:
            while True:
                i = 10
        "\n        The list of limits that will be applied while expanding the array.\n\n        Examples\n        ========\n\n        >>> from sympy.tensor.array import ArrayComprehension\n        >>> from sympy import symbols\n        >>> i, j = symbols('i j')\n        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))\n        >>> a.limits\n        ((i, 1, 4), (j, 1, 3))\n        "
        return self._limits

    @property
    def free_symbols(self):
        if False:
            while True:
                i = 10
        "\n        The set of the free_symbols in the array.\n        Variables appeared in the bounds are supposed to be excluded\n        from the free symbol set.\n\n        Examples\n        ========\n\n        >>> from sympy.tensor.array import ArrayComprehension\n        >>> from sympy import symbols\n        >>> i, j, k = symbols('i j k')\n        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))\n        >>> a.free_symbols\n        set()\n        >>> b = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k+3))\n        >>> b.free_symbols\n        {k}\n        "
        expr_free_sym = self.function.free_symbols
        for (var, inf, sup) in self._limits:
            expr_free_sym.discard(var)
            curr_free_syms = inf.free_symbols.union(sup.free_symbols)
            expr_free_sym = expr_free_sym.union(curr_free_syms)
        return expr_free_sym

    @property
    def variables(self):
        if False:
            print('Hello World!')
        "The tuples of the variables in the limits.\n\n        Examples\n        ========\n\n        >>> from sympy.tensor.array import ArrayComprehension\n        >>> from sympy import symbols\n        >>> i, j, k = symbols('i j k')\n        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))\n        >>> a.variables\n        [i, j]\n        "
        return [l[0] for l in self._limits]

    @property
    def bound_symbols(self):
        if False:
            return 10
        'The list of dummy variables.\n\n        Note\n        ====\n\n        Note that all variables are dummy variables since a limit without\n        lower bound or upper bound is not accepted.\n        '
        return [l[0] for l in self._limits if len(l) != 1]

    @property
    def shape(self):
        if False:
            print('Hello World!')
        "\n        The shape of the expanded array, which may have symbols.\n\n        Note\n        ====\n\n        Both the lower and the upper bounds are included while\n        calculating the shape.\n\n        Examples\n        ========\n\n        >>> from sympy.tensor.array import ArrayComprehension\n        >>> from sympy import symbols\n        >>> i, j, k = symbols('i j k')\n        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))\n        >>> a.shape\n        (4, 3)\n        >>> b = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k+3))\n        >>> b.shape\n        (4, k + 3)\n        "
        return self._shape

    @property
    def is_shape_numeric(self):
        if False:
            while True:
                i = 10
        "\n        Test if the array is shape-numeric which means there is no symbolic\n        dimension.\n\n        Examples\n        ========\n\n        >>> from sympy.tensor.array import ArrayComprehension\n        >>> from sympy import symbols\n        >>> i, j, k = symbols('i j k')\n        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))\n        >>> a.is_shape_numeric\n        True\n        >>> b = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k+3))\n        >>> b.is_shape_numeric\n        False\n        "
        for (_, inf, sup) in self._limits:
            if Basic(inf, sup).atoms(Symbol):
                return False
        return True

    def rank(self):
        if False:
            return 10
        "The rank of the expanded array.\n\n        Examples\n        ========\n\n        >>> from sympy.tensor.array import ArrayComprehension\n        >>> from sympy import symbols\n        >>> i, j, k = symbols('i j k')\n        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))\n        >>> a.rank()\n        2\n        "
        return self._rank

    def __len__(self):
        if False:
            while True:
                i = 10
        "\n        The length of the expanded array which means the number\n        of elements in the array.\n\n        Raises\n        ======\n\n        ValueError : When the length of the array is symbolic\n\n        Examples\n        ========\n\n        >>> from sympy.tensor.array import ArrayComprehension\n        >>> from sympy import symbols\n        >>> i, j = symbols('i j')\n        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))\n        >>> len(a)\n        12\n        "
        if self._loop_size.free_symbols:
            raise ValueError('Symbolic length is not supported')
        return self._loop_size

    @classmethod
    def _check_limits_validity(cls, function, limits):
        if False:
            i = 10
            return i + 15
        new_limits = []
        for (var, inf, sup) in limits:
            var = _sympify(var)
            inf = _sympify(inf)
            if isinstance(sup, list):
                sup = Tuple(*sup)
            else:
                sup = _sympify(sup)
            new_limits.append(Tuple(var, inf, sup))
            if any((not isinstance(i, Expr) or i.atoms(Symbol, Integer) != i.atoms() for i in [inf, sup])):
                raise TypeError('Bounds should be an Expression(combination of Integer and Symbol)')
            if (inf > sup) == True:
                raise ValueError('Lower bound should be inferior to upper bound')
            if var in inf.free_symbols or var in sup.free_symbols:
                raise ValueError('Variable should not be part of its bounds')
        return new_limits

    @classmethod
    def _calculate_shape_from_limits(cls, limits):
        if False:
            while True:
                i = 10
        return tuple([sup - inf + 1 for (_, inf, sup) in limits])

    @classmethod
    def _calculate_loop_size(cls, shape):
        if False:
            return 10
        if not shape:
            return 0
        loop_size = 1
        for l in shape:
            loop_size = loop_size * l
        return loop_size

    def doit(self, **hints):
        if False:
            for i in range(10):
                print('nop')
        if not self.is_shape_numeric:
            return self
        return self._expand_array()

    def _expand_array(self):
        if False:
            for i in range(10):
                print('nop')
        res = []
        for values in itertools.product(*[range(inf, sup + 1) for (var, inf, sup) in self._limits]):
            res.append(self._get_element(values))
        return ImmutableDenseNDimArray(res, self.shape)

    def _get_element(self, values):
        if False:
            return 10
        temp = self.function
        for (var, val) in zip(self.variables, values):
            temp = temp.subs(var, val)
        return temp

    def tolist(self):
        if False:
            for i in range(10):
                print('nop')
        "Transform the expanded array to a list.\n\n        Raises\n        ======\n\n        ValueError : When there is a symbolic dimension\n\n        Examples\n        ========\n\n        >>> from sympy.tensor.array import ArrayComprehension\n        >>> from sympy import symbols\n        >>> i, j = symbols('i j')\n        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))\n        >>> a.tolist()\n        [[11, 12, 13], [21, 22, 23], [31, 32, 33], [41, 42, 43]]\n        "
        if self.is_shape_numeric:
            return self._expand_array().tolist()
        raise ValueError('A symbolic array cannot be expanded to a list')

    def tomatrix(self):
        if False:
            print('Hello World!')
        "Transform the expanded array to a matrix.\n\n        Raises\n        ======\n\n        ValueError : When there is a symbolic dimension\n        ValueError : When the rank of the expanded array is not equal to 2\n\n        Examples\n        ========\n\n        >>> from sympy.tensor.array import ArrayComprehension\n        >>> from sympy import symbols\n        >>> i, j = symbols('i j')\n        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))\n        >>> a.tomatrix()\n        Matrix([\n        [11, 12, 13],\n        [21, 22, 23],\n        [31, 32, 33],\n        [41, 42, 43]])\n        "
        from sympy.matrices import Matrix
        if not self.is_shape_numeric:
            raise ValueError('A symbolic array cannot be expanded to a matrix')
        if self._rank != 2:
            raise ValueError('Dimensions must be of size of 2')
        return Matrix(self._expand_array().tomatrix())

def isLambda(v):
    if False:
        i = 10
        return i + 15
    LAMBDA = lambda : 0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__

class ArrayComprehensionMap(ArrayComprehension):
    """
    A subclass of ArrayComprehension dedicated to map external function lambda.

    Notes
    =====

    Only the lambda function is considered.
    At most one argument in lambda function is accepted in order to avoid ambiguity
    in value assignment.

    Examples
    ========

    >>> from sympy.tensor.array import ArrayComprehensionMap
    >>> from sympy import symbols
    >>> i, j, k = symbols('i j k')
    >>> a = ArrayComprehensionMap(lambda: 1, (i, 1, 4))
    >>> a.doit()
    [1, 1, 1, 1]
    >>> b = ArrayComprehensionMap(lambda a: a+1, (j, 1, 4))
    >>> b.doit()
    [2, 3, 4, 5]

    """

    def __new__(cls, function, *symbols, **assumptions):
        if False:
            return 10
        if any((len(l) != 3 or None for l in symbols)):
            raise ValueError('ArrayComprehension requires values lower and upper bound for the expression')
        if not isLambda(function):
            raise ValueError('Data type not supported')
        arglist = cls._check_limits_validity(function, symbols)
        obj = Basic.__new__(cls, *arglist, **assumptions)
        obj._limits = obj._args
        obj._shape = cls._calculate_shape_from_limits(obj._limits)
        obj._rank = len(obj._shape)
        obj._loop_size = cls._calculate_loop_size(obj._shape)
        obj._lambda = function
        return obj

    @property
    def func(self):
        if False:
            return 10

        class _(ArrayComprehensionMap):

            def __new__(cls, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                return ArrayComprehensionMap(self._lambda, *args, **kwargs)
        return _

    def _get_element(self, values):
        if False:
            i = 10
            return i + 15
        temp = self._lambda
        if self._lambda.__code__.co_argcount == 0:
            temp = temp()
        elif self._lambda.__code__.co_argcount == 1:
            temp = temp(functools.reduce(lambda a, b: a * b, values))
        return temp