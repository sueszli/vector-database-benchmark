import collections.abc
import operator
from collections import defaultdict, Counter
from functools import reduce
import itertools
from itertools import accumulate
from typing import Optional, List, Tuple as tTuple
import typing
from sympy.core.numbers import Integer
from sympy.core.relational import Equality
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function, Lambda
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.matrices.common import MatrixCommon
from sympy.matrices.expressions.diagonal import diagonalize_vector
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.tensor.array.arrayop import permutedims, tensorcontraction, tensordiagonal, tensorproduct
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.tensor.array.ndim_array import NDimArray
from sympy.tensor.indexed import Indexed, IndexedBase
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.utils import _apply_recursively_over_nested_lists, _sort_contraction_indices, _get_mapping_from_subranks, _build_push_indices_up_func_transformation, _get_contraction_links, _build_push_indices_down_func_transformation
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import _af_invert
from sympy.core.sympify import _sympify

class _ArrayExpr(Expr):
    shape: tTuple[Expr, ...]

    def __getitem__(self, item):
        if False:
            print('Hello World!')
        if not isinstance(item, collections.abc.Iterable):
            item = (item,)
        ArrayElement._check_shape(self, item)
        return self._get(item)

    def _get(self, item):
        if False:
            print('Hello World!')
        return _get_array_element_or_slice(self, item)

class ArraySymbol(_ArrayExpr):
    """
    Symbol representing an array expression
    """

    def __new__(cls, symbol, shape: typing.Iterable) -> 'ArraySymbol':
        if False:
            print('Hello World!')
        if isinstance(symbol, str):
            symbol = Symbol(symbol)
        shape = Tuple(*map(_sympify, shape))
        obj = Expr.__new__(cls, symbol, shape)
        return obj

    @property
    def name(self):
        if False:
            return 10
        return self._args[0]

    @property
    def shape(self):
        if False:
            i = 10
            return i + 15
        return self._args[1]

    def as_explicit(self):
        if False:
            return 10
        if not all((i.is_Integer for i in self.shape)):
            raise ValueError('cannot express explicit array with symbolic shape')
        data = [self[i] for i in itertools.product(*[range(j) for j in self.shape])]
        return ImmutableDenseNDimArray(data).reshape(*self.shape)

class ArrayElement(Expr):
    """
    An element of an array.
    """
    _diff_wrt = True
    is_symbol = True
    is_commutative = True

    def __new__(cls, name, indices):
        if False:
            return 10
        if isinstance(name, str):
            name = Symbol(name)
        name = _sympify(name)
        if not isinstance(indices, collections.abc.Iterable):
            indices = (indices,)
        indices = _sympify(tuple(indices))
        cls._check_shape(name, indices)
        obj = Expr.__new__(cls, name, indices)
        return obj

    @classmethod
    def _check_shape(cls, name, indices):
        if False:
            return 10
        indices = tuple(indices)
        if hasattr(name, 'shape'):
            index_error = IndexError('number of indices does not match shape of the array')
            if len(indices) != len(name.shape):
                raise index_error
            if any(((i >= s) == True for (i, s) in zip(indices, name.shape))):
                raise ValueError('shape is out of bounds')
        if any(((i < 0) == True for i in indices)):
            raise ValueError('shape contains negative values')

    @property
    def name(self):
        if False:
            print('Hello World!')
        return self._args[0]

    @property
    def indices(self):
        if False:
            print('Hello World!')
        return self._args[1]

    def _eval_derivative(self, s):
        if False:
            i = 10
            return i + 15
        if not isinstance(s, ArrayElement):
            return S.Zero
        if s == self:
            return S.One
        if s.name != self.name:
            return S.Zero
        return Mul.fromiter((KroneckerDelta(i, j) for (i, j) in zip(self.indices, s.indices)))

class ZeroArray(_ArrayExpr):
    """
    Symbolic array of zeros. Equivalent to ``ZeroMatrix`` for matrices.
    """

    def __new__(cls, *shape):
        if False:
            print('Hello World!')
        if len(shape) == 0:
            return S.Zero
        shape = map(_sympify, shape)
        obj = Expr.__new__(cls, *shape)
        return obj

    @property
    def shape(self):
        if False:
            while True:
                i = 10
        return self._args

    def as_explicit(self):
        if False:
            i = 10
            return i + 15
        if not all((i.is_Integer for i in self.shape)):
            raise ValueError('Cannot return explicit form for symbolic shape.')
        return ImmutableDenseNDimArray.zeros(*self.shape)

    def _get(self, item):
        if False:
            print('Hello World!')
        return S.Zero

class OneArray(_ArrayExpr):
    """
    Symbolic array of ones.
    """

    def __new__(cls, *shape):
        if False:
            print('Hello World!')
        if len(shape) == 0:
            return S.One
        shape = map(_sympify, shape)
        obj = Expr.__new__(cls, *shape)
        return obj

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        return self._args

    def as_explicit(self):
        if False:
            return 10
        if not all((i.is_Integer for i in self.shape)):
            raise ValueError('Cannot return explicit form for symbolic shape.')
        return ImmutableDenseNDimArray([S.One for i in range(reduce(operator.mul, self.shape))]).reshape(*self.shape)

    def _get(self, item):
        if False:
            return 10
        return S.One

class _CodegenArrayAbstract(Basic):

    @property
    def subranks(self):
        if False:
            print('Hello World!')
        '\n        Returns the ranks of the objects in the uppermost tensor product inside\n        the current object.  In case no tensor products are contained, return\n        the atomic ranks.\n\n        Examples\n        ========\n\n        >>> from sympy.tensor.array import tensorproduct, tensorcontraction\n        >>> from sympy import MatrixSymbol\n        >>> M = MatrixSymbol("M", 3, 3)\n        >>> N = MatrixSymbol("N", 3, 3)\n        >>> P = MatrixSymbol("P", 3, 3)\n\n        Important: do not confuse the rank of the matrix with the rank of an array.\n\n        >>> tp = tensorproduct(M, N, P)\n        >>> tp.subranks\n        [2, 2, 2]\n\n        >>> co = tensorcontraction(tp, (1, 2), (3, 4))\n        >>> co.subranks\n        [2, 2, 2]\n        '
        return self._subranks[:]

    def subrank(self):
        if False:
            i = 10
            return i + 15
        '\n        The sum of ``subranks``.\n        '
        return sum(self.subranks)

    @property
    def shape(self):
        if False:
            i = 10
            return i + 15
        return self._shape

    def doit(self, **hints):
        if False:
            while True:
                i = 10
        deep = hints.get('deep', True)
        if deep:
            return self.func(*[arg.doit(**hints) for arg in self.args])._canonicalize()
        else:
            return self._canonicalize()

class ArrayTensorProduct(_CodegenArrayAbstract):
    """
    Class to represent the tensor product of array-like objects.
    """

    def __new__(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        args = [_sympify(arg) for arg in args]
        canonicalize = kwargs.pop('canonicalize', False)
        ranks = [get_rank(arg) for arg in args]
        obj = Basic.__new__(cls, *args)
        obj._subranks = ranks
        shapes = [get_shape(i) for i in args]
        if any((i is None for i in shapes)):
            obj._shape = None
        else:
            obj._shape = tuple((j for i in shapes for j in i))
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        if False:
            for i in range(10):
                print('nop')
        args = self.args
        args = self._flatten(args)
        ranks = [get_rank(arg) for arg in args]
        permutation_cycles = []
        for (i, arg) in enumerate(args):
            if not isinstance(arg, PermuteDims):
                continue
            permutation_cycles.extend([[k + sum(ranks[:i]) for k in j] for j in arg.permutation.cyclic_form])
            args[i] = arg.expr
        if permutation_cycles:
            return _permute_dims(_array_tensor_product(*args), Permutation(sum(ranks) - 1) * Permutation(permutation_cycles))
        if len(args) == 1:
            return args[0]
        if any((isinstance(arg, (ZeroArray, ZeroMatrix)) for arg in args)):
            shapes = reduce(operator.add, [get_shape(i) for i in args], ())
            return ZeroArray(*shapes)
        contractions = {i: arg for (i, arg) in enumerate(args) if isinstance(arg, ArrayContraction)}
        if contractions:
            ranks = [_get_subrank(arg) if isinstance(arg, ArrayContraction) else get_rank(arg) for arg in args]
            cumulative_ranks = list(accumulate([0] + ranks))[:-1]
            tp = _array_tensor_product(*[arg.expr if isinstance(arg, ArrayContraction) else arg for arg in args])
            contraction_indices = [tuple((cumulative_ranks[i] + k for k in j)) for (i, arg) in contractions.items() for j in arg.contraction_indices]
            return _array_contraction(tp, *contraction_indices)
        diagonals = {i: arg for (i, arg) in enumerate(args) if isinstance(arg, ArrayDiagonal)}
        if diagonals:
            inverse_permutation = []
            last_perm = []
            ranks = [get_rank(arg) for arg in args]
            cumulative_ranks = list(accumulate([0] + ranks))[:-1]
            for (i, arg) in enumerate(args):
                if isinstance(arg, ArrayDiagonal):
                    i1 = get_rank(arg) - len(arg.diagonal_indices)
                    i2 = len(arg.diagonal_indices)
                    inverse_permutation.extend([cumulative_ranks[i] + j for j in range(i1)])
                    last_perm.extend([cumulative_ranks[i] + j for j in range(i1, i1 + i2)])
                else:
                    inverse_permutation.extend([cumulative_ranks[i] + j for j in range(get_rank(arg))])
            inverse_permutation.extend(last_perm)
            tp = _array_tensor_product(*[arg.expr if isinstance(arg, ArrayDiagonal) else arg for arg in args])
            ranks2 = [_get_subrank(arg) if isinstance(arg, ArrayDiagonal) else get_rank(arg) for arg in args]
            cumulative_ranks2 = list(accumulate([0] + ranks2))[:-1]
            diagonal_indices = [tuple((cumulative_ranks2[i] + k for k in j)) for (i, arg) in diagonals.items() for j in arg.diagonal_indices]
            return _permute_dims(_array_diagonal(tp, *diagonal_indices), _af_invert(inverse_permutation))
        return self.func(*args, canonicalize=False)

    @classmethod
    def _flatten(cls, args):
        if False:
            while True:
                i = 10
        args = [i for arg in args for i in (arg.args if isinstance(arg, cls) else [arg])]
        return args

    def as_explicit(self):
        if False:
            for i in range(10):
                print('nop')
        return tensorproduct(*[arg.as_explicit() if hasattr(arg, 'as_explicit') else arg for arg in self.args])

class ArrayAdd(_CodegenArrayAbstract):
    """
    Class for elementwise array additions.
    """

    def __new__(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        args = [_sympify(arg) for arg in args]
        ranks = [get_rank(arg) for arg in args]
        ranks = list(set(ranks))
        if len(ranks) != 1:
            raise ValueError('summing arrays of different ranks')
        shapes = [arg.shape for arg in args]
        if len({i for i in shapes if i is not None}) > 1:
            raise ValueError('mismatching shapes in addition')
        canonicalize = kwargs.pop('canonicalize', False)
        obj = Basic.__new__(cls, *args)
        obj._subranks = ranks
        if any((i is None for i in shapes)):
            obj._shape = None
        else:
            obj._shape = shapes[0]
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        if False:
            for i in range(10):
                print('nop')
        args = self.args
        args = self._flatten_args(args)
        shapes = [get_shape(arg) for arg in args]
        args = [arg for arg in args if not isinstance(arg, (ZeroArray, ZeroMatrix))]
        if len(args) == 0:
            if any((i for i in shapes if i is None)):
                raise NotImplementedError('cannot handle addition of ZeroMatrix/ZeroArray and undefined shape object')
            return ZeroArray(*shapes[0])
        elif len(args) == 1:
            return args[0]
        return self.func(*args, canonicalize=False)

    @classmethod
    def _flatten_args(cls, args):
        if False:
            return 10
        new_args = []
        for arg in args:
            if isinstance(arg, ArrayAdd):
                new_args.extend(arg.args)
            else:
                new_args.append(arg)
        return new_args

    def as_explicit(self):
        if False:
            i = 10
            return i + 15
        return reduce(operator.add, [arg.as_explicit() if hasattr(arg, 'as_explicit') else arg for arg in self.args])

class PermuteDims(_CodegenArrayAbstract):
    """
    Class to represent permutation of axes of arrays.

    Examples
    ========

    >>> from sympy.tensor.array import permutedims
    >>> from sympy import MatrixSymbol
    >>> M = MatrixSymbol("M", 3, 3)
    >>> cg = permutedims(M, [1, 0])

    The object ``cg`` represents the transposition of ``M``, as the permutation
    ``[1, 0]`` will act on its indices by switching them:

    `M_{ij} \\Rightarrow M_{ji}`

    This is evident when transforming back to matrix form:

    >>> from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
    >>> convert_array_to_matrix(cg)
    M.T

    >>> N = MatrixSymbol("N", 3, 2)
    >>> cg = permutedims(N, [1, 0])
    >>> cg.shape
    (2, 3)

    There are optional parameters that can be used as alternative to the permutation:

    >>> from sympy.tensor.array.expressions import ArraySymbol, PermuteDims
    >>> M = ArraySymbol("M", (1, 2, 3, 4, 5))
    >>> expr = PermuteDims(M, index_order_old="ijklm", index_order_new="kijml")
    >>> expr
    PermuteDims(M, (0 2 1)(3 4))
    >>> expr.shape
    (3, 1, 2, 5, 4)

    Permutations of tensor products are simplified in order to achieve a
    standard form:

    >>> from sympy.tensor.array import tensorproduct
    >>> M = MatrixSymbol("M", 4, 5)
    >>> tp = tensorproduct(M, N)
    >>> tp.shape
    (4, 5, 3, 2)
    >>> perm1 = permutedims(tp, [2, 3, 1, 0])

    The args ``(M, N)`` have been sorted and the permutation has been
    simplified, the expression is equivalent:

    >>> perm1.expr.args
    (N, M)
    >>> perm1.shape
    (3, 2, 5, 4)
    >>> perm1.permutation
    (2 3)

    The permutation in its array form has been simplified from
    ``[2, 3, 1, 0]`` to ``[0, 1, 3, 2]``, as the arguments of the tensor
    product `M` and `N` have been switched:

    >>> perm1.permutation.array_form
    [0, 1, 3, 2]

    We can nest a second permutation:

    >>> perm2 = permutedims(perm1, [1, 0, 2, 3])
    >>> perm2.shape
    (2, 3, 5, 4)
    >>> perm2.permutation.array_form
    [1, 0, 3, 2]
    """

    def __new__(cls, expr, permutation=None, index_order_old=None, index_order_new=None, **kwargs):
        if False:
            return 10
        from sympy.combinatorics import Permutation
        expr = _sympify(expr)
        expr_rank = get_rank(expr)
        permutation = cls._get_permutation_from_arguments(permutation, index_order_old, index_order_new, expr_rank)
        permutation = Permutation(permutation)
        permutation_size = permutation.size
        if permutation_size != expr_rank:
            raise ValueError('Permutation size must be the length of the shape of expr')
        canonicalize = kwargs.pop('canonicalize', False)
        obj = Basic.__new__(cls, expr, permutation)
        obj._subranks = [get_rank(expr)]
        shape = get_shape(expr)
        if shape is None:
            obj._shape = None
        else:
            obj._shape = tuple((shape[permutation(i)] for i in range(len(shape))))
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        if False:
            while True:
                i = 10
        expr = self.expr
        permutation = self.permutation
        if isinstance(expr, PermuteDims):
            subexpr = expr.expr
            subperm = expr.permutation
            permutation = permutation * subperm
            expr = subexpr
        if isinstance(expr, ArrayContraction):
            (expr, permutation) = self._PermuteDims_denestarg_ArrayContraction(expr, permutation)
        if isinstance(expr, ArrayTensorProduct):
            (expr, permutation) = self._PermuteDims_denestarg_ArrayTensorProduct(expr, permutation)
        if isinstance(expr, (ZeroArray, ZeroMatrix)):
            return ZeroArray(*[expr.shape[i] for i in permutation.array_form])
        plist = permutation.array_form
        if plist == sorted(plist):
            return expr
        return self.func(expr, permutation, canonicalize=False)

    @property
    def expr(self):
        if False:
            i = 10
            return i + 15
        return self.args[0]

    @property
    def permutation(self):
        if False:
            return 10
        return self.args[1]

    @classmethod
    def _PermuteDims_denestarg_ArrayTensorProduct(cls, expr, permutation):
        if False:
            return 10
        perm_image_form = _af_invert(permutation.array_form)
        args = list(expr.args)
        cumul = list(accumulate([0] + expr.subranks))
        perm_image_form_in_components = [perm_image_form[cumul[i]:cumul[i + 1]] for i in range(len(args))]
        ps = [(i, sorted(comp)) for (i, comp) in enumerate(perm_image_form_in_components)]
        ps.sort(key=lambda x: x[1])
        perm_args_image_form = [i[0] for i in ps]
        args_sorted = [args[i] for i in perm_args_image_form]
        perm_image_form_sorted_args = [perm_image_form_in_components[i] for i in perm_args_image_form]
        new_permutation = Permutation(_af_invert([j for i in perm_image_form_sorted_args for j in i]))
        return (_array_tensor_product(*args_sorted), new_permutation)

    @classmethod
    def _PermuteDims_denestarg_ArrayContraction(cls, expr, permutation):
        if False:
            return 10
        if not isinstance(expr, ArrayContraction):
            return (expr, permutation)
        if not isinstance(expr.expr, ArrayTensorProduct):
            return (expr, permutation)
        args = expr.expr.args
        subranks = [get_rank(arg) for arg in expr.expr.args]
        contraction_indices = expr.contraction_indices
        contraction_indices_flat = [j for i in contraction_indices for j in i]
        cumul = list(accumulate([0] + subranks))
        permutation_array_blocks_up = []
        image_form = _af_invert(permutation.array_form)
        counter = 0
        for (i, e) in enumerate(subranks):
            current = []
            for j in range(cumul[i], cumul[i + 1]):
                if j in contraction_indices_flat:
                    continue
                current.append(image_form[counter])
                counter += 1
            permutation_array_blocks_up.append(current)
        index_blocks = [list(range(cumul[i], cumul[i + 1])) for (i, e) in enumerate(expr.subranks)]
        index_blocks_up = expr._push_indices_up(expr.contraction_indices, index_blocks)
        inverse_permutation = permutation ** (-1)
        index_blocks_up_permuted = [[inverse_permutation(j) for j in i if j is not None] for i in index_blocks_up]
        sorting_keys = list(enumerate(index_blocks_up_permuted))
        sorting_keys.sort(key=lambda x: x[1])
        new_perm_image_form = [i[0] for i in sorting_keys]
        new_index_blocks = [index_blocks[i] for i in new_perm_image_form]
        new_index_perm_array_form = _af_invert([j for i in new_index_blocks for j in i])
        new_args = [args[i] for i in new_perm_image_form]
        new_contraction_indices = [tuple((new_index_perm_array_form[j] for j in i)) for i in contraction_indices]
        new_expr = _array_contraction(_array_tensor_product(*new_args), *new_contraction_indices)
        new_permutation = Permutation(_af_invert([j for i in [permutation_array_blocks_up[k] for k in new_perm_image_form] for j in i]))
        return (new_expr, new_permutation)

    @classmethod
    def _check_permutation_mapping(cls, expr, permutation):
        if False:
            while True:
                i = 10
        subranks = expr.subranks
        index2arg = [i for (i, arg) in enumerate(expr.args) for j in range(expr.subranks[i])]
        permuted_indices = [permutation(i) for i in range(expr.subrank())]
        new_args = list(expr.args)
        arg_candidate_index = index2arg[permuted_indices[0]]
        current_indices = []
        new_permutation = []
        inserted_arg_cand_indices = set()
        for (i, idx) in enumerate(permuted_indices):
            if index2arg[idx] != arg_candidate_index:
                new_permutation.extend(current_indices)
                current_indices = []
                arg_candidate_index = index2arg[idx]
            current_indices.append(idx)
            arg_candidate_rank = subranks[arg_candidate_index]
            if len(current_indices) == arg_candidate_rank:
                new_permutation.extend(sorted(current_indices))
                local_current_indices = [j - min(current_indices) for j in current_indices]
                i1 = index2arg[i]
                new_args[i1] = _permute_dims(new_args[i1], Permutation(local_current_indices))
                inserted_arg_cand_indices.add(arg_candidate_index)
                current_indices = []
        new_permutation.extend(current_indices)
        args_positions = list(range(len(new_args)))
        maps = {}
        cumulative_subranks = [0] + list(accumulate(subranks))
        for i in range(len(subranks)):
            s = {index2arg[new_permutation[j]] for j in range(cumulative_subranks[i], cumulative_subranks[i + 1])}
            if len(s) != 1:
                continue
            elem = next(iter(s))
            if i != elem:
                maps[i] = elem
        lines = []
        current_line = []
        while maps:
            if len(current_line) == 0:
                (k, v) = maps.popitem()
                current_line.append(k)
            else:
                k = current_line[-1]
                if k not in maps:
                    current_line = []
                    continue
                v = maps.pop(k)
            if v in current_line:
                lines.append(current_line)
                current_line = []
                continue
            current_line.append(v)
        for line in lines:
            for (i, e) in enumerate(line):
                args_positions[line[(i + 1) % len(line)]] = e
        permutation_blocks = [[new_permutation[cumulative_subranks[i] + j] for j in range(e)] for (i, e) in enumerate(subranks)]
        new_args = [new_args[i] for i in args_positions]
        new_permutation_blocks = [permutation_blocks[i] for i in args_positions]
        new_permutation2 = [j for i in new_permutation_blocks for j in i]
        return (_array_tensor_product(*new_args), Permutation(new_permutation2))

    @classmethod
    def _check_if_there_are_closed_cycles(cls, expr, permutation):
        if False:
            for i in range(10):
                print('nop')
        args = list(expr.args)
        subranks = expr.subranks
        cyclic_form = permutation.cyclic_form
        cumulative_subranks = [0] + list(accumulate(subranks))
        cyclic_min = [min(i) for i in cyclic_form]
        cyclic_max = [max(i) for i in cyclic_form]
        cyclic_keep = []
        for (i, cycle) in enumerate(cyclic_form):
            flag = True
            for j in range(len(cumulative_subranks) - 1):
                if cyclic_min[i] >= cumulative_subranks[j] and cyclic_max[i] < cumulative_subranks[j + 1]:
                    args[j] = _permute_dims(args[j], Permutation([[k - cumulative_subranks[j] for k in cyclic_form[i]]]))
                    flag = False
                    break
            if flag:
                cyclic_keep.append(cyclic_form[i])
        return (_array_tensor_product(*args), Permutation(cyclic_keep, size=permutation.size))

    def nest_permutation(self):
        if False:
            i = 10
            return i + 15
        '\n        DEPRECATED.\n        '
        ret = self._nest_permutation(self.expr, self.permutation)
        if ret is None:
            return self
        return ret

    @classmethod
    def _nest_permutation(cls, expr, permutation):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(expr, ArrayTensorProduct):
            return _permute_dims(*cls._check_if_there_are_closed_cycles(expr, permutation))
        elif isinstance(expr, ArrayContraction):
            cycles = permutation.cyclic_form
            newcycles = ArrayContraction._convert_outer_indices_to_inner_indices(expr, *cycles)
            newpermutation = Permutation(newcycles)
            new_contr_indices = [tuple((newpermutation(j) for j in i)) for i in expr.contraction_indices]
            return _array_contraction(PermuteDims(expr.expr, newpermutation), *new_contr_indices)
        elif isinstance(expr, ArrayAdd):
            return _array_add(*[PermuteDims(arg, permutation) for arg in expr.args])
        return None

    def as_explicit(self):
        if False:
            i = 10
            return i + 15
        expr = self.expr
        if hasattr(expr, 'as_explicit'):
            expr = expr.as_explicit()
        return permutedims(expr, self.permutation)

    @classmethod
    def _get_permutation_from_arguments(cls, permutation, index_order_old, index_order_new, dim):
        if False:
            i = 10
            return i + 15
        if permutation is None:
            if index_order_new is None or index_order_old is None:
                raise ValueError('Permutation not defined')
            return PermuteDims._get_permutation_from_index_orders(index_order_old, index_order_new, dim)
        else:
            if index_order_new is not None:
                raise ValueError('index_order_new cannot be defined with permutation')
            if index_order_old is not None:
                raise ValueError('index_order_old cannot be defined with permutation')
            return permutation

    @classmethod
    def _get_permutation_from_index_orders(cls, index_order_old, index_order_new, dim):
        if False:
            return 10
        if len(set(index_order_new)) != dim:
            raise ValueError('wrong number of indices in index_order_new')
        if len(set(index_order_old)) != dim:
            raise ValueError('wrong number of indices in index_order_old')
        if len(set.symmetric_difference(set(index_order_new), set(index_order_old))) > 0:
            raise ValueError('index_order_new and index_order_old must have the same indices')
        permutation = [index_order_old.index(i) for i in index_order_new]
        return permutation

class ArrayDiagonal(_CodegenArrayAbstract):
    """
    Class to represent the diagonal operator.

    Explanation
    ===========

    In a 2-dimensional array it returns the diagonal, this looks like the
    operation:

    `A_{ij} \\rightarrow A_{ii}`

    The diagonal over axes 1 and 2 (the second and third) of the tensor product
    of two 2-dimensional arrays `A \\otimes B` is

    `\\Big[ A_{ab} B_{cd} \\Big]_{abcd} \\rightarrow \\Big[ A_{ai} B_{id} \\Big]_{adi}`

    In this last example the array expression has been reduced from
    4-dimensional to 3-dimensional. Notice that no contraction has occurred,
    rather there is a new index `i` for the diagonal, contraction would have
    reduced the array to 2 dimensions.

    Notice that the diagonalized out dimensions are added as new dimensions at
    the end of the indices.
    """

    def __new__(cls, expr, *diagonal_indices, **kwargs):
        if False:
            i = 10
            return i + 15
        expr = _sympify(expr)
        diagonal_indices = [Tuple(*sorted(i)) for i in diagonal_indices]
        canonicalize = kwargs.get('canonicalize', False)
        shape = get_shape(expr)
        if shape is not None:
            cls._validate(expr, *diagonal_indices, **kwargs)
            (positions, shape) = cls._get_positions_shape(shape, diagonal_indices)
        else:
            positions = None
        if len(diagonal_indices) == 0:
            return expr
        obj = Basic.__new__(cls, expr, *diagonal_indices)
        obj._positions = positions
        obj._subranks = _get_subranks(expr)
        obj._shape = shape
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        if False:
            for i in range(10):
                print('nop')
        expr = self.expr
        diagonal_indices = self.diagonal_indices
        trivial_diags = [i for i in diagonal_indices if len(i) == 1]
        if len(trivial_diags) > 0:
            trivial_pos = {e[0]: i for (i, e) in enumerate(diagonal_indices) if len(e) == 1}
            diag_pos = {e: i for (i, e) in enumerate(diagonal_indices) if len(e) > 1}
            diagonal_indices_short = [i for i in diagonal_indices if len(i) > 1]
            rank1 = get_rank(self)
            rank2 = len(diagonal_indices)
            rank3 = rank1 - rank2
            inv_permutation = []
            counter1 = 0
            indices_down = ArrayDiagonal._push_indices_down(diagonal_indices_short, list(range(rank1)), get_rank(expr))
            for i in indices_down:
                if i in trivial_pos:
                    inv_permutation.append(rank3 + trivial_pos[i])
                elif isinstance(i, (Integer, int)):
                    inv_permutation.append(counter1)
                    counter1 += 1
                else:
                    inv_permutation.append(rank3 + diag_pos[i])
            permutation = _af_invert(inv_permutation)
            if len(diagonal_indices_short) > 0:
                return _permute_dims(_array_diagonal(expr, *diagonal_indices_short), permutation)
            else:
                return _permute_dims(expr, permutation)
        if isinstance(expr, ArrayAdd):
            return self._ArrayDiagonal_denest_ArrayAdd(expr, *diagonal_indices)
        if isinstance(expr, ArrayDiagonal):
            return self._ArrayDiagonal_denest_ArrayDiagonal(expr, *diagonal_indices)
        if isinstance(expr, PermuteDims):
            return self._ArrayDiagonal_denest_PermuteDims(expr, *diagonal_indices)
        if isinstance(expr, (ZeroArray, ZeroMatrix)):
            (positions, shape) = self._get_positions_shape(expr.shape, diagonal_indices)
            return ZeroArray(*shape)
        return self.func(expr, *diagonal_indices, canonicalize=False)

    @staticmethod
    def _validate(expr, *diagonal_indices, **kwargs):
        if False:
            while True:
                i = 10
        shape = get_shape(expr)
        for i in diagonal_indices:
            if any((j >= len(shape) for j in i)):
                raise ValueError('index is larger than expression shape')
            if len({shape[j] for j in i}) != 1:
                raise ValueError('diagonalizing indices of different dimensions')
            if not kwargs.get('allow_trivial_diags', False) and len(i) <= 1:
                raise ValueError('need at least two axes to diagonalize')
            if len(set(i)) != len(i):
                raise ValueError('axis index cannot be repeated')

    @staticmethod
    def _remove_trivial_dimensions(shape, *diagonal_indices):
        if False:
            for i in range(10):
                print('nop')
        return [tuple((j for j in i)) for i in diagonal_indices if shape[i[0]] != 1]

    @property
    def expr(self):
        if False:
            for i in range(10):
                print('nop')
        return self.args[0]

    @property
    def diagonal_indices(self):
        if False:
            print('Hello World!')
        return self.args[1:]

    @staticmethod
    def _flatten(expr, *outer_diagonal_indices):
        if False:
            print('Hello World!')
        inner_diagonal_indices = expr.diagonal_indices
        all_inner = [j for i in inner_diagonal_indices for j in i]
        all_inner.sort()
        total_rank = _get_subrank(expr)
        inner_rank = len(all_inner)
        outer_rank = total_rank - inner_rank
        shifts = [0 for i in range(outer_rank)]
        counter = 0
        pointer = 0
        for i in range(outer_rank):
            while pointer < inner_rank and counter >= all_inner[pointer]:
                counter += 1
                pointer += 1
            shifts[i] += pointer
            counter += 1
        outer_diagonal_indices = tuple((tuple((shifts[j] + j for j in i)) for i in outer_diagonal_indices))
        diagonal_indices = inner_diagonal_indices + outer_diagonal_indices
        return _array_diagonal(expr.expr, *diagonal_indices)

    @classmethod
    def _ArrayDiagonal_denest_ArrayAdd(cls, expr, *diagonal_indices):
        if False:
            i = 10
            return i + 15
        return _array_add(*[_array_diagonal(arg, *diagonal_indices) for arg in expr.args])

    @classmethod
    def _ArrayDiagonal_denest_ArrayDiagonal(cls, expr, *diagonal_indices):
        if False:
            for i in range(10):
                print('nop')
        return cls._flatten(expr, *diagonal_indices)

    @classmethod
    def _ArrayDiagonal_denest_PermuteDims(cls, expr: PermuteDims, *diagonal_indices):
        if False:
            return 10
        back_diagonal_indices = [[expr.permutation(j) for j in i] for i in diagonal_indices]
        nondiag = [i for i in range(get_rank(expr)) if not any((i in j for j in diagonal_indices))]
        back_nondiag = [expr.permutation(i) for i in nondiag]
        remap = {e: i for (i, e) in enumerate(sorted(back_nondiag))}
        new_permutation1 = [remap[i] for i in back_nondiag]
        shift = len(new_permutation1)
        diag_block_perm = [i + shift for i in range(len(back_diagonal_indices))]
        new_permutation = new_permutation1 + diag_block_perm
        return _permute_dims(_array_diagonal(expr.expr, *back_diagonal_indices), new_permutation)

    def _push_indices_down_nonstatic(self, indices):
        if False:
            print('Hello World!')
        transform = lambda x: self._positions[x] if x < len(self._positions) else None
        return _apply_recursively_over_nested_lists(transform, indices)

    def _push_indices_up_nonstatic(self, indices):
        if False:
            i = 10
            return i + 15

        def transform(x):
            if False:
                i = 10
                return i + 15
            for (i, e) in enumerate(self._positions):
                if isinstance(e, int) and x == e or (isinstance(e, tuple) and x in e):
                    return i
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _push_indices_down(cls, diagonal_indices, indices, rank):
        if False:
            while True:
                i = 10
        (positions, shape) = cls._get_positions_shape(range(rank), diagonal_indices)
        transform = lambda x: positions[x] if x < len(positions) else None
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _push_indices_up(cls, diagonal_indices, indices, rank):
        if False:
            i = 10
            return i + 15
        (positions, shape) = cls._get_positions_shape(range(rank), diagonal_indices)

        def transform(x):
            if False:
                for i in range(10):
                    print('nop')
            for (i, e) in enumerate(positions):
                if isinstance(e, int) and x == e or (isinstance(e, (tuple, Tuple)) and x in e):
                    return i
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _get_positions_shape(cls, shape, diagonal_indices):
        if False:
            print('Hello World!')
        data1 = tuple(((i, shp) for (i, shp) in enumerate(shape) if not any((i in j for j in diagonal_indices))))
        (pos1, shp1) = zip(*data1) if data1 else ((), ())
        data2 = tuple(((i, shape[i[0]]) for i in diagonal_indices))
        (pos2, shp2) = zip(*data2) if data2 else ((), ())
        positions = pos1 + pos2
        shape = shp1 + shp2
        return (positions, shape)

    def as_explicit(self):
        if False:
            i = 10
            return i + 15
        expr = self.expr
        if hasattr(expr, 'as_explicit'):
            expr = expr.as_explicit()
        return tensordiagonal(expr, *self.diagonal_indices)

class ArrayElementwiseApplyFunc(_CodegenArrayAbstract):

    def __new__(cls, function, element):
        if False:
            return 10
        if not isinstance(function, Lambda):
            d = Dummy('d')
            function = Lambda(d, function(d))
        obj = _CodegenArrayAbstract.__new__(cls, function, element)
        obj._subranks = _get_subranks(element)
        return obj

    @property
    def function(self):
        if False:
            return 10
        return self.args[0]

    @property
    def expr(self):
        if False:
            while True:
                i = 10
        return self.args[1]

    @property
    def shape(self):
        if False:
            return 10
        return self.expr.shape

    def _get_function_fdiff(self):
        if False:
            while True:
                i = 10
        d = Dummy('d')
        function = self.function(d)
        fdiff = function.diff(d)
        if isinstance(fdiff, Function):
            fdiff = type(fdiff)
        else:
            fdiff = Lambda(d, fdiff)
        return fdiff

    def as_explicit(self):
        if False:
            print('Hello World!')
        expr = self.expr
        if hasattr(expr, 'as_explicit'):
            expr = expr.as_explicit()
        return expr.applyfunc(self.function)

class ArrayContraction(_CodegenArrayAbstract):
    """
    This class is meant to represent contractions of arrays in a form easily
    processable by the code printers.
    """

    def __new__(cls, expr, *contraction_indices, **kwargs):
        if False:
            while True:
                i = 10
        contraction_indices = _sort_contraction_indices(contraction_indices)
        expr = _sympify(expr)
        canonicalize = kwargs.get('canonicalize', False)
        obj = Basic.__new__(cls, expr, *contraction_indices)
        obj._subranks = _get_subranks(expr)
        obj._mapping = _get_mapping_from_subranks(obj._subranks)
        free_indices_to_position = {i: i for i in range(sum(obj._subranks)) if all((i not in cind for cind in contraction_indices))}
        obj._free_indices_to_position = free_indices_to_position
        shape = get_shape(expr)
        cls._validate(expr, *contraction_indices)
        if shape:
            shape = tuple((shp for (i, shp) in enumerate(shape) if not any((i in j for j in contraction_indices))))
        obj._shape = shape
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        if False:
            return 10
        expr = self.expr
        contraction_indices = self.contraction_indices
        if len(contraction_indices) == 0:
            return expr
        if isinstance(expr, ArrayContraction):
            return self._ArrayContraction_denest_ArrayContraction(expr, *contraction_indices)
        if isinstance(expr, (ZeroArray, ZeroMatrix)):
            return self._ArrayContraction_denest_ZeroArray(expr, *contraction_indices)
        if isinstance(expr, PermuteDims):
            return self._ArrayContraction_denest_PermuteDims(expr, *contraction_indices)
        if isinstance(expr, ArrayTensorProduct):
            (expr, contraction_indices) = self._sort_fully_contracted_args(expr, contraction_indices)
            (expr, contraction_indices) = self._lower_contraction_to_addends(expr, contraction_indices)
            if len(contraction_indices) == 0:
                return expr
        if isinstance(expr, ArrayDiagonal):
            return self._ArrayContraction_denest_ArrayDiagonal(expr, *contraction_indices)
        if isinstance(expr, ArrayAdd):
            return self._ArrayContraction_denest_ArrayAdd(expr, *contraction_indices)
        contraction_indices = [i for i in contraction_indices if len(i) > 1 or get_shape(expr)[i[0]] != 1]
        if len(contraction_indices) == 0:
            return expr
        return self.func(expr, *contraction_indices, canonicalize=False)

    def __mul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if other == 1:
            return self
        else:
            raise NotImplementedError('Product of N-dim arrays is not uniquely defined. Use another method.')

    def __rmul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if other == 1:
            return self
        else:
            raise NotImplementedError('Product of N-dim arrays is not uniquely defined. Use another method.')

    @staticmethod
    def _validate(expr, *contraction_indices):
        if False:
            print('Hello World!')
        shape = get_shape(expr)
        if shape is None:
            return
        for i in contraction_indices:
            if len({shape[j] for j in i if shape[j] != -1}) != 1:
                raise ValueError('contracting indices of different dimensions')

    @classmethod
    def _push_indices_down(cls, contraction_indices, indices):
        if False:
            for i in range(10):
                print('nop')
        flattened_contraction_indices = [j for i in contraction_indices for j in i]
        flattened_contraction_indices.sort()
        transform = _build_push_indices_down_func_transformation(flattened_contraction_indices)
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _push_indices_up(cls, contraction_indices, indices):
        if False:
            while True:
                i = 10
        flattened_contraction_indices = [j for i in contraction_indices for j in i]
        flattened_contraction_indices.sort()
        transform = _build_push_indices_up_func_transformation(flattened_contraction_indices)
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _lower_contraction_to_addends(cls, expr, contraction_indices):
        if False:
            i = 10
            return i + 15
        if isinstance(expr, ArrayAdd):
            raise NotImplementedError()
        if not isinstance(expr, ArrayTensorProduct):
            return (expr, contraction_indices)
        subranks = expr.subranks
        cumranks = list(accumulate([0] + subranks))
        contraction_indices_remaining = []
        contraction_indices_args = [[] for i in expr.args]
        backshift = set()
        for (i, contraction_group) in enumerate(contraction_indices):
            for j in range(len(expr.args)):
                if not isinstance(expr.args[j], ArrayAdd):
                    continue
                if all((cumranks[j] <= k < cumranks[j + 1] for k in contraction_group)):
                    contraction_indices_args[j].append([k - cumranks[j] for k in contraction_group])
                    backshift.update(contraction_group)
                    break
            else:
                contraction_indices_remaining.append(contraction_group)
        if len(contraction_indices_remaining) == len(contraction_indices):
            return (expr, contraction_indices)
        total_rank = get_rank(expr)
        shifts = list(accumulate([1 if i in backshift else 0 for i in range(total_rank)]))
        contraction_indices_remaining = [Tuple.fromiter((j - shifts[j] for j in i)) for i in contraction_indices_remaining]
        ret = _array_tensor_product(*[_array_contraction(arg, *contr) for (arg, contr) in zip(expr.args, contraction_indices_args)])
        return (ret, contraction_indices_remaining)

    def split_multiple_contractions(self):
        if False:
            i = 10
            return i + 15
        '\n        Recognize multiple contractions and attempt at rewriting them as paired-contractions.\n\n        This allows some contractions involving more than two indices to be\n        rewritten as multiple contractions involving two indices, thus allowing\n        the expression to be rewritten as a matrix multiplication line.\n\n        Examples:\n\n        * `A_ij b_j0 C_jk` ===> `A*DiagMatrix(b)*C`\n\n        Care for:\n        - matrix being diagonalized (i.e. `A_ii`)\n        - vectors being diagonalized (i.e. `a_i0`)\n\n        Multiple contractions can be split into matrix multiplications if\n        not more than two arguments are non-diagonals or non-vectors.\n        Vectors get diagonalized while diagonal matrices remain diagonal.\n        The non-diagonal matrices can be at the beginning or at the end\n        of the final matrix multiplication line.\n        '
        editor = _EditArrayContraction(self)
        contraction_indices = self.contraction_indices
        onearray_insert = []
        for (indl, links) in enumerate(contraction_indices):
            if len(links) <= 2:
                continue
            positions = editor.get_mapping_for_index(indl)
            current_dimension = self.expr.shape[links[0]]
            not_vectors = []
            vectors = []
            for (arg_ind, rel_ind) in positions:
                arg = editor.args_with_ind[arg_ind]
                mat = arg.element
                (abs_arg_start, abs_arg_end) = editor.get_absolute_range(arg)
                other_arg_pos = 1 - rel_ind
                other_arg_abs = abs_arg_start + other_arg_pos
                if 1 not in mat.shape or ((current_dimension == 1) is True and mat.shape != (1, 1)) or any((other_arg_abs in l for (li, l) in enumerate(contraction_indices) if li != indl)):
                    not_vectors.append((arg, rel_ind))
                else:
                    vectors.append((arg, rel_ind))
            if len(not_vectors) > 2:
                continue
            for (v, rel_ind) in vectors:
                v.element = diagonalize_vector(v.element)
            vectors_to_loop = not_vectors[:1] + vectors + not_vectors[1:]
            (first_not_vector, rel_ind) = vectors_to_loop[0]
            new_index = first_not_vector.indices[rel_ind]
            for (v, rel_ind) in vectors_to_loop[1:-1]:
                v.indices[rel_ind] = new_index
                new_index = editor.get_new_contraction_index()
                assert v.indices.index(None) == 1 - rel_ind
                v.indices[v.indices.index(None)] = new_index
                onearray_insert.append(v)
            (last_vec, rel_ind) = vectors_to_loop[-1]
            last_vec.indices[rel_ind] = new_index
        for v in onearray_insert:
            editor.insert_after(v, _ArgE(OneArray(1), [None]))
        return editor.to_array_contraction()

    def flatten_contraction_of_diagonal(self):
        if False:
            print('Hello World!')
        if not isinstance(self.expr, ArrayDiagonal):
            return self
        contraction_down = self.expr._push_indices_down(self.expr.diagonal_indices, self.contraction_indices)
        new_contraction_indices = []
        diagonal_indices = self.expr.diagonal_indices[:]
        for i in contraction_down:
            contraction_group = list(i)
            for j in i:
                diagonal_with = [k for k in diagonal_indices if j in k]
                contraction_group.extend([l for k in diagonal_with for l in k])
                diagonal_indices = [k for k in diagonal_indices if k not in diagonal_with]
            new_contraction_indices.append(sorted(set(contraction_group)))
        new_contraction_indices = ArrayDiagonal._push_indices_up(diagonal_indices, new_contraction_indices)
        return _array_contraction(_array_diagonal(self.expr.expr, *diagonal_indices), *new_contraction_indices)

    @staticmethod
    def _get_free_indices_to_position_map(free_indices, contraction_indices):
        if False:
            return 10
        free_indices_to_position = {}
        flattened_contraction_indices = [j for i in contraction_indices for j in i]
        counter = 0
        for ind in free_indices:
            while counter in flattened_contraction_indices:
                counter += 1
            free_indices_to_position[ind] = counter
            counter += 1
        return free_indices_to_position

    @staticmethod
    def _get_index_shifts(expr):
        if False:
            while True:
                i = 10
        '\n        Get the mapping of indices at the positions before the contraction\n        occurs.\n\n        Examples\n        ========\n\n        >>> from sympy.tensor.array import tensorproduct, tensorcontraction\n        >>> from sympy import MatrixSymbol\n        >>> M = MatrixSymbol("M", 3, 3)\n        >>> N = MatrixSymbol("N", 3, 3)\n        >>> cg = tensorcontraction(tensorproduct(M, N), [1, 2])\n        >>> cg._get_index_shifts(cg)\n        [0, 2]\n\n        Indeed, ``cg`` after the contraction has two dimensions, 0 and 1. They\n        need to be shifted by 0 and 2 to get the corresponding positions before\n        the contraction (that is, 0 and 3).\n        '
        inner_contraction_indices = expr.contraction_indices
        all_inner = [j for i in inner_contraction_indices for j in i]
        all_inner.sort()
        total_rank = _get_subrank(expr)
        inner_rank = len(all_inner)
        outer_rank = total_rank - inner_rank
        shifts = [0 for i in range(outer_rank)]
        counter = 0
        pointer = 0
        for i in range(outer_rank):
            while pointer < inner_rank and counter >= all_inner[pointer]:
                counter += 1
                pointer += 1
            shifts[i] += pointer
            counter += 1
        return shifts

    @staticmethod
    def _convert_outer_indices_to_inner_indices(expr, *outer_contraction_indices):
        if False:
            for i in range(10):
                print('nop')
        shifts = ArrayContraction._get_index_shifts(expr)
        outer_contraction_indices = tuple((tuple((shifts[j] + j for j in i)) for i in outer_contraction_indices))
        return outer_contraction_indices

    @staticmethod
    def _flatten(expr, *outer_contraction_indices):
        if False:
            while True:
                i = 10
        inner_contraction_indices = expr.contraction_indices
        outer_contraction_indices = ArrayContraction._convert_outer_indices_to_inner_indices(expr, *outer_contraction_indices)
        contraction_indices = inner_contraction_indices + outer_contraction_indices
        return _array_contraction(expr.expr, *contraction_indices)

    @classmethod
    def _ArrayContraction_denest_ArrayContraction(cls, expr, *contraction_indices):
        if False:
            print('Hello World!')
        return cls._flatten(expr, *contraction_indices)

    @classmethod
    def _ArrayContraction_denest_ZeroArray(cls, expr, *contraction_indices):
        if False:
            return 10
        contraction_indices_flat = [j for i in contraction_indices for j in i]
        shape = [e for (i, e) in enumerate(expr.shape) if i not in contraction_indices_flat]
        return ZeroArray(*shape)

    @classmethod
    def _ArrayContraction_denest_ArrayAdd(cls, expr, *contraction_indices):
        if False:
            while True:
                i = 10
        return _array_add(*[_array_contraction(i, *contraction_indices) for i in expr.args])

    @classmethod
    def _ArrayContraction_denest_PermuteDims(cls, expr, *contraction_indices):
        if False:
            print('Hello World!')
        permutation = expr.permutation
        plist = permutation.array_form
        new_contraction_indices = [tuple((permutation(j) for j in i)) for i in contraction_indices]
        new_plist = [i for i in plist if not any((i in j for j in new_contraction_indices))]
        new_plist = cls._push_indices_up(new_contraction_indices, new_plist)
        return _permute_dims(_array_contraction(expr.expr, *new_contraction_indices), Permutation(new_plist))

    @classmethod
    def _ArrayContraction_denest_ArrayDiagonal(cls, expr: 'ArrayDiagonal', *contraction_indices):
        if False:
            for i in range(10):
                print('nop')
        diagonal_indices = list(expr.diagonal_indices)
        down_contraction_indices = expr._push_indices_down(expr.diagonal_indices, contraction_indices, get_rank(expr.expr))
        down_contraction_indices = [[k for j in i for k in (j if isinstance(j, (tuple, Tuple)) else [j])] for i in down_contraction_indices]
        new_contraction_indices = []
        for contr_indgrp in down_contraction_indices:
            ind = contr_indgrp[:]
            for (j, diag_indgrp) in enumerate(diagonal_indices):
                if diag_indgrp is None:
                    continue
                if any((i in diag_indgrp for i in contr_indgrp)):
                    ind.extend(diag_indgrp)
                    diagonal_indices[j] = None
            new_contraction_indices.append(sorted(set(ind)))
        new_diagonal_indices_down = [i for i in diagonal_indices if i is not None]
        new_diagonal_indices = ArrayContraction._push_indices_up(new_contraction_indices, new_diagonal_indices_down)
        return _array_diagonal(_array_contraction(expr.expr, *new_contraction_indices), *new_diagonal_indices)

    @classmethod
    def _sort_fully_contracted_args(cls, expr, contraction_indices):
        if False:
            i = 10
            return i + 15
        if expr.shape is None:
            return (expr, contraction_indices)
        cumul = list(accumulate([0] + expr.subranks))
        index_blocks = [list(range(cumul[i], cumul[i + 1])) for i in range(len(expr.args))]
        contraction_indices_flat = {j for i in contraction_indices for j in i}
        fully_contracted = [all((j in contraction_indices_flat for j in range(cumul[i], cumul[i + 1]))) for (i, arg) in enumerate(expr.args)]
        new_pos = sorted(range(len(expr.args)), key=lambda x: (0, default_sort_key(expr.args[x])) if fully_contracted[x] else (1,))
        new_args = [expr.args[i] for i in new_pos]
        new_index_blocks_flat = [j for i in new_pos for j in index_blocks[i]]
        index_permutation_array_form = _af_invert(new_index_blocks_flat)
        new_contraction_indices = [tuple((index_permutation_array_form[j] for j in i)) for i in contraction_indices]
        new_contraction_indices = _sort_contraction_indices(new_contraction_indices)
        return (_array_tensor_product(*new_args), new_contraction_indices)

    def _get_contraction_tuples(self):
        if False:
            i = 10
            return i + 15
        '\n        Return tuples containing the argument index and position within the\n        argument of the index position.\n\n        Examples\n        ========\n\n        >>> from sympy import MatrixSymbol\n        >>> from sympy.abc import N\n        >>> from sympy.tensor.array import tensorproduct, tensorcontraction\n        >>> A = MatrixSymbol("A", N, N)\n        >>> B = MatrixSymbol("B", N, N)\n\n        >>> cg = tensorcontraction(tensorproduct(A, B), (1, 2))\n        >>> cg._get_contraction_tuples()\n        [[(0, 1), (1, 0)]]\n\n        Notes\n        =====\n\n        Here the contraction pair `(1, 2)` meaning that the 2nd and 3rd indices\n        of the tensor product `A\\otimes B` are contracted, has been transformed\n        into `(0, 1)` and `(1, 0)`, identifying the same indices in a different\n        notation. `(0, 1)` is the second index (1) of the first argument (i.e.\n                0 or `A`). `(1, 0)` is the first index (i.e. 0) of the second\n        argument (i.e. 1 or `B`).\n        '
        mapping = self._mapping
        return [[mapping[j] for j in i] for i in self.contraction_indices]

    @staticmethod
    def _contraction_tuples_to_contraction_indices(expr, contraction_tuples):
        if False:
            for i in range(10):
                print('nop')
        ranks = expr.subranks
        cumulative_ranks = [0] + list(accumulate(ranks))
        return [tuple((cumulative_ranks[j] + k for (j, k) in i)) for i in contraction_tuples]

    @property
    def free_indices(self):
        if False:
            print('Hello World!')
        return self._free_indices[:]

    @property
    def free_indices_to_position(self):
        if False:
            return 10
        return dict(self._free_indices_to_position)

    @property
    def expr(self):
        if False:
            i = 10
            return i + 15
        return self.args[0]

    @property
    def contraction_indices(self):
        if False:
            for i in range(10):
                print('nop')
        return self.args[1:]

    def _contraction_indices_to_components(self):
        if False:
            print('Hello World!')
        expr = self.expr
        if not isinstance(expr, ArrayTensorProduct):
            raise NotImplementedError('only for contractions of tensor products')
        ranks = expr.subranks
        mapping = {}
        counter = 0
        for (i, rank) in enumerate(ranks):
            for j in range(rank):
                mapping[counter] = (i, j)
                counter += 1
        return mapping

    def sort_args_by_name(self):
        if False:
            return 10
        '\n        Sort arguments in the tensor product so that their order is lexicographical.\n\n        Examples\n        ========\n\n        >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array\n        >>> from sympy import MatrixSymbol\n        >>> from sympy.abc import N\n        >>> A = MatrixSymbol("A", N, N)\n        >>> B = MatrixSymbol("B", N, N)\n        >>> C = MatrixSymbol("C", N, N)\n        >>> D = MatrixSymbol("D", N, N)\n\n        >>> cg = convert_matrix_to_array(C*D*A*B)\n        >>> cg\n        ArrayContraction(ArrayTensorProduct(A, D, C, B), (0, 3), (1, 6), (2, 5))\n        >>> cg.sort_args_by_name()\n        ArrayContraction(ArrayTensorProduct(A, D, B, C), (0, 3), (1, 4), (2, 7))\n        '
        expr = self.expr
        if not isinstance(expr, ArrayTensorProduct):
            return self
        args = expr.args
        sorted_data = sorted(enumerate(args), key=lambda x: default_sort_key(x[1]))
        (pos_sorted, args_sorted) = zip(*sorted_data)
        reordering_map = {i: pos_sorted.index(i) for (i, arg) in enumerate(args)}
        contraction_tuples = self._get_contraction_tuples()
        contraction_tuples = [[(reordering_map[j], k) for (j, k) in i] for i in contraction_tuples]
        c_tp = _array_tensor_product(*args_sorted)
        new_contr_indices = self._contraction_tuples_to_contraction_indices(c_tp, contraction_tuples)
        return _array_contraction(c_tp, *new_contr_indices)

    def _get_contraction_links(self):
        if False:
            return 10
        '\n        Returns a dictionary of links between arguments in the tensor product\n        being contracted.\n\n        See the example for an explanation of the values.\n\n        Examples\n        ========\n\n        >>> from sympy import MatrixSymbol\n        >>> from sympy.abc import N\n        >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array\n        >>> A = MatrixSymbol("A", N, N)\n        >>> B = MatrixSymbol("B", N, N)\n        >>> C = MatrixSymbol("C", N, N)\n        >>> D = MatrixSymbol("D", N, N)\n\n        Matrix multiplications are pairwise contractions between neighboring\n        matrices:\n\n        `A_{ij} B_{jk} C_{kl} D_{lm}`\n\n        >>> cg = convert_matrix_to_array(A*B*C*D)\n        >>> cg\n        ArrayContraction(ArrayTensorProduct(B, C, A, D), (0, 5), (1, 2), (3, 6))\n\n        >>> cg._get_contraction_links()\n        {0: {0: (2, 1), 1: (1, 0)}, 1: {0: (0, 1), 1: (3, 0)}, 2: {1: (0, 0)}, 3: {0: (1, 1)}}\n\n        This dictionary is interpreted as follows: argument in position 0 (i.e.\n        matrix `A`) has its second index (i.e. 1) contracted to `(1, 0)`, that\n        is argument in position 1 (matrix `B`) on the first index slot of `B`,\n        this is the contraction provided by the index `j` from `A`.\n\n        The argument in position 1 (that is, matrix `B`) has two contractions,\n        the ones provided by the indices `j` and `k`, respectively the first\n        and second indices (0 and 1 in the sub-dict).  The link `(0, 1)` and\n        `(2, 0)` respectively. `(0, 1)` is the index slot 1 (the 2nd) of\n        argument in position 0 (that is, `A_{\\ldot j}`), and so on.\n        '
        (args, dlinks) = _get_contraction_links([self], self.subranks, *self.contraction_indices)
        return dlinks

    def as_explicit(self):
        if False:
            print('Hello World!')
        expr = self.expr
        if hasattr(expr, 'as_explicit'):
            expr = expr.as_explicit()
        return tensorcontraction(expr, *self.contraction_indices)

class Reshape(_CodegenArrayAbstract):
    """
    Reshape the dimensions of an array expression.

    Examples
    ========

    >>> from sympy.tensor.array.expressions import ArraySymbol, Reshape
    >>> A = ArraySymbol("A", (6,))
    >>> A.shape
    (6,)
    >>> Reshape(A, (3, 2)).shape
    (3, 2)

    Check the component-explicit forms:

    >>> A.as_explicit()
    [A[0], A[1], A[2], A[3], A[4], A[5]]
    >>> Reshape(A, (3, 2)).as_explicit()
    [[A[0], A[1]], [A[2], A[3]], [A[4], A[5]]]

    """

    def __new__(cls, expr, shape):
        if False:
            while True:
                i = 10
        expr = _sympify(expr)
        if not isinstance(shape, Tuple):
            shape = Tuple(*shape)
        if Equality(Mul.fromiter(expr.shape), Mul.fromiter(shape)) == False:
            raise ValueError('shape mismatch')
        obj = Expr.__new__(cls, expr, shape)
        obj._shape = tuple(shape)
        obj._expr = expr
        return obj

    @property
    def shape(self):
        if False:
            i = 10
            return i + 15
        return self._shape

    @property
    def expr(self):
        if False:
            while True:
                i = 10
        return self._expr

    def doit(self, *args, **kwargs):
        if False:
            return 10
        if kwargs.get('deep', True):
            expr = self.expr.doit(*args, **kwargs)
        else:
            expr = self.expr
        if isinstance(expr, (MatrixCommon, NDimArray)):
            return expr.reshape(*self.shape)
        return Reshape(expr, self.shape)

    def as_explicit(self):
        if False:
            while True:
                i = 10
        ee = self.expr
        if hasattr(ee, 'as_explicit'):
            ee = ee.as_explicit()
        if isinstance(ee, MatrixCommon):
            from sympy import Array
            ee = Array(ee)
        elif isinstance(ee, MatrixExpr):
            return self
        return ee.reshape(*self.shape)

class _ArgE:
    """
    The ``_ArgE`` object contains references to the array expression
    (``.element``) and a list containing the information about index
    contractions (``.indices``).

    Index contractions are numbered and contracted indices show the number of
    the contraction. Uncontracted indices have ``None`` value.

    For example:
    ``_ArgE(M, [None, 3])``
    This object means that expression ``M`` is part of an array contraction
    and has two indices, the first is not contracted (value ``None``),
    the second index is contracted to the 4th (i.e. number ``3``) group of the
    array contraction object.
    """
    indices: List[Optional[int]]

    def __init__(self, element, indices: Optional[List[Optional[int]]]=None):
        if False:
            while True:
                i = 10
        self.element = element
        if indices is None:
            self.indices = [None for i in range(get_rank(element))]
        else:
            self.indices = indices

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '_ArgE(%s, %s)' % (self.element, self.indices)
    __repr__ = __str__

class _IndPos:
    """
    Index position, requiring two integers in the constructor:

    - arg: the position of the argument in the tensor product,
    - rel: the relative position of the index inside the argument.
    """

    def __init__(self, arg: int, rel: int):
        if False:
            for i in range(10):
                print('nop')
        self.arg = arg
        self.rel = rel

    def __str__(self):
        if False:
            return 10
        return '_IndPos(%i, %i)' % (self.arg, self.rel)
    __repr__ = __str__

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        yield from [self.arg, self.rel]

class _EditArrayContraction:
    """
    Utility class to help manipulate array contraction objects.

    This class takes as input an ``ArrayContraction`` object and turns it into
    an editable object.

    The field ``args_with_ind`` of this class is a list of ``_ArgE`` objects
    which can be used to easily edit the contraction structure of the
    expression.

    Once editing is finished, the ``ArrayContraction`` object may be recreated
    by calling the ``.to_array_contraction()`` method.
    """

    def __init__(self, base_array: typing.Union[ArrayContraction, ArrayDiagonal, ArrayTensorProduct]):
        if False:
            i = 10
            return i + 15
        expr: Basic
        diagonalized: tTuple[tTuple[int, ...], ...]
        contraction_indices: List[tTuple[int]]
        if isinstance(base_array, ArrayContraction):
            mapping = _get_mapping_from_subranks(base_array.subranks)
            expr = base_array.expr
            contraction_indices = base_array.contraction_indices
            diagonalized = ()
        elif isinstance(base_array, ArrayDiagonal):
            if isinstance(base_array.expr, ArrayContraction):
                mapping = _get_mapping_from_subranks(base_array.expr.subranks)
                expr = base_array.expr.expr
                diagonalized = ArrayContraction._push_indices_down(base_array.expr.contraction_indices, base_array.diagonal_indices)
                contraction_indices = base_array.expr.contraction_indices
            elif isinstance(base_array.expr, ArrayTensorProduct):
                mapping = {}
                expr = base_array.expr
                diagonalized = base_array.diagonal_indices
                contraction_indices = []
            else:
                mapping = {}
                expr = base_array.expr
                diagonalized = base_array.diagonal_indices
                contraction_indices = []
        elif isinstance(base_array, ArrayTensorProduct):
            expr = base_array
            contraction_indices = []
            diagonalized = ()
        else:
            raise NotImplementedError()
        if isinstance(expr, ArrayTensorProduct):
            args = list(expr.args)
        else:
            args = [expr]
        args_with_ind: List[_ArgE] = [_ArgE(arg) for arg in args]
        for (i, contraction_tuple) in enumerate(contraction_indices):
            for j in contraction_tuple:
                (arg_pos, rel_pos) = mapping[j]
                args_with_ind[arg_pos].indices[rel_pos] = i
        self.args_with_ind: List[_ArgE] = args_with_ind
        self.number_of_contraction_indices: int = len(contraction_indices)
        self._track_permutation: Optional[List[List[int]]] = None
        mapping = _get_mapping_from_subranks(base_array.subranks)
        for (i, e) in enumerate(diagonalized):
            for j in e:
                (arg_pos, rel_pos) = mapping[j]
                self.args_with_ind[arg_pos].indices[rel_pos] = -1 - i

    def insert_after(self, arg: _ArgE, new_arg: _ArgE):
        if False:
            for i in range(10):
                print('nop')
        pos = self.args_with_ind.index(arg)
        self.args_with_ind.insert(pos + 1, new_arg)

    def get_new_contraction_index(self):
        if False:
            for i in range(10):
                print('nop')
        self.number_of_contraction_indices += 1
        return self.number_of_contraction_indices - 1

    def refresh_indices(self):
        if False:
            print('Hello World!')
        updates = {}
        for arg_with_ind in self.args_with_ind:
            updates.update({i: -1 for i in arg_with_ind.indices if i is not None})
        for (i, e) in enumerate(sorted(updates)):
            updates[e] = i
        self.number_of_contraction_indices = len(updates)
        for arg_with_ind in self.args_with_ind:
            arg_with_ind.indices = [updates.get(i, None) for i in arg_with_ind.indices]

    def merge_scalars(self):
        if False:
            print('Hello World!')
        scalars = []
        for arg_with_ind in self.args_with_ind:
            if len(arg_with_ind.indices) == 0:
                scalars.append(arg_with_ind)
        for i in scalars:
            self.args_with_ind.remove(i)
        scalar = Mul.fromiter([i.element for i in scalars])
        if len(self.args_with_ind) == 0:
            self.args_with_ind.append(_ArgE(scalar))
        else:
            from sympy.tensor.array.expressions.from_array_to_matrix import _a2m_tensor_product
            self.args_with_ind[0].element = _a2m_tensor_product(scalar, self.args_with_ind[0].element)

    def to_array_contraction(self):
        if False:
            while True:
                i = 10
        counter = 0
        diag_indices = defaultdict(list)
        count_index_freq = Counter()
        for arg_with_ind in self.args_with_ind:
            count_index_freq.update(Counter(arg_with_ind.indices))
        free_index_count = count_index_freq[None]
        inv_perm1 = []
        inv_perm2 = []
        done = set()
        counter4 = 0
        for arg_with_ind in self.args_with_ind:
            counter2 = 0
            for i in arg_with_ind.indices:
                if i is None:
                    inv_perm1.append(counter4)
                    counter2 += 1
                    counter4 += 1
                    continue
                if i >= 0:
                    continue
                diag_indices[-1 - i].append(counter + counter2)
                if count_index_freq[i] == 1 and i not in done:
                    inv_perm1.append(free_index_count - 1 - i)
                    done.add(i)
                elif i not in done:
                    inv_perm2.append(free_index_count - 1 - i)
                    done.add(i)
                counter2 += 1
            arg_with_ind.indices = [i if i is not None and i >= 0 else None for i in arg_with_ind.indices]
            counter += len([i for i in arg_with_ind.indices if i is None or i < 0])
        inverse_permutation = inv_perm1 + inv_perm2
        permutation = _af_invert(inverse_permutation)
        diag_indices_filtered = [tuple(v) for v in diag_indices.values() if len(v) > 1]
        self.merge_scalars()
        self.refresh_indices()
        args = [arg.element for arg in self.args_with_ind]
        contraction_indices = self.get_contraction_indices()
        expr = _array_contraction(_array_tensor_product(*args), *contraction_indices)
        expr2 = _array_diagonal(expr, *diag_indices_filtered)
        if self._track_permutation is not None:
            permutation2 = _af_invert([j for i in self._track_permutation for j in i])
            expr2 = _permute_dims(expr2, permutation2)
        expr3 = _permute_dims(expr2, permutation)
        return expr3

    def get_contraction_indices(self) -> List[List[int]]:
        if False:
            while True:
                i = 10
        contraction_indices: List[List[int]] = [[] for i in range(self.number_of_contraction_indices)]
        current_position: int = 0
        for (i, arg_with_ind) in enumerate(self.args_with_ind):
            for j in arg_with_ind.indices:
                if j is not None:
                    contraction_indices[j].append(current_position)
                current_position += 1
        return contraction_indices

    def get_mapping_for_index(self, ind) -> List[_IndPos]:
        if False:
            for i in range(10):
                print('nop')
        if ind >= self.number_of_contraction_indices:
            raise ValueError('index value exceeding the index range')
        positions: List[_IndPos] = []
        for (i, arg_with_ind) in enumerate(self.args_with_ind):
            for (j, arg_ind) in enumerate(arg_with_ind.indices):
                if ind == arg_ind:
                    positions.append(_IndPos(i, j))
        return positions

    def get_contraction_indices_to_ind_rel_pos(self) -> List[List[_IndPos]]:
        if False:
            i = 10
            return i + 15
        contraction_indices: List[List[_IndPos]] = [[] for i in range(self.number_of_contraction_indices)]
        for (i, arg_with_ind) in enumerate(self.args_with_ind):
            for (j, ind) in enumerate(arg_with_ind.indices):
                if ind is not None:
                    contraction_indices[ind].append(_IndPos(i, j))
        return contraction_indices

    def count_args_with_index(self, index: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Count the number of arguments that have the given index.\n        '
        counter: int = 0
        for arg_with_ind in self.args_with_ind:
            if index in arg_with_ind.indices:
                counter += 1
        return counter

    def get_args_with_index(self, index: int) -> List[_ArgE]:
        if False:
            while True:
                i = 10
        '\n        Get a list of arguments having the given index.\n        '
        ret: List[_ArgE] = [i for i in self.args_with_ind if index in i.indices]
        return ret

    @property
    def number_of_diagonal_indices(self):
        if False:
            print('Hello World!')
        data = set()
        for arg in self.args_with_ind:
            data.update({i for i in arg.indices if i is not None and i < 0})
        return len(data)

    def track_permutation_start(self):
        if False:
            for i in range(10):
                print('nop')
        permutation = []
        perm_diag = []
        counter = 0
        counter2 = -1
        for arg_with_ind in self.args_with_ind:
            perm = []
            for i in arg_with_ind.indices:
                if i is not None:
                    if i < 0:
                        perm_diag.append(counter2)
                        counter2 -= 1
                    continue
                perm.append(counter)
                counter += 1
            permutation.append(perm)
        max_ind = max([max(i) if i else -1 for i in permutation]) if permutation else -1
        perm_diag = [max_ind - i for i in perm_diag]
        self._track_permutation = permutation + [perm_diag]

    def track_permutation_merge(self, destination: _ArgE, from_element: _ArgE):
        if False:
            for i in range(10):
                print('nop')
        index_destination = self.args_with_ind.index(destination)
        index_element = self.args_with_ind.index(from_element)
        self._track_permutation[index_destination].extend(self._track_permutation[index_element])
        self._track_permutation.pop(index_element)

    def get_absolute_free_range(self, arg: _ArgE) -> typing.Tuple[int, int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the range of the free indices of the arg as absolute positions\n        among all free indices.\n        '
        counter = 0
        for arg_with_ind in self.args_with_ind:
            number_free_indices = len([i for i in arg_with_ind.indices if i is None])
            if arg_with_ind == arg:
                return (counter, counter + number_free_indices)
            counter += number_free_indices
        raise IndexError('argument not found')

    def get_absolute_range(self, arg: _ArgE) -> typing.Tuple[int, int]:
        if False:
            while True:
                i = 10
        '\n        Return the absolute range of indices for arg, disregarding dummy\n        indices.\n        '
        counter = 0
        for arg_with_ind in self.args_with_ind:
            number_indices = len(arg_with_ind.indices)
            if arg_with_ind == arg:
                return (counter, counter + number_indices)
            counter += number_indices
        raise IndexError('argument not found')

def get_rank(expr):
    if False:
        return 10
    if isinstance(expr, (MatrixExpr, MatrixElement)):
        return 2
    if isinstance(expr, _CodegenArrayAbstract):
        return len(expr.shape)
    if isinstance(expr, NDimArray):
        return expr.rank()
    if isinstance(expr, Indexed):
        return expr.rank
    if isinstance(expr, IndexedBase):
        shape = expr.shape
        if shape is None:
            return -1
        else:
            return len(shape)
    if hasattr(expr, 'shape'):
        return len(expr.shape)
    return 0

def _get_subrank(expr):
    if False:
        return 10
    if isinstance(expr, _CodegenArrayAbstract):
        return expr.subrank()
    return get_rank(expr)

def _get_subranks(expr):
    if False:
        print('Hello World!')
    if isinstance(expr, _CodegenArrayAbstract):
        return expr.subranks
    else:
        return [get_rank(expr)]

def get_shape(expr):
    if False:
        i = 10
        return i + 15
    if hasattr(expr, 'shape'):
        return expr.shape
    return ()

def nest_permutation(expr):
    if False:
        print('Hello World!')
    if isinstance(expr, PermuteDims):
        return expr.nest_permutation()
    else:
        return expr

def _array_tensor_product(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    return ArrayTensorProduct(*args, canonicalize=True, **kwargs)

def _array_contraction(expr, *contraction_indices, **kwargs):
    if False:
        print('Hello World!')
    return ArrayContraction(expr, *contraction_indices, canonicalize=True, **kwargs)

def _array_diagonal(expr, *diagonal_indices, **kwargs):
    if False:
        i = 10
        return i + 15
    return ArrayDiagonal(expr, *diagonal_indices, canonicalize=True, **kwargs)

def _permute_dims(expr, permutation, **kwargs):
    if False:
        i = 10
        return i + 15
    return PermuteDims(expr, permutation, canonicalize=True, **kwargs)

def _array_add(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return ArrayAdd(*args, canonicalize=True, **kwargs)

def _get_array_element_or_slice(expr, indices):
    if False:
        return 10
    return ArrayElement(expr, indices)