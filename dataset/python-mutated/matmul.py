from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict
from sympy.core import Basic, sympify, S
from sympy.core.mul import mul, Mul
from sympy.core.numbers import Number, Integer
from sympy.core.symbol import Dummy
from sympy.functions import adjoint
from sympy.strategies import rm_id, unpack, typed, flatten, exhaust, do_one, new
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices.matrices import MatrixBase
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.matrices.expressions._shape import validate_matmul_integer as validate
from .inverse import Inverse
from .matexpr import MatrixExpr
from .matpow import MatPow
from .transpose import transpose
from .permutation import PermutationMatrix
from .special import ZeroMatrix, Identity, GenericIdentity, OneMatrix

class MatMul(MatrixExpr, Mul):
    """
    A product of matrix expressions

    Examples
    ========

    >>> from sympy import MatMul, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 4)
    >>> B = MatrixSymbol('B', 4, 3)
    >>> C = MatrixSymbol('C', 3, 6)
    >>> MatMul(A, B, C)
    A*B*C
    """
    is_MatMul = True
    identity = GenericIdentity()

    def __new__(cls, *args, evaluate=False, check=None, _sympify=True):
        if False:
            return 10
        if not args:
            return cls.identity
        args = list(filter(lambda i: cls.identity != i, args))
        if _sympify:
            args = list(map(sympify, args))
        obj = Basic.__new__(cls, *args)
        (factor, matrices) = obj.as_coeff_matrices()
        if check is not None:
            sympy_deprecation_warning('Passing check to MatMul is deprecated and the check argument will be removed in a future version.', deprecated_since_version='1.11', active_deprecations_target='remove-check-argument-from-matrix-operations')
        if check is not False:
            validate(*matrices)
        if not matrices:
            return factor
        if evaluate:
            return cls._evaluate(obj)
        return obj

    @classmethod
    def _evaluate(cls, expr):
        if False:
            print('Hello World!')
        return canonicalize(expr)

    @property
    def shape(self):
        if False:
            i = 10
            return i + 15
        matrices = [arg for arg in self.args if arg.is_Matrix]
        return (matrices[0].rows, matrices[-1].cols)

    def _entry(self, i, j, expand=True, **kwargs):
        if False:
            while True:
                i = 10
        from sympy.concrete.summations import Sum
        from sympy.matrices.immutable import ImmutableMatrix
        (coeff, matrices) = self.as_coeff_matrices()
        if len(matrices) == 1:
            return coeff * matrices[0][i, j]
        indices = [None] * (len(matrices) + 1)
        ind_ranges = [None] * (len(matrices) - 1)
        indices[0] = i
        indices[-1] = j

        def f():
            if False:
                for i in range(10):
                    print('nop')
            counter = 1
            while True:
                yield Dummy('i_%i' % counter)
                counter += 1
        dummy_generator = kwargs.get('dummy_generator', f())
        for i in range(1, len(matrices)):
            indices[i] = next(dummy_generator)
        for (i, arg) in enumerate(matrices[:-1]):
            ind_ranges[i] = arg.shape[1] - 1
        matrices = [arg._entry(indices[i], indices[i + 1], dummy_generator=dummy_generator) for (i, arg) in enumerate(matrices)]
        expr_in_sum = Mul.fromiter(matrices)
        if any((v.has(ImmutableMatrix) for v in matrices)):
            expand = True
        result = coeff * Sum(expr_in_sum, *zip(indices[1:-1], [0] * len(ind_ranges), ind_ranges))
        if not any((isinstance(v, (Integer, int)) for v in ind_ranges)):
            expand = False
        return result.doit() if expand else result

    def as_coeff_matrices(self):
        if False:
            return 10
        scalars = [x for x in self.args if not x.is_Matrix]
        matrices = [x for x in self.args if x.is_Matrix]
        coeff = Mul(*scalars)
        if coeff.is_commutative is False:
            raise NotImplementedError('noncommutative scalars in MatMul are not supported.')
        return (coeff, matrices)

    def as_coeff_mmul(self):
        if False:
            i = 10
            return i + 15
        (coeff, matrices) = self.as_coeff_matrices()
        return (coeff, MatMul(*matrices))

    def expand(self, **kwargs):
        if False:
            print('Hello World!')
        expanded = super(MatMul, self).expand(**kwargs)
        return self._evaluate(expanded)

    def _eval_transpose(self):
        if False:
            print('Hello World!')
        'Transposition of matrix multiplication.\n\n        Notes\n        =====\n\n        The following rules are applied.\n\n        Transposition for matrix multiplied with another matrix:\n        `\\left(A B\\right)^{T} = B^{T} A^{T}`\n\n        Transposition for matrix multiplied with scalar:\n        `\\left(c A\\right)^{T} = c A^{T}`\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Transpose\n        '
        (coeff, matrices) = self.as_coeff_matrices()
        return MatMul(coeff, *[transpose(arg) for arg in matrices[::-1]]).doit()

    def _eval_adjoint(self):
        if False:
            print('Hello World!')
        return MatMul(*[adjoint(arg) for arg in self.args[::-1]]).doit()

    def _eval_trace(self):
        if False:
            while True:
                i = 10
        (factor, mmul) = self.as_coeff_mmul()
        if factor != 1:
            from .trace import trace
            return factor * trace(mmul.doit())

    def _eval_determinant(self):
        if False:
            i = 10
            return i + 15
        from sympy.matrices.expressions.determinant import Determinant
        (factor, matrices) = self.as_coeff_matrices()
        square_matrices = only_squares(*matrices)
        return factor ** self.rows * Mul(*list(map(Determinant, square_matrices)))

    def _eval_inverse(self):
        if False:
            return 10
        if all((arg.is_square for arg in self.args if isinstance(arg, MatrixExpr))):
            return MatMul(*(arg.inverse() if isinstance(arg, MatrixExpr) else arg ** (-1) for arg in self.args[::-1])).doit()
        return Inverse(self)

    def doit(self, **hints):
        if False:
            i = 10
            return i + 15
        deep = hints.get('deep', True)
        if deep:
            args = tuple((arg.doit(**hints) for arg in self.args))
        else:
            args = self.args
        expr = canonicalize(MatMul(*args))
        return expr

    def args_cnc(self, cset=False, warn=True, **kwargs):
        if False:
            return 10
        coeff_c = [x for x in self.args if x.is_commutative]
        coeff_nc = [x for x in self.args if not x.is_commutative]
        if cset:
            clen = len(coeff_c)
            coeff_c = set(coeff_c)
            if clen and warn and (len(coeff_c) != clen):
                raise ValueError('repeated commutative arguments: %s' % [ci for ci in coeff_c if list(self.args).count(ci) > 1])
        return [coeff_c, coeff_nc]

    def _eval_derivative_matrix_lines(self, x):
        if False:
            i = 10
            return i + 15
        from .transpose import Transpose
        with_x_ind = [i for (i, arg) in enumerate(self.args) if arg.has(x)]
        lines = []
        for ind in with_x_ind:
            left_args = self.args[:ind]
            right_args = self.args[ind + 1:]
            if right_args:
                right_mat = MatMul.fromiter(right_args)
            else:
                right_mat = Identity(self.shape[1])
            if left_args:
                left_rev = MatMul.fromiter([Transpose(i).doit() if i.is_Matrix else i for i in reversed(left_args)])
            else:
                left_rev = Identity(self.shape[0])
            d = self.args[ind]._eval_derivative_matrix_lines(x)
            for i in d:
                i.append_first(left_rev)
                i.append_second(right_mat)
                lines.append(i)
        return lines
mul.register_handlerclass((Mul, MatMul), MatMul)

def newmul(*args):
    if False:
        i = 10
        return i + 15
    if args[0] == 1:
        args = args[1:]
    return new(MatMul, *args)

def any_zeros(mul):
    if False:
        print('Hello World!')
    if any((arg.is_zero or (arg.is_Matrix and arg.is_ZeroMatrix) for arg in mul.args)):
        matrices = [arg for arg in mul.args if arg.is_Matrix]
        return ZeroMatrix(matrices[0].rows, matrices[-1].cols)
    return mul

def merge_explicit(matmul):
    if False:
        while True:
            i = 10
    " Merge explicit MatrixBase arguments\n\n    >>> from sympy import MatrixSymbol, Matrix, MatMul, pprint\n    >>> from sympy.matrices.expressions.matmul import merge_explicit\n    >>> A = MatrixSymbol('A', 2, 2)\n    >>> B = Matrix([[1, 1], [1, 1]])\n    >>> C = Matrix([[1, 2], [3, 4]])\n    >>> X = MatMul(A, B, C)\n    >>> pprint(X)\n      [1  1] [1  2]\n    A*[    ]*[    ]\n      [1  1] [3  4]\n    >>> pprint(merge_explicit(X))\n      [4  6]\n    A*[    ]\n      [4  6]\n\n    >>> X = MatMul(B, A, C)\n    >>> pprint(X)\n    [1  1]   [1  2]\n    [    ]*A*[    ]\n    [1  1]   [3  4]\n    >>> pprint(merge_explicit(X))\n    [1  1]   [1  2]\n    [    ]*A*[    ]\n    [1  1]   [3  4]\n    "
    if not any((isinstance(arg, MatrixBase) for arg in matmul.args)):
        return matmul
    newargs = []
    last = matmul.args[0]
    for arg in matmul.args[1:]:
        if isinstance(arg, (MatrixBase, Number)) and isinstance(last, (MatrixBase, Number)):
            last = last * arg
        else:
            newargs.append(last)
            last = arg
    newargs.append(last)
    return MatMul(*newargs)

def remove_ids(mul):
    if False:
        return 10
    ' Remove Identities from a MatMul\n\n    This is a modified version of sympy.strategies.rm_id.\n    This is necesssary because MatMul may contain both MatrixExprs and Exprs\n    as args.\n\n    See Also\n    ========\n\n    sympy.strategies.rm_id\n    '
    (factor, mmul) = mul.as_coeff_mmul()
    result = rm_id(lambda x: x.is_Identity is True)(mmul)
    if result != mmul:
        return newmul(factor, *result.args)
    else:
        return mul

def factor_in_front(mul):
    if False:
        print('Hello World!')
    (factor, matrices) = mul.as_coeff_matrices()
    if factor != 1:
        return newmul(factor, *matrices)
    return mul

def combine_powers(mul):
    if False:
        return 10
    'Combine consecutive powers with the same base into one, e.g.\n    $$A \\times A^2 \\Rightarrow A^3$$\n\n    This also cancels out the possible matrix inverses using the\n    knowledgebase of :class:`~.Inverse`, e.g.,\n    $$ Y \\times X \\times X^{-1} \\Rightarrow Y $$\n    '
    (factor, args) = mul.as_coeff_matrices()
    new_args = [args[0]]
    for i in range(1, len(args)):
        A = new_args[-1]
        B = args[i]
        if isinstance(B, Inverse) and isinstance(B.arg, MatMul):
            Bargs = B.arg.args
            l = len(Bargs)
            if list(Bargs) == new_args[-l:]:
                new_args = new_args[:-l] + [Identity(B.shape[0])]
                continue
        if isinstance(A, Inverse) and isinstance(A.arg, MatMul):
            Aargs = A.arg.args
            l = len(Aargs)
            if list(Aargs) == args[i:i + l]:
                identity = Identity(A.shape[0])
                new_args[-1] = identity
                for j in range(i, i + l):
                    args[j] = identity
                continue
        if A.is_square == False or B.is_square == False:
            new_args.append(B)
            continue
        if isinstance(A, MatPow):
            (A_base, A_exp) = A.args
        else:
            (A_base, A_exp) = (A, S.One)
        if isinstance(B, MatPow):
            (B_base, B_exp) = B.args
        else:
            (B_base, B_exp) = (B, S.One)
        if A_base == B_base:
            new_exp = A_exp + B_exp
            new_args[-1] = MatPow(A_base, new_exp).doit(deep=False)
            continue
        elif not isinstance(B_base, MatrixBase):
            try:
                B_base_inv = B_base.inverse()
            except NonInvertibleMatrixError:
                B_base_inv = None
            if B_base_inv is not None and A_base == B_base_inv:
                new_exp = A_exp - B_exp
                new_args[-1] = MatPow(A_base, new_exp).doit(deep=False)
                continue
        new_args.append(B)
    return newmul(factor, *new_args)

def combine_permutations(mul):
    if False:
        for i in range(10):
            print('nop')
    'Refine products of permutation matrices as the products of cycles.\n    '
    args = mul.args
    l = len(args)
    if l < 2:
        return mul
    result = [args[0]]
    for i in range(1, l):
        A = result[-1]
        B = args[i]
        if isinstance(A, PermutationMatrix) and isinstance(B, PermutationMatrix):
            cycle_1 = A.args[0]
            cycle_2 = B.args[0]
            result[-1] = PermutationMatrix(cycle_1 * cycle_2)
        else:
            result.append(B)
    return MatMul(*result)

def combine_one_matrices(mul):
    if False:
        i = 10
        return i + 15
    '\n    Combine products of OneMatrix\n\n    e.g. OneMatrix(2, 3) * OneMatrix(3, 4) -> 3 * OneMatrix(2, 4)\n    '
    (factor, args) = mul.as_coeff_matrices()
    new_args = [args[0]]
    for B in args[1:]:
        A = new_args[-1]
        if not isinstance(A, OneMatrix) or not isinstance(B, OneMatrix):
            new_args.append(B)
            continue
        new_args.pop()
        new_args.append(OneMatrix(A.shape[0], B.shape[1]))
        factor *= A.shape[1]
    return newmul(factor, *new_args)

def distribute_monom(mul):
    if False:
        for i in range(10):
            print('nop')
    '\n    Simplify MatMul expressions but distributing\n    rational term to MatMul.\n\n    e.g. 2*(A+B) -> 2*A + 2*B\n    '
    args = mul.args
    if len(args) == 2:
        from .matadd import MatAdd
        if args[0].is_MatAdd and args[1].is_Rational:
            return MatAdd(*[MatMul(mat, args[1]).doit() for mat in args[0].args])
        if args[1].is_MatAdd and args[0].is_Rational:
            return MatAdd(*[MatMul(args[0], mat).doit() for mat in args[1].args])
    return mul
rules = (distribute_monom, any_zeros, remove_ids, combine_one_matrices, combine_powers, unpack, rm_id(lambda x: x == 1), merge_explicit, factor_in_front, flatten, combine_permutations)
canonicalize = exhaust(typed({MatMul: do_one(*rules)}))

def only_squares(*matrices):
    if False:
        return 10
    'factor matrices only if they are square'
    if matrices[0].rows != matrices[-1].cols:
        raise RuntimeError('Invalid matrices being multiplied')
    out = []
    start = 0
    for (i, M) in enumerate(matrices):
        if M.cols == matrices[start].rows:
            out.append(MatMul(*matrices[start:i + 1]).doit())
            start = i + 1
    return out

def refine_MatMul(expr, assumptions):
    if False:
        i = 10
        return i + 15
    "\n    >>> from sympy import MatrixSymbol, Q, assuming, refine\n    >>> X = MatrixSymbol('X', 2, 2)\n    >>> expr = X * X.T\n    >>> print(expr)\n    X*X.T\n    >>> with assuming(Q.orthogonal(X)):\n    ...     print(refine(expr))\n    I\n    "
    newargs = []
    exprargs = []
    for args in expr.args:
        if args.is_Matrix:
            exprargs.append(args)
        else:
            newargs.append(args)
    last = exprargs[0]
    for arg in exprargs[1:]:
        if arg == last.T and ask(Q.orthogonal(arg), assumptions):
            last = Identity(arg.shape[0])
        elif arg == last.conjugate() and ask(Q.unitary(arg), assumptions):
            last = Identity(arg.shape[0])
        else:
            newargs.append(last)
            last = arg
    newargs.append(last)
    return MatMul(*newargs)
handlers_dict['MatMul'] = refine_MatMul