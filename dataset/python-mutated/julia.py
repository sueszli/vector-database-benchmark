"""
Julia code printer

The `JuliaCodePrinter` converts SymPy expressions into Julia expressions.

A complete code generator, which uses `julia_code` extensively, can be found
in `sympy.utilities.codegen`.  The `codegen` module can be used to generate
complete source code files.

"""
from __future__ import annotations
from typing import Any
from sympy.core import Mul, Pow, S, Rational
from sympy.core.mul import _keep_coeff
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from re import search
known_fcns_src1 = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'asin', 'acos', 'atan', 'acot', 'asec', 'acsc', 'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch', 'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch', 'sinc', 'atan2', 'sign', 'floor', 'log', 'exp', 'cbrt', 'sqrt', 'erf', 'erfc', 'erfi', 'factorial', 'gamma', 'digamma', 'trigamma', 'polygamma', 'beta', 'airyai', 'airyaiprime', 'airybi', 'airybiprime', 'besselj', 'bessely', 'besseli', 'besselk', 'erfinv', 'erfcinv']
known_fcns_src2 = {'Abs': 'abs', 'ceiling': 'ceil', 'conjugate': 'conj', 'hankel1': 'hankelh1', 'hankel2': 'hankelh2', 'im': 'imag', 're': 'real'}

class JuliaCodePrinter(CodePrinter):
    """
    A printer to convert expressions to strings of Julia code.
    """
    printmethod = '_julia'
    language = 'Julia'
    _operators = {'and': '&&', 'or': '||', 'not': '!'}
    _default_settings: dict[str, Any] = {'order': None, 'full_prec': 'auto', 'precision': 17, 'user_functions': {}, 'human': True, 'allow_unknown_functions': False, 'contract': True, 'inline': True}

    def __init__(self, settings={}):
        if False:
            while True:
                i = 10
        super().__init__(settings)
        self.known_functions = dict(zip(known_fcns_src1, known_fcns_src1))
        self.known_functions.update(dict(known_fcns_src2))
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)

    def _rate_index_position(self, p):
        if False:
            for i in range(10):
                print('nop')
        return p * 5

    def _get_statement(self, codestring):
        if False:
            while True:
                i = 10
        return '%s' % codestring

    def _get_comment(self, text):
        if False:
            i = 10
            return i + 15
        return '# {}'.format(text)

    def _declare_number_const(self, name, value):
        if False:
            return 10
        return 'const {} = {}'.format(name, value)

    def _format_code(self, lines):
        if False:
            while True:
                i = 10
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        if False:
            print('Hello World!')
        (rows, cols) = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))

    def _get_loop_opening_ending(self, indices):
        if False:
            return 10
        open_lines = []
        close_lines = []
        for i in indices:
            (var, start, stop) = map(self._print, [i.label, i.lower + 1, i.upper + 1])
            open_lines.append('for %s = %s:%s' % (var, start, stop))
            close_lines.append('end')
        return (open_lines, close_lines)

    def _print_Mul(self, expr):
        if False:
            for i in range(10):
                print('nop')
        if expr.is_number and expr.is_imaginary and expr.as_coeff_Mul()[0].is_integer:
            return '%sim' % self._print(-S.ImaginaryUnit * expr)
        prec = precedence(expr)
        (c, e) = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = '-'
        else:
            sign = ''
        a = []
        b = []
        pow_paren = []
        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            args = Mul.make_args(expr)
        for item in args:
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity and (item.p == 1):
                b.append(Rational(item.q))
            else:
                a.append(item)
        a = a or [S.One]
        a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = '(%s)' % b_str[b.index(item.base)]

        def multjoin(a, a_str):
            if False:
                return 10
            r = a_str[0]
            for i in range(1, len(a)):
                mulsym = '*' if a[i - 1].is_number else '.*'
                r = '%s %s %s' % (r, mulsym, a_str[i])
            return r
        if not b:
            return sign + multjoin(a, a_str)
        elif len(b) == 1:
            divsym = '/' if b[0].is_number else './'
            return '%s %s %s' % (sign + multjoin(a, a_str), divsym, b_str[0])
        else:
            divsym = '/' if all((bi.is_number for bi in b)) else './'
            return '%s %s (%s)' % (sign + multjoin(a, a_str), divsym, multjoin(b, b_str))

    def _print_Relational(self, expr):
        if False:
            while True:
                i = 10
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return '{} {} {}'.format(lhs_code, op, rhs_code)

    def _print_Pow(self, expr):
        if False:
            i = 10
            return i + 15
        powsymbol = '^' if all((x.is_number for x in expr.args)) else '.^'
        PREC = precedence(expr)
        if equal_valued(expr.exp, 0.5):
            return 'sqrt(%s)' % self._print(expr.base)
        if expr.is_commutative:
            if equal_valued(expr.exp, -0.5):
                sym = '/' if expr.base.is_number else './'
                return '1 %s sqrt(%s)' % (sym, self._print(expr.base))
            if equal_valued(expr.exp, -1):
                sym = '/' if expr.base.is_number else './'
                return '1 %s %s' % (sym, self.parenthesize(expr.base, PREC))
        return '%s %s %s' % (self.parenthesize(expr.base, PREC), powsymbol, self.parenthesize(expr.exp, PREC))

    def _print_MatPow(self, expr):
        if False:
            while True:
                i = 10
        PREC = precedence(expr)
        return '%s ^ %s' % (self.parenthesize(expr.base, PREC), self.parenthesize(expr.exp, PREC))

    def _print_Pi(self, expr):
        if False:
            while True:
                i = 10
        if self._settings['inline']:
            return 'pi'
        else:
            return super()._print_NumberSymbol(expr)

    def _print_ImaginaryUnit(self, expr):
        if False:
            return 10
        return 'im'

    def _print_Exp1(self, expr):
        if False:
            while True:
                i = 10
        if self._settings['inline']:
            return 'e'
        else:
            return super()._print_NumberSymbol(expr)

    def _print_EulerGamma(self, expr):
        if False:
            i = 10
            return i + 15
        if self._settings['inline']:
            return 'eulergamma'
        else:
            return super()._print_NumberSymbol(expr)

    def _print_Catalan(self, expr):
        if False:
            for i in range(10):
                print('nop')
        if self._settings['inline']:
            return 'catalan'
        else:
            return super()._print_NumberSymbol(expr)

    def _print_GoldenRatio(self, expr):
        if False:
            while True:
                i = 10
        if self._settings['inline']:
            return 'golden'
        else:
            return super()._print_NumberSymbol(expr)

    def _print_Assignment(self, expr):
        if False:
            while True:
                i = 10
        from sympy.codegen.ast import Assignment
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.tensor.indexed import IndexedBase
        lhs = expr.lhs
        rhs = expr.rhs
        if not self._settings['inline'] and isinstance(expr.rhs, Piecewise):
            expressions = []
            conditions = []
            for (e, c) in rhs.args:
                expressions.append(Assignment(lhs, e))
                conditions.append(c)
            temp = Piecewise(*zip(expressions, conditions))
            return self._print(temp)
        if self._settings['contract'] and (lhs.has(IndexedBase) or rhs.has(IndexedBase)):
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement('%s = %s' % (lhs_code, rhs_code))

    def _print_Infinity(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return 'Inf'

    def _print_NegativeInfinity(self, expr):
        if False:
            i = 10
            return i + 15
        return '-Inf'

    def _print_NaN(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return 'NaN'

    def _print_list(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return 'Any[' + ', '.join((self._print(a) for a in expr)) + ']'

    def _print_tuple(self, expr):
        if False:
            while True:
                i = 10
        if len(expr) == 1:
            return '(%s,)' % self._print(expr[0])
        else:
            return '(%s)' % self.stringify(expr, ', ')
    _print_Tuple = _print_tuple

    def _print_BooleanTrue(self, expr):
        if False:
            return 10
        return 'true'

    def _print_BooleanFalse(self, expr):
        if False:
            i = 10
            return i + 15
        return 'false'

    def _print_bool(self, expr):
        if False:
            while True:
                i = 10
        return str(expr).lower()

    def _print_MatrixBase(self, A):
        if False:
            while True:
                i = 10
        if S.Zero in A.shape:
            return 'zeros(%s, %s)' % (A.rows, A.cols)
        elif (A.rows, A.cols) == (1, 1):
            return '[%s]' % A[0, 0]
        elif A.rows == 1:
            return '[%s]' % A.table(self, rowstart='', rowend='', colsep=' ')
        elif A.cols == 1:
            return '[%s]' % ', '.join([self._print(a) for a in A])
        return '[%s]' % A.table(self, rowstart='', rowend='', rowsep=';\n', colsep=' ')

    def _print_SparseRepMatrix(self, A):
        if False:
            while True:
                i = 10
        from sympy.matrices import Matrix
        L = A.col_list()
        I = Matrix([k[0] + 1 for k in L])
        J = Matrix([k[1] + 1 for k in L])
        AIJ = Matrix([k[2] for k in L])
        return 'sparse(%s, %s, %s, %s, %s)' % (self._print(I), self._print(J), self._print(AIJ), A.rows, A.cols)

    def _print_MatrixElement(self, expr):
        if False:
            print('Hello World!')
        return self.parenthesize(expr.parent, PRECEDENCE['Atom'], strict=True) + '[%s,%s]' % (expr.i + 1, expr.j + 1)

    def _print_MatrixSlice(self, expr):
        if False:
            i = 10
            return i + 15

        def strslice(x, lim):
            if False:
                for i in range(10):
                    print('nop')
            l = x[0] + 1
            h = x[1]
            step = x[2]
            lstr = self._print(l)
            hstr = 'end' if h == lim else self._print(h)
            if step == 1:
                if l == 1 and h == lim:
                    return ':'
                if l == h:
                    return lstr
                else:
                    return lstr + ':' + hstr
            else:
                return ':'.join((lstr, self._print(step), hstr))
        return self._print(expr.parent) + '[' + strslice(expr.rowslice, expr.parent.shape[0]) + ',' + strslice(expr.colslice, expr.parent.shape[1]) + ']'

    def _print_Indexed(self, expr):
        if False:
            while True:
                i = 10
        inds = [self._print(i) for i in expr.indices]
        return '%s[%s]' % (self._print(expr.base.label), ','.join(inds))

    def _print_Idx(self, expr):
        if False:
            while True:
                i = 10
        return self._print(expr.label)

    def _print_Identity(self, expr):
        if False:
            return 10
        return 'eye(%s)' % self._print(expr.shape[0])

    def _print_HadamardProduct(self, expr):
        if False:
            i = 10
            return i + 15
        return ' .* '.join([self.parenthesize(arg, precedence(expr)) for arg in expr.args])

    def _print_HadamardPower(self, expr):
        if False:
            print('Hello World!')
        PREC = precedence(expr)
        return '.**'.join([self.parenthesize(expr.base, PREC), self.parenthesize(expr.exp, PREC)])

    def _print_Rational(self, expr):
        if False:
            for i in range(10):
                print('nop')
        if expr.q == 1:
            return str(expr.p)
        return '%s // %s' % (expr.p, expr.q)

    def _print_jn(self, expr):
        if False:
            for i in range(10):
                print('nop')
        from sympy.functions import sqrt, besselj
        x = expr.argument
        expr2 = sqrt(S.Pi / (2 * x)) * besselj(expr.order + S.Half, x)
        return self._print(expr2)

    def _print_yn(self, expr):
        if False:
            print('Hello World!')
        from sympy.functions import sqrt, bessely
        x = expr.argument
        expr2 = sqrt(S.Pi / (2 * x)) * bessely(expr.order + S.Half, x)
        return self._print(expr2)

    def _print_Piecewise(self, expr):
        if False:
            return 10
        if expr.args[-1].cond != True:
            raise ValueError('All Piecewise expressions must contain an (expr, True) statement to be used as a default condition. Without one, the generated expression may not evaluate to anything under some condition.')
        lines = []
        if self._settings['inline']:
            ecpairs = ['({}) ? ({}) :'.format(self._print(c), self._print(e)) for (e, c) in expr.args[:-1]]
            elast = ' (%s)' % self._print(expr.args[-1].expr)
            pw = '\n'.join(ecpairs) + elast
            return '(' + pw + ')'
        else:
            for (i, (e, c)) in enumerate(expr.args):
                if i == 0:
                    lines.append('if (%s)' % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append('else')
                else:
                    lines.append('elseif (%s)' % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                if i == len(expr.args) - 1:
                    lines.append('end')
            return '\n'.join(lines)

    def _print_MatMul(self, expr):
        if False:
            i = 10
            return i + 15
        (c, m) = expr.as_coeff_mmul()
        sign = ''
        if c.is_number:
            (re, im) = c.as_real_imag()
            if im.is_zero and re.is_negative:
                expr = _keep_coeff(-c, m)
                sign = '-'
            elif re.is_zero and im.is_negative:
                expr = _keep_coeff(-c, m)
                sign = '-'
        return sign + ' * '.join((self.parenthesize(arg, precedence(expr)) for arg in expr.args))

    def indent_code(self, code):
        if False:
            return 10
        'Accepts a string of code or a list of code lines'
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)
        tab = '    '
        inc_regex = ('^function ', '^if ', '^elseif ', '^else$', '^for ')
        dec_regex = ('^end$', '^elseif ', '^else$')
        code = [line.lstrip(' \t') for line in code]
        increase = [int(any((search(re, line) for re in inc_regex))) for line in code]
        decrease = [int(any((search(re, line) for re in dec_regex))) for line in code]
        pretty = []
        level = 0
        for (n, line) in enumerate(code):
            if line in ('', '\n'):
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append('%s%s' % (tab * level, line))
            level += increase[n]
        return pretty

def julia_code(expr, assign_to=None, **settings):
    if False:
        while True:
            i = 10
    'Converts `expr` to a string of Julia code.\n\n    Parameters\n    ==========\n\n    expr : Expr\n        A SymPy expression to be converted.\n    assign_to : optional\n        When given, the argument is used as the name of the variable to which\n        the expression is assigned.  Can be a string, ``Symbol``,\n        ``MatrixSymbol``, or ``Indexed`` type.  This can be helpful for\n        expressions that generate multi-line statements.\n    precision : integer, optional\n        The precision for numbers such as pi  [default=16].\n    user_functions : dict, optional\n        A dictionary where keys are ``FunctionClass`` instances and values are\n        their string representations.  Alternatively, the dictionary value can\n        be a list of tuples i.e. [(argument_test, cfunction_string)].  See\n        below for examples.\n    human : bool, optional\n        If True, the result is a single string that may contain some constant\n        declarations for the number symbols.  If False, the same information is\n        returned in a tuple of (symbols_to_declare, not_supported_functions,\n        code_text).  [default=True].\n    contract: bool, optional\n        If True, ``Indexed`` instances are assumed to obey tensor contraction\n        rules and the corresponding nested loops over indices are generated.\n        Setting contract=False will not generate loops, instead the user is\n        responsible to provide values for the indices in the code.\n        [default=True].\n    inline: bool, optional\n        If True, we try to create single-statement code instead of multiple\n        statements.  [default=True].\n\n    Examples\n    ========\n\n    >>> from sympy import julia_code, symbols, sin, pi\n    >>> x = symbols(\'x\')\n    >>> julia_code(sin(x).series(x).removeO())\n    \'x .^ 5 / 120 - x .^ 3 / 6 + x\'\n\n    >>> from sympy import Rational, ceiling\n    >>> x, y, tau = symbols("x, y, tau")\n    >>> julia_code((2*tau)**Rational(7, 2))\n    \'8 * sqrt(2) * tau .^ (7 // 2)\'\n\n    Note that element-wise (Hadamard) operations are used by default between\n    symbols.  This is because its possible in Julia to write "vectorized"\n    code.  It is harmless if the values are scalars.\n\n    >>> julia_code(sin(pi*x*y), assign_to="s")\n    \'s = sin(pi * x .* y)\'\n\n    If you need a matrix product "*" or matrix power "^", you can specify the\n    symbol as a ``MatrixSymbol``.\n\n    >>> from sympy import Symbol, MatrixSymbol\n    >>> n = Symbol(\'n\', integer=True, positive=True)\n    >>> A = MatrixSymbol(\'A\', n, n)\n    >>> julia_code(3*pi*A**3)\n    \'(3 * pi) * A ^ 3\'\n\n    This class uses several rules to decide which symbol to use a product.\n    Pure numbers use "*", Symbols use ".*" and MatrixSymbols use "*".\n    A HadamardProduct can be used to specify componentwise multiplication ".*"\n    of two MatrixSymbols.  There is currently there is no easy way to specify\n    scalar symbols, so sometimes the code might have some minor cosmetic\n    issues.  For example, suppose x and y are scalars and A is a Matrix, then\n    while a human programmer might write "(x^2*y)*A^3", we generate:\n\n    >>> julia_code(x**2*y*A**3)\n    \'(x .^ 2 .* y) * A ^ 3\'\n\n    Matrices are supported using Julia inline notation.  When using\n    ``assign_to`` with matrices, the name can be specified either as a string\n    or as a ``MatrixSymbol``.  The dimensions must align in the latter case.\n\n    >>> from sympy import Matrix, MatrixSymbol\n    >>> mat = Matrix([[x**2, sin(x), ceiling(x)]])\n    >>> julia_code(mat, assign_to=\'A\')\n    \'A = [x .^ 2 sin(x) ceil(x)]\'\n\n    ``Piecewise`` expressions are implemented with logical masking by default.\n    Alternatively, you can pass "inline=False" to use if-else conditionals.\n    Note that if the ``Piecewise`` lacks a default term, represented by\n    ``(expr, True)`` then an error will be thrown.  This is to prevent\n    generating an expression that may not evaluate to anything.\n\n    >>> from sympy import Piecewise\n    >>> pw = Piecewise((x + 1, x > 0), (x, True))\n    >>> julia_code(pw, assign_to=tau)\n    \'tau = ((x > 0) ? (x + 1) : (x))\'\n\n    Note that any expression that can be generated normally can also exist\n    inside a Matrix:\n\n    >>> mat = Matrix([[x**2, pw, sin(x)]])\n    >>> julia_code(mat, assign_to=\'A\')\n    \'A = [x .^ 2 ((x > 0) ? (x + 1) : (x)) sin(x)]\'\n\n    Custom printing can be defined for certain types by passing a dictionary of\n    "type" : "function" to the ``user_functions`` kwarg.  Alternatively, the\n    dictionary value can be a list of tuples i.e., [(argument_test,\n    cfunction_string)].  This can be used to call a custom Julia function.\n\n    >>> from sympy import Function\n    >>> f = Function(\'f\')\n    >>> g = Function(\'g\')\n    >>> custom_functions = {\n    ...   "f": "existing_julia_fcn",\n    ...   "g": [(lambda x: x.is_Matrix, "my_mat_fcn"),\n    ...         (lambda x: not x.is_Matrix, "my_fcn")]\n    ... }\n    >>> mat = Matrix([[1, x]])\n    >>> julia_code(f(x) + g(x) + g(mat), user_functions=custom_functions)\n    \'existing_julia_fcn(x) + my_fcn(x) + my_mat_fcn([1 x])\'\n\n    Support for loops is provided through ``Indexed`` types. With\n    ``contract=True`` these expressions will be turned into loops, whereas\n    ``contract=False`` will just print the assignment expression that should be\n    looped over:\n\n    >>> from sympy import Eq, IndexedBase, Idx\n    >>> len_y = 5\n    >>> y = IndexedBase(\'y\', shape=(len_y,))\n    >>> t = IndexedBase(\'t\', shape=(len_y,))\n    >>> Dy = IndexedBase(\'Dy\', shape=(len_y-1,))\n    >>> i = Idx(\'i\', len_y-1)\n    >>> e = Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))\n    >>> julia_code(e.rhs, assign_to=e.lhs, contract=False)\n    \'Dy[i] = (y[i + 1] - y[i]) ./ (t[i + 1] - t[i])\'\n    '
    return JuliaCodePrinter(settings).doprint(expr, assign_to)

def print_julia_code(expr, **settings):
    if False:
        print('Hello World!')
    'Prints the Julia representation of the given expression.\n\n    See `julia_code` for the meaning of the optional arguments.\n    '
    print(julia_code(expr, **settings))