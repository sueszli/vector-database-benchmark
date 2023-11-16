"""
Rust code printer

The `RustCodePrinter` converts SymPy expressions into Rust expressions.

A complete code generator, which uses `rust_code` extensively, can be found
in `sympy.utilities.codegen`. The `codegen` module can be used to generate
complete source code files.

"""
from __future__ import annotations
from typing import Any
from sympy.core import S, Rational, Float, Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
known_functions = {'floor': 'floor', 'ceiling': 'ceil', 'Abs': 'abs', 'sign': 'signum', 'Pow': [(lambda base, exp: equal_valued(exp, -1), 'recip', 2), (lambda base, exp: equal_valued(exp, 0.5), 'sqrt', 2), (lambda base, exp: equal_valued(exp, -0.5), 'sqrt().recip', 2), (lambda base, exp: exp == Rational(1, 3), 'cbrt', 2), (lambda base, exp: equal_valued(base, 2), 'exp2', 3), (lambda base, exp: exp.is_integer, 'powi', 1), (lambda base, exp: not exp.is_integer, 'powf', 1)], 'exp': [(lambda exp: True, 'exp', 2)], 'log': 'ln', 'Max': 'max', 'Min': 'min', 'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'asin': 'asin', 'acos': 'acos', 'atan': 'atan', 'atan2': 'atan2', 'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh', 'asinh': 'asinh', 'acosh': 'acosh', 'atanh': 'atanh', 'sqrt': 'sqrt'}
reserved_words = ['abstract', 'alignof', 'as', 'become', 'box', 'break', 'const', 'continue', 'crate', 'do', 'else', 'enum', 'extern', 'false', 'final', 'fn', 'for', 'if', 'impl', 'in', 'let', 'loop', 'macro', 'match', 'mod', 'move', 'mut', 'offsetof', 'override', 'priv', 'proc', 'pub', 'pure', 'ref', 'return', 'Self', 'self', 'sizeof', 'static', 'struct', 'super', 'trait', 'true', 'type', 'typeof', 'unsafe', 'unsized', 'use', 'virtual', 'where', 'while', 'yield']

class RustCodePrinter(CodePrinter):
    """A printer to convert SymPy expressions to strings of Rust code"""
    printmethod = '_rust_code'
    language = 'Rust'
    _default_settings: dict[str, Any] = {'order': None, 'full_prec': 'auto', 'precision': 17, 'user_functions': {}, 'human': True, 'contract': True, 'dereference': set(), 'error_on_reserved': False, 'reserved_word_suffix': '_', 'inline': False}

    def __init__(self, settings={}):
        if False:
            return 10
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        self._dereference = set(settings.get('dereference', []))
        self.reserved_words = set(reserved_words)

    def _rate_index_position(self, p):
        if False:
            i = 10
            return i + 15
        return p * 5

    def _get_statement(self, codestring):
        if False:
            print('Hello World!')
        return '%s;' % codestring

    def _get_comment(self, text):
        if False:
            while True:
                i = 10
        return '// %s' % text

    def _declare_number_const(self, name, value):
        if False:
            i = 10
            return i + 15
        return 'const %s: f64 = %s;' % (name, value)

    def _format_code(self, lines):
        if False:
            for i in range(10):
                print('nop')
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        if False:
            i = 10
            return i + 15
        (rows, cols) = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    def _get_loop_opening_ending(self, indices):
        if False:
            while True:
                i = 10
        open_lines = []
        close_lines = []
        loopstart = 'for %(var)s in %(start)s..%(end)s {'
        for i in indices:
            open_lines.append(loopstart % {'var': self._print(i), 'start': self._print(i.lower), 'end': self._print(i.upper + 1)})
            close_lines.append('}')
        return (open_lines, close_lines)

    def _print_caller_var(self, expr):
        if False:
            i = 10
            return i + 15
        if len(expr.args) > 1:
            return '(' + self._print(expr) + ')'
        elif expr.is_number:
            return self._print(expr, _type=True)
        else:
            return self._print(expr)

    def _print_Function(self, expr):
        if False:
            while True:
                i = 10
        '\n        basic function for printing `Function`\n\n        Function Style :\n\n        1. args[0].func(args[1:]), method with arguments\n        2. args[0].func(), method without arguments\n        3. args[1].func(), method without arguments (e.g. (e, x) => x.exp())\n        4. func(args), function with arguments\n        '
        if expr.func.__name__ in self.known_functions:
            cond_func = self.known_functions[expr.func.__name__]
            func = None
            style = 1
            if isinstance(cond_func, str):
                func = cond_func
            else:
                for (cond, func, style) in cond_func:
                    if cond(*expr.args):
                        break
            if func is not None:
                if style == 1:
                    ret = '%(var)s.%(method)s(%(args)s)' % {'var': self._print_caller_var(expr.args[0]), 'method': func, 'args': self.stringify(expr.args[1:], ', ') if len(expr.args) > 1 else ''}
                elif style == 2:
                    ret = '%(var)s.%(method)s()' % {'var': self._print_caller_var(expr.args[0]), 'method': func}
                elif style == 3:
                    ret = '%(var)s.%(method)s()' % {'var': self._print_caller_var(expr.args[1]), 'method': func}
                else:
                    ret = '%(func)s(%(args)s)' % {'func': func, 'args': self.stringify(expr.args, ', ')}
                return ret
        elif hasattr(expr, '_imp_') and isinstance(expr._imp_, Lambda):
            return self._print(expr._imp_(*expr.args))
        elif expr.func.__name__ in self._rewriteable_functions:
            (target_f, required_fs) = self._rewriteable_functions[expr.func.__name__]
            if self._can_print(target_f) and all((self._can_print(f) for f in required_fs)):
                return self._print(expr.rewrite(target_f))
        else:
            return self._print_not_supported(expr)

    def _print_Pow(self, expr):
        if False:
            for i in range(10):
                print('nop')
        if expr.base.is_integer and (not expr.exp.is_integer):
            expr = type(expr)(Float(expr.base), expr.exp)
            return self._print(expr)
        return self._print_Function(expr)

    def _print_Float(self, expr, _type=False):
        if False:
            for i in range(10):
                print('nop')
        ret = super()._print_Float(expr)
        if _type:
            return ret + '_f64'
        else:
            return ret

    def _print_Integer(self, expr, _type=False):
        if False:
            print('Hello World!')
        ret = super()._print_Integer(expr)
        if _type:
            return ret + '_i32'
        else:
            return ret

    def _print_Rational(self, expr):
        if False:
            return 10
        (p, q) = (int(expr.p), int(expr.q))
        return '%d_f64/%d.0' % (p, q)

    def _print_Relational(self, expr):
        if False:
            return 10
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return '{} {} {}'.format(lhs_code, op, rhs_code)

    def _print_Indexed(self, expr):
        if False:
            print('Hello World!')
        dims = expr.shape
        elem = S.Zero
        offset = S.One
        for i in reversed(range(expr.rank)):
            elem += expr.indices[i] * offset
            offset *= dims[i]
        return '%s[%s]' % (self._print(expr.base.label), self._print(elem))

    def _print_Idx(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return expr.label.name

    def _print_Dummy(self, expr):
        if False:
            while True:
                i = 10
        return expr.name

    def _print_Exp1(self, expr, _type=False):
        if False:
            return 10
        return 'E'

    def _print_Pi(self, expr, _type=False):
        if False:
            return 10
        return 'PI'

    def _print_Infinity(self, expr, _type=False):
        if False:
            while True:
                i = 10
        return 'INFINITY'

    def _print_NegativeInfinity(self, expr, _type=False):
        if False:
            print('Hello World!')
        return 'NEG_INFINITY'

    def _print_BooleanTrue(self, expr, _type=False):
        if False:
            for i in range(10):
                print('nop')
        return 'true'

    def _print_BooleanFalse(self, expr, _type=False):
        if False:
            return 10
        return 'false'

    def _print_bool(self, expr, _type=False):
        if False:
            while True:
                i = 10
        return str(expr).lower()

    def _print_NaN(self, expr, _type=False):
        if False:
            return 10
        return 'NAN'

    def _print_Piecewise(self, expr):
        if False:
            while True:
                i = 10
        if expr.args[-1].cond != True:
            raise ValueError('All Piecewise expressions must contain an (expr, True) statement to be used as a default condition. Without one, the generated expression may not evaluate to anything under some condition.')
        lines = []
        for (i, (e, c)) in enumerate(expr.args):
            if i == 0:
                lines.append('if (%s) {' % self._print(c))
            elif i == len(expr.args) - 1 and c == True:
                lines[-1] += ' else {'
            else:
                lines[-1] += ' else if (%s) {' % self._print(c)
            code0 = self._print(e)
            lines.append(code0)
            lines.append('}')
        if self._settings['inline']:
            return ' '.join(lines)
        else:
            return '\n'.join(lines)

    def _print_ITE(self, expr):
        if False:
            while True:
                i = 10
        from sympy.functions import Piecewise
        return self._print(expr.rewrite(Piecewise, deep=False))

    def _print_MatrixBase(self, A):
        if False:
            for i in range(10):
                print('nop')
        if A.cols == 1:
            return '[%s]' % ', '.join((self._print(a) for a in A))
        else:
            raise ValueError('Full Matrix Support in Rust need Crates (https://crates.io/keywords/matrix).')

    def _print_SparseRepMatrix(self, mat):
        if False:
            for i in range(10):
                print('nop')
        return self._print_not_supported(mat)

    def _print_MatrixElement(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return '%s[%s]' % (expr.parent, expr.j + expr.i * expr.parent.shape[1])

    def _print_Symbol(self, expr):
        if False:
            for i in range(10):
                print('nop')
        name = super()._print_Symbol(expr)
        if expr in self._dereference:
            return '(*%s)' % name
        else:
            return name

    def _print_Assignment(self, expr):
        if False:
            i = 10
            return i + 15
        from sympy.tensor.indexed import IndexedBase
        lhs = expr.lhs
        rhs = expr.rhs
        if self._settings['contract'] and (lhs.has(IndexedBase) or rhs.has(IndexedBase)):
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement('%s = %s' % (lhs_code, rhs_code))

    def indent_code(self, code):
        if False:
            return 10
        'Accepts a string of code or a list of code lines'
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)
        tab = '    '
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')
        code = [line.lstrip(' \t') for line in code]
        increase = [int(any(map(line.endswith, inc_token))) for line in code]
        decrease = [int(any(map(line.startswith, dec_token))) for line in code]
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

def rust_code(expr, assign_to=None, **settings):
    if False:
        return 10
    'Converts an expr to a string of Rust code\n\n    Parameters\n    ==========\n\n    expr : Expr\n        A SymPy expression to be converted.\n    assign_to : optional\n        When given, the argument is used as the name of the variable to which\n        the expression is assigned. Can be a string, ``Symbol``,\n        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of\n        line-wrapping, or for expressions that generate multi-line statements.\n    precision : integer, optional\n        The precision for numbers such as pi [default=15].\n    user_functions : dict, optional\n        A dictionary where the keys are string representations of either\n        ``FunctionClass`` or ``UndefinedFunction`` instances and the values\n        are their desired C string representations. Alternatively, the\n        dictionary value can be a list of tuples i.e. [(argument_test,\n        cfunction_string)].  See below for examples.\n    dereference : iterable, optional\n        An iterable of symbols that should be dereferenced in the printed code\n        expression. These would be values passed by address to the function.\n        For example, if ``dereference=[a]``, the resulting code would print\n        ``(*a)`` instead of ``a``.\n    human : bool, optional\n        If True, the result is a single string that may contain some constant\n        declarations for the number symbols. If False, the same information is\n        returned in a tuple of (symbols_to_declare, not_supported_functions,\n        code_text). [default=True].\n    contract: bool, optional\n        If True, ``Indexed`` instances are assumed to obey tensor contraction\n        rules and the corresponding nested loops over indices are generated.\n        Setting contract=False will not generate loops, instead the user is\n        responsible to provide values for the indices in the code.\n        [default=True].\n\n    Examples\n    ========\n\n    >>> from sympy import rust_code, symbols, Rational, sin, ceiling, Abs, Function\n    >>> x, tau = symbols("x, tau")\n    >>> rust_code((2*tau)**Rational(7, 2))\n    \'8*1.4142135623731*tau.powf(7_f64/2.0)\'\n    >>> rust_code(sin(x), assign_to="s")\n    \'s = x.sin();\'\n\n    Simple custom printing can be defined for certain types by passing a\n    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.\n    Alternatively, the dictionary value can be a list of tuples i.e.\n    [(argument_test, cfunction_string)].\n\n    >>> custom_functions = {\n    ...   "ceiling": "CEIL",\n    ...   "Abs": [(lambda x: not x.is_integer, "fabs", 4),\n    ...           (lambda x: x.is_integer, "ABS", 4)],\n    ...   "func": "f"\n    ... }\n    >>> func = Function(\'func\')\n    >>> rust_code(func(Abs(x) + ceiling(x)), user_functions=custom_functions)\n    \'(fabs(x) + x.CEIL()).f()\'\n\n    ``Piecewise`` expressions are converted into conditionals. If an\n    ``assign_to`` variable is provided an if statement is created, otherwise\n    the ternary operator is used. Note that if the ``Piecewise`` lacks a\n    default term, represented by ``(expr, True)`` then an error will be thrown.\n    This is to prevent generating an expression that may not evaluate to\n    anything.\n\n    >>> from sympy import Piecewise\n    >>> expr = Piecewise((x + 1, x > 0), (x, True))\n    >>> print(rust_code(expr, tau))\n    tau = if (x > 0) {\n        x + 1\n    } else {\n        x\n    };\n\n    Support for loops is provided through ``Indexed`` types. With\n    ``contract=True`` these expressions will be turned into loops, whereas\n    ``contract=False`` will just print the assignment expression that should be\n    looped over:\n\n    >>> from sympy import Eq, IndexedBase, Idx\n    >>> len_y = 5\n    >>> y = IndexedBase(\'y\', shape=(len_y,))\n    >>> t = IndexedBase(\'t\', shape=(len_y,))\n    >>> Dy = IndexedBase(\'Dy\', shape=(len_y-1,))\n    >>> i = Idx(\'i\', len_y-1)\n    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))\n    >>> rust_code(e.rhs, assign_to=e.lhs, contract=False)\n    \'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);\'\n\n    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions\n    must be provided to ``assign_to``. Note that any expression that can be\n    generated normally can also exist inside a Matrix:\n\n    >>> from sympy import Matrix, MatrixSymbol\n    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])\n    >>> A = MatrixSymbol(\'A\', 3, 1)\n    >>> print(rust_code(mat, A))\n    A = [x.powi(2), if (x > 0) {\n        x + 1\n    } else {\n        x\n    }, x.sin()];\n    '
    return RustCodePrinter(settings).doprint(expr, assign_to)

def print_rust_code(expr, **settings):
    if False:
        return 10
    'Prints Rust representation of the given expression.'
    print(rust_code(expr, **settings))