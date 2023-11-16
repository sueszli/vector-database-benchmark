"""
Javascript code printer

The JavascriptCodePrinter converts single SymPy expressions into single
Javascript expressions, using the functions defined in the Javascript
Math object where possible.

"""
from __future__ import annotations
from typing import Any
from sympy.core import S
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
known_functions = {'Abs': 'Math.abs', 'acos': 'Math.acos', 'acosh': 'Math.acosh', 'asin': 'Math.asin', 'asinh': 'Math.asinh', 'atan': 'Math.atan', 'atan2': 'Math.atan2', 'atanh': 'Math.atanh', 'ceiling': 'Math.ceil', 'cos': 'Math.cos', 'cosh': 'Math.cosh', 'exp': 'Math.exp', 'floor': 'Math.floor', 'log': 'Math.log', 'Max': 'Math.max', 'Min': 'Math.min', 'sign': 'Math.sign', 'sin': 'Math.sin', 'sinh': 'Math.sinh', 'tan': 'Math.tan', 'tanh': 'Math.tanh'}

class JavascriptCodePrinter(CodePrinter):
    """"A Printer to convert Python expressions to strings of JavaScript code
    """
    printmethod = '_javascript'
    language = 'JavaScript'
    _default_settings: dict[str, Any] = {'order': None, 'full_prec': 'auto', 'precision': 17, 'user_functions': {}, 'human': True, 'allow_unknown_functions': False, 'contract': True}

    def __init__(self, settings={}):
        if False:
            for i in range(10):
                print('nop')
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)

    def _rate_index_position(self, p):
        if False:
            for i in range(10):
                print('nop')
        return p * 5

    def _get_statement(self, codestring):
        if False:
            return 10
        return '%s;' % codestring

    def _get_comment(self, text):
        if False:
            return 10
        return '// {}'.format(text)

    def _declare_number_const(self, name, value):
        if False:
            print('Hello World!')
        return 'var {} = {};'.format(name, value.evalf(self._settings['precision']))

    def _format_code(self, lines):
        if False:
            i = 10
            return i + 15
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        if False:
            return 10
        (rows, cols) = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    def _get_loop_opening_ending(self, indices):
        if False:
            return 10
        open_lines = []
        close_lines = []
        loopstart = 'for (var %(varble)s=%(start)s; %(varble)s<%(end)s; %(varble)s++){'
        for i in indices:
            open_lines.append(loopstart % {'varble': self._print(i.label), 'start': self._print(i.lower), 'end': self._print(i.upper + 1)})
            close_lines.append('}')
        return (open_lines, close_lines)

    def _print_Pow(self, expr):
        if False:
            while True:
                i = 10
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '1/%s' % self.parenthesize(expr.base, PREC)
        elif equal_valued(expr.exp, 0.5):
            return 'Math.sqrt(%s)' % self._print(expr.base)
        elif expr.exp == S.One / 3:
            return 'Math.cbrt(%s)' % self._print(expr.base)
        else:
            return 'Math.pow(%s, %s)' % (self._print(expr.base), self._print(expr.exp))

    def _print_Rational(self, expr):
        if False:
            for i in range(10):
                print('nop')
        (p, q) = (int(expr.p), int(expr.q))
        return '%d/%d' % (p, q)

    def _print_Mod(self, expr):
        if False:
            for i in range(10):
                print('nop')
        (num, den) = expr.args
        PREC = precedence(expr)
        (snum, sden) = [self.parenthesize(arg, PREC) for arg in expr.args]
        if num.is_nonnegative and den.is_nonnegative or (num.is_nonpositive and den.is_nonpositive):
            return f'{snum} % {sden}'
        return f'(({snum} % {sden}) + {sden}) % {sden}'

    def _print_Relational(self, expr):
        if False:
            print('Hello World!')
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return '{} {} {}'.format(lhs_code, op, rhs_code)

    def _print_Indexed(self, expr):
        if False:
            while True:
                i = 10
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
        return self._print(expr.label)

    def _print_Exp1(self, expr):
        if False:
            i = 10
            return i + 15
        return 'Math.E'

    def _print_Pi(self, expr):
        if False:
            i = 10
            return i + 15
        return 'Math.PI'

    def _print_Infinity(self, expr):
        if False:
            while True:
                i = 10
        return 'Number.POSITIVE_INFINITY'

    def _print_NegativeInfinity(self, expr):
        if False:
            return 10
        return 'Number.NEGATIVE_INFINITY'

    def _print_Piecewise(self, expr):
        if False:
            for i in range(10):
                print('nop')
        from sympy.codegen.ast import Assignment
        if expr.args[-1].cond != True:
            raise ValueError('All Piecewise expressions must contain an (expr, True) statement to be used as a default condition. Without one, the generated expression may not evaluate to anything under some condition.')
        lines = []
        if expr.has(Assignment):
            for (i, (e, c)) in enumerate(expr.args):
                if i == 0:
                    lines.append('if (%s) {' % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append('else {')
                else:
                    lines.append('else if (%s) {' % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                lines.append('}')
            return '\n'.join(lines)
        else:
            ecpairs = ['((%s) ? (\n%s\n)\n' % (self._print(c), self._print(e)) for (e, c) in expr.args[:-1]]
            last_line = ': (\n%s\n)' % self._print(expr.args[-1].expr)
            return ': '.join(ecpairs) + last_line + ' '.join([')' * len(ecpairs)])

    def _print_MatrixElement(self, expr):
        if False:
            while True:
                i = 10
        return '{}[{}]'.format(self.parenthesize(expr.parent, PRECEDENCE['Atom'], strict=True), expr.j + expr.i * expr.parent.shape[1])

    def indent_code(self, code):
        if False:
            return 10
        'Accepts a string of code or a list of code lines'
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)
        tab = '   '
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

def jscode(expr, assign_to=None, **settings):
    if False:
        return 10
    'Converts an expr to a string of javascript code\n\n    Parameters\n    ==========\n\n    expr : Expr\n        A SymPy expression to be converted.\n    assign_to : optional\n        When given, the argument is used as the name of the variable to which\n        the expression is assigned. Can be a string, ``Symbol``,\n        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of\n        line-wrapping, or for expressions that generate multi-line statements.\n    precision : integer, optional\n        The precision for numbers such as pi [default=15].\n    user_functions : dict, optional\n        A dictionary where keys are ``FunctionClass`` instances and values are\n        their string representations. Alternatively, the dictionary value can\n        be a list of tuples i.e. [(argument_test, js_function_string)]. See\n        below for examples.\n    human : bool, optional\n        If True, the result is a single string that may contain some constant\n        declarations for the number symbols. If False, the same information is\n        returned in a tuple of (symbols_to_declare, not_supported_functions,\n        code_text). [default=True].\n    contract: bool, optional\n        If True, ``Indexed`` instances are assumed to obey tensor contraction\n        rules and the corresponding nested loops over indices are generated.\n        Setting contract=False will not generate loops, instead the user is\n        responsible to provide values for the indices in the code.\n        [default=True].\n\n    Examples\n    ========\n\n    >>> from sympy import jscode, symbols, Rational, sin, ceiling, Abs\n    >>> x, tau = symbols("x, tau")\n    >>> jscode((2*tau)**Rational(7, 2))\n    \'8*Math.sqrt(2)*Math.pow(tau, 7/2)\'\n    >>> jscode(sin(x), assign_to="s")\n    \'s = Math.sin(x);\'\n\n    Custom printing can be defined for certain types by passing a dictionary of\n    "type" : "function" to the ``user_functions`` kwarg. Alternatively, the\n    dictionary value can be a list of tuples i.e. [(argument_test,\n    js_function_string)].\n\n    >>> custom_functions = {\n    ...   "ceiling": "CEIL",\n    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),\n    ...           (lambda x: x.is_integer, "ABS")]\n    ... }\n    >>> jscode(Abs(x) + ceiling(x), user_functions=custom_functions)\n    \'fabs(x) + CEIL(x)\'\n\n    ``Piecewise`` expressions are converted into conditionals. If an\n    ``assign_to`` variable is provided an if statement is created, otherwise\n    the ternary operator is used. Note that if the ``Piecewise`` lacks a\n    default term, represented by ``(expr, True)`` then an error will be thrown.\n    This is to prevent generating an expression that may not evaluate to\n    anything.\n\n    >>> from sympy import Piecewise\n    >>> expr = Piecewise((x + 1, x > 0), (x, True))\n    >>> print(jscode(expr, tau))\n    if (x > 0) {\n       tau = x + 1;\n    }\n    else {\n       tau = x;\n    }\n\n    Support for loops is provided through ``Indexed`` types. With\n    ``contract=True`` these expressions will be turned into loops, whereas\n    ``contract=False`` will just print the assignment expression that should be\n    looped over:\n\n    >>> from sympy import Eq, IndexedBase, Idx\n    >>> len_y = 5\n    >>> y = IndexedBase(\'y\', shape=(len_y,))\n    >>> t = IndexedBase(\'t\', shape=(len_y,))\n    >>> Dy = IndexedBase(\'Dy\', shape=(len_y-1,))\n    >>> i = Idx(\'i\', len_y-1)\n    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))\n    >>> jscode(e.rhs, assign_to=e.lhs, contract=False)\n    \'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);\'\n\n    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions\n    must be provided to ``assign_to``. Note that any expression that can be\n    generated normally can also exist inside a Matrix:\n\n    >>> from sympy import Matrix, MatrixSymbol\n    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])\n    >>> A = MatrixSymbol(\'A\', 3, 1)\n    >>> print(jscode(mat, A))\n    A[0] = Math.pow(x, 2);\n    if (x > 0) {\n       A[1] = x + 1;\n    }\n    else {\n       A[1] = x;\n    }\n    A[2] = Math.sin(x);\n    '
    return JavascriptCodePrinter(settings).doprint(expr, assign_to)

def print_jscode(expr, **settings):
    if False:
        while True:
            i = 10
    'Prints the Javascript representation of the given expression.\n\n       See jscode for the meaning of the optional arguments.\n    '
    print(jscode(expr, **settings))