"""
R code printer

The RCodePrinter converts single SymPy expressions into single R expressions,
using the functions defined in math.h where possible.



"""
from __future__ import annotations
from typing import Any
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range
known_functions = {'Abs': 'abs', 'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'asin': 'asin', 'acos': 'acos', 'atan': 'atan', 'atan2': 'atan2', 'exp': 'exp', 'log': 'log', 'erf': 'erf', 'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh', 'asinh': 'asinh', 'acosh': 'acosh', 'atanh': 'atanh', 'floor': 'floor', 'ceiling': 'ceiling', 'sign': 'sign', 'Max': 'max', 'Min': 'min', 'factorial': 'factorial', 'gamma': 'gamma', 'digamma': 'digamma', 'trigamma': 'trigamma', 'beta': 'beta', 'sqrt': 'sqrt'}
reserved_words = ['if', 'else', 'repeat', 'while', 'function', 'for', 'in', 'next', 'break', 'TRUE', 'FALSE', 'NULL', 'Inf', 'NaN', 'NA', 'NA_integer_', 'NA_real_', 'NA_complex_', 'NA_character_', 'volatile']

class RCodePrinter(CodePrinter):
    """A printer to convert SymPy expressions to strings of R code"""
    printmethod = '_rcode'
    language = 'R'
    _default_settings: dict[str, Any] = {'order': None, 'full_prec': 'auto', 'precision': 15, 'user_functions': {}, 'human': True, 'contract': True, 'dereference': set(), 'error_on_reserved': False, 'reserved_word_suffix': '_'}
    _operators = {'and': '&', 'or': '|', 'not': '!'}
    _relationals: dict[str, str] = {}

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
            while True:
                i = 10
        return p * 5

    def _get_statement(self, codestring):
        if False:
            while True:
                i = 10
        return '%s;' % codestring

    def _get_comment(self, text):
        if False:
            while True:
                i = 10
        return '// {}'.format(text)

    def _declare_number_const(self, name, value):
        if False:
            while True:
                i = 10
        return '{} = {};'.format(name, value)

    def _format_code(self, lines):
        if False:
            return 10
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        if False:
            print('Hello World!')
        (rows, cols) = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    def _get_loop_opening_ending(self, indices):
        if False:
            print('Hello World!')
        'Returns a tuple (open_lines, close_lines) containing lists of codelines\n        '
        open_lines = []
        close_lines = []
        loopstart = 'for (%(var)s in %(start)s:%(end)s){'
        for i in indices:
            open_lines.append(loopstart % {'var': self._print(i.label), 'start': self._print(i.lower + 1), 'end': self._print(i.upper + 1)})
            close_lines.append('}')
        return (open_lines, close_lines)

    def _print_Pow(self, expr):
        if False:
            for i in range(10):
                print('nop')
        if 'Pow' in self.known_functions:
            return self._print_Function(expr)
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '1.0/%s' % self.parenthesize(expr.base, PREC)
        elif equal_valued(expr.exp, 0.5):
            return 'sqrt(%s)' % self._print(expr.base)
        else:
            return '%s^%s' % (self.parenthesize(expr.base, PREC), self.parenthesize(expr.exp, PREC))

    def _print_Rational(self, expr):
        if False:
            for i in range(10):
                print('nop')
        (p, q) = (int(expr.p), int(expr.q))
        return '%d.0/%d.0' % (p, q)

    def _print_Indexed(self, expr):
        if False:
            i = 10
            return i + 15
        inds = [self._print(i) for i in expr.indices]
        return '%s[%s]' % (self._print(expr.base.label), ', '.join(inds))

    def _print_Idx(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return self._print(expr.label)

    def _print_Exp1(self, expr):
        if False:
            while True:
                i = 10
        return 'exp(1)'

    def _print_Pi(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return 'pi'

    def _print_Infinity(self, expr):
        if False:
            while True:
                i = 10
        return 'Inf'

    def _print_NegativeInfinity(self, expr):
        if False:
            i = 10
            return i + 15
        return '-Inf'

    def _print_Assignment(self, expr):
        if False:
            i = 10
            return i + 15
        from sympy.codegen.ast import Assignment
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.tensor.indexed import IndexedBase
        lhs = expr.lhs
        rhs = expr.rhs
        if isinstance(lhs, MatrixSymbol):
            lines = []
            for (i, j) in self._traverse_matrix_indices(lhs):
                temp = Assignment(lhs[i, j], rhs[i, j])
                code0 = self._print(temp)
                lines.append(code0)
            return '\n'.join(lines)
        elif self._settings['contract'] and (lhs.has(IndexedBase) or rhs.has(IndexedBase)):
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement('%s = %s' % (lhs_code, rhs_code))

    def _print_Piecewise(self, expr):
        if False:
            return 10
        if expr.args[-1].cond == True:
            last_line = '%s' % self._print(expr.args[-1].expr)
        else:
            last_line = 'ifelse(%s,%s,NA)' % (self._print(expr.args[-1].cond), self._print(expr.args[-1].expr))
        code = last_line
        for (e, c) in reversed(expr.args[:-1]):
            code = 'ifelse(%s,%s,' % (self._print(c), self._print(e)) + code + ')'
        return code

    def _print_ITE(self, expr):
        if False:
            while True:
                i = 10
        from sympy.functions import Piecewise
        return self._print(expr.rewrite(Piecewise))

    def _print_MatrixElement(self, expr):
        if False:
            i = 10
            return i + 15
        return '{}[{}]'.format(self.parenthesize(expr.parent, PRECEDENCE['Atom'], strict=True), expr.j + expr.i * expr.parent.shape[1])

    def _print_Symbol(self, expr):
        if False:
            i = 10
            return i + 15
        name = super()._print_Symbol(expr)
        if expr in self._dereference:
            return '(*{})'.format(name)
        else:
            return name

    def _print_Relational(self, expr):
        if False:
            for i in range(10):
                print('nop')
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return '{} {} {}'.format(lhs_code, op, rhs_code)

    def _print_AugmentedAssignment(self, expr):
        if False:
            i = 10
            return i + 15
        lhs_code = self._print(expr.lhs)
        op = expr.op
        rhs_code = self._print(expr.rhs)
        return '{} {} {};'.format(lhs_code, op, rhs_code)

    def _print_For(self, expr):
        if False:
            print('Hello World!')
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            (start, stop, step) = expr.iterable.args
        else:
            raise NotImplementedError('Only iterable currently supported is Range')
        body = self._print(expr.body)
        return 'for({target} in seq(from={start}, to={stop}, by={step}){{\n{body}\n}}'.format(target=target, start=start, stop=stop - 1, step=step, body=body)

    def indent_code(self, code):
        if False:
            for i in range(10):
                print('nop')
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

def rcode(expr, assign_to=None, **settings):
    if False:
        while True:
            i = 10
    'Converts an expr to a string of r code\n\n    Parameters\n    ==========\n\n    expr : Expr\n        A SymPy expression to be converted.\n    assign_to : optional\n        When given, the argument is used as the name of the variable to which\n        the expression is assigned. Can be a string, ``Symbol``,\n        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of\n        line-wrapping, or for expressions that generate multi-line statements.\n    precision : integer, optional\n        The precision for numbers such as pi [default=15].\n    user_functions : dict, optional\n        A dictionary where the keys are string representations of either\n        ``FunctionClass`` or ``UndefinedFunction`` instances and the values\n        are their desired R string representations. Alternatively, the\n        dictionary value can be a list of tuples i.e. [(argument_test,\n        rfunction_string)] or [(argument_test, rfunction_formater)]. See below\n        for examples.\n    human : bool, optional\n        If True, the result is a single string that may contain some constant\n        declarations for the number symbols. If False, the same information is\n        returned in a tuple of (symbols_to_declare, not_supported_functions,\n        code_text). [default=True].\n    contract: bool, optional\n        If True, ``Indexed`` instances are assumed to obey tensor contraction\n        rules and the corresponding nested loops over indices are generated.\n        Setting contract=False will not generate loops, instead the user is\n        responsible to provide values for the indices in the code.\n        [default=True].\n\n    Examples\n    ========\n\n    >>> from sympy import rcode, symbols, Rational, sin, ceiling, Abs, Function\n    >>> x, tau = symbols("x, tau")\n    >>> rcode((2*tau)**Rational(7, 2))\n    \'8*sqrt(2)*tau^(7.0/2.0)\'\n    >>> rcode(sin(x), assign_to="s")\n    \'s = sin(x);\'\n\n    Simple custom printing can be defined for certain types by passing a\n    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.\n    Alternatively, the dictionary value can be a list of tuples i.e.\n    [(argument_test, cfunction_string)].\n\n    >>> custom_functions = {\n    ...   "ceiling": "CEIL",\n    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),\n    ...           (lambda x: x.is_integer, "ABS")],\n    ...   "func": "f"\n    ... }\n    >>> func = Function(\'func\')\n    >>> rcode(func(Abs(x) + ceiling(x)), user_functions=custom_functions)\n    \'f(fabs(x) + CEIL(x))\'\n\n    or if the R-function takes a subset of the original arguments:\n\n    >>> rcode(2**x + 3**x, user_functions={\'Pow\': [\n    ...   (lambda b, e: b == 2, lambda b, e: \'exp2(%s)\' % e),\n    ...   (lambda b, e: b != 2, \'pow\')]})\n    \'exp2(x) + pow(3, x)\'\n\n    ``Piecewise`` expressions are converted into conditionals. If an\n    ``assign_to`` variable is provided an if statement is created, otherwise\n    the ternary operator is used. Note that if the ``Piecewise`` lacks a\n    default term, represented by ``(expr, True)`` then an error will be thrown.\n    This is to prevent generating an expression that may not evaluate to\n    anything.\n\n    >>> from sympy import Piecewise\n    >>> expr = Piecewise((x + 1, x > 0), (x, True))\n    >>> print(rcode(expr, assign_to=tau))\n    tau = ifelse(x > 0,x + 1,x);\n\n    Support for loops is provided through ``Indexed`` types. With\n    ``contract=True`` these expressions will be turned into loops, whereas\n    ``contract=False`` will just print the assignment expression that should be\n    looped over:\n\n    >>> from sympy import Eq, IndexedBase, Idx\n    >>> len_y = 5\n    >>> y = IndexedBase(\'y\', shape=(len_y,))\n    >>> t = IndexedBase(\'t\', shape=(len_y,))\n    >>> Dy = IndexedBase(\'Dy\', shape=(len_y-1,))\n    >>> i = Idx(\'i\', len_y-1)\n    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))\n    >>> rcode(e.rhs, assign_to=e.lhs, contract=False)\n    \'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);\'\n\n    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions\n    must be provided to ``assign_to``. Note that any expression that can be\n    generated normally can also exist inside a Matrix:\n\n    >>> from sympy import Matrix, MatrixSymbol\n    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])\n    >>> A = MatrixSymbol(\'A\', 3, 1)\n    >>> print(rcode(mat, A))\n    A[0] = x^2;\n    A[1] = ifelse(x > 0,x + 1,x);\n    A[2] = sin(x);\n\n    '
    return RCodePrinter(settings).doprint(expr, assign_to)

def print_rcode(expr, **settings):
    if False:
        for i in range(10):
            print('nop')
    'Prints R representation of the given expression.'
    print(rcode(expr, **settings))