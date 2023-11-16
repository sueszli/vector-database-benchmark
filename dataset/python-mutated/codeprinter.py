from __future__ import annotations
from typing import Any
from functools import wraps
from sympy.core import Add, Mul, Pow, S, sympify, Float
from sympy.core.basic import Basic
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import Lambda
from sympy.core.mul import _keep_coeff
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import re
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE

class requires:
    """ Decorator for registering requirements on print methods. """

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        self._req = kwargs

    def __call__(self, method):
        if False:
            for i in range(10):
                print('nop')

        def _method_wrapper(self_, *args, **kwargs):
            if False:
                return 10
            for (k, v) in self._req.items():
                getattr(self_, k).update(v)
            return method(self_, *args, **kwargs)
        return wraps(method)(_method_wrapper)

class AssignmentError(Exception):
    """
    Raised if an assignment variable for a loop is missing.
    """
    pass

def _convert_python_lists(arg):
    if False:
        while True:
            i = 10
    if isinstance(arg, list):
        from sympy.codegen.abstract_nodes import List
        return List(*(_convert_python_lists(e) for e in arg))
    elif isinstance(arg, tuple):
        return tuple((_convert_python_lists(e) for e in arg))
    else:
        return arg

class CodePrinter(StrPrinter):
    """
    The base class for code-printing subclasses.
    """
    _operators = {'and': '&&', 'or': '||', 'not': '!'}
    _default_settings: dict[str, Any] = {'order': None, 'full_prec': 'auto', 'error_on_reserved': False, 'reserved_word_suffix': '_', 'human': True, 'inline': False, 'allow_unknown_functions': False}
    _rewriteable_functions = {'cot': ('tan', []), 'csc': ('sin', []), 'sec': ('cos', []), 'acot': ('atan', []), 'acsc': ('asin', []), 'asec': ('acos', []), 'coth': ('exp', []), 'csch': ('exp', []), 'sech': ('exp', []), 'acoth': ('log', []), 'acsch': ('log', []), 'asech': ('log', []), 'catalan': ('gamma', []), 'fibonacci': ('sqrt', []), 'lucas': ('sqrt', []), 'beta': ('gamma', []), 'sinc': ('sin', ['Piecewise']), 'Mod': ('floor', []), 'factorial': ('gamma', []), 'factorial2': ('gamma', ['Piecewise']), 'subfactorial': ('uppergamma', []), 'RisingFactorial': ('gamma', ['Piecewise']), 'FallingFactorial': ('gamma', ['Piecewise']), 'binomial': ('gamma', []), 'frac': ('floor', []), 'Max': ('Piecewise', []), 'Min': ('Piecewise', []), 'Heaviside': ('Piecewise', []), 'erf2': ('erf', []), 'erfc': ('erf', []), 'Li': ('li', []), 'Ei': ('li', []), 'dirichlet_eta': ('zeta', []), 'riemann_xi': ('zeta', ['gamma']), 'SingularityFunction': ('Piecewise', [])}

    def __init__(self, settings=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(settings=settings)
        if not hasattr(self, 'reserved_words'):
            self.reserved_words = set()

    def _handle_UnevaluatedExpr(self, expr):
        if False:
            while True:
                i = 10
        return expr.replace(re, lambda arg: arg if isinstance(arg, UnevaluatedExpr) and arg.args[0].is_real else re(arg))

    def doprint(self, expr, assign_to=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Print the expression as code.\n\n        Parameters\n        ----------\n        expr : Expression\n            The expression to be printed.\n\n        assign_to : Symbol, string, MatrixSymbol, list of strings or Symbols (optional)\n            If provided, the printed code will set the expression to a variable or multiple variables\n            with the name or names given in ``assign_to``.\n        '
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.codegen.ast import CodeBlock, Assignment

        def _handle_assign_to(expr, assign_to):
            if False:
                while True:
                    i = 10
            if assign_to is None:
                return sympify(expr)
            if isinstance(assign_to, (list, tuple)):
                if len(expr) != len(assign_to):
                    raise ValueError('Failed to assign an expression of length {} to {} variables'.format(len(expr), len(assign_to)))
                return CodeBlock(*[_handle_assign_to(lhs, rhs) for (lhs, rhs) in zip(expr, assign_to)])
            if isinstance(assign_to, str):
                if expr.is_Matrix:
                    assign_to = MatrixSymbol(assign_to, *expr.shape)
                else:
                    assign_to = Symbol(assign_to)
            elif not isinstance(assign_to, Basic):
                raise TypeError('{} cannot assign to object of type {}'.format(type(self).__name__, type(assign_to)))
            return Assignment(assign_to, expr)
        expr = _convert_python_lists(expr)
        expr = _handle_assign_to(expr, assign_to)
        expr = self._handle_UnevaluatedExpr(expr)
        self._not_supported = set()
        self._number_symbols = set()
        lines = self._print(expr).splitlines()
        if self._settings['human']:
            frontlines = []
            if self._not_supported:
                frontlines.append(self._get_comment('Not supported in {}:'.format(self.language)))
                for expr in sorted(self._not_supported, key=str):
                    frontlines.append(self._get_comment(type(expr).__name__))
            for (name, value) in sorted(self._number_symbols, key=str):
                frontlines.append(self._declare_number_const(name, value))
            lines = frontlines + lines
            lines = self._format_code(lines)
            result = '\n'.join(lines)
        else:
            lines = self._format_code(lines)
            num_syms = {(k, self._print(v)) for (k, v) in self._number_symbols}
            result = (num_syms, self._not_supported, '\n'.join(lines))
        self._not_supported = set()
        self._number_symbols = set()
        return result

    def _doprint_loops(self, expr, assign_to=None):
        if False:
            i = 10
            return i + 15
        if self._settings.get('contract', True):
            from sympy.tensor import get_contraction_structure
            indices = self._get_expression_indices(expr, assign_to)
            dummies = get_contraction_structure(expr)
        else:
            indices = []
            dummies = {None: (expr,)}
        (openloop, closeloop) = self._get_loop_opening_ending(indices)
        if None in dummies:
            text = StrPrinter.doprint(self, Add(*dummies[None]))
        else:
            text = StrPrinter.doprint(self, 0)
        lhs_printed = self._print(assign_to)
        lines = []
        if text != lhs_printed:
            lines.extend(openloop)
            if assign_to is not None:
                text = self._get_statement('%s = %s' % (lhs_printed, text))
            lines.append(text)
            lines.extend(closeloop)
        for d in dummies:
            if isinstance(d, tuple):
                indices = self._sort_optimized(d, expr)
                (openloop_d, closeloop_d) = self._get_loop_opening_ending(indices)
                for term in dummies[d]:
                    if term in dummies and (not [list(f.keys()) for f in dummies[term]] == [[None] for f in dummies[term]]):
                        raise NotImplementedError('FIXME: no support for contractions in factor yet')
                    else:
                        if assign_to is None:
                            raise AssignmentError('need assignment variable for loops')
                        if term.has(assign_to):
                            raise ValueError('FIXME: lhs present in rhs,                                this is undefined in CodePrinter')
                        lines.extend(openloop)
                        lines.extend(openloop_d)
                        text = '%s = %s' % (lhs_printed, StrPrinter.doprint(self, assign_to + term))
                        lines.append(self._get_statement(text))
                        lines.extend(closeloop_d)
                        lines.extend(closeloop)
        return '\n'.join(lines)

    def _get_expression_indices(self, expr, assign_to):
        if False:
            while True:
                i = 10
        from sympy.tensor import get_indices
        (rinds, junk) = get_indices(expr)
        (linds, junk) = get_indices(assign_to)
        if linds and (not rinds):
            rinds = linds
        if rinds != linds:
            raise ValueError('lhs indices must match non-dummy rhs indices in %s' % expr)
        return self._sort_optimized(rinds, assign_to)

    def _sort_optimized(self, indices, expr):
        if False:
            print('Hello World!')
        from sympy.tensor.indexed import Indexed
        if not indices:
            return []
        score_table = {}
        for i in indices:
            score_table[i] = 0
        arrays = expr.atoms(Indexed)
        for arr in arrays:
            for (p, ind) in enumerate(arr.indices):
                try:
                    score_table[ind] += self._rate_index_position(p)
                except KeyError:
                    pass
        return sorted(indices, key=lambda x: score_table[x])

    def _rate_index_position(self, p):
        if False:
            for i in range(10):
                print('nop')
        'function to calculate score based on position among indices\n\n        This method is used to sort loops in an optimized order, see\n        CodePrinter._sort_optimized()\n        '
        raise NotImplementedError('This function must be implemented by subclass of CodePrinter.')

    def _get_statement(self, codestring):
        if False:
            for i in range(10):
                print('nop')
        'Formats a codestring with the proper line ending.'
        raise NotImplementedError('This function must be implemented by subclass of CodePrinter.')

    def _get_comment(self, text):
        if False:
            i = 10
            return i + 15
        'Formats a text string as a comment.'
        raise NotImplementedError('This function must be implemented by subclass of CodePrinter.')

    def _declare_number_const(self, name, value):
        if False:
            print('Hello World!')
        'Declare a numeric constant at the top of a function'
        raise NotImplementedError('This function must be implemented by subclass of CodePrinter.')

    def _format_code(self, lines):
        if False:
            i = 10
            return i + 15
        'Take in a list of lines of code, and format them accordingly.\n\n        This may include indenting, wrapping long lines, etc...'
        raise NotImplementedError('This function must be implemented by subclass of CodePrinter.')

    def _get_loop_opening_ending(self, indices):
        if False:
            i = 10
            return i + 15
        'Returns a tuple (open_lines, close_lines) containing lists\n        of codelines'
        raise NotImplementedError('This function must be implemented by subclass of CodePrinter.')

    def _print_Dummy(self, expr):
        if False:
            return 10
        if expr.name.startswith('Dummy_'):
            return '_' + expr.name
        else:
            return '%s_%d' % (expr.name, expr.dummy_index)

    def _print_CodeBlock(self, expr):
        if False:
            while True:
                i = 10
        return '\n'.join([self._print(i) for i in expr.args])

    def _print_String(self, string):
        if False:
            print('Hello World!')
        return str(string)

    def _print_QuotedString(self, arg):
        if False:
            while True:
                i = 10
        return '"%s"' % arg.text

    def _print_Comment(self, string):
        if False:
            return 10
        return self._get_comment(str(string))

    def _print_Assignment(self, expr):
        if False:
            for i in range(10):
                print('nop')
        from sympy.codegen.ast import Assignment
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.tensor.indexed import IndexedBase
        lhs = expr.lhs
        rhs = expr.rhs
        if isinstance(expr.rhs, Piecewise):
            expressions = []
            conditions = []
            for (e, c) in rhs.args:
                expressions.append(Assignment(lhs, e))
                conditions.append(c)
            temp = Piecewise(*zip(expressions, conditions))
            return self._print(temp)
        elif isinstance(lhs, MatrixSymbol):
            lines = []
            for (i, j) in self._traverse_matrix_indices(lhs):
                temp = Assignment(lhs[i, j], rhs[i, j])
                code0 = self._print(temp)
                lines.append(code0)
            return '\n'.join(lines)
        elif self._settings.get('contract', False) and (lhs.has(IndexedBase) or rhs.has(IndexedBase)):
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement('%s = %s' % (lhs_code, rhs_code))

    def _print_AugmentedAssignment(self, expr):
        if False:
            for i in range(10):
                print('nop')
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        return self._get_statement('{} {} {}'.format(*(self._print(arg) for arg in [lhs_code, expr.op, rhs_code])))

    def _print_FunctionCall(self, expr):
        if False:
            print('Hello World!')
        return '%s(%s)' % (expr.name, ', '.join((self._print(arg) for arg in expr.function_args)))

    def _print_Variable(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return self._print(expr.symbol)

    def _print_Symbol(self, expr):
        if False:
            i = 10
            return i + 15
        name = super()._print_Symbol(expr)
        if name in self.reserved_words:
            if self._settings['error_on_reserved']:
                msg = 'This expression includes the symbol "{}" which is a reserved keyword in this language.'
                raise ValueError(msg.format(name))
            return name + self._settings['reserved_word_suffix']
        else:
            return name

    def _can_print(self, name):
        if False:
            while True:
                i = 10
        ' Check if function ``name`` is either a known function or has its own\n            printing method. Used to check if rewriting is possible.'
        return name in self.known_functions or getattr(self, '_print_{}'.format(name), False)

    def _print_Function(self, expr):
        if False:
            for i in range(10):
                print('nop')
        if expr.func.__name__ in self.known_functions:
            cond_func = self.known_functions[expr.func.__name__]
            if isinstance(cond_func, str):
                return '%s(%s)' % (cond_func, self.stringify(expr.args, ', '))
            else:
                for (cond, func) in cond_func:
                    if cond(*expr.args):
                        break
                if func is not None:
                    try:
                        return func(*[self.parenthesize(item, 0) for item in expr.args])
                    except TypeError:
                        return '%s(%s)' % (func, self.stringify(expr.args, ', '))
        elif hasattr(expr, '_imp_') and isinstance(expr._imp_, Lambda):
            return self._print(expr._imp_(*expr.args))
        elif expr.func.__name__ in self._rewriteable_functions:
            (target_f, required_fs) = self._rewriteable_functions[expr.func.__name__]
            if self._can_print(target_f) and all((self._can_print(f) for f in required_fs)):
                return '(' + self._print(expr.rewrite(target_f)) + ')'
        if expr.is_Function and self._settings.get('allow_unknown_functions', False):
            return '%s(%s)' % (self._print(expr.func), ', '.join(map(self._print, expr.args)))
        else:
            return self._print_not_supported(expr)
    _print_Expr = _print_Function
    _print_Heaviside = None

    def _print_NumberSymbol(self, expr):
        if False:
            while True:
                i = 10
        if self._settings.get('inline', False):
            return self._print(Float(expr.evalf(self._settings['precision'])))
        else:
            self._number_symbols.add((expr, Float(expr.evalf(self._settings['precision']))))
            return str(expr)

    def _print_Catalan(self, expr):
        if False:
            print('Hello World!')
        return self._print_NumberSymbol(expr)

    def _print_EulerGamma(self, expr):
        if False:
            return 10
        return self._print_NumberSymbol(expr)

    def _print_GoldenRatio(self, expr):
        if False:
            i = 10
            return i + 15
        return self._print_NumberSymbol(expr)

    def _print_TribonacciConstant(self, expr):
        if False:
            while True:
                i = 10
        return self._print_NumberSymbol(expr)

    def _print_Exp1(self, expr):
        if False:
            print('Hello World!')
        return self._print_NumberSymbol(expr)

    def _print_Pi(self, expr):
        if False:
            return 10
        return self._print_NumberSymbol(expr)

    def _print_And(self, expr):
        if False:
            print('Hello World!')
        PREC = precedence(expr)
        return (' %s ' % self._operators['and']).join((self.parenthesize(a, PREC) for a in sorted(expr.args, key=default_sort_key)))

    def _print_Or(self, expr):
        if False:
            return 10
        PREC = precedence(expr)
        return (' %s ' % self._operators['or']).join((self.parenthesize(a, PREC) for a in sorted(expr.args, key=default_sort_key)))

    def _print_Xor(self, expr):
        if False:
            return 10
        if self._operators.get('xor') is None:
            return self._print(expr.to_nnf())
        PREC = precedence(expr)
        return (' %s ' % self._operators['xor']).join((self.parenthesize(a, PREC) for a in expr.args))

    def _print_Equivalent(self, expr):
        if False:
            i = 10
            return i + 15
        if self._operators.get('equivalent') is None:
            return self._print(expr.to_nnf())
        PREC = precedence(expr)
        return (' %s ' % self._operators['equivalent']).join((self.parenthesize(a, PREC) for a in expr.args))

    def _print_Not(self, expr):
        if False:
            for i in range(10):
                print('nop')
        PREC = precedence(expr)
        return self._operators['not'] + self.parenthesize(expr.args[0], PREC)

    def _print_BooleanFunction(self, expr):
        if False:
            return 10
        return self._print(expr.to_nnf())

    def _print_Mul(self, expr):
        if False:
            i = 10
            return i + 15
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
            else:
                a.append(item)
        a = a or [S.One]
        if len(a) == 1 and sign == '-':
            a_str = [self.parenthesize(a[0], 0.5 * (PRECEDENCE['Pow'] + PRECEDENCE['Mul']))]
        else:
            a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = '(%s)' % b_str[b.index(item.base)]
        if not b:
            return sign + '*'.join(a_str)
        elif len(b) == 1:
            return sign + '*'.join(a_str) + '/' + b_str[0]
        else:
            return sign + '*'.join(a_str) + '/(%s)' % '*'.join(b_str)

    def _print_not_supported(self, expr):
        if False:
            i = 10
            return i + 15
        try:
            self._not_supported.add(expr)
        except TypeError:
            pass
        return self.emptyPrinter(expr)
    _print_Basic = _print_not_supported
    _print_ComplexInfinity = _print_not_supported
    _print_Derivative = _print_not_supported
    _print_ExprCondPair = _print_not_supported
    _print_GeometryEntity = _print_not_supported
    _print_Infinity = _print_not_supported
    _print_Integral = _print_not_supported
    _print_Interval = _print_not_supported
    _print_AccumulationBounds = _print_not_supported
    _print_Limit = _print_not_supported
    _print_MatrixBase = _print_not_supported
    _print_DeferredVector = _print_not_supported
    _print_NaN = _print_not_supported
    _print_NegativeInfinity = _print_not_supported
    _print_Order = _print_not_supported
    _print_RootOf = _print_not_supported
    _print_RootsOf = _print_not_supported
    _print_RootSum = _print_not_supported
    _print_Uniform = _print_not_supported
    _print_Unit = _print_not_supported
    _print_Wild = _print_not_supported
    _print_WildFunction = _print_not_supported
    _print_Relational = _print_not_supported

def ccode(expr, assign_to=None, standard='c99', **settings):
    if False:
        return 10
    'Converts an expr to a string of c code\n\n    Parameters\n    ==========\n\n    expr : Expr\n        A SymPy expression to be converted.\n    assign_to : optional\n        When given, the argument is used as the name of the variable to which\n        the expression is assigned. Can be a string, ``Symbol``,\n        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of\n        line-wrapping, or for expressions that generate multi-line statements.\n    standard : str, optional\n        String specifying the standard. If your compiler supports a more modern\n        standard you may set this to \'c99\' to allow the printer to use more math\n        functions. [default=\'c89\'].\n    precision : integer, optional\n        The precision for numbers such as pi [default=17].\n    user_functions : dict, optional\n        A dictionary where the keys are string representations of either\n        ``FunctionClass`` or ``UndefinedFunction`` instances and the values\n        are their desired C string representations. Alternatively, the\n        dictionary value can be a list of tuples i.e. [(argument_test,\n        cfunction_string)] or [(argument_test, cfunction_formater)]. See below\n        for examples.\n    dereference : iterable, optional\n        An iterable of symbols that should be dereferenced in the printed code\n        expression. These would be values passed by address to the function.\n        For example, if ``dereference=[a]``, the resulting code would print\n        ``(*a)`` instead of ``a``.\n    human : bool, optional\n        If True, the result is a single string that may contain some constant\n        declarations for the number symbols. If False, the same information is\n        returned in a tuple of (symbols_to_declare, not_supported_functions,\n        code_text). [default=True].\n    contract: bool, optional\n        If True, ``Indexed`` instances are assumed to obey tensor contraction\n        rules and the corresponding nested loops over indices are generated.\n        Setting contract=False will not generate loops, instead the user is\n        responsible to provide values for the indices in the code.\n        [default=True].\n\n    Examples\n    ========\n\n    >>> from sympy import ccode, symbols, Rational, sin, ceiling, Abs, Function\n    >>> x, tau = symbols("x, tau")\n    >>> expr = (2*tau)**Rational(7, 2)\n    >>> ccode(expr)\n    \'8*M_SQRT2*pow(tau, 7.0/2.0)\'\n    >>> ccode(expr, math_macros={})\n    \'8*sqrt(2)*pow(tau, 7.0/2.0)\'\n    >>> ccode(sin(x), assign_to="s")\n    \'s = sin(x);\'\n    >>> from sympy.codegen.ast import real, float80\n    >>> ccode(expr, type_aliases={real: float80})\n    \'8*M_SQRT2l*powl(tau, 7.0L/2.0L)\'\n\n    Simple custom printing can be defined for certain types by passing a\n    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.\n    Alternatively, the dictionary value can be a list of tuples i.e.\n    [(argument_test, cfunction_string)].\n\n    >>> custom_functions = {\n    ...   "ceiling": "CEIL",\n    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),\n    ...           (lambda x: x.is_integer, "ABS")],\n    ...   "func": "f"\n    ... }\n    >>> func = Function(\'func\')\n    >>> ccode(func(Abs(x) + ceiling(x)), standard=\'C89\', user_functions=custom_functions)\n    \'f(fabs(x) + CEIL(x))\'\n\n    or if the C-function takes a subset of the original arguments:\n\n    >>> ccode(2**x + 3**x, standard=\'C99\', user_functions={\'Pow\': [\n    ...   (lambda b, e: b == 2, lambda b, e: \'exp2(%s)\' % e),\n    ...   (lambda b, e: b != 2, \'pow\')]})\n    \'exp2(x) + pow(3, x)\'\n\n    ``Piecewise`` expressions are converted into conditionals. If an\n    ``assign_to`` variable is provided an if statement is created, otherwise\n    the ternary operator is used. Note that if the ``Piecewise`` lacks a\n    default term, represented by ``(expr, True)`` then an error will be thrown.\n    This is to prevent generating an expression that may not evaluate to\n    anything.\n\n    >>> from sympy import Piecewise\n    >>> expr = Piecewise((x + 1, x > 0), (x, True))\n    >>> print(ccode(expr, tau, standard=\'C89\'))\n    if (x > 0) {\n    tau = x + 1;\n    }\n    else {\n    tau = x;\n    }\n\n    Support for loops is provided through ``Indexed`` types. With\n    ``contract=True`` these expressions will be turned into loops, whereas\n    ``contract=False`` will just print the assignment expression that should be\n    looped over:\n\n    >>> from sympy import Eq, IndexedBase, Idx\n    >>> len_y = 5\n    >>> y = IndexedBase(\'y\', shape=(len_y,))\n    >>> t = IndexedBase(\'t\', shape=(len_y,))\n    >>> Dy = IndexedBase(\'Dy\', shape=(len_y-1,))\n    >>> i = Idx(\'i\', len_y-1)\n    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))\n    >>> ccode(e.rhs, assign_to=e.lhs, contract=False, standard=\'C89\')\n    \'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);\'\n\n    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions\n    must be provided to ``assign_to``. Note that any expression that can be\n    generated normally can also exist inside a Matrix:\n\n    >>> from sympy import Matrix, MatrixSymbol\n    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])\n    >>> A = MatrixSymbol(\'A\', 3, 1)\n    >>> print(ccode(mat, A, standard=\'C89\'))\n    A[0] = pow(x, 2);\n    if (x > 0) {\n       A[1] = x + 1;\n    }\n    else {\n       A[1] = x;\n    }\n    A[2] = sin(x);\n    '
    from sympy.printing.c import c_code_printers
    return c_code_printers[standard.lower()](settings).doprint(expr, assign_to)

def print_ccode(expr, **settings):
    if False:
        for i in range(10):
            print('nop')
    'Prints C representation of the given expression.'
    print(ccode(expr, **settings))

def fcode(expr, assign_to=None, **settings):
    if False:
        return 10
    'Converts an expr to a string of fortran code\n\n    Parameters\n    ==========\n\n    expr : Expr\n        A SymPy expression to be converted.\n    assign_to : optional\n        When given, the argument is used as the name of the variable to which\n        the expression is assigned. Can be a string, ``Symbol``,\n        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of\n        line-wrapping, or for expressions that generate multi-line statements.\n    precision : integer, optional\n        DEPRECATED. Use type_mappings instead. The precision for numbers such\n        as pi [default=17].\n    user_functions : dict, optional\n        A dictionary where keys are ``FunctionClass`` instances and values are\n        their string representations. Alternatively, the dictionary value can\n        be a list of tuples i.e. [(argument_test, cfunction_string)]. See below\n        for examples.\n    human : bool, optional\n        If True, the result is a single string that may contain some constant\n        declarations for the number symbols. If False, the same information is\n        returned in a tuple of (symbols_to_declare, not_supported_functions,\n        code_text). [default=True].\n    contract: bool, optional\n        If True, ``Indexed`` instances are assumed to obey tensor contraction\n        rules and the corresponding nested loops over indices are generated.\n        Setting contract=False will not generate loops, instead the user is\n        responsible to provide values for the indices in the code.\n        [default=True].\n    source_format : optional\n        The source format can be either \'fixed\' or \'free\'. [default=\'fixed\']\n    standard : integer, optional\n        The Fortran standard to be followed. This is specified as an integer.\n        Acceptable standards are 66, 77, 90, 95, 2003, and 2008. Default is 77.\n        Note that currently the only distinction internally is between\n        standards before 95, and those 95 and after. This may change later as\n        more features are added.\n    name_mangling : bool, optional\n        If True, then the variables that would become identical in\n        case-insensitive Fortran are mangled by appending different number\n        of ``_`` at the end. If False, SymPy Will not interfere with naming of\n        variables. [default=True]\n\n    Examples\n    ========\n\n    >>> from sympy import fcode, symbols, Rational, sin, ceiling, floor\n    >>> x, tau = symbols("x, tau")\n    >>> fcode((2*tau)**Rational(7, 2))\n    \'      8*sqrt(2.0d0)*tau**(7.0d0/2.0d0)\'\n    >>> fcode(sin(x), assign_to="s")\n    \'      s = sin(x)\'\n\n    Custom printing can be defined for certain types by passing a dictionary of\n    "type" : "function" to the ``user_functions`` kwarg. Alternatively, the\n    dictionary value can be a list of tuples i.e. [(argument_test,\n    cfunction_string)].\n\n    >>> custom_functions = {\n    ...   "ceiling": "CEIL",\n    ...   "floor": [(lambda x: not x.is_integer, "FLOOR1"),\n    ...             (lambda x: x.is_integer, "FLOOR2")]\n    ... }\n    >>> fcode(floor(x) + ceiling(x), user_functions=custom_functions)\n    \'      CEIL(x) + FLOOR1(x)\'\n\n    ``Piecewise`` expressions are converted into conditionals. If an\n    ``assign_to`` variable is provided an if statement is created, otherwise\n    the ternary operator is used. Note that if the ``Piecewise`` lacks a\n    default term, represented by ``(expr, True)`` then an error will be thrown.\n    This is to prevent generating an expression that may not evaluate to\n    anything.\n\n    >>> from sympy import Piecewise\n    >>> expr = Piecewise((x + 1, x > 0), (x, True))\n    >>> print(fcode(expr, tau))\n          if (x > 0) then\n             tau = x + 1\n          else\n             tau = x\n          end if\n\n    Support for loops is provided through ``Indexed`` types. With\n    ``contract=True`` these expressions will be turned into loops, whereas\n    ``contract=False`` will just print the assignment expression that should be\n    looped over:\n\n    >>> from sympy import Eq, IndexedBase, Idx\n    >>> len_y = 5\n    >>> y = IndexedBase(\'y\', shape=(len_y,))\n    >>> t = IndexedBase(\'t\', shape=(len_y,))\n    >>> Dy = IndexedBase(\'Dy\', shape=(len_y-1,))\n    >>> i = Idx(\'i\', len_y-1)\n    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))\n    >>> fcode(e.rhs, assign_to=e.lhs, contract=False)\n    \'      Dy(i) = (y(i + 1) - y(i))/(t(i + 1) - t(i))\'\n\n    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions\n    must be provided to ``assign_to``. Note that any expression that can be\n    generated normally can also exist inside a Matrix:\n\n    >>> from sympy import Matrix, MatrixSymbol\n    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])\n    >>> A = MatrixSymbol(\'A\', 3, 1)\n    >>> print(fcode(mat, A))\n          A(1, 1) = x**2\n             if (x > 0) then\n          A(2, 1) = x + 1\n             else\n          A(2, 1) = x\n             end if\n          A(3, 1) = sin(x)\n    '
    from sympy.printing.fortran import FCodePrinter
    return FCodePrinter(settings).doprint(expr, assign_to)

def print_fcode(expr, **settings):
    if False:
        print('Hello World!')
    'Prints the Fortran representation of the given expression.\n\n       See fcode for the meaning of the optional arguments.\n    '
    print(fcode(expr, **settings))

def cxxcode(expr, assign_to=None, standard='c++11', **settings):
    if False:
        for i in range(10):
            print('nop')
    ' C++ equivalent of :func:`~.ccode`. '
    from sympy.printing.cxx import cxx_code_printers
    return cxx_code_printers[standard.lower()](settings).doprint(expr, assign_to)