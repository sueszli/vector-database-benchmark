"""
Python code printers

This module contains Python code printers for plain Python as well as NumPy & SciPy enabled code.
"""
from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
_kw = {'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield', 'None', 'False', 'nonlocal', 'True'}
_known_functions = {'Abs': 'abs', 'Min': 'min', 'Max': 'max'}
_known_functions_math = {'acos': 'acos', 'acosh': 'acosh', 'asin': 'asin', 'asinh': 'asinh', 'atan': 'atan', 'atan2': 'atan2', 'atanh': 'atanh', 'ceiling': 'ceil', 'cos': 'cos', 'cosh': 'cosh', 'erf': 'erf', 'erfc': 'erfc', 'exp': 'exp', 'expm1': 'expm1', 'factorial': 'factorial', 'floor': 'floor', 'gamma': 'gamma', 'hypot': 'hypot', 'isnan': 'isnan', 'loggamma': 'lgamma', 'log': 'log', 'ln': 'log', 'log10': 'log10', 'log1p': 'log1p', 'log2': 'log2', 'sin': 'sin', 'sinh': 'sinh', 'Sqrt': 'sqrt', 'tan': 'tan', 'tanh': 'tanh'}
_known_constants_math = {'Exp1': 'e', 'Pi': 'pi', 'E': 'e', 'Infinity': 'inf', 'NaN': 'nan', 'ComplexInfinity': 'nan'}

def _print_known_func(self, expr):
    if False:
        while True:
            i = 10
    known = self.known_functions[expr.__class__.__name__]
    return '{name}({args})'.format(name=self._module_format(known), args=', '.join((self._print(arg) for arg in expr.args)))

def _print_known_const(self, expr):
    if False:
        for i in range(10):
            print('nop')
    known = self.known_constants[expr.__class__.__name__]
    return self._module_format(known)

class AbstractPythonCodePrinter(CodePrinter):
    printmethod = '_pythoncode'
    language = 'Python'
    reserved_words = _kw
    modules = None
    tab = '    '
    _kf = dict(chain(_known_functions.items(), [(k, 'math.' + v) for (k, v) in _known_functions_math.items()]))
    _kc = {k: 'math.' + v for (k, v) in _known_constants_math.items()}
    _operators = {'and': 'and', 'or': 'or', 'not': 'not'}
    _default_settings = dict(CodePrinter._default_settings, user_functions={}, precision=17, inline=True, fully_qualified_modules=True, contract=False, standard='python3')

    def __init__(self, settings=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(settings)
        std = self._settings['standard']
        if std is None:
            import sys
            std = 'python{}'.format(sys.version_info.major)
        if std != 'python3':
            raise ValueError('Only Python 3 is supported.')
        self.standard = std
        self.module_imports = defaultdict(set)
        self.known_functions = dict(self._kf, **(settings or {}).get('user_functions', {}))
        self.known_constants = dict(self._kc, **(settings or {}).get('user_constants', {}))

    def _declare_number_const(self, name, value):
        if False:
            for i in range(10):
                print('nop')
        return '%s = %s' % (name, value)

    def _module_format(self, fqn, register=True):
        if False:
            for i in range(10):
                print('nop')
        parts = fqn.split('.')
        if register and len(parts) > 1:
            self.module_imports['.'.join(parts[:-1])].add(parts[-1])
        if self._settings['fully_qualified_modules']:
            return fqn
        else:
            return fqn.split('(')[0].split('[')[0].split('.')[-1]

    def _format_code(self, lines):
        if False:
            while True:
                i = 10
        return lines

    def _get_statement(self, codestring):
        if False:
            print('Hello World!')
        return '{}'.format(codestring)

    def _get_comment(self, text):
        if False:
            print('Hello World!')
        return '  # {}'.format(text)

    def _expand_fold_binary_op(self, op, args):
        if False:
            return 10
        '\n        This method expands a fold on binary operations.\n\n        ``functools.reduce`` is an example of a folded operation.\n\n        For example, the expression\n\n        `A + B + C + D`\n\n        is folded into\n\n        `((A + B) + C) + D`\n        '
        if len(args) == 1:
            return self._print(args[0])
        else:
            return '%s(%s, %s)' % (self._module_format(op), self._expand_fold_binary_op(op, args[:-1]), self._print(args[-1]))

    def _expand_reduce_binary_op(self, op, args):
        if False:
            i = 10
            return i + 15
        '\n        This method expands a reductin on binary operations.\n\n        Notice: this is NOT the same as ``functools.reduce``.\n\n        For example, the expression\n\n        `A + B + C + D`\n\n        is reduced into:\n\n        `(A + B) + (C + D)`\n        '
        if len(args) == 1:
            return self._print(args[0])
        else:
            N = len(args)
            Nhalf = N // 2
            return '%s(%s, %s)' % (self._module_format(op), self._expand_reduce_binary_op(args[:Nhalf]), self._expand_reduce_binary_op(args[Nhalf:]))

    def _print_NaN(self, expr):
        if False:
            return 10
        return "float('nan')"

    def _print_Infinity(self, expr):
        if False:
            print('Hello World!')
        return "float('inf')"

    def _print_NegativeInfinity(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return "float('-inf')"

    def _print_ComplexInfinity(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return self._print_NaN(expr)

    def _print_Mod(self, expr):
        if False:
            for i in range(10):
                print('nop')
        PREC = precedence(expr)
        return '{} % {}'.format(*(self.parenthesize(x, PREC) for x in expr.args))

    def _print_Piecewise(self, expr):
        if False:
            return 10
        result = []
        i = 0
        for arg in expr.args:
            e = arg.expr
            c = arg.cond
            if i == 0:
                result.append('(')
            result.append('(')
            result.append(self._print(e))
            result.append(')')
            result.append(' if ')
            result.append(self._print(c))
            result.append(' else ')
            i += 1
        result = result[:-1]
        if result[-1] == 'True':
            result = result[:-2]
            result.append(')')
        else:
            result.append(' else None)')
        return ''.join(result)

    def _print_Relational(self, expr):
        if False:
            i = 10
            return i + 15
        'Relational printer for Equality and Unequality'
        op = {'==': 'equal', '!=': 'not_equal', '<': 'less', '<=': 'less_equal', '>': 'greater', '>=': 'greater_equal'}
        if expr.rel_op in op:
            lhs = self._print(expr.lhs)
            rhs = self._print(expr.rhs)
            return '({lhs} {op} {rhs})'.format(op=expr.rel_op, lhs=lhs, rhs=rhs)
        return super()._print_Relational(expr)

    def _print_ITE(self, expr):
        if False:
            return 10
        from sympy.functions.elementary.piecewise import Piecewise
        return self._print(expr.rewrite(Piecewise))

    def _print_Sum(self, expr):
        if False:
            while True:
                i = 10
        loops = ('for {i} in range({a}, {b}+1)'.format(i=self._print(i), a=self._print(a), b=self._print(b)) for (i, a, b) in expr.limits)
        return '(builtins.sum({function} {loops}))'.format(function=self._print(expr.function), loops=' '.join(loops))

    def _print_ImaginaryUnit(self, expr):
        if False:
            while True:
                i = 10
        return '1j'

    def _print_KroneckerDelta(self, expr):
        if False:
            while True:
                i = 10
        (a, b) = expr.args
        return '(1 if {a} == {b} else 0)'.format(a=self._print(a), b=self._print(b))

    def _print_MatrixBase(self, expr):
        if False:
            while True:
                i = 10
        name = expr.__class__.__name__
        func = self.known_functions.get(name, name)
        return '%s(%s)' % (func, self._print(expr.tolist()))
    _print_SparseRepMatrix = _print_MutableSparseMatrix = _print_ImmutableSparseMatrix = _print_Matrix = _print_DenseMatrix = _print_MutableDenseMatrix = _print_ImmutableMatrix = _print_ImmutableDenseMatrix = lambda self, expr: self._print_MatrixBase(expr)

    def _indent_codestring(self, codestring):
        if False:
            return 10
        return '\n'.join([self.tab + line for line in codestring.split('\n')])

    def _print_FunctionDefinition(self, fd):
        if False:
            i = 10
            return i + 15
        body = '\n'.join((self._print(arg) for arg in fd.body))
        return 'def {name}({parameters}):\n{body}'.format(name=self._print(fd.name), parameters=', '.join([self._print(var.symbol) for var in fd.parameters]), body=self._indent_codestring(body))

    def _print_While(self, whl):
        if False:
            while True:
                i = 10
        body = '\n'.join((self._print(arg) for arg in whl.body))
        return 'while {cond}:\n{body}'.format(cond=self._print(whl.condition), body=self._indent_codestring(body))

    def _print_Declaration(self, decl):
        if False:
            return 10
        return '%s = %s' % (self._print(decl.variable.symbol), self._print(decl.variable.value))

    def _print_BreakToken(self, bt):
        if False:
            while True:
                i = 10
        return 'break'

    def _print_Return(self, ret):
        if False:
            while True:
                i = 10
        (arg,) = ret.args
        return 'return %s' % self._print(arg)

    def _print_Raise(self, rs):
        if False:
            print('Hello World!')
        (arg,) = rs.args
        return 'raise %s' % self._print(arg)

    def _print_RuntimeError_(self, re):
        if False:
            print('Hello World!')
        (message,) = re.args
        return 'RuntimeError(%s)' % self._print(message)

    def _print_Print(self, prnt):
        if False:
            while True:
                i = 10
        print_args = ', '.join((self._print(arg) for arg in prnt.print_args))
        from sympy.codegen.ast import none
        if prnt.format_string != none:
            print_args = '{} % ({}), end=""'.format(self._print(prnt.format_string), print_args)
        if prnt.file != None:
            print_args += ', file=%s' % self._print(prnt.file)
        return 'print(%s)' % print_args

    def _print_Stream(self, strm):
        if False:
            print('Hello World!')
        if str(strm.name) == 'stdout':
            return self._module_format('sys.stdout')
        elif str(strm.name) == 'stderr':
            return self._module_format('sys.stderr')
        else:
            return self._print(strm.name)

    def _print_NoneToken(self, arg):
        if False:
            i = 10
            return i + 15
        return 'None'

    def _hprint_Pow(self, expr, rational=False, sqrt='math.sqrt'):
        if False:
            while True:
                i = 10
        "Printing helper function for ``Pow``\n\n        Notes\n        =====\n\n        This preprocesses the ``sqrt`` as math formatter and prints division\n\n        Examples\n        ========\n\n        >>> from sympy import sqrt\n        >>> from sympy.printing.pycode import PythonCodePrinter\n        >>> from sympy.abc import x\n\n        Python code printer automatically looks up ``math.sqrt``.\n\n        >>> printer = PythonCodePrinter()\n        >>> printer._hprint_Pow(sqrt(x), rational=True)\n        'x**(1/2)'\n        >>> printer._hprint_Pow(sqrt(x), rational=False)\n        'math.sqrt(x)'\n        >>> printer._hprint_Pow(1/sqrt(x), rational=True)\n        'x**(-1/2)'\n        >>> printer._hprint_Pow(1/sqrt(x), rational=False)\n        '1/math.sqrt(x)'\n        >>> printer._hprint_Pow(1/x, rational=False)\n        '1/x'\n        >>> printer._hprint_Pow(1/x, rational=True)\n        'x**(-1)'\n\n        Using sqrt from numpy or mpmath\n\n        >>> printer._hprint_Pow(sqrt(x), sqrt='numpy.sqrt')\n        'numpy.sqrt(x)'\n        >>> printer._hprint_Pow(sqrt(x), sqrt='mpmath.sqrt')\n        'mpmath.sqrt(x)'\n\n        See Also\n        ========\n\n        sympy.printing.str.StrPrinter._print_Pow\n        "
        PREC = precedence(expr)
        if expr.exp == S.Half and (not rational):
            func = self._module_format(sqrt)
            arg = self._print(expr.base)
            return '{func}({arg})'.format(func=func, arg=arg)
        if expr.is_commutative and (not rational):
            if -expr.exp is S.Half:
                func = self._module_format(sqrt)
                num = self._print(S.One)
                arg = self._print(expr.base)
                return f'{num}/{func}({arg})'
            if expr.exp is S.NegativeOne:
                num = self._print(S.One)
                arg = self.parenthesize(expr.base, PREC, strict=False)
                return f'{num}/{arg}'
        base_str = self.parenthesize(expr.base, PREC, strict=False)
        exp_str = self.parenthesize(expr.exp, PREC, strict=False)
        return '{}**{}'.format(base_str, exp_str)

class ArrayPrinter:

    def _arrayify(self, indexed):
        if False:
            for i in range(10):
                print('nop')
        from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
        try:
            return convert_indexed_to_array(indexed)
        except Exception:
            return indexed

    def _get_einsum_string(self, subranks, contraction_indices):
        if False:
            while True:
                i = 10
        letters = self._get_letter_generator_for_einsum()
        contraction_string = ''
        counter = 0
        d = {j: min(i) for i in contraction_indices for j in i}
        indices = []
        for rank_arg in subranks:
            lindices = []
            for i in range(rank_arg):
                if counter in d:
                    lindices.append(d[counter])
                else:
                    lindices.append(counter)
                counter += 1
            indices.append(lindices)
        mapping = {}
        letters_free = []
        letters_dum = []
        for i in indices:
            for j in i:
                if j not in mapping:
                    l = next(letters)
                    mapping[j] = l
                else:
                    l = mapping[j]
                contraction_string += l
                if j in d:
                    if l not in letters_dum:
                        letters_dum.append(l)
                else:
                    letters_free.append(l)
            contraction_string += ','
        contraction_string = contraction_string[:-1]
        return (contraction_string, letters_free, letters_dum)

    def _get_letter_generator_for_einsum(self):
        if False:
            return 10
        for i in range(97, 123):
            yield chr(i)
        for i in range(65, 91):
            yield chr(i)
        raise ValueError('out of letters')

    def _print_ArrayTensorProduct(self, expr):
        if False:
            print('Hello World!')
        letters = self._get_letter_generator_for_einsum()
        contraction_string = ','.join([''.join([next(letters) for j in range(i)]) for i in expr.subranks])
        return '%s("%s", %s)' % (self._module_format(self._module + '.' + self._einsum), contraction_string, ', '.join([self._print(arg) for arg in expr.args]))

    def _print_ArrayContraction(self, expr):
        if False:
            return 10
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        base = expr.expr
        contraction_indices = expr.contraction_indices
        if isinstance(base, ArrayTensorProduct):
            elems = ','.join(['%s' % self._print(arg) for arg in base.args])
            ranks = base.subranks
        else:
            elems = self._print(base)
            ranks = [len(base.shape)]
        (contraction_string, letters_free, letters_dum) = self._get_einsum_string(ranks, contraction_indices)
        if not contraction_indices:
            return self._print(base)
        if isinstance(base, ArrayTensorProduct):
            elems = ','.join(['%s' % self._print(arg) for arg in base.args])
        else:
            elems = self._print(base)
        return '%s("%s", %s)' % (self._module_format(self._module + '.' + self._einsum), '{}->{}'.format(contraction_string, ''.join(sorted(letters_free))), elems)

    def _print_ArrayDiagonal(self, expr):
        if False:
            i = 10
            return i + 15
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        diagonal_indices = list(expr.diagonal_indices)
        if isinstance(expr.expr, ArrayTensorProduct):
            subranks = expr.expr.subranks
            elems = expr.expr.args
        else:
            subranks = expr.subranks
            elems = [expr.expr]
        (diagonal_string, letters_free, letters_dum) = self._get_einsum_string(subranks, diagonal_indices)
        elems = [self._print(i) for i in elems]
        return '%s("%s", %s)' % (self._module_format(self._module + '.' + self._einsum), '{}->{}'.format(diagonal_string, ''.join(letters_free + letters_dum)), ', '.join(elems))

    def _print_PermuteDims(self, expr):
        if False:
            return 10
        return '%s(%s, %s)' % (self._module_format(self._module + '.' + self._transpose), self._print(expr.expr), self._print(expr.permutation.array_form))

    def _print_ArrayAdd(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return self._expand_fold_binary_op(self._module + '.' + self._add, expr.args)

    def _print_OneArray(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return '%s((%s,))' % (self._module_format(self._module + '.' + self._ones), ','.join(map(self._print, expr.args)))

    def _print_ZeroArray(self, expr):
        if False:
            i = 10
            return i + 15
        return '%s((%s,))' % (self._module_format(self._module + '.' + self._zeros), ','.join(map(self._print, expr.args)))

    def _print_Assignment(self, expr):
        if False:
            for i in range(10):
                print('nop')
        lhs = self._print(self._arrayify(expr.lhs))
        rhs = self._print(self._arrayify(expr.rhs))
        return '%s = %s' % (lhs, rhs)

    def _print_IndexedBase(self, expr):
        if False:
            print('Hello World!')
        return self._print_ArraySymbol(expr)

class PythonCodePrinter(AbstractPythonCodePrinter):

    def _print_sign(self, e):
        if False:
            for i in range(10):
                print('nop')
        return '(0.0 if {e} == 0 else {f}(1, {e}))'.format(f=self._module_format('math.copysign'), e=self._print(e.args[0]))

    def _print_Not(self, expr):
        if False:
            return 10
        PREC = precedence(expr)
        return self._operators['not'] + self.parenthesize(expr.args[0], PREC)

    def _print_Indexed(self, expr):
        if False:
            i = 10
            return i + 15
        base = expr.args[0]
        index = expr.args[1:]
        return '{}[{}]'.format(str(base), ', '.join([self._print(ind) for ind in index]))

    def _print_Pow(self, expr, rational=False):
        if False:
            return 10
        return self._hprint_Pow(expr, rational=rational)

    def _print_Rational(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return '{}/{}'.format(expr.p, expr.q)

    def _print_Half(self, expr):
        if False:
            print('Hello World!')
        return self._print_Rational(expr)

    def _print_frac(self, expr):
        if False:
            return 10
        return self._print_Mod(Mod(expr.args[0], 1))

    def _print_Symbol(self, expr):
        if False:
            print('Hello World!')
        name = super()._print_Symbol(expr)
        if name in self.reserved_words:
            if self._settings['error_on_reserved']:
                msg = 'This expression includes the symbol "{}" which is a reserved keyword in this language.'
                raise ValueError(msg.format(name))
            return name + self._settings['reserved_word_suffix']
        elif '{' in name:
            return name.replace('{', '').replace('}', '')
        else:
            return name
    _print_lowergamma = CodePrinter._print_not_supported
    _print_uppergamma = CodePrinter._print_not_supported
    _print_fresnelc = CodePrinter._print_not_supported
    _print_fresnels = CodePrinter._print_not_supported
for k in PythonCodePrinter._kf:
    setattr(PythonCodePrinter, '_print_%s' % k, _print_known_func)
for k in _known_constants_math:
    setattr(PythonCodePrinter, '_print_%s' % k, _print_known_const)

def pycode(expr, **settings):
    if False:
        for i in range(10):
            print('nop')
    " Converts an expr to a string of Python code\n\n    Parameters\n    ==========\n\n    expr : Expr\n        A SymPy expression.\n    fully_qualified_modules : bool\n        Whether or not to write out full module names of functions\n        (``math.sin`` vs. ``sin``). default: ``True``.\n    standard : str or None, optional\n        Only 'python3' (default) is supported.\n        This parameter may be removed in the future.\n\n    Examples\n    ========\n\n    >>> from sympy import pycode, tan, Symbol\n    >>> pycode(tan(Symbol('x')) + 1)\n    'math.tan(x) + 1'\n\n    "
    return PythonCodePrinter(settings).doprint(expr)
_not_in_mpmath = 'log1p log2'.split()
_in_mpmath = [(k, v) for (k, v) in _known_functions_math.items() if k not in _not_in_mpmath]
_known_functions_mpmath = dict(_in_mpmath, **{'beta': 'beta', 'frac': 'frac', 'fresnelc': 'fresnelc', 'fresnels': 'fresnels', 'sign': 'sign', 'loggamma': 'loggamma', 'hyper': 'hyper', 'meijerg': 'meijerg', 'besselj': 'besselj', 'bessely': 'bessely', 'besseli': 'besseli', 'besselk': 'besselk'})
_known_constants_mpmath = {'Exp1': 'e', 'Pi': 'pi', 'GoldenRatio': 'phi', 'EulerGamma': 'euler', 'Catalan': 'catalan', 'NaN': 'nan', 'Infinity': 'inf', 'NegativeInfinity': 'ninf'}

def _unpack_integral_limits(integral_expr):
    if False:
        print('Hello World!')
    ' helper function for _print_Integral that\n        - accepts an Integral expression\n        - returns a tuple of\n           - a list variables of integration\n           - a list of tuples of the upper and lower limits of integration\n    '
    integration_vars = []
    limits = []
    for integration_range in integral_expr.limits:
        if len(integration_range) == 3:
            (integration_var, lower_limit, upper_limit) = integration_range
        else:
            raise NotImplementedError('Only definite integrals are supported')
        integration_vars.append(integration_var)
        limits.append((lower_limit, upper_limit))
    return (integration_vars, limits)

class MpmathPrinter(PythonCodePrinter):
    """
    Lambda printer for mpmath which maintains precision for floats
    """
    printmethod = '_mpmathcode'
    language = 'Python with mpmath'
    _kf = dict(chain(_known_functions.items(), [(k, 'mpmath.' + v) for (k, v) in _known_functions_mpmath.items()]))
    _kc = {k: 'mpmath.' + v for (k, v) in _known_constants_mpmath.items()}

    def _print_Float(self, e):
        if False:
            for i in range(10):
                print('nop')
        args = str(tuple(map(int, e._mpf_)))
        return '{func}({args})'.format(func=self._module_format('mpmath.mpf'), args=args)

    def _print_Rational(self, e):
        if False:
            print('Hello World!')
        return '{func}({p})/{func}({q})'.format(func=self._module_format('mpmath.mpf'), q=self._print(e.q), p=self._print(e.p))

    def _print_Half(self, e):
        if False:
            i = 10
            return i + 15
        return self._print_Rational(e)

    def _print_uppergamma(self, e):
        if False:
            i = 10
            return i + 15
        return '{}({}, {}, {})'.format(self._module_format('mpmath.gammainc'), self._print(e.args[0]), self._print(e.args[1]), self._module_format('mpmath.inf'))

    def _print_lowergamma(self, e):
        if False:
            i = 10
            return i + 15
        return '{}({}, 0, {})'.format(self._module_format('mpmath.gammainc'), self._print(e.args[0]), self._print(e.args[1]))

    def _print_log2(self, e):
        if False:
            i = 10
            return i + 15
        return '{0}({1})/{0}(2)'.format(self._module_format('mpmath.log'), self._print(e.args[0]))

    def _print_log1p(self, e):
        if False:
            while True:
                i = 10
        return '{}({})'.format(self._module_format('mpmath.log1p'), self._print(e.args[0]))

    def _print_Pow(self, expr, rational=False):
        if False:
            print('Hello World!')
        return self._hprint_Pow(expr, rational=rational, sqrt='mpmath.sqrt')

    def _print_Integral(self, e):
        if False:
            print('Hello World!')
        (integration_vars, limits) = _unpack_integral_limits(e)
        return '{}(lambda {}: {}, {})'.format(self._module_format('mpmath.quad'), ', '.join(map(self._print, integration_vars)), self._print(e.args[0]), ', '.join(('(%s, %s)' % tuple(map(self._print, l)) for l in limits)))
for k in MpmathPrinter._kf:
    setattr(MpmathPrinter, '_print_%s' % k, _print_known_func)
for k in _known_constants_mpmath:
    setattr(MpmathPrinter, '_print_%s' % k, _print_known_const)

class SymPyPrinter(AbstractPythonCodePrinter):
    language = 'Python with SymPy'

    def _print_Function(self, expr):
        if False:
            for i in range(10):
                print('nop')
        mod = expr.func.__module__ or ''
        return '%s(%s)' % (self._module_format(mod + ('.' if mod else '') + expr.func.__name__), ', '.join((self._print(arg) for arg in expr.args)))

    def _print_Pow(self, expr, rational=False):
        if False:
            return 10
        return self._hprint_Pow(expr, rational=rational, sqrt='sympy.sqrt')