"""
C code printer

The C89CodePrinter & C99CodePrinter converts single SymPy expressions into
single C expressions, using the functions defined in math.h where possible.

A complete code generator, which uses ccode extensively, can be found in
sympy.utilities.codegen. The codegen module can be used to generate complete
source code files that are compilable without further modifications.


"""
from __future__ import annotations
from typing import Any
from functools import wraps
from itertools import chain
from sympy.core import S
from sympy.core.numbers import equal_valued, Float
from sympy.codegen.ast import Assignment, Pointer, Variable, Declaration, Type, real, complex_, integer, bool_, float32, float64, float80, complex64, complex128, intc, value_const, pointer_const, int8, int16, int32, int64, uint8, uint16, uint32, uint64, untyped, none
from sympy.printing.codeprinter import CodePrinter, requires
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range
from sympy.printing.codeprinter import ccode, print_ccode
known_functions_C89 = {'Abs': [(lambda x: not x.is_integer, 'fabs'), (lambda x: x.is_integer, 'abs')], 'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'asin': 'asin', 'acos': 'acos', 'atan': 'atan', 'atan2': 'atan2', 'exp': 'exp', 'log': 'log', 'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh', 'floor': 'floor', 'ceiling': 'ceil', 'sqrt': 'sqrt'}
known_functions_C99 = dict(known_functions_C89, **{'exp2': 'exp2', 'expm1': 'expm1', 'log10': 'log10', 'log2': 'log2', 'log1p': 'log1p', 'Cbrt': 'cbrt', 'hypot': 'hypot', 'fma': 'fma', 'loggamma': 'lgamma', 'erfc': 'erfc', 'Max': 'fmax', 'Min': 'fmin', 'asinh': 'asinh', 'acosh': 'acosh', 'atanh': 'atanh', 'erf': 'erf', 'gamma': 'tgamma'})
reserved_words = ['auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if', 'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static', 'struct', 'entry', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while']
reserved_words_c99 = ['inline', 'restrict']

def get_math_macros():
    if False:
        i = 10
        return i + 15
    ' Returns a dictionary with math-related macros from math.h/cmath\n\n    Note that these macros are not strictly required by the C/C++-standard.\n    For MSVC they are enabled by defining "_USE_MATH_DEFINES" (preferably\n    via a compilation flag).\n\n    Returns\n    =======\n\n    Dictionary mapping SymPy expressions to strings (macro names)\n\n    '
    from sympy.codegen.cfunctions import log2, Sqrt
    from sympy.functions.elementary.exponential import log
    from sympy.functions.elementary.miscellaneous import sqrt
    return {S.Exp1: 'M_E', log2(S.Exp1): 'M_LOG2E', 1 / log(2): 'M_LOG2E', log(2): 'M_LN2', log(10): 'M_LN10', S.Pi: 'M_PI', S.Pi / 2: 'M_PI_2', S.Pi / 4: 'M_PI_4', 1 / S.Pi: 'M_1_PI', 2 / S.Pi: 'M_2_PI', 2 / sqrt(S.Pi): 'M_2_SQRTPI', 2 / Sqrt(S.Pi): 'M_2_SQRTPI', sqrt(2): 'M_SQRT2', Sqrt(2): 'M_SQRT2', 1 / sqrt(2): 'M_SQRT1_2', 1 / Sqrt(2): 'M_SQRT1_2'}

def _as_macro_if_defined(meth):
    if False:
        for i in range(10):
            print('nop')
    " Decorator for printer methods\n\n    When a Printer's method is decorated using this decorator the expressions printed\n    will first be looked for in the attribute ``math_macros``, and if present it will\n    print the macro name in ``math_macros`` followed by a type suffix for the type\n    ``real``. e.g. printing ``sympy.pi`` would print ``M_PIl`` if real is mapped to float80.\n\n    "

    @wraps(meth)
    def _meth_wrapper(self, expr, **kwargs):
        if False:
            i = 10
            return i + 15
        if expr in self.math_macros:
            return '%s%s' % (self.math_macros[expr], self._get_math_macro_suffix(real))
        else:
            return meth(self, expr, **kwargs)
    return _meth_wrapper

class C89CodePrinter(CodePrinter):
    """A printer to convert Python expressions to strings of C code"""
    printmethod = '_ccode'
    language = 'C'
    standard = 'C89'
    reserved_words = set(reserved_words)
    _default_settings: dict[str, Any] = {'order': None, 'full_prec': 'auto', 'precision': 17, 'user_functions': {}, 'human': True, 'allow_unknown_functions': False, 'contract': True, 'dereference': set(), 'error_on_reserved': False, 'reserved_word_suffix': '_'}
    type_aliases = {real: float64, complex_: complex128, integer: intc}
    type_mappings: dict[Type, Any] = {real: 'double', intc: 'int', float32: 'float', float64: 'double', integer: 'int', bool_: 'bool', int8: 'int8_t', int16: 'int16_t', int32: 'int32_t', int64: 'int64_t', uint8: 'int8_t', uint16: 'int16_t', uint32: 'int32_t', uint64: 'int64_t'}
    type_headers = {bool_: {'stdbool.h'}, int8: {'stdint.h'}, int16: {'stdint.h'}, int32: {'stdint.h'}, int64: {'stdint.h'}, uint8: {'stdint.h'}, uint16: {'stdint.h'}, uint32: {'stdint.h'}, uint64: {'stdint.h'}}
    type_macros: dict[Type, tuple[str, ...]] = {}
    type_func_suffixes = {float32: 'f', float64: '', float80: 'l'}
    type_literal_suffixes = {float32: 'F', float64: '', float80: 'L'}
    type_math_macro_suffixes = {float80: 'l'}
    math_macros = None
    _ns = ''
    _kf: dict[str, Any] = known_functions_C89

    def __init__(self, settings=None):
        if False:
            print('Hello World!')
        settings = settings or {}
        if self.math_macros is None:
            self.math_macros = settings.pop('math_macros', get_math_macros())
        self.type_aliases = dict(chain(self.type_aliases.items(), settings.pop('type_aliases', {}).items()))
        self.type_mappings = dict(chain(self.type_mappings.items(), settings.pop('type_mappings', {}).items()))
        self.type_headers = dict(chain(self.type_headers.items(), settings.pop('type_headers', {}).items()))
        self.type_macros = dict(chain(self.type_macros.items(), settings.pop('type_macros', {}).items()))
        self.type_func_suffixes = dict(chain(self.type_func_suffixes.items(), settings.pop('type_func_suffixes', {}).items()))
        self.type_literal_suffixes = dict(chain(self.type_literal_suffixes.items(), settings.pop('type_literal_suffixes', {}).items()))
        self.type_math_macro_suffixes = dict(chain(self.type_math_macro_suffixes.items(), settings.pop('type_math_macro_suffixes', {}).items()))
        super().__init__(settings)
        self.known_functions = dict(self._kf, **settings.get('user_functions', {}))
        self._dereference = set(settings.get('dereference', []))
        self.headers = set()
        self.libraries = set()
        self.macros = set()

    def _rate_index_position(self, p):
        if False:
            while True:
                i = 10
        return p * 5

    def _get_statement(self, codestring):
        if False:
            print('Hello World!')
        ' Get code string as a statement - i.e. ending with a semicolon. '
        return codestring if codestring.endswith(';') else codestring + ';'

    def _get_comment(self, text):
        if False:
            print('Hello World!')
        return '/* {} */'.format(text)

    def _declare_number_const(self, name, value):
        if False:
            while True:
                i = 10
        type_ = self.type_aliases[real]
        var = Variable(name, type=type_, value=value.evalf(type_.decimal_dig), attrs={value_const})
        decl = Declaration(var)
        return self._get_statement(self._print(decl))

    def _format_code(self, lines):
        if False:
            for i in range(10):
                print('nop')
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        if False:
            while True:
                i = 10
        (rows, cols) = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    @_as_macro_if_defined
    def _print_Mul(self, expr, **kwargs):
        if False:
            return 10
        return super()._print_Mul(expr, **kwargs)

    @_as_macro_if_defined
    def _print_Pow(self, expr):
        if False:
            for i in range(10):
                print('nop')
        if 'Pow' in self.known_functions:
            return self._print_Function(expr)
        PREC = precedence(expr)
        suffix = self._get_func_suffix(real)
        if equal_valued(expr.exp, -1):
            return '%s/%s' % (self._print_Float(Float(1.0)), self.parenthesize(expr.base, PREC))
        elif equal_valued(expr.exp, 0.5):
            return '%ssqrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        elif expr.exp == S.One / 3 and self.standard != 'C89':
            return '%scbrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        else:
            return '%spow%s(%s, %s)' % (self._ns, suffix, self._print(expr.base), self._print(expr.exp))

    def _print_Mod(self, expr):
        if False:
            return 10
        (num, den) = expr.args
        if num.is_integer and den.is_integer:
            PREC = precedence(expr)
            (snum, sden) = [self.parenthesize(arg, PREC) for arg in expr.args]
            if num.is_nonnegative and den.is_nonnegative or (num.is_nonpositive and den.is_nonpositive):
                return f'{snum} % {sden}'
            return f'(({snum} % {sden}) + {sden}) % {sden}'
        return self._print_math_func(expr, known='fmod')

    def _print_Rational(self, expr):
        if False:
            while True:
                i = 10
        (p, q) = (int(expr.p), int(expr.q))
        suffix = self._get_literal_suffix(real)
        return '%d.0%s/%d.0%s' % (p, suffix, q, suffix)

    def _print_Indexed(self, expr):
        if False:
            print('Hello World!')
        offset = getattr(expr.base, 'offset', S.Zero)
        strides = getattr(expr.base, 'strides', None)
        indices = expr.indices
        if strides is None or isinstance(strides, str):
            dims = expr.shape
            shift = S.One
            temp = ()
            if strides == 'C' or strides is None:
                traversal = reversed(range(expr.rank))
                indices = indices[::-1]
            elif strides == 'F':
                traversal = range(expr.rank)
            for i in traversal:
                temp += (shift,)
                shift *= dims[i]
            strides = temp
        flat_index = sum([x[0] * x[1] for x in zip(indices, strides)]) + offset
        return '%s[%s]' % (self._print(expr.base.label), self._print(flat_index))

    def _print_Idx(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return self._print(expr.label)

    @_as_macro_if_defined
    def _print_NumberSymbol(self, expr):
        if False:
            return 10
        return super()._print_NumberSymbol(expr)

    def _print_Infinity(self, expr):
        if False:
            while True:
                i = 10
        return 'HUGE_VAL'

    def _print_NegativeInfinity(self, expr):
        if False:
            print('Hello World!')
        return '-HUGE_VAL'

    def _print_Piecewise(self, expr):
        if False:
            i = 10
            return i + 15
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

    def _print_ITE(self, expr):
        if False:
            return 10
        from sympy.functions import Piecewise
        return self._print(expr.rewrite(Piecewise, deep=False))

    def _print_MatrixElement(self, expr):
        if False:
            return 10
        return '{}[{}]'.format(self.parenthesize(expr.parent, PRECEDENCE['Atom'], strict=True), expr.j + expr.i * expr.parent.shape[1])

    def _print_Symbol(self, expr):
        if False:
            while True:
                i = 10
        name = super()._print_Symbol(expr)
        if expr in self._settings['dereference']:
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

    def _print_For(self, expr):
        if False:
            for i in range(10):
                print('nop')
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            (start, stop, step) = expr.iterable.args
        else:
            raise NotImplementedError('Only iterable currently supported is Range')
        body = self._print(expr.body)
        return 'for ({target} = {start}; {target} < {stop}; {target} += {step}) {{\n{body}\n}}'.format(target=target, start=start, stop=stop, step=step, body=body)

    def _print_sign(self, func):
        if False:
            for i in range(10):
                print('nop')
        return '((({0}) > 0) - (({0}) < 0))'.format(self._print(func.args[0]))

    def _print_Max(self, expr):
        if False:
            i = 10
            return i + 15
        if 'Max' in self.known_functions:
            return self._print_Function(expr)

        def inner_print_max(args):
            if False:
                print('Hello World!')
            if len(args) == 1:
                return self._print(args[0])
            half = len(args) // 2
            return '((%(a)s > %(b)s) ? %(a)s : %(b)s)' % {'a': inner_print_max(args[:half]), 'b': inner_print_max(args[half:])}
        return inner_print_max(expr.args)

    def _print_Min(self, expr):
        if False:
            for i in range(10):
                print('nop')
        if 'Min' in self.known_functions:
            return self._print_Function(expr)

        def inner_print_min(args):
            if False:
                print('Hello World!')
            if len(args) == 1:
                return self._print(args[0])
            half = len(args) // 2
            return '((%(a)s < %(b)s) ? %(a)s : %(b)s)' % {'a': inner_print_min(args[:half]), 'b': inner_print_min(args[half:])}
        return inner_print_min(expr.args)

    def indent_code(self, code):
        if False:
            while True:
                i = 10
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

    def _get_func_suffix(self, type_):
        if False:
            i = 10
            return i + 15
        return self.type_func_suffixes[self.type_aliases.get(type_, type_)]

    def _get_literal_suffix(self, type_):
        if False:
            print('Hello World!')
        return self.type_literal_suffixes[self.type_aliases.get(type_, type_)]

    def _get_math_macro_suffix(self, type_):
        if False:
            return 10
        alias = self.type_aliases.get(type_, type_)
        dflt = self.type_math_macro_suffixes.get(alias, '')
        return self.type_math_macro_suffixes.get(type_, dflt)

    def _print_Tuple(self, expr):
        if False:
            i = 10
            return i + 15
        return '{' + ', '.join((self._print(e) for e in expr)) + '}'
    _print_List = _print_Tuple

    def _print_Type(self, type_):
        if False:
            while True:
                i = 10
        self.headers.update(self.type_headers.get(type_, set()))
        self.macros.update(self.type_macros.get(type_, set()))
        return self._print(self.type_mappings.get(type_, type_.name))

    def _print_Declaration(self, decl):
        if False:
            for i in range(10):
                print('nop')
        from sympy.codegen.cnodes import restrict
        var = decl.variable
        val = var.value
        if var.type == untyped:
            raise ValueError('C does not support untyped variables')
        if isinstance(var, Pointer):
            result = '{vc}{t} *{pc} {r}{s}'.format(vc='const ' if value_const in var.attrs else '', t=self._print(var.type), pc=' const' if pointer_const in var.attrs else '', r='restrict ' if restrict in var.attrs else '', s=self._print(var.symbol))
        elif isinstance(var, Variable):
            result = '{vc}{t} {s}'.format(vc='const ' if value_const in var.attrs else '', t=self._print(var.type), s=self._print(var.symbol))
        else:
            raise NotImplementedError('Unknown type of var: %s' % type(var))
        if val != None:
            result += ' = %s' % self._print(val)
        return result

    def _print_Float(self, flt):
        if False:
            for i in range(10):
                print('nop')
        type_ = self.type_aliases.get(real, real)
        self.macros.update(self.type_macros.get(type_, set()))
        suffix = self._get_literal_suffix(type_)
        num = str(flt.evalf(type_.decimal_dig))
        if 'e' not in num and '.' not in num:
            num += '.0'
        num_parts = num.split('e')
        num_parts[0] = num_parts[0].rstrip('0')
        if num_parts[0].endswith('.'):
            num_parts[0] += '0'
        return 'e'.join(num_parts) + suffix

    @requires(headers={'stdbool.h'})
    def _print_BooleanTrue(self, expr):
        if False:
            print('Hello World!')
        return 'true'

    @requires(headers={'stdbool.h'})
    def _print_BooleanFalse(self, expr):
        if False:
            return 10
        return 'false'

    def _print_Element(self, elem):
        if False:
            while True:
                i = 10
        if elem.strides == None:
            if elem.offset != None:
                raise ValueError('Expected strides when offset is given')
            idxs = ']['.join((self._print(arg) for arg in elem.indices))
        else:
            global_idx = sum([i * s for (i, s) in zip(elem.indices, elem.strides)])
            if elem.offset != None:
                global_idx += elem.offset
            idxs = self._print(global_idx)
        return '{symb}[{idxs}]'.format(symb=self._print(elem.symbol), idxs=idxs)

    def _print_CodeBlock(self, expr):
        if False:
            print('Hello World!')
        ' Elements of code blocks printed as statements. '
        return '\n'.join([self._get_statement(self._print(i)) for i in expr.args])

    def _print_While(self, expr):
        if False:
            print('Hello World!')
        return 'while ({condition}) {{\n{body}\n}}'.format(**expr.kwargs(apply=lambda arg: self._print(arg)))

    def _print_Scope(self, expr):
        if False:
            i = 10
            return i + 15
        return '{\n%s\n}' % self._print_CodeBlock(expr.body)

    @requires(headers={'stdio.h'})
    def _print_Print(self, expr):
        if False:
            return 10
        if expr.file == none:
            template = 'printf({fmt}, {pargs})'
        else:
            template = 'fprintf(%(out)s, {fmt}, {pargs})' % {'out': self._print(expr.file)}
        return template.format(fmt='%s\n' if expr.format_string == none else self._print(expr.format_string), pargs=', '.join((self._print(arg) for arg in expr.print_args)))

    def _print_Stream(self, strm):
        if False:
            return 10
        return strm.name

    def _print_FunctionPrototype(self, expr):
        if False:
            return 10
        pars = ', '.join((self._print(Declaration(arg)) for arg in expr.parameters))
        return '%s %s(%s)' % (tuple((self._print(arg) for arg in (expr.return_type, expr.name))) + (pars,))

    def _print_FunctionDefinition(self, expr):
        if False:
            while True:
                i = 10
        return '%s%s' % (self._print_FunctionPrototype(expr), self._print_Scope(expr))

    def _print_Return(self, expr):
        if False:
            for i in range(10):
                print('nop')
        (arg,) = expr.args
        return 'return %s' % self._print(arg)

    def _print_CommaOperator(self, expr):
        if False:
            return 10
        return '(%s)' % ', '.join((self._print(arg) for arg in expr.args))

    def _print_Label(self, expr):
        if False:
            while True:
                i = 10
        if expr.body == none:
            return '%s:' % str(expr.name)
        if len(expr.body.args) == 1:
            return '%s:\n%s' % (str(expr.name), self._print_CodeBlock(expr.body))
        return '%s:\n{\n%s\n}' % (str(expr.name), self._print_CodeBlock(expr.body))

    def _print_goto(self, expr):
        if False:
            return 10
        return 'goto %s' % expr.label.name

    def _print_PreIncrement(self, expr):
        if False:
            for i in range(10):
                print('nop')
        (arg,) = expr.args
        return '++(%s)' % self._print(arg)

    def _print_PostIncrement(self, expr):
        if False:
            i = 10
            return i + 15
        (arg,) = expr.args
        return '(%s)++' % self._print(arg)

    def _print_PreDecrement(self, expr):
        if False:
            return 10
        (arg,) = expr.args
        return '--(%s)' % self._print(arg)

    def _print_PostDecrement(self, expr):
        if False:
            return 10
        (arg,) = expr.args
        return '(%s)--' % self._print(arg)

    def _print_struct(self, expr):
        if False:
            print('Hello World!')
        return '%(keyword)s %(name)s {\n%(lines)s}' % {'keyword': expr.__class__.__name__, 'name': expr.name, 'lines': ';\n'.join([self._print(decl) for decl in expr.declarations] + [''])}

    def _print_BreakToken(self, _):
        if False:
            i = 10
            return i + 15
        return 'break'

    def _print_ContinueToken(self, _):
        if False:
            print('Hello World!')
        return 'continue'
    _print_union = _print_struct

class C99CodePrinter(C89CodePrinter):
    standard = 'C99'
    reserved_words = set(reserved_words + reserved_words_c99)
    type_mappings = dict(chain(C89CodePrinter.type_mappings.items(), {complex64: 'float complex', complex128: 'double complex'}.items()))
    type_headers = dict(chain(C89CodePrinter.type_headers.items(), {complex64: {'complex.h'}, complex128: {'complex.h'}}.items()))
    _kf: dict[str, Any] = known_functions_C99
    _prec_funcs = 'fabs fmod remainder remquo fma fmax fmin fdim nan exp exp2 expm1 log log10 log2 log1p pow sqrt cbrt hypot sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh atanh erf erfc tgamma lgamma ceil floor trunc round nearbyint rint frexp ldexp modf scalbn ilogb logb nextafter copysign'.split()

    def _print_Infinity(self, expr):
        if False:
            i = 10
            return i + 15
        return 'INFINITY'

    def _print_NegativeInfinity(self, expr):
        if False:
            i = 10
            return i + 15
        return '-INFINITY'

    def _print_NaN(self, expr):
        if False:
            while True:
                i = 10
        return 'NAN'

    @requires(headers={'math.h'}, libraries={'m'})
    @_as_macro_if_defined
    def _print_math_func(self, expr, nest=False, known=None):
        if False:
            i = 10
            return i + 15
        if known is None:
            known = self.known_functions[expr.__class__.__name__]
        if not isinstance(known, str):
            for (cb, name) in known:
                if cb(*expr.args):
                    known = name
                    break
            else:
                raise ValueError('No matching printer')
        try:
            return known(self, *expr.args)
        except TypeError:
            suffix = self._get_func_suffix(real) if self._ns + known in self._prec_funcs else ''
        if nest:
            args = self._print(expr.args[0])
            if len(expr.args) > 1:
                paren_pile = ''
                for curr_arg in expr.args[1:-1]:
                    paren_pile += ')'
                    args += ', {ns}{name}{suffix}({next}'.format(ns=self._ns, name=known, suffix=suffix, next=self._print(curr_arg))
                args += ', %s%s' % (self._print(expr.func(expr.args[-1])), paren_pile)
        else:
            args = ', '.join((self._print(arg) for arg in expr.args))
        return '{ns}{name}{suffix}({args})'.format(ns=self._ns, name=known, suffix=suffix, args=args)

    def _print_Max(self, expr):
        if False:
            print('Hello World!')
        return self._print_math_func(expr, nest=True)

    def _print_Min(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return self._print_math_func(expr, nest=True)

    def _get_loop_opening_ending(self, indices):
        if False:
            print('Hello World!')
        open_lines = []
        close_lines = []
        loopstart = 'for (int %(var)s=%(start)s; %(var)s<%(end)s; %(var)s++){'
        for i in indices:
            open_lines.append(loopstart % {'var': self._print(i.label), 'start': self._print(i.lower), 'end': self._print(i.upper + 1)})
            close_lines.append('}')
        return (open_lines, close_lines)
for k in 'Abs Sqrt exp exp2 expm1 log log10 log2 log1p Cbrt hypot fma loggamma sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh atanh erf erfc loggamma gamma ceiling floor'.split():
    setattr(C99CodePrinter, '_print_%s' % k, C99CodePrinter._print_math_func)

class C11CodePrinter(C99CodePrinter):

    @requires(headers={'stdalign.h'})
    def _print_alignof(self, expr):
        if False:
            i = 10
            return i + 15
        (arg,) = expr.args
        return 'alignof(%s)' % self._print(arg)
c_code_printers = {'c89': C89CodePrinter, 'c99': C99CodePrinter, 'c11': C11CodePrinter}