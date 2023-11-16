import typing
import sympy
from sympy.core import Add, Mul
from sympy.core import Symbol, Expr, Float, Rational, Integer, Basic
from sympy.core.function import UndefinedFunction, Function
from sympy.core.relational import Relational, Unequality, Equality, LessThan, GreaterThan, StrictLessThan, StrictGreaterThan
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log, Pow
from sympy.functions.elementary.hyperbolic import sinh, cosh, tanh
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin, cos, tan, asin, acos, atan, atan2
from sympy.logic.boolalg import And, Or, Xor, Implies, Boolean
from sympy.logic.boolalg import BooleanTrue, BooleanFalse, BooleanFunction, Not, ITE
from sympy.printing.printer import Printer
from sympy.sets import Interval
from mpmath.libmp.libmpf import prec_to_dps, to_str as mlib_to_str
from sympy.assumptions.assume import AppliedPredicate
from sympy.assumptions.relation.binrel import AppliedBinaryRelation
from sympy.assumptions.ask import Q
from sympy.assumptions.relation.equality import StrictGreaterThanPredicate, StrictLessThanPredicate, GreaterThanPredicate, LessThanPredicate, EqualityPredicate

class SMTLibPrinter(Printer):
    printmethod = '_smtlib'
    _default_settings: dict = {'precision': None, 'known_types': {bool: 'Bool', int: 'Int', float: 'Real'}, 'known_constants': {}, 'known_functions': {Add: '+', Mul: '*', Equality: '=', LessThan: '<=', GreaterThan: '>=', StrictLessThan: '<', StrictGreaterThan: '>', EqualityPredicate(): '=', LessThanPredicate(): '<=', GreaterThanPredicate(): '>=', StrictLessThanPredicate(): '<', StrictGreaterThanPredicate(): '>', exp: 'exp', log: 'log', Abs: 'abs', sin: 'sin', cos: 'cos', tan: 'tan', asin: 'arcsin', acos: 'arccos', atan: 'arctan', atan2: 'arctan2', sinh: 'sinh', cosh: 'cosh', tanh: 'tanh', Min: 'min', Max: 'max', Pow: 'pow', And: 'and', Or: 'or', Xor: 'xor', Not: 'not', ITE: 'ite', Implies: '=>'}}
    symbol_table: dict

    def __init__(self, settings: typing.Optional[dict]=None, symbol_table=None):
        if False:
            for i in range(10):
                print('nop')
        settings = settings or {}
        self.symbol_table = symbol_table or {}
        Printer.__init__(self, settings)
        self._precision = self._settings['precision']
        self._known_types = dict(self._settings['known_types'])
        self._known_constants = dict(self._settings['known_constants'])
        self._known_functions = dict(self._settings['known_functions'])
        for _ in self._known_types.values():
            assert self._is_legal_name(_)
        for _ in self._known_constants.values():
            assert self._is_legal_name(_)

    def _is_legal_name(self, s: str):
        if False:
            print('Hello World!')
        if not s:
            return False
        if s[0].isnumeric():
            return False
        return all((_.isalnum() or _ == '_' for _ in s))

    def _s_expr(self, op: str, args: typing.Union[list, tuple]) -> str:
        if False:
            i = 10
            return i + 15
        args_str = ' '.join((a if isinstance(a, str) else self._print(a) for a in args))
        return f'({op} {args_str})'

    def _print_Function(self, e):
        if False:
            for i in range(10):
                print('nop')
        if e in self._known_functions:
            op = self._known_functions[e]
        elif type(e) in self._known_functions:
            op = self._known_functions[type(e)]
        elif type(type(e)) == UndefinedFunction:
            op = e.name
        elif isinstance(e, AppliedBinaryRelation) and e.function in self._known_functions:
            op = self._known_functions[e.function]
            return self._s_expr(op, e.arguments)
        else:
            op = self._known_functions[e]
        return self._s_expr(op, e.args)

    def _print_Relational(self, e: Relational):
        if False:
            i = 10
            return i + 15
        return self._print_Function(e)

    def _print_BooleanFunction(self, e: BooleanFunction):
        if False:
            for i in range(10):
                print('nop')
        return self._print_Function(e)

    def _print_Expr(self, e: Expr):
        if False:
            return 10
        return self._print_Function(e)

    def _print_Unequality(self, e: Unequality):
        if False:
            return 10
        if type(e) in self._known_functions:
            return self._print_Relational(e)
        else:
            eq_op = self._known_functions[Equality]
            not_op = self._known_functions[Not]
            return self._s_expr(not_op, [self._s_expr(eq_op, e.args)])

    def _print_Piecewise(self, e: Piecewise):
        if False:
            while True:
                i = 10

        def _print_Piecewise_recursive(args: typing.Union[list, tuple]):
            if False:
                i = 10
                return i + 15
            (e, c) = args[0]
            if len(args) == 1:
                assert c is True or isinstance(c, BooleanTrue)
                return self._print(e)
            else:
                ite = self._known_functions[ITE]
                return self._s_expr(ite, [c, e, _print_Piecewise_recursive(args[1:])])
        return _print_Piecewise_recursive(e.args)

    def _print_Interval(self, e: Interval):
        if False:
            for i in range(10):
                print('nop')
        if e.start.is_infinite and e.end.is_infinite:
            return ''
        elif e.start.is_infinite != e.end.is_infinite:
            raise ValueError(f'One-sided intervals (`{e}`) are not supported in SMT.')
        else:
            return f'[{e.start}, {e.end}]'

    def _print_AppliedPredicate(self, e: AppliedPredicate):
        if False:
            i = 10
            return i + 15
        if e.function == Q.positive:
            rel = Q.gt(e.arguments[0], 0)
        elif e.function == Q.negative:
            rel = Q.lt(e.arguments[0], 0)
        elif e.function == Q.zero:
            rel = Q.eq(e.arguments[0], 0)
        elif e.function == Q.nonpositive:
            rel = Q.le(e.arguments[0], 0)
        elif e.function == Q.nonnegative:
            rel = Q.ge(e.arguments[0], 0)
        elif e.function == Q.nonzero:
            rel = Q.ne(e.arguments[0], 0)
        else:
            raise ValueError(f'Predicate (`{e}`) is not handled.')
        return self._print_AppliedBinaryRelation(rel)

    def _print_AppliedBinaryRelation(self, e: AppliedPredicate):
        if False:
            for i in range(10):
                print('nop')
        if e.function == Q.ne:
            return self._print_Unequality(Unequality(*e.arguments))
        else:
            return self._print_Function(e)

    def _print_BooleanTrue(self, x: BooleanTrue):
        if False:
            i = 10
            return i + 15
        return 'true'

    def _print_BooleanFalse(self, x: BooleanFalse):
        if False:
            while True:
                i = 10
        return 'false'

    def _print_Float(self, x: Float):
        if False:
            for i in range(10):
                print('nop')
        dps = prec_to_dps(x._prec)
        str_real = mlib_to_str(x._mpf_, dps, strip_zeros=True, min_fixed=None, max_fixed=None)
        if 'e' in str_real:
            (mant, exp) = str_real.split('e')
            if exp[0] == '+':
                exp = exp[1:]
            mul = self._known_functions[Mul]
            pow = self._known_functions[Pow]
            return '(%s %s (%s 10 %s))' % (mul, mant, pow, exp)
        elif str_real in ['+inf', '-inf']:
            raise ValueError('Infinite values are not supported in SMT.')
        else:
            return str_real

    def _print_float(self, x: float):
        if False:
            while True:
                i = 10
        return self._print(Float(x))

    def _print_Rational(self, x: Rational):
        if False:
            for i in range(10):
                print('nop')
        return self._s_expr('/', [x.p, x.q])

    def _print_Integer(self, x: Integer):
        if False:
            i = 10
            return i + 15
        assert x.q == 1
        return str(x.p)

    def _print_int(self, x: int):
        if False:
            for i in range(10):
                print('nop')
        return str(x)

    def _print_Symbol(self, x: Symbol):
        if False:
            return 10
        assert self._is_legal_name(x.name)
        return x.name

    def _print_NumberSymbol(self, x):
        if False:
            return 10
        name = self._known_constants.get(x)
        if name:
            return name
        else:
            f = x.evalf(self._precision) if self._precision else x.evalf()
            return self._print_Float(f)

    def _print_UndefinedFunction(self, x):
        if False:
            i = 10
            return i + 15
        assert self._is_legal_name(x.name)
        return x.name

    def _print_Exp1(self, x):
        if False:
            while True:
                i = 10
        return self._print_Function(exp(1, evaluate=False)) if exp in self._known_functions else self._print_NumberSymbol(x)

    def emptyPrinter(self, expr):
        if False:
            while True:
                i = 10
        raise NotImplementedError(f'Cannot convert `{repr(expr)}` of type `{type(expr)}` to SMT.')

def smtlib_code(expr, auto_assert=True, auto_declare=True, precision=None, symbol_table=None, known_types=None, known_constants=None, known_functions=None, prefix_expressions=None, suffix_expressions=None, log_warn=None):
    if False:
        while True:
            i = 10
    'Converts ``expr`` to a string of smtlib code.\n\n    Parameters\n    ==========\n\n    expr : Expr | List[Expr]\n        A SymPy expression or system to be converted.\n    auto_assert : bool, optional\n        If false, do not modify expr and produce only the S-Expression equivalent of expr.\n        If true, assume expr is a system and assert each boolean element.\n    auto_declare : bool, optional\n        If false, do not produce declarations for the symbols used in expr.\n        If true, prepend all necessary declarations for variables used in expr based on symbol_table.\n    precision : integer, optional\n        The ``evalf(..)`` precision for numbers such as pi.\n    symbol_table : dict, optional\n        A dictionary where keys are ``Symbol`` or ``Function`` instances and values are their Python type i.e. ``bool``, ``int``, ``float``, or ``Callable[...]``.\n        If incomplete, an attempt will be made to infer types from ``expr``.\n    known_types: dict, optional\n        A dictionary where keys are ``bool``, ``int``, ``float`` etc. and values are their corresponding SMT type names.\n        If not given, a partial listing compatible with several solvers will be used.\n    known_functions : dict, optional\n        A dictionary where keys are ``Function``, ``Relational``, ``BooleanFunction``, or ``Expr`` instances and values are their SMT string representations.\n        If not given, a partial listing optimized for dReal solver (but compatible with others) will be used.\n    known_constants: dict, optional\n        A dictionary where keys are ``NumberSymbol`` instances and values are their SMT variable names.\n        When using this feature, extra caution must be taken to avoid naming collisions between user symbols and listed constants.\n        If not given, constants will be expanded inline i.e. ``3.14159`` instead of ``MY_SMT_VARIABLE_FOR_PI``.\n    prefix_expressions: list, optional\n        A list of lists of ``str`` and/or expressions to convert into SMTLib and prefix to the output.\n    suffix_expressions: list, optional\n        A list of lists of ``str`` and/or expressions to convert into SMTLib and postfix to the output.\n    log_warn: lambda function, optional\n        A function to record all warnings during potentially risky operations.\n        Soundness is a core value in SMT solving, so it is good to log all assumptions made.\n\n    Examples\n    ========\n    >>> from sympy import smtlib_code, symbols, sin, Eq\n    >>> x = symbols(\'x\')\n    >>> smtlib_code(sin(x).series(x).removeO(), log_warn=print)\n    Could not infer type of `x`. Defaulting to float.\n    Non-Boolean expression `x**5/120 - x**3/6 + x` will not be asserted. Converting to SMTLib verbatim.\n    \'(declare-const x Real)\\n(+ x (* (/ -1 6) (pow x 3)) (* (/ 1 120) (pow x 5)))\'\n\n    >>> from sympy import Rational\n    >>> x, y, tau = symbols("x, y, tau")\n    >>> smtlib_code((2*tau)**Rational(7, 2), log_warn=print)\n    Could not infer type of `tau`. Defaulting to float.\n    Non-Boolean expression `8*sqrt(2)*tau**(7/2)` will not be asserted. Converting to SMTLib verbatim.\n    \'(declare-const tau Real)\\n(* 8 (pow 2 (/ 1 2)) (pow tau (/ 7 2)))\'\n\n    ``Piecewise`` expressions are implemented with ``ite`` expressions by default.\n    Note that if the ``Piecewise`` lacks a default term, represented by\n    ``(expr, True)`` then an error will be thrown.  This is to prevent\n    generating an expression that may not evaluate to anything.\n\n    >>> from sympy import Piecewise\n    >>> pw = Piecewise((x + 1, x > 0), (x, True))\n    >>> smtlib_code(Eq(pw, 3), symbol_table={x: float}, log_warn=print)\n    \'(declare-const x Real)\\n(assert (= (ite (> x 0) (+ 1 x) x) 3))\'\n\n    Custom printing can be defined for certain types by passing a dictionary of\n    PythonType : "SMT Name" to the ``known_types``, ``known_constants``, and ``known_functions`` kwargs.\n\n    >>> from typing import Callable\n    >>> from sympy import Function, Add\n    >>> f = Function(\'f\')\n    >>> g = Function(\'g\')\n    >>> smt_builtin_funcs = {  # functions our SMT solver will understand\n    ...   f: "existing_smtlib_fcn",\n    ...   Add: "sum",\n    ... }\n    >>> user_def_funcs = {  # functions defined by the user must have their types specified explicitly\n    ...   g: Callable[[int], float],\n    ... }\n    >>> smtlib_code(f(x) + g(x), symbol_table=user_def_funcs, known_functions=smt_builtin_funcs, log_warn=print)\n    Non-Boolean expression `f(x) + g(x)` will not be asserted. Converting to SMTLib verbatim.\n    \'(declare-const x Int)\\n(declare-fun g (Int) Real)\\n(sum (existing_smtlib_fcn x) (g x))\'\n    '
    log_warn = log_warn or (lambda _: None)
    if not isinstance(expr, list):
        expr = [expr]
    expr = [sympy.sympify(_, strict=True, evaluate=False, convert_xor=False) for _ in expr]
    if not symbol_table:
        symbol_table = {}
    symbol_table = _auto_infer_smtlib_types(*expr, symbol_table=symbol_table)
    settings = {}
    if precision:
        settings['precision'] = precision
    del precision
    if known_types:
        settings['known_types'] = known_types
    del known_types
    if known_functions:
        settings['known_functions'] = known_functions
    del known_functions
    if known_constants:
        settings['known_constants'] = known_constants
    del known_constants
    if not prefix_expressions:
        prefix_expressions = []
    if not suffix_expressions:
        suffix_expressions = []
    p = SMTLibPrinter(settings, symbol_table)
    del symbol_table
    for e in expr:
        for sym in e.atoms(Symbol, Function):
            if sym.is_Symbol and sym not in p._known_constants and (sym not in p.symbol_table):
                log_warn(f'Could not infer type of `{sym}`. Defaulting to float.')
                p.symbol_table[sym] = float
            if sym.is_Function and type(sym) not in p._known_functions and (type(sym) not in p.symbol_table) and (not sym.is_Piecewise):
                raise TypeError(f'Unknown type of undefined function `{sym}`. Must be mapped to ``str`` in known_functions or mapped to ``Callable[..]`` in symbol_table.')
    declarations = []
    if auto_declare:
        constants = {sym.name: sym for e in expr for sym in e.free_symbols if sym not in p._known_constants}
        functions = {fnc.name: fnc for e in expr for fnc in e.atoms(Function) if type(fnc) not in p._known_functions and (not fnc.is_Piecewise)}
        declarations = [_auto_declare_smtlib(sym, p, log_warn) for sym in constants.values()] + [_auto_declare_smtlib(fnc, p, log_warn) for fnc in functions.values()]
        declarations = [decl for decl in declarations if decl]
    if auto_assert:
        expr = [_auto_assert_smtlib(e, p, log_warn) for e in expr]
    return '\n'.join([*[e if isinstance(e, str) else p.doprint(e) for e in prefix_expressions], *sorted((e for e in declarations)), *[e if isinstance(e, str) else p.doprint(e) for e in expr], *[e if isinstance(e, str) else p.doprint(e) for e in suffix_expressions]])

def _auto_declare_smtlib(sym: typing.Union[Symbol, Function], p: SMTLibPrinter, log_warn: typing.Callable[[str], None]):
    if False:
        print('Hello World!')
    if sym.is_Symbol:
        type_signature = p.symbol_table[sym]
        assert isinstance(type_signature, type)
        type_signature = p._known_types[type_signature]
        return p._s_expr('declare-const', [sym, type_signature])
    elif sym.is_Function:
        type_signature = p.symbol_table[type(sym)]
        assert callable(type_signature)
        type_signature = [p._known_types[_] for _ in type_signature.__args__]
        assert len(type_signature) > 0
        params_signature = f"({' '.join(type_signature[:-1])})"
        return_signature = type_signature[-1]
        return p._s_expr('declare-fun', [type(sym), params_signature, return_signature])
    else:
        log_warn(f'Non-Symbol/Function `{sym}` will not be declared.')
        return None

def _auto_assert_smtlib(e: Expr, p: SMTLibPrinter, log_warn: typing.Callable[[str], None]):
    if False:
        while True:
            i = 10
    if isinstance(e, Boolean) or (e in p.symbol_table and p.symbol_table[e] == bool) or (e.is_Function and type(e) in p.symbol_table and (p.symbol_table[type(e)].__args__[-1] == bool)):
        return p._s_expr('assert', [e])
    else:
        log_warn(f'Non-Boolean expression `{e}` will not be asserted. Converting to SMTLib verbatim.')
        return e

def _auto_infer_smtlib_types(*exprs: Basic, symbol_table: typing.Optional[dict]=None) -> dict:
    if False:
        print('Hello World!')
    _symbols = dict(symbol_table) if symbol_table else {}

    def safe_update(syms: set, inf):
        if False:
            print('Hello World!')
        for s in syms:
            assert s.is_Symbol
            if (old_type := _symbols.setdefault(s, inf)) != inf:
                raise TypeError(f'Could not infer type of `{s}`. Apparently both `{old_type}` and `{inf}`?')
    safe_update({e for e in exprs if e.is_Symbol}, bool)
    safe_update({symbol for e in exprs for boolfunc in e.atoms(BooleanFunction) for symbol in boolfunc.args if symbol.is_Symbol}, bool)
    safe_update({symbol for e in exprs for boolfunc in e.atoms(Function) if type(boolfunc) in _symbols for (symbol, param) in zip(boolfunc.args, _symbols[type(boolfunc)].__args__) if symbol.is_Symbol and param == bool}, bool)
    safe_update({symbol for e in exprs for intfunc in e.atoms(Function) if type(intfunc) in _symbols for (symbol, param) in zip(intfunc.args, _symbols[type(intfunc)].__args__) if symbol.is_Symbol and param == int}, int)
    safe_update({symbol for e in exprs for symbol in e.atoms(Symbol) if symbol.is_integer}, int)
    safe_update({symbol for e in exprs for symbol in e.atoms(Symbol) if symbol.is_real and (not symbol.is_integer)}, float)
    rels = [rel for expr in exprs for rel in expr.atoms(Equality)]
    rels = [(rel.lhs, rel.rhs) for rel in rels if rel.lhs.is_Symbol] + [(rel.rhs, rel.lhs) for rel in rels if rel.rhs.is_Symbol]
    for (infer, reltd) in rels:
        inference = _symbols[infer] if infer in _symbols else _symbols[reltd] if reltd in _symbols else _symbols[type(reltd)].__args__[-1] if reltd.is_Function and type(reltd) in _symbols else bool if reltd.is_Boolean else int if reltd.is_integer or reltd.is_Integer else float if reltd.is_real else None
        if inference:
            safe_update({infer}, inference)
    return _symbols