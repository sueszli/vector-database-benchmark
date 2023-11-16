from sympy.core.containers import Tuple
from sympy.core.numbers import oo
from sympy.core.relational import Gt, Lt
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.logic.boolalg import And
from sympy.codegen.ast import Assignment, AddAugmentedAssignment, break_, CodeBlock, Declaration, FunctionDefinition, Print, Return, Scope, While, Variable, Pointer, real
from sympy.codegen.cfunctions import isnan
' This module collects functions for constructing ASTs representing algorithms. '

def newtons_method(expr, wrt, atol=1e-12, delta=None, *, rtol=4e-16, debug=False, itermax=None, counter=None, delta_fn=lambda e, x: -e / e.diff(x), cse=False, handle_nan=None, bounds=None):
    if False:
        while True:
            i = 10
    " Generates an AST for Newton-Raphson method (a root-finding algorithm).\n\n    Explanation\n    ===========\n\n    Returns an abstract syntax tree (AST) based on ``sympy.codegen.ast`` for Netwon's\n    method of root-finding.\n\n    Parameters\n    ==========\n\n    expr : expression\n    wrt : Symbol\n        With respect to, i.e. what is the variable.\n    atol : number or expression\n        Absolute tolerance (stopping criterion)\n    rtol : number or expression\n        Relative tolerance (stopping criterion)\n    delta : Symbol\n        Will be a ``Dummy`` if ``None``.\n    debug : bool\n        Whether to print convergence information during iterations\n    itermax : number or expr\n        Maximum number of iterations.\n    counter : Symbol\n        Will be a ``Dummy`` if ``None``.\n    delta_fn: Callable[[Expr, Symbol], Expr]\n        computes the step, default is newtons method. For e.g. Halley's method\n        use delta_fn=lambda e, x: -2*e*e.diff(x)/(2*e.diff(x)**2 - e*e.diff(x, 2))\n    cse: bool\n        Perform common sub-expression elimination on delta expression\n    handle_nan: Token\n        How to handle occurrence of not-a-number (NaN).\n    bounds: Optional[tuple[Expr, Expr]]\n        Perform optimization within bounds\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, cos\n    >>> from sympy.codegen.ast import Assignment\n    >>> from sympy.codegen.algorithms import newtons_method\n    >>> x, dx, atol = symbols('x dx atol')\n    >>> expr = cos(x) - x**3\n    >>> algo = newtons_method(expr, x, atol=atol, delta=dx)\n    >>> algo.has(Assignment(dx, -expr/expr.diff(x)))\n    True\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Newton%27s_method\n\n    "
    if delta is None:
        delta = Dummy()
        Wrapper = Scope
        name_d = 'delta'
    else:
        Wrapper = lambda x: x
        name_d = delta.name
    delta_expr = delta_fn(expr, wrt)
    if cse:
        from sympy.simplify.cse_main import cse
        (cses, (red,)) = cse([delta_expr.factor()])
        whl_bdy = [Assignment(dum, sub_e) for (dum, sub_e) in cses]
        whl_bdy += [Assignment(delta, red)]
    else:
        whl_bdy = [Assignment(delta, delta_expr)]
    if handle_nan is not None:
        whl_bdy += [While(isnan(delta), CodeBlock(handle_nan, break_))]
    whl_bdy += [AddAugmentedAssignment(wrt, delta)]
    if bounds is not None:
        whl_bdy += [Assignment(wrt, Min(Max(wrt, bounds[0]), bounds[1]))]
    if debug:
        prnt = Print([wrt, delta], '{}=%12.5g {}=%12.5g\\n'.format(wrt.name, name_d))
        whl_bdy += [prnt]
    req = Gt(Abs(delta), atol + rtol * Abs(wrt))
    declars = [Declaration(Variable(delta, type=real, value=oo))]
    if itermax is not None:
        counter = counter or Dummy(integer=True)
        v_counter = Variable.deduced(counter, 0)
        declars.append(Declaration(v_counter))
        whl_bdy.append(AddAugmentedAssignment(counter, 1))
        req = And(req, Lt(counter, itermax))
    whl = While(req, CodeBlock(*whl_bdy))
    blck = declars
    if debug:
        blck.append(Print([wrt], '{}=%12.5g\\n'.format(wrt.name)))
    blck += [whl]
    return Wrapper(CodeBlock(*blck))

def _symbol_of(arg):
    if False:
        i = 10
        return i + 15
    if isinstance(arg, Declaration):
        arg = arg.variable.symbol
    elif isinstance(arg, Variable):
        arg = arg.symbol
    return arg

def newtons_method_function(expr, wrt, params=None, func_name='newton', attrs=Tuple(), *, delta=None, **kwargs):
    if False:
        print('Hello World!')
    " Generates an AST for a function implementing the Newton-Raphson method.\n\n    Parameters\n    ==========\n\n    expr : expression\n    wrt : Symbol\n        With respect to, i.e. what is the variable\n    params : iterable of symbols\n        Symbols appearing in expr that are taken as constants during the iterations\n        (these will be accepted as parameters to the generated function).\n    func_name : str\n        Name of the generated function.\n    attrs : Tuple\n        Attribute instances passed as ``attrs`` to ``FunctionDefinition``.\n    \\*\\*kwargs :\n        Keyword arguments passed to :func:`sympy.codegen.algorithms.newtons_method`.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, cos\n    >>> from sympy.codegen.algorithms import newtons_method_function\n    >>> from sympy.codegen.pyutils import render_as_module\n    >>> x = symbols('x')\n    >>> expr = cos(x) - x**3\n    >>> func = newtons_method_function(expr, x)\n    >>> py_mod = render_as_module(func)  # source code as string\n    >>> namespace = {}\n    >>> exec(py_mod, namespace, namespace)\n    >>> res = eval('newton(0.5)', namespace)\n    >>> abs(res - 0.865474033102) < 1e-12\n    True\n\n    See Also\n    ========\n\n    sympy.codegen.algorithms.newtons_method\n\n    "
    if params is None:
        params = (wrt,)
    pointer_subs = {p.symbol: Symbol('(*%s)' % p.symbol.name) for p in params if isinstance(p, Pointer)}
    if delta is None:
        delta = Symbol('d_' + wrt.name)
        if expr.has(delta):
            delta = None
    algo = newtons_method(expr, wrt, delta=delta, **kwargs).xreplace(pointer_subs)
    if isinstance(algo, Scope):
        algo = algo.body
    not_in_params = expr.free_symbols.difference({_symbol_of(p) for p in params})
    if not_in_params:
        raise ValueError('Missing symbols in params: %s' % ', '.join(map(str, not_in_params)))
    declars = tuple((Variable(p, real) for p in params))
    body = CodeBlock(algo, Return(wrt))
    return FunctionDefinition(real, func_name, declars, body, attrs=attrs)