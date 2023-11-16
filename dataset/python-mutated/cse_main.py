""" Tools for doing common subexpression elimination.
"""
from collections import defaultdict
from sympy.core import Basic, Mul, Add, Pow, sympify
from sympy.core.containers import Tuple, OrderedSet
from sympy.core.exprtools import factor_terms
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import symbols, Symbol
from sympy.matrices import MatrixBase, Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix
from sympy.matrices.expressions import MatrixExpr, MatrixSymbol, MatMul, MatAdd, MatPow, Inverse
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.polys.rootoftools import RootOf
from sympy.utilities.iterables import numbered_symbols, sift, topological_sort, iterable
from . import cse_opts
basic_optimizations = [(cse_opts.sub_pre, cse_opts.sub_post), (factor_terms, None)]

def reps_toposort(r):
    if False:
        return 10
    "Sort replacements ``r`` so (k1, v1) appears before (k2, v2)\n    if k2 is in v1's free symbols. This orders items in the\n    way that cse returns its results (hence, in order to use the\n    replacements in a substitution option it would make sense\n    to reverse the order).\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.cse_main import reps_toposort\n    >>> from sympy.abc import x, y\n    >>> from sympy import Eq\n    >>> for l, r in reps_toposort([(x, y + 1), (y, 2)]):\n    ...     print(Eq(l, r))\n    ...\n    Eq(y, 2)\n    Eq(x, y + 1)\n\n    "
    r = sympify(r)
    E = []
    for (c1, (k1, v1)) in enumerate(r):
        for (c2, (k2, v2)) in enumerate(r):
            if k1 in v2.free_symbols:
                E.append((c1, c2))
    return [r[i] for i in topological_sort((range(len(r)), E))]

def cse_separate(r, e):
    if False:
        return 10
    "Move expressions that are in the form (symbol, expr) out of the\n    expressions and sort them into the replacements using the reps_toposort.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.cse_main import cse_separate\n    >>> from sympy.abc import x, y, z\n    >>> from sympy import cos, exp, cse, Eq, symbols\n    >>> x0, x1 = symbols('x:2')\n    >>> eq = (x + 1 + exp((x + 1)/(y + 1)) + cos(y + 1))\n    >>> cse([eq, Eq(x, z + 1), z - 2], postprocess=cse_separate) in [\n    ... [[(x0, y + 1), (x, z + 1), (x1, x + 1)],\n    ...  [x1 + exp(x1/x0) + cos(x0), z - 2]],\n    ... [[(x1, y + 1), (x, z + 1), (x0, x + 1)],\n    ...  [x0 + exp(x0/x1) + cos(x1), z - 2]]]\n    ...\n    True\n    "
    d = sift(e, lambda w: w.is_Equality and w.lhs.is_Symbol)
    r = r + [w.args for w in d[True]]
    e = d[False]
    return [reps_toposort(r), e]

def cse_release_variables(r, e):
    if False:
        return 10
    '\n    Return tuples giving ``(a, b)`` where ``a`` is a symbol and ``b`` is\n    either an expression or None. The value of None is used when a\n    symbol is no longer needed for subsequent expressions.\n\n    Use of such output can reduce the memory footprint of lambdified\n    expressions that contain large, repeated subexpressions.\n\n    Examples\n    ========\n\n    >>> from sympy import cse\n    >>> from sympy.simplify.cse_main import cse_release_variables\n    >>> from sympy.abc import x, y\n    >>> eqs = [(x + y - 1)**2, x, x + y, (x + y)/(2*x + 1) + (x + y - 1)**2, (2*x + 1)**(x + y)]\n    >>> defs, rvs = cse_release_variables(*cse(eqs))\n    >>> for i in defs:\n    ...   print(i)\n    ...\n    (x0, x + y)\n    (x1, (x0 - 1)**2)\n    (x2, 2*x + 1)\n    (_3, x0/x2 + x1)\n    (_4, x2**x0)\n    (x2, None)\n    (_0, x1)\n    (x1, None)\n    (_2, x0)\n    (x0, None)\n    (_1, x)\n    >>> print(rvs)\n    (_0, _1, _2, _3, _4)\n    '
    if not r:
        return (r, e)
    (s, p) = zip(*r)
    esyms = symbols('_:%d' % len(e))
    syms = list(esyms)
    s = list(s)
    in_use = set(s)
    p = list(p)
    e = [(e[i], syms[i]) for i in range(len(e))]
    (e, syms) = zip(*sorted(e, key=lambda x: -sum([p[s.index(i)].count_ops() for i in x[0].free_symbols & in_use])))
    syms = list(syms)
    p += e
    rv = []
    i = len(p) - 1
    while i >= 0:
        _p = p.pop()
        c = in_use & _p.free_symbols
        if c:
            rv.extend([(s, None) for s in sorted(c, key=str)])
        if i >= len(r):
            rv.append((syms.pop(), _p))
        else:
            rv.append((s[i], _p))
        in_use -= c
        i -= 1
    rv.reverse()
    return (rv, esyms)

def preprocess_for_cse(expr, optimizations):
    if False:
        for i in range(10):
            print('nop')
    ' Preprocess an expression to optimize for common subexpression\n    elimination.\n\n    Parameters\n    ==========\n\n    expr : SymPy expression\n        The target expression to optimize.\n    optimizations : list of (callable, callable) pairs\n        The (preprocessor, postprocessor) pairs.\n\n    Returns\n    =======\n\n    expr : SymPy expression\n        The transformed expression.\n    '
    for (pre, post) in optimizations:
        if pre is not None:
            expr = pre(expr)
    return expr

def postprocess_for_cse(expr, optimizations):
    if False:
        return 10
    'Postprocess an expression after common subexpression elimination to\n    return the expression to canonical SymPy form.\n\n    Parameters\n    ==========\n\n    expr : SymPy expression\n        The target expression to transform.\n    optimizations : list of (callable, callable) pairs, optional\n        The (preprocessor, postprocessor) pairs.  The postprocessors will be\n        applied in reversed order to undo the effects of the preprocessors\n        correctly.\n\n    Returns\n    =======\n\n    expr : SymPy expression\n        The transformed expression.\n    '
    for (pre, post) in reversed(optimizations):
        if post is not None:
            expr = post(expr)
    return expr

class FuncArgTracker:
    """
    A class which manages a mapping from functions to arguments and an inverse
    mapping from arguments to functions.
    """

    def __init__(self, funcs):
        if False:
            i = 10
            return i + 15
        self.value_numbers = {}
        self.value_number_to_value = []
        self.arg_to_funcset = []
        self.func_to_argset = []
        for (func_i, func) in enumerate(funcs):
            func_argset = OrderedSet()
            for func_arg in func.args:
                arg_number = self.get_or_add_value_number(func_arg)
                func_argset.add(arg_number)
                self.arg_to_funcset[arg_number].add(func_i)
            self.func_to_argset.append(func_argset)

    def get_args_in_value_order(self, argset):
        if False:
            return 10
        '\n        Return the list of arguments in sorted order according to their value\n        numbers.\n        '
        return [self.value_number_to_value[argn] for argn in sorted(argset)]

    def get_or_add_value_number(self, value):
        if False:
            i = 10
            return i + 15
        '\n        Return the value number for the given argument.\n        '
        nvalues = len(self.value_numbers)
        value_number = self.value_numbers.setdefault(value, nvalues)
        if value_number == nvalues:
            self.value_number_to_value.append(value)
            self.arg_to_funcset.append(OrderedSet())
        return value_number

    def stop_arg_tracking(self, func_i):
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove the function func_i from the argument to function mapping.\n        '
        for arg in self.func_to_argset[func_i]:
            self.arg_to_funcset[arg].remove(func_i)

    def get_common_arg_candidates(self, argset, min_func_i=0):
        if False:
            print('Hello World!')
        'Return a dict whose keys are function numbers. The entries of the dict are\n        the number of arguments said function has in common with\n        ``argset``. Entries have at least 2 items in common.  All keys have\n        value at least ``min_func_i``.\n        '
        count_map = defaultdict(lambda : 0)
        if not argset:
            return count_map
        funcsets = [self.arg_to_funcset[arg] for arg in argset]
        largest_funcset = max(funcsets, key=len)
        for funcset in funcsets:
            if largest_funcset is funcset:
                continue
            for func_i in funcset:
                if func_i >= min_func_i:
                    count_map[func_i] += 1
        (smaller_funcs_container, larger_funcs_container) = sorted([largest_funcset, count_map], key=len)
        for func_i in smaller_funcs_container:
            if count_map[func_i] < 1:
                continue
            if func_i in larger_funcs_container:
                count_map[func_i] += 1
        return {k: v for (k, v) in count_map.items() if v >= 2}

    def get_subset_candidates(self, argset, restrict_to_funcset=None):
        if False:
            while True:
                i = 10
        '\n        Return a set of functions each of which whose argument list contains\n        ``argset``, optionally filtered only to contain functions in\n        ``restrict_to_funcset``.\n        '
        iarg = iter(argset)
        indices = OrderedSet((fi for fi in self.arg_to_funcset[next(iarg)]))
        if restrict_to_funcset is not None:
            indices &= restrict_to_funcset
        for arg in iarg:
            indices &= self.arg_to_funcset[arg]
        return indices

    def update_func_argset(self, func_i, new_argset):
        if False:
            return 10
        '\n        Update a function with a new set of arguments.\n        '
        new_args = OrderedSet(new_argset)
        old_args = self.func_to_argset[func_i]
        for deleted_arg in old_args - new_args:
            self.arg_to_funcset[deleted_arg].remove(func_i)
        for added_arg in new_args - old_args:
            self.arg_to_funcset[added_arg].add(func_i)
        self.func_to_argset[func_i].clear()
        self.func_to_argset[func_i].update(new_args)

class Unevaluated:

    def __init__(self, func, args):
        if False:
            for i in range(10):
                print('nop')
        self.func = func
        self.args = args

    def __str__(self):
        if False:
            while True:
                i = 10
        return 'Uneval<{}>({})'.format(self.func, ', '.join((str(a) for a in self.args)))

    def as_unevaluated_basic(self):
        if False:
            i = 10
            return i + 15
        return self.func(*self.args, evaluate=False)

    @property
    def free_symbols(self):
        if False:
            print('Hello World!')
        return set().union(*[a.free_symbols for a in self.args])
    __repr__ = __str__

def match_common_args(func_class, funcs, opt_subs):
    if False:
        i = 10
        return i + 15
    '\n    Recognize and extract common subexpressions of function arguments within a\n    set of function calls. For instance, for the following function calls::\n\n        x + z + y\n        sin(x + y)\n\n    this will extract a common subexpression of `x + y`::\n\n        w = x + y\n        w + z\n        sin(w)\n\n    The function we work with is assumed to be associative and commutative.\n\n    Parameters\n    ==========\n\n    func_class: class\n        The function class (e.g. Add, Mul)\n    funcs: list of functions\n        A list of function calls.\n    opt_subs: dict\n        A dictionary of substitutions which this function may update.\n    '
    funcs = sorted(funcs, key=lambda f: len(f.args))
    arg_tracker = FuncArgTracker(funcs)
    changed = OrderedSet()
    for i in range(len(funcs)):
        common_arg_candidates_counts = arg_tracker.get_common_arg_candidates(arg_tracker.func_to_argset[i], min_func_i=i + 1)
        common_arg_candidates = OrderedSet(sorted(common_arg_candidates_counts.keys(), key=lambda k: (common_arg_candidates_counts[k], k)))
        while common_arg_candidates:
            j = common_arg_candidates.pop(last=False)
            com_args = arg_tracker.func_to_argset[i].intersection(arg_tracker.func_to_argset[j])
            if len(com_args) <= 1:
                continue
            diff_i = arg_tracker.func_to_argset[i].difference(com_args)
            if diff_i:
                com_func = Unevaluated(func_class, arg_tracker.get_args_in_value_order(com_args))
                com_func_number = arg_tracker.get_or_add_value_number(com_func)
                arg_tracker.update_func_argset(i, diff_i | OrderedSet([com_func_number]))
                changed.add(i)
            else:
                com_func_number = arg_tracker.get_or_add_value_number(funcs[i])
            diff_j = arg_tracker.func_to_argset[j].difference(com_args)
            arg_tracker.update_func_argset(j, diff_j | OrderedSet([com_func_number]))
            changed.add(j)
            for k in arg_tracker.get_subset_candidates(com_args, common_arg_candidates):
                diff_k = arg_tracker.func_to_argset[k].difference(com_args)
                arg_tracker.update_func_argset(k, diff_k | OrderedSet([com_func_number]))
                changed.add(k)
        if i in changed:
            opt_subs[funcs[i]] = Unevaluated(func_class, arg_tracker.get_args_in_value_order(arg_tracker.func_to_argset[i]))
        arg_tracker.stop_arg_tracking(i)

def opt_cse(exprs, order='canonical'):
    if False:
        print('Hello World!')
    "Find optimization opportunities in Adds, Muls, Pows and negative\n    coefficient Muls.\n\n    Parameters\n    ==========\n\n    exprs : list of SymPy expressions\n        The expressions to optimize.\n    order : string, 'none' or 'canonical'\n        The order by which Mul and Add arguments are processed. For large\n        expressions where speed is a concern, use the setting order='none'.\n\n    Returns\n    =======\n\n    opt_subs : dictionary of expression substitutions\n        The expression substitutions which can be useful to optimize CSE.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.cse_main import opt_cse\n    >>> from sympy.abc import x\n    >>> opt_subs = opt_cse([x**-2])\n    >>> k, v = list(opt_subs.keys())[0], list(opt_subs.values())[0]\n    >>> print((k, v.as_unevaluated_basic()))\n    (x**(-2), 1/(x**2))\n    "
    opt_subs = {}
    adds = OrderedSet()
    muls = OrderedSet()
    seen_subexp = set()
    collapsible_subexp = set()

    def _find_opts(expr):
        if False:
            i = 10
            return i + 15
        if not isinstance(expr, (Basic, Unevaluated)):
            return
        if expr.is_Atom or expr.is_Order:
            return
        if iterable(expr):
            list(map(_find_opts, expr))
            return
        if expr in seen_subexp:
            return expr
        seen_subexp.add(expr)
        list(map(_find_opts, expr.args))
        if not isinstance(expr, MatrixExpr) and expr.could_extract_minus_sign():
            if isinstance(expr, Add):
                neg_expr = Add(*(-i for i in expr.args))
            else:
                neg_expr = -expr
            if not neg_expr.is_Atom:
                opt_subs[expr] = Unevaluated(Mul, (S.NegativeOne, neg_expr))
                seen_subexp.add(neg_expr)
                expr = neg_expr
        if isinstance(expr, (Mul, MatMul)):
            if len(expr.args) == 1:
                collapsible_subexp.add(expr)
            else:
                muls.add(expr)
        elif isinstance(expr, (Add, MatAdd)):
            if len(expr.args) == 1:
                collapsible_subexp.add(expr)
            else:
                adds.add(expr)
        elif isinstance(expr, Inverse):
            pass
        elif isinstance(expr, (Pow, MatPow)):
            (base, exp) = (expr.base, expr.exp)
            if exp.could_extract_minus_sign():
                opt_subs[expr] = Unevaluated(Pow, (Pow(base, -exp), -1))
    for e in exprs:
        if isinstance(e, (Basic, Unevaluated)):
            _find_opts(e)
    edges = [(s, s.args[0]) for s in collapsible_subexp if s.args[0] in collapsible_subexp]
    for e in reversed(topological_sort((collapsible_subexp, edges))):
        opt_subs[e] = opt_subs.get(e.args[0], e.args[0])
    commutative_muls = OrderedSet()
    for m in muls:
        (c, nc) = m.args_cnc(cset=False)
        if c:
            c_mul = m.func(*c)
            if nc:
                if c_mul == 1:
                    new_obj = m.func(*nc)
                elif isinstance(m, MatMul):
                    new_obj = m.func(c_mul, *nc, evaluate=False)
                else:
                    new_obj = m.func(c_mul, m.func(*nc), evaluate=False)
                opt_subs[m] = new_obj
            if len(c) > 1:
                commutative_muls.add(c_mul)
    match_common_args(Add, adds, opt_subs)
    match_common_args(Mul, commutative_muls, opt_subs)
    return opt_subs

def tree_cse(exprs, symbols, opt_subs=None, order='canonical', ignore=()):
    if False:
        while True:
            i = 10
    "Perform raw CSE on expression tree, taking opt_subs into account.\n\n    Parameters\n    ==========\n\n    exprs : list of SymPy expressions\n        The expressions to reduce.\n    symbols : infinite iterator yielding unique Symbols\n        The symbols used to label the common subexpressions which are pulled\n        out.\n    opt_subs : dictionary of expression substitutions\n        The expressions to be substituted before any CSE action is performed.\n    order : string, 'none' or 'canonical'\n        The order by which Mul and Add arguments are processed. For large\n        expressions where speed is a concern, use the setting order='none'.\n    ignore : iterable of Symbols\n        Substitutions containing any Symbol from ``ignore`` will be ignored.\n    "
    if opt_subs is None:
        opt_subs = {}
    to_eliminate = set()
    seen_subexp = set()
    excluded_symbols = set()

    def _find_repeated(expr):
        if False:
            print('Hello World!')
        if not isinstance(expr, (Basic, Unevaluated)):
            return
        if isinstance(expr, RootOf):
            return
        if isinstance(expr, Basic) and (expr.is_Atom or expr.is_Order or isinstance(expr, (MatrixSymbol, MatrixElement))):
            if expr.is_Symbol:
                excluded_symbols.add(expr.name)
            return
        if iterable(expr):
            args = expr
        else:
            if expr in seen_subexp:
                for ign in ignore:
                    if ign in expr.free_symbols:
                        break
                else:
                    to_eliminate.add(expr)
                    return
            seen_subexp.add(expr)
            if expr in opt_subs:
                expr = opt_subs[expr]
            args = expr.args
        list(map(_find_repeated, args))
    for e in exprs:
        if isinstance(e, Basic):
            _find_repeated(e)
    symbols = (_ for _ in symbols if _.name not in excluded_symbols)
    replacements = []
    subs = {}

    def _rebuild(expr):
        if False:
            while True:
                i = 10
        if not isinstance(expr, (Basic, Unevaluated)):
            return expr
        if not expr.args:
            return expr
        if iterable(expr):
            new_args = [_rebuild(arg) for arg in expr.args]
            return expr.func(*new_args)
        if expr in subs:
            return subs[expr]
        orig_expr = expr
        if expr in opt_subs:
            expr = opt_subs[expr]
        if order != 'none':
            if isinstance(expr, (Mul, MatMul)):
                (c, nc) = expr.args_cnc()
                if c == [1]:
                    args = nc
                else:
                    args = list(ordered(c)) + nc
            elif isinstance(expr, (Add, MatAdd)):
                args = list(ordered(expr.args))
            else:
                args = expr.args
        else:
            args = expr.args
        new_args = list(map(_rebuild, args))
        if isinstance(expr, Unevaluated) or new_args != args:
            new_expr = expr.func(*new_args)
        else:
            new_expr = expr
        if orig_expr in to_eliminate:
            try:
                sym = next(symbols)
            except StopIteration:
                raise ValueError('Symbols iterator ran out of symbols.')
            if isinstance(orig_expr, MatrixExpr):
                sym = MatrixSymbol(sym.name, orig_expr.rows, orig_expr.cols)
            subs[orig_expr] = sym
            replacements.append((sym, new_expr))
            return sym
        else:
            return new_expr
    reduced_exprs = []
    for e in exprs:
        if isinstance(e, Basic):
            reduced_e = _rebuild(e)
        else:
            reduced_e = e
        reduced_exprs.append(reduced_e)
    return (replacements, reduced_exprs)

def cse(exprs, symbols=None, optimizations=None, postprocess=None, order='canonical', ignore=(), list=True):
    if False:
        for i in range(10):
            print('nop')
    ' Perform common subexpression elimination on an expression.\n\n    Parameters\n    ==========\n\n    exprs : list of SymPy expressions, or a single SymPy expression\n        The expressions to reduce.\n    symbols : infinite iterator yielding unique Symbols\n        The symbols used to label the common subexpressions which are pulled\n        out. The ``numbered_symbols`` generator is useful. The default is a\n        stream of symbols of the form "x0", "x1", etc. This must be an\n        infinite iterator.\n    optimizations : list of (callable, callable) pairs\n        The (preprocessor, postprocessor) pairs of external optimization\n        functions. Optionally \'basic\' can be passed for a set of predefined\n        basic optimizations. Such \'basic\' optimizations were used by default\n        in old implementation, however they can be really slow on larger\n        expressions. Now, no pre or post optimizations are made by default.\n    postprocess : a function which accepts the two return values of cse and\n        returns the desired form of output from cse, e.g. if you want the\n        replacements reversed the function might be the following lambda:\n        lambda r, e: return reversed(r), e\n    order : string, \'none\' or \'canonical\'\n        The order by which Mul and Add arguments are processed. If set to\n        \'canonical\', arguments will be canonically ordered. If set to \'none\',\n        ordering will be faster but dependent on expressions hashes, thus\n        machine dependent and variable. For large expressions where speed is a\n        concern, use the setting order=\'none\'.\n    ignore : iterable of Symbols\n        Substitutions containing any Symbol from ``ignore`` will be ignored.\n    list : bool, (default True)\n        Returns expression in list or else with same type as input (when False).\n\n    Returns\n    =======\n\n    replacements : list of (Symbol, expression) pairs\n        All of the common subexpressions that were replaced. Subexpressions\n        earlier in this list might show up in subexpressions later in this\n        list.\n    reduced_exprs : list of SymPy expressions\n        The reduced expressions with all of the replacements above.\n\n    Examples\n    ========\n\n    >>> from sympy import cse, SparseMatrix\n    >>> from sympy.abc import x, y, z, w\n    >>> cse(((w + x + y + z)*(w + y + z))/(w + x)**3)\n    ([(x0, y + z), (x1, w + x)], [(w + x0)*(x0 + x1)/x1**3])\n\n\n    List of expressions with recursive substitutions:\n\n    >>> m = SparseMatrix([x + y, x + y + z])\n    >>> cse([(x+y)**2, x + y + z, y + z, x + z + y, m])\n    ([(x0, x + y), (x1, x0 + z)], [x0**2, x1, y + z, x1, Matrix([\n    [x0],\n    [x1]])])\n\n    Note: the type and mutability of input matrices is retained.\n\n    >>> isinstance(_[1][-1], SparseMatrix)\n    True\n\n    The user may disallow substitutions containing certain symbols:\n\n    >>> cse([y**2*(x + 1), 3*y**2*(x + 1)], ignore=(y,))\n    ([(x0, x + 1)], [x0*y**2, 3*x0*y**2])\n\n    The default return value for the reduced expression(s) is a list, even if there is only\n    one expression. The `list` flag preserves the type of the input in the output:\n\n    >>> cse(x)\n    ([], [x])\n    >>> cse(x, list=False)\n    ([], x)\n    '
    if not list:
        return _cse_homogeneous(exprs, symbols=symbols, optimizations=optimizations, postprocess=postprocess, order=order, ignore=ignore)
    if isinstance(exprs, (int, float)):
        exprs = sympify(exprs)
    if isinstance(exprs, (Basic, MatrixBase)):
        exprs = [exprs]
    copy = exprs
    temp = []
    for e in exprs:
        if isinstance(e, (Matrix, ImmutableMatrix)):
            temp.append(Tuple(*e.flat()))
        elif isinstance(e, (SparseMatrix, ImmutableSparseMatrix)):
            temp.append(Tuple(*e.todok().items()))
        else:
            temp.append(e)
    exprs = temp
    del temp
    if optimizations is None:
        optimizations = []
    elif optimizations == 'basic':
        optimizations = basic_optimizations
    reduced_exprs = [preprocess_for_cse(e, optimizations) for e in exprs]
    if symbols is None:
        symbols = numbered_symbols(cls=Symbol)
    else:
        symbols = iter(symbols)
    opt_subs = opt_cse(reduced_exprs, order)
    (replacements, reduced_exprs) = tree_cse(reduced_exprs, symbols, opt_subs, order, ignore)
    exprs = copy
    for (i, (sym, subtree)) in enumerate(replacements):
        subtree = postprocess_for_cse(subtree, optimizations)
        replacements[i] = (sym, subtree)
    reduced_exprs = [postprocess_for_cse(e, optimizations) for e in reduced_exprs]
    for (i, e) in enumerate(exprs):
        if isinstance(e, (Matrix, ImmutableMatrix)):
            reduced_exprs[i] = Matrix(e.rows, e.cols, reduced_exprs[i])
            if isinstance(e, ImmutableMatrix):
                reduced_exprs[i] = reduced_exprs[i].as_immutable()
        elif isinstance(e, (SparseMatrix, ImmutableSparseMatrix)):
            m = SparseMatrix(e.rows, e.cols, {})
            for (k, v) in reduced_exprs[i]:
                m[k] = v
            if isinstance(e, ImmutableSparseMatrix):
                m = m.as_immutable()
            reduced_exprs[i] = m
    if postprocess is None:
        return (replacements, reduced_exprs)
    return postprocess(replacements, reduced_exprs)

def _cse_homogeneous(exprs, **kwargs):
    if False:
        return 10
    "\n    Same as ``cse`` but the ``reduced_exprs`` are returned\n    with the same type as ``exprs`` or a sympified version of the same.\n\n    Parameters\n    ==========\n\n    exprs : an Expr, iterable of Expr or dictionary with Expr values\n        the expressions in which repeated subexpressions will be identified\n    kwargs : additional arguments for the ``cse`` function\n\n    Returns\n    =======\n\n    replacements : list of (Symbol, expression) pairs\n        All of the common subexpressions that were replaced. Subexpressions\n        earlier in this list might show up in subexpressions later in this\n        list.\n    reduced_exprs : list of SymPy expressions\n        The reduced expressions with all of the replacements above.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.cse_main import cse\n    >>> from sympy import cos, Tuple, Matrix\n    >>> from sympy.abc import x\n    >>> output = lambda x: type(cse(x, list=False)[1])\n    >>> output(1)\n    <class 'sympy.core.numbers.One'>\n    >>> output('cos(x)')\n    <class 'str'>\n    >>> output(cos(x))\n    cos\n    >>> output(Tuple(1, x))\n    <class 'sympy.core.containers.Tuple'>\n    >>> output(Matrix([[1,0], [0,1]]))\n    <class 'sympy.matrices.dense.MutableDenseMatrix'>\n    >>> output([1, x])\n    <class 'list'>\n    >>> output((1, x))\n    <class 'tuple'>\n    >>> output({1, x})\n    <class 'set'>\n    "
    if isinstance(exprs, str):
        (replacements, reduced_exprs) = _cse_homogeneous(sympify(exprs), **kwargs)
        return (replacements, repr(reduced_exprs))
    if isinstance(exprs, (list, tuple, set)):
        (replacements, reduced_exprs) = cse(exprs, **kwargs)
        return (replacements, type(exprs)(reduced_exprs))
    if isinstance(exprs, dict):
        keys = list(exprs.keys())
        (replacements, values) = cse([exprs[k] for k in keys], **kwargs)
        reduced_exprs = dict(zip(keys, values))
        return (replacements, reduced_exprs)
    try:
        (replacements, (reduced_exprs,)) = cse(exprs, **kwargs)
    except TypeError:
        return ([], exprs)
    else:
        return (replacements, reduced_exprs)