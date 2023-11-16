"""
Module to evaluate the proposition with assumptions using SAT algorithm.
"""
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.kind import NumberKind, UndefinedKind
from sympy.assumptions.ask_generated import get_all_known_matrix_facts, get_all_known_number_facts
from sympy.assumptions.assume import global_assumptions, AppliedPredicate
from sympy.assumptions.sathandlers import class_fact_registry
from sympy.core import oo
from sympy.logic.inference import satisfiable
from sympy.assumptions.cnf import CNF, EncodedCNF
from sympy.matrices.common import MatrixKind

def satask(proposition, assumptions=True, context=global_assumptions, use_known_facts=True, iterations=oo):
    if False:
        print('Hello World!')
    '\n    Function to evaluate the proposition with assumptions using SAT algorithm.\n\n    This function extracts every fact relevant to the expressions composing\n    proposition and assumptions. For example, if a predicate containing\n    ``Abs(x)`` is proposed, then ``Q.zero(Abs(x)) | Q.positive(Abs(x))``\n    will be found and passed to SAT solver because ``Q.nonnegative`` is\n    registered as a fact for ``Abs``.\n\n    Proposition is evaluated to ``True`` or ``False`` if the truth value can be\n    determined. If not, ``None`` is returned.\n\n    Parameters\n    ==========\n\n    proposition : Any boolean expression.\n        Proposition which will be evaluated to boolean value.\n\n    assumptions : Any boolean expression, optional.\n        Local assumptions to evaluate the *proposition*.\n\n    context : AssumptionsContext, optional.\n        Default assumptions to evaluate the *proposition*. By default,\n        this is ``sympy.assumptions.global_assumptions`` variable.\n\n    use_known_facts : bool, optional.\n        If ``True``, facts from ``sympy.assumptions.ask_generated``\n        module are passed to SAT solver as well.\n\n    iterations : int, optional.\n        Number of times that relevant facts are recursively extracted.\n        Default is infinite times until no new fact is found.\n\n    Returns\n    =======\n\n    ``True``, ``False``, or ``None``\n\n    Examples\n    ========\n\n    >>> from sympy import Abs, Q\n    >>> from sympy.assumptions.satask import satask\n    >>> from sympy.abc import x\n    >>> satask(Q.zero(Abs(x)), Q.zero(x))\n    True\n\n    '
    props = CNF.from_prop(proposition)
    _props = CNF.from_prop(~proposition)
    assumptions = CNF.from_prop(assumptions)
    context_cnf = CNF()
    if context:
        context_cnf = context_cnf.extend(context)
    sat = get_all_relevant_facts(props, assumptions, context_cnf, use_known_facts=use_known_facts, iterations=iterations)
    sat.add_from_cnf(assumptions)
    if context:
        sat.add_from_cnf(context_cnf)
    return check_satisfiability(props, _props, sat)

def check_satisfiability(prop, _prop, factbase):
    if False:
        return 10
    sat_true = factbase.copy()
    sat_false = factbase.copy()
    sat_true.add_from_cnf(prop)
    sat_false.add_from_cnf(_prop)
    can_be_true = satisfiable(sat_true)
    can_be_false = satisfiable(sat_false)
    if can_be_true and can_be_false:
        return None
    if can_be_true and (not can_be_false):
        return True
    if not can_be_true and can_be_false:
        return False
    if not can_be_true and (not can_be_false):
        raise ValueError('Inconsistent assumptions')

def extract_predargs(proposition, assumptions=None, context=None):
    if False:
        return 10
    '\n    Extract every expression in the argument of predicates from *proposition*,\n    *assumptions* and *context*.\n\n    Parameters\n    ==========\n\n    proposition : sympy.assumptions.cnf.CNF\n\n    assumptions : sympy.assumptions.cnf.CNF, optional.\n\n    context : sympy.assumptions.cnf.CNF, optional.\n        CNF generated from assumptions context.\n\n    Examples\n    ========\n\n    >>> from sympy import Q, Abs\n    >>> from sympy.assumptions.cnf import CNF\n    >>> from sympy.assumptions.satask import extract_predargs\n    >>> from sympy.abc import x, y\n    >>> props = CNF.from_prop(Q.zero(Abs(x*y)))\n    >>> assump = CNF.from_prop(Q.zero(x) & Q.zero(y))\n    >>> extract_predargs(props, assump)\n    {x, y, Abs(x*y)}\n\n    '
    req_keys = find_symbols(proposition)
    keys = proposition.all_predicates()
    lkeys = set()
    if assumptions:
        lkeys |= assumptions.all_predicates()
    if context:
        lkeys |= context.all_predicates()
    lkeys = lkeys - {S.true, S.false}
    tmp_keys = None
    while tmp_keys != set():
        tmp = set()
        for l in lkeys:
            syms = find_symbols(l)
            if syms & req_keys != set():
                tmp |= syms
        tmp_keys = tmp - req_keys
        req_keys |= tmp_keys
    keys |= {l for l in lkeys if find_symbols(l) & req_keys != set()}
    exprs = set()
    for key in keys:
        if isinstance(key, AppliedPredicate):
            exprs |= set(key.arguments)
        else:
            exprs.add(key)
    return exprs

def find_symbols(pred):
    if False:
        i = 10
        return i + 15
    '\n    Find every :obj:`~.Symbol` in *pred*.\n\n    Parameters\n    ==========\n\n    pred : sympy.assumptions.cnf.CNF, or any Expr.\n\n    '
    if isinstance(pred, CNF):
        symbols = set()
        for a in pred.all_predicates():
            symbols |= find_symbols(a)
        return symbols
    return pred.atoms(Symbol)

def get_relevant_clsfacts(exprs, relevant_facts=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Extract relevant facts from the items in *exprs*. Facts are defined in\n    ``assumptions.sathandlers`` module.\n\n    This function is recursively called by ``get_all_relevant_facts()``.\n\n    Parameters\n    ==========\n\n    exprs : set\n        Expressions whose relevant facts are searched.\n\n    relevant_facts : sympy.assumptions.cnf.CNF, optional.\n        Pre-discovered relevant facts.\n\n    Returns\n    =======\n\n    exprs : set\n        Candidates for next relevant fact searching.\n\n    relevant_facts : sympy.assumptions.cnf.CNF\n        Updated relevant facts.\n\n    Examples\n    ========\n\n    Here, we will see how facts relevant to ``Abs(x*y)`` are recursively\n    extracted. On the first run, set containing the expression is passed\n    without pre-discovered relevant facts. The result is a set containing\n    candidates for next run, and ``CNF()`` instance containing facts\n    which are relevant to ``Abs`` and its argument.\n\n    >>> from sympy import Abs\n    >>> from sympy.assumptions.satask import get_relevant_clsfacts\n    >>> from sympy.abc import x, y\n    >>> exprs = {Abs(x*y)}\n    >>> exprs, facts = get_relevant_clsfacts(exprs)\n    >>> exprs\n    {x*y}\n    >>> facts.clauses #doctest: +SKIP\n    {frozenset({Literal(Q.odd(Abs(x*y)), False), Literal(Q.odd(x*y), True)}),\n    frozenset({Literal(Q.zero(Abs(x*y)), False), Literal(Q.zero(x*y), True)}),\n    frozenset({Literal(Q.even(Abs(x*y)), False), Literal(Q.even(x*y), True)}),\n    frozenset({Literal(Q.zero(Abs(x*y)), True), Literal(Q.zero(x*y), False)}),\n    frozenset({Literal(Q.even(Abs(x*y)), False),\n                Literal(Q.odd(Abs(x*y)), False),\n                Literal(Q.odd(x*y), True)}),\n    frozenset({Literal(Q.even(Abs(x*y)), False),\n                Literal(Q.even(x*y), True),\n                Literal(Q.odd(Abs(x*y)), False)}),\n    frozenset({Literal(Q.positive(Abs(x*y)), False),\n                Literal(Q.zero(Abs(x*y)), False)})}\n\n    We pass the first run's results to the second run, and get the expressions\n    for next run and updated facts.\n\n    >>> exprs, facts = get_relevant_clsfacts(exprs, relevant_facts=facts)\n    >>> exprs\n    {x, y}\n\n    On final run, no more candidate is returned thus we know that all\n    relevant facts are successfully retrieved.\n\n    >>> exprs, facts = get_relevant_clsfacts(exprs, relevant_facts=facts)\n    >>> exprs\n    set()\n\n    "
    if not relevant_facts:
        relevant_facts = CNF()
    newexprs = set()
    for expr in exprs:
        for fact in class_fact_registry(expr):
            newfact = CNF.to_CNF(fact)
            relevant_facts = relevant_facts._and(newfact)
            for key in newfact.all_predicates():
                if isinstance(key, AppliedPredicate):
                    newexprs |= set(key.arguments)
    return (newexprs - exprs, relevant_facts)

def get_all_relevant_facts(proposition, assumptions, context, use_known_facts=True, iterations=oo):
    if False:
        for i in range(10):
            print('nop')
    '\n    Extract all relevant facts from *proposition* and *assumptions*.\n\n    This function extracts the facts by recursively calling\n    ``get_relevant_clsfacts()``. Extracted facts are converted to\n    ``EncodedCNF`` and returned.\n\n    Parameters\n    ==========\n\n    proposition : sympy.assumptions.cnf.CNF\n        CNF generated from proposition expression.\n\n    assumptions : sympy.assumptions.cnf.CNF\n        CNF generated from assumption expression.\n\n    context : sympy.assumptions.cnf.CNF\n        CNF generated from assumptions context.\n\n    use_known_facts : bool, optional.\n        If ``True``, facts from ``sympy.assumptions.ask_generated``\n        module are encoded as well.\n\n    iterations : int, optional.\n        Number of times that relevant facts are recursively extracted.\n        Default is infinite times until no new fact is found.\n\n    Returns\n    =======\n\n    sympy.assumptions.cnf.EncodedCNF\n\n    Examples\n    ========\n\n    >>> from sympy import Q\n    >>> from sympy.assumptions.cnf import CNF\n    >>> from sympy.assumptions.satask import get_all_relevant_facts\n    >>> from sympy.abc import x, y\n    >>> props = CNF.from_prop(Q.nonzero(x*y))\n    >>> assump = CNF.from_prop(Q.nonzero(x))\n    >>> context = CNF.from_prop(Q.nonzero(y))\n    >>> get_all_relevant_facts(props, assump, context) #doctest: +SKIP\n    <sympy.assumptions.cnf.EncodedCNF at 0x7f09faa6ccd0>\n\n    '
    i = 0
    relevant_facts = CNF()
    all_exprs = set()
    while True:
        if i == 0:
            exprs = extract_predargs(proposition, assumptions, context)
        all_exprs |= exprs
        (exprs, relevant_facts) = get_relevant_clsfacts(exprs, relevant_facts)
        i += 1
        if i >= iterations:
            break
        if not exprs:
            break
    if use_known_facts:
        known_facts_CNF = CNF()
        if any((expr.kind == MatrixKind(NumberKind) for expr in all_exprs)):
            known_facts_CNF.add_clauses(get_all_known_matrix_facts())
        if any((expr.kind == NumberKind or expr.kind == UndefinedKind for expr in all_exprs)):
            known_facts_CNF.add_clauses(get_all_known_number_facts())
        kf_encoded = EncodedCNF()
        kf_encoded.from_cnf(known_facts_CNF)

        def translate_literal(lit, delta):
            if False:
                print('Hello World!')
            if lit > 0:
                return lit + delta
            else:
                return lit - delta

        def translate_data(data, delta):
            if False:
                print('Hello World!')
            return [{translate_literal(i, delta) for i in clause} for clause in data]
        data = []
        symbols = []
        n_lit = len(kf_encoded.symbols)
        for (i, expr) in enumerate(all_exprs):
            symbols += [pred(expr) for pred in kf_encoded.symbols]
            data += translate_data(kf_encoded.data, i * n_lit)
        encoding = dict(list(zip(symbols, range(1, len(symbols) + 1))))
        ctx = EncodedCNF(data, encoding)
    else:
        ctx = EncodedCNF()
    ctx.add_from_cnf(relevant_facts)
    return ctx