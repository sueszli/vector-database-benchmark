import itertools
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import build_scope

def eliminate_subqueries(expression):
    if False:
        return 10
    '\n    Rewrite derived tables as CTES, deduplicating if possible.\n\n    Example:\n        >>> import sqlglot\n        >>> expression = sqlglot.parse_one("SELECT a FROM (SELECT * FROM x) AS y")\n        >>> eliminate_subqueries(expression).sql()\n        \'WITH y AS (SELECT * FROM x) SELECT a FROM y AS y\'\n\n    This also deduplicates common subqueries:\n        >>> expression = sqlglot.parse_one("SELECT a FROM (SELECT * FROM x) AS y CROSS JOIN (SELECT * FROM x) AS z")\n        >>> eliminate_subqueries(expression).sql()\n        \'WITH y AS (SELECT * FROM x) SELECT a FROM y AS y CROSS JOIN y AS z\'\n\n    Args:\n        expression (sqlglot.Expression): expression\n    Returns:\n        sqlglot.Expression: expression\n    '
    if isinstance(expression, exp.Subquery):
        eliminate_subqueries(expression.this)
        return expression
    root = build_scope(expression)
    if not root:
        return expression
    taken = {}
    for scope in root.cte_scopes:
        taken[scope.expression.parent.alias] = scope
    for scope in root.traverse():
        taken.update({source.name: source for (_, source) in scope.sources.items() if isinstance(source, exp.Table)})
    existing_ctes = {}
    with_ = root.expression.args.get('with')
    recursive = False
    if with_:
        recursive = with_.args.get('recursive')
        for cte in with_.expressions:
            existing_ctes[cte.this] = cte.alias
    new_ctes = []
    for cte_scope in root.cte_scopes:
        for scope in cte_scope.traverse():
            if scope is cte_scope:
                continue
            new_cte = _eliminate(scope, existing_ctes, taken)
            if new_cte:
                new_ctes.append(new_cte)
        new_ctes.append(cte_scope.expression.parent)
    for scope in itertools.chain(root.union_scopes, root.subquery_scopes, root.table_scopes):
        for child_scope in scope.traverse():
            new_cte = _eliminate(child_scope, existing_ctes, taken)
            if new_cte:
                new_ctes.append(new_cte)
    if new_ctes:
        expression.set('with', exp.With(expressions=new_ctes, recursive=recursive))
    return expression

def _eliminate(scope, existing_ctes, taken):
    if False:
        for i in range(10):
            print('nop')
    if scope.is_union:
        return _eliminate_union(scope, existing_ctes, taken)
    if scope.is_derived_table:
        return _eliminate_derived_table(scope, existing_ctes, taken)
    if scope.is_cte:
        return _eliminate_cte(scope, existing_ctes, taken)

def _eliminate_union(scope, existing_ctes, taken):
    if False:
        while True:
            i = 10
    duplicate_cte_alias = existing_ctes.get(scope.expression)
    alias = duplicate_cte_alias or find_new_name(taken=taken, base='cte')
    taken[alias] = scope
    expressions = scope.expression.selects
    selects = [exp.alias_(exp.column(e.alias_or_name, table=alias), alias=e.alias_or_name, copy=False) for e in expressions if e.alias_or_name]
    if len(selects) != len(expressions):
        selects = ['*']
    scope.expression.replace(exp.select(*selects).from_(exp.alias_(exp.table_(alias), alias=alias, copy=False)))
    if not duplicate_cte_alias:
        existing_ctes[scope.expression] = alias
        return exp.CTE(this=scope.expression, alias=exp.TableAlias(this=exp.to_identifier(alias)))

def _eliminate_derived_table(scope, existing_ctes, taken):
    if False:
        return 10
    if scope.parent.pivots or isinstance(scope.parent.expression, exp.Lateral):
        return None
    to_replace = scope.expression.parent.unwrap()
    (name, cte) = _new_cte(scope, existing_ctes, taken)
    table = exp.alias_(exp.table_(name), alias=to_replace.alias or name)
    table.set('joins', to_replace.args.get('joins'))
    to_replace.replace(table)
    return cte

def _eliminate_cte(scope, existing_ctes, taken):
    if False:
        for i in range(10):
            print('nop')
    parent = scope.expression.parent
    (name, cte) = _new_cte(scope, existing_ctes, taken)
    with_ = parent.parent
    parent.pop()
    if not with_.expressions:
        with_.pop()
    for child_scope in scope.parent.traverse():
        for (table, source) in child_scope.selected_sources.values():
            if source is scope:
                new_table = exp.alias_(exp.table_(name), alias=table.alias_or_name, copy=False)
                table.replace(new_table)
    return cte

def _new_cte(scope, existing_ctes, taken):
    if False:
        return 10
    '\n    Returns:\n        tuple of (name, cte)\n        where `name` is a new name for this CTE in the root scope and `cte` is a new CTE instance.\n        If this CTE duplicates an existing CTE, `cte` will be None.\n    '
    duplicate_cte_alias = existing_ctes.get(scope.expression)
    parent = scope.expression.parent
    name = parent.alias
    if not name:
        name = find_new_name(taken=taken, base='cte')
    if duplicate_cte_alias:
        name = duplicate_cte_alias
    elif taken.get(name):
        name = find_new_name(taken=taken, base=name)
    taken[name] = scope
    if not duplicate_cte_alias:
        existing_ctes[scope.expression] = name
        cte = exp.CTE(this=scope.expression, alias=exp.TableAlias(this=exp.to_identifier(name)))
    else:
        cte = None
    return (name, cte)