from sqlglot import exp
from sqlglot.optimizer.normalize import normalized
from sqlglot.optimizer.scope import build_scope, find_in_scope
from sqlglot.optimizer.simplify import simplify

def pushdown_predicates(expression):
    if False:
        print('Hello World!')
    '\n    Rewrite sqlglot AST to pushdown predicates in FROMS and JOINS\n\n    Example:\n        >>> import sqlglot\n        >>> sql = "SELECT y.a AS a FROM (SELECT x.a AS a FROM x AS x) AS y WHERE y.a = 1"\n        >>> expression = sqlglot.parse_one(sql)\n        >>> pushdown_predicates(expression).sql()\n        \'SELECT y.a AS a FROM (SELECT x.a AS a FROM x AS x WHERE x.a = 1) AS y WHERE TRUE\'\n\n    Args:\n        expression (sqlglot.Expression): expression to optimize\n    Returns:\n        sqlglot.Expression: optimized expression\n    '
    root = build_scope(expression)
    if root:
        scope_ref_count = root.ref_count()
        for scope in reversed(list(root.traverse())):
            select = scope.expression
            where = select.args.get('where')
            if where:
                selected_sources = scope.selected_sources
                for (k, (node, source)) in selected_sources.items():
                    parent = node.find_ancestor(exp.Join, exp.From)
                    if isinstance(parent, exp.Join) and parent.side == 'RIGHT':
                        selected_sources = {k: (node, source)}
                        break
                pushdown(where.this, selected_sources, scope_ref_count)
            for join in select.args.get('joins') or []:
                name = join.alias_or_name
                if name in scope.selected_sources:
                    pushdown(join.args.get('on'), {name: scope.selected_sources[name]}, scope_ref_count)
    return expression

def pushdown(condition, sources, scope_ref_count):
    if False:
        for i in range(10):
            print('nop')
    if not condition:
        return
    condition = condition.replace(simplify(condition))
    cnf_like = normalized(condition) or not normalized(condition, dnf=True)
    predicates = list(condition.flatten() if isinstance(condition, exp.And if cnf_like else exp.Or) else [condition])
    if cnf_like:
        pushdown_cnf(predicates, sources, scope_ref_count)
    else:
        pushdown_dnf(predicates, sources, scope_ref_count)

def pushdown_cnf(predicates, scope, scope_ref_count):
    if False:
        print('Hello World!')
    '\n    If the predicates are in CNF like form, we can simply replace each block in the parent.\n    '
    for predicate in predicates:
        for node in nodes_for_predicate(predicate, scope, scope_ref_count).values():
            if isinstance(node, exp.Join):
                predicate.replace(exp.true())
                node.on(predicate, copy=False)
                break
            if isinstance(node, exp.Select):
                predicate.replace(exp.true())
                inner_predicate = replace_aliases(node, predicate)
                if find_in_scope(inner_predicate, exp.AggFunc):
                    node.having(inner_predicate, copy=False)
                else:
                    node.where(inner_predicate, copy=False)

def pushdown_dnf(predicates, scope, scope_ref_count):
    if False:
        i = 10
        return i + 15
    "\n    If the predicates are in DNF form, we can only push down conditions that are in all blocks.\n    Additionally, we can't remove predicates from their original form.\n    "
    pushdown_tables = set()
    for a in predicates:
        a_tables = exp.column_table_names(a)
        for b in predicates:
            a_tables &= exp.column_table_names(b)
        pushdown_tables.update(a_tables)
    conditions = {}
    for table in sorted(pushdown_tables):
        for predicate in predicates:
            nodes = nodes_for_predicate(predicate, scope, scope_ref_count)
            if table not in nodes:
                continue
            predicate_condition = None
            for column in predicate.find_all(exp.Column):
                if column.table == table:
                    condition = column.find_ancestor(exp.Condition)
                    predicate_condition = exp.and_(predicate_condition, condition) if predicate_condition else condition
            if predicate_condition:
                conditions[table] = exp.or_(conditions[table], predicate_condition) if table in conditions else predicate_condition
        for (name, node) in nodes.items():
            if name not in conditions:
                continue
            predicate = conditions[name]
            if isinstance(node, exp.Join):
                node.on(predicate, copy=False)
            elif isinstance(node, exp.Select):
                inner_predicate = replace_aliases(node, predicate)
                if find_in_scope(inner_predicate, exp.AggFunc):
                    node.having(inner_predicate, copy=False)
                else:
                    node.where(inner_predicate, copy=False)

def nodes_for_predicate(predicate, sources, scope_ref_count):
    if False:
        print('Hello World!')
    nodes = {}
    tables = exp.column_table_names(predicate)
    where_condition = isinstance(predicate.find_ancestor(exp.Join, exp.Where), exp.Where)
    for table in sorted(tables):
        (node, source) = sources.get(table) or (None, None)
        if node and where_condition:
            node = node.find_ancestor(exp.Join, exp.From)
        if isinstance(node, exp.From) and (not isinstance(source, exp.Table)):
            with_ = source.parent.expression.args.get('with')
            if with_ and with_.recursive:
                return {}
            node = source.expression
        if isinstance(node, exp.Join):
            if node.side and node.side != 'RIGHT':
                return {}
            nodes[table] = node
        elif isinstance(node, exp.Select) and len(tables) == 1:
            has_window_expression = any((select for select in node.selects if select.find(exp.Window)))
            if not node.args.get('group') and scope_ref_count[id(source)] < 2 and (not has_window_expression):
                nodes[table] = node
    return nodes

def replace_aliases(source, predicate):
    if False:
        while True:
            i = 10
    aliases = {}
    for select in source.selects:
        if isinstance(select, exp.Alias):
            aliases[select.alias] = select.this
        else:
            aliases[select.name] = select

    def _replace_alias(column):
        if False:
            while True:
                i = 10
        if isinstance(column, exp.Column) and column.name in aliases:
            return aliases[column.name].copy()
        return column
    return predicate.transform(_replace_alias)