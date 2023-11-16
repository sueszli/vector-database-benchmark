from collections import defaultdict
from sqlglot import alias, exp
from sqlglot.optimizer.qualify_columns import Resolver
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import ensure_schema
SELECT_ALL = object()
DEFAULT_SELECTION = lambda is_agg: alias(exp.Max(this=exp.Literal.number(1)) if is_agg else '1', '_')

def pushdown_projections(expression, schema=None, remove_unused_selections=True):
    if False:
        i = 10
        return i + 15
    '\n    Rewrite sqlglot AST to remove unused columns projections.\n\n    Example:\n        >>> import sqlglot\n        >>> sql = "SELECT y.a AS a FROM (SELECT x.a AS a, x.b AS b FROM x) AS y"\n        >>> expression = sqlglot.parse_one(sql)\n        >>> pushdown_projections(expression).sql()\n        \'SELECT y.a AS a FROM (SELECT x.a AS a FROM x) AS y\'\n\n    Args:\n        expression (sqlglot.Expression): expression to optimize\n        remove_unused_selections (bool): remove selects that are unused\n    Returns:\n        sqlglot.Expression: optimized expression\n    '
    schema = ensure_schema(schema)
    source_column_alias_count = {}
    referenced_columns = defaultdict(set)
    for scope in reversed(traverse_scope(expression)):
        parent_selections = referenced_columns.get(scope, {SELECT_ALL})
        alias_count = source_column_alias_count.get(scope, 0)
        if scope.expression.args.get('distinct') or (scope.parent and scope.parent.pivots):
            parent_selections = {SELECT_ALL}
        if isinstance(scope.expression, exp.Union):
            (left, right) = scope.union_scopes
            referenced_columns[left] = parent_selections
            if any((select.is_star for select in right.expression.selects)):
                referenced_columns[right] = parent_selections
            elif not any((select.is_star for select in left.expression.selects)):
                referenced_columns[right] = [right.expression.selects[i].alias_or_name for (i, select) in enumerate(left.expression.selects) if SELECT_ALL in parent_selections or select.alias_or_name in parent_selections]
        if isinstance(scope.expression, exp.Select):
            if remove_unused_selections:
                _remove_unused_selections(scope, parent_selections, schema, alias_count)
            if scope.expression.is_star:
                continue
            selects = defaultdict(set)
            for col in scope.columns:
                table_name = col.table
                col_name = col.name
                selects[table_name].add(col_name)
            for (name, (node, source)) in scope.selected_sources.items():
                if isinstance(source, Scope):
                    columns = selects.get(name) or set()
                    referenced_columns[source].update(columns)
                column_aliases = node.alias_column_names
                if column_aliases:
                    source_column_alias_count[source] = len(column_aliases)
    return expression

def _remove_unused_selections(scope, parent_selections, schema, alias_count):
    if False:
        for i in range(10):
            print('nop')
    order = scope.expression.args.get('order')
    if order:
        order_refs = {c.name for c in order.find_all(exp.Column) if not c.table}
    else:
        order_refs = set()
    new_selections = []
    removed = False
    star = False
    is_agg = False
    select_all = SELECT_ALL in parent_selections
    for selection in scope.expression.selects:
        name = selection.alias_or_name
        if select_all or name in parent_selections or name in order_refs or (alias_count > 0):
            new_selections.append(selection)
            alias_count -= 1
        else:
            if selection.is_star:
                star = True
            removed = True
        if not is_agg and selection.find(exp.AggFunc):
            is_agg = True
    if star:
        resolver = Resolver(scope, schema)
        names = {s.alias_or_name for s in new_selections}
        for name in sorted(parent_selections):
            if name not in names:
                new_selections.append(alias(exp.column(name, table=resolver.get_table(name)), name, copy=False))
    if not new_selections:
        new_selections.append(DEFAULT_SELECTION(is_agg))
    scope.expression.select(*new_selections, append=False, copy=False)
    if removed:
        scope.clear_cache()