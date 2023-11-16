from sqlglot.optimizer.scope import Scope, build_scope

def eliminate_ctes(expression):
    if False:
        print('Hello World!')
    '\n    Remove unused CTEs from an expression.\n\n    Example:\n        >>> import sqlglot\n        >>> sql = "WITH y AS (SELECT a FROM x) SELECT a FROM z"\n        >>> expression = sqlglot.parse_one(sql)\n        >>> eliminate_ctes(expression).sql()\n        \'SELECT a FROM z\'\n\n    Args:\n        expression (sqlglot.Expression): expression to optimize\n    Returns:\n        sqlglot.Expression: optimized expression\n    '
    root = build_scope(expression)
    if root:
        ref_count = root.ref_count()
        for scope in reversed(list(root.traverse())):
            if scope.is_cte:
                count = ref_count[id(scope)]
                if count <= 0:
                    cte_node = scope.expression.parent
                    with_node = cte_node.parent
                    cte_node.pop()
                    if len(with_node.expressions) <= 0:
                        with_node.pop()
                    for (_, source) in scope.selected_sources.values():
                        if isinstance(source, Scope):
                            ref_count[id(source)] -= 1
    return expression