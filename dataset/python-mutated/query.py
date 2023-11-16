def _quote(to_quote):
    if False:
        return 10
    if '"' not in to_quote:
        return '"%s"' % to_quote
    return to_quote

class Query(object):
    """
     Dumb implementation of a Query object, using 3 string lists so far
     for backwards compatibility with the (table, where_clause, where_params) previously used.

     TODO: To be improved after v6.0 to rewrite part of the ORM and add support for:
      - auto-generated multiple table aliases
      - multiple joins to the same table with different conditions
      - dynamic right-hand-side values in domains  (e.g. a.name = a.description)
      - etc.
    """

    def __init__(self, tables=None, where_clause=None, where_clause_params=None, joins=None, extras=None):
        if False:
            while True:
                i = 10
        self.tables = tables or []
        self.where_clause = where_clause or []
        self.where_clause_params = where_clause_params or []
        self.joins = joins or {}
        self.extras = extras or {}

    def _get_table_aliases(self):
        if False:
            for i in range(10):
                print('nop')
        from odoo.osv.expression import get_alias_from_query
        return [get_alias_from_query(from_statement)[1] for from_statement in self.tables]

    def _get_alias_mapping(self):
        if False:
            for i in range(10):
                print('nop')
        from odoo.osv.expression import get_alias_from_query
        mapping = {}
        for table in self.tables:
            (alias, statement) = get_alias_from_query(table)
            mapping[statement] = table
        return mapping

    def add_join(self, connection, implicit=True, outer=False, extra=None, extra_params=[]):
        if False:
            for i in range(10):
                print('nop')
        ' Join a destination table to the current table.\n\n            :param implicit: False if the join is an explicit join. This allows\n                to fall back on the previous implementation of ``join`` before\n                OpenERP 7.0. It therefore adds the JOIN specified in ``connection``\n                If True, the join is done implicitely, by adding the table alias\n                in the from clause and the join condition in the where clause\n                of the query. Implicit joins do not handle outer, extra, extra_params parameters.\n            :param connection: a tuple ``(lhs, table, lhs_col, col, link)``.\n                The join corresponds to the SQL equivalent of::\n\n                (lhs.lhs_col = table.col)\n\n                Note that all connection elements are strings. Please refer to expression.py for more details about joins.\n\n            :param outer: True if a LEFT OUTER JOIN should be used, if possible\n                      (no promotion to OUTER JOIN is supported in case the JOIN\n                      was already present in the query, as for the moment\n                      implicit INNER JOINs are only connected from NON-NULL\n                      columns so it would not be correct (e.g. for\n                      ``_inherits`` or when a domain criterion explicitly\n                      adds filtering)\n\n            :param extra: A string with the extra join condition (SQL), or None.\n                This is used to provide an additional condition to the join\n                clause that cannot be added in the where clause (e.g., for LEFT\n                JOIN concerns). The condition string should refer to the table\n                aliases as "{lhs}" and "{rhs}".\n\n            :param extra_params: a list of parameters for the `extra` condition.\n        '
        from odoo.osv.expression import generate_table_alias
        (lhs, table, lhs_col, col, link) = connection
        (alias, alias_statement) = generate_table_alias(lhs, [(table, link)])
        if implicit:
            if alias_statement not in self.tables:
                self.tables.append(alias_statement)
                condition = '("%s"."%s" = "%s"."%s")' % (lhs, lhs_col, alias, col)
                self.where_clause.append(condition)
            else:
                pass
            return (alias, alias_statement)
        else:
            aliases = self._get_table_aliases()
            assert lhs in aliases, 'Left-hand-side table %s must already be part of the query tables %s!' % (lhs, str(self.tables))
            if alias_statement in self.tables:
                pass
            else:
                self.tables.append(alias_statement)
                join_tuple = (alias, lhs_col, col, outer and 'LEFT JOIN' or 'JOIN')
                self.joins.setdefault(lhs, []).append(join_tuple)
                if extra or extra_params:
                    extra = (extra or '').format(lhs=lhs, rhs=alias)
                    self.extras[lhs, join_tuple] = (extra, extra_params)
            return (alias, alias_statement)

    def get_sql(self):
        if False:
            print('Hello World!')
        ' Returns (query_from, query_where, query_params). '
        from odoo.osv.expression import get_alias_from_query
        tables_to_process = list(self.tables)
        alias_mapping = self._get_alias_mapping()
        from_clause = []
        from_params = []

        def add_joins_for_table(lhs):
            if False:
                while True:
                    i = 10
            for (rhs, lhs_col, rhs_col, join) in self.joins.get(lhs, []):
                tables_to_process.remove(alias_mapping[rhs])
                from_clause.append(' %s %s ON ("%s"."%s" = "%s"."%s"' % (join, alias_mapping[rhs], lhs, lhs_col, rhs, rhs_col))
                extra = self.extras.get((lhs, (rhs, lhs_col, rhs_col, join)))
                if extra:
                    if extra[0]:
                        from_clause.append(' AND ')
                        from_clause.append(extra[0])
                    if extra[1]:
                        from_params.extend(extra[1])
                from_clause.append(')')
                add_joins_for_table(rhs)
        for (pos, table) in enumerate(tables_to_process):
            if pos > 0:
                from_clause.append(',')
            from_clause.append(table)
            table_alias = get_alias_from_query(table)[1]
            if table_alias in self.joins:
                add_joins_for_table(table_alias)
        return (''.join(from_clause), ' AND '.join(self.where_clause), from_params + self.where_clause_params)

    def __str__(self):
        if False:
            print('Hello World!')
        return '<osv.Query: "SELECT ... FROM %s WHERE %s" with params: %r>' % self.get_sql()