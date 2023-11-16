from __future__ import annotations
from typing import NamedTuple
import ibis.expr.analysis as an
import ibis.expr.operations as ops

class _LimitSpec(NamedTuple):
    n: ops.Value | int | None
    offset: ops.Value | int = 0

class SelectBuilder:
    """Transforms expression IR to a query pipeline.

    There will typically be a primary SELECT query, perhaps with some
    subqueries and other DDL to ingest and tear down intermediate data sources.

    Walks the expression tree and catalogues distinct query units,
    builds select statements (and other DDL types, where necessary), and
    records relevant query unit aliases to be used when actually
    generating SQL.
    """

    def to_select(self, select_class, table_set_formatter_class, node, context, translator_class):
        if False:
            print('Hello World!')
        self.select_class = select_class
        self.table_set_formatter_class = table_set_formatter_class
        self.context = context
        self.translator_class = translator_class
        self.op = node.to_expr().as_table().op()
        assert isinstance(self.op, ops.Node), type(self.op)
        self.table_set = None
        self.select_set = None
        self.group_by = None
        self.having = None
        self.filters = []
        self.limit = None
        self.order_by = []
        self.subqueries = []
        self.distinct = False
        select_query = self._build_result_query()
        self.queries = [select_query]
        return select_query

    def _build_result_query(self):
        if False:
            while True:
                i = 10
        self._collect_elements()
        self._analyze_subqueries()
        self._populate_context()
        return self.select_class(self.table_set, list(self.select_set), translator_class=self.translator_class, table_set_formatter_class=self.table_set_formatter_class, context=self.context, subqueries=self.subqueries, where=self.filters, group_by=self.group_by, having=self.having, limit=self.limit, order_by=self.order_by, distinct=self.distinct, parent_op=self.op)

    def _populate_context(self):
        if False:
            for i in range(10):
                print('nop')
        if self.table_set is not None:
            self._make_table_aliases(self.table_set)

    def _make_table_aliases(self, node):
        if False:
            for i in range(10):
                print('nop')
        ctx = self.context
        if isinstance(node, ops.Join):
            for arg in node.args:
                if isinstance(arg, ops.TableNode):
                    self._make_table_aliases(arg)
        elif not ctx.is_extracted(node):
            ctx.make_alias(node)
        else:
            ctx.set_ref(node, ctx.top_context.get_ref(node))

    def _collect_elements(self):
        if False:
            print('Hello World!')
        if isinstance(self.op, ops.DummyTable):
            self.select_set = list(self.op.values)
        elif isinstance(self.op, ops.TableNode):
            self._collect(self.op, toplevel=True)
        else:
            self.select_set = [self.op]

    def _collect(self, op, toplevel=False):
        if False:
            while True:
                i = 10
        method = f'_collect_{type(op).__name__}'
        if hasattr(self, method):
            f = getattr(self, method)
            f(op, toplevel=toplevel)
        elif isinstance(op, (ops.PhysicalTable, ops.SQLQueryResult)):
            self._collect_PhysicalTable(op, toplevel=toplevel)
        elif isinstance(op, ops.Join):
            self._collect_Join(op, toplevel=toplevel)
        elif isinstance(op, ops.WindowingTVF):
            self._collect_WindowingTVF(op, toplevel=toplevel)
        else:
            raise NotImplementedError(type(op))

    def _collect_Distinct(self, op, toplevel=False):
        if False:
            while True:
                i = 10
        if toplevel:
            self.distinct = True
        self._collect(op.table, toplevel=toplevel)

    def _collect_Limit(self, op, toplevel=False):
        if False:
            return 10
        if toplevel:
            if isinstance((table := op.table), ops.Limit):
                self.table_set = table
                self.select_set = [table]
            else:
                self._collect(table, toplevel=toplevel)
            assert self.limit is None
            self.limit = _LimitSpec(op.n, op.offset)

    def _collect_Sample(self, op, toplevel=False):
        if False:
            while True:
                i = 10
        if toplevel:
            self.table_set = op
            self.select_set = [op]

    def _collect_Union(self, op, toplevel=False):
        if False:
            return 10
        if toplevel:
            self.table_set = op
            self.select_set = [op]

    def _collect_Difference(self, op, toplevel=False):
        if False:
            i = 10
            return i + 15
        if toplevel:
            self.table_set = op
            self.select_set = [op]

    def _collect_Intersection(self, op, toplevel=False):
        if False:
            return 10
        if toplevel:
            self.table_set = op
            self.select_set = [op]

    def _collect_Aggregation(self, op, toplevel=False):
        if False:
            print('Hello World!')
        if toplevel:
            self.group_by = self._convert_group_by(op.by)
            self.having = op.having
            self.select_set = op.by + op.metrics
            self.table_set = op.table
            self.filters = op.predicates
            self.order_by = op.sort_keys
            self._collect(op.table)

    def _collect_Selection(self, op, toplevel=False):
        if False:
            return 10
        table = op.table
        if toplevel:
            if isinstance(table, ops.Join):
                self._collect_Join(table)
            else:
                self._collect(table)
            selections = op.selections
            sort_keys = op.sort_keys
            filters = op.predicates
            if not selections:
                selections = [table]
            self.order_by = sort_keys
            self.select_set = selections
            self.table_set = table
            self.filters = filters

    def _collect_InMemoryTable(self, node, toplevel=False):
        if False:
            for i in range(10):
                print('nop')
        if toplevel:
            self.select_set = [node]
            self.table_set = node

    def _convert_group_by(self, nodes):
        if False:
            print('Hello World!')
        return list(range(len(nodes)))

    def _collect_Join(self, op, toplevel=False):
        if False:
            print('Hello World!')
        if toplevel:
            self.table_set = op
            self.select_set = [op]

    def _collect_PhysicalTable(self, op, toplevel=False):
        if False:
            return 10
        if toplevel:
            self.select_set = [op]
            self.table_set = op

    def _collect_DummyTable(self, op, toplevel=False):
        if False:
            i = 10
            return i + 15
        if toplevel:
            self.select_set = list(op.values)
            self.table_set = None

    def _collect_SelfReference(self, op, toplevel=False):
        if False:
            return 10
        if toplevel:
            self._collect(op.table, toplevel=toplevel)

    def _collect_WindowingTVF(self, op, toplevel=False):
        if False:
            return 10
        if toplevel:
            self.table_set = op
            self.select_set = [op]

    def _analyze_subqueries(self):
        if False:
            return 10
        subqueries = an.find_subqueries([self.table_set, *self.filters], min_dependents=2)
        self.subqueries = []
        for node in subqueries:
            if not self.context.is_extracted(node):
                self.subqueries.append(node)
                self.context.set_extracted(node)