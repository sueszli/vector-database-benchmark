from __future__ import annotations
import functools
import sqlalchemy as sa
import sqlglot as sg
import toolz
from sqlalchemy import sql
import ibis.common.exceptions as com
import ibis.expr.analysis as an
import ibis.expr.operations as ops
from ibis.backends.base import _SQLALCHEMY_TO_SQLGLOT_DIALECT
from ibis.backends.base.sql.alchemy.translator import AlchemyContext, AlchemyExprTranslator
from ibis.backends.base.sql.compiler import Compiler, Select, SelectBuilder, TableSetFormatter
from ibis.backends.base.sql.compiler.base import SetOp

class _AlchemyTableSetFormatter(TableSetFormatter):

    def get_result(self):
        if False:
            return 10
        op = self.node
        if isinstance(op, ops.Join):
            self._walk_join_tree(op)
        else:
            self.join_tables.append(self._format_table(op))
        result = self.join_tables[0]
        for (jtype, table, preds) in zip(self.join_types, self.join_tables[1:], self.join_predicates):
            if preds:
                sqla_preds = [self._translate(pred) for pred in preds]
                onclause = functools.reduce(sql.and_, sqla_preds)
            else:
                onclause = None
            if jtype is ops.InnerJoin:
                result = result.join(table, onclause)
            elif jtype is ops.CrossJoin:
                result = result.join(table, sa.literal(True))
            elif jtype is ops.LeftJoin:
                result = result.join(table, onclause, isouter=True)
            elif jtype is ops.RightJoin:
                result = table.join(result, onclause, isouter=True)
            elif jtype is ops.OuterJoin:
                result = result.outerjoin(table, onclause, full=True)
            elif jtype is ops.LeftSemiJoin:
                result = result.select().where(sa.exists(sa.select(1).where(onclause))).subquery()
            elif jtype is ops.LeftAntiJoin:
                result = result.select().where(~sa.exists(sa.select(1).where(onclause))).subquery()
            else:
                raise NotImplementedError(jtype)
        self.context.set_ref(op, result)
        return result

    def _get_join_type(self, op):
        if False:
            return 10
        return type(op)

    def _format_table(self, op):
        if False:
            while True:
                i = 10
        ctx = self.context
        orig_op = op
        if isinstance(op, (ops.SelfReference, ops.Sample)):
            op = op.table
        alias = ctx.get_ref(orig_op)
        translator = ctx.compiler.translator_class(op, ctx)
        if isinstance(op, ops.DatabaseTable):
            namespace = op.namespace
            result = op.source._get_sqla_table(op.name, namespace=namespace)
        elif isinstance(op, ops.UnboundTable):
            name = op.name
            namespace = op.namespace
            result = sa.Table(name, sa.MetaData(), *translator._schema_to_sqlalchemy_columns(op.schema), quote=translator._quote_table_names)
            dialect = translator._dialect_name
            result.fullname = sg.table(name, db=namespace.schema, catalog=namespace.database).sql(dialect=_SQLALCHEMY_TO_SQLGLOT_DIALECT.get(dialect, dialect))
        elif isinstance(op, ops.SQLQueryResult):
            columns = translator._schema_to_sqlalchemy_columns(op.schema)
            result = sa.text(op.query).columns(*columns)
        elif isinstance(op, ops.SQLStringView):
            columns = translator._schema_to_sqlalchemy_columns(op.schema)
            result = sa.text(op.query).columns(*columns).cte(op.name)
        elif isinstance(op, ops.View):
            child_expr = op.child.to_expr()
            definition = child_expr.compile()
            result = sa.Table(op.name, sa.MetaData(), *translator._schema_to_sqlalchemy_columns(op.schema), quote=translator._quote_table_names)
            backend = child_expr._find_backend()
            backend._create_temp_view(view=result, definition=definition)
        elif isinstance(op, ops.InMemoryTable):
            result = self._format_in_memory_table(op, translator)
        elif isinstance(op, ops.DummyTable):
            result = sa.select(*(translator.translate(value).label(name) for (name, value) in zip(op.schema.names, op.values)))
        elif ctx.is_extracted(op):
            if isinstance(orig_op, ops.SelfReference):
                result = ctx.get_ref(op)
            else:
                result = alias
        else:
            result = ctx.get_compiled_expr(op)
        result = alias if hasattr(alias, 'name') else result.alias(alias)
        if isinstance(orig_op, ops.Sample):
            result = self._format_sample(orig_op, result)
        ctx.set_ref(orig_op, result)
        return result

    def _format_sample(self, op, table):
        if False:
            for i in range(10):
                print('nop')
        raise com.UnsupportedOperationError('`Table.sample` is not supported')

    def _format_in_memory_table(self, op, translator):
        if False:
            for i in range(10):
                print('nop')
        columns = translator._schema_to_sqlalchemy_columns(op.schema)
        if self.context.compiler.cheap_in_memory_tables:
            result = sa.Table(op.name, sa.MetaData(), *columns, quote=translator._quote_table_names)
        elif not op.data:
            result = sa.select(*(translator.translate(ops.Literal(None, dtype=type_)).label(name) for (name, type_) in op.schema.items())).limit(0)
        elif self.context.compiler.support_values_syntax_in_select:
            rows = list(op.data.to_frame().itertuples(index=False))
            result = sa.values(*columns, name=op.name).data(rows).select().subquery()
        else:
            raw_rows = (sa.select(*(translator.translate(ops.Literal(val, dtype=type_)).label(name) for (val, (name, type_)) in zip(row, op.schema.items()))) for row in op.data.to_frame().itertuples(index=False))
            result = sa.union_all(*raw_rows).alias(op.name)
        return result

class AlchemySelect(Select):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.exists = kwargs.pop('exists', False)
        super().__init__(*args, **kwargs)

    def compile(self):
        if False:
            i = 10
            return i + 15
        self.context.set_query(self)
        self._compile_subqueries()
        frag = self._compile_table_set()
        steps = [self._add_select, self._add_group_by, self._add_where, self._add_order_by, self._add_limit]
        for step in steps:
            frag = step(frag)
        return frag

    def _compile_subqueries(self):
        if False:
            i = 10
            return i + 15
        if not self.subqueries:
            return
        for expr in self.subqueries:
            result = self.context.get_compiled_expr(expr)
            alias = self.context.get_ref(expr)
            result = result.cte(alias)
            self.context.set_ref(expr, result)

    def _compile_table_set(self):
        if False:
            for i in range(10):
                print('nop')
        if self.table_set is None:
            return None
        return self.table_set_formatter_class(self, self.table_set).get_result()

    def _add_select(self, table_set):
        if False:
            while True:
                i = 10
        if not self.select_set:
            return table_set.element
        to_select = []
        context = self.context
        select_set = self.select_set
        has_select_star = False
        for op in select_set:
            if isinstance(op, ops.Value):
                arg = self._translate(op, named=True)
            elif isinstance(op, ops.TableNode):
                arg = context.get_ref(op)
                if op.equals(self.table_set):
                    if (has_select_star := (arg is None)):
                        continue
                    else:
                        arg = table_set
                elif arg is None:
                    raise ValueError(op)
            else:
                raise TypeError(op)
            to_select.append(arg)
        if has_select_star:
            if table_set is None:
                raise ValueError('table_set cannot be None here')
            clauses = [table_set] + to_select
        else:
            clauses = to_select
        result_func = sa.exists if self.exists else sa.select
        result = result_func(*clauses)
        if self.distinct:
            result = result.distinct()
        unnest_children = []
        if not self.translator_class.supports_unnest_in_select:
            unnest_children.extend(map(context.get_ref, toolz.unique(an.find_toplevel_unnest_children(select_set))))
        if (has_select_star or table_set is None) and (not unnest_children):
            return result
        if unnest_children:
            table_set = functools.reduce(functools.partial(sa.sql.FromClause.join, onclause=sa.true()), toolz.unique(toolz.concatv(unnest_children, result.get_final_froms())))
        return result.select_from(table_set)

    def _add_group_by(self, fragment):
        if False:
            i = 10
            return i + 15
        nkeys = len(self.group_by)
        if not nkeys:
            return fragment
        if self.context.compiler.supports_indexed_grouping_keys:
            group_keys = map(sa.literal_column, map(str, range(1, nkeys + 1)))
        else:
            group_keys = map(self._translate, self.group_by)
        fragment = fragment.group_by(*group_keys)
        if self.having:
            having_args = [self._translate(arg) for arg in self.having]
            having_clause = functools.reduce(sql.and_, having_args)
            fragment = fragment.having(having_clause)
        return fragment

    def _add_where(self, fragment):
        if False:
            return 10
        if not self.where:
            return fragment
        args = [self._translate(pred, permit_subquery=True, within_where=True) for pred in self.where]
        clause = functools.reduce(sql.and_, args)
        return fragment.where(clause)

    def _add_order_by(self, fragment):
        if False:
            print('Hello World!')
        if not self.order_by:
            return fragment
        clauses = []
        for key in self.order_by:
            sort_expr = key.expr
            arg = self._translate(sort_expr)
            fn = sa.asc if key.ascending else sa.desc
            clauses.append(fn(arg))
        return fragment.order_by(*clauses)

    def _among_select_set(self, expr):
        if False:
            i = 10
            return i + 15
        return any((expr.equals(other) for other in self.select_set))

    def _add_limit(self, fragment):
        if False:
            for i in range(10):
                print('nop')
        if self.limit is None:
            return fragment
        frag = fragment
        n = self.limit.n
        if n is None:
            n = self.context.compiler.null_limit
        elif not isinstance(n, int):
            n = sa.select(self._translate(n)).select_from(frag.subquery()).scalar_subquery()
        if n is not None:
            try:
                fragment = fragment.limit(n)
            except AttributeError:
                fragment = fragment.subquery().select().limit(n)
        offset = self.limit.offset
        if not isinstance(offset, int):
            offset = sa.select(self._translate(offset)).select_from(frag.subquery()).scalar_subquery()
        if offset != 0 and n != 0:
            fragment = fragment.offset(offset)
        return fragment

class AlchemySelectBuilder(SelectBuilder):

    def _convert_group_by(self, exprs):
        if False:
            for i in range(10):
                print('nop')
        return exprs

    def _collect_SQLQueryResult(self, op, toplevel=False):
        if False:
            while True:
                i = 10
        if toplevel:
            self.table_set = op
            self.select_set = []

class AlchemySetOp(SetOp):

    def compile(self):
        if False:
            for i in range(10):
                print('nop')
        context = self.context
        distincts = self.distincts
        assert len(set(distincts)) == 1, "more than one distinct found; this shouldn't be possible because all unions are projected"
        func = self.distinct_func if distincts[0] else self.non_distinct_func
        return func(*(context.get_compiled_expr(table).cte().select() for table in self.tables))

class AlchemyUnion(AlchemySetOp):
    distinct_func = staticmethod(sa.union)
    non_distinct_func = staticmethod(sa.union_all)

class AlchemyIntersection(AlchemySetOp):
    distinct_func = staticmethod(sa.intersect)
    non_distinct_func = staticmethod(sa.intersect_all)

class AlchemyDifference(AlchemySetOp):
    distinct_func = staticmethod(sa.except_)
    non_distinct_func = staticmethod(sa.except_all)

class AlchemyCompiler(Compiler):
    translator_class = AlchemyExprTranslator
    context_class = AlchemyContext
    table_set_formatter_class = _AlchemyTableSetFormatter
    select_builder_class = AlchemySelectBuilder
    select_class = AlchemySelect
    union_class = AlchemyUnion
    intersect_class = AlchemyIntersection
    difference_class = AlchemyDifference
    supports_indexed_grouping_keys = True
    null_limit = sa.null()

    @classmethod
    def to_sql(cls, expr, context=None, params=None, exists=False):
        if False:
            for i in range(10):
                print('nop')
        if context is None:
            context = cls.make_context(params=params)
        query = cls.to_ast(expr, context).queries[0]
        if exists:
            query.exists = True
        return query.compile()