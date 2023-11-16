"""Module provides ``CalciteBuilder`` class."""
from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import CalciteAggregateNode, CalciteBaseNode, CalciteCollation, CalciteFilterNode, CalciteInputIdxExpr, CalciteInputRefExpr, CalciteJoinNode, CalciteProjectionNode, CalciteScanNode, CalciteSortNode, CalciteUnionNode
from .dataframe.utils import ColNameCodec
from .df_algebra import FilterNode, FrameNode, GroupbyAggNode, JoinNode, MaskNode, SortNode, TransformNode, UnionNode
from .expr import AggregateExpr, InputRefExpr, LiteralExpr, OpExpr, _quantile_agg_dtype, build_if_then_else, build_row_idx_filter_expr

class CalciteBuilder:
    """Translator used to transform ``DFAlgNode`` tree into a calcite node sequence."""

    class CompoundAggregate:
        """
        A base class for a compound aggregate translation.

        Translation is done in three steps. Step 1 is an additional
        values generation using a projection. Step 2 is a generation
        of aggregates that will be later used for a compound aggregate
        value computation. Step 3 is a final aggregate value generation
        using another projection.

        Parameters
        ----------
        builder : CalciteBuilder
            A builder to use for translation.
        arg : BaseExpr or List of BaseExpr
            An aggregated values.
        """

        def __init__(self, builder, arg):
            if False:
                i = 10
                return i + 15
            self._builder = builder
            self._arg = arg

        def gen_proj_exprs(self):
            if False:
                return 10
            '\n            Generate values required for intermediate aggregates computation.\n\n            Returns\n            -------\n            dict\n                New column expressions mapped to their names.\n            '
            return []

        def gen_agg_exprs(self):
            if False:
                return 10
            '\n            Generate intermediate aggregates required for a compound aggregate computation.\n\n            Returns\n            -------\n            dict\n                New aggregate expressions mapped to their names.\n            '
            pass

        def gen_reduce_expr(self):
            if False:
                i = 10
                return i + 15
            '\n            Generate an expression for a compound aggregate.\n\n            Returns\n            -------\n            BaseExpr\n                A final compound aggregate expression.\n            '
            pass

    class CompoundAggregateWithColArg(CompoundAggregate):
        """
        A base class for a compound aggregate that require a `LiteralExpr` column argument.

        This aggregate requires 2 arguments. The first argument is an `InputRefExpr`,
        refering to the aggregation column. The second argument is a `LiteralExpr`,
        this expression is added into the frame as a new column.

        Parameters
        ----------
        agg : str
            Aggregate name.
        builder : CalciteBuilder
            A builder to use for translation.
        arg : List of BaseExpr
            Aggregate arguments.
        dtype : dtype, optional
            Aggregate data type. If not specified, `_dtype` from the first argument is used.
        """

        def __init__(self, agg, builder, arg, dtype=None):
            if False:
                i = 10
                return i + 15
            assert isinstance(arg[0], InputRefExpr)
            assert isinstance(arg[1], LiteralExpr)
            super().__init__(builder, arg)
            self._agg = agg
            self._agg_column = f'{arg[0].column}__{agg}__'
            self._dtype = dtype or arg[0]._dtype

        def gen_proj_exprs(self):
            if False:
                while True:
                    i = 10
            return {self._agg_column: self._arg[1]}

        def gen_agg_exprs(self):
            if False:
                i = 10
                return i + 15
            frame = self._arg[0].modin_frame
            return {self._agg_column: AggregateExpr(self._agg, [self._builder._ref_idx(frame, self._arg[0].column), self._builder._ref_idx(frame, self._agg_column)], dtype=self._dtype)}

        def gen_reduce_expr(self):
            if False:
                for i in range(10):
                    print('nop')
            return self._builder._ref(self._arg[0].modin_frame, self._agg_column)

    class StdAggregate(CompoundAggregate):
        """
        A sample standard deviation aggregate generator.

        Parameters
        ----------
        builder : CalciteBuilder
            A builder to use for translation.
        arg : list of BaseExpr
            An aggregated value.
        """

        def __init__(self, builder, arg):
            if False:
                return 10
            assert isinstance(arg[0], InputRefExpr)
            super().__init__(builder, arg[0])
            self._quad_name = self._arg.column + '__quad__'
            self._sum_name = self._arg.column + '__sum__'
            self._quad_sum_name = self._arg.column + '__quad_sum__'
            self._count_name = self._arg.column + '__count__'

        def gen_proj_exprs(self):
            if False:
                return 10
            '\n            Generate values required for intermediate aggregates computation.\n\n            Returns\n            -------\n            dict\n                New column expressions mapped to their names.\n            '
            expr = self._builder._translate(self._arg.mul(self._arg))
            return {self._quad_name: expr}

        def gen_agg_exprs(self):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Generate intermediate aggregates required for a compound aggregate computation.\n\n            Returns\n            -------\n            dict\n                New aggregate expressions mapped to their names.\n            '
            count_expr = self._builder._translate(AggregateExpr('count', self._arg))
            sum_expr = self._builder._translate(AggregateExpr('sum', self._arg))
            self._sum_dtype = sum_expr._dtype
            qsum_expr = AggregateExpr('SUM', self._builder._ref_idx(self._arg.modin_frame, self._quad_name), dtype=sum_expr._dtype)
            return {self._sum_name: sum_expr, self._quad_sum_name: qsum_expr, self._count_name: count_expr}

        def gen_reduce_expr(self):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Generate an expression for a compound aggregate.\n\n            Returns\n            -------\n            BaseExpr\n                A final compound aggregate expression.\n            '
            count_expr = self._builder._ref(self._arg.modin_frame, self._count_name)
            count_expr._dtype = _get_dtype(int)
            sum_expr = self._builder._ref(self._arg.modin_frame, self._sum_name)
            sum_expr._dtype = self._sum_dtype
            qsum_expr = self._builder._ref(self._arg.modin_frame, self._quad_sum_name)
            qsum_expr._dtype = self._sum_dtype
            null_expr = LiteralExpr(None)
            count_or_null = build_if_then_else(count_expr.eq(LiteralExpr(0)), null_expr, count_expr, count_expr._dtype)
            count_m_1_or_null = build_if_then_else(count_expr.eq(LiteralExpr(1)), null_expr, count_expr.sub(LiteralExpr(1)), count_expr._dtype)
            return qsum_expr.sub(sum_expr.mul(sum_expr).truediv(count_or_null)).truediv(count_m_1_or_null).pow(LiteralExpr(0.5))

    class SkewAggregate(CompoundAggregate):
        """
        An unbiased skew aggregate generator.

        Parameters
        ----------
        builder : CalciteBuilder
            A builder to use for translation.
        arg : list of BaseExpr
            An aggregated value.
        """

        def __init__(self, builder, arg):
            if False:
                return 10
            assert isinstance(arg[0], InputRefExpr)
            super().__init__(builder, arg[0])
            self._quad_name = self._arg.column + '__quad__'
            self._cube_name = self._arg.column + '__cube__'
            self._sum_name = self._arg.column + '__sum__'
            self._quad_sum_name = self._arg.column + '__quad_sum__'
            self._cube_sum_name = self._arg.column + '__cube_sum__'
            self._count_name = self._arg.column + '__count__'

        def gen_proj_exprs(self):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Generate values required for intermediate aggregates computation.\n\n            Returns\n            -------\n            dict\n                New column expressions mapped to their names.\n            '
            quad_expr = self._builder._translate(self._arg.mul(self._arg))
            cube_expr = self._builder._translate(self._arg.mul(self._arg).mul(self._arg))
            return {self._quad_name: quad_expr, self._cube_name: cube_expr}

        def gen_agg_exprs(self):
            if False:
                return 10
            '\n            Generate intermediate aggregates required for a compound aggregate computation.\n\n            Returns\n            -------\n            dict\n                New aggregate expressions mapped to their names.\n            '
            count_expr = self._builder._translate(AggregateExpr('count', self._arg))
            sum_expr = self._builder._translate(AggregateExpr('sum', self._arg))
            self._sum_dtype = sum_expr._dtype
            qsum_expr = AggregateExpr('SUM', self._builder._ref_idx(self._arg.modin_frame, self._quad_name), dtype=sum_expr._dtype)
            csum_expr = AggregateExpr('SUM', self._builder._ref_idx(self._arg.modin_frame, self._cube_name), dtype=sum_expr._dtype)
            return {self._sum_name: sum_expr, self._quad_sum_name: qsum_expr, self._cube_sum_name: csum_expr, self._count_name: count_expr}

        def gen_reduce_expr(self):
            if False:
                while True:
                    i = 10
            '\n            Generate an expression for a compound aggregate.\n\n            Returns\n            -------\n            BaseExpr\n                A final compound aggregate expression.\n            '
            count_expr = self._builder._ref(self._arg.modin_frame, self._count_name)
            count_expr._dtype = _get_dtype(int)
            sum_expr = self._builder._ref(self._arg.modin_frame, self._sum_name)
            sum_expr._dtype = self._sum_dtype
            qsum_expr = self._builder._ref(self._arg.modin_frame, self._quad_sum_name)
            qsum_expr._dtype = self._sum_dtype
            csum_expr = self._builder._ref(self._arg.modin_frame, self._cube_sum_name)
            csum_expr._dtype = self._sum_dtype
            mean_expr = sum_expr.truediv(count_expr)
            part1 = count_expr.mul(count_expr.sub(LiteralExpr(1)).pow(LiteralExpr(0.5))).truediv(count_expr.sub(LiteralExpr(2)))
            part2 = csum_expr.sub(mean_expr.mul(qsum_expr).mul(LiteralExpr(3.0))).add(mean_expr.mul(mean_expr).mul(sum_expr).mul(LiteralExpr(2.0)))
            part3 = qsum_expr.sub(mean_expr.mul(sum_expr)).pow(LiteralExpr(1.5))
            skew_expr = part1.mul(part2).truediv(part3)
            return build_if_then_else(count_expr.le(LiteralExpr(2)), LiteralExpr(None), skew_expr, skew_expr._dtype)

    class TopkAggregate(CompoundAggregateWithColArg):
        """
        A TOP_K aggregate generator.

        Parameters
        ----------
        builder : CalciteBuilder
            A builder to use for translation.
        arg : List of BaseExpr
            An aggregated values.
        """

        def __init__(self, builder, arg):
            if False:
                i = 10
                return i + 15
            super().__init__('TOP_K', builder, arg)

        def gen_reduce_expr(self):
            if False:
                print('Hello World!')
            return OpExpr('PG_UNNEST', [super().gen_reduce_expr()], self._dtype)

    class QuantileAggregate(CompoundAggregateWithColArg):
        """
        A QUANTILE aggregate generator.

        Parameters
        ----------
        builder : CalciteBuilder
            A builder to use for translation.
        arg : List of BaseExpr
            A list of 3 values:
                0. InputRefExpr - the column to compute the quantiles for.
                1. LiteralExpr - the quantile value.
                2. str - the interpolation method to use.
        """

        def __init__(self, builder, arg):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__('QUANTILE', builder, arg, _quantile_agg_dtype(arg[0]._dtype))
            self._interpolation = arg[2].val.upper()

        def gen_agg_exprs(self):
            if False:
                return 10
            exprs = super().gen_agg_exprs()
            for expr in exprs.values():
                expr.interpolation = self._interpolation
            return exprs
    _compound_aggregates = {'std': StdAggregate, 'skew': SkewAggregate, 'nlargest': TopkAggregate, 'nsmallest': TopkAggregate, 'quantile': QuantileAggregate}

    class InputContext:
        """
        A class to track current input frames and corresponding nodes.

        Used to translate input column references to numeric indices.

        Parameters
        ----------
        input_frames : list of DFAlgNode
            Input nodes of the currently translated node.
        input_nodes : list of CalciteBaseNode
            Translated input nodes.

        Attributes
        ----------
        input_nodes : list of CalciteBaseNode
            Input nodes of the currently translated node.
        frame_to_node : dict
            Maps input frames to corresponding calcite nodes.
        input_offsets : dict
            Maps input frame to an input index used for its first column.
        replacements : dict
            Maps input frame to a new list of columns to use. Used when
            a single `DFAlgNode` is lowered into multiple computation
            steps, e.g. for compound aggregates requiring additional
            projections.
        """
        _simple_aggregates = {'sum': 'SUM', 'mean': 'AVG', 'max': 'MAX', 'min': 'MIN', 'size': 'COUNT', 'count': 'COUNT'}
        _no_arg_aggregates = {'size'}

        def __init__(self, input_frames, input_nodes):
            if False:
                while True:
                    i = 10
            self.input_nodes = input_nodes
            self.frame_to_node = {x: y for (x, y) in zip(input_frames, input_nodes)}
            self.input_offsets = {}
            self.replacements = {}
            offs = 0
            for frame in input_frames:
                self.input_offsets[frame] = offs
                offs += len(frame._table_cols)
                if isinstance(frame._op, FrameNode):
                    offs += 1

        def replace_input_node(self, frame, node, new_cols):
            if False:
                print('Hello World!')
            '\n            Use `node` as an input node for references to columns of `frame`.\n\n            Parameters\n            ----------\n            frame : DFAlgNode\n                Replaced input frame.\n            node : CalciteBaseNode\n                A new node to use.\n            new_cols : list of str\n                A new columns list to use.\n            '
            self.replacements[frame] = new_cols

        def _idx(self, frame, col):
            if False:
                return 10
            '\n            Get a numeric input index for an input column.\n\n            Parameters\n            ----------\n            frame : DFAlgNode\n                An input frame.\n            col : str\n                An input column.\n\n            Returns\n            -------\n            int\n            '
            assert frame in self.input_offsets, f'unexpected reference to {frame.id_str()}'
            offs = self.input_offsets[frame]
            if frame in self.replacements:
                return self.replacements[frame].index(col) + offs
            if col == ColNameCodec.ROWID_COL_NAME:
                if not isinstance(self.frame_to_node[frame], CalciteScanNode):
                    raise NotImplementedError('rowid can be accessed in materialized frames only')
                return len(frame._table_cols) + offs
            assert col in frame._table_cols, f"unexpected reference to '{col}' in {frame.id_str()}"
            return frame._table_cols.index(col) + offs

        def ref(self, frame, col):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Translate input column into ``CalciteInputRefExpr``.\n\n            Parameters\n            ----------\n            frame : DFAlgNode\n                An input frame.\n            col : str\n                An input column.\n\n            Returns\n            -------\n            CalciteInputRefExpr\n            '
            return CalciteInputRefExpr(self._idx(frame, col))

        def ref_idx(self, frame, col):
            if False:
                while True:
                    i = 10
            '\n            Translate input column into ``CalciteInputIdxExpr``.\n\n            Parameters\n            ----------\n            frame : DFAlgNode\n                An input frame.\n            col : str\n                An input column.\n\n            Returns\n            -------\n            CalciteInputIdxExpr\n            '
            return CalciteInputIdxExpr(self._idx(frame, col))

        def input_ids(self):
            if False:
                return 10
            '\n            Get ids of all input nodes.\n\n            Returns\n            -------\n            list of int\n            '
            return [x.id for x in self.input_nodes]

        def translate(self, expr):
            if False:
                i = 10
                return i + 15
            '\n            Translate an expression.\n\n            Translation is done by replacing ``InputRefExpr`` with\n            ``CalciteInputRefExpr`` and ``CalciteInputIdxExpr``.\n\n            Parameters\n            ----------\n            expr : BaseExpr\n                An expression to translate.\n\n            Returns\n            -------\n            BaseExpr\n                Translated expression.\n            '
            return self._maybe_copy_and_translate_expr(expr)

        def _maybe_copy_and_translate_expr(self, expr, ref_idx=False):
            if False:
                print('Hello World!')
            '\n            Translate an expression.\n\n            Translate an expression replacing ``InputRefExpr`` with ``CalciteInputRefExpr``\n            and ``CalciteInputIdxExpr``. An expression tree branches with input columns\n            are copied into a new tree, other branches are used as is.\n\n            Parameters\n            ----------\n            expr : BaseExpr\n                An expression to translate.\n            ref_idx : bool, default: False\n                If True then translate ``InputRefExpr`` to ``CalciteInputIdxExpr``,\n                use ``CalciteInputRefExr`` otherwise.\n\n            Returns\n            -------\n            BaseExpr\n                Translated expression.\n            '
            if isinstance(expr, InputRefExpr):
                if ref_idx:
                    return self.ref_idx(expr.modin_frame, expr.column)
                else:
                    return self.ref(expr.modin_frame, expr.column)
            if isinstance(expr, AggregateExpr):
                expr = expr.copy()
                if expr.agg in self._no_arg_aggregates:
                    expr.operands = []
                else:
                    expr.operands[0] = self._maybe_copy_and_translate_expr(expr.operands[0], True)
                expr.agg = self._simple_aggregates[expr.agg]
                return expr
            gen = expr.nested_expressions()
            for op in gen:
                expr = gen.send(self._maybe_copy_and_translate_expr(op))
            return expr

    class InputContextMgr:
        """
        A helper class to manage an input context stack.

        The class is designed to be used in a recursion with nested
        'with' statements.

        Parameters
        ----------
        builder : CalciteBuilder
            An outer builder.
        input_frames : list of DFAlgNode
            Input nodes for the new context.
        input_nodes : list of CalciteBaseNode
            Translated input nodes.

        Attributes
        ----------
        builder : CalciteBuilder
            An outer builder.
        input_frames : list of DFAlgNode
            Input nodes for the new context.
        input_nodes : list of CalciteBaseNode
            Translated input nodes.
        """

        def __init__(self, builder, input_frames, input_nodes):
            if False:
                print('Hello World!')
            self.builder = builder
            self.input_frames = input_frames
            self.input_nodes = input_nodes

        def __enter__(self):
            if False:
                while True:
                    i = 10
            '\n            Push new input context into the input context stack.\n\n            Returns\n            -------\n            InputContext\n                New input context.\n            '
            self.builder._input_ctx_stack.append(self.builder.InputContext(self.input_frames, self.input_nodes))
            return self.builder._input_ctx_stack[-1]

        def __exit__(self, type, value, traceback):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Pop current input context.\n\n            Parameters\n            ----------\n            type : Any\n                An exception type.\n            value : Any\n                An exception value.\n            traceback : Any\n                A traceback.\n            '
            self.builder._input_ctx_stack.pop()
    type_strings = {int: 'INTEGER', bool: 'BOOLEAN'}
    _bool_cast_aggregates = {'sum': _get_dtype(int), 'mean': _get_dtype(float), 'quantile': _get_dtype(float)}

    def __init__(self):
        if False:
            print('Hello World!')
        self._input_ctx_stack = []
        self.has_join = False
        self.has_groupby = False

    def build(self, op):
        if False:
            return 10
        '\n        Translate a ``DFAlgNode`` tree into a calcite nodes sequence.\n\n        Parameters\n        ----------\n        op : DFAlgNode\n            A tree to translate.\n\n        Returns\n        -------\n        list of CalciteBaseNode\n            The resulting calcite nodes sequence.\n        '
        CalciteBaseNode.reset_id()
        self.res = []
        self._to_calcite(op)
        return self.res

    def _add_projection(self, frame):
        if False:
            return 10
        "\n        Add a projection node to the resulting sequence.\n\n        Added node simply selects all frame's columns. This method can be used\n        to discard a virtual 'rowid' column provided by all scan nodes.\n\n        Parameters\n        ----------\n        frame : HdkOnNativeDataframe\n            An input frame for a projection.\n\n        Returns\n        -------\n        CalciteProjectionNode\n            Created projection node.\n        "
        proj = CalciteProjectionNode(frame._table_cols, [self._ref(frame, col) for col in frame._table_cols])
        self._push(proj)
        return proj

    def _input_ctx(self):
        if False:
            print('Hello World!')
        '\n        Get current input context.\n\n        Returns\n        -------\n        InputContext\n        '
        return self._input_ctx_stack[-1]

    def _set_input_ctx(self, op):
        if False:
            return 10
        '\n        Create input context manager for a node translation.\n\n        Parameters\n        ----------\n        op : DFAlgNode\n            A translated node.\n\n        Returns\n        -------\n        InputContextMgr\n            Created input context manager.\n        '
        input_frames = getattr(op, 'input', [])
        input_nodes = [self._to_calcite(x._op) for x in input_frames]
        return self.InputContextMgr(self, input_frames, input_nodes)

    def _set_tmp_ctx(self, input_frames, input_nodes):
        if False:
            return 10
        '\n        Create a temporary input context manager.\n\n        This method is deprecated.\n\n        Parameters\n        ----------\n        input_frames : list of DFAlgNode\n            Input nodes of the currently translated node.\n        input_nodes : list of CalciteBaseNode\n            Translated input nodes.\n\n        Returns\n        -------\n        InputContextMgr\n            Created input context manager.\n        '
        return self.InputContextMgr(self, input_frames, input_nodes)

    def _ref(self, frame, col):
        if False:
            while True:
                i = 10
        '\n        Translate input column into ``CalciteInputRefExpr``.\n\n        Parameters\n        ----------\n        frame : DFAlgNode\n            An input frame.\n        col : str\n            An input column.\n\n        Returns\n        -------\n        CalciteInputRefExpr\n        '
        return self._input_ctx().ref(frame, col)

    def _ref_idx(self, frame, col):
        if False:
            return 10
        '\n        Translate input column into ``CalciteInputIdxExpr``.\n\n        Parameters\n        ----------\n        frame : DFAlgNode\n            An input frame.\n        col : str\n            An input column.\n\n        Returns\n        -------\n        CalciteInputIdxExpr\n        '
        return self._input_ctx().ref_idx(frame, col)

    def _translate(self, exprs):
        if False:
            i = 10
            return i + 15
        '\n        Translate expressions.\n\n        Translate expressions replacing ``InputRefExpr`` with ``CalciteInputRefExpr`` and\n        ``CalciteInputIdxExpr``.\n\n        Parameters\n        ----------\n        exprs : BaseExpr or list-like of BaseExpr\n            Expressions to translate.\n\n        Returns\n        -------\n        BaseExpr or list of BaseExpr\n            Translated expression.\n        '
        if isinstance(exprs, abc.Iterable):
            return [self._input_ctx().translate(x) for x in exprs]
        return self._input_ctx().translate(exprs)

    def _push(self, node):
        if False:
            i = 10
            return i + 15
        '\n        Append node to the resulting sequence.\n\n        Parameters\n        ----------\n        node : CalciteBaseNode\n            A node to add.\n        '
        if len(self.res) != 0 and isinstance(node, CalciteProjectionNode) and isinstance(self.res[-1], CalciteProjectionNode) and all((isinstance(expr, CalciteInputRefExpr) for expr in node.exprs)):
            last = self.res.pop()
            exprs = last.exprs
            last.reset_id(int(last.id))
            node = CalciteProjectionNode(node.fields, [exprs[expr.input] for expr in node.exprs])
        self.res.append(node)

    def _last(self):
        if False:
            while True:
                i = 10
        '\n        Get the last node of the resulting calcite node sequence.\n\n        Returns\n        -------\n        CalciteBaseNode\n        '
        return self.res[-1]

    def _input_nodes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get current input calcite nodes.\n\n        Returns\n        -------\n        list if CalciteBaseNode\n        '
        return self._input_ctx().input_nodes

    def _input_node(self, idx):
        if False:
            while True:
                i = 10
        "\n        Get an input calcite node by index.\n\n        Parameters\n        ----------\n        idx : int\n            An input node's index.\n\n        Returns\n        -------\n        CalciteBaseNode\n        "
        return self._input_nodes()[idx]

    def _input_ids(self):
        if False:
            i = 10
            return i + 15
        '\n        Get ids of the current input nodes.\n\n        Returns\n        -------\n        list of int\n        '
        return self._input_ctx().input_ids()

    def _to_calcite(self, op):
        if False:
            print('Hello World!')
        '\n        Translate tree to a calcite node sequence.\n\n        Parameters\n        ----------\n        op : DFAlgNode\n            A tree to translate.\n\n        Returns\n        -------\n        CalciteBaseNode\n            The last node of the generated sequence.\n        '
        with self._set_input_ctx(op):
            if isinstance(op, FrameNode):
                self._process_frame(op)
            elif isinstance(op, MaskNode):
                self._process_mask(op)
            elif isinstance(op, GroupbyAggNode):
                self._process_groupby(op)
            elif isinstance(op, TransformNode):
                self._process_transform(op)
            elif isinstance(op, JoinNode):
                self._process_join(op)
            elif isinstance(op, UnionNode):
                self._process_union(op)
            elif isinstance(op, SortNode):
                self._process_sort(op)
            elif isinstance(op, FilterNode):
                self._process_filter(op)
            else:
                raise NotImplementedError(f"CalciteBuilder doesn't support {type(op).__name__}")
        return self.res[-1]

    def _process_frame(self, op):
        if False:
            i = 10
            return i + 15
        '\n        Translate ``FrameNode`` node.\n\n        Parameters\n        ----------\n        op : FrameNode\n            A frame to translate.\n        '
        self._push(CalciteScanNode(op.modin_frame))

    def _process_mask(self, op):
        if False:
            while True:
                i = 10
        '\n        Translate ``MaskNode`` node.\n\n        Parameters\n        ----------\n        op : MaskNode\n            An operation to translate.\n        '
        if op.row_labels is not None:
            raise NotImplementedError('row indices masking is not yet supported')
        frame = op.input[0]
        rowid_col = self._ref(frame, ColNameCodec.ROWID_COL_NAME)
        condition = build_row_idx_filter_expr(op.row_positions, rowid_col)
        self._push(CalciteFilterNode(condition))
        self._add_projection(frame)

    def _process_groupby(self, op):
        if False:
            return 10
        '\n        Translate ``GroupbyAggNode`` node.\n\n        Parameters\n        ----------\n        op : GroupbyAggNode\n            An operation to translate.\n        '
        self.has_groupby = True
        frame = op.input[0]
        proj_cols = op.by.copy()
        for col in frame._table_cols:
            if col not in op.by:
                proj_cols.append(col)
        agg_exprs = op.agg_exprs
        cast_agg = self._bool_cast_aggregates
        if any((v.agg in cast_agg for v in agg_exprs.values())) and (bool_cols := {c: cast_agg[agg_exprs[c].agg] for (c, t) in frame.dtypes.items() if not isinstance(t, pandas.CategoricalDtype) and is_bool_dtype(t) and (agg_exprs[c].agg in cast_agg)}):
            trans = self._input_ctx()._maybe_copy_and_translate_expr
            proj_exprs = [trans(frame.ref(c).cast(bool_cols[c])) if c in bool_cols else self._ref(frame, c) for c in proj_cols]
        else:
            proj_exprs = [self._ref(frame, col) for col in proj_cols]
        compound_aggs = {}
        for (agg, expr) in agg_exprs.items():
            if expr.agg in self._compound_aggregates:
                compound_aggs[agg] = self._compound_aggregates[expr.agg](self, expr.operands)
                extra_exprs = compound_aggs[agg].gen_proj_exprs()
                proj_cols.extend(extra_exprs.keys())
                proj_exprs.extend(extra_exprs.values())
        proj = CalciteProjectionNode(proj_cols, proj_exprs)
        self._push(proj)
        self._input_ctx().replace_input_node(frame, proj, proj_cols)
        group = [self._ref_idx(frame, col) for col in op.by]
        fields = op.by.copy()
        aggs = []
        for (agg, expr) in agg_exprs.items():
            if agg in compound_aggs:
                extra_aggs = compound_aggs[agg].gen_agg_exprs()
                fields.extend(extra_aggs.keys())
                aggs.extend(extra_aggs.values())
            else:
                fields.append(agg)
                aggs.append(self._translate(expr))
        node = CalciteAggregateNode(fields, group, aggs)
        self._push(node)
        if compound_aggs:
            self._input_ctx().replace_input_node(frame, node, fields)
            proj_cols = op.by.copy()
            proj_exprs = [self._ref(frame, col) for col in proj_cols]
            proj_cols.extend(agg_exprs.keys())
            for agg in agg_exprs:
                if agg in compound_aggs:
                    proj_exprs.append(compound_aggs[agg].gen_reduce_expr())
                else:
                    proj_exprs.append(self._ref(frame, agg))
            proj = CalciteProjectionNode(proj_cols, proj_exprs)
            self._push(proj)
        if op.groupby_opts['sort']:
            collation = [CalciteCollation(col) for col in group]
            self._push(CalciteSortNode(collation))

    def _process_transform(self, op):
        if False:
            while True:
                i = 10
        '\n        Translate ``TransformNode`` node.\n\n        Parameters\n        ----------\n        op : TransformNode\n            An operation to translate.\n        '
        fields = list(op.exprs.keys())
        exprs = self._translate(op.exprs.values())
        self._push(CalciteProjectionNode(fields, exprs))

    def _process_join(self, op):
        if False:
            return 10
        '\n        Translate ``JoinNode`` node.\n\n        Parameters\n        ----------\n        op : JoinNode\n            An operation to translate.\n        '
        self.has_join = True
        node = CalciteJoinNode(left_id=self._input_node(0).id, right_id=self._input_node(1).id, how=op.how, condition=self._translate(op.condition))
        self._push(node)
        self._push(CalciteProjectionNode(op.exprs.keys(), [self._translate(val) for val in op.exprs.values()]))

    def _process_union(self, op):
        if False:
            print('Hello World!')
        '\n        Translate ``UnionNode`` node.\n\n        Parameters\n        ----------\n        op : UnionNode\n            An operation to translate.\n        '
        self._push(CalciteUnionNode(self._input_ids(), True))

    def _process_sort(self, op):
        if False:
            return 10
        '\n        Translate ``SortNode`` node.\n\n        Parameters\n        ----------\n        op : SortNode\n            An operation to translate.\n        '
        frame = op.input[0]
        if not isinstance(self._input_node(0), CalciteProjectionNode):
            proj = self._add_projection(frame)
            self._input_ctx().replace_input_node(frame, proj, frame._table_cols)
        nulls = op.na_position.upper()
        collations = []
        for (col, asc) in zip(op.columns, op.ascending):
            ascending = 'ASCENDING' if asc else 'DESCENDING'
            collations.append(CalciteCollation(self._ref_idx(frame, col), ascending, nulls))
        self._push(CalciteSortNode(collations))

    def _process_filter(self, op):
        if False:
            i = 10
            return i + 15
        '\n        Translate ``FilterNode`` node.\n\n        Parameters\n        ----------\n        op : FilterNode\n            An operation to translate.\n        '
        condition = self._translate(op.condition)
        self._push(CalciteFilterNode(condition))
        if isinstance(self._input_node(0), CalciteScanNode):
            self._add_projection(op.input[0])