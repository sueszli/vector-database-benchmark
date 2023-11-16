"""Module provides classes for lazy DataFrame algebra operations."""
import abc
import typing
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Union
import numpy as np
import pandas
import pyarrow as pa
from pandas.core.dtypes.common import is_string_dtype
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import EMPTY_ARROW_TABLE, ColNameCodec, get_common_arrow_type
from .db_worker import DbTable
from .expr import InputRefExpr, LiteralExpr, OpExpr
if TYPE_CHECKING:
    from .dataframe.dataframe import HdkOnNativeDataframe

class TransformMapper:
    """
    A helper class for ``InputMapper``.

    This class is used to map column references to expressions used
    for their computation. This mapper is used to fold expressions
    from multiple ``TransformNode``-s into a single expression.

    Parameters
    ----------
    op : TransformNode
        Transformation used for mapping.

    Attributes
    ----------
    _op : TransformNode
        Transformation used for mapping.
    """

    def __init__(self, op):
        if False:
            return 10
        self._op = op

    def translate(self, col):
        if False:
            return 10
        '\n        Translate column reference by its name.\n\n        Parameters\n        ----------\n        col : str\n            A name of the column to translate.\n\n        Returns\n        -------\n        BaseExpr\n            Translated expression.\n        '
        if col == ColNameCodec.ROWID_COL_NAME:
            return self._op.input[0].ref(col)
        return self._op.exprs[col]

class FrameMapper:
    """
    A helper class for ``InputMapper``.

    This class is used to map column references to another frame.
    This mapper is used to replace input frame in expressions.

    Parameters
    ----------
    frame : HdkOnNativeDataframe
        Target frame.

    Attributes
    ----------
    _frame : HdkOnNativeDataframe
        Target frame.
    """

    def __init__(self, frame):
        if False:
            i = 10
            return i + 15
        self._frame = frame

    def translate(self, col):
        if False:
            i = 10
            return i + 15
        '\n        Translate column reference by its name.\n\n        Parameters\n        ----------\n        col : str\n            A name of the column to translate.\n\n        Returns\n        -------\n        BaseExpr\n            Translated expression.\n        '
        return self._frame.ref(col)

class InputMapper:
    """
    Input reference mapper.

    This class is used for input translation/replacement in
    expressions via ``BaseExpr.translate_input`` method.

    Translation is performed using column mappers registered via
    `add_mapper` method. Each input frame can have at most one mapper.
    References to frames with no registered mapper are not translated.

    Attributes
    ----------
    _mappers : dict
        Column mappers to use for translation.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._mappers = {}

    def add_mapper(self, frame, mapper):
        if False:
            print('Hello World!')
        '\n        Register a mapper for a frame.\n\n        Parameters\n        ----------\n        frame : HdkOnNativeDataframe\n            A frame for which a mapper is registered.\n        mapper : object\n            A mapper to register.\n        '
        self._mappers[frame] = mapper

    def translate(self, ref):
        if False:
            print('Hello World!')
        '\n        Translate column reference by its name.\n\n        Parameters\n        ----------\n        ref : InputRefExpr\n            A column reference to translate.\n\n        Returns\n        -------\n        BaseExpr\n            Translated expression.\n        '
        if ref.modin_frame in self._mappers:
            return self._mappers[ref.modin_frame].translate(ref.column)
        return ref

class DFAlgNode(abc.ABC):
    """
    A base class for dataframe algebra tree node.

    A dataframe algebra tree is used to describe how dataframe is computed.

    Attributes
    ----------
    input : list of DFAlgNode, optional
        Holds child nodes.
    """

    @abc.abstractmethod
    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make a shallow copy of the node.\n\n        Returns\n        -------\n        DFAlgNode\n        '
        pass

    def walk_dfs(self, cb, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform a depth-first walk over a tree.\n\n        Walk over an input in the depth-first order and call a callback function\n        for each node.\n\n        Parameters\n        ----------\n        cb : callable\n            A callback function.\n        *args : list\n            Arguments for the callback.\n        **kwargs : dict\n            Keyword arguments for the callback.\n        '
        if hasattr(self, 'input'):
            for i in self.input:
                i._op.walk_dfs(cb, *args, **kwargs)
        cb(self, *args, **kwargs)

    def collect_partitions(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Collect all partitions participating in a tree.\n\n        Returns\n        -------\n        list\n            A list of collected partitions.\n        '
        partitions = []
        self.walk_dfs(lambda a, b: a._append_partitions(b), partitions)
        return partitions

    def collect_frames(self):
        if False:
            print('Hello World!')
        '\n        Collect all frames participating in a tree.\n\n        Returns\n        -------\n        list\n            A list of collected frames.\n        '
        frames = []
        self.walk_dfs(lambda a, b: a._append_frames(b), frames)
        return frames

    def require_executed_base(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if materialization of input frames is required.\n\n        Returns\n        -------\n        bool\n        '
        return False

    def can_execute_hdk(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Check for possibility of HDK execution.\n\n        Check if the computation can be executed using an HDK query.\n\n        Returns\n        -------\n        bool\n        '
        return True

    def can_execute_arrow(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Check for possibility of Arrow execution.\n\n        Check if the computation can be executed using\n        the Arrow API instead of HDK query.\n\n        Returns\n        -------\n        bool\n        '
        return False

    def execute_arrow(self, arrow_input: Union[None, pa.Table, List[pa.Table]]) -> pa.Table:
        if False:
            print('Hello World!')
        '\n        Compute the frame data using the Arrow API.\n\n        Parameters\n        ----------\n        arrow_input : None, pa.Table or list of pa.Table\n            The input, converted to arrow.\n\n        Returns\n        -------\n        pyarrow.Table\n            The resulting table.\n        '
        raise RuntimeError(f'Arrow execution is not supported by {type(self)}')

    def _append_partitions(self, partitions):
        if False:
            for i in range(10):
                print('nop')
        "\n        Append all used by the node partitions to `partitions` list.\n\n        The default implementation is no-op. This method should be\n        overriden by all nodes referencing frame's partitions.\n\n        Parameters\n        ----------\n        partitions : list\n            Output list of partitions.\n        "
        pass

    def _append_frames(self, frames):
        if False:
            return 10
        '\n        Append all used by the node frames to `frames` list.\n\n        The default implementation is no-op. This method should be\n        overriden by all nodes referencing frames.\n\n        Parameters\n        ----------\n        frames : list\n            Output list of frames.\n        '
        pass

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a string representation of the tree.\n\n        Returns\n        -------\n        str\n        '
        return self.dumps()

    def dump(self, prefix=''):
        if False:
            while True:
                i = 10
        "\n        Dump the tree.\n\n        Parameters\n        ----------\n        prefix : str, default: ''\n            A prefix to add at each string of the dump.\n        "
        print(self.dumps(prefix))

    def dumps(self, prefix=''):
        if False:
            return 10
        "\n        Return a string representation of the tree.\n\n        Parameters\n        ----------\n        prefix : str, default: ''\n            A prefix to add at each string of the dump.\n\n        Returns\n        -------\n        str\n        "
        return self._prints(prefix)

    @abc.abstractmethod
    def _prints(self, prefix):
        if False:
            return 10
        '\n        Return a string representation of the tree.\n\n        Parameters\n        ----------\n        prefix : str\n            A prefix to add at each string of the dump.\n\n        Returns\n        -------\n        str\n        '
        pass

    def _prints_input(self, prefix):
        if False:
            return 10
        "\n        Return a string representation of node's operands.\n\n        A helper method for `_prints` implementation in derived classes.\n\n        Parameters\n        ----------\n        prefix : str\n            A prefix to add at each string of the dump.\n\n        Returns\n        -------\n        str\n        "
        res = ''
        if hasattr(self, 'input'):
            for (i, node) in enumerate(self.input):
                if isinstance(node._op, FrameNode):
                    res += f'{prefix}input[{i}]: {node._op}\n'
                else:
                    res += f'{prefix}input[{i}]:\n' + node._op._prints(prefix + '  ')
        return res

class FrameNode(DFAlgNode):
    """
    A node to reference a materialized frame.

    Parameters
    ----------
    modin_frame : HdkOnNativeDataframe
        Referenced frame.

    Attributes
    ----------
    modin_frame : HdkOnNativeDataframe
        Referenced frame.
    """

    def __init__(self, modin_frame: 'HdkOnNativeDataframe'):
        if False:
            while True:
                i = 10
        self.modin_frame = modin_frame

    @_inherit_docstrings(DFAlgNode.can_execute_arrow)
    def can_execute_arrow(self) -> bool:
        if False:
            print('Hello World!')
        return self.modin_frame._has_arrow_table()

    def execute_arrow(self, ignore=None) -> Union[DbTable, pa.Table, pandas.DataFrame]:
        if False:
            print('Hello World!')
        '\n        Materialized frame.\n\n        If `can_execute_arrow` returns True, this method returns an arrow table,\n        otherwise - a pandas Dataframe or DbTable.\n\n        Parameters\n        ----------\n        ignore : None, pa.Table or list of pa.Table, default: None\n\n        Returns\n        -------\n        DbTable or pa.Table or pandas.Dataframe\n        '
        frame = self.modin_frame
        if frame._partitions is not None:
            return frame._partitions[0][0].get()
        if frame._has_unsupported_data:
            return pandas.DataFrame(index=frame._index_cache, columns=frame._columns_cache)
        if frame._index_cache or frame._columns_cache:
            return pa.Table.from_pandas(pandas.DataFrame(index=frame._index_cache, columns=frame._columns_cache))
        return EMPTY_ARROW_TABLE

    def copy(self):
        if False:
            while True:
                i = 10
        '\n        Make a shallow copy of the node.\n\n        Returns\n        -------\n        FrameNode\n        '
        return FrameNode(self.modin_frame)

    def _append_partitions(self, partitions):
        if False:
            i = 10
            return i + 15
        '\n        Append all partitions of the referenced frame to `partitions` list.\n\n        Parameters\n        ----------\n        partitions : list\n            Output list of partitions.\n        '
        partitions += self.modin_frame._partitions.flatten()

    def _append_frames(self, frames):
        if False:
            i = 10
            return i + 15
        '\n        Append the referenced frame to `frames` list.\n\n        Parameters\n        ----------\n        frames : list\n            Output list of frames.\n        '
        frames.append(self.modin_frame)

    def _prints(self, prefix):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a string representation of the tree.\n\n        Parameters\n        ----------\n        prefix : str\n            A prefix to add at each string of the dump.\n\n        Returns\n        -------\n        str\n        '
        return f'{prefix}{self.modin_frame.id_str()}'

class MaskNode(DFAlgNode):
    """
    A filtering node which filters rows by index values or row id.

    Parameters
    ----------
    base : HdkOnNativeDataframe
        A filtered frame.
    row_labels : list, optional
        List of row labels to select.
    row_positions : list of int, optional
        List of rows ids to select.

    Attributes
    ----------
    input : list of HdkOnNativeDataframe
        Holds a single filtered frame.
    row_labels : list or None
        List of row labels to select.
    row_positions : list of int or None
        List of rows ids to select.
    """

    def __init__(self, base: 'HdkOnNativeDataframe', row_labels: List[str]=None, row_positions: List[int]=None):
        if False:
            print('Hello World!')
        self.input = [base]
        self.row_labels = row_labels
        self.row_positions = row_positions

    @_inherit_docstrings(DFAlgNode.require_executed_base)
    def require_executed_base(self) -> bool:
        if False:
            while True:
                i = 10
        return True

    @_inherit_docstrings(DFAlgNode.can_execute_arrow)
    def can_execute_arrow(self) -> bool:
        if False:
            while True:
                i = 10
        return self.row_labels is None

    def execute_arrow(self, table: pa.Table) -> pa.Table:
        if False:
            return 10
        '\n        Perform row selection on the frame using Arrow API.\n\n        Parameters\n        ----------\n        table : pa.Table\n\n        Returns\n        -------\n        pyarrow.Table\n            The resulting table.\n        '
        row_positions = self.row_positions
        if not isinstance(row_positions, slice) and (not is_range_like(row_positions)):
            if not isinstance(row_positions, (pa.Array, np.ndarray, list)):
                row_positions = pa.array(row_positions)
            return table.take(row_positions)
        if isinstance(row_positions, slice):
            row_positions = range(*row_positions.indices(table.num_rows))
        (start, stop, step) = (row_positions.start, row_positions.stop, row_positions.step)
        if step == 1:
            return table.slice(start, len(row_positions))
        else:
            indices = np.arange(start, stop, step)
            return table.take(indices)

    def copy(self):
        if False:
            while True:
                i = 10
        '\n        Make a shallow copy of the node.\n\n        Returns\n        -------\n        MaskNode\n        '
        return MaskNode(self.input[0], self.row_labels, self.row_positions)

    def _prints(self, prefix):
        if False:
            print('Hello World!')
        '\n        Return a string representation of the tree.\n\n        Parameters\n        ----------\n        prefix : str\n            A prefix to add at each string of the dump.\n\n        Returns\n        -------\n        str\n        '
        return f'{prefix}MaskNode:\n' + f'{prefix}  row_labels: {self.row_labels}\n' + f'{prefix}  row_positions: {self.row_positions}\n' + self._prints_input(prefix + '  ')

class GroupbyAggNode(DFAlgNode):
    """
    A node to represent a groupby aggregation operation.

    Parameters
    ----------
    base : DFAlgNode
        An aggregated frame.
    by : list of str
        A list of columns used for grouping.
    agg_exprs : dict
        Aggregates to compute.
    groupby_opts : dict
        Additional groupby parameters.

    Attributes
    ----------
    input : list of DFAlgNode
        Holds a single aggregated frame.
    by : list of str
        A list of columns used for grouping.
    agg_exprs : dict
        Aggregates to compute.
    groupby_opts : dict
        Additional groupby parameters.
    """

    def __init__(self, base, by, agg_exprs, groupby_opts):
        if False:
            while True:
                i = 10
        self.by = by
        self.agg_exprs = agg_exprs
        self.groupby_opts = groupby_opts
        self.input = [base]

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make a shallow copy of the node.\n\n        Returns\n        -------\n        GroupbyAggNode\n        '
        return GroupbyAggNode(self.input[0], self.by, self.agg_exprs, self.groupby_opts)

    def _prints(self, prefix):
        if False:
            return 10
        '\n        Return a string representation of the tree.\n\n        Parameters\n        ----------\n        prefix : str\n            A prefix to add at each string of the dump.\n\n        Returns\n        -------\n        str\n        '
        return f'{prefix}AggNode:\n' + f'{prefix}  by: {self.by}\n' + f'{prefix}  aggs: {self.agg_exprs}\n' + f'{prefix}  groupby_opts: {self.groupby_opts}\n' + self._prints_input(prefix + '  ')

class TransformNode(DFAlgNode):
    """
    A node to represent a projection of a single frame.

    Provides expressions to compute each column of the projection.

    Parameters
    ----------
    base : HdkOnNativeDataframe
        A transformed frame.
    exprs : dict
        Expressions for frame's columns computation.
    fold : bool

    Attributes
    ----------
    input : list of HdkOnNativeDataframe
        Holds a single projected frame.
    exprs : dict
        Expressions used to compute frame's columns.
    """

    def __init__(self, base: 'HdkOnNativeDataframe', exprs: Dict[str, Union[InputRefExpr, LiteralExpr, OpExpr]], fold: bool=True):
        if False:
            return 10
        if fold and isinstance(base._op, TransformNode):
            self.input = [base._op.input[0]]
            self.exprs = exprs = translate_exprs_to_base(exprs, self.input[0])
            for (col, expr) in exprs.items():
                exprs[col] = expr.fold()
        else:
            self.input = [base]
            self.exprs = exprs

    @_inherit_docstrings(DFAlgNode.can_execute_hdk)
    def can_execute_hdk(self) -> bool:
        if False:
            while True:
                i = 10
        return self._check_exprs('can_execute_hdk')

    @_inherit_docstrings(DFAlgNode.can_execute_arrow)
    def can_execute_arrow(self) -> bool:
        if False:
            while True:
                i = 10
        return self._check_exprs('can_execute_arrow')

    def execute_arrow(self, table: pa.Table) -> pa.Table:
        if False:
            i = 10
            return i + 15
        '\n        Perform column selection on the frame using Arrow API.\n\n        Parameters\n        ----------\n        table : pa.Table\n\n        Returns\n        -------\n        pyarrow.Table\n            The resulting table.\n        '
        cols = [expr.execute_arrow(table) for expr in self.exprs.values()]
        names = [ColNameCodec.encode(c) for c in self.exprs]
        return pa.table(cols, names)

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make a shallow copy of the node.\n\n        Returns\n        -------\n        TransformNode\n        '
        return TransformNode(self.input[0], self.exprs)

    def is_simple_select(self):
        if False:
            i = 10
            return i + 15
        '\n        Check if transform node is a simple selection.\n\n        Simple selection can only use InputRefExpr expressions.\n\n        Returns\n        -------\n        bool\n            True for simple select and False otherwise.\n        '
        return all((isinstance(expr, InputRefExpr) for expr in self.exprs.values()))

    def _prints(self, prefix):
        if False:
            while True:
                i = 10
        '\n        Return a string representation of the tree.\n\n        Parameters\n        ----------\n        prefix : str\n            A prefix to add at each string of the dump.\n\n        Returns\n        -------\n        str\n        '
        res = f'{prefix}TransformNode:\n'
        for (k, v) in self.exprs.items():
            res += f'{prefix}  {k}: {v}\n'
        res += self._prints_input(prefix + '  ')
        return res

    def _check_exprs(self, attr) -> bool:
        if False:
            return 10
        '\n        Check if the specified attribute is True for all expressions.\n\n        Parameters\n        ----------\n        attr : str\n\n        Returns\n        -------\n        bool\n        '
        stack = list(self.exprs.values())
        while stack:
            expr = stack.pop()
            if not getattr(expr, attr)():
                return False
            if isinstance(expr, OpExpr):
                stack.extend(expr.operands)
        return True

class JoinNode(DFAlgNode):
    """
    A node to represent a join of two frames.

    Parameters
    ----------
    left : DFAlgNode
        A left frame to join.
    right : DFAlgNode
        A right frame to join.
    how : str, default: "inner"
        A type of join.
    exprs : dict, default: None
        Expressions for the resulting frame's columns.
    condition : BaseExpr, default: None
        Join condition.

    Attributes
    ----------
    input : list of DFAlgNode
        Holds joined frames. The first frame in the list is considered as
        the left join operand.
    how : str
        A type of join.
    exprs : dict
        Expressions for the resulting frame's columns.
    condition : BaseExpr
        Join condition.
    """

    def __init__(self, left, right, how='inner', exprs=None, condition=None):
        if False:
            i = 10
            return i + 15
        self.input = [left, right]
        self.how = how
        self.exprs = exprs
        self.condition = condition

    def copy(self):
        if False:
            i = 10
            return i + 15
        '\n        Make a shallow copy of the node.\n\n        Returns\n        -------\n        JoinNode\n        '
        return JoinNode(self.input[0], self.input[1], self.how, self.exprs, self.condition)

    def _prints(self, prefix):
        if False:
            while True:
                i = 10
        '\n        Return a string representation of the tree.\n\n        Parameters\n        ----------\n        prefix : str\n            A prefix to add at each string of the dump.\n\n        Returns\n        -------\n        str\n        '
        exprs_str = ''
        for (k, v) in self.exprs.items():
            exprs_str += f'{prefix}    {k}: {v}\n'
        return f'{prefix}JoinNode:\n' + f'{prefix}  Fields:\n' + exprs_str + f'{prefix}  How: {self.how}\n' + f'{prefix}  Condition: {self.condition}\n' + self._prints_input(prefix + '  ')

class UnionNode(DFAlgNode):
    """
    A node to represent rows union of input frames.

    Parameters
    ----------
    frames : list of HdkOnNativeDataframe
        Input frames.
    columns : dict
        Column names and dtypes.
    ignore_index : bool

    Attributes
    ----------
    input : list of HdkOnNativeDataframe
        Input frames.
    """

    def __init__(self, frames: List['HdkOnNativeDataframe'], columns: Dict[str, np.dtype], ignore_index: bool):
        if False:
            return 10
        self.input = frames
        self.columns = columns
        self.ignore_index = ignore_index

    @_inherit_docstrings(DFAlgNode.require_executed_base)
    def require_executed_base(self) -> bool:
        if False:
            print('Hello World!')
        return not self.can_execute_hdk()

    @_inherit_docstrings(DFAlgNode.can_execute_hdk)
    def can_execute_hdk(self) -> bool:
        if False:
            while True:
                i = 10
        if len(self.input) > 2:
            return False
        if len(self.input) == 0 or len(self.columns) == 0:
            return False
        dtypes = self.input[0]._dtypes.to_dict()
        if any((is_string_dtype(t) for t in dtypes.values())) or any((f._dtypes.to_dict() != dtypes for f in self.input[1:])):
            return False
        return True

    @_inherit_docstrings(DFAlgNode.can_execute_arrow)
    def can_execute_arrow(self) -> bool:
        if False:
            while True:
                i = 10
        return True

    def execute_arrow(self, tables: Union[pa.Table, List[pa.Table]]) -> pa.Table:
        if False:
            return 10
        "\n        Concat frames' rows using Arrow API.\n\n        Parameters\n        ----------\n        tables : pa.Table or list of pa.Table\n\n        Returns\n        -------\n        pyarrow.Table\n            The resulting table.\n        "
        if len(self.columns) == 0:
            frames = self.input
            if len(frames) == 0:
                return EMPTY_ARROW_TABLE
            elif self.ignore_index:
                idx = pandas.RangeIndex(0, sum((len(frame.index) for frame in frames)))
            else:
                idx = frames[0].index.append([f.index for f in frames[1:]])
            idx_cols = ColNameCodec.mangle_index_names(idx.names)
            idx_df = pandas.DataFrame(index=idx).reset_index()
            obj_cols = idx_df.select_dtypes(include=['object']).columns.tolist()
            if len(obj_cols) != 0:
                idx_df[obj_cols] = idx_df[obj_cols].astype(str)
            idx_table = pa.Table.from_pandas(idx_df, preserve_index=False)
            return idx_table.rename_columns(idx_cols)
        if isinstance(tables, pa.Table):
            assert len(self.input) == 1
            return tables
        try:
            return pa.concat_tables(tables)
        except pa.lib.ArrowInvalid:
            fields: typing.OrderedDict[str, pa.Field] = OrderedDict()
            for table in tables:
                for col_name in table.column_names:
                    field = table.field(col_name)
                    cur_field = fields.get(col_name, None)
                    if cur_field is None or cur_field.type != get_common_arrow_type(cur_field.type, field.type):
                        fields[col_name] = field
            schema = pa.schema(list(fields.values()))
            for (i, table) in enumerate(tables):
                tables[i] = pa.table(table.columns, schema=schema)
            return pa.concat_tables(tables)

    def copy(self):
        if False:
            print('Hello World!')
        '\n        Make a shallow copy of the node.\n\n        Returns\n        -------\n        UnionNode\n        '
        return UnionNode(self.input, self.columns, self.ignore_index)

    def _prints(self, prefix):
        if False:
            return 10
        '\n        Return a string representation of the tree.\n\n        Parameters\n        ----------\n        prefix : str\n            A prefix to add at each string of the dump.\n\n        Returns\n        -------\n        str\n        '
        return f'{prefix}UnionNode:\n' + self._prints_input(prefix + '  ')

class SortNode(DFAlgNode):
    """
    A sort node to order frame's rows in a specified order.

    Parameters
    ----------
    frame : DFAlgNode
        Sorted frame.
    columns : list of str
        A list of key columns for a sort.
    ascending : list of bool
        Ascending or descending sort.
    na_position : {"first", "last"}
        "first" to put NULLs at the start of the result,
        "last" to put NULLs at the end of the result.

    Attributes
    ----------
    input : list of DFAlgNode
        Holds a single sorted frame.
    columns : list of str
        A list of key columns for a sort.
    ascending : list of bool
        Ascending or descending sort.
    na_position : {"first", "last"}
        "first" to put NULLs at the start of the result,
        "last" to put NULLs at the end of the result.
    """

    def __init__(self, frame, columns, ascending, na_position):
        if False:
            for i in range(10):
                print('nop')
        self.input = [frame]
        self.columns = columns
        self.ascending = ascending
        self.na_position = na_position

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make a shallow copy of the node.\n\n        Returns\n        -------\n        SortNode\n        '
        return SortNode(self.input[0], self.columns, self.ascending, self.na_position)

    def _prints(self, prefix):
        if False:
            while True:
                i = 10
        '\n        Return a string representation of the tree.\n\n        Parameters\n        ----------\n        prefix : str\n            A prefix to add at each string of the dump.\n\n        Returns\n        -------\n        str\n        '
        return f'{prefix}SortNode:\n' + f'{prefix}  Columns: {self.columns}\n' + f'{prefix}  Ascending: {self.ascending}\n' + f'{prefix}  NULLs position: {self.na_position}\n' + self._prints_input(prefix + '  ')

class FilterNode(DFAlgNode):
    """
    A node for generic rows filtering.

    For rows filter by row id a ``MaskNode`` should be preferred.

    Parameters
    ----------
    frame : DFAlgNode
        A filtered frame.
    condition : BaseExpr
        Filter condition.

    Attributes
    ----------
    input : list of DFAlgNode
        Holds a single filtered frame.
    condition : BaseExpr
        Filter condition.
    """

    def __init__(self, frame, condition):
        if False:
            i = 10
            return i + 15
        self.input = [frame]
        self.condition = condition

    def copy(self):
        if False:
            i = 10
            return i + 15
        '\n        Make a shallow copy of the node.\n\n        Returns\n        -------\n        FilterNode\n        '
        return FilterNode(self.input[0], self.condition)

    def _prints(self, prefix):
        if False:
            return 10
        '\n        Return a string representation of the tree.\n\n        Parameters\n        ----------\n        prefix : str\n            A prefix to add at each string of the dump.\n\n        Returns\n        -------\n        str\n        '
        return f'{prefix}FilterNode:\n' + f'{prefix}  Condition: {self.condition}\n' + self._prints_input(prefix + '  ')

def translate_exprs_to_base(exprs, base):
    if False:
        print('Hello World!')
    '\n    Fold expressions.\n\n    Fold expressions with their input nodes until `base`\n    frame is the only input frame.\n\n    Parameters\n    ----------\n    exprs : dict\n        Expressions to translate.\n    base : HdkOnNativeDataframe\n        Required input frame for translated expressions.\n\n    Returns\n    -------\n    dict\n        Translated expressions.\n    '
    new_exprs = dict(exprs)
    frames = set()
    for expr in new_exprs.values():
        expr.collect_frames(frames)
    frames.discard(base)
    while len(frames) > 0:
        mapper = InputMapper()
        new_frames = set()
        for frame in frames:
            frame_base = frame._op.input[0]
            if frame_base != base:
                new_frames.add(frame_base)
            assert isinstance(frame._op, TransformNode)
            mapper.add_mapper(frame, TransformMapper(frame._op))
        for (k, v) in new_exprs.items():
            new_expr = v.translate_input(mapper)
            new_expr.collect_frames(new_frames)
            new_exprs[k] = new_expr
        new_frames.discard(base)
        frames = new_frames
    res = OrderedDict()
    for col in exprs.keys():
        res[col] = new_exprs[col]
    return res

def replace_frame_in_exprs(exprs, old_frame, new_frame):
    if False:
        while True:
            i = 10
    '\n    Translate input expression replacing an input frame in them.\n\n    Parameters\n    ----------\n    exprs : dict\n        Expressions to translate.\n    old_frame : HdkOnNativeDataframe\n        An input frame to replace.\n    new_frame : HdkOnNativeDataframe\n        A new input frame to use.\n\n    Returns\n    -------\n    dict\n        Translated expressions.\n    '
    mapper = InputMapper()
    mapper.add_mapper(old_frame, FrameMapper(new_frame))
    res = OrderedDict()
    for col in exprs.keys():
        res[col] = exprs[col].translate_input(mapper)
    return res