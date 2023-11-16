"""Module provides a partition manager class for ``HdkOnNativeDataframe`` frame."""
import re
import numpy as np
import pandas
import pyarrow
from modin.config import DoUseCalcite
from modin.core.dataframe.pandas.partitioning.partition_manager import PandasDataframePartitionManager
from modin.error_message import ErrorMessage
from modin.pandas.utils import is_scalar
from ..calcite_builder import CalciteBuilder
from ..calcite_serializer import CalciteSerializer
from ..dataframe.utils import ColNameCodec
from ..db_worker import DbTable, DbWorker
from ..partitioning.partition import HdkOnNativeDataframePartition

class HdkOnNativeDataframePartitionManager(PandasDataframePartitionManager):
    """
    Frame manager for ``HdkOnNativeDataframe``.

    This class handles several features of ``HdkOnNativeDataframe``:
      - frame always has a single partition
      - frame cannot process some data types
      - frame has to use mangling for index labels
      - frame uses HDK storage format for execution
    """
    _partition_class = HdkOnNativeDataframePartition

    @classmethod
    def from_pandas(cls, df, return_dims=False, encode_col_names=True):
        if False:
            for i in range(10):
                print('nop')
        "\n        Build partitions from a ``pandas.DataFrame``.\n\n        Parameters\n        ----------\n        df : pandas.DataFrame\n            Source frame.\n        return_dims : bool, default: False\n            Include resulting dimensions into the returned value.\n        encode_col_names : bool, default: True\n            Encode column names.\n\n        Returns\n        -------\n        tuple\n            Tuple holding array of partitions, list of columns with unsupported\n            data and optionally partitions' dimensions.\n        "
        (at, unsupported_cols) = cls._get_unsupported_cols(df)
        if len(unsupported_cols) > 0:
            parts = [[cls._partition_class(df)]]
            if not return_dims:
                return (np.array(parts), unsupported_cols)
            else:
                row_lengths = [len(df)]
                col_widths = [len(df.columns)]
                return (np.array(parts), row_lengths, col_widths, unsupported_cols)
        else:
            return cls.from_arrow(at, return_dims, unsupported_cols, encode_col_names)

    @classmethod
    def from_arrow(cls, at, return_dims=False, unsupported_cols=None, encode_col_names=True):
        if False:
            print('Hello World!')
        "\n        Build partitions from a ``pyarrow.Table``.\n\n        Parameters\n        ----------\n        at : pyarrow.Table\n            Input table.\n        return_dims : bool, default: False\n            True to include dimensions into returned tuple.\n        unsupported_cols : list of str, optional\n            List of columns holding unsupported data. If None then\n            check all columns to compute the list.\n        encode_col_names : bool, default: True\n            Encode column names.\n\n        Returns\n        -------\n        tuple\n            Tuple holding array of partitions, list of columns with unsupported\n            data and optionally partitions' dimensions.\n        "
        if encode_col_names:
            encoded_names = [ColNameCodec.encode(n) for n in at.column_names]
            encoded_at = at
            if encoded_names != at.column_names:
                encoded_at = at.rename_columns(encoded_names)
        else:
            encoded_at = at
        parts = [[cls._partition_class(encoded_at)]]
        if unsupported_cols is None:
            (_, unsupported_cols) = cls._get_unsupported_cols(at)
        if not return_dims:
            return (np.array(parts), unsupported_cols)
        else:
            row_lengths = [at.num_rows]
            col_widths = [at.num_columns]
            return (np.array(parts), row_lengths, col_widths, unsupported_cols)

    @classmethod
    def _get_unsupported_cols(cls, obj):
        if False:
            i = 10
            return i + 15
        '\n        Return a list of columns with unsupported by HDK data types.\n\n        Parameters\n        ----------\n        obj : pandas.DataFrame or pyarrow.Table\n            Object to inspect on unsupported column types.\n\n        Returns\n        -------\n        tuple\n            Arrow representation of `obj` (for future using) and a list of\n            unsupported columns.\n        '
        if isinstance(obj, (pandas.Series, pandas.DataFrame)):
            if obj.empty:
                unsupported_cols = []
            elif isinstance(obj.columns, pandas.MultiIndex):
                unsupported_cols = [str(c) for c in obj.columns]
            else:
                cols = [name for (name, col) in obj.dtypes.items() if col == 'object']
                type_samples = obj.iloc[0][cols]
                unsupported_cols = [name for (name, col) in type_samples.items() if not isinstance(col, str) and (not (is_scalar(col) and pandas.isna(col)))]
            if len(unsupported_cols) > 0:
                return (None, unsupported_cols)
            try:
                at = pyarrow.Table.from_pandas(obj, preserve_index=False)
            except (pyarrow.lib.ArrowTypeError, pyarrow.lib.ArrowInvalid, ValueError, TypeError) as err:
                if type(err) is TypeError:
                    if any([isinstance(t, pandas.SparseDtype) for t in obj.dtypes]):
                        ErrorMessage.single_warning('Sparse data is not currently supported!')
                    else:
                        raise err
                if type(err) is ValueError and obj.columns.is_unique:
                    raise err
                regex = 'Conversion failed for column ([^\\W]*)'
                unsupported_cols = []
                for msg in err.args:
                    match = re.findall(regex, msg)
                    unsupported_cols.extend(match)
                if len(unsupported_cols) == 0:
                    unsupported_cols = obj.columns
                return (None, unsupported_cols)
            else:
                obj = at

        def is_supported_dtype(dtype):
            if False:
                while True:
                    i = 10
            'Check whether the passed pyarrow `dtype` is supported by HDK.'
            if pyarrow.types.is_string(dtype) or pyarrow.types.is_time(dtype) or pyarrow.types.is_dictionary(dtype) or pyarrow.types.is_null(dtype):
                return True
            if isinstance(dtype, pyarrow.ExtensionType) or pyarrow.types.is_duration(dtype):
                return False
            try:
                pandas_dtype = dtype.to_pandas_dtype()
                return pandas_dtype != np.dtype('O')
            except NotImplementedError:
                return False
        return (obj, [field.name for field in obj.schema if not is_supported_dtype(field.type)])

    @classmethod
    def run_exec_plan(cls, plan):
        if False:
            for i in range(10):
                print('nop')
        "\n        Run execution plan in HDK storage format to materialize frame.\n\n        Parameters\n        ----------\n        plan : DFAlgNode\n            A root of an execution plan tree.\n\n        Returns\n        -------\n        np.array\n            Created frame's partitions.\n        "
        worker = DbWorker()
        frames = plan.collect_frames()
        for frame in frames:
            cls.import_table(frame, worker)
        builder = CalciteBuilder()
        calcite_plan = builder.build(plan)
        calcite_json = CalciteSerializer().serialize(calcite_plan)
        if DoUseCalcite.get():
            exec_calcite = True
            calcite_json = 'execute calcite ' + calcite_json
        else:
            exec_calcite = False
        exec_args = {}
        if builder.has_groupby and (not builder.has_join):
            exec_args = {'enable_lazy_fetch': 0, 'enable_columnar_output': 0}
        elif not builder.has_groupby and builder.has_join:
            exec_args = {'enable_lazy_fetch': 1, 'enable_columnar_output': 1}
        table = worker.executeRA(calcite_json, exec_calcite, **exec_args)
        res = np.empty((1, 1), dtype=np.dtype(object))
        res[0][0] = cls._partition_class(table)
        return res

    @classmethod
    def import_table(cls, frame, worker=DbWorker()) -> DbTable:
        if False:
            while True:
                i = 10
        "\n        Import the frame's partition data, if required.\n\n        Parameters\n        ----------\n        frame : HdkOnNativeDataframe\n        worker : DbWorker, optional\n\n        Returns\n        -------\n        DbTable\n        "
        table = frame._partitions[0][0].get()
        if isinstance(table, pandas.DataFrame):
            table = worker.import_pandas_dataframe(table)
            frame._partitions[0][0] = cls._partition_class(table)
        elif isinstance(table, pyarrow.Table):
            if table.num_columns == 0:
                idx_names = frame.index.names if frame.has_materialized_index else [None]
                idx_names = ColNameCodec.mangle_index_names(idx_names)
                table = pyarrow.table({n: [] for n in idx_names}, schema=pyarrow.schema({n: pyarrow.int64() for n in idx_names}))
            table = worker.import_arrow_table(table)
            frame._partitions[0][0] = cls._partition_class(table)
        return table

    @classmethod
    def _names_from_index_cols(cls, cols):
        if False:
            i = 10
            return i + 15
        '\n        Get index labels.\n\n        Deprecated.\n\n        Parameters\n        ----------\n        cols : list of str\n            Index columns.\n\n        Returns\n        -------\n        list of str\n        '
        if len(cols) == 1:
            return cls._name_from_index_col(cols[0])
        return [cls._name_from_index_col(n) for n in cols]

    @classmethod
    def _name_from_index_col(cls, col):
        if False:
            print('Hello World!')
        '\n        Get index label.\n\n        Deprecated.\n\n        Parameters\n        ----------\n        col : str\n            Index column.\n\n        Returns\n        -------\n        str\n        '
        if col.startswith(ColNameCodec.IDX_COL_NAME):
            return None
        return col

    @classmethod
    def _maybe_scalar(cls, lst):
        if False:
            while True:
                i = 10
        '\n        Transform list with a single element to scalar.\n\n        Deprecated.\n\n        Parameters\n        ----------\n        lst : list\n            Input list.\n\n        Returns\n        -------\n        Any\n        '
        if len(lst) == 1:
            return lst[0]
        return lst