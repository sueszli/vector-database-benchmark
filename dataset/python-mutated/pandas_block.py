import collections
import heapq
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.data._internal.table_block import TableBlockAccessor, TableBlockBuilder
from ray.data._internal.util import find_partitions
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata, KeyType, U
from ray.data.context import DataContext
from ray.data.row import TableRow
if TYPE_CHECKING:
    import pandas
    import pyarrow
    from ray.data._internal.sort import SortKey
    from ray.data.aggregate import AggregateFn
T = TypeVar('T')
_pandas = None

def lazy_import_pandas():
    if False:
        i = 10
        return i + 15
    global _pandas
    if _pandas is None:
        import pandas
        _pandas = pandas
    return _pandas

class PandasRow(TableRow):
    """
    Row of a tabular Dataset backed by a Pandas DataFrame block.
    """

    def __getitem__(self, key: Union[str, List[str]]) -> Any:
        if False:
            print('Hello World!')
        from ray.data.extensions import TensorArrayElement

        def get_item(keys: List[str]) -> Any:
            if False:
                print('Hello World!')
            col = self._row[keys]
            if len(col) == 0:
                return None
            items = col.iloc[0]
            if isinstance(items[0], TensorArrayElement):
                return tuple([item.to_numpy() for item in items])
            try:
                return tuple([item.as_py() for item in items])
            except (AttributeError, ValueError):
                return items
        is_single_item = isinstance(key, str)
        keys = [key] if is_single_item else key
        items = get_item(keys)
        if items is None:
            return None
        elif is_single_item:
            return items[0]
        else:
            return items

    def __iter__(self) -> Iterator:
        if False:
            while True:
                i = 10
        for k in self._row.columns:
            yield k

    def __len__(self):
        if False:
            while True:
                i = 10
        return self._row.shape[1]

class PandasBlockBuilder(TableBlockBuilder):

    def __init__(self):
        if False:
            print('Hello World!')
        pandas = lazy_import_pandas()
        super().__init__(pandas.DataFrame)

    @staticmethod
    def _table_from_pydict(columns: Dict[str, List[Any]]) -> 'pandas.DataFrame':
        if False:
            return 10
        pandas = lazy_import_pandas()
        for (key, value) in columns.items():
            if key == TENSOR_COLUMN_NAME or isinstance(next(iter(value), None), np.ndarray):
                from ray.data.extensions.tensor_extension import TensorArray
                columns[key] = TensorArray(value)
        return pandas.DataFrame(columns)

    @staticmethod
    def _concat_tables(tables: List['pandas.DataFrame']) -> 'pandas.DataFrame':
        if False:
            i = 10
            return i + 15
        pandas = lazy_import_pandas()
        from ray.air.util.data_batch_conversion import _cast_ndarray_columns_to_tensor_extension
        if len(tables) > 1:
            df = pandas.concat(tables, ignore_index=True)
            df.reset_index(drop=True, inplace=True)
        else:
            df = tables[0]
        ctx = DataContext.get_current()
        if ctx.enable_tensor_extension_casting:
            df = _cast_ndarray_columns_to_tensor_extension(df)
        return df

    @staticmethod
    def _concat_would_copy() -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def _empty_table() -> 'pandas.DataFrame':
        if False:
            print('Hello World!')
        pandas = lazy_import_pandas()
        return pandas.DataFrame()
PandasBlockSchema = collections.namedtuple('PandasBlockSchema', ['names', 'types'])

class PandasBlockAccessor(TableBlockAccessor):
    ROW_TYPE = PandasRow

    def __init__(self, table: 'pandas.DataFrame'):
        if False:
            i = 10
            return i + 15
        super().__init__(table)

    def column_names(self) -> List[str]:
        if False:
            print('Hello World!')
        return self._table.columns.tolist()

    @staticmethod
    def _build_tensor_row(row: PandasRow) -> np.ndarray:
        if False:
            while True:
                i = 10
        from ray.data.extensions import TensorArrayElement
        tensor = row[TENSOR_COLUMN_NAME].iloc[0]
        if isinstance(tensor, TensorArrayElement):
            tensor = tensor.to_numpy()
        return tensor

    def slice(self, start: int, end: int, copy: bool=False) -> 'pandas.DataFrame':
        if False:
            i = 10
            return i + 15
        view = self._table[start:end]
        view.reset_index(drop=True, inplace=True)
        if copy:
            view = view.copy(deep=True)
        return view

    def take(self, indices: List[int]) -> 'pandas.DataFrame':
        if False:
            print('Hello World!')
        table = self._table.take(indices)
        table.reset_index(drop=True, inplace=True)
        return table

    def select(self, columns: List[str]) -> 'pandas.DataFrame':
        if False:
            for i in range(10):
                print('nop')
        if not all((isinstance(col, str) for col in columns)):
            raise ValueError(f'Columns must be a list of column name strings when aggregating on Pandas blocks, but got: {columns}.')
        return self._table[columns]

    def random_shuffle(self, random_seed: Optional[int]) -> 'pandas.DataFrame':
        if False:
            while True:
                i = 10
        table = self._table.sample(frac=1, random_state=random_seed)
        table.reset_index(drop=True, inplace=True)
        return table

    def schema(self) -> PandasBlockSchema:
        if False:
            while True:
                i = 10
        dtypes = self._table.dtypes
        schema = PandasBlockSchema(names=dtypes.index.tolist(), types=dtypes.values.tolist())
        if any((not isinstance(name, str) for name in schema.names)):
            raise ValueError(f'A Pandas DataFrame with column names of non-str types is not supported by Ray Dataset. Column names of this DataFrame: {schema.names!r}.')
        return schema

    def to_pandas(self) -> 'pandas.DataFrame':
        if False:
            i = 10
            return i + 15
        from ray.air.util.data_batch_conversion import _cast_tensor_columns_to_ndarrays
        ctx = DataContext.get_current()
        table = self._table
        if ctx.enable_tensor_extension_casting:
            table = _cast_tensor_columns_to_ndarrays(table)
        return table

    def to_numpy(self, columns: Optional[Union[str, List[str]]]=None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if False:
            for i in range(10):
                print('nop')
        if columns is None:
            columns = self._table.columns.tolist()
            should_be_single_ndarray = False
        elif isinstance(columns, list):
            should_be_single_ndarray = False
        else:
            columns = [columns]
            should_be_single_ndarray = True
        for column in columns:
            if column not in self._table.columns:
                raise ValueError(f'Cannot find column {column}, available columns: {self._table.columns.tolist()}')
        arrays = []
        for column in columns:
            arrays.append(self._table[column].to_numpy())
        if should_be_single_ndarray:
            arrays = arrays[0]
        else:
            arrays = dict(zip(columns, arrays))
        return arrays

    def to_arrow(self) -> 'pyarrow.Table':
        if False:
            print('Hello World!')
        import pyarrow
        return pyarrow.table(self._table)

    def num_rows(self) -> int:
        if False:
            return 10
        return self._table.shape[0]

    def size_bytes(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return int(self._table.memory_usage(index=True, deep=True).sum())

    def _zip(self, acc: BlockAccessor) -> 'pandas.DataFrame':
        if False:
            return 10
        r = self.to_pandas().copy(deep=False)
        s = acc.to_pandas()
        for col_name in s.columns:
            col = s[col_name]
            column_names = list(r.columns)
            if col_name in column_names:
                i = 1
                new_name = col_name
                while new_name in column_names:
                    new_name = '{}_{}'.format(col_name, i)
                    i += 1
                col_name = new_name
            r[col_name] = col
        return r

    @staticmethod
    def builder() -> PandasBlockBuilder:
        if False:
            return 10
        return PandasBlockBuilder()

    @staticmethod
    def _empty_table() -> 'pandas.DataFrame':
        if False:
            while True:
                i = 10
        return PandasBlockBuilder._empty_table()

    def _sample(self, n_samples: int, sort_key: 'SortKey') -> 'pandas.DataFrame':
        if False:
            for i in range(10):
                print('nop')
        return self._table[sort_key.get_columns()].sample(n_samples, ignore_index=True)

    def _apply_agg(self, agg_fn: Callable[['pandas.Series', bool], U], on: str) -> Optional[U]:
        if False:
            return 10
        'Helper providing null handling around applying an aggregation to a column.'
        pd = lazy_import_pandas()
        if on is not None and (not isinstance(on, str)):
            raise ValueError(f'on must be a string or None when aggregating on Pandas blocks, but got: {type(on)}.')
        if self.num_rows() == 0:
            return None
        col = self._table[on]
        try:
            val = agg_fn(col)
        except TypeError as e:
            if np.issubdtype(col.dtype, np.object_) and col.isnull().all():
                return None
            raise e from None
        if pd.isnull(val):
            return None
        return val

    def count(self, on: str) -> Optional[U]:
        if False:
            for i in range(10):
                print('nop')
        return self._apply_agg(lambda col: col.count(), on)

    def sum(self, on: str, ignore_nulls: bool) -> Optional[U]:
        if False:
            print('Hello World!')
        pd = lazy_import_pandas()
        if on is not None and (not isinstance(on, str)):
            raise ValueError(f'on must be a string or None when aggregating on Pandas blocks, but got: {type(on)}.')
        if self.num_rows() == 0:
            return None
        col = self._table[on]
        if col.isnull().all():
            return None
        val = col.sum(skipna=ignore_nulls)
        if pd.isnull(val):
            return None
        return val

    def min(self, on: str, ignore_nulls: bool) -> Optional[U]:
        if False:
            for i in range(10):
                print('nop')
        return self._apply_agg(lambda col: col.min(skipna=ignore_nulls), on)

    def max(self, on: str, ignore_nulls: bool) -> Optional[U]:
        if False:
            return 10
        return self._apply_agg(lambda col: col.max(skipna=ignore_nulls), on)

    def mean(self, on: str, ignore_nulls: bool) -> Optional[U]:
        if False:
            i = 10
            return i + 15
        return self._apply_agg(lambda col: col.mean(skipna=ignore_nulls), on)

    def sum_of_squared_diffs_from_mean(self, on: str, ignore_nulls: bool, mean: Optional[U]=None) -> Optional[U]:
        if False:
            return 10
        if mean is None:
            mean = self.mean(on, ignore_nulls)
        return self._apply_agg(lambda col: ((col - mean) ** 2).sum(skipna=ignore_nulls), on)

    def sort_and_partition(self, boundaries: List[T], sort_key: 'SortKey') -> List[Block]:
        if False:
            for i in range(10):
                print('nop')
        if self._table.shape[0] == 0:
            return [self._empty_table() for _ in range(len(boundaries) + 1)]
        (columns, ascending) = sort_key.to_pandas_sort_args()
        table = self._table.sort_values(by=columns, ascending=ascending)
        if len(boundaries) == 0:
            return [table]
        return find_partitions(table, boundaries, sort_key)

    def combine(self, key: Union[str, List[str]], aggs: Tuple['AggregateFn']) -> 'pandas.DataFrame':
        if False:
            for i in range(10):
                print('nop')
        'Combine rows with the same key into an accumulator.\n\n        This assumes the block is already sorted by key in ascending order.\n\n        Args:\n            key: A column name or list of column names.\n            If this is ``None``, place all rows in a single group.\n\n            aggs: The aggregations to do.\n\n        Returns:\n            A sorted block of [k, v_1, ..., v_n] columns where k is the groupby\n            key and v_i is the partially combined accumulator for the ith given\n            aggregation.\n            If key is None then the k column is omitted.\n        '
        if key is not None and (not isinstance(key, (str, list))):
            raise ValueError(f'key must be a string, list of strings or None when aggregating on Pandas blocks, but got: {type(key)}.')

        def iter_groups() -> Iterator[Tuple[KeyType, Block]]:
            if False:
                while True:
                    i = 10
            'Creates an iterator over zero-copy group views.'
            if key is None:
                yield (None, self.to_block())
                return
            start = end = 0
            iter = self.iter_rows(public_row_format=False)
            next_row = None
            while True:
                try:
                    if next_row is None:
                        next_row = next(iter)
                    next_key = next_row[key]
                    while np.all(next_row[key] == next_key):
                        end += 1
                        try:
                            next_row = next(iter)
                        except StopIteration:
                            next_row = None
                            break
                    yield (next_key, self.slice(start, end, copy=False))
                    start = end
                except StopIteration:
                    break
        builder = PandasBlockBuilder()
        for (group_key, group_view) in iter_groups():
            accumulators = [agg.init(group_key) for agg in aggs]
            for i in range(len(aggs)):
                accumulators[i] = aggs[i].accumulate_block(accumulators[i], group_view)
            row = {}
            if key is not None:
                if isinstance(key, list):
                    keys = key
                    group_keys = group_key
                else:
                    keys = [key]
                    group_keys = [group_key]
                for (k, gk) in zip(keys, group_keys):
                    row[k] = gk
            count = collections.defaultdict(int)
            for (agg, accumulator) in zip(aggs, accumulators):
                name = agg.name
                if count[name] > 0:
                    name = self._munge_conflict(name, count[name])
                count[name] += 1
                row[name] = accumulator
            builder.add(row)
        return builder.build()

    @staticmethod
    def merge_sorted_blocks(blocks: List[Block], sort_key: 'SortKey') -> Tuple['pandas.DataFrame', BlockMetadata]:
        if False:
            return 10
        pd = lazy_import_pandas()
        stats = BlockExecStats.builder()
        blocks = [b for b in blocks if b.shape[0] > 0]
        if len(blocks) == 0:
            ret = PandasBlockAccessor._empty_table()
        else:
            ret = pd.concat(blocks, ignore_index=True)
            (columns, ascending) = sort_key.to_pandas_sort_args()
            ret = ret.sort_values(by=columns, ascending=ascending)
        return (ret, PandasBlockAccessor(ret).get_metadata(None, exec_stats=stats.build()))

    @staticmethod
    def aggregate_combined_blocks(blocks: List['pandas.DataFrame'], key: Union[str, List[str]], aggs: Tuple['AggregateFn'], finalize: bool) -> Tuple['pandas.DataFrame', BlockMetadata]:
        if False:
            return 10
        'Aggregate sorted, partially combined blocks with the same key range.\n\n        This assumes blocks are already sorted by key in ascending order,\n        so we can do merge sort to get all the rows with the same key.\n\n        Args:\n            blocks: A list of partially combined and sorted blocks.\n            key: The column name of key or None for global aggregation.\n            aggs: The aggregations to do.\n            finalize: Whether to finalize the aggregation. This is used as an\n                optimization for cases where we repeatedly combine partially\n                aggregated groups.\n\n        Returns:\n            A block of [k, v_1, ..., v_n] columns and its metadata where k is\n            the groupby key and v_i is the corresponding aggregation result for\n            the ith given aggregation.\n            If key is None then the k column is omitted.\n        '
        stats = BlockExecStats.builder()
        keys = key if isinstance(key, list) else [key]
        key_fn = (lambda r: tuple(r[r._row.columns[:len(keys)]])) if key is not None else lambda r: (0,)
        iter = heapq.merge(*[PandasBlockAccessor(block).iter_rows(public_row_format=False) for block in blocks], key=key_fn)
        next_row = None
        builder = PandasBlockBuilder()
        while True:
            try:
                if next_row is None:
                    next_row = next(iter)
                next_keys = key_fn(next_row)
                next_key_names = next_row._row.columns[:len(keys)] if key is not None else None

                def gen():
                    if False:
                        while True:
                            i = 10
                    nonlocal iter
                    nonlocal next_row
                    while key_fn(next_row) == next_keys:
                        yield next_row
                        try:
                            next_row = next(iter)
                        except StopIteration:
                            next_row = None
                            break
                first = True
                accumulators = [None] * len(aggs)
                resolved_agg_names = [None] * len(aggs)
                for r in gen():
                    if first:
                        count = collections.defaultdict(int)
                        for i in range(len(aggs)):
                            name = aggs[i].name
                            if count[name] > 0:
                                name = PandasBlockAccessor._munge_conflict(name, count[name])
                            count[name] += 1
                            resolved_agg_names[i] = name
                            accumulators[i] = r[name]
                        first = False
                    else:
                        for i in range(len(aggs)):
                            accumulators[i] = aggs[i].merge(accumulators[i], r[resolved_agg_names[i]])
                row = {}
                if key is not None:
                    for (next_key, next_key_name) in zip(next_keys, next_key_names):
                        row[next_key_name] = next_key
                for (agg, agg_name, accumulator) in zip(aggs, resolved_agg_names, accumulators):
                    if finalize:
                        row[agg_name] = agg.finalize(accumulator)
                    else:
                        row[agg_name] = accumulator
                builder.add(row)
            except StopIteration:
                break
        ret = builder.build()
        return (ret, PandasBlockAccessor(ret).get_metadata(None, exec_stats=stats.build()))