from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from ._internal.table_block import TableBlockAccessor
from ray.data._internal import sort
from ray.data._internal.compute import ComputeStrategy
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.logical.interfaces import LogicalPlan
from ray.data._internal.logical.operators.all_to_all_operator import Aggregate
from ray.data._internal.plan import AllToAllStage
from ray.data._internal.push_based_shuffle import PushBasedShufflePlan
from ray.data._internal.shuffle import ShuffleOp, SimpleShufflePlan
from ray.data._internal.sort import SortKey
from ray.data.aggregate import AggregateFn, Count, Max, Mean, Min, Std, Sum
from ray.data.aggregate._aggregate import _AggregateOnKeyBase
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata, KeyType, UserDefinedFunction
from ray.data.context import DataContext
from ray.data.dataset import DataBatch, Dataset
from ray.util.annotations import PublicAPI

class _GroupbyOp(ShuffleOp):

    @staticmethod
    def map(idx: int, block: Block, output_num_blocks: int, boundaries: List[KeyType], key: str, aggs: Tuple[AggregateFn]) -> List[Union[BlockMetadata, Block]]:
        if False:
            while True:
                i = 10
        'Partition the block and combine rows with the same key.'
        stats = BlockExecStats.builder()
        block = _GroupbyOp._prune_unused_columns(block, key, aggs)
        if key is None:
            partitions = [block]
        else:
            partitions = BlockAccessor.for_block(block).sort_and_partition(boundaries, SortKey(key))
        parts = [BlockAccessor.for_block(p).combine(key, aggs) for p in partitions]
        meta = BlockAccessor.for_block(block).get_metadata(input_files=None, exec_stats=stats.build())
        return parts + [meta]

    @staticmethod
    def reduce(key: str, aggs: Tuple[AggregateFn], *mapper_outputs: List[Block], partial_reduce: bool=False) -> (Block, BlockMetadata):
        if False:
            i = 10
            return i + 15
        'Aggregate sorted and partially combined blocks.'
        return BlockAccessor.for_block(mapper_outputs[0]).aggregate_combined_blocks(list(mapper_outputs), key, aggs, finalize=not partial_reduce)

    @staticmethod
    def _prune_unused_columns(block: Block, key: str, aggs: Tuple[AggregateFn]) -> Block:
        if False:
            for i in range(10):
                print('nop')
        'Prune unused columns from block before aggregate.'
        prune_columns = True
        columns = set()
        if isinstance(key, str):
            columns.add(key)
        elif callable(key):
            prune_columns = False
        for agg in aggs:
            if isinstance(agg, _AggregateOnKeyBase) and isinstance(agg._key_fn, str):
                columns.add(agg._key_fn)
            elif not isinstance(agg, Count):
                prune_columns = False
        block_accessor = BlockAccessor.for_block(block)
        if prune_columns and isinstance(block_accessor, TableBlockAccessor) and (block_accessor.num_rows() > 0):
            return block_accessor.select(list(columns))
        else:
            return block

class SimpleShuffleGroupbyOp(_GroupbyOp, SimpleShufflePlan):
    pass

class PushBasedGroupbyOp(_GroupbyOp, PushBasedShufflePlan):
    pass

@PublicAPI
class GroupedData:
    """Represents a grouped dataset created by calling ``Dataset.groupby()``.

    The actual groupby is deferred until an aggregation is applied.
    """

    def __init__(self, dataset: Dataset, key: Union[str, List[str]]):
        if False:
            i = 10
            return i + 15
        'Construct a dataset grouped by key (internal API).\n\n        The constructor is not part of the GroupedData API.\n        Use the ``Dataset.groupby()`` method to construct one.\n        '
        self._dataset = dataset
        self._key = key

    def __repr__(self) -> str:
        if False:
            return 10
        return f'{self.__class__.__name__}(dataset={self._dataset}, key={self._key!r})'

    def aggregate(self, *aggs: AggregateFn) -> Dataset:
        if False:
            for i in range(10):
                print('nop')
        'Implements an accumulator-based aggregation.\n\n        Args:\n            aggs: Aggregations to do.\n\n        Returns:\n            The output is an dataset of ``n + 1`` columns where the first column\n            is the groupby key and the second through ``n + 1`` columns are the\n            results of the aggregations.\n            If groupby key is ``None`` then the key part of return is omitted.\n        '

        def do_agg(blocks, task_ctx: TaskContext, clear_input_blocks: bool, *_):
            if False:
                print('Hello World!')
            stage_info = {}
            if len(aggs) == 0:
                raise ValueError('Aggregate requires at least one aggregation')
            for agg in aggs:
                agg._validate(self._dataset.schema(fetch_if_missing=True))
            if blocks.initial_num_blocks() == 0:
                return (blocks, stage_info)
            num_mappers = blocks.initial_num_blocks()
            num_reducers = num_mappers
            if self._key is None:
                num_reducers = 1
                boundaries = []
            else:
                boundaries = sort.sample_boundaries(blocks.get_blocks(), SortKey(self._key), num_reducers, task_ctx)
            ctx = DataContext.get_current()
            if ctx.use_push_based_shuffle:
                shuffle_op_cls = PushBasedGroupbyOp
            else:
                shuffle_op_cls = SimpleShuffleGroupbyOp
            shuffle_op = shuffle_op_cls(map_args=[boundaries, self._key, aggs], reduce_args=[self._key, aggs])
            return shuffle_op.execute(blocks, num_reducers, clear_input_blocks, ctx=task_ctx)
        plan = self._dataset._plan.with_stage(AllToAllStage('Aggregate', None, do_agg, sub_stage_names=['SortSample', 'ShuffleMap', 'ShuffleReduce']))
        logical_plan = self._dataset._logical_plan
        if logical_plan is not None:
            op = Aggregate(logical_plan.dag, key=self._key, aggs=aggs)
            logical_plan = LogicalPlan(op)
        return Dataset(plan, logical_plan)

    def _aggregate_on(self, agg_cls: type, on: Union[str, List[str]], ignore_nulls: bool, *args, **kwargs):
        if False:
            print('Hello World!')
        'Helper for aggregating on a particular subset of the dataset.\n\n        This validates the `on` argument, and converts a list of column names\n        to a multi-aggregation. A null `on` results in a\n        multi-aggregation on all columns for an Arrow Dataset, and a single\n        aggregation on the entire row for a simple Dataset.\n        '
        aggs = self._dataset._build_multicolumn_aggs(agg_cls, on, ignore_nulls, *args, skip_cols=self._key, **kwargs)
        return self.aggregate(*aggs)

    def map_groups(self, fn: UserDefinedFunction[DataBatch, DataBatch], *, compute: Union[str, ComputeStrategy]=None, batch_format: Optional[str]='default', fn_args: Optional[Iterable[Any]]=None, fn_kwargs: Optional[Dict[str, Any]]=None, **ray_remote_args) -> 'Dataset':
        if False:
            i = 10
            return i + 15
        'Apply the given function to each group of records of this dataset.\n\n        While map_groups() is very flexible, note that it comes with downsides:\n            * It may be slower than using more specific methods such as min(), max().\n            * It requires that each group fits in memory on a single node.\n\n        In general, prefer to use aggregate() instead of map_groups().\n\n        Examples:\n            >>> # Return a single record per group (list of multiple records in,\n            >>> # list of a single record out).\n            >>> import ray\n            >>> import pandas as pd\n            >>> import numpy as np\n            >>> # Get first value per group.\n            >>> ds = ray.data.from_items([ # doctest: +SKIP\n            ...     {"group": 1, "value": 1},\n            ...     {"group": 1, "value": 2},\n            ...     {"group": 2, "value": 3},\n            ...     {"group": 2, "value": 4}])\n            >>> ds.groupby("group").map_groups( # doctest: +SKIP\n            ...     lambda g: {"result": np.array([g["value"][0]])})\n\n            >>> # Return multiple records per group (dataframe in, dataframe out).\n            >>> df = pd.DataFrame(\n            ...     {"A": ["a", "a", "b"], "B": [1, 1, 3], "C": [4, 6, 5]}\n            ... )\n            >>> ds = ray.data.from_pandas(df) # doctest: +SKIP\n            >>> grouped = ds.groupby("A") # doctest: +SKIP\n            >>> grouped.map_groups( # doctest: +SKIP\n            ...     lambda g: g.apply(\n            ...         lambda c: c / g[c.name].sum() if c.name in ["B", "C"] else c\n            ...     )\n            ... ) # doctest: +SKIP\n\n        Args:\n            fn: The function to apply to each group of records, or a class type\n                that can be instantiated to create such a callable. It takes as\n                input a batch of all records from a single group, and returns a\n                batch of zero or more records, similar to map_batches().\n            compute: The compute strategy, either "tasks" (default) to use Ray\n                tasks, ``ray.data.ActorPoolStrategy(size=n)`` to use a fixed-size actor\n                pool, or ``ray.data.ActorPoolStrategy(min_size=m, max_size=n)`` for an\n                autoscaling actor pool.\n            batch_format: Specify ``"default"`` to use the default block format\n                (NumPy), ``"pandas"`` to select ``pandas.DataFrame``, "pyarrow" to\n                select ``pyarrow.Table``, or ``"numpy"`` to select\n                ``Dict[str, numpy.ndarray]``, or None to return the underlying block\n                exactly as is with no additional formatting.\n            fn_args: Arguments to `fn`.\n            fn_kwargs: Keyword arguments to `fn`.\n            ray_remote_args: Additional resource requirements to request from\n                ray (e.g., num_gpus=1 to request GPUs for the map tasks).\n\n        Returns:\n            The return type is determined by the return type of ``fn``, and the return\n            value is combined from results of all groups.\n        '
        if self._key is not None:
            sorted_ds = self._dataset.sort(self._key)
        else:
            sorted_ds = self._dataset.repartition(1)

        def get_key_boundaries(block_accessor: BlockAccessor):
            if False:
                print('Hello World!')
            import numpy as np
            boundaries = []
            keys = block_accessor.to_numpy(self._key)
            start = 0
            while start < keys.size:
                end = start + np.searchsorted(keys[start:], keys[start], side='right')
                boundaries.append(end)
                start = end
            return boundaries

        def group_fn(batch, *args, **kwargs):
            if False:
                while True:
                    i = 10
            block = BlockAccessor.batch_to_block(batch)
            block_accessor = BlockAccessor.for_block(block)
            if self._key:
                boundaries = get_key_boundaries(block_accessor)
            else:
                boundaries = [block_accessor.num_rows()]
            builder = DelegatingBlockBuilder()
            start = 0
            for end in boundaries:
                group_block = block_accessor.slice(start, end)
                group_block_accessor = BlockAccessor.for_block(group_block)
                group_batch = group_block_accessor.to_batch_format(batch_format)
                applied = fn(group_batch, *args, **kwargs)
                builder.add_batch(applied)
                start = end
            rs = builder.build()
            return rs
        return sorted_ds.map_batches(group_fn, batch_size=None, compute=compute, batch_format=batch_format, fn_args=fn_args, fn_kwargs=fn_kwargs, **ray_remote_args)

    def count(self) -> Dataset:
        if False:
            print('Hello World!')
        'Compute count aggregation.\n\n        Examples:\n            >>> import ray\n            >>> ray.data.from_items([ # doctest: +SKIP\n            ...     {"A": x % 3, "B": x} for x in range(100)]).groupby( # doctest: +SKIP\n            ...     "A").count() # doctest: +SKIP\n\n        Returns:\n            A dataset of ``[k, v]`` columns where ``k`` is the groupby key and\n            ``v`` is the number of rows with that key.\n            If groupby key is ``None`` then the key part of return is omitted.\n        '
        return self.aggregate(Count())

    def sum(self, on: Union[str, List[str]]=None, ignore_nulls: bool=True) -> Dataset:
        if False:
            return 10
        'Compute grouped sum aggregation.\n\n        Examples:\n            >>> import ray\n            >>> ray.data.from_items([ # doctest: +SKIP\n            ...     (i % 3, i, i**2) # doctest: +SKIP\n            ...     for i in range(100)]) \\ # doctest: +SKIP\n            ...     .groupby(lambda x: x[0] % 3) \\ # doctest: +SKIP\n            ...     .sum(lambda x: x[2]) # doctest: +SKIP\n            >>> ray.data.range(100).groupby("id").sum() # doctest: +SKIP\n            >>> ray.data.from_items([ # doctest: +SKIP\n            ...     {"A": i % 3, "B": i, "C": i**2} # doctest: +SKIP\n            ...     for i in range(100)]) \\ # doctest: +SKIP\n            ...     .groupby("A") \\ # doctest: +SKIP\n            ...     .sum(["B", "C"]) # doctest: +SKIP\n\n        Args:\n            on: a column name or a list of column names to aggregate.\n            ignore_nulls: Whether to ignore null values. If ``True``, null\n                values will be ignored when computing the sum; if ``False``,\n                if a null value is encountered, the output will be null.\n                We consider np.nan, None, and pd.NaT to be null values.\n                Default is ``True``.\n\n        Returns:\n            The sum result.\n\n            For different values of ``on``, the return varies:\n\n            - ``on=None``: a dataset containing a groupby key column,\n              ``"k"``, and a column-wise sum column for each original column\n              in the dataset.\n            - ``on=["col_1", ..., "col_n"]``: a dataset of ``n + 1``\n              columns where the first column is the groupby key and the second\n              through ``n + 1`` columns are the results of the aggregations.\n\n            If groupby key is ``None`` then the key part of return is omitted.\n        '
        return self._aggregate_on(Sum, on, ignore_nulls)

    def min(self, on: Union[str, List[str]]=None, ignore_nulls: bool=True) -> Dataset:
        if False:
            while True:
                i = 10
        'Compute grouped min aggregation.\n\n        Examples:\n            >>> import ray\n            >>> ray.data.le(100).groupby("value").min() # doctest: +SKIP\n            >>> ray.data.from_items([ # doctest: +SKIP\n            ...     {"A": i % 3, "B": i, "C": i**2} # doctest: +SKIP\n            ...     for i in range(100)]) \\ # doctest: +SKIP\n            ...     .groupby("A") \\ # doctest: +SKIP\n            ...     .min(["B", "C"]) # doctest: +SKIP\n\n        Args:\n            on: a column name or a list of column names to aggregate.\n            ignore_nulls: Whether to ignore null values. If ``True``, null\n                values will be ignored when computing the min; if ``False``,\n                if a null value is encountered, the output will be null.\n                We consider np.nan, None, and pd.NaT to be null values.\n                Default is ``True``.\n\n        Returns:\n            The min result.\n\n            For different values of ``on``, the return varies:\n\n            - ``on=None``: a dataset containing a groupby key column,\n              ``"k"``, and a column-wise min column for each original column in\n              the dataset.\n            - ``on=["col_1", ..., "col_n"]``: a dataset of ``n + 1``\n              columns where the first column is the groupby key and the second\n              through ``n + 1`` columns are the results of the aggregations.\n\n            If groupby key is ``None`` then the key part of return is omitted.\n        '
        return self._aggregate_on(Min, on, ignore_nulls)

    def max(self, on: Union[str, List[str]]=None, ignore_nulls: bool=True) -> Dataset:
        if False:
            print('Hello World!')
        'Compute grouped max aggregation.\n\n        Examples:\n            >>> import ray\n            >>> ray.data.le(100).groupby("value").max() # doctest: +SKIP\n            >>> ray.data.from_items([ # doctest: +SKIP\n            ...     {"A": i % 3, "B": i, "C": i**2} # doctest: +SKIP\n            ...     for i in range(100)]) \\ # doctest: +SKIP\n            ...     .groupby("A") \\ # doctest: +SKIP\n            ...     .max(["B", "C"]) # doctest: +SKIP\n\n        Args:\n            on: a column name or a list of column names to aggregate.\n            ignore_nulls: Whether to ignore null values. If ``True``, null\n                values will be ignored when computing the max; if ``False``,\n                if a null value is encountered, the output will be null.\n                We consider np.nan, None, and pd.NaT to be null values.\n                Default is ``True``.\n\n        Returns:\n            The max result.\n\n            For different values of ``on``, the return varies:\n\n            - ``on=None``: a dataset containing a groupby key column,\n              ``"k"``, and a column-wise max column for each original column in\n              the dataset.\n            - ``on=["col_1", ..., "col_n"]``: a dataset of ``n + 1``\n              columns where the first column is the groupby key and the second\n              through ``n + 1`` columns are the results of the aggregations.\n\n            If groupby key is ``None`` then the key part of return is omitted.\n        '
        return self._aggregate_on(Max, on, ignore_nulls)

    def mean(self, on: Union[str, List[str]]=None, ignore_nulls: bool=True) -> Dataset:
        if False:
            i = 10
            return i + 15
        'Compute grouped mean aggregation.\n\n        Examples:\n            >>> import ray\n            >>> ray.data.le(100).groupby("value").mean() # doctest: +SKIP\n            >>> ray.data.from_items([ # doctest: +SKIP\n            ...     {"A": i % 3, "B": i, "C": i**2} # doctest: +SKIP\n            ...     for i in range(100)]) \\ # doctest: +SKIP\n            ...     .groupby("A") \\ # doctest: +SKIP\n            ...     .mean(["B", "C"]) # doctest: +SKIP\n\n        Args:\n            on: a column name or a list of column names to aggregate.\n            ignore_nulls: Whether to ignore null values. If ``True``, null\n                values will be ignored when computing the mean; if ``False``,\n                if a null value is encountered, the output will be null.\n                We consider np.nan, None, and pd.NaT to be null values.\n                Default is ``True``.\n\n        Returns:\n            The mean result.\n\n            For different values of ``on``, the return varies:\n\n            - ``on=None``: a dataset containing a groupby key column,\n              ``"k"``, and a column-wise mean column for each original column\n              in the dataset.\n            - ``on=["col_1", ..., "col_n"]``: a dataset of ``n + 1``\n              columns where the first column is the groupby key and the second\n              through ``n + 1`` columns are the results of the aggregations.\n\n            If groupby key is ``None`` then the key part of return is omitted.\n        '
        return self._aggregate_on(Mean, on, ignore_nulls)

    def std(self, on: Union[str, List[str]]=None, ddof: int=1, ignore_nulls: bool=True) -> Dataset:
        if False:
            print('Hello World!')
        'Compute grouped standard deviation aggregation.\n\n        Examples:\n            >>> import ray\n            >>> ray.data.range(100).groupby("id").std(ddof=0) # doctest: +SKIP\n            >>> ray.data.from_items([ # doctest: +SKIP\n            ...     {"A": i % 3, "B": i, "C": i**2} # doctest: +SKIP\n            ...     for i in range(100)]) \\ # doctest: +SKIP\n            ...     .groupby("A") \\ # doctest: +SKIP\n            ...     .std(["B", "C"]) # doctest: +SKIP\n\n        NOTE: This uses Welford\'s online method for an accumulator-style\n        computation of the standard deviation. This method was chosen due to\n        it\'s numerical stability, and it being computable in a single pass.\n        This may give different (but more accurate) results than NumPy, Pandas,\n        and sklearn, which use a less numerically stable two-pass algorithm.\n        See\n        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford\'s_online_algorithm\n\n        Args:\n            on: a column name or a list of column names to aggregate.\n            ddof: Delta Degrees of Freedom. The divisor used in calculations\n                is ``N - ddof``, where ``N`` represents the number of elements.\n            ignore_nulls: Whether to ignore null values. If ``True``, null\n                values will be ignored when computing the std; if ``False``,\n                if a null value is encountered, the output will be null.\n                We consider np.nan, None, and pd.NaT to be null values.\n                Default is ``True``.\n\n        Returns:\n            The standard deviation result.\n\n            For different values of ``on``, the return varies:\n\n            - ``on=None``: a dataset containing a groupby key column,\n              ``"k"``, and a column-wise std column for each original column in\n              the dataset.\n            - ``on=["col_1", ..., "col_n"]``: a dataset of ``n + 1``\n              columns where the first column is the groupby key and the second\n              through ``n + 1`` columns are the results of the aggregations.\n\n            If groupby key is ``None`` then the key part of return is omitted.\n        '
        return self._aggregate_on(Std, on, ignore_nulls, ddof=ddof)
GroupedDataset = GroupedData