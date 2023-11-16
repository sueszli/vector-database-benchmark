import math
from typing import TYPE_CHECKING, Callable, List, Optional, Union
from ray.data._internal.null_aggregate import _null_wrap_accumulate_block, _null_wrap_accumulate_row, _null_wrap_finalize, _null_wrap_init, _null_wrap_merge
from ray.data._internal.sort import SortKey
from ray.data.block import AggType, Block, BlockAccessor, KeyType, T, U
from ray.util.annotations import PublicAPI
if TYPE_CHECKING:
    import pyarrow as pa

@PublicAPI
class AggregateFn:

    def __init__(self, init: Callable[[KeyType], AggType], merge: Callable[[AggType, AggType], AggType], accumulate_row: Callable[[AggType, T], AggType]=None, accumulate_block: Callable[[AggType, Block], AggType]=None, finalize: Callable[[AggType], U]=lambda a: a, name: Optional[str]=None):
        if False:
            print('Hello World!')
        'Defines an aggregate function in the accumulator style.\n\n        Aggregates a collection of inputs of type T into\n        a single output value of type U.\n        See https://www.sigops.org/s/conferences/sosp/2009/papers/yu-sosp09.pdf\n        for more details about accumulator-based aggregation.\n\n        Args:\n            init: This is called once for each group to return the empty accumulator.\n                For example, an empty accumulator for a sum would be 0.\n            merge: This may be called multiple times, each time to merge\n                two accumulators into one.\n            accumulate_row: This is called once per row of the same group.\n                This combines the accumulator and the row, returns the updated\n                accumulator. Exactly one of accumulate_row and accumulate_block must\n                be provided.\n            accumulate_block: This is used to calculate the aggregation for a\n                single block, and is vectorized alternative to accumulate_row. This will\n                be given a base accumulator and the entire block, allowing for\n                vectorized accumulation of the block. Exactly one of accumulate_row and\n                accumulate_block must be provided.\n            finalize: This is called once to compute the final aggregation\n                result from the fully merged accumulator.\n            name: The name of the aggregation. This will be used as the output\n                column name in the case of Arrow dataset.\n        '
        if accumulate_row is None and accumulate_block is None or (accumulate_row is not None and accumulate_block is not None):
            raise ValueError('Exactly one of accumulate_row or accumulate_block must be provided.')
        if accumulate_block is None:

            def accumulate_block(a: AggType, block: Block) -> AggType:
                if False:
                    while True:
                        i = 10
                block_acc = BlockAccessor.for_block(block)
                for r in block_acc.iter_rows(public_row_format=False):
                    a = accumulate_row(a, r)
                return a
        self.init = init
        self.merge = merge
        self.accumulate_block = accumulate_block
        self.finalize = finalize
        self.name = name

    def _validate(self, schema: Optional[Union[type, 'pa.lib.Schema']]) -> None:
        if False:
            return 10
        'Raise an error if this cannot be applied to the given schema.'
        pass

class _AggregateOnKeyBase(AggregateFn):

    def _set_key_fn(self, on: str):
        if False:
            while True:
                i = 10
        self._key_fn = on

    def _validate(self, schema: Optional[Union[type, 'pa.lib.Schema']]) -> None:
        if False:
            while True:
                i = 10
        SortKey(self._key_fn).validate_schema(schema)

@PublicAPI
class Count(AggregateFn):
    """Defines count aggregation."""

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__(init=lambda k: 0, accumulate_block=lambda a, block: a + BlockAccessor.for_block(block).num_rows(), merge=lambda a1, a2: a1 + a2, name='count()')

@PublicAPI
class Sum(_AggregateOnKeyBase):
    """Defines sum aggregation."""

    def __init__(self, on: Optional[str]=None, ignore_nulls: bool=True, alias_name: Optional[str]=None):
        if False:
            print('Hello World!')
        self._set_key_fn(on)
        if alias_name:
            self._rs_name = alias_name
        else:
            self._rs_name = f'sum({str(on)})'
        null_merge = _null_wrap_merge(ignore_nulls, lambda a1, a2: a1 + a2)
        super().__init__(init=_null_wrap_init(lambda k: 0), merge=null_merge, accumulate_block=_null_wrap_accumulate_block(ignore_nulls, lambda block: BlockAccessor.for_block(block).sum(on, ignore_nulls), null_merge), finalize=_null_wrap_finalize(lambda a: a), name=self._rs_name)

@PublicAPI
class Min(_AggregateOnKeyBase):
    """Defines min aggregation."""

    def __init__(self, on: Optional[str]=None, ignore_nulls: bool=True, alias_name: Optional[str]=None):
        if False:
            print('Hello World!')
        self._set_key_fn(on)
        if alias_name:
            self._rs_name = alias_name
        else:
            self._rs_name = f'min({str(on)})'
        null_merge = _null_wrap_merge(ignore_nulls, min)
        super().__init__(init=_null_wrap_init(lambda k: float('inf')), merge=null_merge, accumulate_block=_null_wrap_accumulate_block(ignore_nulls, lambda block: BlockAccessor.for_block(block).min(on, ignore_nulls), null_merge), finalize=_null_wrap_finalize(lambda a: a), name=self._rs_name)

@PublicAPI
class Max(_AggregateOnKeyBase):
    """Defines max aggregation."""

    def __init__(self, on: Optional[str]=None, ignore_nulls: bool=True, alias_name: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        self._set_key_fn(on)
        if alias_name:
            self._rs_name = alias_name
        else:
            self._rs_name = f'max({str(on)})'
        null_merge = _null_wrap_merge(ignore_nulls, max)
        super().__init__(init=_null_wrap_init(lambda k: float('-inf')), merge=null_merge, accumulate_block=_null_wrap_accumulate_block(ignore_nulls, lambda block: BlockAccessor.for_block(block).max(on, ignore_nulls), null_merge), finalize=_null_wrap_finalize(lambda a: a), name=self._rs_name)

@PublicAPI
class Mean(_AggregateOnKeyBase):
    """Defines mean aggregation."""

    def __init__(self, on: Optional[str]=None, ignore_nulls: bool=True, alias_name: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        self._set_key_fn(on)
        if alias_name:
            self._rs_name = alias_name
        else:
            self._rs_name = f'mean({str(on)})'
        null_merge = _null_wrap_merge(ignore_nulls, lambda a1, a2: [a1[0] + a2[0], a1[1] + a2[1]])

        def vectorized_mean(block: Block) -> AggType:
            if False:
                print('Hello World!')
            block_acc = BlockAccessor.for_block(block)
            count = block_acc.count(on)
            if count == 0 or count is None:
                return None
            sum_ = block_acc.sum(on, ignore_nulls)
            if sum_ is None:
                return None
            return [sum_, count]
        super().__init__(init=_null_wrap_init(lambda k: [0, 0]), merge=null_merge, accumulate_block=_null_wrap_accumulate_block(ignore_nulls, vectorized_mean, null_merge), finalize=_null_wrap_finalize(lambda a: a[0] / a[1]), name=self._rs_name)

@PublicAPI
class Std(_AggregateOnKeyBase):
    """Defines standard deviation aggregation.

    Uses Welford's online method for an accumulator-style computation of the
    standard deviation. This method was chosen due to its numerical
    stability, and it being computable in a single pass.
    This may give different (but more accurate) results than NumPy, Pandas,
    and sklearn, which use a less numerically stable two-pass algorithm.
    See
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self, on: Optional[str]=None, ddof: int=1, ignore_nulls: bool=True, alias_name: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        self._set_key_fn(on)
        if alias_name:
            self._rs_name = alias_name
        else:
            self._rs_name = f'std({str(on)})'

        def merge(a: List[float], b: List[float]):
            if False:
                i = 10
                return i + 15
            (M2_a, mean_a, count_a) = a
            (M2_b, mean_b, count_b) = b
            delta = mean_b - mean_a
            count = count_a + count_b
            mean = (mean_a * count_a + mean_b * count_b) / count
            M2 = M2_a + M2_b + delta ** 2 * count_a * count_b / count
            return [M2, mean, count]
        null_merge = _null_wrap_merge(ignore_nulls, merge)

        def vectorized_std(block: Block) -> AggType:
            if False:
                for i in range(10):
                    print('nop')
            block_acc = BlockAccessor.for_block(block)
            count = block_acc.count(on)
            if count == 0 or count is None:
                return None
            sum_ = block_acc.sum(on, ignore_nulls)
            if sum_ is None:
                return None
            mean = sum_ / count
            M2 = block_acc.sum_of_squared_diffs_from_mean(on, ignore_nulls, mean)
            return [M2, mean, count]

        def finalize(a: List[float]):
            if False:
                i = 10
                return i + 15
            (M2, mean, count) = a
            if count < 2:
                return 0.0
            return math.sqrt(M2 / (count - ddof))
        super().__init__(init=_null_wrap_init(lambda k: [0, 0, 0]), merge=null_merge, accumulate_block=_null_wrap_accumulate_block(ignore_nulls, vectorized_std, null_merge), finalize=_null_wrap_finalize(finalize), name=self._rs_name)

@PublicAPI
class AbsMax(_AggregateOnKeyBase):
    """Defines absolute max aggregation."""

    def __init__(self, on: Optional[str]=None, ignore_nulls: bool=True, alias_name: Optional[str]=None):
        if False:
            while True:
                i = 10
        self._set_key_fn(on)
        on_fn = _to_on_fn(on)
        if alias_name:
            self._rs_name = alias_name
        else:
            self._rs_name = f'abs_max({str(on)})'
        super().__init__(init=_null_wrap_init(lambda k: 0), merge=_null_wrap_merge(ignore_nulls, max), accumulate_row=_null_wrap_accumulate_row(ignore_nulls, on_fn, lambda a, r: max(a, abs(r))), finalize=_null_wrap_finalize(lambda a: a), name=self._rs_name)

def _to_on_fn(on: Optional[str]):
    if False:
        for i in range(10):
            print('nop')
    if on is None:
        return lambda r: r
    elif isinstance(on, str):
        return lambda r: r[on]
    else:
        return on

@PublicAPI
class Quantile(_AggregateOnKeyBase):
    """Defines Quantile aggregation."""

    def __init__(self, on: Optional[str]=None, q: float=0.5, ignore_nulls: bool=True, alias_name: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        self._set_key_fn(on)
        self._q = q
        if alias_name:
            self._rs_name = alias_name
        else:
            self._rs_name = f'quantile({str(on)})'

        def merge(a: List[int], b: List[int]):
            if False:
                while True:
                    i = 10
            if isinstance(a, List) and isinstance(b, List):
                a.extend(b)
                return a
            if isinstance(a, List) and (not isinstance(b, List)):
                if b is not None and b != '':
                    a.append(b)
                return a
            if isinstance(b, List) and (not isinstance(a, List)):
                if a is not None and a != '':
                    b.append(a)
                return b
            ls = []
            if a is not None and a != '':
                ls.append(a)
            if b is not None and b != '':
                ls.append(b)
            return ls
        null_merge = _null_wrap_merge(ignore_nulls, merge)

        def block_row_ls(block: Block) -> AggType:
            if False:
                return 10
            block_acc = BlockAccessor.for_block(block)
            ls = []
            for row in block_acc.iter_rows(public_row_format=False):
                ls.append(row.get(on))
            return ls
        import math

        def percentile(input_values, key=lambda x: x):
            if False:
                return 10
            if not input_values:
                return None
            input_values = sorted(input_values)
            k = (len(input_values) - 1) * self._q
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return key(input_values[int(k)])
            d0 = key(input_values[int(f)]) * (c - k)
            d1 = key(input_values[int(c)]) * (k - f)
            return round(d0 + d1, 5)
        super().__init__(init=_null_wrap_init(lambda k: [0]), merge=null_merge, accumulate_block=_null_wrap_accumulate_block(ignore_nulls, block_row_ls, null_merge), finalize=_null_wrap_finalize(percentile), name=self._rs_name)