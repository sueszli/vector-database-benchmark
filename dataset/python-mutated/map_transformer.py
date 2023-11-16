import itertools
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union
from ray.data._internal.block_batching.block_batching import batch_blocks
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.output_buffer import BlockOutputBuffer
from ray.data.block import Block, BlockAccessor, DataBatch
Row = Dict[str, Any]
MapTransformFnData = Union[Block, Row, DataBatch]
IN = TypeVar('IN')
OUT = TypeVar('OUT')
MapTransformCallable = Callable[[Iterable[IN], TaskContext], Iterable[OUT]]

class MapTransformFnDataType(Enum):
    """An enum that represents the input/output data type of a MapTransformFn."""
    Block = 0
    Row = 1
    Batch = 2

class MapTransformFn:
    """Represents a single transform function in a MapTransformer."""

    def __init__(self, input_type: MapTransformFnDataType, output_type: MapTransformFnDataType):
        if False:
            return 10
        '\n        Args:\n            callable: the underlying Python callable object.\n            input_type: the type of the input data.\n            output_type: the type of the output data.\n        '
        self._callable = callable
        self._input_type = input_type
        self._output_type = output_type
        self._target_max_block_size = None

    @abstractmethod
    def __call__(self, input: Iterable[MapTransformFnData], ctx: TaskContext) -> Iterable[MapTransformFnData]:
        if False:
            i = 10
            return i + 15
        ...

    @property
    def input_type(self) -> MapTransformFnDataType:
        if False:
            for i in range(10):
                print('nop')
        return self._input_type

    @property
    def output_type(self) -> MapTransformFnDataType:
        if False:
            return 10
        return self._output_type

    def set_target_max_block_size(self, target_max_block_size: int):
        if False:
            while True:
                i = 10
        self._target_max_block_size = target_max_block_size

class MapTransformer:
    """Encapsulates the data transformation logic of a physical MapOperator.

    A MapTransformer may consist of one or more steps, each of which is represented
    as a MapTransformFn. The first MapTransformFn must take blocks as input, and
    the last MapTransformFn must output blocks. The intermediate data types can
    be blocks, rows, or batches.
    """

    def __init__(self, transform_fns: List[MapTransformFn], init_fn: Optional[Callable[[], None]]=None):
        if False:
            print('Hello World!')
        '\n        Args:\n        transform_fns: A list of `MapTransformFn`s that will be executed sequentially\n            to transform data.\n        init_fn: A function that will be called before transforming data.\n            Used for the actor-based map operator.\n        '
        self.set_transform_fns(transform_fns)
        self._init_fn = init_fn if init_fn is not None else lambda : None
        self._target_max_block_size = None

    def set_transform_fns(self, transform_fns: List[MapTransformFn]) -> None:
        if False:
            return 10
        'Set the transform functions.'
        assert len(transform_fns) > 0
        assert transform_fns[0].input_type == MapTransformFnDataType.Block, 'The first transform function must take blocks as input.'
        assert transform_fns[-1].output_type == MapTransformFnDataType.Block, 'The last transform function must output blocks.'
        for i in range(len(transform_fns) - 1):
            assert transform_fns[i].output_type == transform_fns[i + 1].input_type, 'The output type of the previous transform function must match the input type of the next transform function.'
        self._transform_fns = transform_fns

    def get_transform_fns(self) -> List[MapTransformFn]:
        if False:
            i = 10
            return i + 15
        'Get the transform functions.'
        return self._transform_fns

    def set_target_max_block_size(self, target_max_block_size: int):
        if False:
            while True:
                i = 10
        self._target_max_block_size = target_max_block_size

    def init(self) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize the transformer.\n\n        Should be called before applying the transform.\n        '
        self._init_fn()

    def apply_transform(self, input_blocks: Iterable[Block], ctx: TaskContext) -> Iterable[Block]:
        if False:
            for i in range(10):
                print('nop')
        'Apply the transform functions to the input blocks.'
        assert self._target_max_block_size is not None, 'target_max_block_size must be set before running'
        for transform_fn in self._transform_fns:
            transform_fn.set_target_max_block_size(self._target_max_block_size)
        iter = input_blocks
        for transform_fn in self._transform_fns:
            iter = transform_fn(iter, ctx)
        return iter

    def fuse(self, other: 'MapTransformer') -> 'MapTransformer':
        if False:
            return 10
        'Fuse two `MapTransformer`s together.'
        assert self._target_max_block_size == other._target_max_block_size or (self._target_max_block_size is None or other._target_max_block_size is None)
        target_max_block_size = self._target_max_block_size or other._target_max_block_size
        self_init_fn = self._init_fn
        other_init_fn = other._init_fn

        def fused_init_fn():
            if False:
                while True:
                    i = 10
            self_init_fn()
            other_init_fn()
        fused_transform_fns = self._transform_fns + other._transform_fns
        transformer = MapTransformer(fused_transform_fns, init_fn=fused_init_fn)
        transformer.set_target_max_block_size(target_max_block_size)
        return transformer

def create_map_transformer_from_block_fn(block_fn: MapTransformCallable[Block, Block], init_fn: Optional[Callable[[], None]]=None):
    if False:
        while True:
            i = 10
    'Create a MapTransformer from a single block-based transform function.\n\n    This method should only be used for testing and legacy compatibility.\n    '
    return MapTransformer([BlockMapTransformFn(block_fn)], init_fn)

class RowMapTransformFn(MapTransformFn):
    """A rows-to-rows MapTransformFn."""

    def __init__(self, row_fn: MapTransformCallable[Row, Row]):
        if False:
            while True:
                i = 10
        self._row_fn = row_fn
        super().__init__(MapTransformFnDataType.Row, MapTransformFnDataType.Row)

    def __call__(self, input: Iterable[Row], ctx: TaskContext) -> Iterable[Row]:
        if False:
            print('Hello World!')
        yield from self._row_fn(input, ctx)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'RowMapTransformFn({self._row_fn})'

class BatchMapTransformFn(MapTransformFn):
    """A batch-to-batch MapTransformFn."""

    def __init__(self, batch_fn: MapTransformCallable[DataBatch, DataBatch]):
        if False:
            while True:
                i = 10
        self._batch_fn = batch_fn
        super().__init__(MapTransformFnDataType.Batch, MapTransformFnDataType.Batch)

    def __call__(self, input: Iterable[DataBatch], ctx: TaskContext) -> Iterable[DataBatch]:
        if False:
            i = 10
            return i + 15
        yield from self._batch_fn(input, ctx)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'BatchMapTransformFn({self._batch_fn})'

class BlockMapTransformFn(MapTransformFn):
    """A block-to-block MapTransformFn."""

    def __init__(self, block_fn: MapTransformCallable[Block, Block]):
        if False:
            i = 10
            return i + 15
        self._block_fn = block_fn
        super().__init__(MapTransformFnDataType.Block, MapTransformFnDataType.Block)

    def __call__(self, input: Iterable[Block], ctx: TaskContext) -> Iterable[Block]:
        if False:
            while True:
                i = 10
        yield from self._block_fn(input, ctx)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'BlockMapTransformFn({self._block_fn})'

class BlocksToRowsMapTransformFn(MapTransformFn):
    """A MapTransformFn that converts input blocks to rows."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(MapTransformFnDataType.Block, MapTransformFnDataType.Row)

    def __call__(self, blocks: Iterable[Block], _: TaskContext) -> Iterable[Row]:
        if False:
            while True:
                i = 10
        for block in blocks:
            block = BlockAccessor.for_block(block)
            for row in block.iter_rows(public_row_format=True):
                yield row

    @classmethod
    def instance(cls) -> 'BlocksToRowsMapTransformFn':
        if False:
            print('Hello World!')
        'Returns the singleton instance.'
        if getattr(cls, '_instance', None) is None:
            cls._instance = cls()
        return cls._instance

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'BlocksToRowsMapTransformFn()'

class BlocksToBatchesMapTransformFn(MapTransformFn):
    """A MapTransformFn that converts input blocks to batches."""

    def __init__(self, batch_size: Optional[int]=None, batch_format: str='default', zero_copy_batch: bool=False):
        if False:
            print('Hello World!')
        self._batch_size = batch_size
        self._batch_format = batch_format
        self._ensure_copy = not zero_copy_batch and batch_size is not None
        super().__init__(MapTransformFnDataType.Block, MapTransformFnDataType.Batch)

    def __call__(self, blocks: Iterable[Block], _: TaskContext) -> Iterable[DataBatch]:
        if False:
            return 10
        'Converts input blocks to batches.'
        block_iter = iter(blocks)
        first = next(block_iter, None)
        if first is None:
            return []
        blocks = itertools.chain([first], block_iter)
        empty_block = BlockAccessor.for_block(first).builder().build()
        first = None
        formatted_batch_iter = batch_blocks(blocks=blocks, stats=None, batch_size=self._batch_size, batch_format=self._batch_format, ensure_copy=self._ensure_copy)
        first = next(formatted_batch_iter, None)
        if first is None:
            return [empty_block]
        else:
            return itertools.chain([first], formatted_batch_iter)

    @property
    def batch_size(self) -> Optional[int]:
        if False:
            return 10
        return self._batch_size

    @property
    def batch_format(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._batch_format

    @property
    def zero_copy_batch(self) -> bool:
        if False:
            print('Hello World!')
        return not self._ensure_copy

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'BlocksToBatchesMapTransformFn(batch_size={self._batch_size}, batch_format={self._batch_format}, zero_copy_batch={self.zero_copy_batch})'

class BuildOutputBlocksMapTransformFn(MapTransformFn):
    """A MapTransformFn that converts UDF-returned data to output blocks."""

    def __init__(self, input_type: MapTransformFnDataType):
        if False:
            return 10
        '\n        Args:\n            input_type: the type of input data.\n        '
        self._input_type = input_type
        super().__init__(input_type, MapTransformFnDataType.Block)

    def __call__(self, iter: Iterable[MapTransformFnData], _: TaskContext) -> Iterable[Block]:
        if False:
            for i in range(10):
                print('nop')
        'Convert UDF-returned data to output blocks.\n\n        Args:\n            iter: the iterable of UDF-returned data, whose type\n                must match self._input_type.\n        '
        assert self._target_max_block_size is not None, 'target_max_block_size must be set before running'
        output_buffer = BlockOutputBuffer(self._target_max_block_size)
        if self._input_type == MapTransformFnDataType.Block:
            add_fn = output_buffer.add_block
        elif self._input_type == MapTransformFnDataType.Batch:
            add_fn = output_buffer.add_batch
        else:
            assert self._input_type == MapTransformFnDataType.Row
            add_fn = output_buffer.add
        for data in iter:
            add_fn(data)
            while output_buffer.has_next():
                yield output_buffer.next()
        output_buffer.finalize()
        while output_buffer.has_next():
            yield output_buffer.next()

    @classmethod
    def for_rows(cls) -> 'BuildOutputBlocksMapTransformFn':
        if False:
            for i in range(10):
                print('nop')
        'Return a BuildOutputBlocksMapTransformFn for row input.'
        return cls(MapTransformFnDataType.Row)

    @classmethod
    def for_batches(cls) -> 'BuildOutputBlocksMapTransformFn':
        if False:
            while True:
                i = 10
        'Return a BuildOutputBlocksMapTransformFn for batch input.'
        return cls(MapTransformFnDataType.Batch)

    @classmethod
    def for_blocks(cls) -> 'BuildOutputBlocksMapTransformFn':
        if False:
            i = 10
            return i + 15
        'Return a BuildOutputBlocksMapTransformFn for block input.'
        return cls(MapTransformFnDataType.Block)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'BuildOutputBlocksMapTransformFn(input_type={self._input_type})'

def _splitrange(n, k):
    if False:
        while True:
            i = 10
    'Calculates array lens of np.array_split().\n\n    This is the equivalent of\n    `[len(x) for x in np.array_split(range(n), k)]`.\n    '
    base = n // k
    output = [base] * k
    rem = n - sum(output)
    for i in range(len(output)):
        if rem > 0:
            output[i] += 1
            rem -= 1
    assert rem == 0, (rem, output, n, k)
    assert sum(output) == n, (output, n, k)
    return output

class ApplyAdditionalSplitToOutputBlocks(MapTransformFn):
    """Do additional splits on output blocks."""

    def __init__(self, additional_split_factor: int):
        if False:
            print('Hello World!')
        '\n        Args:\n          additional_output_splits: The number of additional splits, must be\n          greater than 1.\n        '
        assert additional_split_factor > 1
        self._additional_split_factor = additional_split_factor
        super().__init__(MapTransformFnDataType.Block, MapTransformFnDataType.Block)

    def __call__(self, blocks: Iterable[Block], ctx: TaskContext) -> Iterable[Block]:
        if False:
            return 10
        for block in blocks:
            block = BlockAccessor.for_block(block)
            offset = 0
            split_sizes = _splitrange(block.num_rows(), self._additional_split_factor)
            for size in split_sizes:
                yield block.slice(offset, offset + size, copy=True)
                offset += size