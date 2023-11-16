import inspect
from typing import Any, Dict, Iterable, Optional, Union
from ray.data._internal.compute import ComputeStrategy, TaskPoolStrategy
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators.one_to_one_operator import AbstractOneToOne
from ray.data.block import UserDefinedFunction
from ray.data.context import DEFAULT_BATCH_SIZE
logger = DatasetLogger(__name__)

class AbstractMap(AbstractOneToOne):
    """Abstract class for logical operators that should be converted to physical
    MapOperator.
    """

    def __init__(self, name: str, input_op: Optional[LogicalOperator]=None, ray_remote_args: Optional[Dict[str, Any]]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            name: Name for this operator. This is the name that will appear when\n                inspecting the logical plan of a Dataset.\n            input_op: The operator preceding this operator in the plan DAG. The outputs\n                of `input_op` will be the inputs to this operator.\n            ray_remote_args: Args to provide to ray.remote.\n        '
        super().__init__(name, input_op)
        self._ray_remote_args = ray_remote_args or {}

def _get_udf_name(fn: UserDefinedFunction) -> str:
    if False:
        for i in range(10):
            print('nop')
    try:
        if inspect.isclass(fn):
            return fn.__name__
        elif inspect.ismethod(fn):
            return f'{fn.__self__.__class__.__name__}.{fn.__name__}'
        elif inspect.isfunction(fn):
            return fn.__name__
        else:
            return fn.__class__.__name__
    except AttributeError as e:
        logger.get_logger().error('Failed to get name of UDF %s: %s', fn, e)
        return '<unknown>'

class AbstractUDFMap(AbstractMap):
    """Abstract class for logical operators performing a UDF that should be converted
    to physical MapOperator.
    """

    def __init__(self, name: str, input_op: LogicalOperator, fn: UserDefinedFunction, fn_args: Optional[Iterable[Any]]=None, fn_kwargs: Optional[Dict[str, Any]]=None, fn_constructor_args: Optional[Iterable[Any]]=None, fn_constructor_kwargs: Optional[Dict[str, Any]]=None, min_rows_per_block: Optional[int]=None, compute: Optional[Union[str, ComputeStrategy]]=None, ray_remote_args: Optional[Dict[str, Any]]=None):
        if False:
            print('Hello World!')
        '\n        Args:\n            name: Name for this operator. This is the name that will appear when\n                inspecting the logical plan of a Dataset.\n            input_op: The operator preceding this operator in the plan DAG. The outputs\n                of `input_op` will be the inputs to this operator.\n            fn: User-defined function to be called.\n            fn_args: Arguments to `fn`.\n            fn_kwargs: Keyword arguments to `fn`.\n            fn_constructor_args: Arguments to provide to the initializor of `fn` if\n                `fn` is a callable class.\n            fn_constructor_kwargs: Keyword Arguments to provide to the initializor of\n                `fn` if `fn` is a callable class.\n            min_rows_per_block: The target size for blocks outputted by this operator.\n            compute: The compute strategy, either ``"tasks"`` (default) to use Ray\n                tasks, or ``"actors"`` to use an autoscaling actor pool.\n            ray_remote_args: Args to provide to ray.remote.\n        '
        name = f'{name}({_get_udf_name(fn)})'
        super().__init__(name, input_op, ray_remote_args)
        self._fn = fn
        self._fn_args = fn_args
        self._fn_kwargs = fn_kwargs
        self._fn_constructor_args = fn_constructor_args
        self._fn_constructor_kwargs = fn_constructor_kwargs
        self._min_rows_per_block = min_rows_per_block
        self._compute = compute or TaskPoolStrategy()

class MapBatches(AbstractUDFMap):
    """Logical operator for map_batches."""

    def __init__(self, input_op: LogicalOperator, fn: UserDefinedFunction, batch_size: Optional[int]=DEFAULT_BATCH_SIZE, batch_format: str='default', zero_copy_batch: bool=False, fn_args: Optional[Iterable[Any]]=None, fn_kwargs: Optional[Dict[str, Any]]=None, fn_constructor_args: Optional[Iterable[Any]]=None, fn_constructor_kwargs: Optional[Dict[str, Any]]=None, min_rows_per_block: Optional[int]=None, compute: Optional[Union[str, ComputeStrategy]]=None, ray_remote_args: Optional[Dict[str, Any]]=None):
        if False:
            i = 10
            return i + 15
        super().__init__('MapBatches', input_op, fn, fn_args=fn_args, fn_kwargs=fn_kwargs, fn_constructor_args=fn_constructor_args, fn_constructor_kwargs=fn_constructor_kwargs, min_rows_per_block=min_rows_per_block, compute=compute, ray_remote_args=ray_remote_args)
        self._batch_size = batch_size
        self._batch_format = batch_format
        self._zero_copy_batch = zero_copy_batch

    @property
    def can_modify_num_rows(self) -> bool:
        if False:
            print('Hello World!')
        return False

class MapRows(AbstractUDFMap):
    """Logical operator for map."""

    def __init__(self, input_op: LogicalOperator, fn: UserDefinedFunction, fn_args: Optional[Iterable[Any]]=None, fn_kwargs: Optional[Dict[str, Any]]=None, fn_constructor_args: Optional[Iterable[Any]]=None, fn_constructor_kwargs: Optional[Dict[str, Any]]=None, compute: Optional[Union[str, ComputeStrategy]]=None, ray_remote_args: Optional[Dict[str, Any]]=None):
        if False:
            print('Hello World!')
        super().__init__('Map', input_op, fn, fn_args=fn_args, fn_kwargs=fn_kwargs, fn_constructor_args=fn_constructor_args, fn_constructor_kwargs=fn_constructor_kwargs, compute=compute, ray_remote_args=ray_remote_args)

    @property
    def can_modify_num_rows(self) -> bool:
        if False:
            return 10
        return False

class Filter(AbstractUDFMap):
    """Logical operator for filter."""

    def __init__(self, input_op: LogicalOperator, fn: UserDefinedFunction, compute: Optional[Union[str, ComputeStrategy]]=None, ray_remote_args: Optional[Dict[str, Any]]=None):
        if False:
            while True:
                i = 10
        super().__init__('Filter', input_op, fn, compute=compute, ray_remote_args=ray_remote_args)

    @property
    def can_modify_num_rows(self) -> bool:
        if False:
            print('Hello World!')
        return True

class FlatMap(AbstractUDFMap):
    """Logical operator for flat_map."""

    def __init__(self, input_op: LogicalOperator, fn: UserDefinedFunction, fn_args: Optional[Iterable[Any]]=None, fn_kwargs: Optional[Dict[str, Any]]=None, fn_constructor_args: Optional[Iterable[Any]]=None, fn_constructor_kwargs: Optional[Dict[str, Any]]=None, compute: Optional[Union[str, ComputeStrategy]]=None, ray_remote_args: Optional[Dict[str, Any]]=None):
        if False:
            print('Hello World!')
        super().__init__('FlatMap', input_op, fn, fn_args=fn_args, fn_kwargs=fn_kwargs, fn_constructor_args=fn_constructor_args, fn_constructor_kwargs=fn_constructor_kwargs, compute=compute, ray_remote_args=ray_remote_args)

    @property
    def can_modify_num_rows(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True