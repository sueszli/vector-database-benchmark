"""Module houses classes responsible for storing a virtual partition and applying a function to it."""
import pandas
import ray
from ray.util import get_node_ip_address
from modin.core.dataframe.pandas.partitioning.axis_partition import PandasDataframeAxisPartition
from modin.core.execution.ray.common import RayWrapper
from modin.utils import _inherit_docstrings
from .partition import PandasOnRayDataframePartition

class PandasOnRayDataframeVirtualPartition(PandasDataframeAxisPartition):
    """
    The class implements the interface in ``PandasDataframeAxisPartition``.

    Parameters
    ----------
    list_of_partitions : Union[list, PandasOnRayDataframePartition]
        List of ``PandasOnRayDataframePartition`` and
        ``PandasOnRayDataframeVirtualPartition`` objects, or a single
        ``PandasOnRayDataframePartition``.
    get_ip : bool, default: False
        Whether to get node IP addresses to conforming partitions or not.
    full_axis : bool, default: True
        Whether or not the virtual partition encompasses the whole axis.
    call_queue : list, optional
        A list of tuples (callable, args, kwargs) that contains deferred calls.
    length : ray.ObjectRef or int, optional
        Length, or reference to length, of wrapped ``pandas.DataFrame``.
    width : ray.ObjectRef or int, optional
        Width, or reference to width, of wrapped ``pandas.DataFrame``.
    """
    _PARTITIONS_METADATA_LEN = 3
    partition_type = PandasOnRayDataframePartition
    instance_type = ray.ObjectRef
    axis = None
    _DEPLOY_AXIS_FUNC = None
    _DEPLOY_SPLIT_FUNC = None
    _DRAIN_FUNC = None

    @classmethod
    def _get_deploy_axis_func(cls):
        if False:
            for i in range(10):
                print('nop')
        if cls._DEPLOY_AXIS_FUNC is None:
            cls._DEPLOY_AXIS_FUNC = RayWrapper.put(PandasDataframeAxisPartition.deploy_axis_func)
        return cls._DEPLOY_AXIS_FUNC

    @classmethod
    def _get_deploy_split_func(cls):
        if False:
            while True:
                i = 10
        if cls._DEPLOY_SPLIT_FUNC is None:
            cls._DEPLOY_SPLIT_FUNC = RayWrapper.put(PandasDataframeAxisPartition.deploy_splitting_func)
        return cls._DEPLOY_SPLIT_FUNC

    @classmethod
    def _get_drain_func(cls):
        if False:
            return 10
        if cls._DRAIN_FUNC is None:
            cls._DRAIN_FUNC = RayWrapper.put(PandasDataframeAxisPartition.drain)
        return cls._DRAIN_FUNC

    @property
    def list_of_ips(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the IPs holding the physical objects composing this partition.\n\n        Returns\n        -------\n        List\n            A list of IPs as ``ray.ObjectRef`` or str.\n        '
        result = [None] * len(self.list_of_block_partitions)
        for (idx, partition) in enumerate(self.list_of_block_partitions):
            partition.drain_call_queue()
            result[idx] = partition.ip(materialize=False)
        return result

    @classmethod
    @_inherit_docstrings(PandasDataframeAxisPartition.deploy_splitting_func)
    def deploy_splitting_func(cls, axis, func, f_args, f_kwargs, num_splits, *partitions, extract_metadata=False):
        if False:
            for i in range(10):
                print('nop')
        return _deploy_ray_func.options(num_returns=num_splits * (1 + cls._PARTITIONS_METADATA_LEN) if extract_metadata else num_splits).remote(cls._get_deploy_split_func(), *f_args, num_splits, *partitions, axis=axis, f_to_deploy=func, f_len_args=len(f_args), f_kwargs=f_kwargs, extract_metadata=extract_metadata)

    @classmethod
    def deploy_axis_func(cls, axis, func, f_args, f_kwargs, num_splits, maintain_partitioning, *partitions, lengths=None, manual_partition=False, max_retries=None):
        if False:
            return 10
        '\n        Deploy a function along a full axis.\n\n        Parameters\n        ----------\n        axis : {0, 1}\n            The axis to perform the function along.\n        func : callable\n            The function to perform.\n        f_args : list or tuple\n            Positional arguments to pass to ``func``.\n        f_kwargs : dict\n            Keyword arguments to pass to ``func``.\n        num_splits : int\n            The number of splits to return (see ``split_result_of_axis_func_pandas``).\n        maintain_partitioning : bool\n            If True, keep the old partitioning if possible.\n            If False, create a new partition layout.\n        *partitions : iterable\n            All partitions that make up the full axis (row or column).\n        lengths : list, optional\n            The list of lengths to shuffle the object.\n        manual_partition : bool, default: False\n            If True, partition the result with `lengths`.\n        max_retries : int, default: None\n            The max number of times to retry the func.\n\n        Returns\n        -------\n        list\n            A list of ``ray.ObjectRef``-s.\n        '
        return _deploy_ray_func.options(num_returns=(num_splits if lengths is None else len(lengths)) * (1 + cls._PARTITIONS_METADATA_LEN), **{'max_retries': max_retries} if max_retries is not None else {}).remote(cls._get_deploy_axis_func(), *f_args, num_splits, maintain_partitioning, *partitions, axis=axis, f_to_deploy=func, f_len_args=len(f_args), f_kwargs=f_kwargs, manual_partition=manual_partition, lengths=lengths)

    @classmethod
    def deploy_func_between_two_axis_partitions(cls, axis, func, f_args, f_kwargs, num_splits, len_of_left, other_shape, *partitions):
        if False:
            i = 10
            return i + 15
        '\n        Deploy a function along a full axis between two data sets.\n\n        Parameters\n        ----------\n        axis : {0, 1}\n            The axis to perform the function along.\n        func : callable\n            The function to perform.\n        f_args : list or tuple\n            Positional arguments to pass to ``func``.\n        f_kwargs : dict\n            Keyword arguments to pass to ``func``.\n        num_splits : int\n            The number of splits to return (see ``split_result_of_axis_func_pandas``).\n        len_of_left : int\n            The number of values in `partitions` that belong to the left data set.\n        other_shape : np.ndarray\n            The shape of right frame in terms of partitions, i.e.\n            (other_shape[i-1], other_shape[i]) will indicate slice to restore i-1 axis partition.\n        *partitions : iterable\n            All partitions that make up the full axis (row or column) for both data sets.\n\n        Returns\n        -------\n        list\n            A list of ``ray.ObjectRef``-s.\n        '
        return _deploy_ray_func.options(num_returns=num_splits * (1 + cls._PARTITIONS_METADATA_LEN)).remote(PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions, *f_args, num_splits, len_of_left, other_shape, *partitions, axis=axis, f_to_deploy=func, f_len_args=len(f_args), f_kwargs=f_kwargs)

    def wait(self):
        if False:
            for i in range(10):
                print('nop')
        'Wait completing computations on the object wrapped by the partition.'
        self.drain_call_queue()
        futures = self.list_of_blocks
        RayWrapper.wait(futures)

@_inherit_docstrings(PandasOnRayDataframeVirtualPartition.__init__)
class PandasOnRayDataframeColumnPartition(PandasOnRayDataframeVirtualPartition):
    axis = 0

@_inherit_docstrings(PandasOnRayDataframeVirtualPartition.__init__)
class PandasOnRayDataframeRowPartition(PandasOnRayDataframeVirtualPartition):
    axis = 1

@ray.remote
def _deploy_ray_func(deployer, *positional_args, axis, f_to_deploy, f_len_args, f_kwargs, extract_metadata=True, **kwargs):
    if False:
        print('Hello World!')
    '\n    Execute a function on an axis partition in a worker process.\n\n    This is ALWAYS called on either ``PandasDataframeAxisPartition.deploy_axis_func``\n    or ``PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions``, which both\n    serve to deploy another dataframe function on a Ray worker process. The provided `positional_args`\n    contains positional arguments for both: `deployer` and for `f_to_deploy`, the parameters can be separated\n    using the `f_len_args` value. The parameters are combined so they will be deserialized by Ray before the\n    kernel is executed (`f_kwargs` will never contain more Ray objects, and thus does not require deserialization).\n\n    Parameters\n    ----------\n    deployer : callable\n        A `PandasDataFrameAxisPartition.deploy_*` method that will call ``f_to_deploy``.\n    *positional_args : list\n        The first `f_len_args` elements in this list represent positional arguments\n        to pass to the `f_to_deploy`. The rest are positional arguments that will be\n        passed to `deployer`.\n    axis : {0, 1}\n        The axis to perform the function along. This argument is keyword only.\n    f_to_deploy : callable or RayObjectID\n        The function to deploy. This argument is keyword only.\n    f_len_args : int\n        Number of positional arguments to pass to ``f_to_deploy``. This argument is keyword only.\n    f_kwargs : dict\n        Keyword arguments to pass to ``f_to_deploy``. This argument is keyword only.\n    extract_metadata : bool, default: True\n        Whether to return metadata (length, width, ip) of the result. Passing `False` may relax\n        the load on object storage as the remote function would return 4 times fewer futures.\n        Passing `False` makes sense for temporary results where you know for sure that the\n        metadata will never be requested. This argument is keyword only.\n    **kwargs : dict\n        Keyword arguments to pass to ``deployer``.\n\n    Returns\n    -------\n    list : Union[tuple, list]\n        The result of the function call, and metadata for it.\n\n    Notes\n    -----\n    Ray functions are not detected by codecov (thus pragma: no cover).\n    '
    f_args = positional_args[:f_len_args]
    deploy_args = positional_args[f_len_args:]
    result = deployer(axis, f_to_deploy, f_args, f_kwargs, *deploy_args, **kwargs)
    if not extract_metadata:
        return result
    ip = get_node_ip_address()
    if isinstance(result, pandas.DataFrame):
        return (result, len(result), len(result.columns), ip)
    elif all((isinstance(r, pandas.DataFrame) for r in result)):
        return [i for r in result for i in [r, len(r), len(r.columns), ip]]
    else:
        return [i for r in result for i in [r, None, None, ip]]