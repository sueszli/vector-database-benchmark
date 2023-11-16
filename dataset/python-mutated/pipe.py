"""The Pipe interface."""
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast
import torch
from torch import Tensor, nn
from torch.distributed.rpc import RRef
import torch.autograd
import torch.cuda
from . import microbatch
from .batchnorm import DeferredBatchNorm
from .pipeline import Pipeline
from .skip.layout import inspect_skip_layout
from .skip.skippable import verify_skippables
from .stream import AbstractStream, new_stream
__all__ = ['Pipe', 'BalanceError', 'PipeSequential', 'WithDevice']
Device = Union[torch.device, int, str]
Devices = Union[Iterable[Device], List[Device]]
Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]
if TYPE_CHECKING:
    Module = nn.Module[TensorOrTensors]
    NamedModules = OrderedDict[str, Module]
else:
    Module = nn.Module
    NamedModules = OrderedDict

def _recommend_auto_balance(message: str) -> str:
    if False:
        i = 10
        return i + 15
    'Expands a message with recommendation to :mod:`torchpipe.balance`.'
    return f"{message}\n\nIf your model is still under development, its optimal balance would change\nfrequently. In this case, we highly recommend 'torch.distributed.pipeline.sync.balance' for\nnaive automatic balancing:\n\n  from torch.distributed.pipeline.sync import Pipe\n  from torch.distributed.pipeline.sync.balance import balance_by_time\n\n  partitions = torch.cuda.device_count()\n  sample = torch.empty(...)\n  balance = balance_by_time(partitions, model, sample)\n\n  model = Pipe(model, balance, ...)\n"

def _verify_module(module: nn.Sequential) -> None:
    if False:
        print('Hello World!')
    if not isinstance(module, nn.Sequential):
        raise TypeError('module must be nn.Sequential to be partitioned')
    named_children = list(module.named_children())
    if len(named_children) != len(module):
        raise ValueError('module with duplicate children is not supported')

def _verify_splitting(module: nn.Sequential, partitions: List[nn.Sequential], devices: List[torch.device]) -> None:
    if False:
        for i in range(10):
            print('nop')
    num_parameters = len(list(module.parameters()))
    num_child_parameters = sum((len(list(child.parameters())) for child in module.children()))
    if num_parameters == num_child_parameters:
        return
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            parti = partitions[i]
            partj = partitions[j]
            if devices[i] == devices[j]:
                continue
            for p in parti.parameters():
                for q in partj.parameters():
                    if p is q:
                        raise ValueError('module with duplicate parameters on distinct devices is not supported')

class BalanceError(ValueError):
    pass

def _retrieve_device(module: nn.Module) -> torch.device:
    if False:
        for i in range(10):
            print('nop')
    'Validates all parameters in the Module have the same device and returns\n    the appropriate device.\n\n    Args:\n        An ``nn.Module`` to process.\n\n    Returns:\n        ``torch.Device`` for the entire module.\n\n    Raises:\n        ValueError:\n            If devices for ``nn.Module`` parameters are not all same.\n    '
    device = None
    for parameter in module.parameters():
        if device is None:
            device = parameter.device
        elif device != parameter.device:
            raise ValueError(f'nn.Module: {module}, should have all parameters on a single device, please use .to() to place the module on a single device')
    return device if device is not None else torch.device('cpu')

class PipeSequential(nn.Sequential):
    """
    Pipe variant of ``nn.Sequential`` which supports multiple inputs.
    """

    def forward(self, *inputs):
        if False:
            while True:
                i = 10
        for module in self:
            if isinstance(inputs, Tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class WithDevice(nn.Module):
    """
    Wraps an ``nn.Module`` which is part of ``nn.Sequential`` passed into :class:`Pipe`
    that overrides the device for that module. In cases where :class:`Pipe`
    can't implicitly determine the device for the module and places it on CPU,
    this wrapper can be used to override the implicit behavior and explicitly
    specify which device a module should run on.

    The provided module is also moved to the given device via ``.to(device)``
    by :class:`Pipe`

    Args:
        module(:class:`torch.nn.Module`): The module to be wrapped.
        device(:class:`torch.device`): The device to run the module on.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> fc1 = nn.Linear(16, 8).cuda(0)
        >>> fc2 = nn.Linear(8, 4).cuda(1)
        >>> dropout = nn.Dropout()
        >>>
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
        >>> # Dropout does not have any parameters/buffers, but we want to
        >>> # run it on cuda:1 to avoid any GPU to CPU transfers.
        >>> model = nn.Sequential(fc1, fc2, WithDevice(dropout, 'cuda:1'))
        >>> # xdoctest: +SKIP("Needs RPC framework init")
        >>> model = Pipe(model, chunks=8)
    """

    def __init__(self, module: nn.Module, device: torch.device):
        if False:
            return 10
        super().__init__()
        self._module = module
        self._device = torch.device(device)

    def forward(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return self._module(*args, **kwargs)

    @property
    def module(self):
        if False:
            for i in range(10):
                print('nop')
        return self._module

    @property
    def device(self):
        if False:
            i = 10
            return i + 15
        return self._device

def _assemble_partition(modules: List[nn.Module]):
    if False:
        i = 10
        return i + 15
    modules_list: List[nn.Module] = []
    for module in modules:
        if isinstance(module, nn.Sequential):
            modules_list.extend(module.children())
        else:
            modules_list.append(module)
    return PipeSequential(*modules_list)

def _split_module(modules: nn.Sequential) -> Tuple[List[nn.Sequential], List[torch.device]]:
    if False:
        print('Hello World!')
    partitions = []
    devices = []
    current_partition = []
    current_device = None
    for (name, module) in modules.named_children():
        if isinstance(module, WithDevice):
            device = module.device
            module = module.module
            module.to(device)
        else:
            device = _retrieve_device(module)
        if current_device is not None and (current_device != device or device.type == 'cpu'):
            partitions.append(_assemble_partition(current_partition))
            devices.append(current_device)
            current_partition = []
        current_device = device
        current_partition.append(module)
    if current_device is not None:
        partitions.append(_assemble_partition(current_partition))
        devices.append(current_device)
    partitions = cast(List[nn.Sequential], nn.ModuleList(partitions))
    return (partitions, devices)
MOVING_DENIED = TypeError('denied to move parameters and buffers, because Pipe should manage device placement')

class Pipe(Module):
    """Wraps an arbitrary :class:`nn.Sequential <torch.nn.Sequential>` module
    to train on using synchronous pipeline parallelism. If the module requires
    lots of memory and doesn't fit on a single GPU, pipeline parallelism is a
    useful technique to employ for training.

    The implementation is based on the torchgpipe_ paper.

    .. _torchgpipe: https://arxiv.org/abs/2004.09910

    Pipe combines pipeline parallelism with checkpointing to reduce peak
    memory required to train while minimizing device under-utilization.

    You should place all the modules on the appropriate devices and wrap them
    into an :class:`nn.Sequential <torch.nn.Sequential>` module defining the
    desired order of execution. If a module does not contain any
    parameters/buffers, it is assumed this module should be executed on CPU
    and appropriate input tensors to the module are moved to CPU before
    execution. This behavior can be overridden by the :class:`WithDevice`
    wrapper which can be used to explicitly specify which device a module
    should run on.

    Args:
        module (:class:`nn.Sequential <torch.nn.Sequential>`):
            sequential module to be parallelized using pipelining. Each module
            in the sequence has to have all of its parameters on a single
            device. Each module in the sequence has to either be an nn.Module
            or :class:`nn.Sequential <torch.nn.Sequential>` (to combine multiple
            sequential modules on a single device)
        chunks (int):
            number of micro-batches (default: ``1``)
        checkpoint (str):
            when to enable checkpointing, one of ``'always'``,
            ``'except_last'``, or ``'never'`` (default: ``'except_last'``).
            ``'never'`` disables checkpointing completely, ``'except_last'``
            enables checkpointing for all micro-batches except the last one
            and ``'always'`` enables checkpointing for all micro-batches.
        deferred_batch_norm (bool):
            whether to use deferred ``BatchNorm`` moving statistics (default:
            :data:`False`). If set to :data:`True`, we track statistics across
            multiple micro-batches to update the running statistics per
            mini-batch.

    Raises:
        TypeError:
            the module is not a :class:`nn.Sequential <torch.nn.Sequential>`.
        ValueError:
            invalid arguments

    Example::
        Pipeline of two FC layers across GPUs 0 and 1.

        >>> # Need to initialize RPC framework first.
        >>> # xdoctest: +SKIP
        >>> os.environ['MASTER_ADDR'] = 'localhost'
        >>> os.environ['MASTER_PORT'] = '29500'
        >>> torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
        >>>
        >>> # Build pipe.
        >>> fc1 = nn.Linear(16, 8).cuda(0)
        >>> fc2 = nn.Linear(8, 4).cuda(1)
        >>> model = nn.Sequential(fc1, fc2)
        >>> model = Pipe(model, chunks=8)
        >>> input = torch.rand(16, 16).cuda(0)
        >>> output_rref = model(input)

    .. note::
        You can wrap a :class:`Pipe` model with
        :class:`torch.nn.parallel.DistributedDataParallel` only when the
        checkpoint parameter of :class:`Pipe` is ``'never'``.

    .. note::
        :class:`Pipe` only supports intra-node pipelining currently, but
        will be expanded to support inter-node pipelining in the future.
        The forward function returns an :class:`~torch.distributed.rpc.RRef`
        to allow for inter-node pipelining in the future, where the output
        might be on a remote host. For intra-node pipelining you can use
        :meth:`~torch.distributed.rpc.RRef.local_value` to retrieve the
        output locally.

    .. warning::
        :class:`Pipe` is experimental and subject to change.
    """

    def __init__(self, module: nn.Sequential, chunks: int=1, checkpoint: str='except_last', deferred_batch_norm: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        if not torch.distributed.rpc._is_current_rpc_agent_set():
            raise RuntimeError('Please initialize RPC framework for Pipe using torch.distributed.rpc.init_rpc')
        chunks = int(chunks)
        checkpoint = str(checkpoint)
        if chunks <= 0:
            raise ValueError('number of chunks must be positive integer')
        if checkpoint not in ['always', 'except_last', 'never']:
            raise ValueError("checkpoint is not one of 'always', 'except_last', or 'never'")
        _verify_module(module)
        verify_skippables(module)
        self.chunks = chunks
        self.checkpoint = checkpoint
        if deferred_batch_norm:
            module = DeferredBatchNorm.convert_deferred_batch_norm(module, chunks)
        (self.partitions, self.devices) = _split_module(module)
        _verify_splitting(module, self.partitions, self.devices)
        self._copy_streams: List[List[AbstractStream]] = []
        self._skip_layout = inspect_skip_layout(self.partitions)
        copy_streams = self._ensure_copy_streams()
        checkpoint_stop = {'always': self.chunks, 'except_last': self.chunks - 1, 'never': 0}[self.checkpoint]
        self.pipeline = Pipeline(self.partitions, self.devices, copy_streams, self._skip_layout, checkpoint_stop)

    def __len__(self) -> int:
        if False:
            return 10
        'Counts the length of the underlying sequential module.'
        return sum((len(p) for p in self.partitions))

    def __getitem__(self, index: int) -> nn.Module:
        if False:
            while True:
                i = 10
        'Gets a layer in the underlying sequential module.'
        partitions = self.partitions
        if index < 0:
            partitions = partitions[::-1]
        for partition in partitions:
            try:
                return partition[index]
            except IndexError:
                pass
            shift = len(partition)
            if index < 0:
                index += shift
            else:
                index -= shift
        raise IndexError

    def __iter__(self) -> Iterator[nn.Module]:
        if False:
            while True:
                i = 10
        'Iterates over children of the underlying sequential module.'
        for partition in self.partitions:
            yield from partition

    def cuda(self, device: Optional[Device]=None) -> 'Pipe':
        if False:
            while True:
                i = 10
        raise MOVING_DENIED

    def cpu(self) -> 'Pipe':
        if False:
            i = 10
            return i + 15
        raise MOVING_DENIED

    def to(self, *args: Any, **kwargs: Any) -> 'Pipe':
        if False:
            return 10
        if 'device' in kwargs or 'tensor' in kwargs:
            raise MOVING_DENIED
        if args:
            if isinstance(args[0], (torch.device, int, str)):
                raise MOVING_DENIED
            if torch.is_tensor(args[0]):
                raise MOVING_DENIED
        return super().to(*args, **kwargs)

    def _ensure_copy_streams(self) -> List[List[AbstractStream]]:
        if False:
            i = 10
            return i + 15
        "Ensures that :class:`Pipe` caches CUDA streams for copy.\n\n        It's worth to cache CUDA streams although PyTorch already manages a\n        pool of pre-allocated CUDA streams, because it may reduce GPU memory\n        fragmentation when the number of micro-batches is small.\n\n        "
        if not self._copy_streams:
            for device in self.devices:
                self._copy_streams.append([new_stream(device) for _ in range(self.chunks)])
        return self._copy_streams

    def forward(self, *inputs) -> RRef:
        if False:
            i = 10
            return i + 15
        "\n        Processes a single input mini-batch through the pipe and returns an\n        :class:`~torch.distributed.rpc.RRef` pointing to the output.\n        :class:`Pipe` is a fairly transparent module wrapper. It doesn't\n        modify the input and output signature of the underlying module. But\n        there's type restriction. Input and output have to contain at least one\n        tensor. This restriction is applied at partition boundaries too.\n\n        The sequence of inputs are fed into the first stage of the pipeline as\n        ``*inputs``. As a result the positional args for this function should\n        match the positional args for the first stage of the pipeline. The same\n        condition applies for output of one stage of the pipeline which is the\n        input for the next stage.\n\n        The input tensor is split into multiple micro-batches based on the\n        ``chunks`` parameter used to initialize :class:`Pipe`. The batch size\n        is assumed to be the first dimension of the tensor and if the batch\n        size is less than ``chunks``, the number of micro-batches is equal to\n        the batch size.\n\n        Only tensors are split into multiple micro-batches, non-Tensor inputs\n        are just replicated as-is in each micro-batch. For non-Tensor outputs\n        in the last stage of the pipeline, they are aggregated as a ``List``\n        and returned the user. For example, if you have 2 micro-batches\n        returning the integer 5, the user would receive the consolidated\n        output of `[5, 5]`\n\n        All the input tensors need to be on the same device as the first\n        partition of the pipeline.\n\n        If a tensor is wrapped with the :class:`NoChunk` wrapper, the tensor\n        is not split across micro-batches and is replicated as-is similar to\n        non-tensors.\n\n        Args:\n            inputs: input mini-batch\n\n        Returns:\n            :class:`~torch.distributed.rpc.RRef` to the output of the mini-batch\n\n        Raises:\n            TypeError: input doesn't contain at least one tensor\n\n        "
        first_partition_device = self.devices[0] if len(self.devices) != 0 else torch.device('cpu')
        microbatch.check(first_partition_device, *inputs)
        if not self.devices:
            return RRef(*inputs)
        batches = microbatch.scatter(*inputs, chunks=self.chunks)
        self.pipeline.run(batches)
        output = microbatch.gather(batches)
        return RRef(output)