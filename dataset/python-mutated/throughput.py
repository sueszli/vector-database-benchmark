from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, TypeVar, Union
import torch
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from lightning.fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
if TYPE_CHECKING:
    from lightning.fabric import Fabric
    from lightning.fabric.plugins import Precision
_THROUGHPUT_METRICS = Dict[str, Union[int, float]]

class Throughput:
    """Computes throughput.

    +------------------------+-------------------------------------------------------------------------------------+
    | Key                    | Value                                                                               |
    +========================+=====================================================================================+
    | batches_per_sec        | Rolling average (over ``window_size`` most recent updates) of the number of batches |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | samples_per_sec        | Rolling average (over ``window_size`` most recent updates) of the number of samples |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | items_per_sec          | Rolling average (over ``window_size`` most recent updates) of the number of items   |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | flpps_per_sec          | Rolling average (over ``window_size`` most recent updates) of the number of flops   |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/batches_per_sec | batches_per_sec divided by world size                                               |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/samples_per_sec | samples_per_sec divided by world size                                               |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/items_per_sec   | items_per_sec divided by world size. This may include padding depending on the data |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/flops_per_sec   | flops_per_sec divided by world size.                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/mfu             | device/flops_per_sec divided by world size.                                         |
    +--------------------------+-----------------------------------------------------------------------------------+
    | time                   | Total elapsed time                                                                  |
    +--------------------------+-----------------------------------------------------------------------------------+
    | batches                | Total batches seen                                                                  |
    +--------------------------+-----------------------------------------------------------------------------------+
    | samples                | Total samples seen                                                                  |
    +--------------------------+-----------------------------------------------------------------------------------+
    | lengths                | Total items seen                                                                    |
    +--------------------------+-----------------------------------------------------------------------------------+

    Example::

        throughput = Throughput()
        t0 = time()
        for i in range(1000):
            do_work()
            if torch.cuda.is_available(): torch.cuda.synchronize()  # required or else time() won't be correct
            throughput.update(time=time() - t0, samples=i)
            if i % 10 == 0:
                print(throughput.compute())

    Notes:
        - The implementation assumes that devices FLOPs are all the same as it normalizes by the world size and only
          takes a single ``available_flops`` value.
        - items_per_sec, flops_per_sec and MFU do not account for padding if present. We suggest using
          samples_per_sec or batches_per_sec to measure throughput under this circumstance.

    Args:
        available_flops: Number of theoretical flops available for a single device.
        world_size: Number of devices available across hosts. Global metrics are not included if the world size is 1.
        window_size: Number of batches to use for a rolling average.
        separator: Key separator to use when creating per-device and global metrics.

    """

    def __init__(self, available_flops: Optional[float]=None, world_size: int=1, window_size: int=100, separator: str='/') -> None:
        if False:
            return 10
        self.available_flops = available_flops
        self.separator = separator
        assert world_size > 0
        self.world_size = world_size
        assert window_size > 1
        self._time: _MonotonicWindow[float] = _MonotonicWindow(maxlen=window_size)
        self._batches: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._samples: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._lengths: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._flops: Deque[int] = deque(maxlen=window_size)

    def update(self, *, time: float, batches: int, samples: int, lengths: Optional[int]=None, flops: Optional[int]=None) -> None:
        if False:
            while True:
                i = 10
        'Update throughput metrics.\n\n        Args:\n            time: Total elapsed time in seconds. It should monotonically increase by the iteration time with each\n                call.\n            batches: Total batches seen per device. It should monotonically increase with each call.\n            samples: Total samples seen per device. It should monotonically increase by the batch size with each call.\n            lengths: Total length of the samples seen. It should monotonically increase by the lengths of a batch with\n                each call.\n            flops: Flops elapased per device since last ``update()`` call. You can easily compute this by using\n                :func:`measure_flops` and multiplying it by the number of batches that have been processed.\n                The value might be different in each device if the batch size is not the same.\n\n        '
        self._time.append(time)
        if samples < batches:
            raise ValueError(f'Expected samples ({samples}) to be greater or equal than batches ({batches})')
        self._batches.append(batches)
        self._samples.append(samples)
        if lengths is not None:
            if lengths < samples:
                raise ValueError(f'Expected lengths ({lengths}) to be greater or equal than samples ({samples})')
            self._lengths.append(lengths)
            if len(self._samples) != len(self._lengths):
                raise RuntimeError(f'If lengths are passed ({len(self._lengths)}), there needs to be the same number of samples ({len(self._samples)})')
        if flops is not None:
            self._flops.append(flops * self.world_size)

    def compute(self) -> _THROUGHPUT_METRICS:
        if False:
            for i in range(10):
                print('nop')
        'Compute throughput metrics.'
        metrics = {'time': self._time[-1], 'batches': self._batches[-1], 'samples': self._samples[-1]}
        if self._lengths:
            metrics['lengths'] = self._lengths[-1]
        add_global_metrics = self.world_size > 1
        if len(self._time) == self._time.maxlen:
            elapsed_time = self._time[-1] - self._time[0]
            elapsed_batches = self._batches[-1] - self._batches[0]
            elapsed_samples = self._samples[-1] - self._samples[0]
            dev_samples_per_sec = elapsed_samples / elapsed_time
            dev_batches_per_sec = elapsed_batches / elapsed_time
            metrics.update({f'device{self.separator}batches_per_sec': elapsed_batches / elapsed_time, f'device{self.separator}samples_per_sec': dev_samples_per_sec})
            if add_global_metrics:
                samples_per_sec = dev_batches_per_sec * self.world_size
                metrics.update({'batches_per_sec': samples_per_sec, 'samples_per_sec': dev_samples_per_sec * self.world_size})
            if len(self._lengths) == self._lengths.maxlen:
                elapsed_lengths = self._lengths[-1] - self._lengths[0]
                avg_length = elapsed_lengths / elapsed_batches
                if add_global_metrics:
                    metrics['items_per_sec'] = samples_per_sec * avg_length
                metrics[f'device{self.separator}items_per_sec'] = dev_samples_per_sec * avg_length
        if len(self._flops) == self._flops.maxlen:
            elapsed_flops = sum(self._flops) - self._flops[0]
            elapsed_time = self._time[-1] - self._time[0]
            flops_per_sec = elapsed_flops / elapsed_time
            dev_flops_per_sec = flops_per_sec / self.world_size
            if add_global_metrics:
                metrics['flops_per_sec'] = flops_per_sec
            metrics[f'device{self.separator}flops_per_sec'] = dev_flops_per_sec
            if self.available_flops:
                metrics[f'device{self.separator}mfu'] = dev_flops_per_sec / self.available_flops
        return metrics

    def reset(self) -> None:
        if False:
            i = 10
            return i + 15
        self._time.clear()
        self._batches.clear()
        self._samples.clear()
        self._lengths.clear()
        self._flops.clear()

class ThroughputMonitor(Throughput):
    """Computes throughput.

    This class will automatically keep a count of the number of log calls (``step``). But that can be modified as
    desired. For manual logging, using :class:`Throughput` directly might be desired.

    Example::

        logger = ...
        fabric = Fabric(logger=logger)
        throughput = ThroughputMonitor()
        t0 = time()
        for i in range(1, 100):
            do_work()
            if torch.cuda.is_available(): torch.cuda.synchronize()  # required or else time() won't be correct
            throughput.update(time=time() - t0, batches=i, samples=i)
            if i % 10 == 0:
                throughput.compute_and_log(step=i)

    Args:
        fabric: The Fabric object.
        \\**kwargs: See available parameters in :class:`Throughput`

    """

    def __init__(self, fabric: 'Fabric', **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        fabric._validate_launched()
        dtype = _plugin_to_compute_dtype(fabric.strategy.precision)
        available_flops = get_available_flops(fabric.device, dtype)
        super().__init__(available_flops=available_flops, world_size=fabric.world_size, **kwargs)
        self._fabric = fabric
        self.step = -1
        self.update = rank_zero_only(self.update)
        self.compute = rank_zero_only(self.compute, default={})
        self.compute_and_log = rank_zero_only(self.compute_and_log, default={})
        self.reset = rank_zero_only(self.reset)

    def compute_and_log(self, step: Optional[int]=None, **kwargs: Any) -> _THROUGHPUT_METRICS:
        if False:
            while True:
                i = 10
        'See :meth:`Throughput.compute`\n\n        Args:\n            step: Can be used to override the logging step.\n            \\**kwargs: See available parameters in :meth:`Throughput.compute`\n\n        '
        self.step = self.step + 1 if step is None else step
        metrics = self.compute(**kwargs)
        self._fabric.log_dict(metrics=metrics, step=self.step)
        return metrics

def measure_flops(model: torch.nn.Module, forward_fn: Callable[[], torch.Tensor], loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]=None) -> int:
    if False:
        i = 10
        return i + 15
    'Utility to compute the total number of FLOPs used by a module during training or during inference.\n\n    It\'s recommended to create a meta-device model for this:\n\n    Example::\n\n        with torch.device("meta"):\n            model = MyModel()\n            x = torch.randn(2, 32)\n\n        model_fwd = lambda: model(x)\n        fwd_flops = measure_flops(model, model_fwd)\n\n        model_loss = lambda y: y.sum()\n        fwd_and_bwd_flops = measure_flops(model, model_fwd, model_loss)\n\n    Args:\n        model: The model whose FLOPs should be measured.\n        forward_fn: A function that runs ``forward`` on the model and returns the result.\n        loss_fn: A function that computes the loss given the ``forward_fn`` output. If provided, the loss and `backward`\n            FLOPs will be included in the result.\n\n    '
    if not _TORCH_GREATER_EQUAL_2_1:
        raise ImportError('`measure_flops` requires PyTorch >= 2.1.')
    from torch.utils.flop_counter import FlopCounterMode
    flop_counter = FlopCounterMode(model, display=False)
    with flop_counter:
        if loss_fn is None:
            forward_fn()
        else:
            loss_fn(forward_fn()).backward()
    return flop_counter.get_total_flops()
_CUDA_FLOPS: Dict[str, Dict[Union[str, torch.dtype], float]] = {'h100 nvl': {torch.float64: 67000000000000.0, torch.float32: 133800000000000.0, 'tfloat32': 989400000000000.0, torch.bfloat16: 1978800000000000.0, torch.float16: 1978800000000000.0, torch.int8: 3957800000000000.0}, 'h100 sxm': {torch.float64: 33500000000000.0, torch.float32: 66900000000000.0, 'tfloat32': 494700000000000.0, torch.bfloat16: 989400000000000.0, torch.float16: 989400000000000.0, torch.int8: 1978900000000000.0}, 'h100 pcie': {torch.float64: 25600000000000.0, torch.float32: 51200000000000.0, 'tfloat32': 378000000000000.0, torch.bfloat16: 756000000000000.0, torch.float16: 756000000000000.0, torch.int8: 1513000000000000.0}, 'rtx 4090': {torch.float32: 82600000000000.0, 'tfloat32': 82600000000000.0, torch.bfloat16: 82600000000000.0, torch.float16: 82600000000000.0, torch.int8: 660600000000000.0, 'int4': 1321200000000000.0}, 'rtx 4080': {torch.float32: 48700000000000.0, 'tfloat32': 48700000000000.0, torch.bfloat16: 48700000000000.0, torch.float16: 48700000000000.0, torch.int8: 389900000000000.0, 'int4': 779800000000000.0}, 'l4': {torch.float32: 30300000000000.0, 'tfloat32': 60000000000000.0, torch.bfloat16: 121000000000000.0, torch.float16: 121000000000000.0, torch.int8: 242000000000000.0, 'int4': 484000000000000.0}, 'l40': {torch.float32: 90500000000000.0, 'tfloat32': 90500000000000.0, torch.bfloat16: 181000000000000.0, torch.float16: 181000000000000.0, torch.int8: 362000000000000.0, 'int4': 724000000000000.0}, 'a100': {torch.float64: 9700000000000.0, torch.float32: 19500000000000.0, 'tfloat32': 156000000000000.0, torch.bfloat16: 312000000000000.0, torch.float16: 312000000000000.0, torch.int8: 624000000000000.0}, 'a6000': {torch.float32: 38700000000000.0, 'tfloat32': 77400000000000.0, torch.bfloat16: 38700000000000.0, torch.float16: 38700000000000.0, torch.int8: 309700000000000.0, 'int4': 619300000000000.0}, 'a40': {torch.float32: 37400000000000.0, 'tfloat32': 74800000000000.0, torch.bfloat16: 37400000000000.0, torch.float16: 37400000000000.0, torch.int8: 299300000000000.0, 'int4': 598700000000000.0}, 'a10g': {torch.float32: 31200000000000.0, 'tfloat32': 62500000000000.0, torch.bfloat16: 125000000000000.0, torch.float16: 125000000000000.0, torch.int8: 250000000000000.0, 'int4': 500000000000000.0}, 'rtx 3090 ti': {torch.float32: 40000000000000.0, 'tfloat32': 40000000000000.0, torch.bfloat16: 40000000000000.0, torch.float16: 40000000000000.0, torch.int8: 320000000000000.0, 'int4': 640000000000000.0}, 'rtx 3090': {torch.float32: 35600000000000.0, 'tfloat32': 35600000000000.0, torch.bfloat16: 35600000000000.0, torch.float16: 35600000000000.0, torch.int8: 284000000000000.0, 'int4': 568000000000000.0}, 'rtx 3080 ti': {torch.float32: 34100000000000.0, 'tfloat32': 34100000000000.0, torch.bfloat16: 34100000000000.0, torch.float16: 34100000000000.0, torch.int8: 272800000000000.0, 'int4': 546600000000000.0}, 'rtx 3080': {torch.float32: 29800000000000.0, 'tfloat32': 29800000000000.0, torch.bfloat16: 29800000000000.0, torch.float16: 29800000000000.0, torch.int8: 238000000000000.0, 'int4': 476000000000000.0}, 'rtx 3070': {torch.float32: 20300000000000.0, 'tfloat32': 20300000000000.0, torch.bfloat16: 20300000000000.0, torch.float16: 20300000000000.0, torch.int8: 162600000000000.0, 'int4': 325200000000000.0}, 't4': {torch.float32: 8100000000000.0, torch.float16: 65000000000000.0, torch.int8: 130000000000000.0, 'int4': 260000000000000.0}, 'quadro rtx 5000': {torch.float32: 11200000000000.0, torch.float16: 89200000000000.0}, 'rtx 2080 super': {torch.float32: 11200000000000.0, torch.float16: 22300000000000.0, torch.int8: 178400000000000.0, 'int4': 356800000000000.0}, 'rtx 2080 ti': {torch.float32: 14200000000000.0, torch.float16: 28500000000000.0, torch.int8: 227700000000000.0, 'int4': 455400000000000.0}, 'rtx 2080': {torch.float32: 10600000000000.0, torch.float16: 21200000000000.0, torch.int8: 169600000000000.0, 'int4': 339100000000000.0}, 'rtx 2070 super': {torch.float32: 9100000000000.0, torch.float16: 18100000000000.0, torch.int8: 145000000000000.0, 'int4': 290000000000000.0}, 'titan rtx': {torch.float32: 16300000000000.0, torch.float16: 32600000000000.0, torch.int8: 261000000000000.0, 'int4': 522000000000000.0}, 'v100 sxm': {torch.float64: 7800000000000.0, torch.float32: 15700000000000.0, torch.float16: 125000000000000.0}, 'v100 pcie': {torch.float64: 7000000000000.0, torch.float32: 14000000000000.0, torch.float16: 112000000000000.0}, 'v100s pcie': {torch.float64: 8200000000000.0, torch.float32: 16400000000000.0, torch.float16: 130000000000000.0}}
_TPU_FLOPS = {'v2': 45000000000000.0, 'v3': 123000000000000.0, 'v4': 275000000000000.0, 'v5litepod': 197000000000000.0}

def get_available_flops(device: torch.device, dtype: Union[torch.dtype, str]) -> Optional[int]:
    if False:
        while True:
            i = 10
    'Returns the available theoretical FLOPs.\n\n    This is an optimistic upper limit that could only be achievable if only thick matmuls were run in a benchmark\n    environment.\n\n    '
    if device.type == 'cuda':
        device_name = torch.cuda.get_device_name(device)
        chip = device_name.lower()
        if 'h100' in chip:
            if 'hbm3' in chip:
                chip = 'h100 sxm'
            elif 'nvl' in chip:
                chip = 'h100 nvl'
            elif 'pcie' in chip or 'hbm2e' in chip:
                chip = 'h100 pcie'
        elif 'l4' in chip:
            chip = 'l40' if 'tesla' in chip else 'l4'
        elif 'geforce rtx' in chip:
            number = chip.split(' ')[3]
            extra = ''
            if 'super' in chip:
                extra = ' super'
            elif 'ti' in chip:
                extra = ' ti'
            chip = f'rtx {number}{extra}'
        elif 'a6000' in chip:
            chip = 'a6000'
        elif 'a100' in chip:
            chip = 'a100'
        elif 'a40' in chip:
            chip = 'a40'
        elif 'a10g' in chip:
            chip = 'a10g'
        elif 't4' in chip:
            chip = 't4'
        elif 'quadro rtx 5000' in chip:
            chip = 'quadro rtx 5000'
        elif 'titan rtx' in chip:
            chip = 'titan rtx'
        elif 'v100-sxm' in chip:
            chip = 'v100 sxm'
        elif 'v100-pcie' in chip:
            chip = 'v100 pcie'
        elif 'v100s-pcie' in chip:
            chip = 'v100s pcie'
        else:
            rank_zero_warn(f'FLOPs not found for {device_name!r}')
            return None
        if chip not in _CUDA_FLOPS:
            rank_zero_warn(f'FLOPs not found for {device_name!r}, chip is {chip!r}')
            return None
        dtype_to_flops = _CUDA_FLOPS[chip]
        if dtype is torch.float32:
            from lightning.fabric.accelerators.cuda import _is_ampere_or_later
            if _is_ampere_or_later() and torch.get_float32_matmul_precision() != 'highest':
                dtype = 'tfloat32'
        if dtype not in dtype_to_flops:
            rank_zero_warn(f'{device_name!r} does not support {dtype}')
            return None
        return int(dtype_to_flops[dtype])
    if device.type == 'xla':
        from lightning.fabric.accelerators.xla import _XLA_GREATER_EQUAL_2_1
        if _XLA_GREATER_EQUAL_2_1:
            from torch_xla._internal import tpu
        else:
            from torch_xla.experimental import tpu
        device_name = tpu.get_tpu_env()['TYPE']
        chip = device_name.lower()
        assert isinstance(device_name, str)
        if chip not in _TPU_FLOPS:
            rank_zero_warn(f'FLOPs not found for TPU {device_name!r} with {dtype}')
            return None
        return int(_TPU_FLOPS[chip])

def _plugin_to_compute_dtype(plugin: 'Precision') -> torch.dtype:
    if False:
        for i in range(10):
            print('nop')
    from lightning.fabric.plugins import BitsandbytesPrecision, DeepSpeedPrecision, DoublePrecision, FSDPPrecision, HalfPrecision, MixedPrecision, Precision, TransformerEnginePrecision, XLAPrecision
    if not isinstance(plugin, Precision):
        raise RuntimeError(f'Expected a precision plugin, got {plugin}')
    if isinstance(plugin, BitsandbytesPrecision):
        return plugin.dtype
    if isinstance(plugin, (HalfPrecision, MixedPrecision)):
        return plugin._desired_input_dtype
    if isinstance(plugin, DoublePrecision):
        return torch.double
    if isinstance(plugin, (XLAPrecision, DeepSpeedPrecision)):
        return plugin._desired_dtype
    if isinstance(plugin, TransformerEnginePrecision):
        return torch.int8
    if isinstance(plugin, FSDPPrecision):
        return plugin.mixed_precision_config.reduce_dtype or torch.float32
    if isinstance(plugin, Precision):
        return torch.float32
    raise NotImplementedError(plugin)
T = TypeVar('T', bound=float)

class _MonotonicWindow(List[T]):
    """Custom fixed size list that only supports right-append and ensures that all values increase monotonically."""

    def __init__(self, maxlen: int) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.maxlen = maxlen

    @property
    def last(self) -> Optional[T]:
        if False:
            return 10
        if len(self) > 0:
            return self[-1]
        return None

    def append(self, x: T) -> None:
        if False:
            i = 10
            return i + 15
        last = self.last
        if last is not None and last >= x:
            raise ValueError(f'Expected the value to increase, last: {last}, current: {x}')
        list.append(self, x)
        if len(self) > self.maxlen:
            del self[0]

    def __setitem__(self, key: Any, value: Any) -> None:
        if False:
            return 10
        raise NotImplementedError('__setitem__ is not supported')