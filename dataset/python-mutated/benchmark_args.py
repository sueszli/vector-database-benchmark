from dataclasses import dataclass, field
from typing import Tuple
from ..utils import cached_property, is_torch_available, is_torch_tpu_available, logging, requires_backends
from .benchmark_args_utils import BenchmarkArguments
if is_torch_available():
    import torch
if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
logger = logging.get_logger(__name__)

@dataclass
class PyTorchBenchmarkArguments(BenchmarkArguments):
    deprecated_args = ['no_inference', 'no_cuda', 'no_tpu', 'no_speed', 'no_memory', 'no_env_print', 'no_multi_process']

    def __init__(self, **kwargs):
        if False:
            return 10
        '\n        This __init__ is there for legacy code. When removing deprecated args completely, the class can simply be\n        deleted\n        '
        for deprecated_arg in self.deprecated_args:
            if deprecated_arg in kwargs:
                positive_arg = deprecated_arg[3:]
                setattr(self, positive_arg, not kwargs.pop(deprecated_arg))
                logger.warning(f'{deprecated_arg} is depreciated. Please use --no_{positive_arg} or {positive_arg}={kwargs[positive_arg]}')
        self.torchscript = kwargs.pop('torchscript', self.torchscript)
        self.torch_xla_tpu_print_metrics = kwargs.pop('torch_xla_tpu_print_metrics', self.torch_xla_tpu_print_metrics)
        self.fp16_opt_level = kwargs.pop('fp16_opt_level', self.fp16_opt_level)
        super().__init__(**kwargs)
    torchscript: bool = field(default=False, metadata={'help': 'Trace the models using torchscript'})
    torch_xla_tpu_print_metrics: bool = field(default=False, metadata={'help': 'Print Xla/PyTorch tpu metrics'})
    fp16_opt_level: str = field(default='O1', metadata={'help': "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html"})

    @cached_property
    def _setup_devices(self) -> Tuple['torch.device', int]:
        if False:
            print('Hello World!')
        requires_backends(self, ['torch'])
        logger.info('PyTorch: setting up devices')
        if not self.cuda:
            device = torch.device('cpu')
            n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            n_gpu = 0
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            n_gpu = torch.cuda.device_count()
        return (device, n_gpu)

    @property
    def is_tpu(self):
        if False:
            while True:
                i = 10
        return is_torch_tpu_available() and self.tpu

    @property
    def device_idx(self) -> int:
        if False:
            print('Hello World!')
        requires_backends(self, ['torch'])
        return torch.cuda.current_device()

    @property
    def device(self) -> 'torch.device':
        if False:
            print('Hello World!')
        requires_backends(self, ['torch'])
        return self._setup_devices[0]

    @property
    def n_gpu(self):
        if False:
            return 10
        requires_backends(self, ['torch'])
        return self._setup_devices[1]

    @property
    def is_gpu(self):
        if False:
            i = 10
            return i + 15
        return self.n_gpu > 0