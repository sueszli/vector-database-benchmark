import importlib.util
import json
import os
import warnings
from dataclasses import dataclass, field
import torch
from ..training_args import TrainingArguments
from ..utils import cached_property, is_sagemaker_dp_enabled, logging
logger = logging.get_logger(__name__)

def is_sagemaker_model_parallel_available():
    if False:
        i = 10
        return i + 15
    smp_options = os.getenv('SM_HP_MP_PARAMETERS', '{}')
    try:
        smp_options = json.loads(smp_options)
        if 'partitions' not in smp_options:
            return False
    except json.JSONDecodeError:
        return False
    mpi_options = os.getenv('SM_FRAMEWORK_PARAMS', '{}')
    try:
        mpi_options = json.loads(mpi_options)
        if not mpi_options.get('sagemaker_mpi_enabled', False):
            return False
    except json.JSONDecodeError:
        return False
    return importlib.util.find_spec('smdistributed') is not None
if is_sagemaker_model_parallel_available():
    import smdistributed.modelparallel.torch as smp
    smp.init()

@dataclass
class SageMakerTrainingArguments(TrainingArguments):
    mp_parameters: str = field(default='', metadata={'help': 'Used by the SageMaker launcher to send mp-specific args. Ignored in SageMakerTrainer'})

    def __post_init__(self):
        if False:
            while True:
                i = 10
        super().__post_init__()
        warnings.warn('`SageMakerTrainingArguments` is deprecated and will be removed in v5 of Transformers. You can use `TrainingArguments` instead.', FutureWarning)

    @cached_property
    def _setup_devices(self) -> 'torch.device':
        if False:
            print('Hello World!')
        logger.info('PyTorch: setting up devices')
        if torch.distributed.is_available() and torch.distributed.is_initialized() and (self.local_rank == -1):
            logger.warning('torch.distributed process group is initialized, but local_rank == -1. In order to use Torch DDP, launch your script with `python -m torch.distributed.launch')
        if self.no_cuda:
            device = torch.device('cpu')
            self._n_gpu = 0
        elif is_sagemaker_model_parallel_available():
            local_rank = smp.local_rank()
            device = torch.device('cuda', local_rank)
            self._n_gpu = 1
        elif is_sagemaker_dp_enabled():
            import smdistributed.dataparallel.torch.torch_smddp
            torch.distributed.init_process_group(backend='smddp', timeout=self.ddp_timeout_delta)
            self.local_rank = int(os.getenv('SMDATAPARALLEL_LOCAL_RANK'))
            device = torch.device('cuda', self.local_rank)
            self._n_gpu = 1
        elif self.local_rank == -1:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self._n_gpu = torch.cuda.device_count()
        else:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend='nccl', timeout=self.ddp_timeout_delta)
            device = torch.device('cuda', self.local_rank)
            self._n_gpu = 1
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        return device

    @property
    def world_size(self):
        if False:
            return 10
        if is_sagemaker_model_parallel_available():
            return smp.dp_size()
        return super().world_size

    @property
    def place_model_on_device(self):
        if False:
            i = 10
            return i + 15
        return not is_sagemaker_model_parallel_available()

    @property
    def _no_sync_in_gradient_accumulation(self):
        if False:
            while True:
                i = 10
        return False