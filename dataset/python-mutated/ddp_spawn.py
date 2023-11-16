import os
import copy
import multiprocessing
from typing import Any, List, Optional, Callable
import torch
from torch import nn
from torch.multiprocessing.spawn import ProcessContext
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.core.optimizer import _configure_schedulers_automatic_opt
from pytorch_lightning.core.optimizer import _configure_schedulers_manual_opt
from pytorch_lightning.core.optimizer import _set_scheduler_opt_idx, _validate_scheduler_api
from pytorch_lightning.strategies.launchers import _SpawnLauncher
from pytorch_lightning.strategies import DDPSpawnStrategy as _DDPSpawnStrategy
from pytorch_lightning.plugins.environments import LightningEnvironment
from bigdl.nano.utils.common import schedule_processors
from bigdl.nano.pytorch.dispatcher import _get_patch_status
from bigdl.nano.deps.ipex.ipex_api import ipex_optimize
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.utils.common import EnvContext
from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_1_12
import logging
import warnings
try:
    if TORCH_VERSION_LESS_1_12:
        import torch_ccl
    else:
        import oneccl_bindings_for_pytorch
except Exception as _e:
    pass
log = logging.getLogger(__name__)

class _DDPSpawnLauncher(_SpawnLauncher):

    def __init__(self, strategy: 'DDPSpawnStrategy') -> None:
        if False:
            i = 10
            return i + 15
        self._strategy: DDPSpawnStrategy = strategy
        self._start_method = 'spawn'

    def launch(self, function: Callable, *args: Any, trainer: Optional['pl.Trainer']=None, **kwargs: Any) -> Any:
        if False:
            i = 10
            return i + 15
        invalidInputError(self._strategy.cluster_environment is not None, 'strategy.cluster_environment cannot be None')
        os.environ['MASTER_PORT'] = str(self._strategy.cluster_environment.main_port)
        cpu_procs = self._strategy.cpu_for_each_process
        if cpu_procs is None:
            envs = schedule_processors(self._strategy.num_processes)
        else:
            envs = [{'KMP_AFFINITY': f"granularity=fine,proclist=[{','.join([str(i) for i in cpu_procs[i]])}],explicit", 'OMP_NUM_THREADS': str(len(cpu_procs[i]))} for i in range(self._strategy.num_processes)]
        mp = multiprocessing.get_context(self._start_method)
        return_queue = mp.SimpleQueue()
        error_queues = []
        processes = []
        args = (trainer, function, args, kwargs, return_queue)
        patch_status = _get_patch_status()
        for i in range(self._strategy.num_processes):
            with EnvContext(envs[i]):
                log.debug(f"[Process {i}]: using KMP_AFFINITY: {os.environ['KMP_AFFINITY']}")
                log.debug(f"[Process {i}]: using OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
                error_queue = mp.SimpleQueue()
                process = mp.Process(target=self._wrap, args=(self._wrapping_function, i, args, error_queue, patch_status), daemon=False)
                process.start()
                error_queues.append(error_queue)
                processes.append(process)
        context = ProcessContext(processes, error_queues)
        while not context.join():
            pass
        spawn_output = return_queue.get()
        if trainer is None:
            return spawn_output
        self._recover_results_in_main_process(spawn_output, trainer)
        return spawn_output.trainer_results

    @staticmethod
    def _wrap(fn, i, args, error_queue, patch_status):
        if False:
            while True:
                i = 10
        if patch_status['patch_torch']:
            from bigdl.nano.pytorch.dispatcher import patch_torch
            patch_torch(cuda_to_cpu=patch_status['patch_cuda'])
        from torch.multiprocessing.spawn import _wrap
        _wrap(fn, i, args, error_queue)

class DDPSpawnStrategy(_DDPSpawnStrategy):
    """Extending DDPSpawnStrategy to support launch subprocesses with optimized env variables."""
    strategy_name = 'ddp_spawn'

    def __init__(self, num_processes: int=1, cpu_for_each_process: Optional[List[List[int]]]=None, use_ipex=False, dtype=None, auto_lr=False, **kwargs: Any):
        if False:
            for i in range(10):
                print('nop')
        'Create a DDPSpawnStrategy, adding a cpu_for_each_process parameter.'
        device = 'cpu'
        parallel_devices = [torch.device(device) for _ in range(num_processes)]
        cluster_environment = LightningEnvironment()
        if use_ipex and dtype == torch.bfloat16 and ('precision_plugin' not in kwargs):
            from bigdl.nano.pytorch.strategies import IPEXBF16Precision
            super().__init__(parallel_devices=parallel_devices, cluster_environment=cluster_environment, precision_plugin=IPEXBF16Precision(), **kwargs)
        else:
            super().__init__(parallel_devices=parallel_devices, cluster_environment=cluster_environment, **kwargs)
        self.cpu_for_each_process = cpu_for_each_process
        self.is_distributed = True
        self.use_ipex = use_ipex
        self.dtype = dtype
        self.auto_lr = auto_lr

    def _configure_launcher(self):
        if False:
            while True:
                i = 10
        self._launcher = _DDPSpawnLauncher(self)

    def setup(self, trainer: 'pl.Trainer') -> None:
        if False:
            while True:
                i = 10
        'Setup the distributed environment of sub processes, we add ipex optimization here.'
        invalidInputError(self.model is not None, 'You must specify the model.')
        if self.strategy_name == 'ddp_spawn':
            self.model = copy.deepcopy(self.model)
            self.model.trainer = trainer
        super().setup(trainer)
        if trainer.training and self.auto_lr:

            def _unpack_lightning_optimizer(opt):
                if False:
                    print('Hello World!')
                return opt._optimizer if isinstance(opt, LightningOptimizer) else opt
            optimizers = self.optimizers
            optimizers = [_unpack_lightning_optimizer(opt) for opt in optimizers]
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= self.world_size
            lr_scheduler_configs = self.lr_scheduler_configs
            for config in lr_scheduler_configs:
                scheduler = config.scheduler
                if isinstance(scheduler, _LRScheduler):
                    scheduler.base_lrs = [lr * self.world_size for lr in scheduler.base_lrs]
        if self.use_ipex:
            ipex_optimize(self.model, optimizers=self.optimizers, inplace=True, dtype=self.dtype)

    def on_train_start(self):
        if False:
            while True:
                i = 10
        'Setup warmup lr_schedulers after resetting the train dataloaders.'
        if not self.auto_lr:
            return
        if self.lr_scheduler_configs:
            warnings.warn(f'Nano warmup currently only support no scheduler, but got {len(self.lr_scheduler_configs)}. Skip warmup')
        else:
            trainer = self.lightning_module.trainer
            lr_schedulers = []
            warmup_params = {'start_factor': 1.0 / self.world_size, 'end_factor': 1.0, 'warmup_epochs': trainer.max_epochs // 10, 'interval': 'epoch'}
            supported_keys = {'warmup_epochs'}
            if isinstance(self.auto_lr, dict):
                extra_keys = self.auto_lr.keys() - supported_keys
                if extra_keys:
                    warnings.warn(f'Found unsupported keys in the auto_lr dict: {extra_keys}')
                if 'warmup_epochs' not in self.auto_lr:
                    self.auto_lr = True
                    warnings.warn('Not found "warmup_epochs" in the auto_lr dict warmup_epochs is set by default')
                else:
                    invalidInputError(type(self.auto_lr['warmup_epochs']) is int, f""""warmup_epochs" is {type(self.auto_lr['warmup_epochs'])}""", 'expect "warmup_epochs" is a integer')
                    warmup_params['warmup_epochs'] = self.auto_lr['warmup_epochs']
            if type(self.auto_lr) is bool:
                if warmup_params['warmup_epochs'] == 0:
                    train_loader = trainer.train_dataloader
                    max_steps = len(train_loader) * trainer.max_epochs
                    warmup_params['warmup_epochs'] = max_steps // 10
                    warmup_params['interval'] = 'step'
            for (opt_idx, opt) in enumerate(self.optimizers):
                from torch.optim.lr_scheduler import LambdaLR

                def lr_func(epoch):
                    if False:
                        while True:
                            i = 10
                    current_epoch = trainer.current_epoch
                    start_factor = warmup_params['start_factor']
                    end_factor = warmup_params['end_factor']
                    total_iters = warmup_params['warmup_epochs']
                    if current_epoch > 0 and warmup_params['interval'] == 'step' or epoch > total_iters:
                        return 1.0
                    if epoch == 0:
                        return start_factor
                    return (end_factor - start_factor) * epoch / total_iters + start_factor
                scheduler = LambdaLR(optimizer=opt, lr_lambda=[lr_func] * len(opt.param_groups))
                lr_scheduler = {'scheduler': scheduler, 'opt_idx': opt_idx, 'interval': warmup_params['interval']}
                lr_schedulers.append(lr_scheduler)
            lr_scheduler_configs = _configure_schedulers_automatic_opt(lr_schedulers, None) if self.lightning_module.automatic_optimization else _configure_schedulers_manual_opt(lr_schedulers)
            _set_scheduler_opt_idx(self.optimizers, lr_scheduler_configs)
            _validate_scheduler_api(lr_scheduler_configs, self.lightning_module)
            self.lr_scheduler_configs = lr_scheduler_configs

    def _setup_model(self, model: nn.Module) -> DistributedDataParallel:
        if False:
            while True:
                i = 10
        "Wraps the model into a 'DistributedDataParallel' module."
        self._ddp_kwargs['find_unused_parameters'] = True
        return DistributedDataParallel(model, **self._ddp_kwargs)