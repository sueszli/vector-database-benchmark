import os
import warnings
from typing import Callable, Dict, List, Union, Any, Optional
from collections import defaultdict
import ray
from ray.util.ml_utils.util import find_free_port
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.parallel.distributed import DistributedDataParallel
import pytorch_lightning as pl
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.strategies.launchers import _SpawnLauncher
from pytorch_lightning.strategies import DDPSpawnStrategy
from pytorch_lightning.core.optimizer import _configure_schedulers_automatic_opt
from pytorch_lightning.core.optimizer import _configure_schedulers_manual_opt
from pytorch_lightning.core.optimizer import _set_scheduler_opt_idx, _validate_scheduler_api
from pytorch_lightning.core.optimizer import LightningOptimizer
from .ray_envbase import RayEnvironment
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_1_12
from bigdl.nano.deps.ipex.ipex_api import ipex_optimize
from bigdl.nano.pytorch.dispatcher import _get_patch_status
try:
    if TORCH_VERSION_LESS_1_12:
        import torch_ccl
    else:
        import oneccl_bindings_for_pytorch
except Exception as _e:
    pass

@ray.remote
class RayExecutor:
    """A class to execute any arbitrary function remotely."""

    def set_env_var(self, key: str, value: str):
        if False:
            return 10
        'Set an environment variable with the provided values.'
        if value is not None:
            value = str(value)
            os.environ[key] = value

    def set_env_vars(self, keys: List[str], values: List[str]):
        if False:
            while True:
                i = 10
        'Sets multiple env vars with the provided values.'
        invalidInputError(len(keys) == len(values), "keys length doesn't mathcc values length")
        for (key, value) in zip(keys, values):
            self.set_env_var(key, value)

    def get_env_vars(self, key: str):
        if False:
            for i in range(10):
                print('nop')
        'Return the specified environment variable.'
        return os.environ[key]

    def get_node_ip(self):
        if False:
            return 10
        'Returns the IP address of the node that this Ray actor is on.'
        return ray.util.get_node_ip_address()

    def execute(self, fn: Callable, *args, **kwargs):
        if False:
            print('Hello World!')
        'Execute the provided function and return the result.'
        return fn(*args, **kwargs)

class _RayLauncher(_SpawnLauncher):

    def __init__(self, strategy: 'RayStrategy') -> None:
        if False:
            while True:
                i = 10
        self._strategy: RayStrategy = strategy

    @property
    def is_interactive_compatible(self) -> bool:
        if False:
            print('Hello World!')
        return False

    def launch(self, function: Callable, *args: Any, trainer: Optional['pl.Trainer']=None, **kwargs: Any) -> Any:
        if False:
            return 10
        strategy = self._strategy
        strategy._setup_env_vars()
        strategy.global_to_local = strategy.get_local_ranks()
        torch_backend = os.getenv('PL_TORCH_DISTRIBUTED_BACKEND')
        if torch_backend is not None:
            strategy._process_group_backend = torch_backend
        patch_status = _get_patch_status()
        futures = [strategy.workers[i].execute.remote(self._wrapping_function, (i, trainer, strategy, function, args, kwargs, patch_status)) for i in range(strategy.num_workers)]
        results = ray.get(futures)
        ray.shutdown()
        if trainer is None:
            return results[0]
        self._recover_results_in_main_process(results[0], trainer)
        return results[0].trainer_results

    @staticmethod
    def _wrapping_function(args_pack: tuple) -> Any:
        if False:
            i = 10
            return i + 15
        (global_rank, trainer, strategy, function, args, kwargs, patch_status) = args_pack
        invalidInputError(isinstance(strategy, RayStrategy), 'expect ray strategy here')
        invalidInputError(isinstance(strategy.cluster_environment, RayEnvironment), 'expect ray environment here')
        if patch_status['patch_torch']:
            from bigdl.nano.pytorch import patch_torch
            patch_torch(cuda_to_cpu=patch_status['patch_cuda'])
        strategy.cluster_environment.set_global_rank(global_rank)
        strategy.cluster_environment.set_remote_execution(True)
        strategy._worker_setup(global_rank)
        results = function(*args, **kwargs)
        if trainer is not None:
            results = strategy._launcher._collect_rank_zero_results(trainer, results)
        if strategy.global_rank == 0:
            return move_data_to_device(results, 'cpu')

class RayStrategy(DDPSpawnStrategy):
    """A DDP Strategy which uses ray as backend."""
    strategy_name = 'ray'

    def __init__(self, num_processes: int=1, cpu_for_each_process: Optional[List[List[int]]]=None, num_cpus_per_worker: int=1, use_gpu: bool=False, use_ipex: bool=False, dtype=None, init_hook: Callable=None, auto_lr: Union[bool, dict]=True, **ddp_kwargs: Any):
        if False:
            while True:
                i = 10
        'Create a RayStrategy.'
        os.environ.pop('KMP_AFFINITY', None)
        os.environ.pop('OMP_NUM_THREADS', None)
        if not ray.is_initialized():
            print(ray.init())
        super().__init__(parallel_devices=[], cluster_environment=RayEnvironment(world_size=num_processes), **ddp_kwargs)
        self.num_workers = num_processes
        self.num_cpus_per_worker = num_cpus_per_worker
        self.use_gpu = use_gpu
        self.use_ipex = use_ipex
        self.dtype = dtype
        self.auto_lr = auto_lr
        invalidInputError(not self.use_gpu or not self.use_ipex, 'You can not specify gpu and ipex at the same time.')
        self.workers = self._create_worker()
        self.init_hook = init_hook
        self._local_rank = 0
        if self.init_hook:
            ray.get([w.execute.remote(self.init_hook) for w in self.workers])

    def _configure_launcher(self):
        if False:
            print('Hello World!')
        self._launcher = _RayLauncher(self)

    def _create_worker(self):
        if False:
            print('Hello World!')
        'Creates Ray actor.'
        from bigdl.nano.utils.common import schedule_processors
        envs = schedule_processors(self.num_workers)
        workers = []
        for i in range(self.num_workers):
            worker = RayExecutor.options(num_cpus=self.num_cpus_per_worker, num_gpus=int(self.use_gpu)).remote()
            ray.get(worker.set_env_var.remote('KMP_AFFINITY', envs[i]['KMP_AFFINITY']))
            ray.get(worker.set_env_var.remote('OMP_NUM_THREADS', envs[i]['OMP_NUM_THREADS']))
            ray.get(worker.set_env_var.remote('PYTHONPATH', envs[i].get('PYTHONPATH', '')))
            workers.append(worker)
        return workers

    def _setup_env_vars(self):
        if False:
            print('Hello World!')
        os.environ['MASTER_ADDR'] = ray.get(self.workers[0].get_node_ip.remote())
        os.environ['MASTER_PORT'] = str(ray.get(self.workers[0].execute.remote(find_free_port)))
        keys = ['PL_GLOBAL_SEED', 'PL_TORCH_DISTRIBUTED_BACKEND', 'MASTER_ADDR', 'MASTER_PORT']
        values = [os.getenv(k) for k in keys]
        ray.get([w.set_env_vars.remote(keys, values) for w in self.workers])

    def get_local_ranks(self):
        if False:
            for i in range(10):
                print('nop')
        'Creates a mapping of global ranks to local ranks.'
        node_ips = ray.get([w.get_node_ip.remote() for w in self.workers])
        rank_counter_dict = defaultdict(int)
        global_to_local = [None] * self.num_workers
        for global_rank in range(self.num_workers):
            ip = node_ips[global_rank]
            global_to_local[global_rank] = rank_counter_dict[ip]
            rank_counter_dict[ip] += 1
        return global_to_local

    def set_world_ranks(self, process_idx: int=0):
        if False:
            i = 10
            return i + 15
        'Set the appropriate rank attribues for the trainer.'
        invalidInputError(self.cluster_environment is not None and isinstance(self.cluster_environment, RayEnvironment), 'expect ray environment here')
        if self.cluster_environment.is_remote():
            self._local_rank = self.global_to_local[self.global_rank]
            self.cluster_environment.set_global_rank(self.global_rank)
            self.cluster_environment.set_world_size(self.num_workers)
            rank_zero_only.rank = self.cluster_environment.global_rank()

    def setup(self, trainer: 'pl.Trainer') -> None:
        if False:
            while True:
                i = 10
        'Setup the distributed environment of ray executor, we add ipex optimization here.'
        self.accelerator.setup(trainer)
        trainer_fn = trainer.state.fn
        if trainer_fn == TrainerFn.FITTING:
            if self._layer_sync:
                self.model = self._layer_sync.apply(self.model)
        self.setup_precision_plugin()
        if trainer_fn == TrainerFn.FITTING:
            self.configure_ddp()
        else:
            self.model.eval()
        if trainer.training and self.auto_lr:

            def _unpack_lightning_optimizer(opt):
                if False:
                    return 10
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
            ipex_optimize(self.model, optimizers=optimizer, inplace=True, dtype=self.dtype)
        self.model_to_device()

    @property
    def root_device(self):
        if False:
            print('Hello World!')
        'Return the root device.'
        return torch.device('cpu')

    def _setup_model(self, model: torch.nn.Module) -> DistributedDataParallel:
        if False:
            for i in range(10):
                print('nop')
        "Wraps the model into a 'DistributedDataParallel' module."
        self._ddp_kwargs['find_unused_parameters'] = True
        return DistributedDataParallel(model, **self._ddp_kwargs)

    @property
    def distributed_sampler_kwargs(self):
        if False:
            i = 10
            return i + 15
        'Returns the args to use for torch.data.DistributedSampler.'
        distributed_sampler_kwargs = dict(num_replicas=self.num_workers, rank=self.global_rank)
        return distributed_sampler_kwargs

    def on_train_start(self) -> None:
        if False:
            i = 10
            return i + 15
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
                        print('Hello World!')
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