from filelock import FileLock
import logging
import os
import copy
import tempfile
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from bigdl.orca.learn.metrics import Metric
from bigdl.orca import OrcaContext
from bigdl.orca.learn.pytorch import utils
from bigdl.orca.learn.pytorch.utils import AverageMeterCollection, NUM_SAMPLES, get_batchsize, index_concatenate, split_predict_cols
from bigdl.orca.learn.pytorch.core import BaseRunner
from bigdl.dllib.utils.log4Error import invalidInputError
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

class DistBackend:

    def get_world_size(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def all_reduce(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        pass

class HorovodDistBackend(DistBackend):

    def get_world_size(self):
        if False:
            while True:
                i = 10
        import horovod.torch as hvd
        return hvd.size()

    def all_reduce(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        import horovod.torch as hvd
        return hvd.allreduce(*args, **kwargs)

class TorchDistBackend(DistBackend):

    def get_world_size(self):
        if False:
            while True:
                i = 10
        import torch.distributed as dist
        return dist.get_world_size()

    def all_reduce(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        import torch.distributed as dist
        return dist.all_reduce(*args, **kwargs)

    def is_initialized(self):
        if False:
            print('Hello World!')
        import torch.distributed as dist
        return dist.is_initialized()

    def all_reduce_min(self, tensor, *args, **kwargs):
        if False:
            while True:
                i = 10
        import torch.distributed as dist
        all_reduce_min_kwargs = dict(op=dist.ReduceOp.MIN)
        all_reduce_min_kwargs.update(kwargs)
        return dist.all_reduce(tensor, *args, **all_reduce_min_kwargs)

class TorchRunner(BaseRunner):
    """Manages a PyTorch model for training."""

    def __init__(self, model_creator, optimizer_creator, loss_creator=None, metrics=None, scheduler_creator=None, config=None, sync_stats=True, log_level=logging.INFO):
        if False:
            return 10
        logging.basicConfig(level=log_level, format='[%(asctime)s] %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)
        self.model_creator = model_creator
        self.optimizer_creator = optimizer_creator
        self.loss_creator = loss_creator
        self.scheduler_creator = scheduler_creator
        self.config = {} if config is None else config
        self.timers = utils.TimerCollection()
        self.epochs = 0
        self.global_step = 0
        self.models = None
        self.optimizers = None
        self.metrics = metrics
        self.criterion = None
        self.schedulers = None
        self.train_loader = None
        self.validation_loader = None
        self.sync_stats = sync_stats
        self.epoch_stats = None
        self._mode = 'val'
        self._pocket = dict()
        self.stop = False

    def _create_loss(self):
        if False:
            return 10
        if not self.loss_creator:
            return
        self.logger.debug('Creating loss.')
        if isinstance(self.loss_creator, torch.nn.modules.loss._Loss):
            self.criterion = self.loss_creator
        else:
            import types
            invalidInputError(isinstance(self.loss_creator, types.FunctionType), 'Must provide a torch loss instance or a loss_creator function')
            self.criterion = self.loss_creator(self.config)

    def _create_schedulers_if_available(self):
        if False:
            while True:
                i = 10
        if not self.scheduler_creator:
            return
        self.schedulers = self.scheduler_creator(self.given_optimizers, self.config)
        if not isinstance(self.schedulers, Iterable):
            self.schedulers = [self.schedulers]

    def setup_components(self):
        if False:
            while True:
                i = 10
        'Runs the creator functions without any distributed coordination.'
        self.logger.debug('Creating model')
        if self.model_creator:
            self.models = self.model_creator(self.config)
            if isinstance(self.models, nn.Sequential) or not isinstance(self.models, Iterable):
                self.models = [self.models]
            invalidInputError(all((isinstance(model, nn.Module) for model in self.models)), 'All models must be PyTorch models: {}.'.format(self.models))
            if self.optimizer_creator:
                self.logger.debug('Creating optimizer.')
                self.optimizers = self.optimizer_creator(self.given_models, self.config)
                if self.optimizers and (not isinstance(self.optimizers, Iterable)):
                    self.optimizers = [self.optimizers]
        self._create_schedulers_if_available()
        self._create_loss()

    def setup_ddp_components(self):
        if False:
            while True:
                i = 10
        from torch.nn.parallel import DistributedDataParallel
        self.training_models = [DistributedDataParallel(model) for model in self.models]
        self.setup_operator(self.training_models)

    def setup_operator(self, training_models):
        if False:
            return 10
        'Create the training operator.'
        if self.backend == 'horovod':
            self.dist_backend = HorovodDistBackend()
        else:
            self.dist_backend = TorchDistBackend()

    def train_epochs(self, data_creator, epochs=1, batch_size=32, profile=False, wrap_dataloader=None, callbacks=None, validation_data_creator=None):
        if False:
            while True:
                i = 10
        config = copy.copy(self.config)
        if OrcaContext.serialize_data_creator:
            with FileLock(os.path.join(tempfile.gettempdir(), '.orcadata.lock')):
                self.train_loader = data_creator(config, batch_size)
        else:
            self.train_loader = data_creator(config, batch_size)
        if wrap_dataloader is None:
            if TorchRunner.should_wrap_dataloader(self.train_loader):
                self.train_loader = self.with_sampler(self.train_loader)
        elif wrap_dataloader is True:
            self.train_loader = self.with_sampler(self.train_loader)
        if validation_data_creator:
            if OrcaContext.serialize_data_creator:
                with FileLock(os.path.join(tempfile.gettempdir(), '.orca_val_data.lock')):
                    val_loader = validation_data_creator(config, batch_size)
            else:
                val_loader = validation_data_creator(config, batch_size)
            wrapped = False
            if wrap_dataloader is None:
                if TorchRunner.should_wrap_dataloader(val_loader):
                    val_loader = self.with_sampler(val_loader)
                    wrapped = True
            elif wrap_dataloader is True:
                val_loader = self.with_sampler(val_loader)
                wrapped = True
            if not wrapped:
                validation_tensor = torch.tensor(len(val_loader))
                invalidInputError(self.backend != 'horovod', 'Sanity check failed!')
                self.dist_backend.all_reduce_min(validation_tensor)
                val_steps = validation_tensor.item()
            else:
                val_steps = None
        else:
            val_loader = None
            val_steps = None
        self.val_loader = val_loader
        self.num_epochs = epochs
        self.call_hook(callbacks=callbacks, fn_name='before_run')
        stats_list = list()
        for i in range(self.num_epochs):
            del self.epoch_stats
            self.call_hook(callbacks=callbacks, fn_name='before_train_epoch')
            stats = self.train_epoch(self.train_loader, profile=profile, callbacks=callbacks, val_loader=val_loader, val_steps=val_steps)
            self.epoch_stats = stats
            self.call_hook(callbacks=callbacks, fn_name='after_train_epoch')
            if self.rank == 0:
                if self.sync_stats:
                    self.logger.info(f'Finished training epoch {i + 1}, ' + f'stats averaged over workers: {stats}')
                else:
                    self.logger.info(f'Finished training epoch {i + 1}, ' + f'stats on rank 0: {stats}')
            stats_list.append(stats)
        self.call_hook(callbacks=callbacks, fn_name='after_run')
        return stats_list

    def train_epoch(self, data_loader, profile=False, callbacks=None, val_loader=None, val_steps=None):
        if False:
            i = 10
            return i + 15
        'Runs a training epoch and updates the model parameters.'
        if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
            data_loader.sampler.set_epoch(self.epochs)
        self.logger.debug('Begin Training Step {}'.format(self.epochs + 1))
        if not self.models:
            invalidInputError(False, 'You must provide a model for train and evaluate.')
        if not self.optimizers:
            invalidInputError(False, 'You must provide the optimizer for train.')
        self._toggle_profiling(profile=profile)
        with self.timers.record('train_epoch'):
            data_loader = iter(data_loader)
            train_stats = self._train_epoch(data_loader, callbacks)
        if val_loader:
            with self.timers.record('validation'):
                validation_results = self._validate(val_loader, metrics=self.metrics, num_steps=val_steps, callbacks=callbacks)
                validation_stats = {}
                for (name, value) in validation_results.items():
                    if not name.startswith('val_'):
                        name = 'val_' + name.lower()
                    validation_stats[name] = value
        else:
            validation_stats = {}
        self.epochs += 1
        stats = dict(epoch=self.epochs, **train_stats, **validation_stats)
        if profile:
            stats.update(profile=self.timers.stats())
        return stats

    def _train_epoch(self, iterator, callbacks=None):
        if False:
            return 10
        'Runs one standard training pass over the training dataloader.\n\n        By default, this method will iterate over the given iterator and\n        call ``self.train_batch`` over each batch.\n\n        You do not need to call ``train_batch`` in this method if you plan\n        to implement a custom optimization/training routine here.\n\n        You may find ``ray.util.sgd.utils.AverageMeterCollection`` useful\n        when overriding this method. See example below:\n\n        .. code-block:: python\n\n            def train_epoch(self, ...):\n                meter_collection = AverageMeterCollection()\n                self.model.train()\n                for batch in iterator:\n                    # do some processing\n                    metrics = {"metric_1": 1, "metric_2": 3} # dict of metrics\n\n                    # This keeps track of all metrics across multiple batches\n                    meter_collection.update(metrics, n=len(batch))\n\n                # Returns stats of the meters.\n                stats = meter_collection.summary()\n                return stats\n\n\n        Args:\n            iterator (iter): Iterator over the training data for the entire\n                epoch. This iterator is expected to be entirely consumed.\n\n        Returns:\n            A dict of metrics from training.\n        '
        self._mode = 'train'
        metric_meters = AverageMeterCollection()
        self.model.train()
        from torch.nn.parallel import DistributedDataParallel as DDP
        if isinstance(self.model, DDP):
            with self.model.join():
                self._train_loop(iterator, metric_meters, callbacks)
        else:
            self._train_loop(iterator, metric_meters, callbacks)
        self.call_hook(callbacks=callbacks, fn_name='on_lr_adjust')
        return metric_meters.summary(sync_stats=self.sync_stats, dist_backend=self.dist_backend)

    def _train_loop(self, iterator, metric_meters, callbacks):
        if False:
            return 10
        for (batch_idx, batch) in enumerate(iterator):
            self.batch_idx = batch_idx
            self._train_batch(batch, callbacks=callbacks)
            metric_meters.update(self.metrics_stats, n=self.metrics_stats.pop(NUM_SAMPLES, 1))
            del self.batch_idx
            del self.metrics_stats
            if self.stop:
                break

    def _train_batch(self, batch, callbacks=None):
        if False:
            i = 10
            return i + 15
        'Computes loss and updates the model over one batch.\n\n        This method is responsible for computing the loss and gradient and\n        updating the model.\n\n        By default, this method implementation assumes that batches\n        are in (\\*features, labels) format. So we also support multiple inputs\n        model. If using amp/fp16 training, it will also scale the loss\n        automatically.\n\n        You can provide custom loss metrics and training operations if you\n        override this method. If overriding this method, you can access model,\n        optimizer, criterion via ``self.model``, ``self.optimizer``,\n        and ``self.criterion``.\n\n        You do not need to override this method if you plan to\n        override ``train_epoch``.\n\n        Args:\n            batch: One item of the validation iterator.\n\n        Returns:\n            A dictionary of metrics.\n                By default, this dictionary contains "loss" and "num_samples".\n                "num_samples" corresponds to number of datapoints in the batch.\n                However, you can provide any number of other values.\n                Consider returning "num_samples" in the metrics because\n                by default, ``train_epoch`` uses "num_samples" to\n                calculate averages.\n\n        '
        self.batch = batch
        self.call_hook(callbacks=callbacks, fn_name='before_train_iter')
        with self.timers.record('fwd'):
            self.call_hook(callbacks=callbacks, fn_name='on_train_forward')
        with self.timers.record('bwd'):
            self.call_hook(callbacks=callbacks, fn_name='on_iter_backward')
        loss_item = self.loss.item()
        self.metrics_stats = {'train_loss': loss_item, NUM_SAMPLES: get_batchsize(batch)}
        self.global_step += 1
        self.call_hook(callbacks=callbacks, fn_name='after_train_iter')
        if hasattr(self, 'batch'):
            del self.batch
        if hasattr(self, 'output'):
            del self.output
        if hasattr(self, 'loss'):
            del self.loss

    def validate(self, data_creator, batch_size=32, num_steps=None, profile=False, wrap_dataloader=None, callbacks=None):
        if False:
            print('Hello World!')
        'Evaluates the model on the validation data set.'
        if not self.models:
            invalidInputError(False, 'You must provide a model for train and evaluate.')
        config = copy.copy(self.config)
        self._toggle_profiling(profile=profile)
        if OrcaContext.serialize_data_creator:
            with FileLock(os.path.join(tempfile.gettempdir(), '.orcadata.lock')):
                loader = data_creator(config, batch_size)
        else:
            loader = data_creator(config, batch_size)
        if wrap_dataloader is None:
            if TorchRunner.should_wrap_dataloader(loader):
                loader = self.with_sampler(loader)
        elif wrap_dataloader is True:
            loader = self.with_sampler(loader)
        self.val_loader = loader
        loader = iter(loader)
        with self.timers.record('validation'):
            self.num_steps = num_steps
            validation_stats = self._validate(loader, metrics=self.metrics, num_steps=num_steps, callbacks=callbacks)
            del self.num_steps
        if profile:
            validation_stats.update(profile=self.timers.stats())
        return validation_stats

    def _validate(self, val_iterator, metrics, num_steps=None, callbacks=None):
        if False:
            return 10
        'Runs one standard validation pass over the val_iterator.\n\n        This will call ``model.eval()`` and ``torch.no_grad`` when iterating\n        over the validation dataloader.\n\n        If overriding this method, you can access model, criterion via\n        ``self.model`` and ``self.criterion``. You also do not need to call\n        ``validate_batch`` if overriding this method.\n\n        Args:\n            val_iterator (iter): Iterable constructed from the\n                validation dataloader.\n\n        Returns:\n            A dict of metrics from the evaluation.\n                By default, returns "val_accuracy" and "val_loss"\n                which is computed by aggregating "loss" and "correct" values\n                from ``validate_batch`` and dividing it by the sum of\n                ``num_samples`` from all calls to ``self.validate_batch``.\n        '
        self._mode = 'val'
        self.model.eval()
        metrics = Metric.convert_metrics_dict(metrics, backend='pytorch')
        losses = []
        total_samples = 0
        with torch.no_grad():
            self.call_hook(callbacks=callbacks, fn_name='before_val_epoch')
            for (batch_idx, batch) in enumerate(val_iterator):
                self.batch_idx = batch_idx
                if num_steps and batch_idx == num_steps:
                    break
                (output, target, loss) = self.forward_batch(batch, callbacks)
                num_samples = get_batchsize(output)
                total_samples += num_samples
                losses.append(loss.item() * num_samples)
                for metric in metrics.values():
                    metric(output, target)
                del self.batch_idx
        result = {name: metric.compute() for (name, metric) in metrics.items()}
        result['val_loss'] = sum(losses) / total_samples
        result['num_samples'] = total_samples
        self.call_hook(callbacks=callbacks, fn_name='after_val_epoch')
        return result

    def forward_batch(self, batch, callbacks=None):
        if False:
            print('Hello World!')
        'Calculates the loss and accuracy over a given batch.\n\n        You can override this method to provide arbitrary metrics.\n\n        Same as ``train_batch``, this method implementation assumes that\n        batches are in (\\*features, labels) format by default. So we also\n        support multiple inputs model.\n\n        Args:\n            batch: One item of the validation iterator.\n\n        Returns:\n            A dict of metrics.\n                By default, returns "val_loss", "val_accuracy", and\n                "num_samples". When overriding, consider returning\n                "num_samples" in the metrics because\n                by default, ``validate`` uses "num_samples" to\n                calculate averages.\n        '
        self.batch = batch
        self.call_hook(callbacks=callbacks, fn_name='before_val_iter')
        with self.timers.record('eval_fwd'):
            self.call_hook(callbacks=callbacks, fn_name='on_val_forward')
        self.call_hook(callbacks=callbacks, fn_name='after_val_iter')
        (output, target, loss) = (None, None, None)
        if hasattr(self, 'output'):
            output = self.output
            del self.output
        if hasattr(self, 'target'):
            target = self.target
            del self.target
        del self.batch
        if hasattr(self, 'loss'):
            loss = self.loss
            del self.loss
        return (output, target, loss)

    def predict(self, partition, batch_size=32, profile=False, callbacks=None):
        if False:
            for i in range(10):
                print('nop')
        'Predict the model.'
        config = copy.copy(self.config)
        self._toggle_profiling(profile=profile)
        if not self.models:
            invalidInputError(False, 'You must provide a model for predict.')
        params = {'batch_size': batch_size, 'shuffle': False}
        for arg in ['shuffle', 'sampler', 'batch_sampler', 'num_workers', 'collate_fn', 'pin_memory', 'drop_last', 'timeout', 'worker_init_fn', 'multiprocessing_context']:
            if arg in config:
                params[arg] = config[arg]

        def predict_fn(shard):
            if False:
                while True:
                    i = 10
            if isinstance(partition, IterableDataset):
                y = self._predict(shard, callbacks=callbacks)
            else:
                if isinstance(shard['x'], tuple) or isinstance(shard['x'], list):
                    tensors = [torch.from_numpy(arr) for arr in shard['x']]
                else:
                    tensors = [torch.from_numpy(shard['x'])]
                dataset = torch.utils.data.TensorDataset(*tensors)
                data_loader = DataLoader(dataset, **params)
                y = self._predict(iter(data_loader), callbacks=callbacks)
            return split_predict_cols(y)
        self.call_hook(callbacks, 'before_pred_epoch')
        with self.timers.record('predict'):
            if isinstance(partition, IterableDataset):
                new_part = [predict_fn(shard) for (shard, shard_idx) in partition]
            else:
                new_part = [predict_fn(shard) for shard in partition]
        self.call_hook(callbacks, 'after_pred_epoch')
        return new_part

    def _predict(self, pred_iterator, callbacks=None):
        if False:
            for i in range(10):
                print('nop')
        self._mode = 'predict'
        self.model.eval()
        result = []
        with torch.no_grad():
            for (batch_idx, batch) in enumerate(pred_iterator):
                if isinstance(batch, torch.Tensor):
                    batch = [batch]
                self.batch = batch
                self.batch_idx = batch_idx
                self.call_hook(callbacks, 'before_pred_iter')
                result.append(self.predict_batch(self.batch, callbacks=callbacks))
                self.call_hook(callbacks, 'after_pred_iter')
                del self.batch
                del self.batch_idx
                del self.output
        return index_concatenate(result, axis=0)

    def predict_batch(self, batch, callbacks=None):
        if False:
            return 10
        self.batch = batch
        with self.timers.record('pred_fwd'):
            self.call_hook(callbacks, 'on_pred_forward')
        return self.output

    def _toggle_profiling(self, profile=False):
        if False:
            for i in range(10):
                print('nop')
        'Enables/Disables and resets timing profiles.'
        if profile:
            self.timers.enable()
            self.timers.reset()
        else:
            self.timers.disable()

    def get_state_dict(self):
        if False:
            while True:
                i = 10
        'Returns the state of the runner.'
        state = {'epoch': self.epochs, 'models': [model.state_dict() for model in self.models]}
        if self.optimizers:
            state.update({'optimizers': [opt.state_dict() for opt in self.optimizers]})
        if self.schedulers:
            state.update({'schedulers': [scheduler.state_dict() for scheduler in self.schedulers]})
        return state

    def load_state_dict(self, state):
        if False:
            while True:
                i = 10
        'Sets the state of the model.'
        import collections
        if isinstance(state, collections.OrderedDict):
            for (model, state_dict) in zip(self.models, [state]):
                model.load_state_dict(state_dict)
        elif 'models' in state:
            for (model, state_dict) in zip(self.models, state['models']):
                model.load_state_dict(state_dict)
        else:
            for (model, state_dict) in zip(self.models, state):
                model.load_state_dict(state_dict)
        if self.optimizers and 'optimizers' in state:
            for (optimizer, state_dict) in zip(self.optimizers, state['optimizers']):
                optimizer.load_state_dict(state_dict)
        if self.schedulers and 'schedulers' in state:
            for (scheduler, state_dict) in zip(self.schedulers, state['schedulers']):
                scheduler.load_state_dict(state_dict)
        if 'epoch' in state:
            self.epochs = state['epoch']

    def save_checkpoint(self, filepath, save_weights_only=False):
        if False:
            print('Hello World!')
        if self.rank == 0:
            if save_weights_only:
                checkpoint = {'epoch': self.epochs, 'models': [model.state_dict() for model in self.models]}
            else:
                checkpoint = self.get_state_dict()
            byte_obj = TorchRunner._state_dict2stream(checkpoint)
            file_name = os.path.basename(filepath)
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file_name)
            with open(temp_path, 'wb') as f:
                f.write(byte_obj)
            from bigdl.orca.data.file import put_local_file_to_remote
            put_local_file_to_remote(temp_path, filepath)
            self.logger.debug(f'Saved checkpoint: {filepath}')
        return filepath

    def remove_checkpoint(self, filepath):
        if False:
            return 10
        if self.rank == 0:
            from bigdl.orca.data.file import exists, rmdir
            if exists(filepath):
                rmdir(filepath)
                self.logger.debug(f'Removed checkpoint: {filepath}')

    def shutdown(self):
        if False:
            while True:
                i = 10
        'Attempts to shut down the worker.'
        del self.validation_loader
        del self.train_loader
        del self.criterion
        del self.optimizers
        del self.models

    def call_hook(self, callbacks, fn_name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Call all hooks.\n\n        Args:\n            fn_name (str): The function name in each hook to be called, such as\n                "on_iter_begin".\n        '
        for hook in callbacks:
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self)

    def put(self, k, v):
        if False:
            while True:
                i = 10
        if k in self._pocket.keys():
            self.logger.warning(f'Key {k} has already been in runner._pocket,please use runner.update instead.')
        self._pocket[k] = v

    def update(self, k, v):
        if False:
            while True:
                i = 10
        if k not in self._pocket.keys():
            self.logger.warning(f'Key {k} is not in runner._pocket,please use runner.put instead.')
        self._pocket[k] = v

    def get(self, k):
        if False:
            i = 10
            return i + 15
        invalidInputError(k in self._pocket.keys(), f'KeyError, key {k} is not in runner._pocket,please check your input.')
        return self._pocket[k]

    def remove(self, k):
        if False:
            return 10
        invalidInputError(k in self._pocket.keys(), f'KeyError, key {k} is not in runner.pocket,please check your input.')
        del self._pocket[k]

    @property
    def given_models(self):
        if False:
            print('Hello World!')
        if len(self.models) > 1:
            return self.models
        else:
            return self.models[0]

    @property
    def given_optimizers(self):
        if False:
            print('Hello World!')
        if len(self.optimizers) > 1:
            return self.optimizers
        else:
            return self.optimizers[0]

    @property
    def model(self):
        if False:
            print('Hello World!')
        '\n        First or only model(s) created by the ``model_creator``.\n        Discuss whether to return ddp model depending on the mode.\n        '
        if self._mode == 'train':
            if self.training_models:
                return self.training_models[0]
        elif self.models:
            return self.models[0]

    @property
    def optimizer(self):
        if False:
            while True:
                i = 10
        'First or only optimizer(s) created by the ``optimizer_creator``.'
        return self.optimizers[0]

    @property
    def scheduler(self):
        if False:
            print('Hello World!')
        'First or only scheduler(s) created by the ``scheduler_creator``.'
        if self.schedulers:
            return self.schedulers[0]