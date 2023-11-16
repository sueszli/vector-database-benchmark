import ray
import copy
from bigdl.orca.learn.pytorch.utils import find_free_port
from bigdl.orca.learn.pytorch.torch_runner import TorchRunner
import torch.nn as nn
from torch.utils.data import IterableDataset
import logging
from bigdl.dllib.utils.log4Error import *
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

class PytorchRayWorker(TorchRunner):
    """Manages a PyTorch model for training."""

    def __init__(self, model_creator, optimizer_creator, loss_creator=None, metrics=None, scheduler_creator=None, config=None, sync_stats=True, log_level=logging.INFO):
        if False:
            print('Hello World!')
        super().__init__(model_creator=model_creator, optimizer_creator=optimizer_creator, loss_creator=loss_creator, metrics=metrics, scheduler_creator=scheduler_creator, config=config, sync_stats=sync_stats, log_level=log_level)
        self.backend = 'torch-local'
        self.rank = 0
        self.size = 0

    def setup_horovod(self):
        if False:
            i = 10
            return i + 15
        import horovod.torch as hvd
        hvd.init()
        self.backend = 'horovod'
        self.rank = hvd.rank()
        self.size = hvd.size()
        self.setup_components_horovod()
        self.training_models = self.models
        self.setup_operator(self.training_models)

    def get_node_ip_port(self):
        if False:
            for i in range(10):
                print('nop')
        ip = self.get_node_ip()
        port = find_free_port()
        return (ip, port)

    def get_node_ip(self):
        if False:
            while True:
                i = 10
        'Returns the IP address of the current node.'
        return ray._private.services.get_node_ip_address()

    def setup_components_horovod(self):
        if False:
            print('Hello World!')
        import horovod.torch as hvd
        self.logger.debug('Creating model')
        self.models = self.model_creator(self.config)
        if not isinstance(self.models, Iterable):
            self.models = [self.models]
        else:
            invalidInputError(False, 'only support single model for now')
        invalidInputError(all((isinstance(model, nn.Module) for model in self.models)), 'All models must be PyTorch models: {}.'.format(self.models))
        self.logger.debug('Creating optimizer.')
        self.optimizers = self.optimizer_creator(self.given_models, self.config)
        if not isinstance(self.optimizers, Iterable):
            hvd.broadcast_parameters(self.models[0].state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizers, root_rank=0)
            parameters = self.models[0].named_parameters()
            self.optimizers = hvd.DistributedOptimizer(self.optimizers, named_parameters=parameters)
            self.optimizers = [self.optimizers]
        else:
            invalidInputError(False, 'only support one optimizer for now')
        self._create_schedulers_if_available()
        self._create_loss()

    def predict(self, data_creator, batch_size=32, profile=False, callbacks=None):
        if False:
            while True:
                i = 10
        'Evaluates the model on the validation data set.'
        config = copy.copy(self.config)
        self._toggle_profiling(profile=profile)
        shards_ref = data_creator(config, batch_size)
        if isinstance(shards_ref, IterableDataset):
            pred_stats = super().predict(partition=shards_ref, batch_size=batch_size, profile=profile, callbacks=callbacks)
            for pred_stat in pred_stats:
                pred_stat.update(pred_stat)
            worker_stats = pred_stat['prediction']
        else:
            if not isinstance(shards_ref, ray.ObjectID):
                invalidInputError(False, 'Only xshards and Ray Dataset is supported for predict')
            partition = ray.get(shards_ref)
            worker_stats = super().predict(partition=partition, batch_size=batch_size, profile=profile, callbacks=callbacks)
        return worker_stats