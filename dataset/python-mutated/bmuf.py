from dataclasses import dataclass, field
import torch
import torch.distributed as dist
from fairseq.dataclass.configs import FairseqBMUFConfig
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.optim.fairseq_optimizer import FairseqOptimizer

class FairseqBMUF(FairseqOptimizer):
    """
    Implements incremental block distributed data parallelism similar to
    https://ieeexplore.ieee.org/document/7472805

    Paper title: Scalable training of deep learning machines by incremental
    block training with intra-block parallel optimization and blockwise
    model-update filtering
    """

    def __init__(self, cfg: FairseqBMUFConfig, optimizer):
        if False:
            while True:
                i = 10
        super().__init__(cfg)
        self._optimizer = optimizer
        self._num_updates = 0
        self.sync_iter = cfg.global_sync_iter
        self.block_momentum = cfg.block_momentum
        self.block_lr = cfg.block_lr
        self._reset_local_data()
        self.warmup_iteration = cfg.warmup_iterations
        self.use_nbm = cfg.use_nbm
        self.initial_state = self._optimizer.state_dict()
        self.average_sync = self.cfg.average_sync
        self.world_size = self.cfg.distributed_world_size

    @staticmethod
    def add_args(parser):
        if False:
            return 10
        'Add optimizer-specific arguments to the parser.'
        gen_parser_from_dataclass(parser, FairseqBMUFConfig())

    @property
    def optimizer(self):
        if False:
            while True:
                i = 10
        return self._optimizer.optimizer

    @property
    def optimizer_config(self):
        if False:
            for i in range(10):
                print('nop')
        return self._optimizer.optimizer_config

    def get_lr(self):
        if False:
            for i in range(10):
                print('nop')
        return self._optimizer.get_lr()

    def set_lr(self, lr):
        if False:
            print('Hello World!')
        self._optimizer.set_lr(lr)

    def state_dict(self):
        if False:
            i = 10
            return i + 15
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        if False:
            print('Hello World!')
        self._optimizer.load_state_dict(state_dict, optimizer_overrides)
        self.initial_state = self._optimizer.state_dict()

    def multiply_grads(self, c):
        if False:
            for i in range(10):
                print('nop')
        'Multiplies grads by a constant *c*.'
        self._optimizer.multiply_grads(c)

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        if False:
            for i in range(10):
                print('nop')
        'Clips gradient norm.'
        return self._optimizer.clip_grad_norm(max_norm, aggregate_norm_fn)

    def average_params(self):
        if False:
            while True:
                i = 10
        self._optimizer.average_params()

    def _block_sync(self):
        if False:
            for i in range(10):
                print('nop')
        if self.world_size <= 1:
            return
        if self.block_momentum != 0:
            self._calc_grad()
        self._avg_grad_from_all_gpus()
        if self.block_momentum != 0:
            self._update_global_model()
        if self.average_sync:
            self.average_params()

    def _is_warmup_end(self):
        if False:
            print('Hello World!')
        if self.get_num_updates() == self.warmup_iteration:
            return True
        return False

    def _is_bmuf_iter(self):
        if False:
            return 10
        if self.get_num_updates() > self.warmup_iteration and self.get_num_updates() % self.sync_iter == 0:
            return True
        return False

    def _warmup_sync(self, root_rank=0):
        if False:
            print('Hello World!')
        if self.world_size <= 1:
            return
        for param in self.params:
            dist.broadcast(param.data, src=root_rank)
        if self.average_sync:
            self._optimizer.average_params()
        else:
            self._optimizer.load_state_dict(self.initial_state)
        self._reset_local_data()

    def step(self, closure=None):
        if False:
            while True:
                i = 10
        'Performs a single optimization step.'
        self._optimizer.step(closure)
        self.set_num_updates(self.get_num_updates() + 1)
        if self._is_warmup_end():
            self._warmup_sync()
        elif self._is_bmuf_iter():
            self._block_sync()

    def zero_grad(self):
        if False:
            while True:
                i = 10
        'Clears the gradients of all optimized parameters.'
        self._optimizer.zero_grad()

    def get_num_updates(self):
        if False:
            while True:
                i = 10
        'Get the number of parameters updates.'
        return self._num_updates

    def set_num_updates(self, num_updates):
        if False:
            i = 10
            return i + 15
        'Set the number of parameters updates.'
        self._num_updates = num_updates

    @torch.no_grad()
    def _reset_local_data(self):
        if False:
            i = 10
            return i + 15
        self.global_params = [torch.zeros_like(p.data) for p in self.params]
        self.smoothed_grads = [p.data.new_zeros(p.data.size()) for p in self.params]
        self.grads = [p.data.new_zeros(p.data.size()) for p in self.params]
        for (param, global_param) in zip(self.params, self.global_params):
            global_param.copy_(param.data)

    @torch.no_grad()
    def _calc_grad(self):
        if False:
            while True:
                i = 10
        for (index, (param, global_param)) in enumerate(zip(self.params, self.global_params)):
            self.grads[index] = global_param - param.data

    def _avg_grad_from_all_gpus(self):
        if False:
            for i in range(10):
                print('nop')
        for (index, param) in enumerate(self.params):
            sync_para = param.data if self.block_momentum == 0 else self.grads[index]
            sync_para /= float(dist.get_world_size())
            dist.all_reduce(sync_para, op=dist.ReduceOp.SUM)

    @torch.no_grad()
    def _update_global_model(self):
        if False:
            i = 10
            return i + 15
        for (index, (param, global_param, smoothed_grad, grad)) in enumerate(zip(self.params, self.global_params, self.smoothed_grads, self.grads)):
            smoothed_grad = self.block_momentum * smoothed_grad + self.block_lr * grad
            param.data.copy_(global_param - smoothed_grad)
            if self.use_nbm:
                param.data.copy_(param.data - self.block_momentum * smoothed_grad)
            self.smoothed_grads[index] = smoothed_grad
            global_param.copy_(param.data)