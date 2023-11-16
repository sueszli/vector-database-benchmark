import torch
from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from collections import defaultdict

class FairseqOptimizer(object):

    def __init__(self, cfg):
        if False:
            return 10
        super().__init__()
        self.cfg = cfg

    @classmethod
    def add_args(cls, parser):
        if False:
            for i in range(10):
                print('nop')
        'Add optimizer-specific arguments to the parser.'
        dc = getattr(cls, '__dataclass', None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @property
    def optimizer(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a torch.optim.optimizer.Optimizer instance.'
        if not hasattr(self, '_optimizer'):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError('_optimizer must be an instance of torch.optim.Optimizer')
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if False:
            print('Hello World!')
        'Reset optimizer instance.'
        if not hasattr(self, '_optimizer'):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError('_optimizer must be an instance of torch.optim.Optimizer')
        self._optimizer = optimizer

    @property
    def optimizer_config(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a kwarg dictionary that will be used to override optimizer\n        args stored in checkpoints. This allows us to load a checkpoint and\n        resume training using a different set of optimizer args, e.g., with a\n        different learning rate.\n        '
        raise NotImplementedError

    @property
    def params(self):
        if False:
            for i in range(10):
                print('nop')
        'Return an iterable of the parameters held by the optimizer.'
        for param_group in self.param_groups:
            for p in param_group['params']:
                yield p

    @property
    def param_groups(self):
        if False:
            print('Hello World!')
        return self.optimizer.param_groups

    def __getstate__(self):
        if False:
            while True:
                i = 10
        return self._optimizer.__getstate__()

    def get_lr(self):
        if False:
            print('Hello World!')
        'Return the current learning rate.'
        return self.param_groups[0]['lr']

    def set_lr(self, lr):
        if False:
            while True:
                i = 10
        'Set the learning rate.'
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        if False:
            i = 10
            return i + 15
        "Return the optimizer's state dict."
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        if False:
            while True:
                i = 10
        'Load an optimizer state dict.\n\n        In general we should prefer the configuration of the existing optimizer\n        instance (e.g., learning rate) over that found in the state_dict. This\n        allows us to resume training from a checkpoint using a new set of\n        optimizer args.\n        '
        self.optimizer.load_state_dict(state_dict)
        if optimizer_overrides is not None and len(optimizer_overrides) > 0:
            for group in self.param_groups:
                group.update(optimizer_overrides)

    def backward(self, loss):
        if False:
            i = 10
            return i + 15
        'Computes the sum of gradients of the given tensor w.r.t. graph leaves.'
        loss.backward()

    def all_reduce_grads(self, module):
        if False:
            i = 10
            return i + 15
        'Manually all-reduce gradients (if required).'
        if hasattr(module, 'all_reduce_grads'):
            module.all_reduce_grads()

    def multiply_grads(self, c):
        if False:
            print('Hello World!')
        'Multiplies grads by a constant *c*.'
        per_device_and_dtype_grads = defaultdict(lambda : defaultdict(list))
        for p in self.params:
            if p.grad is not None:
                if p.grad.is_sparse:
                    p.grad.data.mul_(c.to(p.grad.device) if torch.is_tensor(c) else c)
                else:
                    per_device_and_dtype_grads[p.grad.device][p.grad.dtype].append(p.grad.data)
        for (device, per_dtype_grads) in per_device_and_dtype_grads.items():
            for grads in per_dtype_grads.values():
                torch._foreach_mul_(grads, c.to(device) if torch.is_tensor(c) else c)

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        if False:
            for i in range(10):
                print('nop')
        'Clips gradient norm.'
        return utils.clip_grad_norm_(self.params, max_norm, aggregate_norm_fn)

    def step(self, closure=None, scale=1.0, groups=None):
        if False:
            while True:
                i = 10
        'Performs a single optimization step.'
        if self.supports_step_with_scale:
            if self.supports_groups:
                self.optimizer.step(closure, scale=scale, groups=groups)
            else:
                self.optimizer.step(closure, scale=scale)
        else:
            if scale != 1.0:
                self.multiply_grads(1.0 / scale)
            if self.supports_groups:
                self.optimizer.step(closure, groups=groups)
            else:
                self.optimizer.step(closure)

    def zero_grad(self):
        if False:
            for i in range(10):
                print('nop')
        'Clears the gradients of all optimized parameters.'
        for p in self.params:
            p.grad = None
        self.optimizer.zero_grad()

    @property
    def supports_memory_efficient_fp16(self):
        if False:
            while True:
                i = 10
        if hasattr(self.optimizer, 'supports_memory_efficient_fp16'):
            return self.optimizer.supports_memory_efficient_fp16
        return False

    @property
    def supports_step_with_scale(self):
        if False:
            i = 10
            return i + 15
        if hasattr(self.optimizer, 'supports_step_with_scale'):
            return self.optimizer.supports_step_with_scale
        return False

    @property
    def supports_groups(self):
        if False:
            print('Hello World!')
        if hasattr(self.optimizer, 'supports_groups'):
            return self.optimizer.supports_groups
        return False

    @property
    def supports_flat_params(self):
        if False:
            while True:
                i = 10
        '\n        Whether the optimizer supports collapsing of the model\n        parameters/gradients into a single contiguous Tensor.\n        '
        if hasattr(self.optimizer, 'supports_flat_params'):
            return self.optimizer.supports_flat_params
        return False

    def average_params(self):
        if False:
            i = 10
            return i + 15
        pass

    def broadcast_global_state_dict(self, state_dict):
        if False:
            while True:
                i = 10
        '\n        Broadcasts a global state dict to all ranks.\n        Useful for optimizers that shard state between ranks.\n        '
        if hasattr(self.optimizer, 'broadcast_global_state_dict'):
            return self.optimizer.broadcast_global_state_dict(state_dict)
        else:
            return state_dict

class LegacyFairseqOptimizer(FairseqOptimizer):

    def __init__(self, args):
        if False:
            i = 10
            return i + 15
        self.args = args