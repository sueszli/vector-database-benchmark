import logging
import torch
from fairseq import optim
from omegaconf import DictConfig
logger = logging.getLogger(__name__)

class AMPOptimizer(optim.FairseqOptimizer):
    """
    Wrap an *optimizer* to support AMP (automatic mixed precision) training.
    """

    def __init__(self, cfg: DictConfig, params, fp32_optimizer, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(cfg.optimizer)
        self.fp32_optimizer = fp32_optimizer
        amp_kwargs = {'init_scale': cfg.common.fp16_init_scale}
        if getattr(cfg.common, 'amp_scale_window', None) is not None:
            amp_kwargs['growth_interval'] = cfg.common.amp_init_scale
        self._grad_scaler = torch.cuda.amp.GradScaler(**amp_kwargs)
        self.min_loss_scale = cfg.common.min_loss_scale

    @classmethod
    def build_optimizer(cls, cfg: DictConfig, params, **kwargs):
        if False:
            print('Hello World!')
        '\n        Args:\n            cfg (omegaconf.DictConfig): fairseq args\n            params (iterable): iterable of parameters to optimize\n        '
        fp32_optimizer = optim.build_optimizer(cfg.optimizer, params)
        return cls(cfg, params, fp32_optimizer, **kwargs)

    def backward(self, loss):
        if False:
            return 10
        'Computes the sum of gradients of the given tensor w.r.t. graph leaves.\n\n        Compared to :func:`fairseq.optim.FairseqOptimizer.backward`, this\n        function additionally dynamically scales the loss to avoid gradient\n        underflow.\n        '
        self._grad_scaler.scale(loss).backward()

    def step(self):
        if False:
            for i in range(10):
                print('nop')
        self.scaler.step(self.fp32_optimizer)
        self.scaler.update()

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        if False:
            return 10
        'Clips gradient norm.'
        self.scaler.unscale_(self.optimizer)
        grad_norm = self.fp32_optimizer.clip_grad_norm(max_norm, aggregate_norm_fn)
        if not torch.isfinite(grad_norm).all():
            new_loss_scale = self.next_loss_scale
            if new_loss_scale <= self.min_loss_scale:
                raise FloatingPointError('AMP: Minimum loss scale reached ({}). Your loss is probably exploding. Try restarting training or use fp32. {}'.format(self.min_loss_scale, new_loss_scale))
            else:
                logger.info(f'AMP: overflow detected, setting scale to to {new_loss_scale}')
        return grad_norm

    @property
    def scaler(self):
        if False:
            print('Hello World!')
        return self._grad_scaler

    @property
    def next_loss_scale(self):
        if False:
            print('Hello World!')
        return self.scaler.get_scale() * self.scaler.get_backoff_factor()

    @property
    def optimizer(self):
        if False:
            print('Hello World!')
        return self.fp32_optimizer.optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if False:
            while True:
                i = 10
        self.fp32_optimizer.optimizer = optimizer

    @property
    def lr_scheduler(self):
        if False:
            return 10
        return getattr(self.fp32_optimizer, 'lr_scheduler', None)

    @property
    def optimizer_config(self):
        if False:
            i = 10
            return i + 15
        return self.fp32_optimizer.optimizer_config

    def get_lr(self):
        if False:
            for i in range(10):
                print('nop')
        return self.fp32_optimizer.get_lr()

    def set_lr(self, lr):
        if False:
            for i in range(10):
                print('nop')
        self.fp32_optimizer.set_lr(lr)

    def all_reduce_grads(self, module):
        if False:
            for i in range(10):
                print('nop')
        self.fp32_optimizer.all_reduce_grads(module)

    @property
    def supports_flat_params(self):
        if False:
            i = 10
            return i + 15
        return self.fp32_optimizer.supports_flat_params