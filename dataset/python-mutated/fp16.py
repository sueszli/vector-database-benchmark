from __future__ import annotations
from ..basics import *
from .progress import *
from torch.cuda.amp import GradScaler, autocast
from torch.cuda.amp.grad_scaler import OptState
__all__ = ['AMPMode', 'MixedPrecision', 'get_master', 'to_master_grads', 'to_model_params', 'test_overflow', 'grad_overflow', 'copy_clone', 'ModelToHalf', 'NonNativeMixedPrecision']

class AMPMode(Enum):
    """Automatic mixed precision modes for ease of completion"""
    FP16 = 'fp16'
    BF16 = 'bf16'

@delegates(GradScaler)
class MixedPrecision(Callback):
    """Mixed precision training using Pytorch's Automatic Mixed Precision (AMP)"""
    order = 10

    def __init__(self, amp_mode: str | AMPMode=AMPMode.FP16, **kwargs):
        if False:
            print('Hello World!')
        amp_mode = AMPMode(amp_mode)
        store_attr(names='amp_mode')
        self.kwargs = kwargs

    def before_fit(self):
        if False:
            while True:
                i = 10
        if self.amp_mode == AMPMode.BF16:
            if torch.cuda.is_available() and (not torch.cuda.is_bf16_supported()):
                raise ValueError('Unsupported GPU for bfloat16 mixed precision training')
            dtype = torch.bfloat16
        elif self.amp_mode == AMPMode.FP16:
            dtype = torch.float16
        else:
            raise ValueError(f'Unrecognized precision: {self.amp_mode}')
        self.kwargs['enabled'] = dtype == torch.float16
        (self.autocast, self.learn.scaler, self.scales) = (autocast(dtype=dtype), GradScaler(**self.kwargs), L())

    def before_batch(self):
        if False:
            print('Hello World!')
        self.autocast.__enter__()

    def after_pred(self):
        if False:
            i = 10
            return i + 15
        self.learn.pred = to_float(self.pred)

    def after_loss(self):
        if False:
            while True:
                i = 10
        self.autocast.__exit__(None, None, None)

    def before_backward(self):
        if False:
            for i in range(10):
                print('nop')
        self.learn.loss_grad = self.scaler.scale(self.loss_grad)

    def before_step(self):
        if False:
            return 10
        'Use `self` as a fake optimizer. `self.skipped` will be set to True `after_step` if gradients overflow.'
        self.skipped = True
        self.scaler.step(self)
        if self.skipped:
            raise CancelStepException()
        self.scales.append(self.scaler.get_scale())

    def after_step(self):
        if False:
            return 10
        self.learn.scaler.update()

    def after_fit(self):
        if False:
            for i in range(10):
                print('nop')
        (self.autocast, self.learn.scaler, self.scales) = (None, None, None)

    @property
    def param_groups(self):
        if False:
            while True:
                i = 10
        'Pretend to be an optimizer for `GradScaler`'
        return self.opt.param_groups

    def step(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Fake optimizer step to detect whether this batch was skipped from `GradScaler`'
        self.skipped = False

@patch
@delegates(GradScaler)
def to_fp16(self: Learner, **kwargs):
    if False:
        print('Hello World!')
    'Set `Learner` to float16 mixed precision using PyTorch AMP'
    return self.add_cb(MixedPrecision(**kwargs))

@patch
def to_bf16(self: Learner):
    if False:
        while True:
            i = 10
    'Set `Learner` to bfloat16 mixed precision using PyTorch AMP'
    return self.add_cb(MixedPrecision(amp_mode=AMPMode.BF16))

@patch
def to_fp32(self: Learner):
    if False:
        return 10
    'Set `Learner` to float32 precision'
    return self.remove_cb(MixedPrecision)
from ..fp16_utils import convert_network, model_grads_to_master_grads, master_params_to_model_params
from torch.nn.utils import parameters_to_vector

def get_master(opt: Optimizer, flat_master: bool=False) -> list:
    if False:
        i = 10
        return i + 15
    'Creates fp16 model params given an initialized `Optimizer`, also returning fp32 model params. '
    model_params = [[param for param in pg if getattr(param, 'requires_grad', False) and hasattr(param, 'data')] for pg in opt.param_lists]
    if flat_master:
        master_params = []
        for pg in model_params:
            mp = parameters_to_vector([param.data.float() for param in pg])
            mp = nn.Parameter(mp, requires_grad=True)
            if mp.grad is None:
                mp.grad = mp.new(*mp.size())
            master_params.append([mp])
    else:
        master_params = [[nn.Parameter(param.data.clone().float().detach(), requires_grad=True) for param in pg] for pg in model_params]
    return (model_params, master_params)

def to_master_grads(model_pgs: list, master_pgs: list, flat_master: bool=False):
    if False:
        i = 10
        return i + 15
    'Move fp16 model gradients to fp32 master gradients'
    for (model_params, master_params) in zip(model_pgs, master_pgs):
        model_grads_to_master_grads(model_params, master_params, flat_master=flat_master)

def to_model_params(model_pgs: list, master_pgs: list, flat_master: bool=False) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Copy updated fp32 master params to fp16 model params after gradient step. '
    for (model_params, master_params) in zip(model_pgs, master_pgs):
        master_params_to_model_params(model_params, master_params, flat_master=flat_master)

def test_overflow(x: torch.Tensor):
    if False:
        for i in range(10):
            print('nop')
    'Tests whether fp16 gradients have overflown.'
    s = float(x.float().sum())
    return s == float('inf') or s == float('-inf') or s != s

def grad_overflow(pgs: list) -> bool:
    if False:
        i = 10
        return i + 15
    'Tests all fp16 parameters in pgs for gradient overflow'
    for pg in pgs:
        for p in pg:
            if p.grad is not None and test_overflow(p.grad.data):
                return True
    return False

def copy_clone(d):
    if False:
        i = 10
        return i + 15
    return {k: v.detach().clone().float() if isinstance(v, Tensor) else v for (k, v) in d.items()}

def _copy_state(opt, pgs1, pgs2):
    if False:
        i = 10
        return i + 15
    opt.param_lists = pgs2
    for (pg1, pg2) in zip(pgs1, pgs2):
        for (p1, p2) in zip(pg1, pg2):
            opt.state[p2] = copy_clone(opt.state.pop(p1, {}))

class ModelToHalf(Callback):
    """Use with NonNativeMixedPrecision callback (but it needs to run at the very beginning)"""
    order = -50

    def before_fit(self):
        if False:
            i = 10
            return i + 15
        self.learn.model = convert_network(self.model, dtype=torch.float16)

    def after_fit(self):
        if False:
            print('Hello World!')
        self.learn.model = convert_network(self.model, dtype=torch.float32)

@docs
class NonNativeMixedPrecision(Callback):
    """Run training in mixed precision"""
    order = 10

    def __init__(self, loss_scale: int=512, flat_master: bool=False, dynamic: bool=True, max_loss_scale: float=2.0 ** 24, div_factor: float=2.0, scale_wait: int=500, clip: float=None):
        if False:
            i = 10
            return i + 15
        assert torch.backends.cudnn.enabled, 'Mixed precision training requires cudnn.'
        (self.flat_master, self.dynamic, self.max_loss_scale) = (flat_master, dynamic, max_loss_scale)
        (self.div_factor, self.scale_wait, self.clip) = (div_factor, scale_wait, clip)
        self.loss_scale = max_loss_scale if dynamic else loss_scale

    def before_fit(self):
        if False:
            return 10
        assert self.dls.device.type == 'cuda', 'Mixed-precision training requires a GPU, remove the call `to_fp16`'
        if self.learn.opt is None:
            self.learn.create_opt()
        (self.model_pgs, self.master_pgs) = get_master(self.opt, self.flat_master)
        self.old_pgs = self.opt.param_lists
        _copy_state(self.learn.opt, self.model_pgs, self.master_pgs)
        if self.dynamic:
            self.count = 0

    def before_batch(self):
        if False:
            for i in range(10):
                print('nop')
        self.learn.xb = to_half(self.xb)

    def after_pred(self):
        if False:
            print('Hello World!')
        self.learn.pred = to_float(self.pred)

    def before_backward(self):
        if False:
            while True:
                i = 10
        self.learn.loss_grad *= self.loss_scale

    def before_step(self):
        if False:
            while True:
                i = 10
        if self.dynamic and grad_overflow(self.model_pgs):
            self.loss_scale /= self.div_factor
            self.learn.loss_grad /= self.div_factor
            self.model.zero_grad()
            raise CancelBatchException()
        to_master_grads(self.model_pgs, self.master_pgs, self.flat_master)
        for master_params in self.master_pgs:
            for param in master_params:
                if param.grad is not None:
                    param.grad.div_(self.loss_scale)
        if self.clip is not None:
            for group in self.master_pgs:
                nn.utils.clip_grad_norm_(group, self.clip)
        if self.dynamic:
            self.count += 1
            if self.count == self.scale_wait:
                self.count = 0
                self.loss_scale *= self.div_factor

    def after_step(self):
        if False:
            for i in range(10):
                print('nop')
        self.model.zero_grad()
        to_model_params(self.model_pgs, self.master_pgs, self.flat_master)

    def after_batch(self):
        if False:
            i = 10
            return i + 15
        if self.training:
            self.learn.loss_grad /= self.loss_scale

    def after_fit(self):
        if False:
            while True:
                i = 10
        if not hasattr(self, 'master_pgs'):
            return
        _copy_state(self.learn.opt, self.master_pgs, self.model_pgs)
        self.learn.opt.param_lists = self.old_pgs
        delattr(self, 'master_pgs')
        delattr(self, 'model_pgs')
        delattr(self, 'old_pgs')
    _docs = dict(before_fit='Put the model in FP16 and prepare the two copies of the parameters', before_batch='Put the input in FP16', after_pred='Put the output back to FP32 so that the loss is computed in FP32', before_backward='Apply loss scaling to avoid gradient underflow', before_step='Update and apply dynamic loss scaling, move gradients to fp32, apply gradient clipping', after_step='Zero fp16 grads and update fp16 params with fp32 params. ', after_batch='Ensure loss is logged correctly', after_fit='Put the model back in FP32')

@patch
@delegates(NonNativeMixedPrecision.__init__)
def to_non_native_fp16(self: Learner, **kwargs):
    if False:
        i = 10
        return i + 15
    return self.add_cbs([ModelToHalf(), NonNativeMixedPrecision(**kwargs)])

@patch
def to_non_native_fp32(self: Learner):
    if False:
        for i in range(10):
            print('nop')
    return self.remove_cbs([ModelToHalf, NonNativeMixedPrecision])