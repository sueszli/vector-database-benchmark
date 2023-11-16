from __future__ import annotations
from ..basics import *
from .all import *
__all__ = ['GANModule', 'basic_critic', 'AddChannels', 'basic_generator', 'DenseResBlock', 'gan_critic', 'GANLoss', 'AdaptiveLoss', 'accuracy_thresh_expand', 'set_freeze_model', 'GANTrainer', 'FixedGANSwitcher', 'AdaptiveGANSwitcher', 'GANDiscriminativeLR', 'InvisibleTensor', 'generate_noise', 'show_batch', 'show_results', 'gan_loss_from_func', 'GANLearner']

class GANModule(Module):
    """Wrapper around a `generator` and a `critic` to create a GAN."""

    def __init__(self, generator: nn.Module=None, critic: nn.Module=None, gen_mode: None | bool=False):
        if False:
            return 10
        if generator is not None:
            self.generator = generator
        if critic is not None:
            self.critic = critic
        store_attr('gen_mode')

    def forward(self, *args):
        if False:
            i = 10
            return i + 15
        return self.generator(*args) if self.gen_mode else self.critic(*args)

    def switch(self, gen_mode: None | bool=None):
        if False:
            for i in range(10):
                print('nop')
        'Put the module in generator mode if `gen_mode` is `True`, in critic mode otherwise.'
        self.gen_mode = not self.gen_mode if gen_mode is None else gen_mode

@delegates(ConvLayer.__init__)
def basic_critic(in_size: int, n_channels: int, n_features: int=64, n_extra_layers: int=0, norm_type: NormType=NormType.Batch, **kwargs) -> nn.Sequential:
    if False:
        while True:
            i = 10
    'A basic critic for images `n_channels` x `in_size` x `in_size`.'
    layers = [ConvLayer(n_channels, n_features, 4, 2, 1, norm_type=None, **kwargs)]
    (cur_size, cur_ftrs) = (in_size // 2, n_features)
    layers += [ConvLayer(cur_ftrs, cur_ftrs, 3, 1, norm_type=norm_type, **kwargs) for _ in range(n_extra_layers)]
    while cur_size > 4:
        layers.append(ConvLayer(cur_ftrs, cur_ftrs * 2, 4, 2, 1, norm_type=norm_type, **kwargs))
        cur_ftrs *= 2
        cur_size //= 2
    init = kwargs.get('init', nn.init.kaiming_normal_)
    layers += [init_default(nn.Conv2d(cur_ftrs, 1, 4, padding=0), init), Flatten()]
    return nn.Sequential(*layers)

class AddChannels(Module):
    """Add `n_dim` channels at the end of the input."""

    def __init__(self, n_dim):
        if False:
            return 10
        self.n_dim = n_dim

    def forward(self, x):
        if False:
            while True:
                i = 10
        return x.view(*list(x.shape) + [1] * self.n_dim)

@delegates(ConvLayer.__init__)
def basic_generator(out_size: int, n_channels: int, in_sz: int=100, n_features: int=64, n_extra_layers: int=0, **kwargs) -> nn.Sequential:
    if False:
        i = 10
        return i + 15
    'A basic generator from `in_sz` to images `n_channels` x `out_size` x `out_size`.'
    (cur_size, cur_ftrs) = (4, n_features // 2)
    while cur_size < out_size:
        cur_size *= 2
        cur_ftrs *= 2
    layers = [AddChannels(2), ConvLayer(in_sz, cur_ftrs, 4, 1, transpose=True, **kwargs)]
    cur_size = 4
    while cur_size < out_size // 2:
        layers.append(ConvLayer(cur_ftrs, cur_ftrs // 2, 4, 2, 1, transpose=True, **kwargs))
        cur_ftrs //= 2
        cur_size *= 2
    layers += [ConvLayer(cur_ftrs, cur_ftrs, 3, 1, 1, transpose=True, **kwargs) for _ in range(n_extra_layers)]
    layers += [nn.ConvTranspose2d(cur_ftrs, n_channels, 4, 2, 1, bias=False), nn.Tanh()]
    return nn.Sequential(*layers)
_conv_args = dict(act_cls=partial(nn.LeakyReLU, negative_slope=0.2), norm_type=NormType.Spectral)

def _conv(ni, nf, ks=3, stride=1, self_attention=False, **kwargs):
    if False:
        return 10
    if self_attention:
        kwargs['xtra'] = SelfAttention(nf)
    return ConvLayer(ni, nf, ks=ks, stride=stride, **_conv_args, **kwargs)

@delegates(ConvLayer)
def DenseResBlock(nf: int, norm_type: NormType=NormType.Batch, **kwargs) -> SequentialEx:
    if False:
        return 10
    'Resnet block of `nf` features. `conv_kwargs` are passed to `conv_layer`.'
    return SequentialEx(ConvLayer(nf, nf, norm_type=norm_type, **kwargs), ConvLayer(nf, nf, norm_type=norm_type, **kwargs), MergeLayer(dense=True))

def gan_critic(n_channels: int=3, nf: int=128, n_blocks: int=3, p: float=0.15) -> nn.Sequential:
    if False:
        return 10
    'Critic to train a `GAN`.'
    layers = [_conv(n_channels, nf, ks=4, stride=2), nn.Dropout2d(p / 2), DenseResBlock(nf, **_conv_args)]
    nf *= 2
    for i in range(n_blocks):
        layers += [nn.Dropout2d(p), _conv(nf, nf * 2, ks=4, stride=2, self_attention=i == 0)]
        nf *= 2
    layers += [ConvLayer(nf, 1, ks=4, bias=False, padding=0, norm_type=NormType.Spectral, act_cls=None), Flatten()]
    return nn.Sequential(*layers)

class GANLoss(GANModule):
    """Wrapper around `crit_loss_func` and `gen_loss_func`"""

    def __init__(self, gen_loss_func: callable, crit_loss_func: callable, gan_model: GANModule):
        if False:
            i = 10
            return i + 15
        super().__init__()
        store_attr('gen_loss_func,crit_loss_func,gan_model')

    def generator(self, output, target):
        if False:
            return 10
        'Evaluate the `output` with the critic then uses `self.gen_loss_func` to evaluate how well the critic was fooled by `output`'
        fake_pred = self.gan_model.critic(output)
        self.gen_loss = self.gen_loss_func(fake_pred, output, target)
        return self.gen_loss

    def critic(self, real_pred, input):
        if False:
            for i in range(10):
                print('nop')
        'Create some `fake_pred` with the generator from `input` and compare them to `real_pred` in `self.crit_loss_func`.'
        fake = self.gan_model.generator(input).requires_grad_(False)
        fake_pred = self.gan_model.critic(fake)
        self.crit_loss = self.crit_loss_func(real_pred, fake_pred)
        return self.crit_loss

class AdaptiveLoss(Module):
    """Expand the `target` to match the `output` size before applying `crit`."""

    def __init__(self, crit: callable):
        if False:
            while True:
                i = 10
        self.crit = crit

    def forward(self, output: Tensor, target: Tensor):
        if False:
            i = 10
            return i + 15
        return self.crit(output, target[:, None].expand_as(output).float())

def accuracy_thresh_expand(y_pred: Tensor, y_true: Tensor, thresh: float=0.5, sigmoid: bool=True):
    if False:
        for i in range(10):
            print('nop')
    'Compute thresholded accuracy after expanding `y_true` to the size of `y_pred`.'
    if sigmoid:
        y_pred = y_pred.sigmoid()
    return ((y_pred > thresh).byte() == y_true[:, None].expand_as(y_pred).byte()).float().mean()

def set_freeze_model(m: nn.Module, rg: bool):
    if False:
        print('Hello World!')
    for p in m.parameters():
        p.requires_grad_(rg)

class GANTrainer(Callback):
    """Callback to handle GAN Training."""
    run_after = TrainEvalCallback

    def __init__(self, switch_eval: bool=False, clip: None | float=None, beta: float=0.98, gen_first: bool=False, show_img: bool=True):
        if False:
            i = 10
            return i + 15
        store_attr('switch_eval,clip,gen_first,show_img')
        (self.gen_loss, self.crit_loss) = (AvgSmoothLoss(beta=beta), AvgSmoothLoss(beta=beta))

    def _set_trainable(self):
        if False:
            print('Hello World!')
        'Appropriately set the generator and critic into a trainable or loss evaluation mode based on `self.gen_mode`.'
        train_model = self.generator if self.gen_mode else self.critic
        loss_model = self.generator if not self.gen_mode else self.critic
        set_freeze_model(train_model, True)
        set_freeze_model(loss_model, False)
        if self.switch_eval:
            train_model.train()
            loss_model.eval()

    def before_fit(self):
        if False:
            return 10
        'Initialization.'
        (self.generator, self.critic) = (self.model.generator, self.model.critic)
        self.gen_mode = self.gen_first
        self.switch(self.gen_mode)
        (self.crit_losses, self.gen_losses) = ([], [])
        self.gen_loss.reset()
        self.crit_loss.reset()

    def before_validate(self):
        if False:
            for i in range(10):
                print('nop')
        'Switch in generator mode for showing results.'
        self.switch(gen_mode=True)

    def before_batch(self):
        if False:
            print('Hello World!')
        "Clamp the weights with `self.clip` if it's not None, set the correct input/target."
        if self.training and self.clip is not None:
            for p in self.critic.parameters():
                p.data.clamp_(-self.clip, self.clip)
        if not self.gen_mode:
            (self.learn.xb, self.learn.yb) = (self.yb, self.xb)

    def after_batch(self):
        if False:
            for i in range(10):
                print('nop')
        'Record `last_loss` in the proper list.'
        if not self.training:
            return
        if self.gen_mode:
            self.gen_loss.accumulate(self.learn)
            self.gen_losses.append(self.gen_loss.value)
            self.last_gen = self.learn.to_detach(self.pred)
        else:
            self.crit_loss.accumulate(self.learn)
            self.crit_losses.append(self.crit_loss.value)

    def before_epoch(self):
        if False:
            print('Hello World!')
        'Put the critic or the generator back to eval if necessary.'
        self.switch(self.gen_mode)

    def switch(self, gen_mode=None):
        if False:
            i = 10
            return i + 15
        'Switch the model and loss function, if `gen_mode` is provided, in the desired mode.'
        self.gen_mode = not self.gen_mode if gen_mode is None else gen_mode
        self._set_trainable()
        self.model.switch(gen_mode)
        self.loss_func.switch(gen_mode)

class FixedGANSwitcher(Callback):
    """Switcher to do `n_crit` iterations of the critic then `n_gen` iterations of the generator."""
    run_after = GANTrainer

    def __init__(self, n_crit: int=1, n_gen: int=1):
        if False:
            while True:
                i = 10
        store_attr('n_crit,n_gen')

    def before_train(self):
        if False:
            print('Hello World!')
        (self.n_c, self.n_g) = (0, 0)

    def after_batch(self):
        if False:
            i = 10
            return i + 15
        'Switch the model if necessary.'
        if not self.training:
            return
        if self.learn.gan_trainer.gen_mode:
            self.n_g += 1
            (n_iter, n_in, n_out) = (self.n_gen, self.n_c, self.n_g)
        else:
            self.n_c += 1
            (n_iter, n_in, n_out) = (self.n_crit, self.n_g, self.n_c)
        target = n_iter if isinstance(n_iter, int) else n_iter(n_in)
        if target == n_out:
            self.learn.gan_trainer.switch()
            (self.n_c, self.n_g) = (0, 0)

class AdaptiveGANSwitcher(Callback):
    """Switcher that goes back to generator/critic when the loss goes below `gen_thresh`/`crit_thresh`."""
    run_after = GANTrainer

    def __init__(self, gen_thresh: None | float=None, critic_thresh: None | float=None):
        if False:
            return 10
        store_attr('gen_thresh,critic_thresh')

    def after_batch(self):
        if False:
            for i in range(10):
                print('nop')
        'Switch the model if necessary.'
        if not self.training:
            return
        if self.gan_trainer.gen_mode:
            if self.gen_thresh is None or self.loss < self.gen_thresh:
                self.gan_trainer.switch()
        elif self.critic_thresh is None or self.loss < self.critic_thresh:
            self.gan_trainer.switch()

class GANDiscriminativeLR(Callback):
    """`Callback` that handles multiplying the learning rate by `mult_lr` for the critic."""
    run_after = GANTrainer

    def __init__(self, mult_lr=5.0):
        if False:
            i = 10
            return i + 15
        self.mult_lr = mult_lr

    def before_batch(self):
        if False:
            i = 10
            return i + 15
        'Multiply the current lr if necessary.'
        if not self.learn.gan_trainer.gen_mode and self.training:
            self.learn.opt.set_hyper('lr', self.learn.opt.hypers[0]['lr'] * self.mult_lr)

    def after_batch(self):
        if False:
            i = 10
            return i + 15
        'Put the LR back to its value if necessary.'
        if not self.learn.gan_trainer.gen_mode:
            self.learn.opt.set_hyper('lr', self.learn.opt.hypers[0]['lr'] / self.mult_lr)

class InvisibleTensor(TensorBase):
    """TensorBase but show method does nothing"""

    def show(self, ctx=None, **kwargs):
        if False:
            return 10
        return ctx

def generate_noise(fn, size=100) -> InvisibleTensor:
    if False:
        while True:
            i = 10
    'Generate noise vector.'
    return cast(torch.randn(size), InvisibleTensor)

@typedispatch
def show_batch(x: InvisibleTensor, y: TensorImage, samples, ctxs=None, max_n=10, nrows=None, ncols=None, figsize=None, **kwargs):
    if False:
        print('Hello World!')
    if ctxs is None:
        ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)
    ctxs = show_batch[object](x, y, samples, ctxs=ctxs, max_n=max_n, **kwargs)
    return ctxs

@typedispatch
def show_results(x: InvisibleTensor, y: TensorImage, samples, outs, ctxs=None, max_n=10, nrows=None, ncols=None, figsize=None, **kwargs):
    if False:
        return 10
    if ctxs is None:
        ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)
    ctxs = [b.show(ctx=c, **kwargs) for (b, c, _) in zip(outs.itemgot(0), ctxs, range(max_n))]
    return ctxs

def gan_loss_from_func(loss_gen: callable, loss_crit: callable, weights_gen: None | MutableSequence | tuple=None):
    if False:
        while True:
            i = 10
    'Define loss functions for a GAN from `loss_gen` and `loss_crit`.'

    def _loss_G(fake_pred, output, target, weights_gen=weights_gen):
        if False:
            for i in range(10):
                print('nop')
        ones = fake_pred.new_ones(fake_pred.shape[0])
        weights_gen = ifnone(weights_gen, (1.0, 1.0))
        return weights_gen[0] * loss_crit(fake_pred, ones) + weights_gen[1] * loss_gen(output, target)

    def _loss_C(real_pred, fake_pred):
        if False:
            return 10
        ones = real_pred.new_ones(real_pred.shape[0])
        zeros = fake_pred.new_zeros(fake_pred.shape[0])
        return (loss_crit(real_pred, ones) + loss_crit(fake_pred, zeros)) / 2
    return (_loss_G, _loss_C)

def _tk_mean(fake_pred, output, target):
    if False:
        print('Hello World!')
    return fake_pred.mean()

def _tk_diff(real_pred, fake_pred):
    if False:
        print('Hello World!')
    return real_pred.mean() - fake_pred.mean()

@delegates()
class GANLearner(Learner):
    """A `Learner` suitable for GANs."""

    def __init__(self, dls: DataLoaders, generator: nn.Module, critic: nn.Module, gen_loss_func: callable, crit_loss_func: callable, switcher: Callback | None=None, gen_first: bool=False, switch_eval: bool=True, show_img: bool=True, clip: None | float=None, cbs: Callback | None | MutableSequence=None, metrics: None | MutableSequence | callable=None, **kwargs):
        if False:
            return 10
        gan = GANModule(generator, critic)
        loss_func = GANLoss(gen_loss_func, crit_loss_func, gan)
        if switcher is None:
            switcher = FixedGANSwitcher()
        trainer = GANTrainer(clip=clip, switch_eval=switch_eval, gen_first=gen_first, show_img=show_img)
        cbs = L(cbs) + L(trainer, switcher)
        metrics = L(metrics) + L(*LossMetrics('gen_loss,crit_loss'))
        super().__init__(dls, gan, loss_func=loss_func, cbs=cbs, metrics=metrics, **kwargs)

    @classmethod
    def from_learners(cls, gen_learn: Learner, crit_learn: Learner, switcher: Callback | None=None, weights_gen: None | MutableSequence | tuple=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Create a GAN from `learn_gen` and `learn_crit`.'
        losses = gan_loss_from_func(gen_learn.loss_func, crit_learn.loss_func, weights_gen=weights_gen)
        return cls(gen_learn.dls, gen_learn.model, crit_learn.model, *losses, switcher=switcher, **kwargs)

    @classmethod
    def wgan(cls, dls: DataLoaders, generator: nn.Module, critic: nn.Module, switcher: Callback | None=None, clip: None | float=0.01, switch_eval: bool=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Create a [WGAN](https://arxiv.org/abs/1701.07875) from `dls`, `generator` and `critic`.'
        if switcher is None:
            switcher = FixedGANSwitcher(n_crit=5, n_gen=1)
        return cls(dls, generator, critic, _tk_mean, _tk_diff, switcher=switcher, clip=clip, switch_eval=switch_eval, **kwargs)
GANLearner.from_learners = delegates(to=GANLearner.__init__)(GANLearner.from_learners)
GANLearner.wgan = delegates(to=GANLearner.__init__)(GANLearner.wgan)