from __future__ import annotations
from ..basics import *
__all__ = ['Hook', 'hook_output', 'Hooks', 'hook_outputs', 'dummy_eval', 'model_sizes', 'num_features_model', 'has_params', 'HookCallback', 'total_params', 'layer_info', 'module_summary', 'ActivationStats']

@docs
class Hook:
    """Create a hook on `m` with `hook_func`."""

    def __init__(self, m, hook_func, is_forward=True, detach=True, cpu=False, gather=False):
        if False:
            while True:
                i = 10
        store_attr('hook_func,detach,cpu,gather')
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        (self.stored, self.removed) = (None, False)

    def hook_fn(self, module, input, output):
        if False:
            return 10
        'Applies `hook_func` to `module`, `input`, `output`.'
        if self.detach:
            (input, output) = (to_detach(input, cpu=self.cpu, gather=self.gather), to_detach(output, cpu=self.cpu, gather=self.gather))
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        if False:
            while True:
                i = 10
        'Remove the hook from the model.'
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        if False:
            return 10
        return self

    def __exit__(self, *args):
        if False:
            while True:
                i = 10
        self.remove()
    _docs = dict(__enter__='Register the hook', __exit__='Remove the hook')

def _hook_inner(m, i, o):
    if False:
        i = 10
        return i + 15
    return o if isinstance(o, Tensor) or is_listy(o) else list(o)

def hook_output(module, detach=True, cpu=False, grad=False):
    if False:
        print('Hello World!')
    'Return a `Hook` that stores activations of `module` in `self.stored`'
    return Hook(module, _hook_inner, detach=detach, cpu=cpu, is_forward=not grad)

@docs
class Hooks:
    """Create several hooks on the modules in `ms` with `hook_func`."""

    def __init__(self, ms, hook_func, is_forward=True, detach=True, cpu=False):
        if False:
            while True:
                i = 10
        self.hooks = [Hook(m, hook_func, is_forward, detach, cpu) for m in ms]

    def __getitem__(self, i):
        if False:
            for i in range(10):
                print('nop')
        return self.hooks[i]

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.hooks)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self.hooks)

    @property
    def stored(self):
        if False:
            i = 10
            return i + 15
        return L((o.stored for o in self))

    def remove(self):
        if False:
            i = 10
            return i + 15
        'Remove the hooks from the model.'
        for h in self.hooks:
            h.remove()

    def __enter__(self, *args):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, *args):
        if False:
            print('Hello World!')
        self.remove()
    _docs = dict(stored='The states saved in each hook.', __enter__='Register the hooks', __exit__='Remove the hooks')

def hook_outputs(modules, detach=True, cpu=False, grad=False):
    if False:
        return 10
    'Return `Hooks` that store activations of all `modules` in `self.stored`'
    return Hooks(modules, _hook_inner, detach=detach, cpu=cpu, is_forward=not grad)

def dummy_eval(m, size=(64, 64)):
    if False:
        while True:
            i = 10
    'Evaluate `m` on a dummy input of a certain `size`'
    ch_in = in_channels(m)
    x = one_param(m).new(1, ch_in, *size).requires_grad_(False).uniform_(-1.0, 1.0)
    with torch.no_grad():
        return m.eval()(x)

def model_sizes(m, size=(64, 64)):
    if False:
        return 10
    'Pass a dummy input through the model `m` to get the various sizes of activations.'
    with hook_outputs(m) as hooks:
        _ = dummy_eval(m, size=size)
        return [o.stored.shape for o in hooks]

def num_features_model(m):
    if False:
        for i in range(10):
            print('nop')
    'Return the number of output features for `m`.'
    (sz, ch_in) = (32, in_channels(m))
    while True:
        try:
            return model_sizes(m, (sz, sz))[-1][1]
        except Exception as e:
            sz *= 2
            if sz > 2048:
                raise e

def has_params(m):
    if False:
        return 10
    'Check if `m` has at least one parameter'
    return len(list(m.parameters())) > 0

@funcs_kwargs
class HookCallback(Callback):
    """`Callback` that can be used to register hooks on `modules`"""
    _methods = ['hook']
    hook = noops

    def __init__(self, modules=None, every=None, remove_end=True, is_forward=True, detach=True, cpu=True, include_paramless=False, **kwargs):
        if False:
            while True:
                i = 10
        store_attr('modules,every,remove_end,is_forward,detach,cpu, include_paramless')
        assert not kwargs

    def before_fit(self):
        if False:
            while True:
                i = 10
        'Register the `Hooks` on `self.modules`.'
        if self.modules is None:
            self.modules = [m for m in flatten_model(self.model) if self.include_paramless or has_params(m)]
        if self.every is None:
            self._register()

    def before_batch(self):
        if False:
            while True:
                i = 10
        if self.every is None:
            return
        if self.training and self.train_iter % self.every == 0:
            self._register()

    def after_batch(self):
        if False:
            return 10
        if self.every is None:
            return
        if self.training and self.train_iter % self.every == 0:
            self._remove()

    def after_fit(self):
        if False:
            print('Hello World!')
        'Remove the `Hooks`.'
        if self.remove_end:
            self._remove()

    def _register(self):
        if False:
            for i in range(10):
                print('nop')
        self.hooks = Hooks(self.modules, self.hook, self.is_forward, self.detach, self.cpu)

    def _remove(self):
        if False:
            return 10
        if getattr(self, 'hooks', None):
            self.hooks.remove()

    def __del__(self):
        if False:
            print('Hello World!')
        self._remove()

def total_params(m):
    if False:
        print('Hello World!')
    "Give the number of parameters of a module and if it's trainable or not"
    params = sum([p.numel() for p in m.parameters()])
    trains = [p.requires_grad for p in m.parameters()]
    return (params, False if len(trains) == 0 else trains[0])

def layer_info(learn, *xb):
    if False:
        while True:
            i = 10
    'Return layer infos of `model` on `xb` (only support batch first inputs)'

    def _track(m, i, o):
        if False:
            return 10
        (params, trainable, shape) = ('', '', '')
        same = any((isinstance(x[0], torch.Tensor) and x[0].shape[1:] == x[1].shape for x in zip(i, o)))
        shape = apply(lambda x: x.shape, o)
        if hasattr(m, 'weight'):
            (params, trainable) = total_params(m)
        return (type(m).__name__, params, trainable, shape, same)
    with Hooks(flatten_model(learn.model), _track) as h:
        batch = apply(lambda o: o[:1], xb)
        train_only_cbs = [cb for cb in learn.cbs if hasattr(cb, '_only_train_loop')]
        with learn.removed_cbs(train_only_cbs), learn.no_logging(), learn as l:
            r = l.get_preds(dl=[batch], inner=True, reorder=False)
        return h.stored

def _get_shapes(o, bs):
    if False:
        print('Hello World!')
    inp = o[first(o)] if isinstance(o, dict) else o
    return ' x '.join([str(bs)] + [str(t) for t in inp[1:]])

def _print_shapes(o, bs):
    if False:
        while True:
            i = 10
    if isinstance(o, torch.Size):
        return _get_shapes(o, bs)
    elif isinstance(o, tuple):
        return _get_shapes(o[0], bs)
    else:
        return str([_print_shapes(x, bs) for x in o])

def module_summary(learn, *xb):
    if False:
        return 10
    'Print a summary of `model` using `xb`'
    infos = layer_info(learn, *xb)
    (n, bs) = (76, find_bs(xb))
    inp_sz = _print_shapes(apply(lambda x: x.shape, xb), bs)
    res = f'{type(learn.model).__name__} (Input shape: {inp_sz})\n'
    res += '=' * n + '\n'
    res += f"{'Layer (type)':<20} {'Output Shape':<20} {'Param #':<10} {'Trainable':<10}\n"
    res += '=' * n
    (ps, trn_ps, j) = (0, 0, 0)
    infos = [o for o in infos if o is not None]
    prev_sz = None
    for (typ, np, trn, sz, chnged) in infos:
        if sz is None:
            continue
        if j == 0:
            res += f"\n{'':<20} {_print_shapes(sz, bs)[:19]:<20}"
        if not chnged and (not prev_sz == sz) and (j > 0):
            res += '\n' + '_' * n + '\n' + f"{'':<20} {_print_shapes(sz, bs)[:19]:<20}"
        j = 1
        res += f"\n{typ:<20} {'':<20} {np:<10} {str(trn):<10}"
        if np != '':
            ps += np
            if trn:
                trn_ps += np
        prev_sz = sz
    res += '\n' + '_' * n + '\n'
    res += f'\nTotal params: {ps:,}\n'
    res += f'Total trainable params: {trn_ps:,}\n'
    res += f'Total non-trainable params: {ps - trn_ps:,}\n\n'
    return PrettyString(res)

@patch
def summary(self: Learner):
    if False:
        return 10
    'Print a summary of the model, optimizer and loss function.'
    xb = self.dls.train.one_batch()[:getattr(self.dls.train, 'n_inp', 1)]
    res = module_summary(self, *xb)
    res += f'Optimizer used: {self.opt_func}\nLoss function: {self.loss_func}\n\n'
    if self.opt is not None:
        res += f'Model ' + ('unfrozen\n\n' if self.opt.frozen_idx == 0 else f'frozen up to parameter group #{self.opt.frozen_idx}\n\n')
    res += 'Callbacks:\n' + '\n'.join((f'  - {cb}' for cb in self.cbs.sorted('order')))
    return PrettyString(res)

@delegates()
class ActivationStats(HookCallback):
    """Callback that record the mean and std of activations."""
    order = -20

    def __init__(self, with_hist=False, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.with_hist = with_hist

    def before_fit(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialize stats.'
        super().before_fit()
        self.stats = L()

    def hook(self, m, i, o):
        if False:
            while True:
                i = 10
        if isinstance(o, tuple):
            return self.hook_multi_ouput(o)
        o = o.float()
        res = {'mean': o.mean().item(), 'std': o.std().item(), 'near_zero': (o <= 0.05).long().sum().item() / o.numel()}
        if self.with_hist:
            res['hist'] = o.histc(40, 0, 10)
        return res

    def hook_multi_ouput(self, o_tuple):
        if False:
            for i in range(10):
                print('nop')
        'For outputs of RNN which are [nested] tuples of tensors'
        res = []
        for o in self._flatten_tuple(o_tuple):
            if not isinstance(o, Tensor):
                continue
            res.append(self.hook(None, None, o))
        return res

    def _flatten_tuple(self, o_tuple):
        if False:
            i = 10
            return i + 15
        'Recursively flatten a [nested] tuple'
        res = []
        for it in o_tuple:
            if isinstance(it, tuple):
                res += self._flatten_tuple(it)
            else:
                res += [it]
        return tuple(res)

    def after_batch(self):
        if False:
            print('Hello World!')
        'Take the stored results and puts it in `self.stats`'
        if self.training and (self.every is None or self.train_iter % self.every == 0):
            self.stats.append(self.hooks.stored)
        super().after_batch()

    def layer_stats(self, idx):
        if False:
            i = 10
            return i + 15
        lstats = self.stats.itemgot(idx)
        return L((lstats.itemgot(o) for o in ('mean', 'std', 'near_zero')))

    def hist(self, idx):
        if False:
            while True:
                i = 10
        res = self.stats.itemgot(idx).itemgot('hist')
        return torch.stack(tuple(res)).t().float().log1p()

    def color_dim(self, idx, figsize=(10, 5), ax=None):
        if False:
            return 10
        "The 'colorful dimension' plot"
        res = self.hist(idx)
        if ax is None:
            ax = subplots(figsize=figsize)[1][0]
        ax.imshow(res, origin='lower')
        ax.axis('off')

    def plot_layer_stats(self, idx):
        if False:
            return 10
        (_, axs) = subplots(1, 3, figsize=(12, 3))
        for (o, ax, title) in zip(self.layer_stats(idx), axs, ('mean', 'std', '% near zero')):
            ax.plot(o)
            ax.set_title(title)