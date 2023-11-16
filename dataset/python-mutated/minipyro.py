"""
Mini Pyro
---------

This file contains a minimal implementation of the Pyro Probabilistic
Programming Language. The API (method signatures, etc.) match that of
the full implementation as closely as possible. This file is independent
of the rest of Pyro, with the exception of the :mod:`pyro.distributions`
module.

An accompanying example that makes use of this implementation can be
found at examples/minipyro.py.
"""
import random
import warnings
import weakref
from collections import OrderedDict
import numpy as np
import torch
from pyro.distributions import validation_enabled
PYRO_STACK = []
PARAM_STORE = {}

def get_param_store():
    if False:
        i = 10
        return i + 15
    return PARAM_STORE

class Messenger:

    def __init__(self, fn=None):
        if False:
            print('Hello World!')
        self.fn = fn

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        PYRO_STACK.append(self)

    def __exit__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        assert PYRO_STACK[-1] is self
        PYRO_STACK.pop()

    def process_message(self, msg):
        if False:
            print('Hello World!')
        pass

    def postprocess_message(self, msg):
        if False:
            return 10
        pass

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        with self:
            return self.fn(*args, **kwargs)

class trace(Messenger):

    def __enter__(self):
        if False:
            return 10
        super().__enter__()
        self.trace = OrderedDict()
        return self.trace

    def postprocess_message(self, msg):
        if False:
            return 10
        assert msg['type'] != 'sample' or msg['name'] not in self.trace, 'sample sites must have unique names'
        self.trace[msg['name']] = msg.copy()

    def get_trace(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self(*args, **kwargs)
        return self.trace

class replay(Messenger):

    def __init__(self, fn, guide_trace):
        if False:
            print('Hello World!')
        self.guide_trace = guide_trace
        super().__init__(fn)

    def process_message(self, msg):
        if False:
            for i in range(10):
                print('nop')
        if msg['name'] in self.guide_trace:
            msg['value'] = self.guide_trace[msg['name']]['value']

class block(Messenger):

    def __init__(self, fn=None, hide_fn=lambda msg: True):
        if False:
            while True:
                i = 10
        self.hide_fn = hide_fn
        super().__init__(fn)

    def process_message(self, msg):
        if False:
            for i in range(10):
                print('nop')
        if self.hide_fn(msg):
            msg['stop'] = True

class seed(Messenger):

    def __init__(self, fn=None, rng_seed=None):
        if False:
            while True:
                i = 10
        self.rng_seed = rng_seed
        super().__init__(fn)

    def __enter__(self):
        if False:
            print('Hello World!')
        self.old_state = {'torch': torch.get_rng_state(), 'random': random.getstate(), 'numpy': np.random.get_state()}
        torch.manual_seed(self.rng_seed)
        random.seed(self.rng_seed)
        np.random.seed(self.rng_seed)

    def __exit__(self, type, value, traceback):
        if False:
            i = 10
            return i + 15
        torch.set_rng_state(self.old_state['torch'])
        random.setstate(self.old_state['random'])
        if 'numpy' in self.old_state:
            import numpy as np
            np.random.set_state(self.old_state['numpy'])

class PlateMessenger(Messenger):

    def __init__(self, fn, size, dim):
        if False:
            i = 10
            return i + 15
        assert dim < 0
        self.size = size
        self.dim = dim
        super().__init__(fn)

    def process_message(self, msg):
        if False:
            for i in range(10):
                print('nop')
        if msg['type'] == 'sample':
            batch_shape = msg['fn'].batch_shape
            if len(batch_shape) < -self.dim or batch_shape[self.dim] != self.size:
                batch_shape = [1] * (-self.dim - len(batch_shape)) + list(batch_shape)
                batch_shape[self.dim] = self.size
                msg['fn'] = msg['fn'].expand(torch.Size(batch_shape))

    def __iter__(self):
        if False:
            while True:
                i = 10
        return range(self.size)

def apply_stack(msg):
    if False:
        print('Hello World!')
    for (pointer, handler) in enumerate(reversed(PYRO_STACK)):
        handler.process_message(msg)
        if msg.get('stop'):
            break
    if msg['value'] is None:
        msg['value'] = msg['fn'](*msg['args'])
    for handler in PYRO_STACK[-pointer - 1:]:
        handler.postprocess_message(msg)
    return msg

def sample(name, fn, *args, **kwargs):
    if False:
        while True:
            i = 10
    obs = kwargs.pop('obs', None)
    if not PYRO_STACK:
        return fn(*args, **kwargs)
    initial_msg = {'type': 'sample', 'name': name, 'fn': fn, 'args': args, 'kwargs': kwargs, 'value': obs}
    msg = apply_stack(initial_msg)
    return msg['value']

def param(name, init_value=None, constraint=torch.distributions.constraints.real, event_dim=None):
    if False:
        while True:
            i = 10
    if event_dim is not None:
        raise NotImplementedError('minipyro.plate does not support the event_dim arg')

    def fn(init_value, constraint):
        if False:
            for i in range(10):
                print('nop')
        if name in PARAM_STORE:
            (unconstrained_value, constraint) = PARAM_STORE[name]
        else:
            assert init_value is not None
            with torch.no_grad():
                constrained_value = init_value.detach()
                unconstrained_value = torch.distributions.transform_to(constraint).inv(constrained_value)
            unconstrained_value.requires_grad_()
            PARAM_STORE[name] = (unconstrained_value, constraint)
        constrained_value = torch.distributions.transform_to(constraint)(unconstrained_value)
        constrained_value.unconstrained = weakref.ref(unconstrained_value)
        return constrained_value
    if not PYRO_STACK:
        return fn(init_value, constraint)
    initial_msg = {'type': 'param', 'name': name, 'fn': fn, 'args': (init_value, constraint), 'value': None}
    msg = apply_stack(initial_msg)
    return msg['value']

def plate(name, size, dim=None):
    if False:
        print('Hello World!')
    if dim is None:
        raise NotImplementedError('minipyro.plate requires a dim arg')
    return PlateMessenger(fn=None, size=size, dim=dim)

class Adam:

    def __init__(self, optim_args):
        if False:
            i = 10
            return i + 15
        self.optim_args = optim_args
        self.optim_objs = {}

    def __call__(self, params):
        if False:
            return 10
        for param in params:
            if param in self.optim_objs:
                optim = self.optim_objs[param]
            else:
                optim = torch.optim.Adam([param], **self.optim_args)
                self.optim_objs[param] = optim
            optim.step()

class SVI:

    def __init__(self, model, guide, optim, loss):
        if False:
            i = 10
            return i + 15
        self.model = model
        self.guide = guide
        self.optim = optim
        self.loss = loss

    def step(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        with trace() as param_capture:
            with block(hide_fn=lambda msg: msg['type'] == 'sample'):
                loss = self.loss(self.model, self.guide, *args, **kwargs)
        loss.backward()
        params = [site['value'].unconstrained() for site in param_capture.values()]
        self.optim(params)
        for p in params:
            p.grad = torch.zeros_like(p)
        return loss.item()

def elbo(model, guide, *args, **kwargs):
    if False:
        return 10
    guide_trace = trace(guide).get_trace(*args, **kwargs)
    model_trace = trace(replay(model, guide_trace)).get_trace(*args, **kwargs)
    elbo = 0.0
    for site in model_trace.values():
        if site['type'] == 'sample':
            elbo = elbo + site['fn'].log_prob(site['value']).sum()
    for site in guide_trace.values():
        if site['type'] == 'sample':
            elbo = elbo - site['fn'].log_prob(site['value']).sum()
    return -elbo

def Trace_ELBO(**kwargs):
    if False:
        i = 10
        return i + 15
    return elbo

class JitTrace_ELBO:

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        self.ignore_jit_warnings = kwargs.pop('ignore_jit_warnings', False)
        self._compiled = None
        self._param_trace = None

    def __call__(self, model, guide, *args):
        if False:
            i = 10
            return i + 15
        if self._param_trace is None:
            with block(), trace() as tr, block(hide_fn=lambda m: m['type'] != 'param'):
                elbo(model, guide, *args)
            self._param_trace = tr
        unconstrained_params = tuple((param(name).unconstrained() for name in self._param_trace))
        params_and_args = unconstrained_params + args
        if self._compiled is None:

            def compiled(*params_and_args):
                if False:
                    i = 10
                    return i + 15
                unconstrained_params = params_and_args[:len(self._param_trace)]
                args = params_and_args[len(self._param_trace):]
                for (name, unconstrained_param) in zip(self._param_trace, unconstrained_params):
                    constrained_param = param(name)
                    assert constrained_param.unconstrained() is unconstrained_param
                    self._param_trace[name]['value'] = constrained_param
                return replay(elbo, guide_trace=self._param_trace)(model, guide, *args)
            with validation_enabled(False), warnings.catch_warnings():
                if self.ignore_jit_warnings:
                    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
                self._compiled = torch.jit.trace(compiled, params_and_args, check_trace=False)
        return self._compiled(*params_and_args)