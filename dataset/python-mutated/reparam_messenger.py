import warnings
from typing import Callable, Dict, Union
import torch
from .messenger import Messenger
from .runtime import effectful

@effectful(type='get_init_messengers')
def _get_init_messengers():
    if False:
        return 10
    return []

class ReparamMessenger(Messenger):
    """
    Reparametrizes each affected sample site into one or more auxiliary sample
    sites followed by a deterministic transformation [1].

    To specify reparameterizers, pass a ``config`` dict or callable to the
    constructor.  See the :mod:`pyro.infer.reparam` module for available
    reparameterizers.

    Note some reparameterizers can examine the ``*args,**kwargs`` inputs of
    functions they affect; these reparameterizers require using
    ``poutine.reparam`` as a decorator rather than as a context manager.

    [1] Maria I. Gorinova, Dave Moore, Matthew D. Hoffman (2019)
        "Automatic Reparameterisation of Probabilistic Programs"
        https://arxiv.org/pdf/1906.03028.pdf

    :param config: Configuration, either a dict mapping site name to
        :class:`~pyro.infer.reparam.reparam.Reparameterizer` , or a function
        mapping site to :class:`~pyro.infer.reparam.reparam.Reparameterizer` or
        None. See :mod:`pyro.infer.reparam.strategies` for built-in
        configuration strategies.
    :type config: dict or callable
    """

    def __init__(self, config: Union[Dict[str, object], Callable]):
        if False:
            while True:
                i = 10
        super().__init__()
        assert isinstance(config, dict) or callable(config)
        self.config = config
        self._args_kwargs = None

    def __call__(self, fn):
        if False:
            return 10
        return ReparamHandler(self, fn)

    def _pyro_sample(self, msg):
        if False:
            print('Hello World!')
        if type(msg['fn']).__name__ == '_Subsample':
            return
        if isinstance(self.config, dict):
            reparam = self.config.get(msg['name'])
        else:
            reparam = self.config(msg)
        if reparam is None:
            return
        for m in _get_init_messengers():
            m._pyro_sample(msg)
        reparam.args_kwargs = self._args_kwargs
        try:
            new_msg = reparam.apply({'name': msg['name'], 'fn': msg['fn'], 'value': msg['value'], 'is_observed': msg['is_observed']})
        finally:
            reparam.args_kwargs = None
        if new_msg['value'] is not None:
            if getattr(msg['fn'], '_validation_enabled', False):
                msg['fn']._validate_sample(new_msg['value'])
            if msg['value'] is not None and msg['value'] is not new_msg['value']:
                if not torch._C._get_tracing_state():
                    assert new_msg['value'].shape == msg['value'].shape
                if getattr(msg['value'], '_pyro_custom_init', True):
                    warnings.warn(f"At pyro.sample({repr(msg['name'])},...), {type(reparam).__name__} does not commute with initialization; falling back to default initialization.", RuntimeWarning)
        msg['fn'] = new_msg['fn']
        msg['value'] = new_msg['value']
        msg['is_observed'] = new_msg['is_observed']

class ReparamHandler(object):
    """
    Reparameterization poutine.
    """

    def __init__(self, msngr, fn):
        if False:
            for i in range(10):
                print('nop')
        self.msngr = msngr
        self.fn = fn
        super().__init__()

    def __call__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.msngr._args_kwargs = (args, kwargs)
        try:
            with self.msngr:
                return self.fn(*args, **kwargs)
        finally:
            self.msngr._args_kwargs = None