import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union, ValuesView
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import Optimizer
import pyro
from pyro.optim.adagrad_rmsprop import AdagradRMSProp as pt_AdagradRMSProp
from pyro.optim.clipped_adam import ClippedAdam as pt_ClippedAdam
from pyro.optim.dct_adam import DCTAdam as pt_DCTAdam
from pyro.params.param_store import module_from_param_with_module_name, normalize_param_name, user_param_name

def is_scheduler(optimizer) -> bool:
    if False:
        return 10
    '\n    Helper method to determine whether a PyTorch object is either a PyTorch\n    optimizer (return false) or a optimizer wrapped in an LRScheduler e.g. a\n    ``ReduceLROnPlateau`` or subclasses of ``_LRScheduler`` (return true).\n    '
    return hasattr(optimizer, 'optimizer')

def _get_state_dict(optimizer) -> dict:
    if False:
        while True:
            i = 10
    '\n    Helper to get the state dict for either a raw optimizer or an optimizer\n    wrapped in an LRScheduler.\n    '
    if is_scheduler(optimizer):
        state = {'scheduler': optimizer.state_dict(), 'optimizer': optimizer.optimizer.state_dict()}
    else:
        state = optimizer.state_dict()
    return state

def _load_state_dict(optimizer, state: dict) -> None:
    if False:
        while True:
            i = 10
    '\n    Helper to load the state dict into either a raw optimizer or an optimizer\n    wrapped in an LRScheduler.\n    '
    if is_scheduler(optimizer):
        optimizer.load_state_dict(state['scheduler'])
        optimizer.optimizer.load_state_dict(state['optimizer'])
    else:
        optimizer.load_state_dict(state)

class PyroOptim:
    """
    A wrapper for torch.optim.Optimizer objects that helps with managing dynamically generated parameters.

    :param optim_constructor: a torch.optim.Optimizer
    :param optim_args: a dictionary of learning arguments for the optimizer or a callable that returns
        such dictionaries
    :param clip_args: a dictionary of clip_norm and/or clip_value args or a callable that returns
        such dictionaries
    """

    def __init__(self, optim_constructor: Union[Callable, Optimizer, Type[Optimizer]], optim_args: Union[Dict, Callable[..., Dict]], clip_args: Optional[Union[Dict, Callable[..., Dict]]]=None):
        if False:
            while True:
                i = 10
        self.pt_optim_constructor = optim_constructor
        assert callable(optim_args) or isinstance(optim_args, dict), 'optim_args must be function that returns defaults or a defaults dictionary'
        if clip_args is None:
            clip_args = {}
        assert callable(clip_args) or isinstance(clip_args, dict), 'clip_args must be function that returns defaults or a defaults dictionary'
        self.pt_optim_args = optim_args
        if callable(optim_args):
            self.pt_optim_args_argc = len(inspect.signature(optim_args).parameters)
        self.pt_clip_args = clip_args
        self.optim_objs: Dict = {}
        self.grad_clip: Dict = {}
        self._state_waiting_to_be_consumed: Dict = {}

    def __call__(self, params: Union[List, ValuesView], *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        :param params: a list of parameters\n        :type params: an iterable of strings\n\n        Do an optimization step for each param in params. If a given param has never been seen before,\n        initialize an optimizer for it.\n        '
        for p in params:
            if p not in self.optim_objs:
                optimizer = self.optim_objs[p] = self._get_optim(p)
                self.grad_clip[p] = self._get_grad_clip(p)
                param_name = pyro.get_param_store().param_name(p)
                state = self._state_waiting_to_be_consumed.pop(param_name, None)
                if state is not None:
                    _load_state_dict(optimizer, state)
            if self.grad_clip[p] is not None:
                self.grad_clip[p](p)
            if hasattr(torch.optim.lr_scheduler, '_LRScheduler') and isinstance(self.optim_objs[p], torch.optim.lr_scheduler._LRScheduler) or (hasattr(torch.optim.lr_scheduler, 'LRScheduler') and isinstance(self.optim_objs[p], torch.optim.lr_scheduler.LRScheduler)) or isinstance(self.optim_objs[p], torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.optim_objs[p].optimizer.step(*args, **kwargs)
            else:
                self.optim_objs[p].step(*args, **kwargs)

    def get_state(self) -> Dict:
        if False:
            print('Hello World!')
        '\n        Get state associated with all the optimizers in the form of a dictionary with\n        key-value pairs (parameter name, optim state dicts)\n        '
        state_dict = {}
        for param in self.optim_objs:
            param_name = pyro.get_param_store().param_name(param)
            state_dict[param_name] = _get_state_dict(self.optim_objs[param])
        return state_dict

    def set_state(self, state_dict: Dict) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Set the state associated with all the optimizers using the state obtained\n        from a previous call to get_state()\n        '
        self._state_waiting_to_be_consumed.update(state_dict)

    def save(self, filename: str) -> None:
        if False:
            print('Hello World!')
        '\n        :param filename: file name to save to\n        :type filename: str\n\n        Save optimizer state to disk\n        '
        with open(filename, 'wb') as output_file:
            torch.save(self.get_state(), output_file)

    def load(self, filename: str, map_location=None) -> None:
        if False:
            while True:
                i = 10
        '\n        :param filename: file name to load from\n        :type filename: str\n        :param map_location: torch.load() map_location parameter\n        :type map_location: function, torch.device, string or a dict\n\n        Load optimizer state from disk\n        '
        with open(filename, 'rb') as input_file:
            state = torch.load(input_file, map_location=map_location)
        self.set_state(state)

    def _get_optim(self, param: Union[Iterable[Tensor], Iterable[Dict[Any, Any]]]):
        if False:
            while True:
                i = 10
        return self.pt_optim_constructor([param], **self._get_optim_args(param))

    def _get_optim_args(self, param: Union[Iterable[Tensor], Iterable[Dict]]):
        if False:
            for i in range(10):
                print('nop')
        if callable(self.pt_optim_args):
            param_name = pyro.get_param_store().param_name(param)
            if self.pt_optim_args_argc == 1:
                normal_name = normalize_param_name(param_name)
                opt_dict = self.pt_optim_args(normal_name)
            else:
                module_name = module_from_param_with_module_name(param_name)
                stripped_param_name = user_param_name(param_name)
                opt_dict = self.pt_optim_args(module_name, stripped_param_name)
            assert isinstance(opt_dict, dict), 'per-param optim arg must return defaults dictionary'
            return opt_dict
        else:
            return self.pt_optim_args

    def _get_grad_clip(self, param: str):
        if False:
            while True:
                i = 10
        grad_clip_args = self._get_grad_clip_args(param)
        if not grad_clip_args:
            return None

        def _clip_grad(params: Union[Tensor, Iterable[Tensor]]):
            if False:
                while True:
                    i = 10
            self._clip_grad(params, **grad_clip_args)
        return _clip_grad

    def _get_grad_clip_args(self, param: str) -> Dict:
        if False:
            print('Hello World!')
        if callable(self.pt_clip_args):
            param_name = pyro.get_param_store().param_name(param)
            module_name = module_from_param_with_module_name(param_name)
            stripped_param_name = user_param_name(param_name)
            clip_dict = self.pt_clip_args(module_name, stripped_param_name)
            assert isinstance(clip_dict, dict), 'per-param clip arg must return defaults dictionary'
            return clip_dict
        else:
            return self.pt_clip_args

    @staticmethod
    def _clip_grad(params: Union[Tensor, Iterable[Tensor]], clip_norm: Optional[Union[int, float]]=None, clip_value: Optional[Union[int, float]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        if clip_norm is not None:
            clip_grad_norm_(params, clip_norm)
        if clip_value is not None:
            clip_grad_value_(params, clip_value)

def AdagradRMSProp(optim_args: Dict) -> PyroOptim:
    if False:
        for i in range(10):
            print('nop')
    '\n    Wraps :class:`pyro.optim.adagrad_rmsprop.AdagradRMSProp` with :class:`~pyro.optim.optim.PyroOptim`.\n    '
    return PyroOptim(pt_AdagradRMSProp, optim_args)

def ClippedAdam(optim_args: Dict) -> PyroOptim:
    if False:
        return 10
    '\n    Wraps :class:`pyro.optim.clipped_adam.ClippedAdam` with :class:`~pyro.optim.optim.PyroOptim`.\n    '
    return PyroOptim(pt_ClippedAdam, optim_args)

def DCTAdam(optim_args: Dict) -> PyroOptim:
    if False:
        return 10
    '\n    Wraps :class:`pyro.optim.dct_adam.DCTAdam` with :class:`~pyro.optim.optim.PyroOptim`.\n    '
    return PyroOptim(pt_DCTAdam, optim_args)