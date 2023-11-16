import warnings
from enum import auto, Enum
from functools import partial
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
import torch
import torch.nn as nn
from torch.autograd.graph import save_on_cpu
from torch.distributed.utils import _pack_kwargs, _replace_by_prefix, _unpack_kwargs
from torch.utils.checkpoint import checkpoint as torch_utils_checkpoint
_CHECKPOINT_WRAPPED_MODULE = '_checkpoint_wrapped_module'
_CHECKPOINT_PREFIX = _CHECKPOINT_WRAPPED_MODULE + '.'

class CheckpointImpl(Enum):
    REENTRANT = auto()
    NO_REENTRANT = auto()

class ActivationWrapper(torch.nn.Module):
    """
    Base class for Activation Checkpoint and Activation Offload.

    Not meant to be instantiated directly.
    """

    def __init__(self, mod):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._checkpoint_wrapped_module = mod
        self._register_state_dict_hook(self._post_state_dict_hook)
        self._register_load_state_dict_pre_hook(self._pre_load_state_dict_hook, with_module=True)

    def forward(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        raise ValueError('Subclasses should implement forward().')

    def __getattr__(self, name: str) -> Any:
        if False:
            return 10
        'Forward missing attributes to wrapped module.'
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._checkpoint_wrapped_module, name)

    def __getitem__(self, key: int) -> Any:
        if False:
            return 10
        'Forward indexing calls in case the module is a nn.Sequential.'
        return self._checkpoint_wrapped_module.__getitem__(key)

    def named_parameters(self, *args, **kwargs) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Override :meth:`named_parameters()` to intercept parameter names.\n\n        remove all occurrences of ``_CHECKPOINT_PREFIX``.\n        '
        for (param_name, param) in super().named_parameters(*args, **kwargs):
            yield (param_name.replace(_CHECKPOINT_PREFIX, ''), param)

    @staticmethod
    def _post_state_dict_hook(module: nn.Module, state_dict: Dict[str, Any], prefix: str, *args: Any) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        '\n        _post_state_dict_hook() is called after the state_dict() of this FSDP module is executed.\n\n        For ``checkpoint_wrapper``, it will strip checkpoint-wrapped module prefix,\n        so that this module can be loaded into non-checkpointed modules.\n        It would still be able to be loaded into checkpoint-wrapped modules as this class,\n        adds the prefix back before loading the state_dict.\n        '
        _replace_by_prefix(state_dict, f'{prefix}{_CHECKPOINT_PREFIX}', prefix)
        return state_dict

    @staticmethod
    def _pre_load_state_dict_hook(module: nn.Module, state_dict: Dict[str, Any], prefix: str, *args: Any) -> None:
        if False:
            while True:
                i = 10
        '\n        ``_pre_state_dict_hook` is called before ``self._load_from_state_dict()`` is called.\n\n        For ``checkpoint_wrapper``, it will add back the module\n        prefix so that non-checkpointed modules can be loaded into\n        checkpoint_wrapper modules properly.\n        '
        _replace_by_prefix(state_dict, prefix, prefix + f'{_CHECKPOINT_PREFIX}')

class OffloadWrapper(ActivationWrapper):

    def __init__(self, mod):
        if False:
            return 10
        super().__init__(mod)

    def forward(self, *args, **kwargs):
        if False:
            return 10
        with save_on_cpu(pin_memory=True):
            return self._checkpoint_wrapped_module(*args, **kwargs)

class CheckpointWrapper(ActivationWrapper):
    """
    An ``nn.Module`` that wraps another ``nn.Module`` with checkpointing.

    Note that this module is not meant to be used directly but instead,
    it is to be used through the ``checkpoint_wrapper`` function.
    """

    def __init__(self, mod: torch.nn.Module, checkpoint_impl: CheckpointImpl=CheckpointImpl.NO_REENTRANT, checkpoint_fn=None, **checkpoint_fn_kwargs):
        if False:
            while True:
                i = 10
        super().__init__(mod)
        self.checkpoint_impl = checkpoint_impl
        if checkpoint_fn is None:
            self.checkpoint_fn = partial(torch_utils_checkpoint, use_reentrant=self.checkpoint_impl == CheckpointImpl.REENTRANT, **checkpoint_fn_kwargs)
        else:
            self.checkpoint_fn = partial(checkpoint_fn, **checkpoint_fn_kwargs)

    def forward(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if self.checkpoint_impl == CheckpointImpl.REENTRANT and kwargs != {}:
            (flat_args, kwarg_keys) = _pack_kwargs(*args, **kwargs)

            def my_function(*inputs):
                if False:
                    while True:
                        i = 10
                (unpacked_args, unpacked_kwargs) = _unpack_kwargs(inputs, kwarg_keys)
                return self._checkpoint_wrapped_module(*unpacked_args, **unpacked_kwargs)
            return self.checkpoint_fn(my_function, *flat_args)
        else:
            return self.checkpoint_fn(self._checkpoint_wrapped_module, *args, **kwargs)

def offload_wrapper(module: torch.nn.Module) -> torch.nn.Module:
    if False:
        print('Hello World!')
    '\n    Wrap a module for activation offloading to CPU.\n\n    Offloads intermediate activations to the CPU for modules wrapped with this function.\n    Wrappers with activation offload can be composed with ones that do recomputation-based\n    checkpoint to trade off increased compute versus increased CPU\n    memory usage and additional H2D transfers.\n\n    Usage::\n        offloaded_module = offload_wrapper(module)\n        outputs = checkpointed_module(inputs)\n    Args:\n        module (nn.Module):\n            The module to be wrapped\n    Returns:\n        (nn.Module):\n            Wrapped module\n    '
    return OffloadWrapper(module)

def checkpoint_wrapper(module: torch.nn.Module, checkpoint_impl: CheckpointImpl=CheckpointImpl.NO_REENTRANT, checkpoint_fn=None, **checkpoint_fn_kwargs) -> torch.nn.Module:
    if False:
        print('Hello World!')
    '\n    Wrap a module for activation checkpointing.\n\n    If the module is wrapped with this function, all subsequent calls to the module will,\n    automatically perform checkpointing without the user having to explicitly call ``checkpoint`` function.\n\n    Usage::\n        checkpointed_module = checkpoint_wrapper(module)\n        outputs = checkpointed_module(inputs)\n    Args:\n        module (nn.Module):\n            The module to be wrapped\n        checkpoint_impl (Optional[CheckpointImpl]):\n            The checkpointing implementation to use. Note that this will only\n            be passed into the ``torch.utils.checkpoint.checkpoint``\n            implementation, and is ignored if a custom ``checkpoint_fn`` is\n            specified. Note that for implementations using reentrant checkpoint\n            from ``torch.utils.checkpoint``, keyword arguments will only be\n            supported if ``checkpoint_impl`` is passed as ``CheckpointImpl.REENTRANT`.\n        checkpoint_fn (Optional[Callable]):\n            Functional checkpoint implementation to use. If this is specified,\n            it will be used over the default ``torch.utils.checkpoint.checkpoint``\n            implementation and the `checkpoint_impl` argument will be ignored.\n        **checkpoint_fn_kwargs: (Dict[str, Any]): Keyword arguments to pass into `checkpoint_fn`.\n\n    Returns:\n        (nn.Module):\n            Wrapped module\n    '
    if checkpoint_impl == CheckpointImpl.REENTRANT:
        warnings.warn(f'Please specify {CheckpointImpl.NO_REENTRANT} as {CheckpointImpl.REENTRANT} will soon be removed as the default and eventually deprecated.', stacklevel=1)
    return CheckpointWrapper(module, checkpoint_impl, checkpoint_fn, **checkpoint_fn_kwargs)

def apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=lambda _: True, auto_wrap_policy: Optional[Callable[[nn.Module, bool, int], bool]]=None):
    if False:
        while True:
            i = 10
    "\n    Apply :func:`checkpoint_wrapper` to modules within `model` based on a user-defined configuration.\n\n    For each module within `model`, the `check_fn` is used to decide\n    whether `module` should be wrapped with :func:`checkpoint_wrapper` or not.\n\n    Note::\n        This function modifies `model` in place and replaces appropriate layers with\n        their checkpoint-wrapped modules.\n    Note::\n        This function will not wrap the overall root module. If this is needed, please directly use\n        :func:`checkpoint_wrapper` or :func:`offload_wrapper`.\n    Usage::\n        model = nn.Sequential(\n            nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10)\n        )\n        check_fn = lambda l: isinstance(l, nn.Linear)\n        # checkpoint activations\n        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)\n        # Or offload activations to CPU\n        apply_activation_checkpointing(model, checkpoint_wrapper_fn=offload_wrapper, check_fn=check_fn)\n    Args:\n        model (nn.Module):\n            The model whose submodules should be wrapped with activation checkpointing.\n        checkpoint_wrapper_fn (Optional[Callable[nn.Module]])\n            A ``Callable`` which will wrap modules\n        check_fn (Optional[Callable[nn.Module, nn.Module]])\n            A lambda function which will be passed each child submodule of ``model`` and returns\n            ``True`` or ``False`` depending on whether the submodule should be wrapped.\n        auto_wrap_policy (Optional[Callable[[nn.Module, bool, int], bool]]): A policy to wrap model's\n            submodules with AC. Note that if this is specified, it takes precedence over ``check_fn``.\n    Returns: None (`model` is modified inplace)\n    "
    from torch.distributed.fsdp.wrap import _recursive_wrap, lambda_auto_wrap_policy, _Policy
    from torch.distributed.fsdp._wrap_utils import _construct_wrap_fn, _post_order_apply
    policy = auto_wrap_policy if auto_wrap_policy is not None else partial(lambda_auto_wrap_policy, lambda_fn=check_fn)
    if not callable(policy):
        if not isinstance(policy, _Policy):
            raise ValueError(f'Expected {policy} to be callable or be a pre-defined wrap policy')
        target_module_to_kwargs = policy._run_policy(model, ignored_modules=set(), root_kwargs={})
        wrap_fn = _construct_wrap_fn(model, target_module_to_kwargs, checkpoint_wrapper_fn)
        _post_order_apply(model, wrap_fn)
        return
    _recursive_wrap(module=model, auto_wrap_policy=policy, wrapper_cls=checkpoint_wrapper_fn, ignored_modules=set(), ignored_params=set(), only_wrap_children=True)