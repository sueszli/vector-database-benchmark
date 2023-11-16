import collections
import functools
import inspect
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Set, Tuple, Type, Union
import torch.nn as nn
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state, _override_module_mixed_precision
from torch.distributed.fsdp.wrap import _construct_wrap_fn, _or_policy, _Policy, _post_order_apply, _recursive_wrap, _run_mixed_precision_override_policy, _wrap_module_cls_individually

def _auto_wrap(root_module: nn.Module, policy: Union[Callable, _Policy], ignored_modules: Set[nn.Module], ignored_params: Set[nn.Parameter], root_kwargs: Dict[str, Any], fsdp_fn: Callable):
    if False:
        return 10
    "\n    Auto wraps modules in ``root_module`` 's tree according to ``policy``\n    following a post-order traversal.\n\n    Precondition: ``root_kwargs`` should contain all arguments except\n    ``module``. This function accepts the kwargs dict directly since it gets\n    forwarded into the post-order traversal function.\n    "
    mixed_precision = root_kwargs['mixed_precision']
    is_wrapper = inspect.isclass(fsdp_fn)
    _check_nested_wrapping(root_module)
    if isinstance(policy, _Policy):
        root_kwargs['auto_wrap_policy' if is_wrapper else 'policy'] = None
        target_module_to_kwargs = policy._run_policy(root_module, ignored_modules, root_kwargs)
        if mixed_precision is not None:
            target_module_to_kwargs = _run_mixed_precision_override_policy(root_module, mixed_precision._module_classes_to_ignore, ignored_modules, root_kwargs, target_module_to_kwargs)
            overridden_module_classes = _override_module_mixed_precision(root_module, mixed_precision._module_classes_to_ignore)
            _warn_on_overridden_mixed_precision(overridden_module_classes)
        use_orig_params = root_kwargs.get('use_orig_params', False)
        _validate_frozen_params(root_module, set(target_module_to_kwargs.keys()), ignored_params, use_orig_params)
        wrap_fn = _construct_wrap_fn(root_module, target_module_to_kwargs, fsdp_fn)
        _post_order_apply(root_module, wrap_fn)
        return
    recursive_wrap_kwargs = {'module': root_module, 'auto_wrap_policy': policy, 'wrapper_cls': fsdp_fn, 'ignored_modules': ignored_modules, 'ignored_params': ignored_params, 'only_wrap_children': True}
    if mixed_precision is not None:
        overridden_module_classes = _override_module_mixed_precision(root_module, mixed_precision._module_classes_to_ignore)
        policy = functools.partial(_or_policy, policies=[policy, partial(_wrap_module_cls_individually, module_classes=mixed_precision._module_classes_to_ignore)])
        recursive_wrap_kwargs['auto_wrap_policy'] = policy
        _warn_on_overridden_mixed_precision(overridden_module_classes)
    _recursive_wrap(**recursive_wrap_kwargs, **root_kwargs)

def _check_nested_wrapping(root_module: nn.Module):
    if False:
        while True:
            i = 10
    for (module_name, module) in root_module.named_modules():
        if _get_module_fsdp_state(module) is not None:
            raise ValueError(f'FSDP auto wrapping requires modules to not already have FSDP applied but found {module_name} in\n{root_module}')

def _warn_on_overridden_mixed_precision(overridden_module_classes: Set[Type[nn.Module]]):
    if False:
        while True:
            i = 10
    if len(overridden_module_classes) == 0:
        return
    warnings.warn(f'Both mixed precision and an auto_wrap_policy were specified to FSDP, where the wrapped module has submodules of type:\n{overridden_module_classes}\nThese modules will be wrapped as separate FSDP instacnes with mixed precision disabled.')

def _validate_frozen_params(root_module: nn.Module, modules_to_wrap: Set[nn.Module], ignored_params: Set[nn.Parameter], use_orig_params: bool):
    if False:
        i = 10
        return i + 15
    '\n    This checks that, given ``modules_to_wrap``, each module would manage\n    parameters that are uniformly frozen or non-frozen. This uniformity\n    requirement is strict for ``use_orig_params=False`` (hard error) and highly\n    recommended for ``use_orig_params=True`` (user warning).\n    '
    post_order_named_modules = _get_post_order_named_modules(root_module)
    visited_modules: Set[nn.Module] = set()
    for (module_name, module) in post_order_named_modules:
        if module in modules_to_wrap:
            param_to_fqn = _get_managed_param_to_fqn(module, ignored_params, visited_modules, module_name)
            frozen_param_fqns: List[str] = []
            frozen_param_numel = 0
            nonfrozen_param_fqns: List[str] = []
            nonfrozen_param_numel = 0
            for (param, fqn) in param_to_fqn.items():
                if param.requires_grad:
                    nonfrozen_param_fqns.append(fqn)
                    nonfrozen_param_numel += param.numel()
                else:
                    frozen_param_fqns.append(fqn)
                    frozen_param_numel += param.numel()
            if len(frozen_param_fqns) > 0 and len(nonfrozen_param_fqns) > 0:
                msg = f'{module_name} has both parameters with requires_grad=True and False.'
                if use_orig_params:
                    total_param_numel = frozen_param_numel + nonfrozen_param_numel
                    msg += f' We do not recommend wrapping such modules since the gradient memory usage will be higher than expected ({total_param_numel} numel instead of {nonfrozen_param_numel} numel before sharding via reduce-scatter). '
                else:
                    msg += ' FSDP does not support wrapping such modules when use_orig_params=False. '
                msg += 'If possible, wrap the frozen parameters with FSDP separately.\n'
                msg += f'The following parameters have requires_grad=True:\n{nonfrozen_param_fqns}\nThe following parameters have requires_grad=False:\n{frozen_param_fqns}'
                if use_orig_params:
                    warnings.warn(msg)
                else:
                    raise ValueError(msg)

def _get_post_order_named_modules(root_module: nn.Module) -> List[Tuple[str, nn.Module]]:
    if False:
        while True:
            i = 10
    '\n    This returns the named modules following a post-order traversal, which is a\n    valid reverse topological sort. We achieve this using the reverse of a\n    stack-based DFS order instead of reversing ``root_module.named_modules()``\n    since the former gives the modules in registration order at each level in\n    the module tree (as opposed to the reverse), which allows us to error/warn\n    on the first registered module that violates the condition.\n\n    For example, consider the following module structure:\n        M(\n          S1(),\n          S2(\n            SS1(),\n            SS2(),\n          ),\n          S3(),\n        )\n    The reverse DFS order is [S1, SS1, SS2, S2, S3, M], while the reverse\n    ``named_modules()`` order is [S3, SS2, SS1, S2, S1, M].\n    '
    visited_modules = {root_module}
    stack = [('', root_module)]
    reverse_post_order_named_modules: List[Tuple[str, nn.Module]] = []
    while stack:
        (module_name, module) = stack.pop()
        reverse_post_order_named_modules.append((module_name, module))
        for (child_module_name, child_module) in module.named_children():
            if child_module is None:
                continue
            if child_module not in visited_modules:
                visited_modules.add(child_module)
                if module_name != '':
                    child_module_name = module_name + '.' + child_module_name
                stack.append((child_module_name, child_module))
    post_order_named_modules = list(reversed(reverse_post_order_named_modules))
    return post_order_named_modules

def _get_managed_param_to_fqn(module_to_wrap: nn.Module, ignored_params: Set[nn.Parameter], visited_modules: Set[nn.Module], root_prefix: str) -> Dict[nn.Parameter, str]:
    if False:
        for i in range(10):
            print('nop')
    "\n    This returns a dict that maps managed parameter to its FQN for the given\n    ``module_to_wrap``. The dict's keys are exactly the parameters that would\n    be managed by the module, where this is achieved by calling this function\n    on the modules to wrap in reverse topological order, destructively updating\n    ``visited_modules``, and not traversing into those modules. The FQNs are\n    prefixed from the root (via ``root_prefix``) to be more informative.\n\n    NOTE: This function is meant to be called pre-wrapping and iteratively in\n    reverse topological order to cover the full module tree. This differs from\n    the ``_get_param_to_fqn()`` function meant to be called post-wrapping and\n    on the full module tree in one shot. Given those differences, we do not try\n    to unify the two.\n    "
    param_to_fqn: Dict[nn.Parameter, str] = {}
    queue = collections.deque([(module_to_wrap, root_prefix)])
    visited_modules.add(module_to_wrap)
    while queue:
        (module, prefix) = queue.popleft()
        for (param_name, param) in module.named_parameters(recurse=False):
            if param not in ignored_params:
                fqn = param_name if prefix == '' else prefix + '.' + param_name
                param_to_fqn[param] = fqn
        for (child_module_name, child_module) in module.named_children():
            if child_module is None:
                continue
            if child_module not in visited_modules:
                visited_modules.add(child_module)
                child_prefix = child_module_name if prefix == '' else prefix + '.' + child_module_name
                queue.append((child_module, child_prefix))
    return param_to_fqn