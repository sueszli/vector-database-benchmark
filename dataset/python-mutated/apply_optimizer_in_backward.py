from typing import Any, Dict, Iterable, List, no_type_check, Type
import torch
__all__: List[str] = []
param_to_optim_hook_handle_map = torch.utils.weak.WeakTensorKeyDictionary()
param_to_acc_grad_map = torch.utils.weak.WeakTensorKeyDictionary()

@no_type_check
def _apply_optimizer_in_backward(optimizer_class: Type[torch.optim.Optimizer], params: Iterable[torch.nn.Parameter], optimizer_kwargs: Dict[str, Any], register_hook: bool=True) -> None:
    if False:
        while True:
            i = 10
    '\n    Upon ``backward()``, the optimizer specified for each parameter will fire after\n    the gradient has been accumulated into the parameter.\n\n    Note - gradients for these parameters will be set to None after ``backward()``.\n    This means that any other optimizer not specified via `_apply_optimizer_in_backward`\n    over this parameter will be a no-op.\n\n    Args:\n        optimizer_class: (Type[torch.optim.Optimizer]): Optimizer to apply to parameter\n        params: (Iterator[nn.Parameter]): parameters to apply optimizer state to\n        optimizer_kwargs: (Dict[str, Any]): kwargs to pass to optimizer constructor\n        register_hook: (bool): whether to register a hook that runs the optimizer\n            after gradient for this parameter is accumulated. This is the default\n            way that optimizer in backward is implemented, but specific use cases\n            (such as DDP) may wish to override this to implement custom behavior.\n            (Default = True)\n\n    Example::\n        params_generator = model.parameters()\n        param_1 = next(params_generator)\n        remainder_params = list(params_generator)\n\n        apply_optimizer_in_backward(torch.optim.SGD, [param_1], {"lr": .02})\n        apply_optimizer_in_backward(torch.optim.Adam, remainder_params, {"lr": .04})\n\n        model(...).sum().backward() # after backward, parameters will already\n        # have their registered optimizer(s) applied.\n\n    '
    torch._C._log_api_usage_once('torch.distributed.optim.apply_optimizer_in_backward')

    @no_type_check
    def _apply_optimizer_in_backward_to_param(param: torch.nn.Parameter) -> None:
        if False:
            return 10
        if param not in param_to_acc_grad_map:
            param_to_acc_grad_map[param] = param.view_as(param).grad_fn.next_functions[0][0]
        optimizer = optimizer_class([param], **optimizer_kwargs)
        if not hasattr(param, '_in_backward_optimizers'):
            param._in_backward_optimizers = []
            param._optimizer_classes = []
            param._optimizer_kwargs = []
        param._in_backward_optimizers.append(optimizer)
        param._optimizer_classes.append(optimizer_class)
        param._optimizer_kwargs.append(optimizer_kwargs)
        if not register_hook:
            return

        def optimizer_hook(*_unused) -> None:
            if False:
                while True:
                    i = 10
            for opt in param._in_backward_optimizers:
                opt.step()
            param.grad = None
        handle = param_to_acc_grad_map[param].register_hook(optimizer_hook)
        if param not in param_to_optim_hook_handle_map:
            param_to_optim_hook_handle_map[param] = []
        param_to_optim_hook_handle_map[param].append(handle)
    for param in params:
        _apply_optimizer_in_backward_to_param(param)

def _get_in_backward_optimizers(module: torch.nn.Module) -> List[torch.optim.Optimizer]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a list of in-backward optimizers applied to ``module``'s parameters. Note that these\n    optimizers are not intended to directly have their ``step`` or ``zero_grad`` methods called\n    by the user and are intended to be used for things like checkpointing.\n\n    Args:\n        module: (torch.nn.Module): model to retrieve in-backward optimizers for\n\n    Returns:\n        List[torch.optim.Optimizer]: the in-backward optimizers.\n\n    Example::\n        _apply_optimizer_in_backward(torch.optim.SGD, model.parameters(), {'lr': 0.01})\n        optims = _get_optimizers_in_backward(model)\n    "
    optims: List[torch.optim.Optimizer] = []
    for param in module.parameters():
        optims.extend(getattr(param, '_in_backward_optimizers', []))
    return optims