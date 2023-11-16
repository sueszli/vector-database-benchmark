import contextlib
import torch
import torch.utils._pytree as pytree

@contextlib.contextmanager
def set_autograd_fallback_mode(mode):
    if False:
        for i in range(10):
            print('nop')
    prev = torch._C._get_autograd_fallback_mode()
    try:
        torch._C._set_autograd_fallback_mode(mode)
        yield
    finally:
        torch._C._set_autograd_fallback_mode(prev)

def autograd_registration_check(op, args, kwargs):
    if False:
        return 10
    'Check if autograd was registered correctly (for the operator).\n\n    Operators should have "autograd support" registered directly to an\n    autograd dispatch key.\n    An incorrect registration may lead to unexpected silent incorrectness.\n    Note that this check won\'t catch all problems but will catch\n    the most common ones.\n\n    Example usage:\n        >>> x = torch.randn(3, requires_grad=True)\n        >>> autograd_registration_check(torch.ops.aten.sin.default, (x,), {})\n\n    Here are some best practices if you do find your autograd is\n    registered incorrectly:\n    - If the operator is composite (i.e. consists of other PyTorch ops)\n      and you wish the operator to decompose and get autograd support\n      that way, then please register the implementation to\n      DispatchKey::CompositeImplicitAutograd\n    - If you\'re adding an autograd formula for the operator, the correct\n      thing to do is to register an autograd.Function to\n      DispatchKey::Autograd (preferred) or one of the\n      DispatchKey::Autograd<BACKEND> keys. It is NOT OK to register\n      an autograd.Function to a backend (e.g. CPU/CUDA) key.\n    - If your operator is non-differentiable, then you should register\n      an implementation to the Autograd key that uses\n      AutoDispatchBelowAutograd and re-invokes the operator.\n\n    '
    assert isinstance(op, torch._ops.OpOverload)
    flat_args = pytree.arg_tree_leaves(*args, **kwargs)
    all_tensors = [arg for arg in flat_args if isinstance(arg, torch.Tensor)]
    if not any((t.requires_grad for t in all_tensors)):
        raise RuntimeError('autograd_registration_check: no inputs have requires_grad=True so we are unable to actually perform this test. Please pass inputs that do require grad.')
    all_device_types = {arg.device.type for arg in all_tensors}
    if not all_device_types.issubset(['cpu', 'cuda']):
        raise NotImplementedError(f'autograd_registration_check: NYI devices other than CPU/CUDA, got {all_device_types}')
    if 'cuda' in all_device_types:
        key = 'AutogradCUDA'
    elif 'cpu' in all_device_types:
        key = 'AutogradCPU'
    if torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), key):
        return
    if torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), 'Autograd'):
        return
    if torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), 'CompositeImplicitAutograd'):
        return
    with set_autograd_fallback_mode('nothing'):
        all_outs = op(*args, **kwargs)
    inp_ids = {id(arg) for arg in flat_args}

    def not_an_input_and_requires_grad(tensor):
        if False:
            return 10
        if not tensor.requires_grad:
            return False
        if id(tensor) in inp_ids:
            return False
        return True
    if not pytree.tree_any_only(torch.Tensor, not_an_input_and_requires_grad, all_outs):
        return
    raise AssertionError(f'{op.name()}: at least one output of this operator has requires_grad=True but the operator does not have an autograd kernel defined at an autograd key (e.g. DispatchKey::Autograd). This could mean that you have incorrectly registered an autograd kernel to a non-Autograd DispatchKey, which may lead to silently incorrect results. If your operator consists of regular PyTorch operations, consider not using an operator at all or registering your operator as CompositeImplicitAutograd. If you have an autograd.Function registered to a backend (CPU/CUDA) key, the correct location for it is the Autograd key.')