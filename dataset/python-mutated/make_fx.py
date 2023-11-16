import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_methods_invocations import wrapper_set_seed
import torch.utils._pytree as pytree

def make_fx_check(func, args, kwargs, tracing_mode, assert_close=torch.testing.assert_close, randomize_data=False):
    if False:
        return 10
    (f, *new_args) = handle_sizes_for_dynamic_shapes(func, args, kwargs)

    def run(f, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return wrapper_set_seed(f, *args, **kwargs)
    traced_f = make_fx(f, tracing_mode=tracing_mode)(*new_args)
    msg = 'op(*args, **kwargs) and make_fx(op)(*args, **kwargs) produced different values. This could mean that your abstract impls (meta/FakeTensor impls) are incorrect, that your operator is not completely traceable (e.g., it relies on some global state), or that there is a bug in make_fx. Note that if you passed a python function (and not an operator) to make_fx_check, it is still possible that the python function will still work with torch.compile because it handles capturing pieces of your python code to compile.'
    if randomize_data:
        new_args = randomize(new_args)
    try:
        expected = run(f, *new_args)
    except Exception:
        if randomize_data:
            return
        raise
    result = run(traced_f, *new_args)
    assert_close(result, expected, msg=msg)

def handle_sizes_for_dynamic_shapes(func, args, kwargs):
    if False:
        return 10

    def f(args, kwargs, extra_args, extra_kwargs):
        if False:
            print('Hello World!')
        if extra_args:
            for (i, t) in extra_args:
                args[i] = t.size()
        if extra_kwargs:
            for (k, t) in extra_kwargs.items():
                kwargs[k] = t.size()
        return func(*args, **kwargs)
    extra_args = []
    extra_kwargs = {}
    for (i, arg) in enumerate(args):
        if isinstance(arg, torch.Size):
            extra_args.append((i, torch.empty(arg, device='cpu')))
    for (key, value) in kwargs.items():
        if isinstance(value, torch.Size):
            extra_kwargs[key] = torch.empty(value, device='cpu')
    return (f, args, kwargs, extra_args, extra_kwargs)

def randomize(args):
    if False:
        print('Hello World!')

    def transform(x):
        if False:
            i = 10
            return i + 15
        if not x.dtype.is_floating_point:
            return x
        return x.detach().clone().uniform_(0, 1).requires_grad_(x.requires_grad)
    return pytree.tree_map_only(torch.Tensor, transform, args)