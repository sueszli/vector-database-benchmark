import inspect
from torch._custom_op.impl import _custom_op_with_schema, _find_custom_op, infer_schema, parse_qualname, validate_namespace
from torch.library import get_ctx
__all__ = ['custom_op', 'impl', 'impl_abstract', 'get_ctx', 'impl_save_for_backward', 'impl_backward']

def custom_op(qualname, func_or_schema=None):
    if False:
        return 10
    'Register a new custom operator\n\n    In PyTorch, defining an op (short for "operator") is a two step-process:\n    - we need to define the op (by providing an operator name and schema)\n    - we need to implement behavior for how the operator interacts with\n      various PyTorch subsystems, like CPU/CUDA Tensors, Autograd, etc.\n\n    This entrypoint defines the custom operator (the first step)\n    you must then perform the second step by calling various\n    ``impl_*`` APIs.\n\n    This API may be used as a decorator (see examples).\n\n    For a detailed guide on custom ops, please see\n    https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk\n\n    Arguments:\n        qualname (str): Should be a string that looks like\n            "namespace::operator_name". Operators in PyTorch need a namespace to\n            avoid name collisions; a given operator may only be created once.\n            If you are writing a Python library, we recommend the namespace to\n            be the name of your top-level module.\n        func_or_schema (Union[Callable, str]): Each PyTorch operator needs a\n            schema that tells PyTorch the types of the inputs/outputs.\n            If this is a Callable, we will automatically infer the schema from\n            the type annotations on the function (see examples). Otherwise,\n            if you don\'t want to use type annotations, you may provide us the\n            schema string.\n\n    Example::\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)\n        >>> import torch\n        >>> import numpy as np\n        >>> from torch import Tensor\n        >>>\n        >>> # Step 1: define the custom op.\n        >>> # We need to provide the API a "prototype function"\n        >>> # (a function that returns NotImplementedError), from which\n        >>> # we will infer the types of the inputs and outputs.\n        >>> @torch._custom_ops.custom_op("mylibrary::numpy_sin")\n        >>> def numpy_sin(x: Tensor) -> Tensor:\n        >>>     raise NotImplementedError()\n        >>>\n        >>> # The custom op is now accessible via the torch.ops module:\n        >>> torch.ops.mylibrary.numpy_sin\n        >>>\n        >>> # Step 2: Register an implementation for various PyTorch subsystems\n        >>>\n        >>> # Register an implementation for CPU tensors\n        >>> @torch._custom_ops.impl("mylibrary::numpy_sin", device_types="cpu")\n        >>> def numpy_sin_impl_cpu(x):\n        >>>     return torch.from_numpy(np.sin(x.numpy()))\n        >>>\n        >>> # Register an implementation for CUDA tensors\n        >>> @torch._custom_ops.impl("mylibrary::numpy_sin", device_types="cuda")\n        >>> def numpy_sin_impl_cuda(x):\n        >>>     return torch.from_numpy(np.sin(x.cpu().numpy())).to(x.device)\n        >>>\n        >>> x = torch.randn(3)\n        >>> torch.ops.mylibrary.numpy_sin(x)  # calls numpy_sin_impl_cpu\n        >>>\n        >>> x_cuda = x.cuda()\n        >>> torch.ops.mylibrary.numpy_sin(x)  # calls numpy_sin_impl_cuda\n\n    '
    (ns, name) = parse_qualname(qualname)
    validate_namespace(ns)

    def inner(func):
        if False:
            while True:
                i = 10
        if not inspect.isfunction(func):
            raise ValueError(f'custom_op(...)(func): Expected `func` to be a Python function, got: {type(func)}')
        if func.__name__ != name:
            raise ValueError(f"custom_op(qualname='{qualname}', ...)(func): expected `func` to have name '{name}' but got '{func.__name__}'. Please either change the name of `func` or the qualname that is passed to `custom_op`")
        schema = infer_schema(func)
        _custom_op_with_schema(qualname, schema)
        return func
    if func_or_schema is None:
        return inner
    if isinstance(func_or_schema, str):
        _custom_op_with_schema(qualname, func_or_schema)
    else:
        return inner(func_or_schema)

def impl(qualname, *, device_types=('cpu', 'cuda'), func=None):
    if False:
        while True:
            i = 10
    'Register an implementation for a device type for this custom op.\n\n    If the op is passed multiple Tensor inputs with different device\n    types, it will dispatch to the registered implementation for the highest\n    priority device type among those present.\n    The supported device types, in order of priority, are {\'cuda\', \'cpu\'}.\n\n    This API may be used as a decorator (see examples).\n\n    For a detailed guide on custom ops, please see\n    https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk\n\n    Arguments:\n        device_types (str or Iterable[str]): the device type(s) to register the function for.\n\n    Example::\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)\n        >>> import torch\n        >>> import numpy as np\n        >>> from torch import Tensor\n        >>>\n        >>> # Step 1: define the custom op.\n        >>> # We need to provide the API a "prototype function"\n        >>> # (a function that returns NotImplementedError), from which\n        >>> # we will infer the types of the inputs and outputs.\n        >>> @torch._custom_ops.custom_op("mylibrary::numpy_cos")\n        >>> def numpy_cos(x: Tensor) -> Tensor:\n        >>>     raise NotImplementedError()\n        >>>\n        >>> # The custom op is now accessible via the torch.ops module:\n        >>> torch.ops.mylibrary.numpy_cos\n        >>>\n        >>> # Step 2: Register an implementation for various PyTorch subsystems\n        >>>\n        >>> # Register an implementation for CPU tensors\n        >>> @torch._custom_ops.impl("mylibrary::numpy_cos", device_types="cpu")\n        >>> def numpy_cos_impl_cpu(x):\n        >>>     return torch.from_numpy(np.cos(x.numpy()))\n        >>>\n        >>> # Register an implementation for CUDA tensors\n        >>> @torch._custom_ops.impl("mylibrary::numpy_cos", device_types="cuda")\n        >>> def numpy_cos_impl_cuda(x):\n        >>>     return torch.from_numpy(np.cos(x.cpu().numpy())).to(x.device)\n        >>>\n        >>> x = torch.randn(3)\n        >>> torch.ops.mylibrary.numpy_cos(x)  # calls numpy_cos_impl_cpu\n        >>>\n        >>> x_cuda = x.cuda()\n        >>> torch.ops.mylibrary.numpy_cos(x)  # calls numpy_cos_impl_cuda\n\n    '

    def inner(func):
        if False:
            for i in range(10):
                print('nop')
        custom_op = _find_custom_op(qualname, also_check_torch_library=True)
        custom_op.impl(device_types, _stacklevel=3)(func)
        return func
    if func is None:
        return inner
    return inner(func)

def impl_abstract(qualname, *, func=None):
    if False:
        while True:
            i = 10
    'Register an abstract implementation for this operator.\n\n    An "abstract implementation" specifies the behavior of this operator on\n    Tensors that carry no data. Given some input Tensors with certain properties\n    (sizes/strides/storage_offset/device), it specifies what the properties of\n    the output Tensors are.\n\n    The abstract implementation has the same signature as the operator.\n    It is run for both FakeTensors and meta tensors. To write an abstract\n    implementation, assume that all Tensor inputs to the operator are\n    regular CPU/CUDA/Meta tensors, but they do not have storage, and\n    you are trying to return regular CPU/CUDA/Meta tensor(s) as output.\n    The abstract implementation must consist of only PyTorch operations\n    (and may not directly access the storage or data of any input or\n    intermediate Tensors).\n\n    This API may be used as a decorator (see examples).\n\n    For a detailed guide on custom ops, please see\n    https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk\n\n    Examples::\n        >>> import numpy as np\n        >>> from torch import Tensor\n        >>>\n        >>> # Example 1: an operator without data-dependent output shape\n        >>> @torch._custom_ops.custom_op("mylibrary::custom_linear")\n        >>> def custom_linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:\n        >>>     raise NotImplementedError()\n        >>>\n        >>> @torch._custom_ops.impl_abstract("mylibrary::custom_linear")\n        >>> def custom_linear_abstract(x, weight):\n        >>>     assert x.dim() == 2\n        >>>     assert weight.dim() == 2\n        >>>     assert bias.dim() == 1\n        >>>     assert x.shape[1] == weight.shape[1]\n        >>>     assert weight.shape[0] == bias.shape[0]\n        >>>     assert x.device == weight.device\n        >>>\n        >>>     return (x @ weight.t()) + bias\n        >>>\n        >>> # Example 2: an operator with data-dependent output shape\n        >>> @torch._custom_ops.custom_op(\'mylibrary::custom_nonzero\')\n        >>> def custom_nonzero(x: Tensor) -> Tensor:\n        >>>     ...\n        >>>\n        >>> @torch._custom_ops.impl_abstract("mylibrary::custom_nonzero")\n        >>> def custom_nonzero_abstract(x):\n        >>>     # Number of nonzero-elements is data-dependent.\n        >>>     # Since we cannot peek at the data in an abstract impl,\n        >>>     # we use the ctx object to construct a new symint that\n        >>>     # represents the data-dependent size.\n        >>>     ctx = torch._custom_ops.get_ctx()\n        >>>     nnz = ctx.create_unbacked_symint()\n        >>>     shape = [x.dim(), nnz]\n        >>>     result = x.new_empty(shape, dtype=torch.long)\n        >>>     return result\n        >>>\n        >>> @torch._custom_ops.impl("mylibrary::custom_nonzero")\n        >>> def custom_nonzero_impl(x):\n        >>>     x_np = to_numpy(x)\n        >>>     res = np.stack(np.nonzero(x_np), axis=1)\n        >>>     # unbacked symbolic ints in PyTorch must be >= 2, so we\n        >>>     # constrain the range to at least 2\n        >>>     if res.shape[0] <= 1:\n        >>>         raise RuntimeError("not supported")\n        >>>     return torch.tensor(res, device=x.device)\n\n    '
    import torch.library
    return torch.library.impl_abstract(qualname, func, _stacklevel=2)

def impl_save_for_backward(qualname, *, func=None):
    if False:
        while True:
            i = 10
    'Register a function that tells us what to save for backward.\n\n    Please see :func:`impl_backward` for more details.\n    '

    def inner(func):
        if False:
            return 10
        custom_op = _find_custom_op(qualname, also_check_torch_library=True)
        custom_op.impl_save_for_backward(_stacklevel=3)(func)
        return func
    if func is None:
        return inner
    return inner(func)

def impl_backward(qualname, output_differentiability=None, *, func=None):
    if False:
        while True:
            i = 10
    'Registers a backward formula for an operator.\n\n    In order for an operator to work with autograd, you need to register\n    a backward formula. There are two pieces to this:\n    1. You must give us a function to specify what to save for backward.\n       Call this the "save for backward" function.\n    2. You must give us a function that computes gradients. Call this the\n       "backward" function.\n\n    Use `impl_save_for_backward` to define a "save for backward" function\n    that specifies what gets saved for backward. The function should accept\n    two arguments ``(inputs, output)`` and return the quantities to be saved\n    for backward.\n\n    During runtime, when you call the operator in a forwards pass, PyTorch\n    will invoke the "save for backward" function with the inputs and output\n    of the operator.\n\n    Use `impl_backward` to define the "backward" function. The backward\n    function must accept ``(ctx, saved, *grads)``:\n    - ``ctx`` is a context object where we may provide information\n    - ``saved`` is exactly what gets returned from the "save for backward"\n      function\n    - ``grads`` is one or more gradients. The number of gradients matches\n      the number of outputs of the operator.\n\n    The backward function must return a dict that maps the name of\n    an input to the operator to its corresponding gradient. All inputs that\n    were declared to be Tensors in the operator definition must be accounted\n    for in the dict. The gradient may be a Tensor or None.\n\n    For a detailed guide on custom ops, please see\n    https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk\n\n    '

    def inner(func):
        if False:
            print('Hello World!')
        custom_op = _find_custom_op(qualname, also_check_torch_library=True)
        custom_op.impl_backward(output_differentiability, _stacklevel=3)(func)
        return func
    if func is None:
        return inner
    return inner(func)

def _destroy(qualname):
    if False:
        print('Hello World!')
    'De-registers a custom op. For testing purposes only'
    custom_op = _find_custom_op(qualname)
    custom_op._destroy()