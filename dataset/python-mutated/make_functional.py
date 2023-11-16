import copy
from typing import Any, Callable, Dict, Iterable, List, NoReturn, Sequence, Tuple, Type, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor

def raise_parameter_tying_error() -> NoReturn:
    if False:
        for i in range(10):
            print('nop')
    raise RuntimeError("make_functional(module): we don't yet support models that do parameter tying (also sometimes known as weight sharing). Please try to rewrite your model by replacing all instances of the tied parameter with another and/or comment your support in https://github.com/pytorch/functorch/issues/446")

def create_names_map(named_params: Union[Dict[str, Tensor], Iterable[Tuple[str, Tensor]]], tied_named_params: Union[Dict[str, Tensor], Iterable[Tuple[str, Tensor]]]) -> Dict[str, List[str]]:
    if False:
        for i in range(10):
            print('nop')
    "\n    named_params is a dictionary of tensors: {'A': A, 'B': B}\n    tied_named_params is another dictionary of tensors {'A': A, 'B': B, 'B_tied': B}\n    with potentially tied (or 'duplicated') tensors\n\n    This function creates a mapping from the names in named_params to the\n    names in tied_named_params: {'A': ['A'], 'B': ['B', 'B_tied']}.\n    "
    named_params = dict(named_params)
    tied_named_params = dict(tied_named_params)
    tensors_dict_keys = set(named_params.keys())
    tied_tensors_dict_keys = set(tied_named_params.keys())
    assert tensors_dict_keys.issubset(tied_tensors_dict_keys)
    tensor_to_mapping: Dict[Tensor, Tuple[str, List[str]]] = {}
    for (key, tensor) in named_params.items():
        tensor_to_mapping[tensor] = (key, [])
    for (key, tensor) in tied_named_params.items():
        assert tensor in tensor_to_mapping
        tensor_to_mapping[tensor][1].append(key)
    return dict(tensor_to_mapping.values())

def _extract_members(mod: nn.Module, named_members: Callable[..., Iterable[Tuple[str, Tensor]]], subclass: Callable[[Tensor], Tensor]) -> Tuple[Tuple[Tensor, ...], Tuple[str, ...], Dict[str, List[str]]]:
    if False:
        print('Hello World!')
    all_named_members = tuple(named_members(remove_duplicate=False))
    unique_named_members = tuple(named_members(remove_duplicate=True))
    names_map = create_names_map(unique_named_members, all_named_members)
    memo = {}
    accessor = NamedMemberAccessor(mod)
    for (name, p) in all_named_members:
        if p not in memo:
            memo[p] = subclass(torch.empty_like(p, device='meta'))
        replacement = memo[p]
        accessor.set_tensor(name, replacement)
    if len(unique_named_members) == 0:
        (names, params) = ((), ())
    else:
        (names, params) = zip(*unique_named_members)
    return (params, names, names_map)

def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], Tuple[str, ...], Dict[str, List[str]]]:
    if False:
        i = 10
        return i + 15
    '\n    This function removes all the Parameters from the model and\n    return them as a tuple as well as their original attribute names.\n    The weights must be re-loaded with `load_weights` before the model\n    can be used again.\n    Note that this function modifies the model in place and after this\n    call, mod.parameters() will be empty.\n    '
    return _extract_members(mod, mod.named_parameters, nn.Parameter)

def extract_buffers(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], Tuple[str, ...], Dict[str, List[str]]]:
    if False:
        i = 10
        return i + 15
    return _extract_members(mod, mod.named_buffers, lambda x: x)

def load_weights(mod: nn.Module, names: Sequence[str], params: Sequence[Tensor], as_params: bool=False) -> None:
    if False:
        print('Hello World!')
    '\n    Reload a set of weights so that `mod` can be used again to perform a forward pass.\n    Note that the `params` are regular Tensors (that can have history) and so are left\n    as Tensors. This means that mod.parameters() will still be empty after this call.\n    '
    accessor = NamedMemberAccessor(mod)
    if as_params:
        params = [nn.Parameter(p) for p in params]
    accessor.set_tensors(names, params)

def _swap_state(mod: nn.Module, names_map: Dict[str, List[str]], elems: Iterable[Tensor]) -> List[Tensor]:
    if False:
        print('Hello World!')
    result: List[Tensor] = []
    accessor = NamedMemberAccessor(mod)
    for ((_, attr_names), elem) in zip(names_map.items(), elems):
        for (i, attr_name) in enumerate(attr_names):
            if i == 0:
                result.append(accessor.swap_tensor(attr_name, elem))
            else:
                accessor.set_tensor(attr_name, elem)
    return result

def load_buffers(mod: nn.Module, names: Sequence[str], buffers: Sequence[Tensor], as_params: bool=False) -> None:
    if False:
        return 10
    accessor = NamedMemberAccessor(mod)
    accessor.set_tensors(names, buffers)

def load_state(model: nn.Module, weights: Sequence[Tensor], weight_names: Sequence[str], buffers: Sequence[Tensor]=(), buffer_names: Sequence[str]=()) -> nn.Module:
    if False:
        print('Hello World!')
    'load_state(model, weights, weight_names, buffers=(), buffer_names=()) -> model\n\n    load_state takes `weights` and `buffers` and assigns them to the model.\n    This is the inverse operation of `make_functional_deprecated_v1`.\n    '
    assert len(weight_names) == len(weights)
    load_weights(model, weight_names, weights)
    if len(buffers) > 0:
        assert len(buffer_names) == len(buffers)
        load_buffers(model, buffer_names, buffers)
    return model

def make_functional_deprecated_v1(model: nn.Module):
    if False:
        for i in range(10):
            print('nop')
    'make_functional_deprecated_v1(model) -> weights, func, weight_names\n\n    Given an nn.Module, make_functional_deprecated_v1 extracts the state (weights)\n    and returns a functional version of the model, `func`. This makes\n    it so that it is possible use transforms over the parameters of\n    `model`.\n\n    `func` can be invoked as follows:\n    ```\n    x = torch.randn(4, 3)\n    model = nn.Linear(3, 3)\n    weights, func, _ = make_functional_deprecated_v1(model)\n    func(weights, (x,))\n    ```\n\n    And here is an example of applying the grad transform:\n    ```\n    x = torch.randn(4, 3)\n    model = nn.Linear(3, 3)\n    weights, _, func = make_functional_deprecated_v1(model)\n    grad_weights = grad(func)(weights, (x,))\n    ```\n\n    To put the state back into a model, use `load_state`.\n    '
    buffers = list(model.buffers())
    if len(buffers) > 0:
        raise RuntimeError('make_functional_deprecated_v1(model): `model` has buffers. Please use make_functional_with_buffers_deprecated_v1(model) instead.')
    (weights, descriptors, _) = extract_weights(model)

    def fun(weights, data):
        if False:
            return 10
        mutable_model = copy.deepcopy(model)
        load_weights(mutable_model, descriptors, weights)
        return mutable_model(*data)
    return (weights, fun, descriptors)

def make_functional_with_buffers_deprecated_v1(model: nn.Module):
    if False:
        return 10
    'make_functional_with_buffers_deprecated_v1(model) -> weights, buffers, func, weight_names, buffer_names\n\n    Given an nn.Module, make_functional_with_buffers_deprecated_v1 extracts the state (weights and buffers)\n    and returns a functional version of the model, `func`.\n\n    `func` can be invoked as follows:\n    ```\n    x = torch.randn(4, 3)\n    model = nn.Linear(3, 3)\n    weights, buffers, func, _, _ = make_functional_with_buffers_deprecated_v1(model)\n    func(weights, buffers, (x,))\n    ```\n\n    And here is an example of applying the grad transform:\n    ```\n    x = torch.randn(4, 3)\n    model = nn.Linear(3, 3)\n    weights, buffers, func, _, _ = make_functional_with_buffers_deprecated_v1(model)\n    func(weights, buffers, (x,))\n    grad_weights = grad(func)(weights, buffers, (x,))\n    ```\n\n    To put the state back into a model, use `load_state`.\n    '
    (weights, weight_descriptors, _) = extract_weights(model)
    (buffers, buf_descriptors, _) = extract_buffers(model)

    def fun(weights, buffers, data):
        if False:
            return 10
        mutable_model = copy.deepcopy(model)
        load_weights(mutable_model, weight_descriptors, weights)
        load_buffers(mutable_model, buf_descriptors, buffers)
        return mutable_model(*data)
    return (weights, buffers, fun, weight_descriptors, buf_descriptors)

class FunctionalModuleWithBuffers(nn.Module):
    """
    This is the callable object returned by :func:`make_functional_with_buffers`.
    """

    def __init__(self, stateless_model: nn.Module, param_names: Tuple[str, ...], buffer_names: Tuple[str, ...], param_names_map: Dict[str, List[str]], buffer_names_map: Dict[str, List[str]]) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.stateless_model = stateless_model
        self.param_names = param_names
        self.buffer_names = buffer_names
        self.all_names_map = dict(param_names_map)
        self.all_names_map.update(buffer_names_map)

    @staticmethod
    def _create_from(model: nn.Module, disable_autograd_tracking: bool=False) -> Tuple['FunctionalModuleWithBuffers', Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        if False:
            print('Hello World!')
        model_copy = copy.deepcopy(model)
        (params, param_names, param_names_map) = extract_weights(model_copy)
        (buffers, buffer_names, buffer_names_map) = extract_buffers(model_copy)
        if disable_autograd_tracking:
            for param in params:
                param.requires_grad_(False)
        return (FunctionalModuleWithBuffers(model_copy, param_names, buffer_names, param_names_map, buffer_names_map), params, buffers)

    def forward(self, params: Iterable[Tensor], buffers: Iterable[Tensor], *args, **kwargs) -> Any:
        if False:
            for i in range(10):
                print('nop')
        old_state = _swap_state(self.stateless_model, self.all_names_map, tuple(params) + tuple(buffers))
        try:
            return self.stateless_model(*args, **kwargs)
        finally:
            _swap_state(self.stateless_model, self.all_names_map, old_state)

class FunctionalModule(nn.Module):
    """
    This is the callable object returned by :func:`make_functional`.
    """

    def __init__(self, stateless_model: nn.Module, param_names: Tuple[str, ...], names_map: Dict[str, List[str]]) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.stateless_model = stateless_model
        self.param_names = param_names
        self.names_map = names_map

    @staticmethod
    def _create_from(model: nn.Module, disable_autograd_tracking: bool=False) -> Tuple['FunctionalModule', Tuple[Tensor, ...]]:
        if False:
            return 10
        model_copy = copy.deepcopy(model)
        (params, param_names, names_map) = extract_weights(model_copy)
        if disable_autograd_tracking:
            for param in params:
                param.requires_grad_(False)
        return (FunctionalModule(model_copy, param_names, names_map), params)

    def forward(self, params: Iterable[Tensor], *args, **kwargs) -> Any:
        if False:
            while True:
                i = 10
        old_state = _swap_state(self.stateless_model, self.names_map, params)
        try:
            return self.stateless_model(*args, **kwargs)
        finally:
            _swap_state(self.stateless_model, self.names_map, old_state)

def make_functional(model: nn.Module, disable_autograd_tracking: bool=False) -> Tuple[FunctionalModule, Tuple[Tensor, ...]]:
    if False:
        while True:
            i = 10
    "make_functional(model, disable_autograd_tracking=False) -> func, params\n\n    Given a ``torch.nn.Module``, :func:`make_functional` extracts the state\n    (params) and returns a functional version of the model, ``func``. This\n    makes it so that it is possible use transforms over the parameters of\n    ``model``.\n\n    ``func`` can be invoked as follows:\n\n    .. code-block:: python\n\n        import torch\n        import torch.nn as nn\n        from functorch import make_functional\n\n        x = torch.randn(4, 3)\n        model = nn.Linear(3, 3)\n        func, params = make_functional(model)\n        func(params, x)\n\n    And here is an example of applying the grad transform over the parameters\n    of a model.\n\n    .. code-block:: python\n\n        import torch\n        import torch.nn as nn\n        from functorch import make_functional, grad\n\n        x = torch.randn(4, 3)\n        t = torch.randn(4, 3)\n        model = nn.Linear(3, 3)\n        func, params = make_functional(model)\n\n        def compute_loss(params, x, t):\n            y = func(params, x)\n            return nn.functional.mse_loss(y, t)\n\n        grad_weights = grad(compute_loss)(params, x, t)\n\n    If the model has any buffers, please use :func:`make_functional_with_buffers` instead.\n\n    Args:\n        model (torch.nn.Module): Input model.\n        disable_autograd_tracking (bool): Flag to disable gradients tracking for output parameters.\n            The returned params are unrelated to the set of params from the original model. If False (default),\n            the params will have ``requires_grad=True`` on them (aka they will be trackable with regular\n            PyTorch autograd), matching the requires_grad-ness of the params from the original model.\n            Otherwise, the returned params will have ``requires_grad=False``. Default, False.\n            If you plan on using regular PyTorch autograd (e.g., if you want to call ``.backward()`` or\n            ``torch.autograd.grad()``, then set ``disable_autograd_tracking=False``.\n            Otherwise, if you're only planning on using functorch's gradient transforms,\n            then please set ``disable_autograd_tracking=True`` to avoid unnecessarily tracking\n            history with PyTorch autograd.\n\n    "
    buffers = list(model.buffers())
    if len(buffers) > 0:
        raise RuntimeError('make_functional(model): `model` has buffers. Please use make_functional_with_buffers(model) instead.')
    return FunctionalModule._create_from(model, disable_autograd_tracking=disable_autograd_tracking)

def make_functional_with_buffers(model: nn.Module, disable_autograd_tracking: bool=False) -> Tuple[FunctionalModuleWithBuffers, Tuple[Tensor, ...], Tuple[Tensor, ...]]:
    if False:
        while True:
            i = 10
    "make_functional_with_buffers(model, disable_autograd_tracking=False) -> func, params, buffers\n\n    Given a ``torch.nn.Module``, make_functional_with_buffers extracts the\n    state (params and buffers) and returns a functional version of the model\n    ``func`` that can be invoked like a function.\n\n    ``func`` can be invoked as follows:\n\n    .. code-block:: python\n\n        import torch\n        import torch.nn as nn\n        from functorch import make_functional_with_buffers\n\n        x = torch.randn(4, 3)\n        model = nn.Linear(3, 3)\n        func, params, buffers = make_functional_with_buffers(model)\n        func(params, buffers, x)\n\n    And here is an example of applying the grad transform over the parameters\n    of a model:\n\n    .. code-block:: python\n\n        import torch\n        import torch.nn as nn\n        from functorch import make_functional_with_buffers, grad\n\n        x = torch.randn(4, 3)\n        t = torch.randn(4, 3)\n        model = nn.Linear(3, 3)\n        func, params, buffers = make_functional_with_buffers(model)\n\n        def compute_loss(params, buffers, x, t):\n            y = func(params, buffers, x)\n            return nn.functional.mse_loss(y, t)\n\n        grad_weights = grad(compute_loss)(params, buffers, x, t)\n\n    Args:\n        model (torch.nn.Module): Input model.\n        disable_autograd_tracking (bool): Flag to disable gradients tracking for output parameters.\n            The returned params are unrelated to the set of params from the original model. If False (default),\n            the params will have ``requires_grad=True`` on them (aka they will be trackable with regular\n            PyTorch autograd), matching the requires_grad-ness of the params from the original model.\n            Otherwise, the returned params will have ``requires_grad=False``. Default, False.\n            If you plan on using regular PyTorch autograd (e.g., if you want to call ``.backward()`` or\n            ``torch.autograd.grad()``, then set ``disable_autograd_tracking=False``.\n            Otherwise, if you're only planning on using functorch's gradient transforms,\n            then please set ``disable_autograd_tracking=True`` to avoid unnecessarily tracking\n            history with PyTorch autograd.\n\n    "
    return FunctionalModuleWithBuffers._create_from(model, disable_autograd_tracking=disable_autograd_tracking)

def transpose_stack(tuple_of_tuple_of_tensors: Tuple[Tuple[Tensor, ...], ...]) -> Tuple[Tensor, ...]:
    if False:
        return 10
    tuple_of_tuple_of_tensors = tuple(zip(*tuple_of_tuple_of_tensors))
    results = tuple((torch.stack(shards).detach() for shards in tuple_of_tuple_of_tensors))
    return results

def combine_state_for_ensemble(models: Sequence[nn.Module]) -> Tuple[FunctionalModuleWithBuffers, Tuple[Tensor, ...], Tuple[Tensor, ...]]:
    if False:
        return 10
    "combine_state_for_ensemble(models) -> func, params, buffers\n\n    Prepares a list of torch.nn.Modules for ensembling with :func:`vmap`.\n\n    Given a list of ``M`` ``nn.Modules`` of the same class, stacks all of their\n    parameters and buffers together to make ``params`` and ``buffers``.\n    Each parameter and buffer in the result will have an additional dimension\n    of size ``M``.\n\n    :func:`combine_state_for_ensemble` also returns ``func``, a functional\n    version of one of the models in :attr:`models`. One cannot directly run\n    ``func(params, buffers, *args, **kwargs)`` directly, you probably want to\n    use ``vmap(func, ...)(params, buffers, *args, **kwargs)``\n\n    Here's an example of how to ensemble over a very simple model:\n\n    .. code-block:: python\n\n        num_models = 5\n        batch_size = 64\n        in_features, out_features = 3, 3\n        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]\n        data = torch.randn(batch_size, 3)\n\n        fmodel, params, buffers = combine_state_for_ensemble(models)\n        output = vmap(fmodel, (0, 0, None))(params, buffers, data)\n\n        assert output.shape == (num_models, batch_size, out_features)\n\n    .. warning::\n        All of the modules being stacked together must be the same (except for\n        the values of their parameters/buffers). For example, they should be in the\n        same mode (training vs eval).\n\n        This API is subject to change -- we're investigating better ways to\n        create ensembles and would love your feedback how to improve this.\n    "
    if len(models) == 0:
        raise RuntimeError('combine_state_for_ensemble: Expected at least one model, got 0.')
    if not (all((m.training for m in models)) or all((not m.training for m in models))):
        raise RuntimeError('combine_state_for_ensemble: Expected all models to have the same training/eval mode.')
    model0_typ = type(models[0])
    if not all((type(m) == model0_typ for m in models)):
        raise RuntimeError('combine_state_for_ensemble: Expected all models to be of the same class.')
    (funcs, params, buffers) = zip(*[make_functional_with_buffers(model) for model in models])
    params = transpose_stack(params)
    buffers = transpose_stack(buffers)
    return (funcs[0], params, buffers)

def functional_init(model_class: Type[nn.Module], ensemble_shape: Union[Tuple[()], Tuple[int]]=(), device: torch.types.Device='cpu'):
    if False:
        while True:
            i = 10

    def wrapped(*args, **kwargs):
        if False:
            return 10
        if len(ensemble_shape) >= 2:
            raise ValueError('NYI: ensemble_shape with more than 1 element')
        if len(ensemble_shape) == 0:
            model = model_class(*args, **kwargs).to(device)
            return make_functional_deprecated_v1(model)
        num_models = ensemble_shape[0]
        if num_models <= 0:
            raise ValueError(f'num_models {num_models} should be > 0')
        models = tuple((model_class(*args, **kwargs).to(device) for _ in range(num_models)))
        (_, fn, names) = make_functional_deprecated_v1(model_class(*args, **kwargs))
        weights = tuple((make_functional_deprecated_v1(model)[0] for model in models))
        weights = tuple(zip(*weights))
        weights = tuple((torch.stack(shards).detach() for shards in weights))
        return (weights, fn, names)
    return wrapped

def functional_init_with_buffers(model_class: Type[nn.Module], ensemble_shape: Union[Tuple[()], Tuple[int]]=(), device: torch.types.Device='cpu'):
    if False:
        while True:
            i = 10

    def wrapped(*args, **kwargs):
        if False:
            print('Hello World!')
        if len(ensemble_shape) >= 2:
            raise ValueError('NYI: ensemble_shape with more than 1 element')
        if len(ensemble_shape) == 0:
            model = model_class(*args, **kwargs).to(device)
            return make_functional_deprecated_v1(model)
        num_models = ensemble_shape[0]
        if num_models <= 0:
            raise ValueError(f'num_models {num_models} should be > 0')
        models = tuple((model_class(*args, **kwargs).to(device) for _ in range(num_models)))
        (_, _, fn, weight_names, buffer_names) = make_functional_with_buffers_deprecated_v1(model_class(*args, **kwargs))
        (weights, buffers) = zip(*tuple((make_functional_with_buffers_deprecated_v1(model)[:2] for model in models)))
        weights = tuple(zip(*weights))
        weights = tuple((torch.stack(shards).detach() for shards in weights))
        buffers = tuple(zip(*buffers))
        buffers = tuple((torch.stack(shards).detach() for shards in buffers))
        return (weights, buffers, fn, weight_names, buffer_names)
    return wrapped