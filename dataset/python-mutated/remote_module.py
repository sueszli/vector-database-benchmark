import collections
import io
import sys
import types
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Set, Tuple, Type, TypeVar, Union
import torch
import torch.distributed.rpc as rpc
from torch import Tensor, device, dtype, nn
from torch.distributed.nn.jit import instantiator
from torch.distributed import _remote_device
from torch.distributed.rpc.internal import _internal_rpc_pickler
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle
__all__ = ['RemoteModule']
_grad_t = Union[Tuple[Tensor, ...], Tensor]
T = TypeVar('T', bound='Module')
_NON_SCRIPTABLE_REMOTE_MODULE_MODULE = instantiator.instantiate_non_scriptable_remote_module_template()
_REMOTE_MODULE_PICKLED_ATTRIBUTES = ('on', 'device', 'is_device_map_set', 'is_scriptable', 'generated_methods', 'module_rref')
_SerializedRemoteModule = collections.namedtuple('_SerializedRemoteModule', _REMOTE_MODULE_PICKLED_ATTRIBUTES)
_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING = ('training', '_parameters', '_buffers', '_non_persistent_buffers_set', '_backward_hooks', '_backward_pre_hooks', '_is_full_backward_hook', '_forward_hooks', '_forward_hooks_with_kwargs', '_forward_hooks_always_called', '_forward_pre_hooks', '_forward_pre_hooks_with_kwargs', '_state_dict_hooks', '_state_dict_pre_hooks', '_load_state_dict_pre_hooks', '_load_state_dict_post_hooks', '_state_dict_pre_hooks', '_modules', 'forward_async', 'forward')

def _instantiate_template(module_interface_cls, enable_moving_cpu_tensors_to_cuda):
    if False:
        return 10
    instantiator.instantiate_scriptable_remote_module_template(module_interface_cls, enable_moving_cpu_tensors_to_cuda)

def _create_module(module_cls, args, kwargs, device):
    if False:
        print('Hello World!')
    module = module_cls(*args, **kwargs)
    if not isinstance(module, nn.Module):
        raise ValueError(f'Expect `module_cls(*args, **kwargs)` returns an instance of <class nn.Module>, but it returns an instance of {type(module)}.')
    module.to(device)
    return module

def _create_module_with_interface(module_cls, args, kwargs, device, module_interface_cls):
    if False:
        for i in range(10):
            print('nop')
    module = _create_module(module_cls, args, kwargs, device)
    if module_interface_cls is not None:
        module = torch.jit.script(module)
    return rpc.RRef(module, module_interface_cls)

def _param_rrefs(module_rref, recurse) -> List[rpc.RRef[Parameter]]:
    if False:
        return 10
    ret: List[rpc.RRef[Parameter]] = []
    for param in module_rref.local_value().parameters(recurse):
        ret.append(rpc.RRef(param))
    return ret

def _raise_not_supported(name: str) -> None:
    if False:
        print('Hello World!')
    raise ValueError(f'Method ``{name}`` not supported for RemoteModule')

class _RemoteModule(nn.Module):

    def __new__(cls, *args, **kwargs):
        if False:
            return 10
        torch._C._log_api_usage_once('torch.distributed.nn.api.remote_module')
        return super().__new__(cls)

    def __init__(self, remote_device: str, module_cls: Type[nn.Module], args: Optional[Tuple]=None, kwargs: Optional[Dict[str, Any]]=None, _module_interface_cls: Any=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        RemoteModule instance can only be created after RPC initialization.\n\n        It creates a user-specified module on a specified remote node.\n        It behaves like a regular ``nn.Module`` except that the ``forward`` method is\n        executed on the remote node.\n        It takes care of autograd recording to ensure the backward pass propagates\n        gradients back to the corresponding remote module.\n        It can be shared across processors using `RPC framework <https://pytorch.org/docs/stable/rpc.html>`__,\n        without incurring any overheads of copying the actual module,\n        which is equivalent to an :class:`~torch.distributed.rpc.RRef`\n        pointing to the remote module.\n\n        The arguments of ``forward_async`` and ``forward`` are the same as\n        the ``forward`` method of the module returned by the ``module_cls``.\n\n        Apart from ``forward_async`` and ``forward``, no other methods are supported from nn.Module for now.\n\n        Particularly, to create a hybrid model, typically the local modules should be\n        created outside of remote modules, rather than as submodules of any remote module (by calling ``add_module``).\n        Hybrid Example:\n                >>> class HybridModel(nn.Module):\n                >>>     def __init__(self):\n                >>>         nn.Module.__init__(self)\n                >>>         self.remote_embedding = RemoteModule(...)\n                >>>         self.local_linear = nn.Linear(...)\n\n        For example, if ``module_cls`` returns an instance of ``nn.Linear``,\n        that has ``forward`` method signature, ``def forward(input: Tensor) -> Tensor:``,\n        the generated ``RemoteModule`` will have 2 methods in signature of\n        ``def forward(input: Tensor) -> Tensor:`` and\n        ``def forward_async(input: Tensor) -> Future[Tensor]:``.\n\n        .. note::\n            If the remote module is placed on a cuda device,\n            any input CPU tensors will be automatically moved to the same cuda device,\n            and GPU tensors are returned over the wire according to the device map of the remote worker on TensorPipe RPC backend.\n\n        Args:\n            remote_device (str): Device on the destination worker where we\'d like to place this module.\n                The device can be a local device or a remote device specified by one of the following remote\n                formats:\n\n                    1. "rank:<rank>/<device>" (ex: "rank:0/cuda:0").\n                    2. "<worker_name>/<device>" (ex: "trainer0/cuda:0").\n\n                In addition, the device field can be optional and the default value is "cpu".\n            module_cls (nn.Module): For example,\n                >>> class MyModule(nn.Module):\n                >>>     def forward(input):\n                >>>         return input + 1\n                >>>\n                >>> module_cls = MyModule\n            args (Sequence, optional): args to be passed to ``module_cls``.\n            kwargs (Dict, optional): kwargs to be passed to ``module_cls``.\n            _module_interface_cls (type, optional): The TorchScript interface type for the module\n                to be created. The type object should be decorated by @torch.jit.interface.\n                If not provided, the generated RemoteModule is not torchscript-able.\n                Warning, this is an experimental API and susceptible to frequent changes.\n\n        Returns:\n            A remote module instance which wraps the :class:`~nn.Module` created by the\n            user-provided ``module_cls``, it has a blocking ``forward`` method and an\n            asynchronous ``forward_async`` method that returns a future of the ``forward`` call\n            on the user-provided module on the remote side.\n\n        Example::\n            Run the following code in two different processes:\n\n            >>> # xdoctest: +SKIP("distributed")\n            >>> # On worker 0:\n            >>> import torch\n            >>> import torch.distributed.rpc as rpc\n            >>> from torch import nn, Tensor\n            >>> from torch.distributed.nn.api.remote_module import RemoteModule\n            >>>\n            >>> rpc.init_rpc("worker0", rank=0, world_size=2)\n            >>> remote_linear_module = RemoteModule(\n            >>>     "worker1/cpu", nn.Linear, args=(20, 30),\n            >>> )\n            >>> input = torch.randn(128, 20)\n            >>> ret_fut = remote_linear_module.forward_async(input)\n            >>> ret = ret_fut.wait()\n            >>> rpc.shutdown()\n\n            >>> # On worker 1:\n            >>> import torch\n            >>> import torch.distributed.rpc as rpc\n            >>>\n            >>> rpc.init_rpc("worker1", rank=1, world_size=2)\n            >>> rpc.shutdown()\n        '
        super().__init__()
        enable_moving_cpu_tensors_to_cuda = self._prepare_init(remote_device)
        args = args if args is not None else ()
        kwargs = kwargs if kwargs is not None else {}
        if _module_interface_cls is not None:
            self.is_scriptable = True
            fut = rpc.rpc_async(self.on, _instantiate_template, (_module_interface_cls, enable_moving_cpu_tensors_to_cuda))
            self._init_template(_module_interface_cls, enable_moving_cpu_tensors_to_cuda)
            fut = rpc.rpc_async(self.on, _instantiate_template, (_module_interface_cls, enable_moving_cpu_tensors_to_cuda))
            fut.wait()
            self.module_rref = rpc.rpc_sync(self.on, _create_module_with_interface, (module_cls, args, kwargs, self.device, _module_interface_cls))
        else:
            self.is_scriptable = False
            self.generated_methods = _NON_SCRIPTABLE_REMOTE_MODULE_MODULE._generated_methods
            self.module_rref = rpc.remote(self.on, _create_module, (module_cls, args, kwargs, self.device))
        self._install_generated_methods()
        self._check_attribute_picklability()

    def remote_parameters(self, recurse: bool=True) -> List[rpc.RRef[Parameter]]:
        if False:
            while True:
                i = 10
        "\n        Return a list of :class:`~torch.distributed.rpc.RRef` pointing to the remote module's parameters.\n\n        This can typically be used in conjunction\n        with :class:`~torch.distributed.optim.DistributedOptimizer`.\n\n        Args:\n            recurse (bool): if True, then returns parameters of the remote\n                module and all submodules of the remote module. Otherwise,\n                returns only parameters that are direct members of the\n                remote module.\n\n        Returns:\n            A list of :class:`~torch.distributed.rpc.RRef` (``List[RRef[nn.Parameter]]``)\n            to remote module's parameters.\n        "
        return rpc.rpc_sync(self.on, _param_rrefs, args=(self.module_rref, recurse))

    def get_module_rref(self) -> rpc.RRef[nn.Module]:
        if False:
            return 10
        'Return an :class:`~torch.distributed.rpc.RRef` (``RRef[nn.Module]``) pointing to the remote module.'
        return self.module_rref

    @torch.jit.export
    def __getstate__(self):
        if False:
            return 10
        raise RuntimeError('Cannot pickle RemoteModule in python pickler. RemoteModule can only be pickled when using RPC')

    @torch.jit.export
    def __setstate__(self, state):
        if False:
            return 10
        raise RuntimeError('Cannot unpickle RemoteModule in python pickler. RemoteModule can only be unpickled when using RPC')

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        _raise_not_supported(self.register_buffer.__name__)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        if False:
            return 10
        _raise_not_supported(self.register_parameter.__name__)

    def add_module(self, name: str, module: Optional[Module]) -> None:
        if False:
            while True:
                i = 10
        _raise_not_supported(self.add_module.__name__)

    def apply(self: T, fn: Callable[[Module], None]) -> T:
        if False:
            for i in range(10):
                print('nop')
        _raise_not_supported(self.apply.__name__)

    def cuda(self: T, device: Optional[Union[int, device]]=None) -> T:
        if False:
            for i in range(10):
                print('nop')
        _raise_not_supported(self.cuda.__name__)

    def ipu(self: T, device: Optional[Union[int, device]]=None) -> T:
        if False:
            for i in range(10):
                print('nop')
        _raise_not_supported(self.ipu.__name__)

    def xpu(self: T, device: Optional[Union[int, device]]=None) -> T:
        if False:
            print('Hello World!')
        _raise_not_supported(self.xpu.__name__)

    def cpu(self: T) -> T:
        if False:
            i = 10
            return i + 15
        _raise_not_supported(self.cpu.__name__)

    def type(self: T, dst_type: Union[dtype, str]) -> T:
        if False:
            return 10
        _raise_not_supported(self.type.__name__)

    def float(self: T) -> T:
        if False:
            i = 10
            return i + 15
        _raise_not_supported(self.float.__name__)

    def double(self: T) -> T:
        if False:
            for i in range(10):
                print('nop')
        _raise_not_supported(self.double.__name__)

    def half(self: T) -> T:
        if False:
            print('Hello World!')
        _raise_not_supported(self.half.__name__)

    def bfloat16(self: T) -> T:
        if False:
            print('Hello World!')
        _raise_not_supported(self.bfloat16.__name__)

    def to(self, *args, **kwargs) -> T:
        if False:
            print('Hello World!')
        _raise_not_supported(self.to.__name__)

    def register_backward_hook(self, hook: Callable[[Module, _grad_t, _grad_t], Union[None, _grad_t]]) -> RemovableHandle:
        if False:
            i = 10
            return i + 15
        _raise_not_supported(self.register_backward_hook.__name__)

    def register_forward_pre_hook(self, hook: Union[Callable[[T, Tuple[Any, ...]], Optional[Any]], Callable[[T, Tuple[Any, ...], Dict[str, Any]], Optional[Tuple[Any, Dict[str, Any]]]]], prepend: bool=False, with_kwargs: bool=False) -> RemovableHandle:
        if False:
            for i in range(10):
                print('nop')
        _raise_not_supported(self.register_forward_pre_hook.__name__)

    def register_forward_hook(self, hook: Union[Callable[[T, Tuple[Any, ...], Any], Optional[Any]], Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]]], prepend: bool=False, with_kwargs: bool=False) -> RemovableHandle:
        if False:
            print('Hello World!')
        _raise_not_supported(self.register_forward_hook.__name__)

    def state_dict(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        _raise_not_supported(self.state_dict.__name__)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool=True, assign: bool=False):
        if False:
            while True:
                i = 10
        _raise_not_supported(self.load_state_dict.__name__)

    def parameters(self, recurse: bool=True) -> Iterator[Parameter]:
        if False:
            return 10
        raise ValueError('Method ``parameters`` not supported for RemoteModule. Please use ``remote_parameters`` instead.')

    def named_parameters(self, prefix: str='', recurse: bool=True, remove_duplicate: bool=True) -> Iterator[Tuple[str, Parameter]]:
        if False:
            for i in range(10):
                print('nop')
        _raise_not_supported(self.named_parameters.__name__)

    def buffers(self, recurse: bool=True) -> Iterator[Tensor]:
        if False:
            i = 10
            return i + 15
        _raise_not_supported(self.buffers.__name__)

    def named_buffers(self, prefix: str='', recurse: bool=True, remove_duplicate: bool=True) -> Iterator[Tuple[str, Tensor]]:
        if False:
            while True:
                i = 10
        _raise_not_supported(self.named_buffers.__name__)

    def children(self) -> Iterator[Module]:
        if False:
            print('Hello World!')
        _raise_not_supported(self.children.__name__)

    def named_children(self) -> Iterator[Tuple[str, Module]]:
        if False:
            i = 10
            return i + 15
        _raise_not_supported(self.named_children.__name__)

    def modules(self) -> Iterator[Module]:
        if False:
            return 10
        _raise_not_supported(self.modules.__name__)

    def named_modules(self, memo: Optional[Set[Module]]=None, prefix: str='', remove_duplicate: bool=True):
        if False:
            i = 10
            return i + 15
        _raise_not_supported(self.named_modules.__name__)

    def train(self: T, mode: bool=True) -> T:
        if False:
            print('Hello World!')
        return self.module_rref.rpc_sync().train()

    def eval(self: T) -> T:
        if False:
            i = 10
            return i + 15
        return self.module_rref.rpc_sync().eval()

    def requires_grad_(self: T, requires_grad: bool=True) -> T:
        if False:
            for i in range(10):
                print('nop')
        _raise_not_supported(self.requires_grad_.__name__)

    def zero_grad(self, set_to_none: bool=True) -> None:
        if False:
            while True:
                i = 10
        _raise_not_supported(self.zero_grad.__name__)

    def share_memory(self: T) -> T:
        if False:
            i = 10
            return i + 15
        _raise_not_supported(self.share_memory.__name__)

    def extra_repr(self) -> str:
        if False:
            print('Hello World!')
        _raise_not_supported(self.extra_repr.__name__)

    def _prepare_init(self, remote_device_str: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Prepare the initialization and returns whether to enable automatically moving CPU tensors to CUDA devices.'
        assert rpc._is_current_rpc_agent_set(), 'RemoteModule only works in RPC.'
        remote_device = _remote_device(remote_device_str)
        self.on = remote_device.worker_name() if remote_device.worker_name() is not None else remote_device.rank()
        self.device = str(remote_device.device())
        agent = rpc._get_current_rpc_agent()
        self.is_device_map_set = bool(agent._get_device_map(agent.get_worker_info(self.on)))
        enable_moving_cpu_tensors_to_cuda = torch.device(self.device).type == 'cuda'
        return enable_moving_cpu_tensors_to_cuda

    def _init_template(self, module_interface_cls, enable_moving_cpu_tensors_to_cuda):
        if False:
            for i in range(10):
                print('nop')
        'Instantiate template on local side.'
        generated_module = instantiator.instantiate_scriptable_remote_module_template(module_interface_cls, enable_moving_cpu_tensors_to_cuda)
        self.generated_methods = generated_module._generated_methods

    def _check_attribute_picklability(self):
        if False:
            i = 10
            return i + 15
        'Check if all the attribute has explicitly defined whether to be pickled (i.e., picklability).'
        for k in self.__dict__.keys():
            if k not in _REMOTE_MODULE_PICKLED_ATTRIBUTES and k not in _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING:
                raise AttributeError(f'Attribute {k} must be either in ``_REMOTE_MODULE_PICKLED_ATTRIBUTES`` or ``_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING``.')

    def _install_generated_methods(self):
        if False:
            return 10
        for method in self.generated_methods:
            method_name = method.__name__
            method = torch.jit.export(method)
            setattr(self, method_name, types.MethodType(method, self))

    @staticmethod
    def init_from_module_rref(remote_device: str, module_rref: rpc.RRef[nn.Module], _module_interface_cls: Any=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Besides the constructor, a RemoteModule instance can also be initialized given a module RRef.\n\n        This alternate initialization method can be particularly useful if we want to create multiple\n        RemoteModule instances that share the same underlying module and reduce memory consumption.\n\n        Moreover, this also provides a workaround for passing script RemoteModule over RPC,\n        which is not supported. The recommended way is as follows:\n\n            1. the sender creates a RemoteModule;\n            2. the sender sends its ``module_rref`` over RPC;\n            3. the receiver calls this method to initialize another RemoteModule using the same ``module_rref``.\n\n        Example::\n            Run the following code in two different processes:\n\n            >>> # xdoctest: +SKIP("distributed")\n            >>> # On worker 0:\n            >>> import torch\n            >>> import torch.distributed.rpc as rpc\n            >>> from torch import nn, Tensor\n            >>> from torch.distributed.nn.api.remote_module import RemoteModule\n            >>>\n            >>> rpc.init_rpc("worker0", rank=0, world_size=2)\n            >>> remote_module = RemoteModule(\n            >>>     "worker1/cpu", nn.Linear, args=(20, 30),\n            >>> )\n            >>>\n            >>> remote_module1 = rpc.rpc_sync(\n            >>>     "worker1/cpu",\n            >>>     RemoteModule.init_from_module_rref,\n            >>>     ("worker1/cpu", remote_module1.get_module_rref()),\n            >>> )\n            >>> rpc.shutdown()\n\n            >>> # On worker 1:\n            >>> import torch\n            >>> import torch.distributed.rpc as rpc\n            >>>\n            >>> rpc.init_rpc("worker1", rank=1, world_size=2)\n            >>> rpc.shutdown()\n\n        Args:\n            remote_device (str): Device on the destination worker where we\'d like to place this module.\n                The device can be a local device or a remote device specified by one of the following remote\n                formats:\n\n                    1. "rank:<rank>/<device>" (ex: "rank:0/cuda:0").\n                    2. "<worker_name>/<device>" (ex: "trainer0/cuda:0").\n\n                In addition, the device field can be optional and the default value is "cpu".\n            module_rref (RRef[nn.Module]): The module reference shared by both the caller and\n                the created remote module.\n            _module_interface_cls (type, optional): The TorchScript interface type for the module\n                to be created. The type object should be decorated by @torch.jit.interface.\n                If not provided, the generated RemoteModule is not torchscript-able.\n                Warning, this is an experimental API and susceptible to frequent changes.\n\n        Returns:\n            A remote module instance which wraps the :class:`~nn.Module` created by the\n            user-provided ``module_rref``, it has a blocking ``forward`` method and an\n            asynchronous ``forward_async`` method that returns a future of the ``forward`` call\n            on the user-provided module on the remote side.\n        '
        remote_module = object.__new__(RemoteModule)
        enable_moving_cpu_tensors_to_cuda = remote_module._prepare_init(remote_device)
        if _module_interface_cls is not None:
            remote_module.is_scriptable = True
            remote_module._init_template(_module_interface_cls, enable_moving_cpu_tensors_to_cuda)
        else:
            remote_module.is_scriptable = False
            remote_module.generated_methods = _NON_SCRIPTABLE_REMOTE_MODULE_MODULE._generated_methods
        remote_module.module_rref = module_rref
        remote_module._install_generated_methods()
        remote_module._check_attribute_picklability()
        return remote_module

class RemoteModule(_RemoteModule):
    """
        A RemoteModule instance can only be created after RPC initialization.

        It creates a user-specified module on a specified remote node.
        It behaves like a regular ``nn.Module`` except that the ``forward`` method is
        executed on the remote node.
        It takes care of autograd recording to ensure the backward pass propagates
        gradients back to the corresponding remote module.

        It generates two methods ``forward_async`` and ``forward`` based on the
        signature of the ``forward`` method of ``module_cls``. ``forward_async``
        runs asynchronously and returns a Future. The arguments of ``forward_async``
        and ``forward`` are the same as the ``forward`` method of the module
        returned by the ``module_cls``.

        For example, if ``module_cls`` returns an instance of ``nn.Linear``,
        that has ``forward`` method signature: ``def forward(input: Tensor) -> Tensor:``,
        the generated ``RemoteModule`` will have 2 methods with the signatures:

        | ``def forward(input: Tensor) -> Tensor:``
        | ``def forward_async(input: Tensor) -> Future[Tensor]:``

    Args:
        remote_device (str): Device on the destination worker where we'd like to place this module.
            The format should be "<workername>/<device>", where the device field can be parsed as torch.device type.
            E.g., "trainer0/cpu", "trainer0", "ps0/cuda:0".
            In addition, the device field can be optional and the default value is "cpu".
        module_cls (nn.Module): Class for the module to be created remotely. For example,

            >>> class MyModule(nn.Module):
            >>>     def forward(input):
            >>>         return input + 1
            >>>
            >>> module_cls = MyModule

        args (Sequence, optional): args to be passed to ``module_cls``.
        kwargs (Dict, optional): kwargs to be passed to ``module_cls``.

    Returns:
        A remote module instance which wraps the :class:`~nn.Module` created by the
        user-provided ``module_cls``, it has a blocking ``forward`` method and an
        asynchronous ``forward_async`` method that returns a future of the ``forward`` call
        on the user-provided module on the remote side.

    Example::
        Run the following code in two different processes:

        >>> # xdoctest: +SKIP("distributed")
        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> from torch import nn, Tensor
        >>> from torch.distributed.nn.api.remote_module import RemoteModule
        >>>
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> remote_linear_module = RemoteModule(
        >>>     "worker1/cpu", nn.Linear, args=(20, 30),
        >>> )
        >>> input = torch.randn(128, 20)
        >>> ret_fut = remote_linear_module.forward_async(input)
        >>> ret = ret_fut.wait()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>>
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

        Furthermore, a more practical example that is combined with
        `DistributedDataParallel <https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel>`__ (DDP)
        can be found in this `tutorial <https://pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html>`__.
    """

    def __init__(self, remote_device: str, module_cls: Type[nn.Module], args: Optional[Tuple]=None, kwargs: Optional[Dict[str, Any]]=None):
        if False:
            while True:
                i = 10
        super().__init__(remote_device, module_cls, args, kwargs)

def _remote_module_receiver(*remote_module_pickled_attrs):
    if False:
        print('Hello World!')
    'Deserializes a RemoteModule.'
    serialized_remote_module = _SerializedRemoteModule._make(remote_module_pickled_attrs)
    m = object.__new__(RemoteModule)
    m.__dict__.update(serialized_remote_module._asdict())
    m.module_rref = rpc.PyRRef._deserialize(m.module_rref)
    for method in m.generated_methods:
        method_name = method.__name__
        method = torch.jit.export(method)
        setattr(m, method_name, types.MethodType(method, m))
    return m

def _remote_module_reducer(remote_module):
    if False:
        print('Hello World!')
    'Serialize a RemoteModule.'
    pickled_attrs = {}
    for (k, v) in remote_module.__dict__.items():
        if k == 'module_rref':
            pickled_attrs[k] = v._serialize()
        elif k in _REMOTE_MODULE_PICKLED_ATTRIBUTES:
            pickled_attrs[k] = v
        elif k not in _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING:
            print(f'The new attribute ``{k}`` of RemoteModule is ignored during RPC pickling. To pickle this attribute, please add it to ``_REMOTE_MODULE_PICKLED_ATTRIBUTES``. Otherwise, please explicitly add it to ``_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING``.', file=sys.stderr)
    return (_remote_module_receiver, tuple(pickled_attrs.values()))

def _recursive_script_module_receiver(recursive_script_module_serialized):
    if False:
        while True:
            i = 10
    'Deserializes a RecursiveScriptModule that does not contain a script RemoteModule.'
    f = io.BytesIO(recursive_script_module_serialized)
    m = torch.jit.load(f)
    return m

def _recursive_script_module_reducer(recursive_script_module):
    if False:
        for i in range(10):
            print('nop')
    'Serialize a RecursiveScriptModule that does not contain a script RemoteModule, and raises an error otherwise.'
    if hasattr(recursive_script_module._c, 'module_rref'):
        raise RuntimeError('Passing a script RemoteModule over RPC is not supported. Please create a RemoteModule in the sender, send the `module_rref` to the receiver, and create a new instance on the receiver end by passing this `module_rref`.')
    f = io.BytesIO()
    torch.jit.save(recursive_script_module, f)
    return (_recursive_script_module_receiver, (f.getvalue(),))
_internal_rpc_pickler._register_reducer(RemoteModule, _remote_module_reducer)
_internal_rpc_pickler._register_reducer(torch.jit.RecursiveScriptModule, _recursive_script_module_reducer)