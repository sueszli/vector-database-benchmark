import torch
from torch._C import _rename_privateuse1_backend, _get_privateuse1_backend_name
from typing import List, Optional, Union
__all__ = ['rename_privateuse1_backend', 'generate_methods_for_privateuse1_backend']
_privateuse1_backend_name = 'privateuseone'

def rename_privateuse1_backend(backend_name: str) -> None:
    if False:
        return 10
    '\n    Rename the privateuse1 backend device to make it more convenient to use as a device name within PyTorch APIs.\n\n    The steps are:\n\n    (1) (In C++) implement kernels for various torch operations, and register them\n        to the PrivateUse1 dispatch key.\n    (2) (In python) call torch.utils.rename_privateuse1_backend("foo")\n\n    You can now use "foo" as an ordinary device string in python.\n\n    Note: this API can only be called once per process. Attempting to change\n    the external backend after it\'s already been set will result in an error.\n\n    Note(AMP): If you want to support AMP on your device, you can register a custom backend module.\n    The backend must register a custom backend module with ``torch._register_device_module("foo", BackendModule)``.\n    BackendModule needs to have the following API\'s:\n\n    (1) ``get_amp_supported_dtype() -> List[torch.dtype]``\n        get the supported dtypes on your "foo" device in AMP, maybe the "foo" device supports one more dtype.\n\n    (2) ``is_autocast_enabled() -> bool``\n        check the AMP is enabled or not on your "foo" device.\n\n    (3) ``get_autocast_dtype() -> torch.dtype``\n        get the supported dtype on your "foo" device in AMP, which is set by ``set_autocast_dtype`` or the\n        default dtype, and the default dtype is ``torch.float16``.\n\n    (4) ``set_autocast_enabled(bool) -> None``\n        enable the AMP or not on your "foo" device.\n\n    (5) ``set_autocast_dtype(dtype) -> None``\n        set the supported dtype on your "foo" device in AMP, and the dtype be contained in the dtypes got\n        from ``get_amp_supported_dtype``.\n\n    Note(random): If you want to support to set seed for your device, BackendModule needs to have the following API\'s:\n\n    (1) ``_is_in_bad_fork() -> bool``\n        Return ``True`` if now it is in bad_fork, else return ``False``.\n\n    (2) ``manual_seed_all(seed int) -> None``\n        Sets the seed for generating random numbers for your devices.\n\n    (3) ``device_count() -> int``\n        Returns the number of "foo"s available.\n\n    (4) ``get_rng_state(device: Union[int, str, torch.device] = \'foo\') -> Tensor``\n        Returns a list of ByteTensor representing the random number states of all devices.\n\n    (5) ``set_rng_state(new_state: Tensor, device: Union[int, str, torch.device] = \'foo\') -> None``\n        Sets the random number generator state of the specified "foo" device.\n\n    And there are some common funcs:\n\n    (1) ``is_available() -> bool``\n        Returns a bool indicating if "foo" is currently available.\n\n    (2) ``current_device() -> int``\n        Returns the index of a currently selected device.\n\n    For more details, see https://pytorch.org/tutorials/advanced/extend_dispatcher.html#get-a-dispatch-key-for-your-backend\n    For an existing example, see https://github.com/bdhirsh/pytorch_open_registration_example\n\n    Example::\n\n        >>> # xdoctest: +SKIP("failing")\n        >>> torch.utils.rename_privateuse1_backend("foo")\n        # This will work, assuming that you\'ve implemented the right C++ kernels\n        # to implement torch.ones.\n        >>> a = torch.ones(2, device="foo")\n\n    '
    _rename_privateuse1_backend(backend_name)
    global _privateuse1_backend_name
    _privateuse1_backend_name = backend_name

def _check_register_once(module, attr):
    if False:
        i = 10
        return i + 15
    if hasattr(module, attr):
        raise RuntimeError(f'The custom device module of {module} has already been registered with {attr}')

def _normalization_device(custom_backend_name: str, device: Optional[Union[int, str, torch.device]]=None) -> int:
    if False:
        i = 10
        return i + 15

    def _get_current_device_index():
        if False:
            print('Hello World!')
        _get_device_index = 'current_device'
        if hasattr(torch, custom_backend_name) and hasattr(getattr(torch, custom_backend_name), _get_device_index):
            return getattr(getattr(torch, custom_backend_name), _get_device_index)()
        else:
            return 0
    if device is None:
        return _get_current_device_index()
    elif isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.type != custom_backend_name:
            raise RuntimeError(f'Invalid device, must be {custom_backend_name} device')
        elif device.index is None:
            device_idx = _get_current_device_index()
        else:
            device_idx = device.index
    else:
        device_idx = device
    return device_idx

def _generate_tensor_methods_for_privateuse1_backend(custom_backend_name: str) -> None:
    if False:
        print('Hello World!')

    @property
    def wrap_tensor_backend(self: torch.Tensor) -> bool:
        if False:
            print('Hello World!')
        return self.device.type == custom_backend_name
    _check_register_once(torch.Tensor, f'is_{custom_backend_name}')
    setattr(torch.Tensor, f'is_{custom_backend_name}', wrap_tensor_backend)

    def wrap_tensor_to(self: torch.Tensor, device: Optional[Union[int, torch.device]]=None, non_blocking=False, **kwargs) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        'Perform Tensor device conversion. Call the to operator implementation.\n\n        .. note::\n            If the ``self`` Tensor already\n            has the correct :class:`torch.device`, then ``self`` is returned.\n            Otherwise, the returned tensor is a copy of ``self`` with the desired :class:`torch.device`.\n\n        Args:\n            device (int, optional): if specified, all parameters will be copied to that device\n            non_blocking (bool): If ``True`` and the source is in pinned memory,\n                the copy will be asynchronous with respect to the host. Otherwise,\n                the argument has no effect.\n            **kwargs (dict): For compatibility, may contain the key ``memory_format`` argument.\n        '
        device_idx = _normalization_device(custom_backend_name, device)
        return self.to(device=torch.device(f'{custom_backend_name}:{device_idx}'), non_blocking=non_blocking, **kwargs)
    _check_register_once(torch.Tensor, custom_backend_name)
    setattr(torch.Tensor, custom_backend_name, wrap_tensor_to)

def _generate_module_methods_for_privateuse1_backend(custom_backend_name: str) -> None:
    if False:
        return 10
    if not hasattr(torch.Tensor, custom_backend_name):
        raise RuntimeError(f"Can not automatically generate {custom_backend_name}() method for torch.nn.Module.Because torch.Tensor doesn't has the method {custom_backend_name}().For this error, you can try setting for_tensor=True.")

    def wrap_module_to(self: torch.nn.modules.module.T, device: Optional[Union[int, torch.device]]=None) -> torch.nn.modules.module.T:
        if False:
            for i in range(10):
                print('nop')
        'Move all model parameters and buffers to the custom device.\n\n        This also makes associated parameters and buffers different objects. So\n        it should be called before constructing optimizer if the module will\n        live on device while being optimized.\n\n        .. note::\n            This method modifies the module in-place.\n\n        Args:\n            device (int, optional): if specified, all parameters will be copied to that device\n        '
        return self._apply(lambda t: getattr(t, custom_backend_name)(device))
    _check_register_once(torch.nn.Module, custom_backend_name)
    setattr(torch.nn.Module, custom_backend_name, wrap_module_to)

def _generate_storage_methods_for_privateuse1_backend(custom_backend_name: str, unsupported_dtype: Optional[List[torch.dtype]]=None) -> None:
    if False:
        i = 10
        return i + 15

    @property
    def wrap_storage_backend(self: torch.storage._StorageBase) -> bool:
        if False:
            while True:
                i = 10
        'Return the internal :class:`torch.UntypedStorage`.'
        return self.device.type == custom_backend_name
    _check_register_once(torch.storage._StorageBase, f'is_{custom_backend_name}')
    setattr(torch.storage._StorageBase, f'is_{custom_backend_name}', wrap_storage_backend)

    def wrap_storage_to(self, device=None, non_blocking=False):
        if False:
            while True:
                i = 10
        'Return a copy of this object in custom device memory.\n\n        If this object is already in device memory and on the correct device, then\n        no copy is performed and the original object is returned.\n\n        Args:\n            device (int): The destination device id. Defaults to the current device.\n            non_blocking (bool): If ``True`` and the source is in pinned memory,\n            the copy will be asynchronous with respect to the host. Otherwise,\n            the argument has no effect.\n        '
        device_idx = _normalization_device(custom_backend_name, device)
        if getattr(self, f'is_{custom_backend_name}'):
            if self.get_device() == device_idx:
                return self
        if self.is_sparse:
            raise RuntimeError(f'Can not support a sparse storage move to {custom_backend_name} backend')
        untyped_storage = torch.UntypedStorage(self.size(), device=torch.device(f'{custom_backend_name}:{device_idx}'))
        untyped_storage.copy_(self, non_blocking)
        return untyped_storage
    _check_register_once(torch.storage._StorageBase, custom_backend_name)
    setattr(torch.storage._StorageBase, custom_backend_name, wrap_storage_to)

    @property
    def wrap_typed_storage_backend(self: torch.storage.TypedStorage) -> bool:
        if False:
            return 10
        torch.storage._warn_typed_storage_removal()
        return self._untyped_storage.device.type == custom_backend_name
    _check_register_once(torch.TypedStorage, f'is_{custom_backend_name}')
    setattr(torch.storage.TypedStorage, f'is_{custom_backend_name}', wrap_typed_storage_backend)

    def wrap_typed_storage_to(self: torch.storage.TypedStorage, device=None, non_blocking=False, **kwargs) -> torch.storage.TypedStorage:
        if False:
            for i in range(10):
                print('nop')
        torch.storage._warn_typed_storage_removal()
        if unsupported_dtype and self.dtype in unsupported_dtype:
            raise RuntimeError(f'Cannot create {custom_backend_name} storage as {self.dtype} dtype is not supported by this backend')
        custom_backend_storage: torch.UntypedStorage = getattr(self._untyped_storage, custom_backend_name)(device, non_blocking, **kwargs)
        return self._new_wrapped_storage(custom_backend_storage)
    _check_register_once(torch.TypedStorage, custom_backend_name)
    setattr(torch.TypedStorage, custom_backend_name, wrap_typed_storage_to)

def generate_methods_for_privateuse1_backend(for_tensor: bool=True, for_module: bool=True, for_storage: bool=False, unsupported_dtype: Optional[List[torch.dtype]]=None) -> None:
    if False:
        while True:
            i = 10
    '\n    Automatically generate attributes and methods for the custom backend after rename privateuse1 backend.\n\n    In the default scenario, storage-related methods will not be generated automatically.\n\n    When you implement kernels for various torch operations, and register them to the PrivateUse1 dispatch key.\n    And call the function torch.rename_privateuse1_backend("foo") to rename your backend name.\n    At this point, you can easily register specific methods and attributes by calling this function.\n    Just like torch.Tensor.foo(), torch.Tensor.is_foo, torch.Storage.foo(), torch.Storage.is_foo.\n\n    Note: We recommend you use generic functions (check devices are equal or to(device=)).\n    We provide these methods for convenience only and they will be "monkey patched" onto the objects\n    and so will not be properly typed. For Storage methods generate, if you need to support sparse data storage,\n    you need to extend the implementation yourself.\n\n    Args:\n        for_tensor (bool): whether register related methods for torch.Tensor class.\n        for_module (bool): whether register related methods for torch.nn.Module class.\n        for_storage (bool): whether register related methods for torch.Storage class.\n        unsupported_dtype (List[torch.dtype]): takes effect only when the storage method needs to be generated,\n            indicating that the storage does not support the torch.dtype type.\n\n    Example::\n\n        >>> # xdoctest: +SKIP("failing")\n        >>> torch.utils.rename_privateuse1_backend("foo")\n        >>> torch.utils.generate_methods_for_privateuse1_backend()\n        # Then automatically generate backend-related attributes and methods.\n        >>> a = torch.tensor(2).foo()\n        >>> a.is_foo\n        >>> hasattr(torch.nn.Module, \'foo\')\n    '
    custom_backend_name = _get_privateuse1_backend_name()
    if for_tensor:
        _generate_tensor_methods_for_privateuse1_backend(custom_backend_name)
    if for_module:
        _generate_module_methods_for_privateuse1_backend(custom_backend_name)
    if for_storage:
        _generate_storage_methods_for_privateuse1_backend(custom_backend_name, unsupported_dtype)

def _get_custom_mod_func(func_name: str):
    if False:
        while True:
            i = 10
    '\n    Return the func named `func_name` defined in custom device module. If not defined,\n    return `None`. And the func is registered with `torch.utils.rename_privateuse1_backend(\'foo\')`\n    and `torch._register_device_module(\'foo\', BackendModule)`.\n    If the custom device module or the func is not defined, it will give warning or error message.\n    Args:\n        func_name (str): return the callable func named func_name defined in custom device module.\n    Example::\n        class DummyfooModule:\n            @staticmethod\n            def is_available():\n                return True\n            @staticmethod\n            def func_name(*args, **kwargs):\n                ....\n        torch.utils.rename_privateuse1_backend("foo")\n        torch._register_device_module("foo", DummyfooModule)\n        foo_is_available_func = torch.utils.backend_registration._get_custom_mod_func("is_available")\n        if foo_is_available_func:\n            foo_is_available = foo_is_available_func()\n        func_ = torch.utils.backend_registration._get_custom_mod_func("func_name")\n        if func_:\n            result = func_(*args, **kwargs)\n    Attention: This function is not meant to be used directly by users, which is why\n    it is marked as private. It is a convenience function for backend implementers to\n    more easily call the hooks into their backend extensions.\n    '
    assert isinstance(func_name, str), f'func_name must be `str`, but got `{type(func_name)}`.'
    backend_name = _get_privateuse1_backend_name()
    custom_device_mod = getattr(torch, backend_name, None)
    function = getattr(custom_device_mod, func_name, None)
    if custom_device_mod is None or function is None:
        message = f'Try to call torch.{backend_name}.{func_name}. The backend must register a custom backend '
        message += f"module with `torch._register_device_module('{backend_name}', BackendModule)`. And "
        message += f"BackendModule needs to have the following API's:\n `{func_name}(*args, **kwargs)`. \n"
        raise RuntimeError(message)
    return function