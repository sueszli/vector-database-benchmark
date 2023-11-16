import copyreg
import enum
import functools
import warnings
from collections import OrderedDict
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch._C as _C
import torch.utils.hooks as hooks
from torch._namedtensor_internals import check_serializing_named_tensor, is_ellipsis, resolve_ellipsis, single_ellipsis_index, unzip_namedshape, update_names
from torch.overrides import get_default_nowrap_functions, handle_torch_function, has_torch_function, has_torch_function_unary, has_torch_function_variadic
from torch.utils.dlpack import DLDeviceType

def _handle_torch_function_and_wrap_type_error_to_not_implemented(f):
    if False:
        i = 10
        return i + 15
    assigned = functools.WRAPPER_ASSIGNMENTS

    @functools.wraps(f, assigned=assigned)
    def wrapped(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        try:
            if has_torch_function(args):
                return handle_torch_function(wrapped, args, *args, **kwargs)
            return f(*args, **kwargs)
        except TypeError:
            return NotImplemented
    return wrapped

def _rebuild_from_type(func, type, args, dict):
    if False:
        i = 10
        return i + 15
    if type is Tensor:
        return func(*args)
    ret = func(*args).as_subclass(type)
    ret.__dict__ = dict
    return ret

def _rebuild_from_type_v2(func, new_type, args, state):
    if False:
        while True:
            i = 10
    ret = func(*args)
    if type(ret) is not new_type:
        ret = ret.as_subclass(new_type)
    if getattr(ret.__class__, '__setstate__', Tensor.__setstate__) is not Tensor.__setstate__:
        ret.__setstate__(state)
    else:
        ret = torch._utils._set_obj_state(ret, state)
    return ret

class Tensor(torch._C.TensorBase):

    def __deepcopy__(self, memo):
        if False:
            return 10
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__deepcopy__, (self,), self, memo)
        if not self.is_leaf:
            raise RuntimeError('Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment.  If you were attempting to deepcopy a module, this may be because of a torch.nn.utils.weight_norm usage, see https://github.com/pytorch/pytorch/pull/103001')
        if id(self) in memo:
            return memo[id(self)]
        with torch.no_grad():
            if self.is_sparse or self.device.type in ['lazy', 'xla', 'mtia', 'mps', 'ort', 'meta', 'ipu'] or (not torch._C._has_storage(self) and self.device.type == torch._C._get_privateuse1_backend_name()) or (type(self) is not Tensor and self.data_ptr() == 0):
                new_tensor = self.clone()
                if type(new_tensor) is not type(self):
                    raise RuntimeError('The default implementation of __deepcopy__() for wrapper subclasses only works for subclass types that implement clone() and for which cloning returns another instance of the same subclass. You should either properly implement clone() for your subclass or override __deepcopy__() if it is intended behavior for clone() to return an instance of a different type.')
            else:
                new_storage = self._typed_storage()._deepcopy(memo)
                if self.is_quantized:
                    quantizer_params: Union[Tuple[torch.qscheme, float, int], Tuple[torch.qscheme, Tensor, Tensor, int]]
                    if self.qscheme() == torch.per_tensor_affine:
                        quantizer_params = (self.qscheme(), self.q_scale(), self.q_zero_point())
                    elif self.qscheme() in (torch.per_channel_affine, torch.per_channel_affine_float_qparams):
                        quantizer_params = (self.qscheme(), self.q_per_channel_scales(), self.q_per_channel_zero_points(), self.q_per_channel_axis())
                    else:
                        raise RuntimeError(f'Unsupported qscheme {self.qscheme()} in deepcopy')
                    new_tensor = torch._utils._rebuild_qtensor(torch.storage.TypedStorage(wrap_storage=new_storage._untyped_storage, dtype=self.dtype, _internal=True), self.storage_offset(), self.size(), self.stride(), quantizer_params, self.requires_grad, self._backward_hooks)
                    if type(new_tensor) is not type(self):
                        raise RuntimeError("The default implementation of __deepcopy__() for quantized tensors expects the tensor returned by torch._utils._rebuild_qtensor() to match the type of the instance being copied. If you encounter this, please open an issue on PyTorch's GitHub.")
                else:
                    new_tensor = self.new_empty([])
                    if type(new_tensor) is not type(self):
                        raise RuntimeError('The default implementation of __deepcopy__() for non-wrapper subclasses only works for subclass types that implement new_empty() and for which that function returns another instance of the same subclass. You should either properly implement new_empty() for your subclass or override __deepcopy__() if it is intended behavior for new_empty() to return an instance of a different type.')
                    new_tensor.set_(new_storage, self.storage_offset(), self.size(), self.stride())
                    if self.is_conj():
                        new_tensor = new_tensor.conj_physical()
                    if self.is_neg():
                        new_tensor = new_tensor.neg()
            if self.requires_grad:
                new_tensor.requires_grad_()
            if self.grad is not None:
                new_tensor.grad = self.grad.__deepcopy__(memo)
            if type(self) is not Tensor:
                if type(new_tensor) is not type(self):
                    raise RuntimeError("Type of deepcopy result does not match the type of the source tensor. If you encounter this, please open an issue on PyTorch's GitHub.")
                slots_to_save = copyreg._slotnames(self.__class__)
                for slot in slots_to_save:
                    if hasattr(self, slot):
                        setattr(new_tensor, slot, deepcopy(getattr(self, slot), memo))
            new_tensor.__dict__ = deepcopy(self.__dict__, memo)
            memo[id(self)] = new_tensor
            return new_tensor

    def __reduce_ex__(self, proto):
        if False:
            print('Hello World!')
        state = torch._utils._get_obj_state(self)
        if type(self) is Tensor and (not state):
            return self._reduce_ex_internal(proto)
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__reduce_ex__, (self,), self, proto)
        (func, args) = self._reduce_ex_internal(proto)
        return (_rebuild_from_type_v2, (func, type(self), args, state))

    def storage(self):
        if False:
            return 10
        '\n        storage() -> torch.TypedStorage\n\n        Returns the underlying :class:`TypedStorage`.\n\n        .. warning::\n\n            :class:`TypedStorage` is deprecated. It will be removed in the future, and\n            :class:`UntypedStorage` will be the only storage class. To access the\n            :class:`UntypedStorage` directly, use :attr:`Tensor.untyped_storage()`.\n        '
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.storage, (self,), self)
        torch.storage._warn_typed_storage_removal(stacklevel=2)
        return self._typed_storage()

    def _typed_storage(self):
        if False:
            while True:
                i = 10
        untyped_storage = self.untyped_storage()
        return torch.TypedStorage(wrap_storage=untyped_storage, dtype=self.dtype, _internal=True)

    def _reduce_ex_internal(self, proto):
        if False:
            while True:
                i = 10
        check_serializing_named_tensor(self)
        torch.utils.hooks.warn_if_has_hooks(self)
        backward_hooks: Dict[Any, Any] = OrderedDict()
        if self.device.type in ['xla', 'mtia', 'ort'] or (not torch._C._has_storage(self) and self.device.type == torch._C._get_privateuse1_backend_name()):
            numpy_tensor = self.cpu().numpy() if self.dtype != torch.bfloat16 else self.cpu().to(torch.float32).numpy()
            return (torch._utils._rebuild_device_tensor_from_numpy, (numpy_tensor, self.dtype, str(self.device), self.requires_grad))
        if self.device.type == 'meta':
            arg_meta = (self.dtype, tuple(self.size()), self.stride(), self.requires_grad)
            return (torch._utils._rebuild_meta_tensor_no_storage, arg_meta)
        if self.is_quantized:
            quantizer_params: Union[Tuple[torch.qscheme, float, int], Tuple[Any, Tensor, Tensor, int]]
            if self.qscheme() == torch.per_tensor_affine:
                quantizer_params = (torch.per_tensor_affine, self.q_scale(), self.q_zero_point())
            elif self.qscheme() in (torch.per_channel_affine, torch.per_channel_affine_float_qparams):
                quantizer_params = (torch.per_channel_affine, self.q_per_channel_scales(), self.q_per_channel_zero_points(), self.q_per_channel_axis())
            else:
                raise RuntimeError(f'Serialization is not supported for tensors of type {self.qscheme()}')
            args_qtensor = (torch.storage.TypedStorage(wrap_storage=self._typed_storage()._untyped_storage, dtype=self.dtype, _internal=True), self.storage_offset(), tuple(self.size()), self.stride(), quantizer_params, self.requires_grad, backward_hooks)
            return (torch._utils._rebuild_qtensor, args_qtensor)
        elif self.is_sparse:
            if self.layout == torch.sparse_coo:
                args_sparse = (self.layout, (self._indices(), self._values(), self.size(), self.is_coalesced()))
            else:
                raise NotImplementedError(f'sparse tensor __reduce_ex__ for layout `{self.layout}`')
            return (torch._utils._rebuild_sparse_tensor, args_sparse)
        elif self.layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:
            if self.layout in {torch.sparse_csr, torch.sparse_bsr}:
                (compressed_indices, plain_indices) = (self.crow_indices(), self.col_indices())
            else:
                (compressed_indices, plain_indices) = (self.ccol_indices(), self.row_indices())
            args_sparse_compressed = (self.layout, (compressed_indices, plain_indices, self.values(), self.size()))
            return (torch._utils._rebuild_sparse_tensor, args_sparse_compressed)
        elif self.is_nested:
            args_nested = (self.values(), self._nested_tensor_size(), self._nested_tensor_strides(), self._nested_tensor_storage_offsets())
            return (torch._utils._rebuild_nested_tensor, args_nested)
        elif self.data_ptr() == 0 and type(self) is not torch.Tensor and (type(self).__torch_dispatch__ is not torch.Tensor.__torch_dispatch__):
            arg_wrapper_subclass = (type(self), self.dtype, tuple(self.size()), self.stride(), self.storage_offset(), self.layout, self.device, self.requires_grad)
            return (torch._utils._rebuild_wrapper_subclass, arg_wrapper_subclass)
        else:
            args = (torch.storage.TypedStorage(wrap_storage=self._typed_storage()._untyped_storage, dtype=self.dtype, _internal=True), self.storage_offset(), tuple(self.size()), self.stride(), self.requires_grad, backward_hooks)
            metadata = torch._utils.get_tensor_metadata(self)
            if metadata:
                args = args + (metadata,)
            return (torch._utils._rebuild_tensor_v2, args)

    def __setstate__(self, state):
        if False:
            while True:
                i = 10
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__setstate__, (self,), self, state)
        if not self.is_leaf:
            raise RuntimeError('__setstate__ can be only called on leaf Tensors')
        if len(state) == 4:
            self.set_(*state)
            return
        elif len(state) == 5:
            self.data = state[0]
            state = (state[3], state[4], state[2])
        (self.requires_grad, _, self._backward_hooks) = state

    def __repr__(self, *, tensor_contents=None):
        if False:
            while True:
                i = 10
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__repr__, (self,), self, tensor_contents=tensor_contents)
        return torch._tensor_str._str(self, tensor_contents=tensor_contents)

    def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
        if False:
            while True:
                i = 10
        "Computes the gradient of current tensor wrt graph leaves.\n\n        The graph is differentiated using the chain rule. If the tensor is\n        non-scalar (i.e. its data has more than one element) and requires\n        gradient, the function additionally requires specifying ``gradient``.\n        It should be a tensor of matching type and location, that contains\n        the gradient of the differentiated function w.r.t. ``self``.\n\n        This function accumulates gradients in the leaves - you might need to zero\n        ``.grad`` attributes or set them to ``None`` before calling it.\n        See :ref:`Default gradient layouts<default-grad-layouts>`\n        for details on the memory layout of accumulated gradients.\n\n        .. note::\n\n            If you run any forward ops, create ``gradient``, and/or call ``backward``\n            in a user-specified CUDA stream context, see\n            :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.\n\n        .. note::\n\n            When ``inputs`` are provided and a given input is not a leaf,\n            the current implementation will call its grad_fn (though it is not strictly needed to get this gradients).\n            It is an implementation detail on which the user should not rely.\n            See https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780 for more details.\n\n        Args:\n            gradient (Tensor or None): Gradient w.r.t. the\n                tensor. If it is a tensor, it will be automatically converted\n                to a Tensor that does not require grad unless ``create_graph`` is True.\n                None values can be specified for scalar Tensors or ones that\n                don't require grad. If a None value would be acceptable then\n                this argument is optional.\n            retain_graph (bool, optional): If ``False``, the graph used to compute\n                the grads will be freed. Note that in nearly all cases setting\n                this option to True is not needed and often can be worked around\n                in a much more efficient way. Defaults to the value of\n                ``create_graph``.\n            create_graph (bool, optional): If ``True``, graph of the derivative will\n                be constructed, allowing to compute higher order derivative\n                products. Defaults to ``False``.\n            inputs (sequence of Tensor): Inputs w.r.t. which the gradient will be\n                accumulated into ``.grad``. All other Tensors will be ignored. If not\n                provided, the gradient is accumulated into all the leaf Tensors that were\n                used to compute the attr::tensors.\n        "
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.backward, (self,), self, gradient=gradient, retain_graph=retain_graph, create_graph=create_graph, inputs=inputs)
        torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)

    def register_hook(self, hook):
        if False:
            print('Hello World!')
        'Registers a backward hook.\n\n        The hook will be called every time a gradient with respect to the\n        Tensor is computed. The hook should have the following signature::\n\n            hook(grad) -> Tensor or None\n\n\n        The hook should not modify its argument, but it can optionally return\n        a new gradient which will be used in place of :attr:`grad`.\n\n        This function returns a handle with a method ``handle.remove()``\n        that removes the hook from the module.\n\n        .. note::\n            See :ref:`backward-hooks-execution` for more information on how when this hook\n            is executed, and how its execution is ordered relative to other hooks.\n\n        Example::\n\n            >>> v = torch.tensor([0., 0., 0.], requires_grad=True)\n            >>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient\n            >>> v.backward(torch.tensor([1., 2., 3.]))\n            >>> v.grad\n\n             2\n             4\n             6\n            [torch.FloatTensor of size (3,)]\n\n            >>> h.remove()  # removes the hook\n        '
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.register_hook, (self,), self, hook)
        if not self.requires_grad:
            raise RuntimeError("cannot register a hook on a tensor that doesn't require gradient")
        if self._backward_hooks is None:
            self._backward_hooks = OrderedDict()
            if self.grad_fn is not None:
                self.grad_fn._register_hook_dict(self)
        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def register_post_accumulate_grad_hook(self, hook):
        if False:
            i = 10
            return i + 15
        'Registers a backward hook that runs after grad accumulation.\n\n        The hook will be called after all gradients for a tensor have been accumulated,\n        meaning that the .grad field has been updated on that tensor. The post\n        accumulate grad hook is ONLY applicable for leaf tensors (tensors without a\n        .grad_fn field). Registering this hook on a non-leaf tensor will error!\n\n        The hook should have the following signature::\n\n            hook(param: Tensor) -> None\n\n        Note that, unlike other autograd hooks, this hook operates on the tensor\n        that requires grad and not the grad itself. The hook can in-place modify\n        and access its Tensor argument, including its .grad field.\n\n        This function returns a handle with a method ``handle.remove()``\n        that removes the hook from the module.\n\n        .. note::\n            See :ref:`backward-hooks-execution` for more information on how when this hook\n            is executed, and how its execution is ordered relative to other hooks. Since\n            this hook runs during the backward pass, it will run in no_grad mode (unless\n            create_graph is True). You can use torch.enable_grad() to re-enable autograd\n            within the hook if you need it.\n\n        Example::\n\n            >>> v = torch.tensor([0., 0., 0.], requires_grad=True)\n            >>> lr = 0.01\n            >>> # simulate a simple SGD update\n            >>> h = v.register_post_accumulate_grad_hook(lambda p: p.add_(p.grad, alpha=-lr))\n            >>> v.backward(torch.tensor([1., 2., 3.]))\n            >>> v\n            tensor([-0.0100, -0.0200, -0.0300], requires_grad=True)\n\n            >>> h.remove()  # removes the hook\n        '
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.register_post_accumulate_grad_hook, (self,), self, hook)
        if not self.requires_grad:
            raise RuntimeError("cannot register a hook on a tensor that doesn't require gradient")
        if self.grad_fn is not None:
            raise RuntimeError('post accumulate grad hooks cannot be registered on non-leaf tensors')
        if self._post_accumulate_grad_hooks is None:
            self._post_accumulate_grad_hooks: Dict[Any, Any] = OrderedDict()
        handle = hooks.RemovableHandle(self._post_accumulate_grad_hooks)
        self._post_accumulate_grad_hooks[handle.id] = hook
        return handle

    def reinforce(self, reward):
        if False:
            while True:
                i = 10

        def trim(str):
            if False:
                for i in range(10):
                    print('nop')
            return '\n'.join([line.strip() for line in str.split('\n')])
        raise RuntimeError(trim('reinforce() was removed.\n            Use torch.distributions instead.\n            See https://pytorch.org/docs/master/distributions.html\n\n            Instead of:\n\n            probs = policy_network(state)\n            action = probs.multinomial()\n            next_state, reward = env.step(action)\n            action.reinforce(reward)\n            action.backward()\n\n            Use:\n\n            probs = policy_network(state)\n            # NOTE: categorical is equivalent to what used to be called multinomial\n            m = torch.distributions.Categorical(probs)\n            action = m.sample()\n            next_state, reward = env.step(action)\n            loss = -m.log_prob(action) * reward\n            loss.backward()\n        '))
    detach = _C._add_docstr(_C.TensorBase.detach, '\n    Returns a new Tensor, detached from the current graph.\n\n    The result will never require gradient.\n\n    This method also affects forward mode AD gradients and the result will never\n    have forward mode AD gradients.\n\n    .. note::\n\n      Returned Tensor shares the same storage with the original one.\n      In-place modifications on either of them will be seen, and may trigger\n      errors in correctness checks.\n      IMPORTANT NOTE: Previously, in-place size / stride / storage changes\n      (such as `resize_` / `resize_as_` / `set_` / `transpose_`) to the returned tensor\n      also update the original tensor. Now, these in-place changes will not update the\n      original tensor anymore, and will instead trigger an error.\n      For sparse tensors:\n      In-place indices / values changes (such as `zero_` / `copy_` / `add_`) to the\n      returned tensor will not update the original tensor anymore, and will instead\n      trigger an error.\n    ')
    detach_ = _C._add_docstr(_C.TensorBase.detach_, '\n    Detaches the Tensor from the graph that created it, making it a leaf.\n    Views cannot be detached in-place.\n\n    This method also affects forward mode AD gradients and the result will never\n    have forward mode AD gradients.\n    ')

    def is_shared(self):
        if False:
            while True:
                i = 10
        'Checks if tensor is in shared memory.\n\n        This is always ``True`` for CUDA tensors.\n        '
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.is_shared, (self,), self)
        return self._typed_storage()._is_shared()

    def share_memory_(self):
        if False:
            print('Hello World!')
        'Moves the underlying storage to shared memory.\n\n        This is a no-op if the underlying storage is already in shared memory\n        and for CUDA tensors. Tensors in shared memory cannot be resized.\n\n        See :meth:`torch.UntypedStorage.share_memory_` for more details.\n        '
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.share_memory_, (self,), self)
        self._typed_storage()._share_memory_()
        return self

    def __reversed__(self):
        if False:
            return 10
        'Reverses the tensor along dimension 0.'
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__reversed__, (self,), self)
        if self.dim() == 0:
            return self
        else:
            return self.flip(0)

    def norm(self, p: Optional[Union[float, str]]='fro', dim=None, keepdim=False, dtype=None):
        if False:
            return 10
        'See :func:`torch.norm`'
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.norm, (self,), self, p=p, dim=dim, keepdim=keepdim, dtype=dtype)
        return torch.norm(self, p, dim, keepdim, dtype=dtype)

    def solve(self, other):
        if False:
            for i in range(10):
                print('nop')
        from ._linalg_utils import solve
        return solve(self, other)

    def lstsq(self, other):
        if False:
            for i in range(10):
                print('nop')
        from ._linalg_utils import lstsq
        return lstsq(self, other)

    def eig(self, eigenvectors=False):
        if False:
            i = 10
            return i + 15
        from ._linalg_utils import eig
        return eig(self, eigenvectors=eigenvectors)

    def symeig(self, eigenvectors=False):
        if False:
            print('Hello World!')
        from ._linalg_utils import _symeig
        return _symeig(self, eigenvectors=eigenvectors)

    def lu(self, pivot=True, get_infos=False):
        if False:
            i = 10
            return i + 15
        'See :func:`torch.lu`'
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.lu, (self,), self, pivot=pivot, get_infos=get_infos)
        (LU, pivots, infos) = torch._lu_with_info(self, pivot=pivot, check_errors=not get_infos)
        if get_infos:
            return (LU, pivots, infos)
        else:
            return (LU, pivots)

    def stft(self, n_fft: int, hop_length: Optional[int]=None, win_length: Optional[int]=None, window: 'Optional[Tensor]'=None, center: bool=True, pad_mode: str='reflect', normalized: bool=False, onesided: Optional[bool]=None, return_complex: Optional[bool]=None):
        if False:
            print('Hello World!')
        'See :func:`torch.stft`\n\n        .. warning::\n          This function changed signature at version 0.4.1. Calling with\n          the previous signature may cause error or return incorrect result.\n        '
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.stft, (self,), self, n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode, normalized=normalized, onesided=onesided, return_complex=return_complex)
        return torch.stft(self, n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided, return_complex=return_complex)

    def istft(self, n_fft: int, hop_length: Optional[int]=None, win_length: Optional[int]=None, window: 'Optional[Tensor]'=None, center: bool=True, normalized: bool=False, onesided: Optional[bool]=None, length: Optional[int]=None, return_complex: bool=False):
        if False:
            while True:
                i = 10
        'See :func:`torch.istft`'
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.istft, (self,), self, n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, normalized=normalized, onesided=onesided, length=length, return_complex=return_complex)
        return torch.istft(self, n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex=return_complex)

    def resize(self, *sizes):
        if False:
            for i in range(10):
                print('nop')
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.resize, (self,), self, *sizes)
        warnings.warn('non-inplace resize is deprecated')
        from torch.autograd._functions import Resize
        return Resize.apply(self, sizes)

    def resize_as(self, tensor):
        if False:
            i = 10
            return i + 15
        if has_torch_function_variadic(self, tensor):
            return handle_torch_function(Tensor.resize_as, (self, tensor), self, tensor)
        warnings.warn('non-inplace resize_as is deprecated')
        from torch.autograd._functions import Resize
        return Resize.apply(self, tensor.size())

    def split(self, split_size, dim=0):
        if False:
            i = 10
            return i + 15
        'See :func:`torch.split`'
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.split, (self,), self, split_size, dim=dim)
        if isinstance(split_size, Tensor):
            try:
                split_size = int(split_size)
            except ValueError:
                pass
        if isinstance(split_size, (int, torch.SymInt)):
            return torch._VF.split(self, split_size, dim)
        else:
            return torch._VF.split_with_sizes(self, split_size, dim)

    def unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the unique elements of the input tensor.\n\n        See :func:`torch.unique`\n        '
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.unique, (self,), self, sorted=sorted, return_inverse=return_inverse, return_counts=return_counts, dim=dim)
        return torch.unique(self, sorted=sorted, return_inverse=return_inverse, return_counts=return_counts, dim=dim)

    def unique_consecutive(self, return_inverse=False, return_counts=False, dim=None):
        if False:
            for i in range(10):
                print('nop')
        'Eliminates all but the first element from every consecutive group of equivalent elements.\n\n        See :func:`torch.unique_consecutive`\n        '
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.unique_consecutive, (self,), self, return_inverse=return_inverse, return_counts=return_counts, dim=dim)
        return torch.unique_consecutive(self, return_inverse=return_inverse, return_counts=return_counts, dim=dim)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rsub__(self, other):
        if False:
            i = 10
            return i + 15
        return _C._VariableFunctions.rsub(self, other)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rdiv__(self, other):
        if False:
            return 10
        return self.reciprocal() * other
    __rtruediv__ = __rdiv__
    __itruediv__ = _C.TensorBase.__idiv__
    __pow__ = _handle_torch_function_and_wrap_type_error_to_not_implemented(_C.TensorBase.pow)
    __ipow__ = _handle_torch_function_and_wrap_type_error_to_not_implemented(_C.TensorBase.pow_)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rmod__(self, other):
        if False:
            return 10
        return torch.remainder(other, self)

    def __format__(self, format_spec):
        if False:
            print('Hello World!')
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__format__, (self,), self, format_spec)
        if self.dim() == 0 and (not self.is_meta) and (type(self) is Tensor):
            return self.item().__format__(format_spec)
        return object.__format__(self, format_spec)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rpow__(self, other):
        if False:
            return 10
        return torch.pow(other, self)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __floordiv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return torch.floor_divide(self, other)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rfloordiv__(self, other):
        if False:
            i = 10
            return i + 15
        return torch.floor_divide(other, self)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rlshift__(self, other):
        if False:
            print('Hello World!')
        return torch.bitwise_left_shift(other, self)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rrshift__(self, other):
        if False:
            print('Hello World!')
        return torch.bitwise_right_shift(other, self)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rmatmul__(self, other):
        if False:
            i = 10
            return i + 15
        return torch.matmul(other, self)
    __pos__ = _C.TensorBase.positive
    __neg__ = _C.TensorBase.neg
    __abs__ = _C.TensorBase.abs

    def __len__(self):
        if False:
            return 10
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__len__, (self,), self)
        if self.dim() == 0:
            raise TypeError('len() of a 0-d tensor')
        if torch._C._get_tracing_state():
            warnings.warn('Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.', category=torch.jit.TracerWarning, stacklevel=2)
        return self.shape[0]

    def __iter__(self):
        if False:
            return 10
        if self.dim() == 0:
            raise TypeError('iteration over a 0-d tensor')
        if torch._C._get_tracing_state():
            warnings.warn("Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).", category=torch.jit.TracerWarning, stacklevel=2)
        return iter(self.unbind(0))

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return id(self)

    def __dir__(self):
        if False:
            for i in range(10):
                print('nop')
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__dir__, (self,), self)
        tensor_methods = dir(self.__class__)
        tensor_methods.remove('volatile')
        attrs = list(self.__dict__.keys())
        keys = tensor_methods + attrs
        if not self.is_cuda or self.is_sparse:
            keys.remove('__cuda_array_interface__')
        return sorted(keys)
    __array_priority__ = 1000

    def __array__(self, dtype=None):
        if False:
            while True:
                i = 10
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__array__, (self,), self, dtype=dtype)
        if dtype is None:
            return self.numpy()
        else:
            return self.numpy().astype(dtype, copy=False)

    def __array_wrap__(self, array):
        if False:
            while True:
                i = 10
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__array_wrap__, (self,), self, array=array)
        if array.dtype == bool:
            array = array.astype('uint8')
        return torch.from_numpy(array)

    def __contains__(self, element):
        if False:
            while True:
                i = 10
        'Check if `element` is present in tensor\n\n        Args:\n            element (Tensor or scalar): element to be checked\n                for presence in current tensor"\n        '
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__contains__, (self,), self, element)
        if isinstance(element, (torch.Tensor, Number, torch.SymInt, torch.SymFloat, torch.SymBool)):
            return (element == self).any().item()
        raise RuntimeError(f'Tensor.__contains__ only supports Tensor or scalar, but you passed in a {type(element)}.')

    @property
    def __cuda_array_interface__(self):
        if False:
            while True:
                i = 10
        'Array view description for cuda tensors.\n\n        See:\n        https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html\n        '
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__cuda_array_interface__.__get__, (self,), self)
        if not self.is_cuda:
            raise AttributeError("Can't get __cuda_array_interface__ on non-CUDA tensor type: %s If CUDA data is required use tensor.cuda() to copy tensor to device memory." % self.type())
        if self.is_sparse:
            raise AttributeError("Can't get __cuda_array_interface__ on sparse type: %s Use Tensor.to_dense() to convert to a dense tensor first." % self.type())
        if self.requires_grad:
            raise RuntimeError("Can't get __cuda_array_interface__ on Variable that requires grad. If gradients aren't required, use var.detach() to get Variable that doesn't require grad.")
        typestr = {torch.complex64: '<c8', torch.complex128: '<c16', torch.float16: '<f2', torch.float32: '<f4', torch.float64: '<f8', torch.uint8: '|u1', torch.int8: '|i1', torch.int16: '<i2', torch.int32: '<i4', torch.int64: '<i8'}[self.dtype]
        itemsize = self.element_size()
        shape = tuple(self.shape)
        if self.is_contiguous():
            strides = None
        else:
            strides = tuple((s * itemsize for s in self.stride()))
        data_ptr = self.data_ptr() if self.numel() > 0 else 0
        data = (data_ptr, False)
        return dict(typestr=typestr, shape=shape, strides=strides, data=data, version=2)

    def storage_type(self):
        if False:
            for i in range(10):
                print('nop')
        'storage_type() -> type\n\n        Returns the type of the underlying storage.\n\n        '
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.storage_type, (self,), self)
        torch.storage._warn_typed_storage_removal()
        return self._typed_storage()._get_legacy_storage_class()

    def refine_names(self, *names):
        if False:
            for i in range(10):
                print('nop')
        'Refines the dimension names of :attr:`self` according to :attr:`names`.\n\n        Refining is a special case of renaming that "lifts" unnamed dimensions.\n        A ``None`` dim can be refined to have any name; a named dim can only be\n        refined to have the same name.\n\n        Because named tensors can coexist with unnamed tensors, refining names\n        gives a nice way to write named-tensor-aware code that works with both\n        named and unnamed tensors.\n\n        :attr:`names` may contain up to one Ellipsis (``...``).\n        The Ellipsis is expanded greedily; it is expanded in-place to fill\n        :attr:`names` to the same length as ``self.dim()`` using names from the\n        corresponding indices of ``self.names``.\n\n        Python 2 does not support Ellipsis but one may use a string literal\n        instead (``\'...\'``).\n\n        Args:\n            names (iterable of str): The desired names of the output tensor. May\n                contain up to one Ellipsis.\n\n        Examples::\n\n            >>> imgs = torch.randn(32, 3, 128, 128)\n            >>> named_imgs = imgs.refine_names(\'N\', \'C\', \'H\', \'W\')\n            >>> named_imgs.names\n            (\'N\', \'C\', \'H\', \'W\')\n\n            >>> tensor = torch.randn(2, 3, 5, 7, 11)\n            >>> tensor = tensor.refine_names(\'A\', ..., \'B\', \'C\')\n            >>> tensor.names\n            (\'A\', None, None, \'B\', \'C\')\n\n        .. warning::\n            The named tensor API is experimental and subject to change.\n\n        '
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.refine_names, (self,), self, *names)
        names = resolve_ellipsis(names, self.names, 'refine_names')
        return super().refine_names(names)

    def align_to(self, *names):
        if False:
            for i in range(10):
                print('nop')
        "Permutes the dimensions of the :attr:`self` tensor to match the order\n        specified in :attr:`names`, adding size-one dims for any new names.\n\n        All of the dims of :attr:`self` must be named in order to use this method.\n        The resulting tensor is a view on the original tensor.\n\n        All dimension names of :attr:`self` must be present in :attr:`names`.\n        :attr:`names` may contain additional names that are not in ``self.names``;\n        the output tensor has a size-one dimension for each of those new names.\n\n        :attr:`names` may contain up to one Ellipsis (``...``).\n        The Ellipsis is expanded to be equal to all dimension names of :attr:`self`\n        that are not mentioned in :attr:`names`, in the order that they appear\n        in :attr:`self`.\n\n        Python 2 does not support Ellipsis but one may use a string literal\n        instead (``'...'``).\n\n        Args:\n            names (iterable of str): The desired dimension ordering of the\n                output tensor. May contain up to one Ellipsis that is expanded\n                to all unmentioned dim names of :attr:`self`.\n\n        Examples::\n\n            >>> tensor = torch.randn(2, 2, 2, 2, 2, 2)\n            >>> named_tensor = tensor.refine_names('A', 'B', 'C', 'D', 'E', 'F')\n\n            # Move the F and E dims to the front while keeping the rest in order\n            >>> named_tensor.align_to('F', 'E', ...)\n\n        .. warning::\n            The named tensor API is experimental and subject to change.\n\n        "
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.align_to, (self,), self, *names)
        ellipsis_idx = single_ellipsis_index(names, 'align_to')
        if ellipsis_idx is None:
            return super().align_to(names)
        return super().align_to([name for name in names if not is_ellipsis(name)], ellipsis_idx)

    def unflatten(self, dim, sizes):
        if False:
            while True:
                i = 10
        '\n        unflatten(dim, sizes) -> Tensor\n\n        See :func:`torch.unflatten`.\n\n        '
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.unflatten, (self,), self, dim, sizes)
        if not sizes:
            raise RuntimeError('unflatten: sizes must be non-empty')
        names = None
        if isinstance(sizes, OrderedDict) or (isinstance(sizes, (tuple, list)) and isinstance(sizes[0], (tuple, list))):
            (names, sizes) = unzip_namedshape(sizes)
            return super().unflatten(dim, sizes, names)
        else:
            return super().unflatten(dim, sizes)

    def rename_(self, *names, **rename_map):
        if False:
            while True:
                i = 10
        'In-place version of :meth:`~Tensor.rename`.'
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.rename_, (self,), self, *names, **rename_map)
        return update_names(self, names, rename_map, inplace=True)

    def rename(self, *names, **rename_map):
        if False:
            for i in range(10):
                print('nop')
        "Renames dimension names of :attr:`self`.\n\n        There are two main usages:\n\n        ``self.rename(**rename_map)`` returns a view on tensor that has dims\n        renamed as specified in the mapping :attr:`rename_map`.\n\n        ``self.rename(*names)`` returns a view on tensor, renaming all\n        dimensions positionally using :attr:`names`.\n        Use ``self.rename(None)`` to drop names on a tensor.\n\n        One cannot specify both positional args :attr:`names` and keyword args\n        :attr:`rename_map`.\n\n        Examples::\n\n            >>> imgs = torch.rand(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))\n            >>> renamed_imgs = imgs.rename(N='batch', C='channels')\n            >>> renamed_imgs.names\n            ('batch', 'channels', 'H', 'W')\n\n            >>> renamed_imgs = imgs.rename(None)\n            >>> renamed_imgs.names\n            (None, None, None, None)\n\n            >>> renamed_imgs = imgs.rename('batch', 'channel', 'height', 'width')\n            >>> renamed_imgs.names\n            ('batch', 'channel', 'height', 'width')\n\n        .. warning::\n            The named tensor API is experimental and subject to change.\n\n        "
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.rename, (self,), self, *names, **rename_map)
        return update_names(self, names, rename_map, inplace=False)

    def to_sparse_coo(self):
        if False:
            print('Hello World!')
        'Convert a tensor to :ref:`coordinate format <sparse-coo-docs>`.\n\n        Examples::\n\n             >>> dense = torch.randn(5, 5)\n             >>> sparse = dense.to_sparse_coo()\n             >>> sparse._nnz()\n             25\n\n        '
        return self.to_sparse()

    def dim_order(self):
        if False:
            while True:
                i = 10
        '\n\n        dim_order() -> tuple\n\n        Returns a tuple of int describing the dim order or physical layout of :attr:`self`.\n\n        Args:\n            None\n\n        Dim order represents how dimensions are laid out in memory,\n        starting from the outermost to the innermost dimension.\n\n        Example::\n            >>> torch.empty((2, 3, 5, 7)).dim_order()\n            (0, 1, 2, 3)\n            >>> torch.empty((2, 3, 5, 7), memory_format=torch.channels_last).dim_order()\n            (0, 2, 3, 1)\n\n        .. warning::\n            The dim_order tensor API is experimental and subject to change.\n\n        '
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.dim_order, (self,), self)
        import torch._prims_common as utils
        return tuple(utils.compute_elementwise_output_logical_to_physical_perm(self))

    def _update_names(self, names, inplace):
        if False:
            return 10
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor._update_names, (self,), self, names, inplace)
        if inplace:
            return super().rename_(names)
        else:
            return super().rename(names)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if False:
            i = 10
            return i + 15
        '\n        This __torch_function__ implementation wraps subclasses such that\n        methods called on subclasses return a subclass instance instead of\n        a ``torch.Tensor`` instance.\n\n        One corollary to this is that you need coverage for torch.Tensor\n        methods if implementing __torch_function__ for subclasses.\n\n        We recommend always calling ``super().__torch_function__`` as the base\n        case when doing the above.\n\n        While not mandatory, we recommend making `__torch_function__` a classmethod.\n        '
        if kwargs is None:
            kwargs = {}
        if not all((issubclass(cls, t) for t in types)):
            return NotImplemented
        with _C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)
            if func in get_default_nowrap_functions():
                return ret
            else:
                return _convert(ret, cls)
    __torch_dispatch__ = _C._disabled_torch_dispatch_impl

    def __dlpack__(self, stream=None):
        if False:
            return 10
        '\n        Creates a DLpack `capsule https://data-apis.org/array-api/latest/design_topics/data_interchange.html#data-interchange`_\n        of the current tensor to be exported to other libraries.\n\n        This function will be called from the `from_dlpack` method\n        of the library that will consume the capsule. `from_dlpack` passes the current\n        stream to this method as part of the specification.\n\n        Args:\n            stream (integer or None): An optional Python integer representing a\n            pointer to a CUDA stream. The current stream is synchronized with\n            this stream before the capsule is created, and since the capsule\n            shares its storage with the tensor this make it safe to access from\n            both streams.  If None or -1 is passed then no synchronization is performed.\n            If 1 (on CUDA) or 0 (on ROCM) then the default stream is used for\n            synchronization.\n        '
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__dlpack__, (self,), self, stream)
        if self.requires_grad:
            raise RuntimeError("Can't export tensors that require gradient, use tensor.detach()")
        if self.is_conj():
            raise RuntimeError("Can't export tensors with the conjugate bit set")
        if self.layout != torch.strided:
            raise RuntimeError("Can't export tensors with layout other than torch.strided")
        if stream is not None and type(stream) is not int:
            raise TypeError('stream must be ``int`` or ``none``')
        elif stream is not None and stream != -1:
            if self.device.type == 'cuda':
                if stream == 1 and torch.version.hip is None:
                    stream = torch.cuda.default_stream()
                elif stream == 0 and torch.version.hip is not None:
                    stream = torch.cuda.default_stream()
                else:
                    stream = torch.cuda.ExternalStream(stream)
                sync_stream = torch.cuda.current_stream()
                if stream != sync_stream:
                    event = torch.cuda.Event()
                    event.record(sync_stream)
                    stream.wait_event(event)
        return torch.to_dlpack(self)

    def __dlpack_device__(self) -> Tuple[enum.IntEnum, int]:
        if False:
            while True:
                i = 10
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__dlpack_device__, (self,), self)
        device = self.device
        idx = device.index if device.index is not None else 0
        torch_device_type = device.type
        if torch_device_type == 'cuda' and torch.version.hip is not None:
            device_type = DLDeviceType.kDLROCM
        elif torch_device_type == 'cpu' and self.is_pinned():
            device_type = DLDeviceType.kDLCPUPinned
        elif torch_device_type == 'cuda':
            device_type = DLDeviceType.kDLGPU
        elif torch_device_type == 'cpu':
            device_type = DLDeviceType.kDLCPU
        elif self.device.type == 'xpu':
            device_type = DLDeviceType.kDLOneAPI
        else:
            raise ValueError(f'Unknown device type {torch_device_type} for Dlpack')
        return (device_type, idx)
    __module__ = 'torch'

def _convert(ret, cls):
    if False:
        print('Hello World!')
    if cls is Tensor:
        return ret
    if isinstance(ret, Tensor) and (not isinstance(ret, cls)):
        ret = ret.as_subclass(cls)
    if isinstance(ret, (tuple, list)):
        ret = type(ret)((_convert(r, cls) for r in ret))
    return ret