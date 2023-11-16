import contextlib
import warnings
import weakref
from typing import ContextManager, List, Optional, Tuple, TYPE_CHECKING
import torch
from torch._C._functorch import _unwrap_functional_tensor, _wrap_functional_tensor, current_level, peek_interpreter_stack, TransformType
from torch._guards import Source
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import is_traceable_wrapper_subclass, transform_subclass
from torch.utils.weak import WeakIdRef
if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import DimConstraint, DimDynamic
DimList = List

def safe_is_leaf(t):
    if False:
        i = 10
        return i + 15
    try:
        return t.is_leaf
    except RuntimeError:
        return False

def safe_grad(t):
    if False:
        print('Hello World!')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'The .grad attribute of a Tensor')
        return t.grad

def assert_eq(a, b):
    if False:
        i = 10
        return i + 15
    assert a == b, f'{a} != {b}'

def assert_metadata_eq(assert_eq, m1, m2, *, skip_symbolic=False):
    if False:
        return 10

    def go(m1, m2):
        if False:
            print('Hello World!')
        assert_eq(m1.dtype, m2.dtype)
        if not skip_symbolic:
            assert_eq(m1.shape, m2.shape)
        assert_eq(m1.requires_grad, m2.requires_grad)
        assert_eq(m1.is_leaf, m2.is_leaf)
        assert_eq(m1.grad_fn is None, m2.grad_fn is None)
        assert_eq(m1.is_sparse, m2.is_sparse)
        assert_eq(m1.is_inference(), m2.is_inference())
        assert_eq(m1.is_conj(), m2.is_conj())
        assert_eq(m1.is_neg(), m2.is_neg())
        assert_eq(safe_grad(m1) is not None, safe_grad(m2) is not None)
        if safe_grad(m1) is not None:
            go(safe_grad(m1), safe_grad(m2))
        if m1.is_sparse:
            assert_eq(m1.dense_dim(), m2.dense_dim())
            assert_eq(m1.sparse_dim(), m2.sparse_dim())
            assert_eq(m1.is_coalesced(), m2.is_coalesced())
        else:
            if not skip_symbolic:
                assert_eq(m1.stride(), m2.stride())
                assert_eq(m1.storage_offset(), m2.storage_offset())
            assert_eq(m1._is_view(), m2._is_view())
            if m1._is_view():
                go(m1._base, m2._base)
    return go(m1, m2)

class MetaConverter:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.storage_memo = {}
        self.tensor_memo: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self.maybe_storages_to_delete = []
        self.check_expired_frequency = 128
        self.check_expired_count = 0
        self.hit = 0
        self.miss = 0
        self.del_hook = None
        self.arg_cnt = 0

    def successful(self):
        if False:
            print('Hello World!')
        return self.hit > 0 and self.miss == 0

    def check_for_expired_weak_storages(self):
        if False:
            for i in range(10):
                print('nop')
        new_li = []
        stor_to_delete = []
        for obj in self.maybe_storages_to_delete:
            if not obj.expired():
                new_li.append(obj)
            else:
                stor_to_delete.append(obj)
        for obj in stor_to_delete:
            self.storage_memo.pop(obj, None)
        self.maybe_storages_to_delete = new_li
        self.check_expired_frequency = max(self.check_expired_frequency, len(self.maybe_storages_to_delete))

    def get_tensor_memo(self, t):
        if False:
            print('Hello World!')
        return self.tensor_memo.get(WeakIdRef(t), None)

    def set_tensor_memo(self, t, v):
        if False:
            i = 10
            return i + 15
        self_weak_ref = weakref.ref(self)
        if t.is_sparse or t.is_mkldnn:
            weak_st = None
        else:
            weak_st = StorageWeakRef(t._typed_storage())
        tensor_ref_key = WeakIdRef(t)

        def del_ten():
            if False:
                return 10
            self_ref = self_weak_ref()
            if self_ref is None:
                return
            self_ref.tensor_memo.pop(tensor_ref_key, None)
            if weak_st and weak_st.expired():
                self_ref.storage_memo.pop(weak_st, None)
            elif weak_st is not None:
                self_ref.maybe_storages_to_delete.append(weak_st)
        weakref.finalize(t, del_ten)
        self.tensor_memo[tensor_ref_key] = v

    def meta_storage(self, s, callback):
        if False:
            for i in range(10):
                print('nop')
        swr = StorageWeakRef(s)
        if swr not in self.storage_memo:
            self.storage_memo[swr] = callback(lambda : torch.empty(s.size(), dtype=torch.uint8, device='meta')).untyped_storage()
        return self.storage_memo[swr]

    def meta_tensor(self, t, shape_env=None, callback=lambda t: t(), source: Optional[Source]=None, dynamic_dims: 'Optional[DimList[DimDynamic]]'=None, constraint_dims: 'Optional[DimList[DimConstraint]]'=None):
        if False:
            return 10
        from torch._subclasses.fake_tensor import FakeTensor
        if source is None:
            from torch._dynamo.source import ConstantSource
            source = ConstantSource(f'__meta_utils_unknown_tensor{len(self.tensor_memo)}')
        assert not torch._C._dispatch_tls_local_exclude_set().has(torch._C.DispatchKey.Python)
        arg_cnt = self.arg_cnt
        self.arg_cnt += 1
        maybe_suppress = contextlib.nullcontext
        if shape_env is not None:
            maybe_suppress = shape_env.suppress_guards

        def sym_sizes_strides_storage_offset(t, src) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
            if False:
                while True:
                    i = 10
            if shape_env is not None:
                if isinstance(t, FakeTensor) and t.fake_mode.shape_env is shape_env:
                    return (t.size(), t.stride(), t.storage_offset())
                else:
                    return shape_env.create_symbolic_sizes_strides_storage_offset(t, src, dynamic_dims=dynamic_dims, constraint_dims=constraint_dims)
            else:
                assert dynamic_dims is None
                assert constraint_dims is None
            return (t.size(), t.stride(), t.storage_offset())
        self.check_expired_count += 1
        if self.check_expired_count >= self.check_expired_frequency:
            self.check_for_expired_weak_storages()
            self.check_expired_count = 0
        if self.get_tensor_memo(t) is None:
            with torch.inference_mode(t.is_inference()):
                if t.is_sparse:
                    is_leaf = safe_is_leaf(t)
                    r = callback(lambda : torch.ops.aten._sparse_coo_tensor_with_dims(t.sparse_dim(), t.dense_dim(), t.shape, dtype=t.dtype, layout=torch.sparse_coo, device='meta'))
                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    r._coalesced_(t.is_coalesced())
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and (not is_leaf):
                        with torch.enable_grad():
                            r = r.clone()
                            r._coalesced_(t.is_coalesced())
                elif t.is_mkldnn:
                    is_leaf = safe_is_leaf(t)
                    (sizes, strides, _storage_offset) = sym_sizes_strides_storage_offset(t, source)
                    r = callback(lambda : torch.empty_strided(sizes, strides, dtype=t.dtype, device='meta'))
                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and (not is_leaf):
                        with torch.enable_grad():
                            r = r.clone()
                elif t._is_view():
                    assert t._is_view()
                    from torch._dynamo.source import AttrSource
                    from torch.fx.experimental.symbolic_shapes import DimDynamic
                    if shape_env and (not t.is_nested) and (not t._base.is_nested):
                        base_dynamic_dims = [DimDynamic.STATIC] * t._base.dim()
                    else:
                        base_dynamic_dims = None
                    base = self.meta_tensor(t._base, shape_env, callback, source=AttrSource(source, '_base'), dynamic_dims=base_dynamic_dims)

                    def is_c_of_r(complex_dtype, real_dtype):
                        if False:
                            i = 10
                            return i + 15
                        return utils.is_complex_dtype(complex_dtype) and utils.corresponding_real_dtype(complex_dtype) == real_dtype
                    old_exclude = torch._C._dispatch_tls_is_dispatch_key_excluded(torch._C.DispatchKey.ADInplaceOrView)
                    torch._C._dispatch_tls_set_dispatch_key_excluded(torch._C.DispatchKey.ADInplaceOrView, False)
                    try:
                        if base.dtype == t.dtype:
                            pass
                        elif is_c_of_r(base.dtype, t.dtype):
                            base = torch.view_as_real(base)
                        elif is_c_of_r(t.dtype, base.dtype):
                            base = torch.view_as_complex(base)
                        else:
                            base = base.view(t.dtype)

                        def _view_from_base(base, t):
                            if False:
                                i = 10
                                return i + 15
                            if t.is_nested:
                                return t._view_func_unsafe(base)
                            else:
                                (sizes, strides, storage_offset) = sym_sizes_strides_storage_offset(t, source)
                                return base.as_strided(sizes, strides, storage_offset)
                        if safe_is_leaf(t):
                            with torch.no_grad(), maybe_suppress():
                                r = _view_from_base(base, t)
                            r.requires_grad = t.requires_grad
                        elif t._base.requires_grad == t.requires_grad:
                            with torch.enable_grad(), maybe_suppress():
                                r = _view_from_base(base, t)
                        else:
                            assert t.requires_grad
                            with torch.no_grad():
                                mid = base.view(base.shape)
                            mid.requires_grad = t.requires_grad
                            with torch.enable_grad(), maybe_suppress():
                                r = _view_from_base(mid, t)
                        torch._C._autograd._set_creation_meta(r, torch._C._autograd._get_creation_meta(t))
                    finally:
                        torch._C._dispatch_tls_set_dispatch_key_excluded(torch._C.DispatchKey.ADInplaceOrView, old_exclude)
                else:
                    is_leaf = safe_is_leaf(t)
                    if not t.is_nested:
                        (sizes, strides, storage_offset) = sym_sizes_strides_storage_offset(t, source)

                    def empty_create(inner_t, inner_src):
                        if False:
                            for i in range(10):
                                print('nop')
                        (inner_sizes, inner_strides, inner_storage_offset) = sym_sizes_strides_storage_offset(inner_t, inner_src)
                        return torch.empty_strided(inner_sizes, inner_strides, dtype=inner_t.dtype, device='meta')
                    if is_traceable_wrapper_subclass(t):
                        from torch._dynamo.source import AttrSource
                        if t.is_nested:
                            from torch._dynamo.source import TensorProperty, TensorPropertySource
                            (attrs, ctx) = t.__tensor_flatten__()
                            transformed_tensors_dict = {}
                            orig_shape_env = None
                            for attr in attrs:
                                inner_t = getattr(t, attr)
                                if orig_shape_env is None:
                                    orig_shape_env = inner_t.fake_mode.shape_env if isinstance(inner_t, FakeTensor) else None
                                transformed_tensors_dict[attr] = callback(lambda : empty_create(inner_t, AttrSource(source, attr)))
                            assert isinstance(ctx, dict)
                            assert 'ragged_size' in ctx
                            assert isinstance(t._size[1], torch.SymInt)
                            if orig_shape_env is shape_env:
                                ctx['ragged_size'] = t._size[1]
                            else:
                                assert t._size[1].node.singleton_int() is not None
                                ctx['ragged_size'] = shape_env.create_symintnode(shape_env.create_symbol(t._size[1], TensorPropertySource(source, TensorProperty.SIZE, 1)), hint=t._size[1])
                            r = type(t).__tensor_unflatten__(transformed_tensors_dict, ctx)
                        else:
                            r = transform_subclass(t, lambda attr, inner_t: callback(lambda : empty_create(inner_t, AttrSource(source, attr))))
                    else:
                        r = callback(lambda : torch.empty_strided(sizes, strides, dtype=t.dtype, device='meta'))
                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    if t.requires_grad:
                        r.requires_grad = t.requires_grad
                        if not is_leaf:
                            with torch.enable_grad():
                                r = r.clone(memory_format=torch.preserve_format)
                    if torch._C._functorch.is_functorch_wrapped_tensor(t):
                        return NotImplemented
                    s = t.untyped_storage()
                    swr = StorageWeakRef(s)
                    if swr not in self.storage_memo and (r.is_nested or (r.stride() == strides and r.storage_offset() == storage_offset)):
                        self.storage_memo[swr] = r.untyped_storage()
                    else:
                        r_s = self.meta_storage(s, callback=callback)
                        maybe_fake_mgr: ContextManager[None] = contextlib.nullcontext()
                        from torch._subclasses.fake_tensor import in_kernel_invocation_manager, maybe_get_fake_mode
                        mb_fake_mode = maybe_get_fake_mode(r)
                        if mb_fake_mode is not None:
                            maybe_fake_mgr = in_kernel_invocation_manager(mb_fake_mode)
                        with maybe_fake_mgr, torch.no_grad():
                            r.set_(r_s, storage_offset, sizes, strides)
                if safe_grad(t) is not None:
                    from torch._dynamo.source import AttrSource
                    r.grad = self.meta_tensor(safe_grad(t), shape_env, callback, source=AttrSource(source, 'grad'), dynamic_dims=dynamic_dims, constraint_dims=constraint_dims)
                torch._C._set_conj(r, t.is_conj())
                torch._C._set_neg(r, t.is_neg())
            assert_metadata_eq(assert_eq, t, r, skip_symbolic=True)
            self.set_tensor_memo(t, r)
        return self.get_tensor_memo(t)

    def __call__(self, t, shape_env=None, *, callback=lambda t: t(), source=None, dynamic_dims=None, constraint_dims=None):
        if False:
            while True:
                i = 10
        if isinstance(t, torch.Tensor) or is_traceable_wrapper_subclass(t):
            if t.device.type != 'xla' and any([t.is_sparse_csr, t.layout in [torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc], t.is_quantized, t._is_view() and t._base is not None and t._base.is_sparse, torch._is_functional_tensor(t), t.device.type in 'lazy']):
                if torch._is_functional_tensor(t) and t.device.type != 'lazy':
                    if t._is_view():
                        raise RuntimeError('Cannot safely fakify a view because this process drops the view information right now.')
                    st = peek_interpreter_stack()
                    assert st is None or st.key() == TransformType.Functionalize, 'Expect st to be either None or have Functionalize transform key.'
                    if st is None:
                        torch._sync(t)
                        unwrap_t = torch._from_functional_tensor(t)
                        with torch._dispatch.python.suspend_functionalization():
                            fake_t = self.meta_tensor(unwrap_t, shape_env=shape_env, callback=callback, source=source, dynamic_dims=dynamic_dims, constraint_dims=constraint_dims)
                        out = torch._to_functional_tensor(fake_t)
                        torch._mirror_autograd_meta_to(fake_t, out)
                        return out
                    else:
                        reapply_views = torch._C._functionalization_reapply_views_tls()
                        unwrap_t = _unwrap_functional_tensor(t, reapply_views)
                        pop_st_ctx = torch._functorch.pyfunctorch.temporarily_pop_interpreter_stack()
                        with pop_st_ctx:
                            fake_t = self.meta_tensor(unwrap_t, shape_env=shape_env, callback=callback, source=source, dynamic_dims=dynamic_dims, constraint_dims=constraint_dims)
                        return _wrap_functional_tensor(fake_t, current_level())
                self.miss += 1
                return NotImplemented
            else:
                self.hit += 1
                r = self.meta_tensor(t, shape_env=shape_env, callback=callback, source=source, dynamic_dims=dynamic_dims, constraint_dims=constraint_dims)
                if type(t) is torch.nn.Parameter:
                    r._is_param = True
                return r
        elif torch.overrides.is_tensor_like(t):
            self.miss += 1
            return NotImplemented
        else:
            return t
import torch._prims_common as utils