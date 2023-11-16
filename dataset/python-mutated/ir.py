import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
import traceback
from contextlib import nullcontext
from enum import Enum
from functools import partial
from inspect import signature
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
from unittest.mock import patch
import sympy
from sympy import Expr, Integer
import torch._export.serde.schema as export_schema
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._prims_common import compute_required_storage_length, is_boolean_dtype, is_float_dtype, make_channels_last_strides_for, make_contiguous_strides_for, StrideType
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .dependencies import extract_input_node_reduction_ranges, extract_read_writes, var_builder
from .utils import argsort, cache_on_self, convert_shape_to_inductor, convert_shape_to_symint, developer_warning, get_kernel_metadata, is_dynamic, pad_listlike, sympy_dot, sympy_product, sympy_subs, sympy_symbol
from .virtualized import ops, V
log = logging.getLogger(__name__)
indent = functools.partial(textwrap.indent, prefix='  ')
aten = torch.ops.aten
" [Note: Inductor IR]\n\nInductor's IR is produced by executing 'lowering' code (see lowering.py).  Each\nlowering is registered to a particular aten operator, and expects inputs that\ncorrespond to the aten schema.  However, in place of torch Tensor inputs, lowerings\nexpect Inductor TensorBox inputs.\n\nTensorBox IR represents torch tensors.  Tensors are sometimes single objects owning\nstorage, and sometimes views of another Tensor's storage.  Mutating tensor operations\n(such as add_()) affect the underlying storage and any associated views.  Other operations\n(such as .t_()) update metadata about the current view but don't modify the underlying storage.\n\nTo model this in Inductor, the IR distinguishes between TensorBox, View, StorageBox and Buffer.\n\nTensorBox is the top level IR construct that any lowering should produce and maps to a torch.Tensor\noutput from an operation.  But just as torch.Tensors take different forms, TensorBox IR can\nreference View IR or directly reference StorageBox IRs.\n\nSome Inductor lowerings produce new sets of 'Box'es, while others (such as .t() or other view ops)\nmay take an existing TensorBox and point it to a new underlying View IR.\n\nTensors that directly own storage are represented as a chain of:\nTensorBox -> StorageBox -> Buffer\nwhere Buffer is a simple (1D) allocation, and StorageBox introduces the concept of a Layout.\n\nIf you mutate the data of such a tensor, we swing the StorageBox pointer to point to a new buffer\n(leaving the old buffer unmodified and functionalizing the operation).\n\nTensors backed by views add one more indirection to the IR.\nTensorBox -> View -> StorageBox -> Buffer\nIn these cases, the underlying StorageBox/Buffer will be shared with the pre-view TensorBox.\n"

def validate_ir(node_or_nodes):
    if False:
        i = 10
        return i + 15

    def _check_tensorbox(nodes):
        if False:
            i = 10
            return i + 15
        if isinstance(nodes, (list, tuple)):
            for node in nodes:
                _check_tensorbox(node)
        elif isinstance(nodes, dict):
            for node in nodes.values():
                _check_tensorbox(node)
        else:
            assert isinstance(nodes, (torch._inductor.ir.ExpandView, DynamicScalar, TensorBox, sympy.Symbol, sympy.logic.boolalg.Boolean, Expr)), f'Found {type(nodes)}, which is not a supported top level IR node. See [Note: Inductor IR]'
    _check_tensorbox(node_or_nodes)

def ops_wrapper(name):
    if False:
        while True:
            i = 10
    assert isinstance(name, str)

    def fn(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return getattr(ops, name)(*args, **kwargs)
    return fn

def inverse_reorder(order):
    if False:
        for i in range(10):
            print('nop')
    inv_order = dict(zip(order, range(len(order))))

    def reindex(index):
        if False:
            while True:
                i = 10
        assert len(index) == len(inv_order)
        return [index[inv_order[i]] for i in range(len(index))]
    return reindex

def same_reorder(order):
    if False:
        while True:
            i = 10

    def reindex(index):
        if False:
            return 10
        assert len(index) == len(order)
        return [index[order[i]] for i in range(len(index))]
    return reindex

def fuse_reindexing(reindex1, reindex2):
    if False:
        for i in range(10):
            print('nop')

    def reindex(index):
        if False:
            for i in range(10):
                print('nop')
        return reindex1(reindex2(index))
    return reindex
NHWC_STRIDE_ORDER = [3, 0, 2, 1]

def stride_order2fill_order(order):
    if False:
        while True:
            i = 10
    '\n    Convert stride order to fill order\n    For channel last format,\n    stride order = [3, 0, 2, 1] and fill order = [1, 3, 2, 0]\n    '
    lookup = {pos: idx for (idx, pos) in enumerate(order)}
    fill_order = [lookup[i] for i in range(len(order))]
    return fill_order

def get_stride_order(seq: Sequence[int]) -> List[int]:
    if False:
        i = 10
        return i + 15
    '\n    Convert strides to stride order\n    '
    sorted_idx: List[int] = argsort(seq)
    out = [0 for _ in range(len(seq))]
    for (i, elem) in enumerate(sorted_idx):
        out[elem] = i
    return out

def ir_node_to_tensor(x, guard_shape=True):
    if False:
        return 10
    if x is None:
        return None
    if not guard_shape:
        shape_fn = V.graph.sizevars.size_hint
    else:
        shape_fn = identity
    size = [shape_fn(s) for s in x.get_size()]
    stride: StrideType
    if is_storage_and_layout(x):
        stride = [shape_fn(s) for s in x.get_layout().stride]
    else:
        stride = make_contiguous_strides_for(size)
    dtype = x.get_dtype()
    device = x.get_device()
    size = convert_shape_to_symint(size)
    stride = convert_shape_to_symint(stride)
    t = torch.empty_strided(size=size, stride=stride, dtype=dtype, device=device).zero_()
    return t

def may_convert_to_optional(value):
    if False:
        while True:
            i = 10
    if isinstance(value, list) and (not value) and V.graph.cpp_wrapper:
        return [None]
    return value

def get_device_type(x):
    if False:
        while True:
            i = 10
    if getattr(x, 'get_device', None):
        return get_device_type(x.get_device())
    if isinstance(x, torch.device):
        return x.type
    return None

def is_triton(x):
    if False:
        i = 10
        return i + 15
    return get_device_type(x) == 'cuda'

def is_cpu(x):
    if False:
        i = 10
        return i + 15
    return get_device_type(x) == 'cpu'

class IRNode:
    _current_origins: ClassVar[Set[Any]] = set()

    @staticmethod
    @contextlib.contextmanager
    def current_origins(origins: Set[torch.fx.Node]):
        if False:
            return 10
        old = IRNode._current_origins
        IRNode._current_origins = old | origins
        try:
            yield
        finally:
            IRNode._current_origins = old

    def __post_init__(self):
        if False:
            while True:
                i = 10
        self.origins = set(self._current_origins)
        self.traceback = traceback.format_stack() if config.debug_ir_traceback else None

    def get_traceback(self):
        if False:
            return 10
        return self.traceback

    def common_repr(self):
        if False:
            return 10
        origins = f"origins={getattr(self, 'origins', '')}"
        if len(origins) > 64:
            origins = f'{origins[:61]}...'
        return [origins]

    def str_helper(self, lines):
        if False:
            for i in range(10):
                print('nop')
        lines = lines + self.common_repr()
        lines = indent(',\n'.join(map(str, lines)))
        return f'{type(self).__name__}(\n{lines}\n)'

    def is_user_of(self, name):
        if False:
            i = 10
            return i + 15
        return name in self.get_read_names()

    @cache_on_self
    def get_read_names(self):
        if False:
            i = 10
            return i + 15
        return {dep.name for dep in self.get_reads()}

    def get_layout(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError(f'get_layout() is not implemented by {type(self)}!')

    def get_size(self):
        if False:
            print('Hello World!')
        raise NotImplementedError(f'get_size() is not implemented by {type(self)}!')

    def get_numel(self):
        if False:
            print('Hello World!')
        return sympy_product(self.get_size())

    def is_zero_elements(self):
        if False:
            return 10
        return V.graph.sizevars.is_expr_static_and_true(sympy.Eq(self.get_numel(), 0))

    def realize(self):
        if False:
            i = 10
            return i + 15
        "\n        If the IRNode refers to data which has not been materialized (e.g.,\n        it is a Pointwise/Reduction that could potentially have more\n        compute fused into it), realize the IRNode into physical memory,\n        ending the possibility of fusing into it, but allowing, e.g., multiple\n        users to access the data without having to recompute.\n\n        Check StorageBox.realize for a particularly notable implementation.\n\n        TODO(ezyang): I think, in principle, every IRNode should have an\n        implementation of this, and most of the time no-op is OK, but you\n        really do have to audit each IRNode for this, so for now, raise\n        an error if it's not implemented.  Note that some code in graph.py\n        will catch this thrown error and suppress it with a warning.\n        "
        raise NotImplementedError(f'realize NYI on {type(self)}')
    get_device: Callable[[], torch.device]
    get_dtype: Callable[[], torch.dtype]
    get_name: Callable[[], str]
    get_reads: Callable[[], Any]
    get_stride: Callable[[], Any]
    get_storage_numel: Callable[[], Any]
    has_exceeded_max_reads: Callable[[], bool]
    make_loader: Callable[[], Callable[[Any], Any]]
    make_indexer: Callable[[], Callable[[Any], Any]]
    mark_reuse: Callable[[List[Any]], None]
    realize_hint: Callable[[], None]

@dataclasses.dataclass
class Loops(IRNode):
    device: torch.device
    dtype: torch.dtype
    inner_fn: Callable[..., Any]
    ranges: List[Expr]

    def __str__(self, names=('ranges',)):
        if False:
            for i in range(10):
                print('nop')
        return self.str_helper([f"'{self.device.type}'", str(self.dtype), self.inner_fn_str()] + [f'{name}={getattr(self, name)}' for name in names] + [f'origin_node={self.origin_node!r}'])

    def __post_init__(self):
        if False:
            while True:
                i = 10
        super().__post_init__()
        self.origin_node = None
    __repr__ = __str__

    def get_dtype(self):
        if False:
            return 10
        return self.dtype

    def get_device(self):
        if False:
            print('Hello World!')
        return self.device

    def get_origin_node(self):
        if False:
            i = 10
            return i + 15
        return self.origin_node

    def get_size(self):
        if False:
            i = 10
            return i + 15
        return self.ranges

    def is_extern(self):
        if False:
            i = 10
            return i + 15
        return False

    @classmethod
    def create(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        origin_node = kwargs.pop('origin_node', None)
        tb = kwargs.pop('traceback', None)
        r = cls(*args, **kwargs)
        r.origin_node = origin_node
        r.traceback = tb or traceback.format_stack() if config.debug_ir_traceback else None
        return TensorBox.create(r)

    @staticmethod
    def _index(ranges, prefix='i'):
        if False:
            for i in range(10):
                print('nop')
        return [sympy.Integer(0) if s == 1 else sympy_symbol(f'{prefix}{n}') for (n, s) in enumerate(ranges)]

    @cache_on_self
    def inner_fn_str_len(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.inner_fn_str())

    def inner_fn_str(self):
        if False:
            i = 10
            return i + 15
        index = self._index(self.ranges)
        return V.KernelFormatterHandler.ir_to_string(self.inner_fn, index)

    def get_reads(self):
        if False:
            for i in range(10):
                print('nop')
        with patch.object(FlexibleLayout, 'allow_indexing', True):
            if self.get_reduction_type():
                return extract_read_writes(self.make_loader(), self.get_size(), self.get_reduction_size()).reads
            else:
                return extract_read_writes(self.make_loader(), self.get_size()).reads

    def get_reduction_size(self):
        if False:
            return 10
        raise NotImplementedError(f'get_reduction_size() is not implemented by {type(self)}!')

    def get_reduction_type(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError(f'get_reduction_type() is not implemented by {type(self)}!')

    def constant_to_device(self, device):
        if False:
            print('Hello World!')
        raise NotImplementedError(f'constant_to_device() is not implemented by {type(self)}!')

def nop_loader_fn(idx, *, dtype):
    if False:
        return 10
    if dtype.is_floating_point:
        return ops.constant(float('nan'), dtype)
    else:
        return ops.constant(0, dtype)

class Pointwise(Loops):

    def make_loader(self):
        if False:
            for i in range(10):
                print('nop')
        if self.is_zero_elements():
            return partial(nop_loader_fn, dtype=self.dtype)
        return self.inner_fn

    def get_reduction_size(self):
        if False:
            print('Hello World!')
        return []

    def get_reduction_type(self):
        if False:
            for i in range(10):
                print('nop')
        return None

    def store_output(self, output_name, indexer, vars):
        if False:
            print('Hello World!')
        loader = self.make_loader()
        return ops.store(output_name, indexer(vars), loader(vars))

    def constant_to_device(self, device):
        if False:
            for i in range(10):
                print('nop')
        'Move this to a given device. Requires that all reads are to constants.'
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, 'override_device', device)(loader)
        return Pointwise(device, self.dtype, loader, self.ranges)

@dataclasses.dataclass
class Scatter(Pointwise):
    output_indexer: Callable[[List[Expr]], Expr]
    scatter_mode: Optional[str] = None

    def constant_to_device(self, device):
        if False:
            print('Hello World!')
        'Move this to a given device. Requires that all reads are to constants.'
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, 'override_device', device)(loader)
        return Scatter(device, self.dtype, loader, self.ranges, self.output_indexer, self.scatter_mode)

    def store_output(self, output_name, indexer, vars):
        if False:
            print('Hello World!')
        loader = self.make_loader()
        return ops.store(output_name, indexer(self.output_indexer(vars)), loader(vars), mode=self.scatter_mode)

class ReductionHint(Enum):
    INNER = 0
    OUTER = 1
    OUTER_TINY = 2
    DEFAULT = 3

class TileHint(Enum):
    SQUARE = 0
    DEFAULT = 1
REDUCTION_COMBINE_FN = {'any': ops_wrapper('logical_or'), 'max': ops_wrapper('maximum'), 'min': ops_wrapper('minimum'), 'prod': ops_wrapper('mul'), 'sum': ops_wrapper('add'), 'xor_sum': ops_wrapper('bitwise_xor')}

def get_reduction_combine_fn(reduction_type, dtype):
    if False:
        while True:
            i = 10
    if reduction_type in REDUCTION_COMBINE_FN:
        combine_fn = REDUCTION_COMBINE_FN[reduction_type]
    elif reduction_type in {'argmax', 'argmin'}:

        def combine_fn(a, b):
            if False:
                print('Hello World!')
            (a_value, a_index) = a
            (b_value, b_index) = b
            if reduction_type == 'argmin':
                mask = ops.lt(a_value, b_value)
            else:
                mask = ops.gt(a_value, b_value)
            equal = ops.eq(a_value, b_value)
            if is_float_dtype(dtype):
                a_isnan = ops.ne(a_value, a_value)
                b_isnan = ops.ne(b_value, b_value)
                mask = ops.logical_or(mask, ops.gt(a_isnan, b_isnan))
                equal = ops.logical_or(equal, ops.logical_and(a_isnan, b_isnan))
            mask = ops.logical_or(mask, ops.logical_and(equal, ops.lt(a_index, b_index)))
            return (ops.where(mask, a_value, b_value), ops.where(mask, a_index, b_index))
    elif reduction_type == 'welford_combine':

        def combine_fn(a, b):
            if False:
                print('Hello World!')
            (a_mean, a_m2, a_weight) = a
            (b_mean, b_m2, b_weight) = b
            delta = b_mean - a_mean
            new_weight = a_weight + b_weight
            w2_over_w = b_weight / new_weight
            return (a_mean + delta * w2_over_w, a_m2 + b_m2 + delta * delta * a_weight * w2_over_w, new_weight)
    else:
        raise NotImplementedError(f'unknown reduction_type={reduction_type}')
    return combine_fn

@dataclasses.dataclass
class Reduction(Loops):
    reduction_ranges: List[Expr]
    reduction_type: str
    src_dtype: torch.dtype
    reduction_hint: ReductionHint

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return Loops.__str__(self, names=('ranges', 'reduction_ranges', 'reduction_type'))

    def __repr__(self):
        if False:
            return 10
        return self.__str__()

    def get_reduction_size(self):
        if False:
            while True:
                i = 10
        return self.reduction_ranges

    def get_reduction_type(self):
        if False:
            for i in range(10):
                print('nop')
        return self.reduction_type

    def store_reduction(self, output_name, indexer, vars, reduction_vars):
        if False:
            return 10
        value = ops.reduction(self.dtype, self.src_dtype, self.reduction_type, self.inner_fn(vars, reduction_vars))
        return ops.store_reduction(output_name, indexer(vars), value)

    def index_length(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.ranges) + len(self.reduction_ranges)

    def inner_fn_str(self):
        if False:
            i = 10
            return i + 15
        index = self._index(self.ranges)
        rindex = self._index(self.reduction_ranges, 'r')
        return V.KernelFormatterHandler.ir_to_string(self.inner_fn, index, rindex)

    def constant_to_device(self, device):
        if False:
            return 10
        'Move this to a given device. Requires that all reads are to constants.'
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, 'override_device', device)(loader)
        return Reduction(device, self.dtype, loader, self.ranges, self.reduction_ranges, self.reduction_type, self.src_dtype, ReductionHint.DEFAULT)

    @staticmethod
    def num_splits(device, dst_dtype, src_dtype, inner_fn, ranges, reduction_ranges, reduction_type, reduction_numel, input_node: Optional[IRNode]=None):
        if False:
            while True:
                i = 10

        def _is_static(x):
            if False:
                print('Hello World!')
            return isinstance(x, (int, sympy.Integer))
        reduction_numel_hint = V.graph.sizevars.symbolic_hint(reduction_numel)
        numel_hint = V.graph.sizevars.symbolic_hint(sympy_product(ranges))
        should_split = is_triton(device) and reduction_type not in {'argmax', 'argmin'} and config.split_reductions and _is_static(reduction_numel_hint) and _is_static(numel_hint)
        if not should_split:
            return (ReductionHint.DEFAULT, 1)
        device_interface = get_interface_for_device(get_device_type(device))
        num_sm = device_interface.Worker.get_device_properties(device).multi_processor_count
        min_elements_per_thread = 32
        max_elements_per_thread = 512
        threads_per_sm = 2048
        min_elements_per_device = min_elements_per_thread * num_sm * threads_per_sm
        max_elements_per_device = max_elements_per_thread * num_sm * threads_per_sm

        def inner_reduction_splits(reduction_numel_hint, numel_hint):
            if False:
                print('Hello World!')
            num_warps = 8
            num_threads = 32 * num_warps
            if numel_hint >= 2 * num_sm:
                return 1
            if reduction_numel_hint <= 8192:
                return 1
            if reduction_numel_hint * numel_hint <= min_elements_per_device:
                split_size = min_elements_per_thread
            elif reduction_numel_hint * numel_hint < max_elements_per_device:
                target_blocks = num_sm * threads_per_sm // (2 * num_threads)
                blocks_per_output = (target_blocks + numel_hint - 1) // numel_hint
                tmp_split_size = (reduction_numel_hint + num_threads * blocks_per_output - 1) // (num_threads * blocks_per_output)
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - tmp_split_size))
                if abs(closest - tmp_split_size) < 30:
                    split_size = max(closest, min_elements_per_thread)
                else:
                    split_size = tmp_split_size
            else:
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - max_elements_per_thread))
                if abs(closest - max_elements_per_thread) < 50:
                    split_size = closest
                else:
                    split_size = max_elements_per_thread
            return (reduction_numel_hint + split_size * num_threads - 1) // (split_size * num_threads)

        def outer_reduction_splits(reduction_numel_hint, numel_hint):
            if False:
                print('Hello World!')
            num_warps = 8
            num_threads = num_warps * 32
            rvals_per_thread = 4
            xvals_per_block = 128
            xblocks = (numel_hint + xvals_per_block - 1) // xvals_per_block
            if reduction_numel_hint * numel_hint < min_elements_per_device:
                split_size = min_elements_per_thread
            elif reduction_numel_hint * numel_hint < max_elements_per_device:
                target_blocks = num_sm * threads_per_sm // num_threads
                target_blocks = (target_blocks + xblocks - 1) // xblocks
                tmp_split_size = (reduction_numel_hint + rvals_per_thread * target_blocks - 1) // (rvals_per_thread * target_blocks)
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - tmp_split_size))
                if abs(tmp_split_size - closest) < 20:
                    split_size = max(closest, min_elements_per_thread)
                else:
                    split_size = tmp_split_size
            else:
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - max_elements_per_thread))
                if abs(closest - max_elements_per_thread) < 50:
                    split_size = closest
                else:
                    split_size = max_elements_per_thread
            return (reduction_numel_hint + rvals_per_thread * split_size - 1) // (rvals_per_thread * split_size)
        if numel_hint == 1:
            split = inner_reduction_splits(reduction_numel_hint, numel_hint)
            if split == 1:
                return (ReductionHint.INNER, split)
            if len(ranges) == 0 and input_node is not None and isinstance(input_node, TensorBox):
                (new_ranges, new_reduction_ranges) = extract_input_node_reduction_ranges(input_node)
                if new_ranges is not None and new_reduction_ranges is not None:
                    extracted_numel_hint = V.graph.sizevars.symbolic_hint(sympy_product(new_ranges + new_reduction_ranges))
                    if reduction_numel_hint == extracted_numel_hint:
                        log.debug("Use previous IRNode's range and reduction_ranges instead of split. current ranges: %s, current reduction ranges: %s, current split: %d, new ranges: %s, new reduction ranges: %s", ranges, reduction_ranges, split, new_ranges, new_reduction_ranges)
                        return (ReductionHint.INNER, -1)
            return (ReductionHint.INNER, split)
        if reduction_numel_hint <= min_elements_per_thread or numel_hint >= num_sm * 2 * 32:
            return (ReductionHint.DEFAULT, 1)
        r = Reduction(device, dst_dtype, inner_fn, ranges, reduction_ranges, reduction_type, src_dtype, ReductionHint.DEFAULT)

        def get_read_indices(r):
            if False:
                i = 10
                return i + 15
            cb = ComputedBuffer(name=None, layout=FlexibleLayout(device=r.get_device(), dtype=r.get_dtype(), size=r.get_size()), data=r)
            read_writes = cb.get_read_writes()
            range_vars = [r for r in read_writes.range_vars if isinstance(r, sympy.Expr) and (not isinstance(r, sympy.Number))]
            indices = []
            changed = False
            for md in sorted(read_writes.reads, key=lambda x: x.name):
                if all((r in md.index.free_symbols for r in range_vars)):
                    indices.append(md.index)
                    if md.name in V.graph.name_to_buffer:
                        buf = V.graph.name_to_buffer[md.name]
                        original_stride = buf.layout.stride
                        buf.decide_layout()
                        if buf.layout.stride != original_stride:
                            changed = True
            return (indices, changed)
        (indices, changed) = get_read_indices(r)
        if changed:
            (indices, _) = get_read_indices(r)
        if len(indices) == 0:
            return (ReductionHint.DEFAULT, 1)
        ((_, reduction_vars), ranges) = dependencies.index_vars_squeeze(r.get_size(), r.get_reduction_size())
        num_outer = 0
        num_inner = 0
        for i in indices:
            i = V.graph.sizevars.simplify_with_ranges(i, ranges)
            strides = V.graph.sizevars.stride_hints(i, reduction_vars, ranges.keys())
            outer = all((s > 1 for s in strides))
            if outer:
                num_outer += 1
            else:
                num_inner += 1
        if num_inner > num_outer:
            return (ReductionHint.INNER, inner_reduction_splits(reduction_numel_hint, numel_hint))
        else:
            return (ReductionHint.OUTER, outer_reduction_splits(reduction_numel_hint, numel_hint))

    @staticmethod
    def _unroll_reduction_fn(inner_fn, reduction_ranges, reduction_type, src_dtype):
        if False:
            while True:
                i = 10
        'Convert inner_fn from a reduction to an pointwise'
        reduction_ranges = [V.graph.sizevars.evaluate_static_shape(x) for x in reduction_ranges]
        combine_fn = get_reduction_combine_fn(reduction_type, src_dtype)

        def fn(index):
            if False:
                return 10
            return functools.reduce(combine_fn, (value_fn(index, rindex) for rindex in itertools.product(*[range(x) for x in reduction_ranges])))
        if reduction_type in ('argmin', 'argmax'):
            flatten_index = FixedLayout(None, None, reduction_ranges, FlexibleLayout.contiguous_strides(reduction_ranges)).make_indexer()

            def value_fn(index, rindex):
                if False:
                    i = 10
                    return i + 15
                rindex = [sympy.expand(i) for i in rindex]
                return (inner_fn(index, rindex), ops.index_expr(flatten_index(rindex), torch.int64))
            return lambda index: fn(index)[1]
        else:
            value_fn = inner_fn
            return fn

    @classmethod
    def create(cls, device: torch.device, dst_dtype: torch.dtype, src_dtype: torch.dtype, inner_fn: Callable[..., Any], ranges: List[Expr], reduction_ranges: List[Expr], reduction_type: str, reduction_hint: ReductionHint=ReductionHint.DEFAULT, input_node: Optional[IRNode]=None):
        if False:
            for i in range(10):
                print('nop')
        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))
        if reduction_numel == 0:

            def py_cnst(val):
                if False:
                    i = 10
                    return i + 15
                return bool(val) if dst_dtype == torch.bool else float(val) if dst_dtype.is_floating_point else int(val)
            rtypes_to_inits = {'sum': py_cnst(0), 'xor_sum': py_cnst(0), 'prod': py_cnst(1), 'any': py_cnst(0)}
            assert reduction_type in rtypes_to_inits.keys(), f'{reduction_type} not supported for zero-dimension tensors!'

            def const_fn(index):
                if False:
                    print('Hello World!')
                return ops.constant(rtypes_to_inits[reduction_type], dst_dtype)
            return Pointwise.create(device=device, dtype=src_dtype, inner_fn=const_fn, ranges=list(ranges))
        if reduction_numel == 1:
            if reduction_type in ('argmin', 'argmax'):

                def fn(index):
                    if False:
                        while True:
                            i = 10
                    return ops.constant(0, dst_dtype)
            else:

                def fn(index):
                    if False:
                        return 10
                    reduction_index = [sympy.Integer(0) for _ in reduction_ranges]
                    return inner_fn(index, reduction_index)
            return Pointwise.create(device, dst_dtype, fn, ranges)
        if isinstance(reduction_numel, sympy.Integer) and V.graph.sizevars.size_hint(reduction_numel) < config.unroll_reductions_threshold and (sympy_product(ranges) != 1):
            return Pointwise.create(device, dst_dtype, cls._unroll_reduction_fn(inner_fn, reduction_ranges, reduction_type, src_dtype), ranges)
        (hint, split) = cls.num_splits(device, dst_dtype, src_dtype, inner_fn, ranges, reduction_ranges, reduction_type, reduction_numel, input_node)
        if reduction_hint == ReductionHint.DEFAULT:
            reduction_hint = hint
        if split == -1:
            assert input_node is not None
            (new_ranges, new_reduction_ranges) = extract_input_node_reduction_ranges(input_node)
            assert new_ranges is not None
            assert new_reduction_ranges is not None
            return cls.create_multilayer_existing_ranges(device, dst_dtype, src_dtype, inner_fn, ranges, reduction_ranges, new_ranges, new_reduction_ranges, reduction_type, reduction_hint)
        elif split > 1:
            return cls.create_multilayer(device, dst_dtype, src_dtype, inner_fn, ranges, reduction_ranges, reduction_type, split, reduction_hint)
        return TensorBox.create(Reduction(device, dst_dtype, inner_fn, ranges, reduction_ranges, reduction_type, src_dtype, reduction_hint))

    @staticmethod
    def default_accumulator(reduction_type, dtype):
        if False:
            print('Hello World!')
        if reduction_type in {'max', 'argmax'}:
            if is_float_dtype(dtype):
                return float('-inf')
            elif is_boolean_dtype(dtype):
                return 0
            else:
                return torch.iinfo(dtype).min
        if reduction_type in {'min', 'argmin'}:
            if is_float_dtype(dtype):
                return float('inf')
            elif is_boolean_dtype(dtype):
                return 1
            else:
                return torch.iinfo(dtype).max
        return {'sum': 0, 'prod': 1, 'xor_sum': 0, 'any': 0, 'welford_reduce': (0, 0, 0), 'welford_combine': (0, 0, 0)}[reduction_type]

    @staticmethod
    def default_value(reduction_type, dtype):
        if False:
            for i in range(10):
                print('nop')
        if reduction_type == 'welford_reduce':
            return 0
        return Reduction.default_accumulator(reduction_type, dtype)

    @staticmethod
    def _multilayer_second_step_hint(split: int, numel_hint: int, reduction_hint: ReductionHint) -> ReductionHint:
        if False:
            print('Hello World!')
        if split == -1:
            return reduction_hint
        if split <= 512 and numel_hint <= 512 and (reduction_hint == ReductionHint.OUTER):
            return ReductionHint.OUTER_TINY
        if split <= 1024 and numel_hint <= 256 and (reduction_hint == ReductionHint.OUTER):
            return ReductionHint.OUTER_TINY
        return reduction_hint

    @classmethod
    def _multilayer_wrap_loader(cls, loader, reduction_ranges, reduction_numel, split, block_size, default):
        if False:
            for i in range(10):
                print('nop')
        reindex = View.dynamic_reshape_indexer(reduction_ranges, [reduction_numel])
        need_mask = not V.graph.sizevars.is_expr_static_and_true(sympy.Eq(reduction_numel % split, 0))

        def wrapper_fn(index, reduction_index):
            if False:
                for i in range(10):
                    print('nop')
            (reduction_index,) = reduction_index
            (*new_index, reduction_block) = index
            indices = block_size * reduction_block + reduction_index

            def body():
                if False:
                    i = 10
                    return i + 15
                return loader(new_index, reindex([indices]))
            if need_mask:
                mask = ops.lt(ops.index_expr(indices, torch.int32), ops.index_expr(reduction_numel, torch.int32))
                return ops.masked(mask, body, default)
            else:
                return body()
        return wrapper_fn

    @classmethod
    def _multilayer_wrap_loader_existing_ranges(cls, loader, original_ranges, original_reduction_ranges, new_ranges, new_reduction_ranges, default):
        if False:
            print('Hello World!')
        assert len(original_ranges) == 0, f'{original_ranges}= is not equal to []'
        reindex = View.dynamic_reshape_indexer(original_reduction_ranges, tuple(new_ranges) + tuple(new_reduction_ranges))

        def wrapper_fn(index, reduction_index):
            if False:
                print('Hello World!')
            return loader([], reindex(tuple(index) + tuple(reduction_index)))
        return wrapper_fn

    @classmethod
    def create_multilayer_helper(cls, device: torch.device, dst_dtype: torch.dtype, src_dtype: torch.dtype, wrapper_fn: Callable[..., Any], original_ranges: List[Expr], original_reduction_ranges: List[Expr], new_ranges: List[Expr], new_reduction_ranges: List[Expr], reduction_type: str, split: int, reduction_hint: ReductionHint):
        if False:
            print('Hello World!')
        '\n        Break a large reduction up into multiple smaller reductions\n        recursively\n        '
        intermediate_dtype = dst_dtype if dst_dtype not in (torch.float16, torch.bfloat16) else torch.float
        intermediate = Reduction.create(device, intermediate_dtype, src_dtype, wrapper_fn, new_ranges, new_reduction_ranges, reduction_type, reduction_hint)
        intermediate.realize()
        intermediate_loader = intermediate.make_loader()

        def intermediate_fn(index, reduction_index):
            if False:
                print('Hello World!')
            return intermediate_loader([*index, *reduction_index])
        numel_hint = V.graph.sizevars.size_hint(sympy_product(original_ranges))
        reduction_hint = cls._multilayer_second_step_hint(split, numel_hint, reduction_hint)
        assert original_ranges == new_ranges[:len(original_ranges)]
        return TensorBox.create(Reduction(device, dst_dtype, intermediate_fn, original_ranges, new_ranges[len(original_ranges):], reduction_type, src_dtype, reduction_hint))

    @classmethod
    def create_multilayer(cls, device: torch.device, dst_dtype: torch.dtype, src_dtype: torch.dtype, inner_fn: Callable[..., Any], ranges: List[Expr], reduction_ranges: List[Expr], reduction_type: str, split: int, reduction_hint: ReductionHint):
        if False:
            return 10
        '\n        Break a large reduction up into multiple smaller reductions\n        recursively\n        '
        reduction_numel = sympy_product(reduction_ranges)
        block_size = FloorDiv(reduction_numel + (split - 1), split)
        default = cls.default_value(reduction_type, dst_dtype)
        wrapper_fn = cls._multilayer_wrap_loader(inner_fn, reduction_ranges, reduction_numel, split, block_size, default)
        return cls.create_multilayer_helper(device, dst_dtype, src_dtype, wrapper_fn, ranges, reduction_ranges, [*ranges, split], [block_size], reduction_type, split, reduction_hint)

    @classmethod
    def create_multilayer_existing_ranges(cls, device: torch.device, dst_dtype: torch.dtype, src_dtype: torch.dtype, inner_fn: Callable[..., Any], original_ranges: List[Expr], original_reduction_ranges: List[Expr], new_ranges: List[Expr], new_reduction_ranges: List[Expr], reduction_type: str, reduction_hint: ReductionHint):
        if False:
            while True:
                i = 10
        '\n        Break a large reduction up into multiple smaller reductions\n        recursively\n        '
        default = cls.default_value(reduction_type, dst_dtype)
        wrapper_fn = cls._multilayer_wrap_loader_existing_ranges(inner_fn, original_ranges, original_reduction_ranges, new_ranges, new_reduction_ranges, default)
        return cls.create_multilayer_helper(device, dst_dtype, src_dtype, wrapper_fn, original_ranges, original_reduction_ranges, new_ranges, new_reduction_ranges, reduction_type, -1, reduction_hint)

def num_reduction_outputs(reduction_type):
    if False:
        return 10
    return 3 if 'welford' in reduction_type else 1

class WelfordReduction(Reduction):
    output_index: int

    def __init__(self, device, dtype, inner_fns, ranges, reduction_ranges, reduction_type, reduction_hint, output_index):
        if False:
            return 10
        if len(inner_fns) == 1:
            loader = inner_fns[0]
        else:

            def loader(idx, reduction_idx):
                if False:
                    for i in range(10):
                        print('nop')
                return tuple((fn(idx, reduction_idx) for fn in inner_fns))
        super().__init__(device, dtype, loader, ranges, reduction_ranges, reduction_type, dtype, reduction_hint)
        self.output_index = output_index

    def store_reduction(self, output_name, indexer, vars, reduction_vars):
        if False:
            for i in range(10):
                print('nop')
        values = ops.reduction(self.dtype, self.src_dtype, self.reduction_type, self.inner_fn(vars, reduction_vars))
        value = values[self.output_index]
        return ops.store_reduction(output_name, indexer(vars), value)

    @classmethod
    def create(cls, device: torch.device, dtype: torch.dtype, inner_fns: Sequence[Callable[..., Any]], ranges: List[Expr], reduction_ranges: List[Expr], reduction_type: str, reduction_hint: ReductionHint=ReductionHint.DEFAULT):
        if False:
            i = 10
            return i + 15
        assert reduction_type in {'welford_reduce', 'welford_combine'}
        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

        def const(val):
            if False:
                print('Hello World!')

            def inner_fn(idx):
                if False:
                    print('Hello World!')
                return ops.constant(val, dtype)
            return Pointwise.create(device=device, dtype=dtype, inner_fn=inner_fn, ranges=list(ranges))
        if reduction_numel == 0:
            mean = const(0)
            m2 = const(0)
            weight = const(0)
            return (mean, m2, weight)
        if reduction_numel == 1:

            def copy(loader):
                if False:
                    for i in range(10):
                        print('nop')

                def inner_fn(idx):
                    if False:
                        while True:
                            i = 10
                    reduction_index = [sympy.Integer(0) for _ in reduction_ranges]
                    return loader(idx, reduction_index)
                return Pointwise.create(device=device, dtype=dtype, inner_fn=inner_fn, ranges=list(ranges))
            if reduction_type == 'welford_reduce':
                return (copy(inner_fns[0]), const(0), const(1))
            else:
                return tuple((copy(fn) for fn in inner_fns))
        (hint, split) = Reduction.num_splits(device, dtype, dtype, inner_fns[0], ranges, reduction_ranges, reduction_type=reduction_type, reduction_numel=reduction_numel)
        if reduction_hint == ReductionHint.DEFAULT:
            reduction_hint = hint
        if split > 1:
            return cls.create_multilayer(device, dtype, inner_fns, ranges, reduction_ranges, reduction_type, split, reduction_hint)
        results = [TensorBox.create(WelfordReduction(device, dtype, inner_fns, ranges, reduction_ranges, reduction_type, reduction_hint, output_idx)) for output_idx in range(3)]
        for t in results:
            t.realize()
        return results

    @staticmethod
    def default_value(reduction_type, dtype):
        if False:
            return 10
        return (0, 0, 0)

    @classmethod
    def create_multilayer(cls, device: torch.device, dtype: torch.dtype, inner_fns: Sequence[Callable[..., Any]], ranges: List[Expr], reduction_ranges: List[Expr], reduction_type: str, split: int, reduction_hint: ReductionHint):
        if False:
            return 10
        '\n        Break a large reduction up into multiple smaller reductions\n        recursively\n        '
        reduction_numel = sympy_product(reduction_ranges)
        need_mask = not V.graph.sizevars.is_expr_static_and_true(sympy.Eq(reduction_numel % split, 0))
        if need_mask and reduction_type != 'welford_combine':

            def constant(idx, reduction_idx, value):
                if False:
                    for i in range(10):
                        print('nop')
                return ops.constant(value, dtype)
            return cls.create_multilayer(device=device, dtype=dtype, inner_fns=(inner_fns[0], partial(constant, value=0), partial(constant, value=1)), ranges=ranges, reduction_ranges=reduction_ranges, reduction_type='welford_combine', split=split, reduction_hint=reduction_hint)
        block_size = FloorDiv(reduction_numel + (split - 1), split)
        intermediates = WelfordReduction.create(device, dtype, tuple((cls._multilayer_wrap_loader(loader, reduction_ranges, reduction_numel, split, block_size, default=0) for loader in inner_fns)), [*ranges, split], [block_size], reduction_type, reduction_hint)
        for i in intermediates:
            i.realize()
        i_loaders = [i.make_loader() for i in intermediates]

        def intermediate_loader_fn(index, reduction_index, loader):
            if False:
                while True:
                    i = 10
            return loader([*index, *reduction_index])
        numel_hint = V.graph.sizevars.size_hint(sympy_product(ranges))
        reduction_hint = cls._multilayer_second_step_hint(split, numel_hint, reduction_hint)
        return WelfordReduction.create(device, dtype, tuple((partial(intermediate_loader_fn, loader=i.make_loader()) for i in intermediates)), ranges, [split], 'welford_combine', reduction_hint)

def is_storage_and_layout(x):
    if False:
        i = 10
        return i + 15
    try:
        as_storage_and_layout(x, freeze=False)
        return True
    except NotImplementedError:
        return False

def is_contiguous_storage_and_layout(x):
    if False:
        i = 10
        return i + 15
    try:
        (buffer, layout) = as_storage_and_layout(x, freeze=False)
        return layout.is_contiguous()
    except NotImplementedError:
        return False

def as_storage_and_layout(x, freeze=True, want_contiguous=False, stride_order=None):
    if False:
        i = 10
        return i + 15
    'Try to simplify x into a StorageBox and a Layout'
    if isinstance(x, TensorBox):
        return as_storage_and_layout(x.data, freeze=freeze, want_contiguous=want_contiguous, stride_order=stride_order)
    if isinstance(x, StorageBox) and isinstance(x.data, Buffer):
        if freeze:
            if want_contiguous:
                x.data.freeze_layout()
                assert x.data.layout.is_contiguous()
            elif stride_order is not None:
                x.data.freeze_layout_with_stride_order(stride_order)
            else:
                x.data.decide_layout()
        return (x, x.data.layout)
    if isinstance(x, ReinterpretView):
        (buffer, _) = as_storage_and_layout(x.data, freeze=freeze)
        return (buffer, x.layout)
    raise NotImplementedError
as_contiguous_storage_and_layout = functools.partial(as_storage_and_layout, want_contiguous=True)

def is_stride_order_storage_and_layout(x, stride_order):
    if False:
        i = 10
        return i + 15
    try:
        (buffer, layout) = as_storage_and_layout(x, freeze=False)
        return layout.is_stride_ordered(stride_order)
    except NotImplementedError:
        return False

@dataclasses.dataclass
class BaseView(IRNode):
    data: IRNode

    def make_reindexer(self):
        if False:
            return 10
        raise NotImplementedError(f'make_reindexer NYI on {self}')

    def make_indexer(self):
        if False:
            return 10
        inner = self.data.make_indexer()
        reindex = self.make_reindexer()

        def indexer(idx):
            if False:
                while True:
                    i = 10
            return inner(reindex(idx))
        return indexer

    def make_loader(self):
        if False:
            while True:
                i = 10
        inner = self.data.make_loader()
        reindex = self.make_reindexer()

        def loader(idx):
            if False:
                print('Hello World!')
            return inner(reindex(idx))
        return loader

    def get_dtype(self):
        if False:
            i = 10
            return i + 15
        return self.data.get_dtype()

    def get_layout(self):
        if False:
            print('Hello World!')
        return self.data.get_layout()

    def get_device(self):
        if False:
            for i in range(10):
                print('nop')
        return self.data.get_device()

    def get_origin_node(self):
        if False:
            print('Hello World!')
        return None

    def get_name(self):
        if False:
            return 10
        return self.data.get_name()

    def mark_reuse(self, users):
        if False:
            while True:
                i = 10
        return self.data.mark_reuse(users)

    def has_exceeded_max_reads(self):
        if False:
            i = 10
            return i + 15
        return self.data.has_exceeded_max_reads()

    def realize(self):
        if False:
            print('Hello World!')
        return self.data.realize()

    def realize_hint(self):
        if False:
            while True:
                i = 10
        return self.data.realize_hint()

    def get_storage_numel(self):
        if False:
            i = 10
            return i + 15
        return self.data.get_storage_numel()

    def is_extern(self):
        if False:
            while True:
                i = 10
        return self.data.is_extern()

    def get_reads(self):
        if False:
            for i in range(10):
                print('nop')
        with patch.object(FlexibleLayout, 'allow_indexing', True):
            return extract_read_writes(self.make_loader(), self.get_size()).reads

    def unwrap_view(self):
        if False:
            while True:
                i = 10
        x: IRNode = self
        while isinstance(x, BaseView):
            x = x.data
        return x

    def constant_to_device(self, device):
        if False:
            for i in range(10):
                print('nop')
        'Move this to a given device. Requires that all reads are to constants.'
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, 'override_device', device)(loader)
        return Pointwise(device, self.get_dtype(), loader, self.get_size())

@dataclasses.dataclass
class ExpandView(BaseView):
    size: List[Expr]

    @staticmethod
    def _normalize_size(x, new_size):
        if False:
            i = 10
            return i + 15
        'Replace `-1` with correct sizes'
        new_size = list(map(sympy.expand, new_size))
        old_size = x.get_size()
        old_size = [None] * (len(new_size) - len(old_size)) + list(old_size)
        assert len(new_size) == len(old_size)
        for i in range(len(new_size)):
            if new_size[i] == -1:
                assert old_size[i] is not None
                new_size[i] = old_size[i]
        return new_size

    @classmethod
    def create(cls, x, new_size):
        if False:
            while True:
                i = 10
        new_size = cls._normalize_size(x, new_size)
        if is_storage_and_layout(x):
            (storage, old_layout) = as_storage_and_layout(x)
            skip = len(new_size) - len(old_layout.size)
            assert skip >= 0
            new_stride = [sympy.Integer(0)] * skip
            for (stride, size) in zip(old_layout.stride, old_layout.size):
                new_stride.append(stride if size != 1 else sympy.Integer(0))
            new_layout = FixedLayout(old_layout.device, old_layout.dtype, list(new_size), new_stride, old_layout.offset)
            return ReinterpretView(storage, new_layout)
        return ExpandView(x, new_size)

    def get_size(self):
        if False:
            i = 10
            return i + 15
        return self.size

    def make_reindexer(self):
        if False:
            return 10
        target = self.get_size()
        actual = self.data.get_size()
        skip = len(target) - len(actual)

        def reindex(index):
            if False:
                while True:
                    i = 10
            index = list(index[skip:])
            assert len(index) == len(actual)
            for i in range(len(actual)):
                if actual[i] == 1:
                    index[i] = sympy.Integer(0)
            return index
        return reindex

@dataclasses.dataclass
class PermuteView(BaseView):
    dims: List[Expr]

    @classmethod
    def create(cls, x, dims):
        if False:
            for i in range(10):
                print('nop')
        dims = cls._map_neg_dims(dims)
        assert set(dims) == set(range(len(dims)))
        if is_storage_and_layout(x):
            (storage, old_layout) = as_storage_and_layout(x)
            new_layout = FixedLayout(old_layout.device, old_layout.dtype, [old_layout.size[i] for i in dims], [old_layout.stride[i] for i in dims], old_layout.offset)
            return ReinterpretView(storage, new_layout)
        return PermuteView(x, dims)

    @classmethod
    def _map_neg_dims(cls, dims):
        if False:
            i = 10
            return i + 15
        return [dim if dim >= 0 else len(dims) + dim for dim in dims]

    def get_size(self):
        if False:
            return 10
        assert set(self._map_neg_dims(self.dims)) == set(range(len(self.dims)))
        size = self.data.get_size()
        return [size[i] for i in self.dims]

    def make_reindexer(self):
        if False:
            while True:
                i = 10
        inv = {j: i for (i, j) in enumerate(self.dims)}
        inv = [inv[i] for i in range(len(self.dims))]
        assert set(inv) == set(range(len(self.dims)))

        def reindex(index):
            if False:
                return 10
            return [index[i] for i in inv]
        return reindex

class SqueezeView(BaseView):

    @classmethod
    def create(cls, x, *, dim=None):
        if False:
            while True:
                i = 10
        if is_storage_and_layout(x):
            (storage, old_layout) = as_storage_and_layout(x)
            new_size = []
            new_stride = []
            if dim is not None:
                assert isinstance(dim, int), 'expected integer dim argument'
                assert 0 <= dim and dim < len(old_layout.size)
            for (i, (size, stride)) in enumerate(zip(old_layout.size, old_layout.stride)):
                if dim is None:
                    if size != 1:
                        new_size.append(size)
                        new_stride.append(stride)
                elif i != dim:
                    new_size.append(size)
                    new_stride.append(stride)
                else:
                    assert size == 1, 'expected squeezed size to be 1'
            new_layout = FixedLayout(old_layout.device, old_layout.dtype, new_size, new_stride, old_layout.offset)
            return ReinterpretView(storage, new_layout)
        if dim is None:
            return View.create(x, [s for s in x.get_size() if s != 1])
        else:
            assert x.get_size()[dim] == 1
            return View.create(x, [s for (i, s) in enumerate(x.get_size()) if i != dim])

    @staticmethod
    def squeezer(size: Tuple[sympy.Expr, ...]):
        if False:
            while True:
                i = 10
        new_size = [s for s in size if s != 1]
        not_one = [i for (i, s) in enumerate(size) if s != 1]
        length = len(size)

        def reindex(index: List[sympy.Expr]) -> Tuple[sympy.Expr, ...]:
            if False:
                return 10
            assert len(index) == len(not_one), f'{index} {not_one}'
            new_index = [sympy.Integer(0)] * length
            for (idx, s) in zip(not_one, index):
                new_index[idx] = s
            return tuple(new_index)
        return (new_size, reindex)

    def __init__(self, data):
        if False:
            return 10
        raise AssertionError('use SqueezeView.create()')

@dataclasses.dataclass
class GenericView(BaseView):
    size: List[Expr]
    reindex: Callable[..., Any]

    def make_reindexer(self):
        if False:
            while True:
                i = 10
        return self.reindex

    def reindex_str(self):
        if False:
            i = 10
            return i + 15
        index_old = [sympy_symbol(f'i{n}') for n in range(len(self.size))]
        index_new = list(self.reindex(index_old))
        return f"lambda {', '.join(map(str, index_old))}: {index_new}"

    def __str__(self):
        if False:
            return 10
        return self.str_helper([self.data, f'size={self.size}', f'reindex={self.reindex_str()}'])
    __repr__ = __str__

    @classmethod
    def create(cls, x, new_size, reindex):
        if False:
            for i in range(10):
                print('nop')
        return cls(x, list(new_size), reindex)

    def get_size(self):
        if False:
            print('Hello World!')
        return self.size

@dataclasses.dataclass
class View(GenericView):

    @staticmethod
    def handle_negative_index(idx, size):
        if False:
            print('Hello World!')
        idx = sympy.expand(idx)
        size = sympy.expand(size)
        evaluate_expr = V.graph.sizevars.shape_env.evaluate_expr
        if evaluate_expr(sympy.Lt(idx, 0)):
            idx = idx + size
        return idx

    @classmethod
    def create(cls, x, new_size):
        if False:
            while True:
                i = 10
        assert isinstance(new_size, (tuple, list))
        (old_size, new_size) = cls.resolve_negative_size(x.get_size(), new_size)
        if V.graph.sizevars.statically_known_list_equals(old_size, new_size):
            return x
        unbacked_symbols_in_sizes = False
        if len(free_unbacked_symbols(old_size)) > 0 or len(free_unbacked_symbols(new_size)) > 0:
            unbacked_symbols_in_sizes = True
        if 0 in new_size:

            def fake_reindex(index):
                if False:
                    print('Hello World!')
                return tuple([0] * len(old_size))
            return cls(x, list(new_size), fake_reindex)
        elif is_contiguous_storage_and_layout(x) or unbacked_symbols_in_sizes:
            if unbacked_symbols_in_sizes and (not is_contiguous_storage_and_layout(x)):
                x.realize()
            (storage, old_layout) = as_contiguous_storage_and_layout(x)
            new_layout = FixedLayout(old_layout.device, old_layout.dtype, new_size, FlexibleLayout.contiguous_strides(new_size), old_layout.offset)
            return ReinterpretView(storage, new_layout)
        reindex = cls.dynamic_reshape_indexer(old_size, new_size)
        return cls(x, list(new_size), reindex)

    @staticmethod
    def resolve_negative_size(old_size, new_size):
        if False:
            while True:
                i = 10
        new_size = [V.graph.sizevars.simplify(x) for x in new_size]
        old_size = [V.graph.sizevars.simplify(x) for x in old_size]
        new_size = list(new_size)
        for i in range(len(new_size)):
            if new_size[i] == -1:
                new_size[i] = sympy.Integer(1)
                new_size[i] = CleanDiv(sympy_product(old_size), sympy_product(new_size))
                break
        V.graph.sizevars.guard_equals(sympy_product(old_size), sympy_product(new_size))
        return (old_size, new_size)

    @classmethod
    def dynamic_reshape_indexer(cls, old_size, new_size):
        if False:
            print('Hello World!')
        try:
            reindex = cls._dynamic_reshape_indexer(old_size, new_size)
        except (AssertionError, IndexError):
            flat = [sympy_product(old_size)]
            reindex1 = cls._dynamic_reshape_indexer(old_size, flat)
            reindex2 = cls._dynamic_reshape_indexer(flat, new_size)
            reindex = fuse_reindexing(reindex1, reindex2)
        return reindex

    @staticmethod
    def _dynamic_reshape_indexer(old_size, new_size):
        if False:
            print('Hello World!')
        '\n        Perform a reshape entirely by modifying indexing math\n        '
        size_hint = V.graph.sizevars.size_hint
        vars = [sympy_symbol(f'view{i}') for i in range(len(new_size))]
        stack_new = list(zip(vars, new_size))
        stack_old = list(old_size)
        view_expr = []
        while stack_new and stack_old:
            size_old = stack_old.pop()
            (var, size_new) = stack_new.pop()
            if size_old == 1:
                view_expr.append(sympy.Integer(0))
                stack_new.append((var, size_new))
            elif size_new == 1:
                stack_old.append(size_old)
            elif size_hint(size_new) == size_hint(size_old):
                view_expr.append(var)
                V.graph.sizevars.guard_equals(size_new, size_old)
            elif size_hint(size_new) < size_hint(size_old):
                while size_hint(size_new) < size_hint(size_old):
                    (var2, size_new2) = stack_new.pop()
                    var = var2 * size_new + var
                    size_new = size_new * size_new2
                view_expr.append(var)
                V.graph.sizevars.guard_equals(size_new, size_old)
            elif size_hint(size_new) > size_hint(size_old):
                divisor = sympy.Integer(1)
                modulus = size_old
                view_expr.append(ModularIndexing(var, divisor, modulus))
                divisor = divisor * modulus
                while size_hint(size_new) > size_hint(size_old):
                    modulus = stack_old.pop()
                    view_expr.append(ModularIndexing(var, divisor, modulus))
                    divisor = divisor * modulus
                    size_old = size_old * modulus
                V.graph.sizevars.guard_equals(size_new, size_old)
            else:
                raise AssertionError()
        while stack_old:
            size_old = stack_old.pop()
            V.graph.sizevars.guard_equals(size_old, 1)
            view_expr.append(sympy.Integer(0))
        while stack_new:
            (var, size_new) = stack_new.pop()
            V.graph.sizevars.guard_equals(size_new, 1)
        view_expr = list(reversed(view_expr))
        assert len(view_expr) == len(old_size)

        def reindex(index):
            if False:
                for i in range(10):
                    print('nop')
            assert len(index) == len(vars), (len(index), len(vars))
            replacements = dict(zip(vars, index))
            return tuple((sympy_subs(x, replacements) for x in view_expr))
        return reindex

@dataclasses.dataclass
class ReinterpretView(BaseView):
    """Pretend our storage has a different layout"""
    layout: 'Layout'

    def __post_init__(self):
        if False:
            i = 10
            return i + 15
        super().__post_init__()
        if isinstance(self.data, BaseView):
            self.data = self.data.unwrap_view()

    def __str__(self):
        if False:
            return 10
        return self.str_helper([self.data, self.layout])
    __repr__ = __str__

    def get_name(self):
        if False:
            while True:
                i = 10
        return self.data.get_name()

    def get_device(self):
        if False:
            print('Hello World!')
        return self.layout.device

    def get_origin_node(self):
        if False:
            while True:
                i = 10
        return None

    def get_dtype(self):
        if False:
            while True:
                i = 10
        return self.layout.dtype

    def get_size(self):
        if False:
            while True:
                i = 10
        return list(self.layout.size)

    def get_stride(self):
        if False:
            i = 10
            return i + 15
        return list(self.layout.stride)

    def make_loader(self):
        if False:
            i = 10
            return i + 15

        def loader(index):
            if False:
                while True:
                    i = 10
            indexer = self.layout.make_indexer()
            return ops.load(self.get_name(), indexer(index))
        return loader

    def make_indexer(self):
        if False:
            i = 10
            return i + 15
        return self.layout.make_indexer()

    def get_layout(self):
        if False:
            print('Hello World!')
        return self.layout

    def freeze_layout(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def codegen_reference(self, writer=None):
        if False:
            for i in range(10):
                print('nop')
        return V.graph.wrapper_code.codegen_reinterpret_view(self.data, self.layout.size, self.layout.stride, self.layout.offset, writer)

class SliceView(View):

    @classmethod
    def create(cls, x, dim, start, end, step=1):
        if False:
            while True:
                i = 10
        step = sympy.expand(step)
        assert step > 0
        try:
            if start == 0 and end >= 2 ** 63 - 1 and (step == 1):
                return x
        except TypeError:
            pass
        sizevars = V.graph.sizevars
        new_size = list(x.get_size())
        start = cls.handle_negative_index(start, new_size[dim])
        end = cls.handle_negative_index(end, new_size[dim])
        end = sizevars.evaluate_min(end, new_size[dim])
        start = sizevars.evaluate_min(start, end)
        if start == 0 and sizevars.size_hint(end - new_size[dim]) == 0 and (step == 1):
            sizevars.guard_equals(end, new_size[dim])
            return x
        new_size[dim] = FloorDiv(end - start + (step - 1), step)
        if is_storage_and_layout(x):
            (storage, old_layout) = as_storage_and_layout(x)
            new_stride = list(old_layout.stride)
            new_stride[dim] = new_stride[dim] * step
            new_layout = FixedLayout(old_layout.device, old_layout.dtype, new_size, new_stride, old_layout.offset + old_layout.stride[dim] * start)
            return ReinterpretView(storage, new_layout)

        def reindex(index):
            if False:
                for i in range(10):
                    print('nop')
            assert len(index) == len(new_size), f'wrong ndim {index} {new_size}'
            index = list(index)
            index[dim] = index[dim] * step + start
            return index
        return SliceView(x, size=new_size, reindex=reindex)

class BaseConstant(IRNode):
    dtype: torch.dtype
    device: torch.device

    def get_size(self):
        if False:
            i = 10
            return i + 15
        return ()

    def get_dtype(self):
        if False:
            print('Hello World!')
        return self.dtype

    def get_device(self):
        if False:
            i = 10
            return i + 15
        return self.device

    def get_origin_node(self):
        if False:
            i = 10
            return i + 15
        return None

    def mark_reuse(self, users):
        if False:
            for i in range(10):
                print('nop')
        pass

    def has_exceeded_max_reads(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def get_reads(self):
        if False:
            while True:
                i = 10
        return ()

    def is_extern(self):
        if False:
            for i in range(10):
                print('nop')
        return False

@dataclasses.dataclass
class Constant(BaseConstant):
    value: Any
    dtype: torch.dtype
    device: torch.device

    def make_loader(self):
        if False:
            for i in range(10):
                print('nop')

        def loader(index):
            if False:
                return 10
            return ops.constant(self.value, self.dtype)
        return loader

    def realize(self):
        if False:
            print('Hello World!')
        pass

    def constant_to_device(self, device):
        if False:
            print('Hello World!')
        return Constant(self.value, self.dtype, device)

@dataclasses.dataclass
class IndexingConstant(BaseConstant):
    index: Any
    dtype: torch.dtype
    device: torch.device

    def make_loader(self):
        if False:
            print('Hello World!')

        def loader(index):
            if False:
                return 10
            return ops.index_expr(self.index, self.dtype)
        return loader

    def constant_to_device(self, device):
        if False:
            return 10
        return IndexingConstant(self.index, self.dtype, device)

@dataclasses.dataclass
class Layout(IRNode):

    def __init__(self, device: torch.device, dtype: torch.dtype, size: List[Expr], stride: Optional[Sequence[Union[Expr, int]]], offset: Expr=Integer(0)):
        if False:
            i = 10
            return i + 15
        assert stride is None or len(size) == len(stride), f'size={size}, stride={stride}'
        self.device = device
        self.dtype = dtype
        assert all((isinstance(s, (Expr, int)) for s in size))
        self.size = size
        self._stride = stride
        self.offset = offset

    @property
    def stride(self):
        if False:
            i = 10
            return i + 15
        return self._stride

    def __str__(self):
        if False:
            while True:
                i = 10
        offset = ''
        if self.offset != 0:
            offset = f', offset={self.offset}'
        return f"{type(self).__name__}('{self.device.type}', {self.dtype}, size={self.size}, stride={self.stride}{offset})"
    __repr__ = __str__

    def is_contiguous(self):
        if False:
            for i in range(10):
                print('nop')
        for (left, right, size) in zip(self.stride, FlexibleLayout.contiguous_strides(self.size), self.size):
            if size != 1 and left != right:
                return False
        return True

    def is_channels_last_contiguous(self):
        if False:
            i = 10
            return i + 15
        ndim = len(self.size)
        if ndim not in [4, 5]:
            return False
        for (left, right, size) in zip(self.stride, make_channels_last_strides_for(self.size), self.size):
            if size != 1 and left != right:
                return False
        return True

    def is_transposed(self):
        if False:
            i = 10
            return i + 15
        for (left, right, size) in zip(self.stride, reversed(FlexibleLayout.contiguous_strides(self.size)), self.size):
            if size != 1 and left != right:
                return False
        return True

    def is_stride_ordered(self, order):
        if False:
            for i in range(10):
                print('nop')
        assert len(self.stride) == len(order)
        non_1_indices = [i for (i, dim) in enumerate(self.size) if V.graph.sizevars.size_hint(dim, fallback=2) != 1]
        stride = [self.stride[i] for i in non_1_indices]
        order = [order[i] for i in non_1_indices]

        def sorted_indices(arr):
            if False:
                return 10
            sorted_arr = sorted(arr)
            return [sorted_arr.index(element) for element in arr]
        order = sorted_indices(order)
        stride_ordered = [-1] * len(order)
        for i in range(len(order)):
            stride_ordered[order[i]] = V.graph.sizevars.size_hint(stride[i])
        for i in range(len(order) - 1):
            if stride_ordered[i] > stride_ordered[i + 1]:
                return False
        return True

    def is_channels_last_stride_ordered(self):
        if False:
            while True:
                i = 10
        order = [0] + list(reversed(range(1, len(self.stride) - 1)))
        order = [len(order)] + order
        return self.is_stride_ordered(order)

    def as_fixed(self):
        if False:
            print('Hello World!')
        return FixedLayout(self.device, self.dtype, self.size, self.stride, self.offset)

    def make_indexer(self):
        if False:
            for i in range(10):
                print('nop')
        assert FlexibleLayout.allow_indexing, f'convert {type(self).__name__} to FixedLayout first'
        return self.as_fixed().make_indexer()

    def __eq__(self, other) -> bool:
        if False:
            return 10
        return self.device == other.device and self.dtype == other.dtype and (self.size == other.size) and (self.stride == other.stride) and (self.offset == other.offset)

    def storage_size(self) -> sympy.Expr:
        if False:
            for i in range(10):
                print('nop')
        return compute_required_storage_length(self.size, self.stride, self.offset)

class FixedLayout(Layout):
    """A Tensor layout we cannot change"""

    def __init__(self, device: torch.device, dtype: torch.dtype, size: Union[List[Expr], List[int]], stride: Optional[Sequence[Union[Expr, int]]]=None, offset: Union[Expr, int]=Integer(0)):
        if False:
            i = 10
            return i + 15
        if stride is None:
            stride = FlexibleLayout.contiguous_strides(size)
        super().__init__(device, dtype, size, stride, offset)

    def make_indexer(self):
        if False:
            return 10
        'A closure containing math to read a given element'

        def indexer(index):
            if False:
                return 10
            assert len(index) == len(self.stride) == len(self.size)
            result = self.offset
            for (idx, stride, sz) in zip(index, self.stride, self.size):
                if sz != 1:
                    result = result + idx * stride
            return result
        return indexer

class FlexibleLayout(Layout):
    """A Tensor layout we are allowed to change"""
    allow_indexing = False

    @staticmethod
    def contiguous_strides(sizes):
        if False:
            i = 10
            return i + 15
        if len(sizes) == 0:
            return []
        reversed_strides = [sympy.Integer(1)]
        for size in reversed(sizes[1:]):
            reversed_strides.append(size * reversed_strides[-1])
        return list(reversed(reversed_strides))

    @staticmethod
    def fill_ordered(sizes, order):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a stride based on the order the dimensions should be filled in.\n\n        In this format, channels last would be:\n            [1, 3, 2, 0]\n        '
        assert set(range(len(sizes))) == set(order)
        next_stride = sympy.Integer(1)
        strides = [None] * len(order)
        for i in order:
            strides[i] = next_stride
            next_stride = next_stride * sizes[i]
        return strides

    @staticmethod
    def stride_ordered(sizes, order):
        if False:
            return 10
        '\n        Create a stride based on the sorted order of a permuted range.\n\n        In this format, channels last would be:\n            [3, 0, 2, 1]\n        '
        assert set(range(len(sizes))) == set(order)
        fill_order = stride_order2fill_order(order)
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    @staticmethod
    def same_ordered(sizes, stride):
        if False:
            i = 10
            return i + 15
        '\n        Create a stride that has the same stride order as given stride\n\n        For example, if given stride is [1000, 1, 100, 10],\n        the fill order should be [1, 3, 2, 0]\n        '
        assert len(sizes) == len(stride)
        stride = [V.graph.sizevars.size_hint(x) for x in stride]
        fill_order = sorted(range(len(stride)), key=stride.__getitem__)
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    def as_stride_order(self, order):
        if False:
            while True:
                i = 10
        return FixedLayout(self.device, self.dtype, self.size, self.stride_ordered(self.size, order), self.offset)

    def as_fill_order(self, order):
        if False:
            print('Hello World!')
        return FixedLayout(self.device, self.dtype, self.size, self.fill_ordered(self.size, order), self.offset)

    def as_same_order(self, stride):
        if False:
            i = 10
            return i + 15
        return FixedLayout(self.device, self.dtype, self.size, self.same_ordered(self.size, stride), self.offset)

    def __init__(self, device, dtype, size, stride_order=None):
        if False:
            while True:
                i = 10
        if stride_order:
            strides = FlexibleLayout.fill_ordered(size, stride_order)
        else:
            strides = FlexibleLayout.contiguous_strides(size)
        super().__init__(device, dtype, size, strides)

class AliasedLayout(Layout):
    """Shares the same storage as another tensor"""

    def __init__(self, view: Union[BaseView, 'TensorBox']):
        if False:
            for i in range(10):
                print('nop')
        layout = view.get_layout()
        super().__init__(layout.device, layout.dtype, layout.size, layout.stride)
        self.view = view

    def make_indexer(self):
        if False:
            while True:
                i = 10
        return self.as_fixed().make_indexer()

    def maybe_guard_aligned(self):
        if False:
            for i in range(10):
                print('nop')
        offset = self.view.get_layout().offset
        if offset == 0:
            return True
        from .compile_fx import ALIGNMENT
        return V.graph.sizevars.statically_known_multiple_of(offset, ALIGNMENT)

class NoneLayout(IRNode):

    def __init__(self, device):
        if False:
            while True:
                i = 10
        self.device = device
        self.size = [0]
        self.stride = [0]

    def storage_size(self):
        if False:
            print('Hello World!')
        return 0

    def as_fixed(self):
        if False:
            i = 10
            return i + 15
        return self

class MutationLayout(Layout):

    def __init__(self, target: IRNode):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(target.get_device(), target.get_dtype(), target.get_size(), None)
        self.target = target
        name = self.get_buffer().get_name()
        V.graph.mark_buffer_mutated(name)

    @Layout.stride.getter
    def stride(self):
        if False:
            for i in range(10):
                print('nop')
        return self.real_layout().stride

    def storage_size(self) -> sympy.Expr:
        if False:
            while True:
                i = 10
        return self.real_layout().storage_size()

    def get_buffer(self) -> 'Buffer':
        if False:
            for i in range(10):
                print('nop')

        def unwrap_views(target):
            if False:
                print('Hello World!')
            if isinstance(target, MutationLayout):
                return unwrap_views(target.target)
            if isinstance(target, BaseView):
                return unwrap_views(target.unwrap_view())
            if isinstance(target, MutableBox):
                return unwrap_views(target.data)
            return target
        result = unwrap_views(self.target)
        assert isinstance(result, Buffer), 'MutationLayout must refer to a buffer'
        return result

    def real_layout(self):
        if False:
            i = 10
            return i + 15
        return self.get_buffer().layout

    @classmethod
    def realize_into(cls, src, dst):
        if False:
            for i in range(10):
                print('nop')
        dst.realize()
        V.graph.mark_buffer_mutated(dst.get_name())
        if isinstance(src, TensorBox):
            src = src.data
        src.realize_hint()
        src = Pointwise.create(device=src.get_device(), dtype=src.get_dtype(), inner_fn=src.make_loader(), ranges=[V.graph.sizevars.guard_equals(a, b) for (a, b) in zip(src.get_size(), dst.get_size())]).data
        src.realize()
        assert isinstance(src.data.layout, FlexibleLayout)
        src.data.layout = MutationLayout(dst)
        return src.data

    def as_fixed(self):
        if False:
            return 10
        return self

    def make_indexer(self):
        if False:
            return 10
        return self.target.make_indexer()

@dataclasses.dataclass
class Buffer(IRNode):
    name: Optional[str]
    layout: Layout

    def __post_init__(self):
        if False:
            print('Hello World!')
        super().__post_init__()
        self.origin_node = None

    def make_indexer(self):
        if False:
            print('Hello World!')
        return self.layout.make_indexer()

    def get_name(self):
        if False:
            i = 10
            return i + 15
        assert self.name
        return self.name

    def get_device(self):
        if False:
            print('Hello World!')
        return self.layout.device

    def get_origin_node(self):
        if False:
            i = 10
            return i + 15
        return self.origin_node

    def get_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self.layout, 'dtype', None)

    def get_size(self):
        if False:
            while True:
                i = 10
        return list(self.layout.size)

    def get_stride(self):
        if False:
            return 10
        return list(self.layout.stride)

    def get_offset(self):
        if False:
            print('Hello World!')
        return self.layout.offset

    def get_layout(self):
        if False:
            i = 10
            return i + 15
        return self.layout

    def get_storage_numel(self):
        if False:
            i = 10
            return i + 15
        return self.get_numel()

    def is_extern(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def freeze_layout(self):
        if False:
            return 10
        if not isinstance(self.layout, (MultiOutputLayout, AliasedLayout)):
            self.layout = self.layout.as_fixed()

    def freeze_layout_with_stride_order(self, order):
        if False:
            while True:
                i = 10
        assert isinstance(self.layout, FlexibleLayout)
        self.layout = self.layout.as_stride_order(order)

    def freeze_layout_with_fill_order(self, order):
        if False:
            return 10
        assert isinstance(self.layout, FlexibleLayout)
        self.layout = self.layout.as_fill_order(order)

    def freeze_layout_with_same_order(self, stride):
        if False:
            while True:
                i = 10
        assert isinstance(self.layout, FlexibleLayout)
        self.layout = self.layout.as_same_order(stride)

    def is_zero_elements(self):
        if False:
            for i in range(10):
                print('nop')
        return V.graph.sizevars.is_expr_static_and_true(sympy.Eq(self.get_numel(), 0))

    def make_loader(self):
        if False:
            while True:
                i = 10
        if self.is_zero_elements():
            return partial(nop_loader_fn, dtype=self.get_dtype())

        def loader(index):
            if False:
                for i in range(10):
                    print('nop')
            indexer = self.layout.make_indexer()
            return ops.load(self.name, indexer(index))
        return loader

    def is_no_op(self):
        if False:
            i = 10
            return i + 15
        return False

    def codegen_reference(self, writer=None):
        if False:
            while True:
                i = 10
        return self.get_name()

    def decide_layout(self):
        if False:
            return 10
        pass

    def get_alias_names(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self.layout, AliasedLayout):
            return [self.layout.view.get_name()]
        return ()

    def get_mutation_names(self):
        if False:
            print('Hello World!')
        if isinstance(self.layout, MutationLayout):
            return [self.layout.target.get_name()]
        return ()

    def get_read_writes(self):
        if False:
            while True:
                i = 10
        with patch.object(FlexibleLayout, 'allow_indexing', True):
            return extract_read_writes(self.make_loader(), self.get_size())

    def get_reads(self):
        if False:
            for i in range(10):
                print('nop')
        return self.get_read_writes().reads

    def get_unbacked_symbol_defs(self):
        if False:
            while True:
                i = 10
        '\n        Returns the unbacked symbols which are defined by this IR node,\n        because this is a data-dependent IR node, or item()\n        '
        if isinstance(self.layout, (NoneLayout, MultiOutputLayout)):
            return set()
        defs = free_unbacked_symbols(self.get_size()) | free_unbacked_symbols(self.get_stride()) | free_unbacked_symbols(self.get_offset())
        return defs - self.get_unbacked_symbol_uses()

    def get_unbacked_symbol_uses(self):
        if False:
            print('Hello World!')
        '\n        Returns the unbacked symbols which are required to be in scope in\n        order to successfully perform codegen for this buffer.  For example,\n        a buffer that corresponds to an extern kernel call that takes i0 as\n        an argument would return {i0} here.  This is used to generate necessary\n        dependencies that ensure we actually bind i0 in codegen before you\n        try to use it.\n\n        Note that this is NOT transitive; in particular, if this buffer takes\n        in as input another buffer with dynamic shape (e.g., (i0,)), we will\n        not report it here, because you will already have a dependency\n        on that buffer, which will eventually have a dependency on i0 if\n        necessary.\n        '
        return set()

    def codegen_unbacked_symbol_defs(self, wrapper):
        if False:
            while True:
                i = 10
        symbols_to_define = self.get_unbacked_symbol_defs()
        for (i, s) in enumerate(self.get_size()):
            if s in symbols_to_define:
                wrapper.writeline(f'{wrapper.codegen_unbacked_symbol_decl(s)} = {self.get_name()}.size({i}){wrapper.ending}')
                symbols_to_define.remove(s)
        for (i, s) in enumerate(self.get_stride()):
            if s in symbols_to_define:
                wrapper.writeline(f'{wrapper.codegen_unbacked_symbol_decl(s)} = {self.get_name()}.stride({i}){wrapper.ending}')
                symbols_to_define.remove(s)
        if (s := self.get_offset()) in symbols_to_define:
            wrapper.writeline(f'{wrapper.codegen_unbacked_symbol_decl(s)} = {self.get_name()}.storage_offset(){wrapper.ending}')
            symbols_to_define.remove(s)
        assert not symbols_to_define, f'unbacked symint {s} not written out, check comment above'

    def realize(self):
        if False:
            print('Hello World!')
        pass

    def get_workspace_size(self):
        if False:
            i = 10
            return i + 15
        '\n        Gets extra global memory size needed by this buffer.\n        Some algorithms (e.g. group gemm) may require extra global memory in the generated code.\n        '
        return 0

    def should_allocate(self):
        if False:
            for i in range(10):
                print('nop')
        return False

class InputBuffer(Buffer):
    pass

class ConstantBuffer(InputBuffer):
    override_device = None

    def make_loader(self):
        if False:
            while True:
                i = 10

        def loader(index):
            if False:
                return 10
            indexer = self.layout.make_indexer()
            return ops.load(V.graph.constant_name(self.name, self.override_device), indexer(index))
        return loader

    def constant_to_device(self, device):
        if False:
            for i in range(10):
                print('nop')
        return ConstantBuffer(V.graph.constant_name(self.name, device), self.layout)

class NoneAsConstantBuffer(IRNode):

    def codegen_reference(self, writer=None):
        if False:
            for i in range(10):
                print('nop')
        return V.graph.wrapper_code.none_str

class ShapeAsConstantBuffer(IRNode):

    def __init__(self, shape):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.shape = shape

    def codegen_reference(self, writer=None):
        if False:
            i = 10
            return i + 15
        expr = V.graph.wrapper_code.expr_printer(V.graph.sizevars.simplify(self.shape))
        if V.graph.cpp_wrapper:
            return f'torch::tensor({expr})'
        else:
            return expr

@dataclasses.dataclass
class ComputedBuffer(Buffer):
    data: Loops

    def get_computed_buffer_name(self):
        if False:
            print('Hello World!')
        '\n        Returns self.name if it exists, otherwise returns the name of the data node if that exists.\n        If neither exist, returns None.\n        '
        if self.name is not None:
            return self.name
        if hasattr(self.data, 'name'):
            return self.data.name
        return None

    @cache_on_self
    def num_reads(self):
        if False:
            print('Hello World!')
        return len(self.get_read_writes().reads)

    def get_read_writes(self):
        if False:
            while True:
                i = 10
        with patch.object(FlexibleLayout, 'allow_indexing', True):
            if self.data.get_reduction_type():
                return extract_read_writes(self.get_store_function(), self.data.get_size(), self.data.get_reduction_size())
            else:
                return extract_read_writes(self.get_store_function(), self.data.get_size())

    def get_unbacked_symbol_uses(self):
        if False:
            for i in range(10):
                print('nop')
        return free_unbacked_symbols(self.get_size()) | free_unbacked_symbols(self.get_stride()) | free_unbacked_symbols(self.get_offset())

    def make_loader(self):
        if False:
            print('Hello World!')
        if hasattr(self.data, 'make_loader') and self.name not in V.graph.mutated_buffers and (self.num_reads() == 0):
            return self.data.make_loader()
        return super().make_loader()

    def get_store_function(self):
        if False:
            return 10
        indexer = self.layout.as_fixed().make_indexer()
        if isinstance(self.data, Reduction):
            return partial(self.data.store_reduction, self.name, indexer)
        else:
            assert isinstance(self.data, Pointwise)
            return partial(self.data.store_output, self.name, indexer)

    def get_fill_order(self):
        if False:
            print('Hello World!')
        '\n        If our layout is still flexible, try to determine the stride order based on stride orders of reads.\n\n        TODO(jansel): A better algorithm here would look at downstream consumers of this\n                      value and try to do global graph-level layout optimization.\n                      This is also something just begging to be autotuned.\n        '
        if isinstance(self.layout, FlexibleLayout):
            ((index_vars, reduction_vars), _) = dependencies.index_vars_squeeze(self.data.get_size(), self.data.get_reduction_size())
            reads = self.get_read_writes().reads
            reads_bufs = [V.graph.name_to_buffer[r.name] if r.name in V.graph.name_to_buffer.keys() else None for r in reads]
            assert all((isinstance(r, (dependencies.StarDep, dependencies.MemoryDep)) for r in reads))
            reads = [sympy_subs(r.index, {v: sympy.Integer(0) for v in reduction_vars if v != 0}) for r in reads if isinstance(r, dependencies.MemoryDep)]
            if reads:
                stride_lengths = [V.graph.sizevars.stride_hints(expr, index_vars) for expr in reads]
                from .scheduler import pick_loop_order
                return pick_loop_order(stride_lengths, self.get_size())
        return None

    def decide_layout(self):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.layout, FlexibleLayout):
            order = self.get_fill_order()
            if order:
                self.freeze_layout_with_fill_order(order)
            else:
                self.freeze_layout()

    def simplify_and_reorder(self):
        if False:
            print('Hello World!')
        '\n        This is a main place where we do loop transformations in a\n        backend-agnostic way.\n\n        Here we:\n            1) Remove any 1 dimensions\n            2) Fuse contiguous dimensions together\n            3) Reorder dimensions based on stride orders\n        '
        (args, var_ranges) = dependencies.index_vars_squeeze(self.data.get_size(), self.data.get_reduction_size(), prefix='q')
        with patch.object(ConstantBuffer, 'override_device', self.get_device()):
            body = LoopBody(self.get_store_function(), args if self.get_reduction_type() else args[:1], var_ranges)
        index_formulas = [*body.indexing_exprs.values()]
        reads_bufs = [V.graph.name_to_buffer[reads_name] if reads_name in V.graph.name_to_buffer.keys() else None for reads_name in body.reads_name2expr.keys()]
        memory_addrs = [*body.reads_name2expr.values(), *body.writes_name2expr.values()]
        index_vars = []
        reduce_vars: List[Any] = []
        index_size = []
        reduce_size = []
        for (v, s) in var_ranges.items():
            if v in args[0]:
                assert not reduce_vars
                index_vars.append(v)
                index_size.append(s)
            else:
                assert v in args[1]
                reduce_vars.append(v)
                reduce_size.append(s)
        reordering_reindex = [same_reorder(range(len(index_vars)))] * len(memory_addrs)
        for (i, reads_buf) in enumerate(reads_bufs):
            if isinstance(reads_buf, ComputedBuffer) and hasattr(reads_buf, 'iter_reordering_reindex'):
                reordering_reindex[i] = reads_buf.iter_reordering_reindex

        def simplify_and_reorder(x_vars, support_vars, sizes, reordering_reindex=None):
            if False:
                i = 10
                return i + 15
            (sizes, reindex0, reindex1) = self._apply_loop_reordering(x_vars, support_vars, sizes, memory_addrs, reordering_reindex)
            x_vars = reindex0(x_vars)
            (sizes, reindex2, prune) = V.graph.sizevars._simplify_loops(x_vars, sizes, index_prevent_reordering(index_formulas, x_vars, sizes))
            x_vars = prune(x_vars)
            reindex = fuse_reindexing(reindex1, reindex2)
            return (sizes, reindex, reindex1)
        support_vars = index_vars + reduce_vars
        (iter_ranges, iter_reindex, iter_reordering_reindex) = simplify_and_reorder(index_vars, support_vars, index_size, reordering_reindex)
        (reduce_ranges, reduce_reindex, _) = simplify_and_reorder(reduce_vars, support_vars, reduce_size)
        if len(iter_ranges) == len(index_vars):
            self.iter_reordering_reindex = iter_reordering_reindex
        ((iter_vars, reduce_vars), var_ranges) = dependencies.index_vars_no_squeeze(iter_ranges, reduce_ranges, prefix='z')
        body = LoopBody(body, [iter_reindex(iter_vars), reduce_reindex(reduce_vars)], var_ranges)
        return ((iter_ranges, reduce_ranges), body)

    @staticmethod
    def _apply_loop_reordering(index_vars, support_vars, sizes, memory_addrs, reordering_reindex=None, priority_idx=None):
        if False:
            while True:
                i = 10
        '\n        Shuffle the order of loops around to hopefully improve performance.\n        '
        from .scheduler import pick_loop_order
        if priority_idx is None:
            priority_idx = []
        try:
            strides = [V.graph.sizevars.stride_hints(expr, index_vars, support_vars) for expr in memory_addrs]
            assert len(strides) == len(memory_addrs) and len(strides[0]) == len(index_vars)
            if reordering_reindex is not None:
                for i in range(len(memory_addrs)):
                    try:
                        strides[i] = reordering_reindex[i](strides[i])
                    except AssertionError:
                        pass
            order = list(reversed(pick_loop_order(strides, sizes, priority_idx)))
        except Exception:
            if config.debug:
                log.warning('Did not simplify complex index:\n%s\n%s', dict(zip(index_vars, sizes)), memory_addrs)
            order = list(range(len(sizes)))
        sizes = [sizes[i] for i in order]
        return (sizes, same_reorder(order), inverse_reorder(order))

    def get_reduction_size(self):
        if False:
            while True:
                i = 10
        return self.data.get_reduction_size()

    def get_reduction_type(self):
        if False:
            for i in range(10):
                print('nop')
        return self.data.get_reduction_type()

    def is_no_op(self):
        if False:
            print('Hello World!')
        return self.data.is_zero_elements()

    def should_allocate(self):
        if False:
            return 10
        return True

    def constant_to_device(self, device):
        if False:
            print('Hello World!')
        'Move this to a given device. Requires that all reads are to constants.'
        return self.data.constant_to_device(device)

class TemplateBuffer(Buffer):
    """
    Represents a Triton (in the future other type) of template operator
    that we can fuse an epilogue onto.
    """

    def __init__(self, layout, inputs, make_kernel_render):
        if False:
            i = 10
            return i + 15
        super().__init__(name=None, layout=layout)
        self.inputs = InputsKernel.unwrap_storage(inputs)
        self.make_kernel_render = make_kernel_render
        self.name = V.graph.register_buffer(self)

    def get_read_writes(self):
        if False:
            print('Hello World!')
        return self.normalized_read_writes()

    def normalized_read_writes(self):
        if False:
            print('Hello World!')
        name = self.get_name()
        indexer = self.layout.make_indexer()

        def dummy(index, rindex):
            if False:
                return 10
            assert len(rindex) == 0
            return ops.store(name, indexer(index), 'fake')
        deps = dependencies.extract_read_writes(dummy, self.get_size(), (), normalize=True)
        deps.reads = {dependencies.StarDep(x.get_name()) for x in self.inputs}
        return deps

    def get_reduction_size(self):
        if False:
            i = 10
            return i + 15
        return 1

    def get_reduction_type(self):
        if False:
            return 10
        return None

    def is_no_op(self):
        if False:
            return 10
        return False

    def should_allocate(self):
        if False:
            i = 10
            return i + 15
        return True

    def simplify_and_reorder(self):
        if False:
            i = 10
            return i + 15
        return ((self.get_size(), ()), None)

class TritonTemplateBuffer(TemplateBuffer):
    pass

class CUDATemplateBuffer(TemplateBuffer):

    def __init__(self, layout, inputs, make_kernel_render, workspace_size: int, template: 'CUDATemplate'):
        if False:
            while True:
                i = 10
        super().__init__(layout, inputs, make_kernel_render)
        self.workspace_size = workspace_size
        self.template = template

    def get_workspace_size(self):
        if False:
            while True:
                i = 10
        return self.workspace_size if self.workspace_size is not None else 0

@dataclasses.dataclass
class InputsKernel(Buffer):
    inputs: List[Buffer]

    def get_read_writes_input(self, x):
        if False:
            for i in range(10):
                print('nop')
        return dependencies.StarDep(x.get_name())

    def get_read_writes(self):
        if False:
            return 10
        star_dep = []
        for input in self.inputs:
            if isinstance(input, list):
                star_dep.extend([self.get_read_writes_input(x) for x in input])
            else:
                star_dep.append(self.get_read_writes_input(input))
        return dependencies.ReadWrites(set(star_dep), {dependencies.StarDep(self.get_name())}, set(), [], None, op_counts=collections.Counter())

    @staticmethod
    def unwrap_storage_for_input(x):
        if False:
            print('Hello World!')
        if isinstance(x, TensorBox):
            x = x.data
        if isinstance(x, StorageBox):
            x = x.data
        if isinstance(x, BaseView) and (not isinstance(x, ReinterpretView)):
            x = ExternKernel.realize_input(x)
        assert isinstance(x, (Buffer, ReinterpretView)), x
        return x

    @staticmethod
    def unwrap_storage(inputs):
        if False:
            print('Hello World!')
        inputs_new = []
        for x in inputs:
            if isinstance(x, list):
                x = [InputsKernel.unwrap_storage_for_input(i) for i in x]
            else:
                x = InputsKernel.unwrap_storage_for_input(x)
            inputs_new.append(x)
        return inputs_new

    def is_extern(self):
        if False:
            for i in range(10):
                print('nop')
        return True

class NopKernel(InputsKernel):

    def is_no_op(self):
        if False:
            return 10
        return True

class ConcatKernel(NopKernel):
    """
    There isn't actually a real kernel for concat, we just change the
    storage for the upstream data.
    """

    @classmethod
    def create(cls, inputs, dim):
        if False:
            while True:
                i = 10
        device = inputs[0].get_device()
        dtype = inputs[0].get_dtype()
        new_size = list(inputs[0].get_size())
        offsets_start = [0]
        offsets_end = [new_size[dim]]
        assert 0 <= dim < len(new_size)
        for i in range(1, len(inputs)):
            input_size = inputs[i].get_size()
            offsets_start.append(new_size[dim])
            assert len(input_size) == len(new_size)
            assert inputs[i].get_dtype() == dtype
            assert inputs[i].get_device() == device
            for j in range(len(new_size)):
                if j == dim:
                    new_size[j] = new_size[j] + input_size[j]
                else:
                    new_size[j] = V.graph.sizevars.guard_equals(new_size[j], input_size[j])
            offsets_end.append(new_size[dim])
        output_stride = FlexibleLayout.contiguous_strides(new_size)
        for i in range(len(inputs)):
            x = inputs[i]
            if is_storage_and_layout(x):
                layout = x.get_layout()
                if isinstance(layout, FixedLayout) and layout.is_channels_last_contiguous():
                    output_stride = make_channels_last_strides_for(new_size)
                    break
        concat_kernel = ConcatKernel(name=None, layout=FixedLayout(device=device, dtype=dtype, size=new_size, stride=output_stride), inputs=[])
        kernel = StorageBox(concat_kernel)
        buffer_names = []
        for i in range(len(inputs)):
            input_buffer = cls.realize_into(inputs[i], SliceView.create(kernel, dim, offsets_start[i], offsets_end[i]))
            concat_kernel.inputs.append(input_buffer)
            if isinstance(inputs[i].data, BaseView):
                input_unwrapped = inputs[i].data.unwrap_view()
            else:
                input_unwrapped = inputs[i].data
            if input_unwrapped.is_input_buffer() and inputs[i].get_device().type == 'cuda' and (not is_dynamic(input_buffer)):
                buffer_names.append(input_buffer.get_name())
        if len(buffer_names) > 1:
            V.graph.register_list(buffer_names)
        concat_kernel.name = V.graph.register_buffer(concat_kernel)
        concat_kernel.inputs = cls.unwrap_storage(concat_kernel.inputs)
        return kernel

    @classmethod
    def can_realize_into_without_copy(cls, src):
        if False:
            return 10
        if isinstance(src, TensorBox):
            return cls.can_realize_into_without_copy(src.data)
        return isinstance(src.data.layout, FlexibleLayout) and (not isinstance(src.data, ExternKernelAlloc))

    @classmethod
    def realize_into(cls, src, dst):
        if False:
            i = 10
            return i + 15
        if not isinstance(dst, ReinterpretView):
            if is_storage_and_layout(dst):
                (storage, layout) = as_storage_and_layout(dst)
                dst = ReinterpretView(storage, layout)
        assert isinstance(dst, ReinterpretView), dst
        if isinstance(src, TensorBox):
            return cls.realize_into(src.data, dst)
        if isinstance(src, StorageBox):
            src.realize()
            assert hasattr(src.data, 'layout')
            if cls.can_realize_into_without_copy(src):
                src.data.layout = AliasedLayout(dst)
                return src.data
        pw = Pointwise.create(device=src.get_device(), dtype=src.get_dtype(), inner_fn=src.make_loader(), ranges=[V.graph.sizevars.guard_equals(a, b) for (a, b) in zip(src.get_size(), dst.get_size())])
        return cls.realize_into(pw, dst)

    def should_allocate(self):
        if False:
            return 10
        return True

@dataclasses.dataclass
class ExternKernel(InputsKernel):
    constant_args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    output_view: Optional[ReinterpretView] = None
    ordered_kwargs_for_cpp_kernel: Iterable[str] = dataclasses.field(default_factory=list)

    def decide_layout(self):
        if False:
            while True:
                i = 10
        if isinstance(self.layout, FlexibleLayout):
            self.apply_constraint()
            self.freeze_layout()

    def codegen_comment(self, wrapper):
        if False:
            for i in range(10):
                print('nop')
        (origin_str, detailed_origin_str) = get_kernel_metadata(self, wrapper)
        if origin_str:
            wrapper.writeline(origin_str)

    def codegen(self, wrapper):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @staticmethod
    def copy_input(x):
        if False:
            i = 10
            return i + 15
        pw = Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=x.make_loader(), ranges=x.get_size(), origin_node=x.get_origin_node(), traceback=x.get_traceback())
        pw.realize()
        return pw

    @classmethod
    def process_kernel(cls, kernel, *args, **kwargs):
        if False:
            print('Hello World!')
        binded_args = signature(kernel).bind(*args, **kwargs).arguments
        (args_flat, args_spec) = pytree.tree_flatten(binded_args)
        is_arg_tensor = []
        tensor_args = []
        non_tensor_args = []
        for arg in args_flat:
            is_arg_tensor.append(isinstance(arg, IRNode))
            if is_arg_tensor[-1]:
                tensor_args.append(arg)
            else:
                if isinstance(arg, sympy.Expr):
                    arg = V.graph.sizevars.shape_env.create_symintnode(arg, hint=None)
                non_tensor_args.append(arg)

        def unflatten_args(new_tensor_args, new_non_tensor_args):
            if False:
                print('Hello World!')
            result = []
            it_tensors = iter(new_tensor_args)
            it_non_tensors = iter(new_non_tensor_args)
            for is_tensor in is_arg_tensor:
                if is_tensor:
                    result.append(next(it_tensors))
                else:
                    result.append(next(it_non_tensors))
            r = pytree.tree_unflatten(result, args_spec)
            return (r.get('args', []), r.get('kwargs', {}))
        tensor_args = [cls.realize_input(x) for x in tensor_args]
        for x in tensor_args:
            if is_storage_and_layout(x):
                as_storage_and_layout(x, freeze=True)
        example_args = []
        for x in tensor_args:
            if x.get_name() in V.graph.constants:
                example_args.append(V.graph.constants[x.get_name()])
            else:
                example_args.append(ir_node_to_tensor(x, guard_shape=True))
        (new_args, new_kwargs) = unflatten_args(example_args, non_tensor_args)
        example_output = kernel(*new_args, **new_kwargs)
        if maybe_free_unbacked_symbols(example_output):
            example_output = V.graph.current_node.meta['val']
        return (example_output, tensor_args, non_tensor_args, unflatten_args)

    @classmethod
    def convert_to_reinterpret_view(cls, x):
        if False:
            return 10
        '\n        In order to pass this to an extern kernel we need a\n        ReinterpretView not a View.  This allows us to avoid some\n        unneeded copies.\n        '
        assert isinstance(x, BaseView)
        if isinstance(x, ReinterpretView):
            return x
        x.unwrap_view().freeze_layout()
        (index_args, var_ranges) = dependencies.index_vars_squeeze(x.get_size(), prefix='r')
        range_vars = index_args[0]
        index = x.make_indexer()(range_vars)
        index = V.graph.sizevars.simplify_with_ranges(index, var_ranges)
        strides = V.graph.sizevars.stride_vars(index, range_vars)
        offset = V.graph.sizevars.offset_var(index, range_vars)
        expected = sympy_dot(range_vars, strides) + offset
        if index != expected:
            log.debug('convert_to_reinterpret_view failed: stride=%s offset=%s index=%s', strides, offset, index)
            raise NotImplementedError()
        return ReinterpretView(data=x.data, layout=FixedLayout(device=x.get_device(), dtype=x.get_dtype(), size=x.get_size(), stride=strides, offset=offset))

    @classmethod
    def realize_input(cls, x):
        if False:
            while True:
                i = 10
        if x is None:
            return NoneAsConstantBuffer()
        if isinstance(x, (sympy.Expr, sympy.logic.boolalg.Boolean, int)):
            return ShapeAsConstantBuffer(x)
        if isinstance(x, Constant):
            return V.graph.add_tensor_constant(torch.tensor(x.value, dtype=x.get_dtype(), device=x.get_device()))
        if isinstance(x, ConstantBuffer):
            return x
        if isinstance(x, TensorBox):
            return cls.realize_input(x.data)
        if isinstance(x, ReinterpretView):
            return x
        if isinstance(x, BaseView):
            x.realize()
            if is_storage_and_layout(x.unwrap_view()):
                try:
                    return cls.convert_to_reinterpret_view(x)
                except NotImplementedError:
                    pass
        if isinstance(x, StorageBox):
            x.realize()
            return x
        return cls.copy_input(x)

    @classmethod
    def require_stride1(cls, x):
        if False:
            for i in range(10):
                print('nop')
        if is_storage_and_layout(x):
            if len(x.get_stride()) == 0:
                return x
            for stride in x.get_stride():
                if stride == 1:
                    return x
        return cls.copy_input(x)

    @classmethod
    def require_stride_order(cls, x, order):
        if False:
            i = 10
            return i + 15
        if x.get_numel() == 0:
            return x
        if is_storage_and_layout(x):
            while isinstance(x.get_layout(), AliasedLayout):
                x = x.get_layout().view
            if isinstance(x.get_layout(), FlexibleLayout):
                as_storage_and_layout(x, freeze=True, want_contiguous=False, stride_order=order)
                return x
            elif isinstance(x.get_layout(), FixedLayout) and x.get_layout().is_stride_ordered(order):
                return x
            elif isinstance(x.get_layout(), MutationLayout):
                if isinstance(x.get_layout().real_layout(), FlexibleLayout):
                    raise AssertionError("the MutationLayout's real layout shouldn't be FlexibleLayout")
                elif isinstance(x.get_layout().real_layout(), FixedLayout) and x.get_layout().real_layout().is_stride_ordered(order):
                    return x
        if isinstance(x, InputBuffer) and x.get_layout().is_stride_ordered(order):
            return x
        if isinstance(x, TensorBox) and isinstance(x.data, BaseView) and (not isinstance(x.data, ReinterpretView)) and is_storage_and_layout(x.unwrap_view()) and (not isinstance(x.unwrap_view().data, ExternKernelAlloc)):
            try:
                x.data = cls.convert_to_reinterpret_view(x.data)
                return cls.require_stride_order(x, order)
            except NotImplementedError:
                pass
        x = cls.copy_input(x)
        as_storage_and_layout(x, freeze=True, want_contiguous=False, stride_order=order)
        assert is_stride_order_storage_and_layout(x, order)
        return x

    @classmethod
    def require_channels_last(cls, x):
        if False:
            return 10
        return cls.require_stride_order(x, NHWC_STRIDE_ORDER)

    @classmethod
    def require_contiguous(cls, x):
        if False:
            print('Hello World!')
        return cls.require_stride_order(x, list(reversed(range(len(x.get_size())))))

    def apply_constraint(self):
        if False:
            return 10
        pass

    def codegen_const_args(self):
        if False:
            for i in range(10):
                print('nop')
        return map(V.graph.wrapper_code.val_to_arg_str, self.constant_args)

    def codegen_args(self):
        if False:
            for i in range(10):
                print('nop')
        args = []
        for x in self.inputs:
            if isinstance(x, list):
                names = [i.codegen_reference() for i in x]
                codegen_reference = f"[{', '.join(names)}]"
                args.append(codegen_reference)
            else:
                args.append(x.codegen_reference())
        args.extend(self.codegen_const_args())
        return args

    def get_kwargs_value(self, arg_name):
        if False:
            return 10
        if arg_name in self.kwargs:
            return self.kwargs.get(arg_name)
        if hasattr(self, 'kwargs_default_value') and arg_name in self.kwargs_default_value:
            return self.kwargs_default_value.get(arg_name).get('value')
        raise AssertionError(f'arg {arg_name} not found in self.kwargs or self.kwargs_default_value')

    def codegen_kwargs(self):
        if False:
            print('Hello World!')
        if not self.kwargs:
            return []
        if V.graph.cpp_wrapper:
            if self.kwargs and (not self.ordered_kwargs_for_cpp_kernel):
                raise AssertionError('ordered_kwargs_for_cpp_kernel is missing')
            kwargs = []
            for arg_name in self.ordered_kwargs_for_cpp_kernel:
                v = self.get_kwargs_value(arg_name)
                if isinstance(v, sympy.Expr):
                    kwargs.append(v)
                else:
                    kwargs.append(V.graph.wrapper_code.val_to_arg_str(v))
        else:
            kwargs = [f'{k}={V.graph.wrapper_code.val_to_arg_str(v)}' for (k, v) in self.kwargs.items()]
        return kwargs

    def codegen_size_asserts(self, wrapper):
        if False:
            while True:
                i = 10
        if config.size_asserts and (not V.graph.cpp_wrapper):
            size = V.graph.wrapper_code.codegen_shape_tuple(self.get_size())
            stride = V.graph.wrapper_code.codegen_shape_tuple(self.get_stride())
            wrapper.writeline(f'assert_size_stride({self.get_name()}, {size}, {stride})')

    def get_group_stride(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        get output sizes and strides, for template_codegen\n        '
        _size = self.get_size()
        _stride = self.get_stride()
        return ([_size, []], _stride)

    def canonicalize(self):
        if False:
            while True:
                i = 10
        '\n        Manually get canonicalization of the output index\n        '
        sizevars = V.graph.sizevars
        sizes = self.get_size()
        strides = self.get_stride()
        strides = [sizevars.size_hint(x) for x in strides]
        index_vars = [sympy_symbol(f'd{i}') for i in range(len(sizes))]
        index_order = sorted(range(len(strides)), key=strides.__getitem__, reverse=True)
        lookup = {pos: idx for (idx, pos) in enumerate(index_order)}
        order = [lookup[i] for i in range(len(lookup))]
        index_vars = [index_vars[i] for i in order]
        indexer = self.make_indexer()
        index = indexer(index_vars)
        (new_sizes, reindex, prune) = V.graph.sizevars._simplify_loops(index_vars, sizes, [index])
        (_, add_var) = var_builder('c')
        replacement = dict(zip(index_vars, reindex([add_var(x) for x in new_sizes])))
        index = sympy_subs(sympy.expand(index), replacement)
        return (index, tuple(new_sizes))

    def get_unbacked_symbol_uses(self):
        if False:
            return 10
        r = set()
        for arg in self.constant_args:
            r |= maybe_free_unbacked_symbols(arg)
        for arg in self.kwargs.values():
            r |= maybe_free_unbacked_symbols(arg)
        return r

    def __str__(self):
        if False:
            i = 10
            return i + 15
        kernel_name = getattr(self, 'kernel', None)
        lines = [f'kernel={kernel_name!r}']
        lines += [f'{field.name}={getattr(self, field.name)}' for field in dataclasses.fields(self)]
        lines.append(f'origin_node={self.origin_node!r}')
        return self.str_helper(lines)
    __repr__ = __str__

@dataclasses.dataclass
class ExternKernelOut(ExternKernel):
    output_view: Optional[ReinterpretView] = None

    def codegen(self, wrapper):
        if False:
            while True:
                i = 10
        self.codegen_comment(wrapper)
        args = [*self.codegen_args(), *self.codegen_kwargs()]
        wrapper.generate_extern_kernel_out(self.output_view, self.codegen_reference(), args, self.kernel)

    def __init__(self, layout, inputs, constant_args=(), kwargs=None, output_view=None, kernel=None, cpp_kernel=None, ordered_kwargs_for_cpp_kernel=()):
        if False:
            return 10
        super().__init__(None, layout, self.unwrap_storage(inputs), constant_args, kwargs or {})
        self.output_view = output_view
        self.name = V.graph.register_buffer(self)
        self.kernel = cpp_kernel if V.graph.cpp_wrapper else kernel
        self.ordered_kwargs_for_cpp_kernel = ordered_kwargs_for_cpp_kernel

    def should_allocate(self):
        if False:
            for i in range(10):
                print('nop')
        return True

class RandomSeeds(ExternKernelOut):

    def __init__(self, count: int, device: torch.device):
        if False:
            for i in range(10):
                print('nop')
        limits = torch.iinfo(torch.int64)
        super().__init__(layout=FixedLayout(device=device, dtype=torch.int64, size=[count]), inputs=[], constant_args=[limits.min, limits.max, [count]], kernel='aten.randint.low_out', cpp_kernel='at::randint_out')

class ExternKernelAlloc(ExternKernel):

    def codegen(self, wrapper):
        if False:
            while True:
                i = 10
        self.codegen_comment(wrapper)
        args = [*self.codegen_args(), *self.codegen_kwargs()]
        V.graph.wrapper_code.generate_extern_kernel_alloc(self, args)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    def __init__(self, layout, inputs, constant_args=(), kwargs=None, kernel=None, cpp_kernel=None, ordered_kwargs_for_cpp_kernel=()):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(None, layout, self.unwrap_storage(inputs), constant_args, kwargs or {})
        self.name = V.graph.register_buffer(self)
        self.kernel = cpp_kernel if V.graph.cpp_wrapper else kernel
        self.ordered_kwargs_for_cpp_kernel = ordered_kwargs_for_cpp_kernel

    def should_allocate(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def apply_constraint(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

class UserDefinedTritonKernel(ExternKernel):

    def get_kernel_and_configs(self):
        if False:
            print('Hello World!')
        from triton.runtime.autotuner import Autotuner
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        kernel = kernel_side_table.get_kernel(self.kernel_idx)
        configs = []
        if isinstance(kernel, Autotuner):
            configs = kernel.configs
            kernel = kernel.fn
        return (kernel, configs)

    def codegen(self, wrapper):
        if False:
            i = 10
            return i + 15
        (kernel, configs) = self.get_kernel_and_configs()
        new_name = wrapper.define_user_defined_triton_kernel(kernel, configs, self.kwargs)
        self.codegen_comment(wrapper)
        wrapper.generate_user_defined_triton_kernel(new_name, self.grid, configs, self.codegen_kwargs())

    def should_allocate(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def has_side_effects(self):
        if False:
            while True:
                i = 10
        return True

    def get_unbacked_symbol_defs(self):
        if False:
            while True:
                i = 10
        return {}

    def get_mutation_names(self):
        if False:
            i = 10
            return i + 15
        return []

    def __init__(self, *, kernel_idx, grid, kernel_args):
        if False:
            for i in range(10):
                print('nop')
        inputs = []
        kwargs = dict()
        constant_args = []
        for (k, v) in kernel_args.items():
            if isinstance(v, TensorBox):
                t = InputsKernel.unwrap_storage_for_input(self.realize_input(v))
                inputs.append(t)
                kwargs[k] = t
            else:
                constant_args.append(v)
                kwargs[k] = v
        assert len(inputs) != 0
        device = inputs[0].get_device()
        super().__init__(None, NoneLayout(device), inputs, tuple(constant_args), kwargs)
        self.name = V.graph.register_buffer(self)
        self.kernel_idx = kernel_idx
        self.grid = grid
        (kernel, _) = self.get_kernel_and_configs()
        self.ordered_kwargs_for_cpp_kernel = [arg for arg in kernel.arg_names if arg in kernel_args]
        mark_node_as_mutating(self, *[a for a in kernel_args.values() if isinstance(a, TensorBox)])

    def get_alias_names(self):
        if False:
            return 10
        return [i.get_name() for i in self.inputs]

def mark_node_as_mutating(cur_buffer, *mutated_ops):
    if False:
        print('Hello World!')
    '\n    Allows ops in mutated_ops to be marked as being mutated as well as\n    indicates to the scheduler that these ops depend on cur_buffer.\n    '
    for op in mutated_ops:
        assert isinstance(op, TensorBox)
        V.graph.mark_buffer_mutated(op.get_name())
        MutationOutput(op.layout, op, cur_buffer)

class MutationOutput(ExternKernel):

    def get_mutation_names(self):
        if False:
            print('Hello World!')
        return [self.inputs[0].get_name()]

    def __init__(self, layout, input, parent):
        if False:
            while True:
                i = 10
        super().__init__(None, layout, [input, parent], ())
        self.name = V.graph.register_buffer(self)

    def should_allocate(self):
        if False:
            return 10
        return False

    def is_no_op(self):
        if False:
            print('Hello World!')
        return True

    def has_side_effects(self):
        if False:
            i = 10
            return i + 15
        return True

    def get_alias_names(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.inputs[0].get_name()]

class InplaceBernoulliFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly
    """
    kernel = 'aten.bernoulli_'

    def codegen(self, wrapper):
        if False:
            for i in range(10):
                print('nop')
        (x,) = (t.codegen_reference() for t in self.inputs)
        wrapper.writeline(f"{self.kernel}({x}, {', '.join(map(repr, self.constant_args))})")

    def should_allocate(self):
        if False:
            i = 10
            return i + 15
        return False

    def get_mutation_names(self):
        if False:
            print('Hello World!')
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self):
        if False:
            while True:
                i = 10
        return {}

    def __init__(self, x, *constant_args):
        if False:
            return 10
        super().__init__(None, NoneLayout(x.get_device()), self.unwrap_storage([x]), constant_args)
        self.name = V.graph.register_buffer(self)
        mark_node_as_mutating(self, x)

class AccumulateGrad(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly
    """
    kernel = 'inductor_ops.accumulate_grad_'

    def codegen(self, wrapper):
        if False:
            print('Hello World!')
        (variable, new_grad) = (t.codegen_reference() for t in self.inputs)
        wrapper.writeline(f'{self.kernel}({variable}, {new_grad})')

    def should_allocate(self):
        if False:
            return 10
        return False

    def get_mutation_names(self):
        if False:
            while True:
                i = 10
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self):
        if False:
            return 10
        return {}

    def __init__(self, variable, new_grad):
        if False:
            i = 10
            return i + 15
        super().__init__(None, NoneLayout(variable.get_device()), self.unwrap_storage([variable, new_grad]))
        self.name = V.graph.register_buffer(self)
        mark_node_as_mutating(self, variable)

class ScatterFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly.
    This class handles both aten.scatter_ and aten.scatter_reduce_.
    It also handle the case `src` being a scalar properly.
    """

    def codegen(self, wrapper):
        if False:
            i = 10
            return i + 15
        if self.src_is_tensor:
            (x, index, src) = (t.codegen_reference() for t in self.inputs)
        else:
            (x, index) = (t.codegen_reference() for t in self.inputs)
            src = self.constant_args[1]
        wrapper.generate_scatter_fallback(x, [x, self.constant_args[0], index, src], self.kernel, self.fn, self.src_is_tensor, self.kwargs['reduce'], self.codegen_kwargs())

    def should_allocate(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def get_cpp_kernel(self, fn, reduce):
        if False:
            while True:
                i = 10
        if fn == 'aten.scatter_':
            if self.src_is_tensor:
                kernel = 'at::scatter_out' if reduce is None else 'at::scatter_reduce_out'
            else:
                assert reduce is None, 'Expect reduce to be None for aten.scatter_ with scalar src'
                kernel = 'at::scatter_out'
        else:
            assert reduce is not None, 'Expect reduce to be not None for aten.scatter_reduce_'
            kernel = 'at::scatter_reduce_out'
        return kernel

    def get_mutation_names(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self):
        if False:
            print('Hello World!')
        return {}

    def __init__(self, fn, x, dim: int, index, src, *, reduce: Optional[str]=None, include_self: bool=True):
        if False:
            for i in range(10):
                print('nop')
        assert fn in {'aten.scatter_', 'aten.scatter_reduce_'}
        self.src_is_tensor = isinstance(src, TensorBox)
        if V.graph.cpp_wrapper:
            get_operator_enum = {'add': 'sum', 'multiply': 'prod'}
            if reduce in get_operator_enum:
                reduce = get_operator_enum[reduce]
            self.kernel = self.get_cpp_kernel(fn, reduce)
        else:
            self.kernel = fn
        self.fn = fn
        constant_args: Tuple[Any, ...]
        if self.src_is_tensor:
            tensors = [self.realize_input(t) for t in [x, index, src]]
            constant_args = (dim,)
        else:
            tensors = [self.realize_input(t) for t in [x, index]]
            constant_args = (dim, src)
        super().__init__(None, NoneLayout(x.get_device()), self.unwrap_storage(tensors), constant_args, {'reduce': reduce, 'include_self': include_self})
        self.ordered_kwargs_for_cpp_kernel = ['reduce', 'include_self']
        self.name = V.graph.register_buffer(self)
        mark_node_as_mutating(self, x)

class IndexPutFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation and indices properly
    """

    def codegen(self, wrapper):
        if False:
            return 10
        (x, values, *valid_indices) = (t.codegen_reference() for t in self.inputs)
        indices = []
        iter_valid_indices = iter(valid_indices)
        for (i, _) in enumerate(self.indices):
            if self.indices[i] is not None:
                indices.append(next(iter_valid_indices))
            else:
                indices.append(V.graph.wrapper_code.none_str)
        indices_str = f"{V.graph.wrapper_code.open_bracket}{', '.join(indices)}{V.graph.wrapper_code.closed_bracket}"
        args = [x, indices_str, values, *self.codegen_const_args()]
        wrapper.writeline(wrapper.wrap_kernel_call(self.kernel, args))

    def should_allocate(self):
        if False:
            return 10
        return False

    def get_mutation_names(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self):
        if False:
            for i in range(10):
                print('nop')
        return {}

    def __init__(self, x, indices, values, accumulate):
        if False:
            for i in range(10):
                print('nop')
        self.indices = indices
        valid_indices = [i for i in indices if i is not None]
        tensors = [self.realize_input(x) for x in [x, values, *valid_indices]]
        super().__init__(None, NoneLayout(x.get_device()), self.unwrap_storage(tensors), (accumulate,))
        self.name = V.graph.register_buffer(self)
        self.kernel = 'at::index_put_' if V.graph.cpp_wrapper else 'aten.index_put_'
        mark_node_as_mutating(self, x)

class DeviceCopy(ExternKernelOut):

    @classmethod
    def create(cls, x, device):
        if False:
            i = 10
            return i + 15
        if not x.is_extern() and all((r.name in V.graph.constants and isinstance(r, dependencies.MemoryDep) for r in x.get_reads())):
            return x.constant_to_device(device)
        V.graph.device_types.add(device.type)
        V.graph.add_device_idx(device.index)
        V.graph.device_types.add(x.get_device().type)
        V.graph.add_device_idx(x.get_device().index)
        developer_warning('DeviceCopy in input program')
        return DeviceCopy(FlexibleLayout(device=device, dtype=x.get_dtype(), size=x.get_size()), [cls.realize_input(x)])

    def codegen(self, wrapper):
        if False:
            i = 10
            return i + 15
        args = self.codegen_args()
        assert len(args) == 1
        if self.output_view:
            wrapper.codegen_device_copy(args[0], self.output_view.codegen_reference())
        else:
            wrapper.codegen_device_copy(args[0], self.codegen_reference())

class DynamicScalar(ExternKernel):
    """
    The result of a call to aten._local_scalar_dense.
    """

    def get_reads(self):
        if False:
            while True:
                i = 10
        return ()

    def should_allocate(self):
        if False:
            i = 10
            return i + 15
        return False

    def __init__(self, sym, data):
        if False:
            print('Hello World!')
        super().__init__(None, NoneLayout(torch.device('cpu')), [data])
        self.sym = sym

    def get_unbacked_symbol_defs(self):
        if False:
            for i in range(10):
                print('nop')
        return {self.sym}

    def codegen(self, wrapper):
        if False:
            return 10
        (data,) = (t.codegen_reference() for t in self.inputs)
        wrapper.writeline(f'{self.sym} = {data}.item()')
        wrapper.writeline(f'{self.get_name()} = None')

@dataclasses.dataclass
class ExternKernelNode:
    name: str
    node: export_schema.Node
fbcode_use_proxy_executor = {torch.ops.aten._scaled_dot_product_efficient_attention.default}

@dataclasses.dataclass
class FallbackKernel(ExternKernelAlloc):

    def __init__(self, layout, kernel, tensor_args, nontensor_args, unflatten_args, kwargs=None):
        if False:
            return 10
        super().__init__(layout, tuple(tensor_args), tuple(nontensor_args))
        self.outputs: Sequence[Any] = []
        self.use_cpp_op_schema = False
        self.op_overload = kernel
        assert isinstance(kernel, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)), f'Fails to create FallbackKernel for {kernel}: {type(kernel)} not supported'
        if kernel.namespace == 'aten':
            assert isinstance(kernel, torch._ops.OpOverload)
            op_base_name = kernel.__name__.split('.')[0]
            if V.graph.cpp_wrapper:
                if config.is_fbcode() and kernel in fbcode_use_proxy_executor:
                    self.use_cpp_op_schema = True
                    self.set_cpp_kernel(kernel)
                else:
                    self.kernel = f'at::{op_base_name}' if kernel._overloadname == 'default' else f"at::_ops::{kernel.__name__.replace('.', '_')}::call"
                    schema = kernel._schema
                    self.args_default_value = [{'type': x.real_type, 'value': x.default_value} for x in schema.arguments if not x.kwarg_only]
                    self.ordered_kwargs_for_cpp_kernel = [x.name for x in schema.arguments if x.kwarg_only]
                    self.kwargs_default_value = {x.name: {'type': x.real_type, 'value': x.default_value} for x in schema.arguments if x.kwarg_only}
            else:
                self.kernel = f'aten.{op_base_name}'
        elif isinstance(kernel, torch._ops.HigherOrderOperator):
            if getattr(torch._prims.rng_prims, kernel.__name__, None) is kernel:
                self.kernel = f'torch._prims.rng_prims.{kernel.__name__}'
            else:
                raise NotImplementedError('Unable to find HigherOrderOperator kernel name')
        elif V.graph.cpp_wrapper:
            self.use_cpp_op_schema = True
            self.set_cpp_kernel(kernel)
        else:
            self.kernel = f"{kernel.__module__.replace('._ops.', '.ops.')}.{kernel.__name__}"
        self.unflatten_args = unflatten_args
        self.kwargs = {} if kwargs is None else kwargs
        V.graph.warn_fallback(self.kernel)

    def set_cpp_kernel(self, kernel):
        if False:
            return 10
        from .codegen.wrapper import get_cpp_op_schema
        assert not kernel._schema.is_mutable, f'mutable {kernel.__name__} is not supported with cpp_wrapper'

        def is_not_write(arg):
            if False:
                while True:
                    i = 10
            return arg.alias_info is None or not arg.alias_info.is_write
        assert all((is_not_write(x) for x in kernel._schema.arguments)), f'{kernel.__name__} with alias_info arguments is not supported with cpp_wrapper'
        assert all((is_not_write(x) for x in kernel._schema.returns)), f'{kernel.__name__} with alias_info returns is not supported with cpp_wrapper'
        self.kernel = kernel._schema.name
        self.cpp_kernel_overlad_name = kernel._schema.overload_name
        self.cpp_kernel_key = f"{self.kernel.replace('::', '_')}_{self.cpp_kernel_overlad_name}"
        self.cpp_op_schema = get_cpp_op_schema(kernel)
        self.ordered_kwargs_for_cpp_kernel = [x.name for x in kernel._schema.arguments if x.kwarg_only]

    def get_arg_default_value(self, pos):
        if False:
            return 10
        assert hasattr(self, 'args_default_value'), 'self.args_default_value has to be provided'
        assert pos < len(self.args_default_value), f'expected the index {pos} to be smaller than len(self.args_default_value): {len(self.args_default_value)}'
        return self.args_default_value[pos]['value']

    def codegen_args(self):
        if False:
            return 10

        @dataclasses.dataclass
        class Shim:
            ref: Any

            def __repr__(self):
                if False:
                    i = 10
                    return i + 15
                return self.ref
        tensor_args = [Shim(x.codegen_reference()) for x in self.inputs]
        (args, kwargs) = self.unflatten_args(tensor_args, self.constant_args)
        args = [V.graph.wrapper_code.val_to_arg_str(x) for x in args]
        if V.graph.cpp_wrapper and hasattr(self, 'args_default_value'):
            n_args = len(args)
            n_pos_args = len(self.args_default_value)
            if n_args < n_pos_args:
                pos_args = [self.get_arg_default_value(i) for i in range(n_args, n_pos_args)]
                pos_args = [V.graph.wrapper_code.val_to_arg_str(x) for x in pos_args]
                args.extend(pos_args)
        self.kwargs.update(kwargs)
        return args

    @staticmethod
    def find_device(tensor_args, example_output):
        if False:
            i = 10
            return i + 15
        if tensor_args:
            return tensor_args[0].get_device()
        if isinstance(example_output, torch.Tensor):
            return example_output.device
        if isinstance(example_output, (list, tuple)):
            devices = {FallbackKernel.find_device(None, x) for x in example_output}
            devices = [device for device in devices if device]
            if len(devices) == 1:
                return devices[0]
            for device in devices:
                if device.type == 'cuda':
                    return device
            return devices[0]
        return None

    def has_side_effects(self):
        if False:
            print('Hello World!')
        if not isinstance(self.op_overload, torch._ops.OpOverload):
            return False
        return get_schema_info(self.op_overload).is_mutable()

    def get_alias_names(self):
        if False:
            return 10
        if not isinstance(self.op_overload, torch._ops.OpOverload):
            return []
        if torch._inductor.utils.is_view(self.op_overload):
            return [inp.get_name() for inp in self.inputs]
        return []

    def export_extern_kernel_node(self):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(self, FallbackKernel)
        (args, kwargs) = self.unflatten_args(self.inputs, self.constant_args)
        ordered_kwargs = [kwargs.get(key, None) for key in self.ordered_kwargs_for_cpp_kernel]
        serializer = GraphModuleSerializer(None, None)
        named_arguments = serializer.serialize_inputs(self.op_overload, args, kwargs)

        def handle_single_output(return_type, output):
            if False:
                i = 10
                return i + 15
            if isinstance(return_type, torch.TensorType):
                out = output
                if isinstance(output, (list, tuple)):
                    assert len(output) == 1
                    out = output[0]
                return export_schema.Argument.create(as_tensor=export_schema.TensorArgument(name=out.get_name()))
            elif isinstance(return_type, torch.ListType) and isinstance(return_type.getElementType(), torch.TensorType):
                return export_schema.Argument.create(as_tensors=[export_schema.TensorArgument(name=out.get_name()) for out in output])
            else:
                raise RuntimeError(f'Unsupported return type {type(return_type)}')
        target = self.op_overload
        returns = target._schema.returns
        if len(returns) == 1:
            return_type = returns[0].real_type
            output_arguments = [handle_single_output(return_type, self.outputs)]
        else:
            assert isinstance(self.outputs, tuple)
            assert len(returns) == len(self.outputs)
            output_arguments = [handle_single_output(return_schema.real_type, output) for (return_schema, output) in zip(returns, self.outputs)]
        node = ExternKernelNode(name=self.get_name(), node=export_schema.Node(target=self.kernel, inputs=named_arguments, outputs=output_arguments, metadata={}))
        V.graph.extern_kernel_nodes.append(node)
        return [*args, *ordered_kwargs]

    def codegen(self, wrapper):
        if False:
            print('Hello World!')
        if self.use_cpp_op_schema:
            self.codegen_comment(wrapper)
            exported_args = None
            args = None
            if config.is_fbcode() and V.graph.cpp_wrapper:
                exported_args = self.export_extern_kernel_node()
            else:
                args = [*self.codegen_args(), *self.codegen_kwargs()]
            wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(self.get_name(), self.kernel, args, self.cpp_op_schema, self.cpp_kernel_key, self.cpp_kernel_overlad_name, self.op_overload, exported_args, self.outputs)
        else:
            super().codegen(wrapper)

    @staticmethod
    def tensor_to_layout(output: torch.Tensor):
        if False:
            return 10
        return FixedLayout(output.device, output.dtype, convert_shape_to_inductor(output.size()), convert_shape_to_inductor(output.stride()))

    @classmethod
    def create(cls, kernel, *args, **kwargs):
        if False:
            return 10
        fake_incorrect_kernels = (aten._fused_moving_avg_obs_fq_helper_functional,)
        context = V.graph.fake_mode if kernel not in fake_incorrect_kernels else nullcontext()
        with context:
            (example_output, tensor_args, non_tensor_args, unflatten_args) = cls.process_kernel(kernel, *args, **kwargs)
        device = cls.find_device(tensor_args, example_output)
        assert device, 'Not sure where to find device info'
        packed = cls(MultiOutputLayout(device), kernel, tensor_args, non_tensor_args, unflatten_args)

        def generate_output(output, indices):
            if False:
                while True:
                    i = 10
            if isinstance(output, (list, tuple)):
                return type(output)((generate_output(output[i], indices + [(type(output), i)]) for i in range(len(output))))
            elif isinstance(output, dict):
                return {key: generate_output(val, indices + [(type(output), key)]) for (key, val) in output.items()}
            elif isinstance(output, torch.Tensor):
                return MultiOutput(cls.tensor_to_layout(output), packed, indices)
            elif isinstance(output, int):
                return output
            elif isinstance(output, torch.SymInt):
                return output.node.expr
            else:
                assert output is None, f'FallbackKernel output type {type(output)} is not supported'
                return None
        outputs = generate_output(example_output, [])
        if isinstance(outputs, (list, tuple, dict)):
            packed.outputs = outputs
        else:
            packed.outputs = [outputs]
        return outputs

    def apply_constraint(self):
        if False:
            i = 10
            return i + 15
        return super().apply_constraint()

@dataclasses.dataclass
class ComplexView(ExternKernelAlloc):
    """View a complex number as two dtyped numbers or vice versa"""

    def should_allocate(self):
        if False:
            print('Hello World!')
        return False

    def get_alias_names(self):
        if False:
            while True:
                i = 10
        return [self.inputs[0].get_name()]

    def __init__(self, layout, kernel, tensor_args, nontensor_args):
        if False:
            while True:
                i = 10
        super().__init__(layout, tuple(tensor_args), tuple(nontensor_args))
        self.outputs: Sequence[Any] = []
        self.kernel = kernel

    @classmethod
    def create(cls, kernel, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        context = V.graph.fake_mode
        with context:
            (example_output, tensor_args, non_tensor_args, unflatten_args) = cls.process_kernel(kernel, *args, **kwargs)
        device = FallbackKernel.find_device(tensor_args, example_output)
        assert device, 'Not sure where to find device info'
        packed = ComplexView(MultiOutputLayout(device), kernel, tensor_args, non_tensor_args)
        layout = FixedLayout(example_output.device, example_output.dtype, convert_shape_to_inductor(example_output.size()), convert_shape_to_inductor(example_output.stride()))
        outputs = MultiOutput(layout, packed, [])
        packed.outputs = [outputs]
        return outputs

@dataclasses.dataclass
class MultiOutputLayout(IRNode):
    device: torch.device

class MultiOutput(ExternKernel):

    def codegen_list_tuple_access(self, basename, indices):
        if False:
            for i in range(10):
                print('nop')
        if len(indices) > 0:
            (itype, i) = indices[0]
            if itype == list:
                return self.codegen_list_tuple_access(f'{basename}[{i}]', indices[1:])
            elif itype == tuple:
                tuple_access = V.graph.wrapper_code.codegen_tuple_access(basename, self.get_name(), str(i))
                return self.codegen_list_tuple_access(tuple_access, indices[1:])
            elif itype == dict:
                return self.codegen_list_tuple_access(f"{basename}['{i}']", indices[1:])
            else:
                raise AssertionError('non supported index type')
        else:
            return basename

    def codegen(self, wrapper):
        if False:
            for i in range(10):
                print('nop')
        wrapper.codegen_multi_output(self.get_name(), self.codegen_list_tuple_access(self.inputs[0].get_name(), self.indices))
        self.codegen_unbacked_symbol_defs(wrapper)

    def __init__(self, layout, input, indices: List[Tuple[Any, ...]]):
        if False:
            while True:
                i = 10
        super().__init__(None, layout, [input], ())
        self.name = V.graph.register_buffer(self)
        self.indices = indices

    def get_unbacked_symbol_uses(self):
        if False:
            i = 10
            return i + 15
        return self.inputs[0].get_unbacked_symbol_uses()

    def should_allocate(self):
        if False:
            i = 10
            return i + 15
        return False

    def get_alias_names(self):
        if False:
            print('Hello World!')
        return [inp.get_name() for inp in self.inputs if isinstance(inp, (FallbackKernel, ComplexView)) and len(inp.get_alias_names()) > 0]

def _prepare_convolution_fusion_create(cls, x: 'TensorBox', weight: 'TensorBox', bias: 'TensorBox', padding: List[int], stride: List[int], dilation: List[int], groups: int, transposed: bool=False, output_padding: Optional[List[int]]=None):
    if False:
        print('Hello World!')
    "\n    This function is a helper function to prepare inputs, layout and constant args\n    for convolution post-op fusion's create function, including deciding the output\n    layout (channels first or channels last), realizing inputs and make them etc. The\n    function only supports the CPU device since conv post-op fusion kernel is only\n    supported on CPU right now.\n    "

    def _conv_input_size(output_size, weight_size, padding, output_padding, stride, dilation, groups):
        if False:
            for i in range(10):
                print('nop')
        assert len(output_size) == len(weight_size), 'Expect input dim == weight dim'
        dim = len(output_size)
        assert dim > 2, 'Expect input dim > 2'
        BATCH_DIM = 0
        WEIGHT_INPUT_CHANNELS_DIM = 1
        input_size = []
        input_size.append(output_size[BATCH_DIM])
        input_size.append(weight_size[WEIGHT_INPUT_CHANNELS_DIM] * groups)
        for d in range(2, dim):
            kernel = (weight_size[d] - 1) * dilation[d - 2] + 1
            input_size_d = (output_size[d] - 1) * stride[d - 2] - padding[d - 2] * 2 + kernel + output_padding[d - 2]
            input_size.append(input_size_d)
        return list(map(int, input_size))

    def _original_deconv_weight_size(prepacked_weight, groups):
        if False:
            print('Hello World!')
        prepacked_weight_size = prepacked_weight.size()
        dim = len(prepacked_weight_size)
        assert dim > 2, 'Expect weight dim > 2'
        if groups > 1:
            weight_size = []
            weight_size.append(prepacked_weight_size[1] * groups)
            weight_size.append(prepacked_weight_size[0] / groups)
            for d in range(2, dim):
                weight_size.append(prepacked_weight_size[d])
        else:
            weight_size = prepacked_weight.transpose(0, 1).size()
        return weight_size
    x.realize()
    weight.realize()
    if bias is not None:
        bias.realize()
    with V.graph.fake_mode:
        x_fake = ir_node_to_tensor(x, guard_shape=True)
        weight_fake = ir_node_to_tensor(weight, guard_shape=True)
        dims = len(x_fake.size()) - 2
        assert 0 < len(padding) <= dims
        assert 0 < len(dilation) <= dims
        assert 0 < len(stride) <= dims
        padding = pad_listlike(padding, dims)
        dilation = pad_listlike(dilation, dims)
        stride = pad_listlike(stride, dims)
        if output_padding is None:
            output_padding = pad_listlike([0], dims)
        else:
            assert 0 < len(output_padding) <= dims
            output_padding = pad_listlike(output_padding, dims)
        assert isinstance(groups, int)
        if transposed:
            weight_size = _original_deconv_weight_size(weight_fake, groups)
            input_size = x_fake.size()
            output_size = _conv_input_size(input_size, weight_size, padding, output_padding, stride, dilation, groups)
        else:
            bias_fake = ir_node_to_tensor(bias, guard_shape=True) if bias is not None else bias
            output = torch.ops.aten.convolution(x_fake, weight_fake, bias_fake, stride, padding, dilation, transposed, output_padding, groups)
            output_size = output.size()
        req_stride_order = [0] + list(reversed(range(1, len(stride) + 1)))
        req_stride_order = [len(req_stride_order)] + req_stride_order
        output_stride = make_channels_last_strides_for(output_size)
    x = cls.require_stride_order(x, req_stride_order)
    assert x.get_device().type == 'cpu' and weight.get_device().type == 'cpu'
    inputs = [x, weight]
    kernel_layout = FixedLayout(x.get_device(), x.get_dtype(), convert_shape_to_inductor(output_size), convert_shape_to_inductor(output_stride))
    constant_args = [padding, stride, dilation, groups]
    if transposed:
        constant_args.insert(1, output_padding)
    if bias is not None:
        inputs.append(bias)
    else:
        constant_args.insert(0, bias)
    return (inputs, constant_args, kernel_layout, req_stride_order)

def _prepare_linear_fusion_create(cls, x: 'TensorBox', weight: 'TensorBox', bias: 'TensorBox'):
    if False:
        i = 10
        return i + 15
    "\n    This function is a helper function to prepare inputs, layout and constant args\n    for linear post-op fusion's create function. The function only supports the CPU device\n    since linear post-op fusion kernel is only supported on CPU right now.\n    "
    x.realize()
    weight.realize()
    if bias is not None:
        bias.realize()
    with V.graph.fake_mode:
        x_fake = ir_node_to_tensor(x, guard_shape=True)
        weight_fake = ir_node_to_tensor(weight, guard_shape=True)
        bias_fake = ir_node_to_tensor(bias, guard_shape=True) if bias is not None else bias
        if bias is not None:
            output = torch.ops.aten.addmm.default(bias_fake, x_fake, weight_fake)
        else:
            output = torch.ops.aten.mm.default(x_fake, weight_fake)
        output_size = output.size()
        req_stride_order = [1, 0]
        output_stride = make_contiguous_strides_for(output_size)
    x = cls.require_stride_order(x, req_stride_order)
    assert x.get_device().type == 'cpu' and weight.get_device().type == 'cpu'
    inputs = [x, weight]
    kernel_layout = FixedLayout(x.get_device(), x.get_dtype(), convert_shape_to_inductor(output_size), convert_shape_to_inductor(output_stride))
    constant_args: List[Any] = []
    if bias is not None:
        inputs.append(bias)
    else:
        constant_args.insert(0, bias)
    return (inputs, constant_args, kernel_layout, req_stride_order)

class ConvolutionUnary(ExternKernelAlloc):

    def __init__(self, layout, inputs, constant_args=(), kernel='torch.ops.mkldnn._convolution_pointwise'):
        if False:
            return 10
        super().__init__(layout, inputs, constant_args, None, kernel='torch.ops.mkldnn._convolution_pointwise', cpp_kernel='mkldnn::_convolution_pointwise')
        self.cpp_kernel_key = 'convolution_pointwise'
        self.cpp_op_schema = '\n            at::Tensor(\n                const at::Tensor& input_t,\n                const at::Tensor& weight_t,\n                const c10::optional<at::Tensor>& bias_opt,\n                at::IntArrayRef padding,\n                at::IntArrayRef stride,\n                at::IntArrayRef dilation,\n                int64_t groups,\n                c10::string_view attr,\n                torch::List<c10::optional<at::Scalar>> scalars,\n                c10::optional<c10::string_view> algorithm)'

    def codegen(self, wrapper):
        if False:
            for i in range(10):
                print('nop')
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(self.get_name(), self.kernel, self.codegen_args(), self.cpp_op_schema, self.cpp_kernel_key)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(cls, x: 'TensorBox', weight: 'TensorBox', bias: 'TensorBox', padding_: List[int], stride_: List[int], dilation_: List[int], groups: int, attr, scalars: Optional[List[Any]], algorithm):
        if False:
            for i in range(10):
                print('nop')
        (inputs, constant_args, kernel_layout, _) = _prepare_convolution_fusion_create(cls, x, weight, bias, padding_, stride_, dilation_, groups)
        constant_args = constant_args + [attr, may_convert_to_optional(scalars), algorithm]
        return ConvolutionUnary(layout=kernel_layout, inputs=inputs, constant_args=constant_args)

class ConvolutionBinary(ExternKernelAlloc):

    def __init__(self, layout, inputs, constant_args=(), cpp_constant_args=()):
        if False:
            i = 10
            return i + 15
        super().__init__(layout, inputs, constant_args, None, kernel='torch.ops.mkldnn._convolution_pointwise.binary', cpp_kernel='mkldnn::_convolution_pointwise')
        self.cpp_kernel_overlad_name = 'binary'
        self.cpp_kernel_key = 'convolution_pointwise_binary'
        self.cpp_op_schema = '\n            at::Tensor(\n                const at::Tensor& input_t,\n                const at::Tensor& other_t,\n                const at::Tensor& weight_t,\n                const c10::optional<at::Tensor>& bias_opt,\n                at::IntArrayRef padding,\n                at::IntArrayRef stride,\n                at::IntArrayRef dilation,\n                int64_t groups,\n                c10::string_view binary_attr,\n                c10::optional<at::Scalar> alpha,\n                c10::optional<c10::string_view> unary_attr,\n                torch::List<c10::optional<at::Scalar>> unary_scalars,\n                c10::optional<c10::string_view> unary_algorithm)'
        self.cpp_constant_args = cpp_constant_args

    def codegen(self, wrapper):
        if False:
            return 10
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(self.get_name(), self.kernel, self.codegen_args(), self.cpp_op_schema, self.cpp_kernel_key, self.cpp_kernel_overlad_name)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(cls, x: 'TensorBox', other: 'TensorBox', weight: 'TensorBox', bias: 'TensorBox', padding_: List[int], stride_: List[int], dilation_: List[int], groups: int, binary_attr: str, binary_alpha: Optional[float], unary_attr: Optional[str], unary_scalars: Optional[List[Any]], unary_algorithm: Optional[str]):
        if False:
            i = 10
            return i + 15
        (inputs, constant_args, kernel_layout, req_stride_order) = _prepare_convolution_fusion_create(cls, x, weight, bias, padding_, stride_, dilation_, groups)
        other = cls.require_stride_order(other, req_stride_order)
        inputs.insert(1, other)
        constant_args = constant_args + [binary_attr, binary_alpha, unary_attr, may_convert_to_optional(unary_scalars), unary_algorithm]
        return ConvolutionBinary(layout=kernel_layout, inputs=inputs, constant_args=constant_args)

class ConvolutionBinaryInplace(ExternKernelAlloc):

    def __init__(self, kernel_layout, inputs, constant_args=()):
        if False:
            while True:
                i = 10
        reordered_inputs = [inputs[1], inputs[0]] + inputs[2:]
        super().__init__(kernel_layout, reordered_inputs, constant_args, None, kernel='torch.ops.mkldnn._convolution_pointwise_.binary', cpp_kernel='mkldnn::_convolution_pointwise_')
        self.cpp_kernel_overlad_name = 'binary'
        self.cpp_kernel_key = 'convolution_pointwise_binary_'
        self.cpp_op_schema = '\n            at::Tensor&(\n                at::Tensor& other_t,\n                const at::Tensor& input_t,\n                const at::Tensor& weight_t,\n                const c10::optional<at::Tensor>& bias_opt,\n                at::IntArrayRef padding,\n                at::IntArrayRef stride,\n                at::IntArrayRef dilation,\n                int64_t groups,\n                c10::string_view binary_attr,\n                c10::optional<at::Scalar> alpha,\n                c10::optional<c10::string_view> unary_attr,\n                torch::List<c10::optional<at::Scalar>> unary_scalars,\n                c10::optional<c10::string_view> unary_algorithm)'

    def codegen(self, wrapper):
        if False:
            i = 10
            return i + 15
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(self.get_name(), self.kernel, self.codegen_args(), self.cpp_op_schema, self.cpp_kernel_key, self.cpp_kernel_overlad_name)

    def get_mutation_names(self):
        if False:
            return 10
        return [self.inputs[1].get_name()]

    def get_unbacked_symbol_defs(self):
        if False:
            i = 10
            return i + 15
        return {}

    @classmethod
    def create(cls, x: 'TensorBox', other: 'TensorBox', weight: 'TensorBox', bias: 'TensorBox', padding_: List[int], stride_: List[int], dilation_: List[int], groups: int, binary_attr: str, binary_alpha: Optional[float], unary_attr: Optional[str], unary_scalars: Optional[List[Any]], unary_algorithm: Optional[str]):
        if False:
            while True:
                i = 10
        (inputs, constant_args, _, req_stride_order) = _prepare_convolution_fusion_create(cls, x, weight, bias, padding_, stride_, dilation_, groups)
        other = cls.require_stride_order(other, req_stride_order)
        inputs.insert(1, other)
        constant_args = constant_args + [binary_attr, binary_alpha, unary_attr, may_convert_to_optional(unary_scalars), unary_algorithm]
        packed = ConvolutionBinaryInplace(kernel_layout=NoneLayout(inputs[1].get_device()), inputs=inputs, constant_args=constant_args)
        mark_node_as_mutating(packed, inputs[1])
        return packed

class MKLPackedLinear(ExternKernelAlloc):

    def __init__(self, layout, inputs, constant_args=()):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(layout, inputs, constant_args, None, kernel='torch.ops.mkl._mkl_linear', cpp_kernel='mkl::_mkl_linear')
        self.cpp_kernel_key = 'mkl_linear'
        self.cpp_op_schema = '\n            at::Tensor(\n                const at::Tensor& self,\n                const at::Tensor& mkl_weight_t,\n                const at::Tensor& origin_weight_t,\n                const c10::optional<at::Tensor>& bias_opt,\n                const int64_t prepack_batch_size)'

    def codegen(self, wrapper):
        if False:
            while True:
                i = 10
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(self.get_name(), self.kernel, self.codegen_args(), self.cpp_op_schema, self.cpp_kernel_key)

    @classmethod
    def create(cls, x, packed_w, orig_w, batch_size):
        if False:
            print('Hello World!')
        x = cls.require_stride1(cls.realize_input(x))
        orig_w = cls.require_stride1(cls.realize_input(orig_w))
        (*m, _) = x.get_size()
        (oc, _) = orig_w.get_size()
        output_size = list(m) + [oc]
        output_stride = make_contiguous_strides_for(output_size)
        inputs = [x, packed_w, orig_w]
        constant_args = [None, batch_size]
        return MKLPackedLinear(layout=FixedLayout(x.get_device(), x.get_dtype(), output_size, output_stride), inputs=inputs, constant_args=constant_args)

class LinearUnary(ExternKernelAlloc):

    def __init__(self, layout, inputs, constant_args=()):
        if False:
            while True:
                i = 10
        super().__init__(layout, inputs, constant_args, None, kernel='torch.ops.mkldnn._linear_pointwise', cpp_kernel='mkldnn::_linear_pointwise')
        self.cpp_kernel_key = 'linear_pointwise'
        self.cpp_op_schema = '\n            at::Tensor(\n                const at::Tensor& input_t,\n                const at::Tensor& weight_t,\n                const c10::optional<at::Tensor>& bias_opt,\n                c10::string_view attr,\n                torch::List<c10::optional<at::Scalar>> scalars,\n                c10::optional<c10::string_view> algorithm)'

    def codegen(self, wrapper):
        if False:
            while True:
                i = 10
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(self.get_name(), self.kernel, self.codegen_args(), self.cpp_op_schema, self.cpp_kernel_key)

    @classmethod
    def create(cls, x, w, b, attr, scalars, algorithm):
        if False:
            while True:
                i = 10
        x = cls.require_contiguous(cls.realize_input(x))
        w = cls.require_contiguous(cls.realize_input(w))
        (*m, ic) = x.get_size()
        (oc, ic) = w.get_size()
        inputs = [x, w]
        constant_args = [attr, scalars if scalars else [-1], algorithm]
        if b is not None:
            b = cls.require_contiguous(cls.realize_input(b))
            inputs.append(b)
        else:
            constant_args.insert(0, None)
        return LinearUnary(layout=FlexibleLayout(device=x.get_device(), dtype=x.get_dtype(), size=list(m) + [oc]), inputs=inputs, constant_args=constant_args)

    def apply_constraint(self):
        if False:
            return 10
        pass

class LinearBinary(ExternKernelAlloc):
    kernel = 'torch.ops.mkldnn._linear_pointwise.binary'

    def __init__(self, layout, inputs, constant_args=()):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(layout, inputs, constant_args, None, kernel='torch.ops.mkldnn._linear_pointwise.binary', cpp_kernel='mkldnn::_linear_pointwise')
        self.cpp_kernel_overlad_name = 'binary'
        self.cpp_kernel_key = 'linear_pointwise_binary'
        self.cpp_op_schema = '\n            at::Tensor(\n                const at::Tensor& input_t,\n                const at::Tensor& other_t,\n                const at::Tensor& weight_t,\n                const c10::optional<at::Tensor>& bias_opt,\n                c10::string_view attr)\n        '

    def codegen(self, wrapper):
        if False:
            return 10
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(self.get_name(), self.kernel, self.codegen_args(), self.cpp_op_schema, self.cpp_kernel_key, self.cpp_kernel_overlad_name)

    @classmethod
    def create(cls, x, y, w, b, attr):
        if False:
            while True:
                i = 10
        x = cls.require_contiguous(cls.realize_input(x))
        y = cls.require_contiguous(cls.realize_input(y))
        w = cls.require_contiguous(cls.realize_input(w))
        (*m, ic) = x.get_size()
        (oc, ic) = w.get_size()
        inputs = [x, y, w]
        constant_args = [attr]
        if b is not None:
            b = cls.require_contiguous(cls.realize_input(b))
            inputs.append(b)
        else:
            constant_args.insert(0, b)
        return LinearBinary(layout=FlexibleLayout(device=x.get_device(), dtype=x.get_dtype(), size=list(m) + [oc]), inputs=inputs, constant_args=constant_args)

    def apply_constraint(self):
        if False:
            while True:
                i = 10
        pass

class ConvolutionTransposeUnary(ExternKernelAlloc):

    def __init__(self, layout, inputs, constant_args=()):
        if False:
            while True:
                i = 10
        super().__init__(layout, inputs, constant_args, None, kernel='torch.ops.mkldnn._convolution_transpose_pointwise', cpp_kernel='mkldnn::_convolution_transpose_pointwise')
        self.cpp_kernel_key = 'convolution_transpose_pointwise'
        self.cpp_op_schema = '\n            at::Tensor(\n                const at::Tensor& input_t,\n                const at::Tensor& weight_t,\n                const c10::optional<at::Tensor>& bias_opt,\n                at::IntArrayRef padding,\n                at::IntArrayRef output_padding,\n                at::IntArrayRef stride,\n                at::IntArrayRef dilation,\n                int64_t groups,\n                c10::string_view attr,\n                torch::List<c10::optional<at::Scalar>> scalars,\n                c10::optional<c10::string_view> algorithm)'

    def codegen(self, wrapper):
        if False:
            return 10
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(self.get_name(), self.kernel, self.codegen_args(), self.cpp_op_schema, self.cpp_kernel_key)

    @classmethod
    def create(cls, x: 'TensorBox', weight: 'TensorBox', bias: 'TensorBox', padding_: List[int], output_padding_: List[int], stride_: List[int], dilation_: List[int], groups_: int, attr, scalars: Optional[List[Any]], algorithm):
        if False:
            print('Hello World!')
        transposed = True
        (inputs, constant_args, kernel_layout, _) = _prepare_convolution_fusion_create(cls, x, weight, bias, padding_, stride_, dilation_, groups_, transposed, output_padding_)
        constant_args = constant_args + [attr, may_convert_to_optional(scalars), algorithm]
        return ConvolutionTransposeUnary(layout=kernel_layout, inputs=inputs, constant_args=constant_args)

class MkldnnRnnLayer(ExternKernelAlloc):

    def __init__(self, layout, inputs, constant_args=()):
        if False:
            while True:
                i = 10
        super().__init__(layout, inputs, constant_args, None, kernel='aten.mkldnn_rnn_layer', cpp_kernel='at::mkldnn_rnn_layer')

    @classmethod
    def create(cls, x: 'TensorBox', w0: 'TensorBox', w1: 'TensorBox', w2: 'TensorBox', w3: 'TensorBox', hx: 'TensorBox', cx: 'TensorBox', reverse: bool, batch_sizes: List[int], mode: int, hidden_size: int, num_layers: int, has_biases: bool, bidirectional: bool, batch_first: bool, train: bool):
        if False:
            for i in range(10):
                print('nop')
        x = cls.require_stride1(cls.realize_input(x))
        x.freeze_layout()
        w0 = cls.require_stride1(cls.realize_input(w0))
        w1 = cls.require_stride1(cls.realize_input(w1))
        w2 = cls.require_stride1(cls.realize_input(w2))
        w3 = cls.require_stride1(cls.realize_input(w3))
        hx = cls.require_stride1(cls.realize_input(hx))
        hx.freeze_layout()
        cx = cls.require_stride1(cls.realize_input(cx))
        cx.freeze_layout()
        input_size = x.get_size()
        assert len(input_size) == 3, 'Expect lstm input to be 3D'
        (seq_length, mini_batch, input_size) = input_size
        output_shape = [seq_length, mini_batch, hidden_size]
        hy_shape = hx.get_size()
        cy_shape = cx.get_size()
        res: List[IRNode] = []
        inputs = [x, w0, w1, w2, w3, hx, cx]
        constant_args = [reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train]
        packed = MkldnnRnnLayer(MultiOutputLayout(x.get_device()), inputs=inputs, constant_args=constant_args)

        def get_strides_of_lstm_output(output_shape, batch_first):
            if False:
                for i in range(10):
                    print('nop')
            assert len(output_shape) == 3, 'Expect output_shape to be 3D'
            return make_contiguous_strides_for(output_shape)
        output_sizes = [output_shape, hy_shape, cy_shape]
        output_strides = [get_strides_of_lstm_output(output_shape, batch_first), make_contiguous_strides_for(hy_shape), make_contiguous_strides_for(cy_shape)]
        output_ir = [MultiOutput(FixedLayout(x.get_device(), x.get_dtype(), output_size, output_stride), packed, [(tuple, i)]) for (i, (output_size, output_stride)) in enumerate(zip(output_sizes, output_strides))]
        return output_ir

class QConvPointWisePT2E(ExternKernelAlloc):

    def __init__(self, layout, inputs, constant_args=()):
        if False:
            return 10
        '\n        if bias is not None\n            - inputs = [x, w, b, weight_scale, weight_zp]\n            - const_args is: [stride, padding, dilation, groups, x_scale, x_zp, o_inv_scale, o_zp,\n              fp32_output, unary_attr, unary_scalars, unary_algorithm]\n        else\n            - inputs = [x, w, weight_scale, weight_zp]\n            - const_args is: [bias, stride, padding, dilation, groups, x_scale, x_zp, o_inv_scale, o_zp,\n              fp32_output, unary_attr, unary_scalars, unary_algorithm]\n        '
        self.has_bias = len(inputs) == 5
        super().__init__(layout, inputs, constant_args, None, kernel='torch.ops.onednn.qconv2d_pointwise', cpp_kernel='onednn::qconv2d_pointwise')
        self.cpp_kernel_key = 'qconv2d_pointwise'
        self.cpp_op_schema = '\n            at::Tensor(\n                at::Tensor act,\n                double act_scale,\n                int64_t act_zero_point,\n                at::Tensor weight,\n                at::Tensor weight_scales,\n                at::Tensor weight_zero_points,\n                c10::optional<at::Tensor> bias,\n                torch::List<int64_t> stride,\n                torch::List<int64_t> padding,\n                torch::List<int64_t> dilation,\n                int64_t groups,\n                double inv_output_scale,\n                int64_t output_zero_point,\n                c10::optional<c10::ScalarType> output_dtype,\n                c10::string_view attr,\n                torch::List<c10::optional<at::Scalar>> scalars,\n                c10::optional<c10::string_view> algorithm)'

    def codegen(self, wrapper):
        if False:
            return 10
        args = [x.codegen_reference() for x in self.inputs]
        const_args = []
        const_args.extend(self.codegen_const_args())
        x = args[0]
        packed_weight = args[1]
        bias = args[2] if self.has_bias else const_args[0]
        (w_scale, w_zp) = (args[-2], args[-1])
        (stride, padding, dilation, groups, x_scale, x_zp, o_inv_scale, o_zp, output_dtype, unary_attr, unary_scalars, unary_algorithm) = const_args[-12:]
        codegen_args = (x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias, stride, padding, dilation, groups, o_inv_scale, o_zp, output_dtype, unary_attr, unary_scalars, unary_algorithm)
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(self.get_name(), self.kernel, codegen_args, self.cpp_op_schema, self.cpp_kernel_key)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(cls, x: 'TensorBox', x_scale: float, x_zp: int, weight: 'TensorBox', w_scale: 'TensorBox', w_zp: 'TensorBox', bias: 'TensorBox', stride_: List[int], padding_: List[int], dilation_: List[int], groups: int, o_inv_scale: float, output_zero_point: int, output_dtype, unary_attr, unary_scalars, unary_algorithm):
        if False:
            i = 10
            return i + 15
        transposed = False
        output_padding = None
        (inputs, constant_args, kernel_layout, _) = _prepare_convolution_fusion_create(cls, x, weight, bias, padding_, stride_, dilation_, groups, transposed, output_padding)
        if bias is None:
            (constant_args[1], constant_args[2]) = (constant_args[2], constant_args[1])
        else:
            (constant_args[0], constant_args[1]) = (constant_args[1], constant_args[0])
        w_scale.realize()
        w_zp.realize()
        inputs = inputs + [w_scale, w_zp]
        constant_args = constant_args + [x_scale, x_zp, o_inv_scale, output_zero_point, output_dtype, unary_attr, may_convert_to_optional(unary_scalars), unary_algorithm]
        if output_dtype is not None:
            assert output_dtype in [torch.float32, torch.bfloat16]
            kernel_layout.dtype = output_dtype
        return QConvPointWisePT2E(layout=kernel_layout, inputs=inputs, constant_args=constant_args)

class QConvPointWiseBinaryPT2E(ExternKernelAlloc):

    def __init__(self, layout, inputs, constant_args=()):
        if False:
            i = 10
            return i + 15
        '\n        Needs input/weight/output qparams\n        if bias is not None\n            - inputs = [x, w, b, accum, w_scale, w_zp]\n            - const_args = [stride, padding, dilation, groups, x_scale, x_zp, accum_scale, accum_zp, o_inv_scale, o_zp,\n            fp32_output, binary_attr, aplha, unary_attr, unary_scalars, unary_algorithm]\n        else\n            - inputs = [x, w, accum, w_scale, w_zp]\n            - const_args = const_args is: [bias, stride, padding, dilation, groups, x_scale, x_zp, accum_scale,\n            accum_zp, o_inv_scale, o_zp, fp32_output, binary_attr, aplha, unary_attr, unary_scalars, unary_algorithm]\n        '
        self.has_bias = len(inputs) == 6
        super().__init__(layout, inputs, constant_args, None, kernel='torch.ops.onednn.qconv2d_pointwise.binary', cpp_kernel='onednn::qconv2d_pointwise')
        self.cpp_kernel_overlad_name = 'binary'
        self.cpp_kernel_key = 'qconv2d_pointwise_binary'
        self.cpp_op_schema = '\n            at::Tensor(\n                at::Tensor act,\n                double act_scale,\n                int64_t act_zero_point,\n                at::Tensor accum,\n                double accum_scale,\n                int64_t accum_zero_point,\n                at::Tensor weight,\n                at::Tensor weight_scales,\n                at::Tensor weight_zero_points,\n                c10::optional<at::Tensor> bias,\n                torch::List<int64_t> stride,\n                torch::List<int64_t> padding,\n                torch::List<int64_t> dilation,\n                int64_t groups,\n                double inv_output_scale,\n                int64_t output_zero_point,\n                c10::optional<c10::ScalarType> output_dtype,\n                c10::string_view binary_attr,\n                c10::optional<at::Scalar> alpha,\n                c10::optional<c10::string_view> attr,\n                torch::List<c10::optional<at::Scalar>> scalars,\n                c10::optional<c10::string_view> algorithm)'

    def codegen(self, wrapper):
        if False:
            print('Hello World!')
        args = [x.codegen_reference() for x in self.inputs]
        const_args = []
        const_args.extend(self.codegen_const_args())
        x = args[0]
        packed_weight = args[1]
        bias = args[2] if self.has_bias else const_args[0]
        (accum, w_scale, w_zp) = (args[-3], args[-2], args[-1])
        (stride, padding, dilation, groups, x_scale, x_zp, accum_scale, accum_zp, o_inv_scale, o_zp, output_dtype, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithm) = const_args[-16:]
        conv_args = (x, x_scale, x_zp, accum, accum_scale, accum_zp, packed_weight, w_scale, w_zp, bias, stride, padding, dilation, groups, o_inv_scale, o_zp, output_dtype, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithm)
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(self.get_name(), self.kernel, conv_args, self.cpp_op_schema, self.cpp_kernel_key, self.cpp_kernel_overlad_name)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(cls, x: 'TensorBox', x_scale, x_zp, accum: 'TensorBox', accum_scale, accum_zp, weight: 'TensorBox', w_scale, w_zp, bias: 'TensorBox', stride_: List[int], padding_: List[int], dilation_: List[int], groups: int, o_inv_scale: 'TensorBox', output_zero_point: 'TensorBox', output_dtype, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithm):
        if False:
            print('Hello World!')
        transposed = False
        output_padding = None
        (inputs, constant_args, kernel_layout, req_stride_order) = _prepare_convolution_fusion_create(cls, x, weight, bias, padding_, stride_, dilation_, groups, transposed, output_padding)
        accum = cls.require_stride_order(accum, req_stride_order)
        inputs.append(accum)
        if bias is None:
            (constant_args[1], constant_args[2]) = (constant_args[2], constant_args[1])
        else:
            (constant_args[0], constant_args[1]) = (constant_args[1], constant_args[0])
        w_scale.realize()
        w_zp.realize()
        inputs = inputs + [w_scale, w_zp]
        constant_args = constant_args + [x_scale, x_zp, accum_scale, accum_zp, o_inv_scale, output_zero_point, output_dtype, binary_attr, alpha, unary_attr, may_convert_to_optional(unary_scalars), unary_algorithm]
        if output_dtype is not None:
            kernel_layout.dtype = output_dtype
        return QConvPointWiseBinaryPT2E(layout=kernel_layout, inputs=inputs, constant_args=constant_args)

class QLinearPointwisePT2E(ExternKernelAlloc):

    def __init__(self, layout, inputs, constant_args=()):
        if False:
            for i in range(10):
                print('nop')
        '\n        if bias is not None\n            - inputs = [x, w, b, weight_scale, weight_zp]\n            - const_args is: [x_scale, x_zp, o_inv_scale, o_zp,\n              fp32_output, unary_attr, unary_scalars, unary_algorithm]\n        else\n            - inputs = [x, w, weight_scale, weight_zp]\n            - const_args is: [bias, x_scale, x_zp, o_inv_scale, o_zp,\n              fp32_output, unary_attr, unary_scalars, unary_algorithm]\n        '
        self.has_bias = len(inputs) == 5
        super().__init__(layout, inputs, constant_args, None, kernel='torch.ops.onednn.qlinear_pointwise', cpp_kernel='onednn::qlinear_pointwise')
        self.cpp_kernel_key = 'qlinear_pointwise'
        self.cpp_op_schema = '\n            at::Tensor(\n                at::Tensor act,\n                double act_scale,\n                int64_t act_zero_point,\n                at::Tensor weight,\n                at::Tensor weight_scales,\n                at::Tensor weight_zero_points,\n                c10::optional<at::Tensor> bias,\n                double inv_output_scale,\n                int64_t output_zero_point,\n                c10::optional<c10::ScalarType> output_dtype,\n                std::string post_op_name,\n                torch::List<c10::optional<at::Scalar>> post_op_args,\n                std::string post_op_algorithm)'

    def codegen(self, wrapper):
        if False:
            for i in range(10):
                print('nop')
        args = [x.codegen_reference() for x in self.inputs]
        const_args = []
        const_args.extend(self.codegen_const_args())
        x = args[0]
        packed_weight = args[1]
        bias = args[2] if self.has_bias else const_args[0]
        (w_scale, w_zp) = (args[-2], args[-1])
        (x_scale, x_zp, o_inv_scale, o_zp, output_dtype, unary_attr, unary_scalars, unary_algorithm) = const_args[-8:]
        codegen_args = (x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias, o_inv_scale, o_zp, output_dtype, unary_attr, unary_scalars, unary_algorithm)
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(self.get_name(), self.kernel, codegen_args, self.cpp_op_schema, self.cpp_kernel_key)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(cls, x: 'TensorBox', x_scale: float, x_zp: int, weight: 'TensorBox', w_scale: 'TensorBox', w_zp: 'TensorBox', bias: 'TensorBox', o_inv_scale: float, output_zero_point: int, output_dtype, unary_attr, unary_scalars, unary_algorithm):
        if False:
            for i in range(10):
                print('nop')
        (inputs, constant_args, kernel_layout, _) = _prepare_linear_fusion_create(cls, x, weight, bias)
        w_scale.realize()
        w_zp.realize()
        inputs = inputs + [w_scale, w_zp]
        constant_args = constant_args + [x_scale, x_zp, o_inv_scale, output_zero_point, output_dtype, unary_attr, may_convert_to_optional(unary_scalars), unary_algorithm]
        if output_dtype is not None:
            assert output_dtype in [torch.float32, torch.bfloat16]
            kernel_layout.dtype = output_dtype
        return QLinearPointwisePT2E(layout=kernel_layout, inputs=inputs, constant_args=constant_args)

@dataclasses.dataclass
class MutableBox(IRNode):
    """
    TensorBox / StorageBox allow in-place mutation of Tensors
    """
    data: IRNode

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        fn = getattr(self.data, name)
        if callable(fn):
            return fn
        raise AttributeError(f'{type(self.data).__name__}.{name} not callable')

    def realize(self):
        if False:
            return 10
        return self.data.realize()

    @property
    def layout(self):
        if False:
            while True:
                i = 10
        return self.data.layout

    def get_layout(self):
        if False:
            i = 10
            return i + 15
        return self.layout

    def get_size(self):
        if False:
            i = 10
            return i + 15
        return self.data.get_size()

    def __str__(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self.data, MutableBox):
            line0 = f'{type(self).__name__}({type(self.data).__name__}('
            endl = '))'
            inner = self.data.data
        else:
            line0 = f'{type(self).__name__}('
            inner = self.data
            endl = ')'
        lines = [line0, indent(str(inner)), endl]
        return '\n'.join(lines)
    __repr__ = __str__

class TensorBox(MutableBox):

    @staticmethod
    def create(data):
        if False:
            return 10
        return TensorBox(StorageBox(data))

class StorageBox(MutableBox):

    def is_input_buffer(self):
        if False:
            while True:
                i = 10
        if isinstance(self.data, (InputBuffer, ReinterpretView)):
            return self.data.get_name() in V.graph.graph_inputs
        return False

    def realize(self):
        if False:
            print('Hello World!')
        if isinstance(self.data, (ComputedBuffer, InputsKernel, InputBuffer, ReinterpretView, TemplateBuffer)):
            return self.data.get_name()
        assert isinstance(self.data, (Pointwise, Reduction)), type(self.data)
        origin_node = self.data.get_origin_node()
        traceback = self.data.get_traceback()
        self.data = ComputedBuffer(name=None, layout=FlexibleLayout(device=self.data.get_device(), dtype=self.data.get_dtype(), size=self.data.get_size()), data=self.data)
        self.data.name = V.graph.register_buffer(self.data)
        self.data.origins = self.origins
        self.data.origin_node = origin_node
        self.data.traceback = traceback
        return self.data.name

    def realize_hint(self):
        if False:
            print('Hello World!')
        '\n        Called on buffers we expect to be forced to realize later.\n        '
        if isinstance(self.data, (Pointwise, Reduction)) and self.num_reads() > 1 and self.is_pointwise_non_scalar_tensor_num_reads_larger_than_one():
            self.realize()

    def has_exceeded_max_reads(self):
        if False:
            return 10
        return isinstance(self.data, Pointwise) and (self.num_reads() > config.realize_acc_reads_threshold or self.inner_fn_str_len() > config.realize_bytes_threshold)

    def mark_reuse(self, users):
        if False:
            i = 10
            return i + 15
        '\n        A heuristic to decide if we should realize a tensor\n        that is used multiple times.\n        '

        def should_realize_on_cpu(loops: Union[Pointwise, Reduction]):
            if False:
                while True:
                    i = 10
            '\n            The heuristic for realizing reused result of heavy ops on cpu\n            '
            heavy_ops = ['exp']
            fn_str = loops.inner_fn_str()
            return any((op + '(' in fn_str for op in heavy_ops))
        if users > 1 and isinstance(self.data, (Pointwise, Reduction)) and (self.num_reads() > config.realize_reads_threshold or len(self.inner_fn_str()) > config.realize_bytes_threshold or (is_cpu(self.data) and should_realize_on_cpu(self.data))):
            self.realize()

    @cache_on_self
    def num_reads(self):
        if False:
            for i in range(10):
                print('nop')
        data = self.data
        if isinstance(data, (InputsKernel, InputBuffer, ReinterpretView)):
            return 1
        if isinstance(data, ComputedBuffer):
            read_writes = data.get_read_writes()
        else:
            assert isinstance(data, (Pointwise, Reduction)), type(data)
            read_writes = ComputedBuffer(name=None, layout=FlexibleLayout(device=data.get_device(), dtype=data.get_dtype(), size=data.get_size()), data=data).get_read_writes()
        return len(read_writes.reads)

    @cache_on_self
    def is_pointwise_non_scalar_tensor_num_reads_larger_than_one(self):
        if False:
            print('Hello World!')
        return sum((read.index != 0 for read in self.data.get_reads())) > 1 if isinstance(self.data, Pointwise) and all((not isinstance(read, dependencies.StarDep) for read in self.data.get_reads())) else True

class InterpreterShim(torch.fx.Interpreter):

    @staticmethod
    @functools.lru_cache(None)
    def _dummy_gm():
        if False:
            print('Hello World!')
        return torch.fx.symbolic_trace(identity)

    def __init__(self, graph, submodules):
        if False:
            while True:
                i = 10
        super().__init__(self._dummy_gm(), garbage_collect_values=False)
        self.module = self
        self.graph = graph
        self.submodules = submodules
        self.extra_traceback = False
        self.fetch_attr = submodules.__getitem__
        self.current_node = None

    def run_node(self, n: torch.fx.Node) -> Any:
        if False:
            print('Hello World!')
        self.current_node = n
        return super().run_node(n)

    def run(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        with V.set_interpreter_handler(self):
            return super().run(*args, **kwargs)

class LoopBody:
    """
    Captures the body of a Loops subclass into an FX graph.  Persists any
    indexing simplifications and makes it easier to analyze loop bodies.
    """

    def __init__(self, fn, args, var_ranges):
        if False:
            while True:
                i = 10
        super().__init__()
        self.var_ranges = var_ranges
        self.indexing_exprs = {}
        self.indexing_exprs_name = {}
        self.reads = []
        self.writes = []
        self.reads_name2expr = {}
        self.writes_name2expr = {}
        self.other = []
        self.submodules = {'get_index': self.get_index}
        self.subblocks = {}
        self.indirect_vars = []
        self.root_block = LoopBodyBlock(self, fn, args)
        self.indexing = None

    @cache_on_self
    def get_nodes(self):
        if False:
            print('Hello World!')
        all_graphs = itertools.chain((self.root_block.graph,), (block.graph for block in self.subblocks.values()))
        return [node for graph in all_graphs for node in graph.nodes]

    @cache_on_self
    def bounds(self):
        if False:
            for i in range(10):
                print('nop')
        from .bounds import BoundVars
        return BoundVars(self)

    def debug_str(self):
        if False:
            print('Hello World!')
        lines = [f'var_ranges = {dict(self.var_ranges)}']
        lines.extend([f'{name} = {val}' for (name, val) in self.indexing_exprs.items()])
        lines.extend([block.debug_str(name) for (name, block) in itertools.chain([('body', self.root_block)], self.subblocks.items())])
        return '\n'.join(lines)

    def add_index_expr(self, expr: sympy.Expr, category, buf_name):
        if False:
            i = 10
            return i + 15
        getattr(self, category).append(expr)
        if buf_name is not None:
            getattr(self, f'{category}_name2expr')[buf_name] = expr
        if expr not in self.indexing_exprs_name:
            name = f'index{len(self.indexing_exprs)}'
            self.indexing_exprs_name[expr] = name
            self.indexing_exprs[name] = expr
        return self.indexing_exprs_name[expr]

    def add_submodule(self, block, prefix):
        if False:
            return 10
        'Not actually for nn.Modules, but subblocks in generated code are mapped to FX call_module opcodes'
        if prefix[-1].isnumeric() and prefix not in self.submodules:
            name = prefix
        else:
            name = f'{prefix}{len(self.submodules)}'
        self.submodules[name] = block
        return name

    def add_indirect(self, size):
        if False:
            print('Hello World!')
        name = f'indirect{len(self.indirect_vars)}'
        var = sympy_symbol(name)
        self.indirect_vars.append(var)
        return var

    def replace_indirect(self, old, new):
        if False:
            for i in range(10):
                print('nop')
        'Swap in a variable used in indirect indexing'
        if str(old) == str(new):
            return
        assert self.indexing is not None
        self.indexing = {k: sympy_subs(v, {old: new}) for (k, v) in self.indexing.items()}

    def get_index(self, name):
        if False:
            while True:
                i = 10
        assert self.indexing is not None
        return self.indexing[name]

    def __call__(self, *indices):
        if False:
            i = 10
            return i + 15
        index = list(itertools.chain(*indices))
        assert len(index) == len(self.var_ranges), (index, self.var_ranges)
        assert all((v not in self.var_ranges for v in index))
        replacements = dict(zip(self.var_ranges.keys(), index))
        self.indexing = {name: sympy_subs(expr, replacements) for (name, expr) in self.indexing_exprs.items()}
        result = self.root_block()
        self.indexing = None
        return result

class LoopBodyBlock:
    """
    Captures the body of a Loops subclass into an FX graph.
    In normal cases there will be a 1:1 mapping between LoopBody and
    LoopBodyBlock, hower in the case of ops.masked() the masked out
    operations will manifest as an extra LoopBodyBlock.
    """

    def __init__(self, body: LoopBody, fn: Callable[..., Any], args: List[Any]):
        if False:
            return 10
        self.body = body

        def add_index(expr, category, buf_name=None):
            if False:
                return 10
            return tracer.create_proxy('call_module', 'get_index', (self.body.add_index_expr(expr, category, buf_name),), {})

        class CaptureIndexing(V.WrapperHandler):
            self.name = 'CaptureIndexing'

            def load(self, name: str, index: sympy.Expr):
                if False:
                    for i in range(10):
                        print('nop')
                index = add_index(index, 'reads', name)
                return self._inner.load(name, index)

            def store(self, name, index, value, mode=None):
                if False:
                    while True:
                        i = 10
                index = add_index(index, 'writes', name)
                return self._inner.store(name, index, value, mode)

            def store_reduction(self, name, index, value):
                if False:
                    while True:
                        i = 10
                index = add_index(index, 'writes', name)
                return self._inner.store_reduction(name, index, value)

            def reduction(self, dtype, src_dtype, reduction_type, value):
                if False:
                    print('Hello World!')
                result = self._inner.reduction(dtype, src_dtype, reduction_type, value)
                if 'welford' in reduction_type:
                    return tuple((result[i] for i in range(3)))
                return result

            def index_expr(self, index, dtype):
                if False:
                    print('Hello World!')
                if isinstance(index, (int, sympy.Integer)):
                    return self._inner.constant(int(index), dtype)
                index = add_index(index, 'other')
                return self._inner.index_expr(index, dtype)

            def bucketize(self, values, offsets_name: str, offsets_size: sympy.Expr, indexing_dtype: torch.dtype, right: bool):
                if False:
                    i = 10
                    return i + 15
                offsets_size = add_index(offsets_size, 'other')
                return self._inner.bucketize(values, offsets_name, offsets_size, indexing_dtype, right)

            @staticmethod
            def masked(mask_proxy, masked_body: Callable[..., Any], other_proxy):
                if False:
                    while True:
                        i = 10
                '\n                Recursively capture the masked out body in another LoopBodyBlock\n                '
                subblock: LoopBodyBlock

                def shim(mask, other):
                    if False:
                        i = 10
                        return i + 15
                    return V.ops.masked(mask, subblock, other)
                name = self.body.add_submodule(shim, 'masked_subblock')
                subblock = LoopBodyBlock(self.body, masked_body, [])
                self.body.subblocks[name] = subblock
                return tracer.create_proxy('call_module', name, (mask_proxy, other_proxy), {})

            @staticmethod
            def indirect_indexing(index_proxy, size, check=True):
                if False:
                    return 10
                '\n                Flow data from tensors into indexing formulas.\n                Introduce a call_module to update the indexing.\n                '
                var = self.body.add_indirect(size)

                def set_indirect(new_var):
                    if False:
                        while True:
                            i = 10
                    self.body.replace_indirect(var, V.ops.indirect_indexing(new_var, size, check))
                tracer.create_proxy('call_module', self.body.add_submodule(set_indirect, f'set_{var}'), (index_proxy,), {})
                return var

            @staticmethod
            def output(result):
                if False:
                    print('Hello World!')
                tracer.create_proxy('output', 'output', (result,), {})
        tracer = torch.fx.Tracer()
        tracer.graph = torch.fx.Graph(tracer_cls=tracer.__class__)
        proxy_ops = tracer.create_proxy('placeholder', 'ops', (), {})
        from .index_propagation import IndexPropagation
        from .sizevars import SimplifyIndexing
        handler: Any = SimplifyIndexing(CaptureIndexing(proxy_ops), self.body.var_ranges)
        if config.constant_and_index_propagation:
            handler = IndexPropagation(handler)
        with V.set_ops_handler(handler):
            ops.output(fn(*args))
        self.graph = tracer.graph

    def __call__(self):
        if False:
            i = 10
            return i + 15
        graph = self.graph
        submodules = self.body.submodules
        return InterpreterShim(graph, submodules).run(V.get_ops_handler())

    def debug_str(self, name='block'):
        if False:
            for i in range(10):
                print('nop')
        code = torch.fx.GraphModule(self.body.submodules, self.graph).code
        return re.sub(';[^\\n]*', '', code.strip().replace('def forward(', f'def {name}('))

class Wait(ExternKernelAlloc):
    """
    Wait should not be used by itself.  It should always be constructed in tandem
    with a collective op that produces a work to wait on.
    """

    def __init__(self, layout, inputs, constant_args=()):
        if False:
            return 10
        super().__init__(layout, inputs, constant_args)

    def should_allocate(self):
        if False:
            return 10
        return False

    def codegen(self, wrapper):
        if False:
            return 10
        from .codegen.wrapper import ReuseLine
        wrapper.add_import_once('from torch.distributed._functional_collectives_impl import _wait_tensor')
        (input_collective,) = (t.codegen_reference() for t in self.inputs)
        wrapper.writeline(f'{input_collective} = _wait_tensor({input_collective})')
        wrapper.writeline(ReuseLine(wrapper, self.inputs[0], self, delete_old=False))

    @classmethod
    def create(cls, collective_op: 'TensorBox'):
        if False:
            while True:
                i = 10
        collective_op.decide_layout()
        return Wait(layout=AliasedLayout(collective_op), inputs=[collective_op])

    def get_alias_names(self):
        if False:
            print('Hello World!')
        return [self.inputs[0].codegen_reference()]

class CollectiveKernel(ExternKernel):
    """
    Each collective should follow the pattern:
    - extend InPlaceCollectiveKernel or OutOfPlaceCollectiveKernel.
    - the kernel delegates into c10d processgroup, which returns a 'work' obj
    - the work obj is registered via _register_tensor_work so it can be waited on later
    """

    def __init__(self, layout, inputs, constant_args):
        if False:
            while True:
                i = 10
        super().__init__(None, layout, inputs, constant_args)
        self.name = V.graph.register_buffer(self)

    def should_emit_register_tensor_work(self):
        if False:
            print('Hello World!')
        return True

    def should_emit_find_or_create_pg(self):
        if False:
            return 10
        return True

    def codegen_collective(self, wrapper, output_name, input_names):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('Must implement')

    def codegen_output(self, wrapper, output_name, input_names):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Must implement')

    @classmethod
    def wrap_inputs_as_inplace(cls, inputs):
        if False:
            return 10

        def wrap_input(var):
            if False:
                return 10
            op = InPlaceHint(FlexibleLayout(var.get_device(), var.get_dtype(), var.get_size()), var)
            return TensorBox.create(op)
        return list(map(wrap_input, inputs))

    def codegen(self, wrapper):
        if False:
            while True:
                i = 10
        wrapper.add_import_once('import torch.distributed as dist')
        wrapper.add_import_once('import torch.distributed.distributed_c10d as c10d')
        wrapper.add_import_once('import torch.distributed._functional_collectives_impl as fun_col_impl')
        input_names = [t.codegen_reference() for t in self.inputs]
        output_name = self.get_name()
        (tag, ranks, group_size) = self.constant_args
        if self.should_emit_find_or_create_pg():
            wrapper.writeline(f"{output_name}_pg = c10d._find_or_create_pg_by_ranks_and_tag('{tag}', {ranks}, {group_size})")
        self.codegen_output(wrapper, output_name, input_names)
        self.codegen_collective(wrapper, output_name, input_names)
        if self.should_emit_register_tensor_work():
            wrapper.writeline(f'fun_col_impl._register_tensor_work({output_name}, {output_name}_work)')

class InPlaceCollectiveKernel(CollectiveKernel):
    """
    InPlaceCollectiveKernel are those with in-out arguments such as all_reduce.
    Extend this kernel if your collective needs to modify its inputs in-place.
    """

    def __init__(self, layout, inputs, constant_args):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(layout, inputs, constant_args)

    def should_allocate(self):
        if False:
            print('Hello World!')
        return False

    def has_side_effects(self):
        if False:
            return 10
        return True

    def codegen_output(self, wrapper, output_name, input_names):
        if False:
            i = 10
            return i + 15
        if len(input_names) > 1:
            wrapper.writeline(f"{output_name} = [{','.join(input_names)}] ")
        else:
            wrapper.writeline(f'{output_name} = {input_names[0]}')

class OutOfPlaceCollectiveKernel(CollectiveKernel):
    """
    OutOfPlaceCollectiveKernel are those that allocate their
    outputs and leave their inputs inplace, such as all_gather.
    """

    def __init__(self, layout, inputs, outputs, constant_args):
        if False:
            print('Hello World!')
        super().__init__(layout, inputs + outputs, constant_args)
        self.outputs = outputs
        self.original_inputs = inputs
        for x in self.outputs:
            V.graph.never_reuse_buffers.add(x.name)

    def should_allocate(self):
        if False:
            i = 10
            return i + 15
        return False

    def has_side_effects(self):
        if False:
            while True:
                i = 10
        return True

    def codegen_output(self, wrapper, output_name, input_names):
        if False:
            for i in range(10):
                print('nop')
        input_names = [t.codegen_reference() for t in self.original_inputs]
        wrapper.writeline(f"{output_name}_inputs = [{','.join(input_names)}]")
        wrapper.writeline(f"{output_name} = [{','.join((x.name for x in self.outputs))}]")

    @classmethod
    def create_output_buffers(cls, inputs, size_cb=None):
        if False:
            for i in range(10):
                print('nop')
        outputs = []
        for input in inputs:
            new_size = input.get_size()
            if size_cb is not None:
                size_cb(new_size)
            buff = OutputBuffer(layout=FlexibleLayout(device=input.get_device(), dtype=input.get_dtype(), size=new_size))
            outputs.append(buff)
        return outputs

    @classmethod
    def create_output_nodes(cls, coll, output_buffers):
        if False:
            i = 10
            return i + 15
        return [MultiOutputNoSizeAssert(out_t.layout, coll, f'[{i}]') for (i, out_t) in enumerate(output_buffers)]

class InPlaceHint(ExternKernel):
    """
    Helper OP to encode an in/out argument that tries to make it inplace whenever possible.
    Wrap the input of your inplace op to enable this behavior.

    The design is based on two key decisions:
    - this node is responsible for allocating the in/out buffer used by the collective.
        This is controlled by the ``should_allocate`` method that returns True here and
        False for the collective node
    - The scheduler special-case this node and enable it to reuse its input.
    """

    def codegen(self, wrapper):
        if False:
            for i in range(10):
                print('nop')
        input_name = self.inputs[0].codegen_reference()
        output_name = self.get_name()
        if not wrapper.did_reuse(self, self.inputs[0]):
            wrapper.writeline(f'{output_name}.copy_({input_name}) #no reuse')

    def __init__(self, layout, input):
        if False:
            while True:
                i = 10
        input = self.realize_input(input)
        super().__init__(None, layout, self.unwrap_storage([input]), ())
        self.name = V.graph.register_buffer(self)

    def should_allocate(self):
        if False:
            return 10
        return True

class OutputBuffer(ExternKernel):
    """
    Represent the output buffer used by ops that require multiple of them
    """

    def __init__(self, layout):
        if False:
            while True:
                i = 10
        super().__init__(name=None, layout=layout, inputs=[])
        self.name = V.graph.register_buffer(self)

    def should_allocate(self):
        if False:
            i = 10
            return i + 15
        return True

    def codegen(self, wrapper):
        if False:
            while True:
                i = 10
        wrapper.writeline(f'# collective out buffer {self.name}')

class MultiOutputNoSizeAssert(MultiOutput):
    """
    Extract partial output from a multi-output OP.
    Works like MultiOutput but doesn't assert size. This must be a property guaranteed by the op emitting this.
    """

    def __init__(self, layout, input, index):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(layout, input, [])
        self.index = index

    def codegen(self, wrapper):
        if False:
            print('Hello World!')
        wrapper.writeline(f'{self.get_name()} = {self.inputs[0].get_name()}{self.index}')

class Broadcast(InPlaceCollectiveKernel):

    def __init__(self, layout, inputs, constant_args, src):
        if False:
            print('Hello World!')
        super().__init__(layout, inputs, constant_args)
        self.src = src

    def get_mutation_names(self):
        if False:
            return 10
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self):
        if False:
            return 10
        return {}

    @classmethod
    def create(cls, x: 'TensorBox', src: int, tag: str, ranks: List[int], group_size: int):
        if False:
            print('Hello World!')
        inplace_inputs = cls.wrap_inputs_as_inplace([x])
        packed = Broadcast(layout=NoneLayout(inplace_inputs[0].get_device()), inputs=inplace_inputs, constant_args=[tag, ranks, group_size], src=src)
        mark_node_as_mutating(packed, inplace_inputs[0])
        return inplace_inputs[0]

    def codegen_collective(self, wrapper, output_name, input_names):
        if False:
            while True:
                i = 10
        wrapper.writeline(f'{output_name}_work = dist.broadcast({output_name}, async_op=True, group={output_name}_pg, src={self.src})')

class AllReduceCoalesced(InPlaceCollectiveKernel):

    def __init__(self, layout, inputs, constant_args, reduce_op):
        if False:
            print('Hello World!')
        super().__init__(layout, inputs, constant_args)
        self.reduce_op = reduce_op

    def should_allocate(self):
        if False:
            i = 10
            return i + 15
        return False

    def get_mutation_names(self):
        if False:
            while True:
                i = 10
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self):
        if False:
            for i in range(10):
                print('nop')
        return {}

    @classmethod
    def create(cls, inputs: List['TensorBox'], reduce_op: str, tag: str, ranks: List[int], group_size: int):
        if False:
            print('Hello World!')
        inplace_inputs = cls.wrap_inputs_as_inplace(inputs)
        packed = AllReduceCoalesced(layout=NoneLayout(inplace_inputs[0].get_device()), inputs=inplace_inputs, constant_args=[tag, ranks, group_size], reduce_op=reduce_op)
        mark_node_as_mutating(packed, inplace_inputs[0])
        return inplace_inputs

    def codegen_collective(self, wrapper, output_name, input_names):
        if False:
            for i in range(10):
                print('nop')
        wrapper.writeline(f"{output_name}_work = dist.all_reduce_coalesced({output_name}, op=fun_col_impl._str_to_reduce_op('{str(self.reduce_op)}'), group={output_name}_pg, async_op=True)")

class AllReduce(InPlaceCollectiveKernel):

    def __init__(self, layout, inputs, constant_args, reduce_op):
        if False:
            print('Hello World!')
        super().__init__(layout, inputs, constant_args)
        self.reduce_op = reduce_op

    def get_mutation_names(self):
        if False:
            while True:
                i = 10
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self):
        if False:
            i = 10
            return i + 15
        return {}

    @classmethod
    def create(cls, x: 'TensorBox', reduce_op: str, tag: str, ranks: List[int], group_size: int):
        if False:
            return 10
        inplace_inputs = cls.wrap_inputs_as_inplace([x])
        packed = AllReduce(layout=NoneLayout(inplace_inputs[0].get_device()), inputs=inplace_inputs, constant_args=[tag, ranks, group_size], reduce_op=reduce_op)
        mark_node_as_mutating(packed, inplace_inputs[0])
        return inplace_inputs[0]

    def codegen_collective(self, wrapper, output_name, input_names):
        if False:
            for i in range(10):
                print('nop')
        wrapper.writeline(f"{output_name}_work = dist.all_reduce({output_name}, async_op=True, group={output_name}_pg, op=fun_col_impl._str_to_reduce_op('{str(self.reduce_op)}'))")

class AllGatherIntoTensor(OutOfPlaceCollectiveKernel):

    def __init__(self, layout, inputs, outputs, constant_args):
        if False:
            while True:
                i = 10
        super().__init__(layout, inputs, outputs, constant_args)

    @classmethod
    def create(cls, x: 'TensorBox', tag: str, ranks: List[int], group_size: int):
        if False:
            return 10
        inputs = [cls.realize_input(x)]

        def compute_size(new_size):
            if False:
                return 10
            new_size[0] *= group_size
        outputs = cls.create_output_buffers(inputs, compute_size)
        layout = MultiOutputLayout(inputs[0].get_device())
        packed = AllGatherIntoTensor(layout=layout, inputs=inputs, outputs=outputs, constant_args=[tag, ranks, group_size])
        return cls.create_output_nodes(packed, outputs)[0]

    def codegen_collective(self, wrapper, output_name, input_names):
        if False:
            while True:
                i = 10
        wrapper.writeline(f'{output_name}_work = dist.all_gather_into_tensor({output_name}[0], {output_name}_inputs[0], async_op=True, group={output_name}_pg)')

class ReduceScatterTensor(OutOfPlaceCollectiveKernel):

    def __init__(self, layout, inputs, outputs, constant_args, reduce_op):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(layout, inputs, outputs, constant_args)
        self.reduce_op = reduce_op

    @classmethod
    def create(cls, x: 'TensorBox', reduce_op: str, tag: str, ranks: List[int], group_size: int):
        if False:
            while True:
                i = 10
        inputs = [cls.realize_input(x)]

        def compute_size(new_size):
            if False:
                i = 10
                return i + 15
            new_size[0] //= group_size
        outputs = cls.create_output_buffers(inputs, compute_size)
        layout = MultiOutputLayout(inputs[0].get_device())
        packed = ReduceScatterTensor(layout=layout, inputs=inputs, outputs=outputs, constant_args=[tag, ranks, group_size], reduce_op=reduce_op)
        return cls.create_output_nodes(packed, outputs)[0]

    def codegen_collective(self, wrapper, output_name, input_names):
        if False:
            print('Hello World!')
        wrapper.writeline(f"{output_name}_work = dist.reduce_scatter_tensor({output_name}[0], {output_name}_inputs[0], async_op=True, group={output_name}_pg, op=fun_col_impl._str_to_reduce_op('{str(self.reduce_op)}'))")

class AllGatherIntoTensorCoalesced(OutOfPlaceCollectiveKernel):

    def __init__(self, layout, inputs, outputs, constant_args):
        if False:
            return 10
        super().__init__(layout, inputs, outputs, constant_args)

    @classmethod
    def create(cls, inputs: List['TensorBox'], tag: str, ranks: List[int], group_size: int):
        if False:
            for i in range(10):
                print('nop')
        inputs = [cls.realize_input(x) for x in inputs]

        def compute_size(new_size):
            if False:
                return 10
            new_size[0] *= group_size
        outputs = cls.create_output_buffers(inputs, compute_size)
        layout = MultiOutputLayout(inputs[0].get_device())
        packed = AllGatherIntoTensorCoalesced(layout=layout, inputs=inputs, outputs=outputs, constant_args=[tag, ranks, group_size])
        return outputs

    def codegen_collective(self, wrapper, output_name, input_names):
        if False:
            for i in range(10):
                print('nop')
        wrapper.writeline(f'{output_name}_work = fun_col_impl._all_gather_into_tensor_coalesced_fallback(output_tensors={output_name}, input_tensors={output_name}_inputs, group={output_name}_pg, async_op=True)')

class ReduceScatterTensorCoalesced(OutOfPlaceCollectiveKernel):

    def __init__(self, layout, inputs, outputs, constant_args, reduce_op):
        if False:
            while True:
                i = 10
        super().__init__(layout, inputs, outputs, constant_args)
        self.reduce_op = reduce_op

    @classmethod
    def create(cls, inputs: List['TensorBox'], reduce_op: str, tag: str, ranks: List[int], group_size: int):
        if False:
            return 10
        inputs = [cls.realize_input(x) for x in inputs]

        def compute_size(new_size):
            if False:
                for i in range(10):
                    print('nop')
            new_size[0] //= group_size
        outputs = cls.create_output_buffers(inputs, compute_size)
        layout = MultiOutputLayout(inputs[0].get_device())
        _ = ReduceScatterTensorCoalesced(layout=layout, inputs=inputs, outputs=outputs, constant_args=[tag, ranks, group_size], reduce_op=reduce_op)
        return outputs

    def codegen_collective(self, wrapper, output_name, input_names):
        if False:
            for i in range(10):
                print('nop')
        wrapper.writeline(f"{output_name}_work = fun_col_impl._reduce_scatter_tensor_coalesced_fallback(output_tensors={output_name}, input_tensors={output_name}_inputs, op=fun_col_impl._str_to_reduce_op('{str(self.reduce_op)}'), group={output_name}_pg, async_op=True)")

class _CollectiveKernel(FallbackKernel):

    def should_allocate(self):
        if False:
            print('Hello World!')
        return False

    def has_side_effects(self):
        if False:
            return 10
        return True

    def set_cpp_kernel(self, kernel):
        if False:
            for i in range(10):
                print('nop')
        from .codegen.wrapper import get_cpp_op_schema
        self.kernel = kernel._schema.name
        self.cpp_kernel_overlad_name = kernel._schema.overload_name
        self.cpp_kernel_key = f"{self.kernel.replace('::', '_')}_{self.cpp_kernel_overlad_name}"
        self.cpp_op_schema = get_cpp_op_schema(kernel)
        self.ordered_kwargs_for_cpp_kernel = [x.name for x in kernel._schema.arguments if x.kwarg_only]

    @classmethod
    def create_inplace(cls, kernel, inputs: Union[TensorBox, List[TensorBox]], *args, **kwargs) -> None:
        if False:
            return 10
        with V.graph.fake_mode:
            (example_output, tensor_args, non_tensor_args, unflatten_args) = cls.process_kernel(kernel, inputs, *args, **kwargs)
        for tensor_arg in tensor_args:
            tensor_arg.realize()
        packed = cls(NoneLayout(tensor_args[0].get_device()), kernel, tensor_args, non_tensor_args, unflatten_args)
        pytree.tree_map(lambda x: MutationOutput(x.layout, x, packed), inputs)

    @classmethod
    def create_out_of_place(cls, kernel, inputs: Union[TensorBox, List[TensorBox]], *args, **kwargs):
        if False:
            print('Hello World!')
        with V.graph.fake_mode:
            (example_output, tensor_args, non_tensor_args, unflatten_args) = cls.process_kernel(kernel, inputs, *args, **kwargs)
        for tensor_arg in tensor_args:
            tensor_arg.realize()
        if isinstance(example_output, list):
            device = cls.find_device(tensor_args, example_output)
            packed = cls(MultiOutputLayout(device), kernel, tensor_args, non_tensor_args, unflatten_args)
            packed.outputs = [MultiOutput(cls.tensor_to_layout(tensor), packed, [(list, i)]) for (i, tensor) in enumerate(example_output)]
            return packed.outputs
        else:
            packed = cls(cls.tensor_to_layout(example_output), kernel, tensor_args, non_tensor_args, unflatten_args)
            packed.outputs = [packed]
            return packed

class _WaitKernel(_CollectiveKernel):

    def get_volatile_reads(self):
        if False:
            while True:
                i = 10
        inp = self.inputs[0]
        if isinstance(inp, _CollectiveKernel):
            return [inp.inputs[0]]
        elif isinstance(inp, MultiOutput):
            coll = inp.inputs[0]
            assert isinstance(coll, _CollectiveKernel)
            (_, idx) = inp.indices[0]
            return [coll.inputs[idx]]
        else:
            return []

    @classmethod
    def create_wait(cls, kernel, inp: TensorBox) -> None:
        if False:
            print('Hello World!')
        with V.graph.fake_mode:
            (example_output, tensor_args, non_tensor_args, unflatten_args) = cls.process_kernel(kernel, inp)
        packed = cls(NoneLayout(inp.get_device()), kernel, tensor_args, non_tensor_args, unflatten_args)
        MutationOutput(inp.layout, inp, packed)

    def get_read_writes(self):
        if False:
            print('Hello World!')
        read_writes = super().get_read_writes()
        volatile_reads = self.get_volatile_reads()
        for vr in volatile_reads:
            read_writes.reads.add(dependencies.StarDep(vr.get_name()))
        return read_writes

def maybe_free_unbacked_symbols(s):
    if False:
        while True:
            i = 10
    if isinstance(s, (SymTypes, sympy.Expr)):
        return free_unbacked_symbols(s)
    elif isinstance(s, (tuple, list)):
        r = set()
        for t in s:
            r |= maybe_free_unbacked_symbols(t)
        return r
    elif isinstance(s, torch.Tensor):
        return free_unbacked_symbols(s)
    else:
        return set()

class AllToAllSingle(OutOfPlaceCollectiveKernel):

    def __init__(self, layout, inputs, outputs, constant_args, output_split_sizes, input_split_sizes):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(layout, inputs, outputs, constant_args)
        self.output_split_sizes = output_split_sizes
        self.input_split_sizes = input_split_sizes

    def get_unbacked_symbol_uses(self):
        if False:
            print('Hello World!')
        r = set()
        if self.output_split_sizes is not None:
            r |= free_unbacked_symbols(self.output_split_sizes)
        if self.input_split_sizes is not None:
            r |= free_unbacked_symbols(self.input_split_sizes)
        return r

    @classmethod
    def create(cls, x: 'TensorBox', output_split_sizes: Optional[List[Expr]], input_split_sizes: Optional[List[Expr]], tag: str, ranks: List[int], group_size: int):
        if False:
            i = 10
            return i + 15
        inputs = [cls.realize_input(x)]

        def compute_size(new_size):
            if False:
                i = 10
                return i + 15
            if output_split_sizes is not None:
                new_size[0] = sum(output_split_sizes)
        outputs = cls.create_output_buffers(inputs, compute_size)
        layout = MultiOutputLayout(inputs[0].get_device())
        packed = AllToAllSingle(layout=layout, inputs=inputs, outputs=outputs, constant_args=[tag, ranks, group_size], output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes)
        return cls.create_output_nodes(packed, outputs)[0]

    def codegen_collective(self, wrapper, output_name, input_names):
        if False:
            while True:
                i = 10
        (tag, ranks, group_size) = self.constant_args
        wrapper.writeline(f'{output_name}_work = dist.all_to_all_single({output_name}[0], {output_name}_inputs[0], output_split_sizes={self.output_split_sizes}, input_split_sizes={self.input_split_sizes}, group={output_name}_pg, async_op=True)')