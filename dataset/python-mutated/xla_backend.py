from collections import defaultdict
import numpy as np
from .. import _full_sync, tensor
from ..core._imperative_rt import CompNode
from ..core._imperative_rt.core2 import Tensor as RawTensor
from ..core._imperative_rt.core2 import is_external_convert, set_external_convert, set_external_convert_hook, set_py_external_type, unset_external_convert
from ..core._imperative_rt.ops import get_global_rng_seed as _get_global_rng_seed
from ..core._trace_option import set_use_xla_backend
from ..device import get_default_device
from ..logger import get_logger
from ..tensor import Tensor
from ..utils.dlpack import from_dlpack, to_dlpack
from .tracing import trace
try:
    from mge_xlalib.xla_extension import ArrayImpl
    from ..xla.lib import xla_client as xc
except ImportError as e:
    pass
logger = get_logger(__name__)
xla_client_compute_stream = None
_max_reserved_before_compile = None
_all_compilation_completed = False
_expect_xlacompile_cnt = 0
_actual_xlacompile_cnt = 0

def _expect_xlacompile_cnt_minus_one():
    if False:
        print('Hello World!')
    global _expect_xlacompile_cnt
    _expect_xlacompile_cnt -= 1

def apply_external_convert_hook(input, cn):
    if False:
        i = 10
        return i + 15
    stream = xla_client_compute_stream
    assert isinstance(input, ArrayImpl)
    dlpack_capsule = xc._xla.buffer_to_dlpack_managed_tensor(input, take_ownership=True)
    output = from_dlpack(dlpack_capsule, stream).to(cn, _borrow=True)
    return output

class xla_trace(trace):
    """Wraps a callable, and provides accelerated evaluation compiled by xla.
        Currently it is an experimental feature.
        Refer to :class:`~.jit.tracing.trace` for more information.


    Examples:

        .. code-block:: python

            import numpy as np
            from basecls.models.resnet import resnet18
            from megengine.autodiff.grad_manager import GradManager
            from megengine.jit import xla_trace
            from megengine.optimizer import Adam

            model = resnet18()
            gm = GradManager()
            opt = Adam(model.parameters(), lr=1e-4)
            gm.attach(model.parameters())

            # Only tensors in wrapped func args/kwargs will be treated as graph inputs,
            # and other tensors will be captured as const value.
            # Module, optimizer, and train data/label should be arguments of the wrapped function.
            @xla_trace(capture_as_const=True)
            def train_step(model, opt, data, label):
                with gm:
                    pred = model(data)
                    loss = F.loss.cross_entropy(pred, label)
                    gm.backward(loss)
                opt.step().clear_grad()
                return loss

    """
    third_party_backend = True

    def __init__(self, function, *, without_host=True, symbolic_shape=False, **kwargs):
        if False:
            while True:
                i = 10
        assert without_host, 'xla trace only support without host mode'
        assert not symbolic_shape, "xla doesn't support dynamic shape currently"
        set_external_convert_hook(apply_external_convert_hook)
        set_py_external_type(ArrayImpl)
        set_external_convert()
        global _expect_xlacompile_cnt
        _expect_xlacompile_cnt = _expect_xlacompile_cnt + 1
        super().__init__(function, without_host=without_host, symbolic_shape=symbolic_shape, **kwargs)

    def setup_env(self):
        if False:
            return 10
        self.orig_use_xla = set_use_xla_backend(True)

    def unset_env(self):
        if False:
            while True:
                i = 10
        set_use_xla_backend(self.orig_use_xla)

    def get_random_seed(self):
        if False:
            i = 10
            return i + 15
        assert self.has_randomstate == True
        return self.random_seed

    def convert_params_to_xla(self):
        if False:
            print('Hello World!')
        from ..utils.module_utils import get_expand_structure
        from ..tensor import Tensor
        backend = self.xla_exec.backend
        devices = backend.local_devices()
        default_cn = CompNode(get_default_device())
        (_, device_id, _) = default_cn.physical_locator
        device_index = 0 if len(devices) == 0 else [d.id for d in devices].index(device_id)
        device = devices[device_index]
        for (attr, _) in self.attr_to_key.items():
            param = get_expand_structure(attr[0], attr[1])
            param._reset(param.to('cpux'))
        for (tensor, _) in self.opt_param_dict.items():
            tensor._reset(tensor.to('cpux'))

        def as_xla_array(tensor, backend, device):
            if False:
                print('Hello World!')
            np_array = tensor.numpy()
            if np_array.shape == ():
                np_array = np_array[np.newaxis]
            xla_array = backend.buffer_from_pyval(np_array, device)
            tensor._reset(Tensor(xla_array, device=default_cn))
        for (attr, _) in self.attr_to_key.items():
            param = get_expand_structure(attr[0], attr[1])
            as_xla_array(param, backend, device)
        for (tensor, _) in self.opt_param_dict.items():
            as_xla_array(tensor, backend, device)

    def compile(self):
        if False:
            while True:
                i = 10
        from ..xla import build_xla
        from ..tensor import Tensor
        from ..distributed import get_mm_server_addr, is_distributed, get_rank
        from ..device import coalesce_free_memory, get_max_reserved_memory, _get_cuda_left_memory, get_cuda_device_property, _make_free_mem_block_device
        assert self.traced
        global _max_reserved_before_compile, _all_compilation_completed
        if _all_compilation_completed is True:
            logger.warning('using xla in two independent workload in one process, which may cause memory fragment')
        if 'gpu' not in get_default_device():
            logger.warning('should specify default device as `gpu` rather than `xpu` before use xla, if not may cause memory fragment')
        if _max_reserved_before_compile is None:
            _full_sync()
            _max_reserved_before_compile = get_max_reserved_memory(get_default_device())
        coalesce_free_memory()
        _full_sync()
        _make_free_mem_block_device(get_default_device(), int(_get_cuda_left_memory(get_rank()) * 0.9))
        _full_sync()
        (self.tr, self.xla_exec, self.inp_ids, self.out_ids) = build_xla(self, return_with_io=True, return_device_array=True, ip=get_mm_server_addr()[0] if is_distributed() else None, port=get_mm_server_addr()[1] + 1 if is_distributed() else None)
        global _expect_xlacompile_cnt, _actual_xlacompile_cnt
        _actual_xlacompile_cnt += 1
        if self.overall:
            self.convert_params_to_xla()
        coalesce_free_memory()
        _full_sync()
        if _actual_xlacompile_cnt == _expect_xlacompile_cnt:
            _all_compilation_completed = True
            total = get_cuda_device_property(get_rank()).total_memory
            left = _get_cuda_left_memory(get_rank())
            should_left = total - _max_reserved_before_compile
            if left > should_left:
                _make_free_mem_block_device(get_default_device(), left - should_left)
        id2inpidx = defaultdict(list)
        id2outidx = defaultdict(list)
        for (idx, inp_id) in enumerate(self.inp_ids):
            id2inpidx[inp_id].append(idx)
        for (idx, oup_id) in enumerate(self.out_ids):
            id2outidx[oup_id].append(idx)
        self.inpkey2idx = {}
        self.outkey2idx = {}
        if self.tr.has_rng_opr:
            assert self.input_num == len(set(self.inp_ids)) - 1, (self.input_num, len(self.inp_ids))
            self.has_randomstate = True
            default_rng_seed = _get_global_rng_seed()
            high = default_rng_seed >> 32
            low = default_rng_seed & 4294967295
            self.random_seed = Tensor([[high, low], [low, high]], dtype='int32')
        else:
            assert self.input_num == len(set(self.inp_ids)), (self.input_num, len(self.inp_ids))
            self.has_randomstate = False
        inpmark2id = dict()
        outmark2id = dict()
        for var in self.vars:
            if var.kind == 'external':
                for mark in var.inp_mark:
                    inpmark2id[mark] = var.id
            elif var.data_required and var.out_mark:
                for mark in var.out_mark:
                    outmark2id[mark] = var.id
        for (k, v) in inpmark2id.items():
            for idx in id2inpidx[v]:
                self.inpkey2idx[k] = idx
        for (k, v) in outmark2id.items():
            for idx in id2outidx[v]:
                self.outkey2idx[k] = idx

    def prepare_xla_inputs(self, tensors):
        if False:
            for i in range(10):
                print('nop')
        from ..utils.module_utils import get_expand_structure
        inp_count = 0
        inp_list = [0] * self.input_num
        for (idx, t) in enumerate(tensors):
            inp = self.inpkey2idx[self.arg_list[idx]]
            inp_list[inp] = t
            inp_count += 1
        if self.overall:
            for (attr, key) in self.attr_to_key.items():
                param = get_expand_structure(attr[0], attr[1])
                inp = self.inpkey2idx[key]
                inp_list[inp] = param
                inp_count += 1
            for (tensor, k) in self.opt_param_dict.items():
                inp = self.inpkey2idx[k]
                inp_list[inp] = tensor
                inp_count += 1
            opt_hyper_inps = []
            for opt in self.optimizers:
                opt_hyper_inps.extend([Tensor(pg['lr']) for pg in opt.param_groups])
            for (tensor, k) in zip(opt_hyper_inps, self.capture_optimizer_hyper_param):
                inp = self.inpkey2idx[k]
                inp_list[inp] = tensor
                inp_count += 1
        assert inp_count == self.input_num
        if self.has_randomstate:
            inp_list.append(self.random_seed)
        return inp_list

    def to_dlpack(self, x, take_ownership: bool=True):
        if False:
            while True:
                i = 10
        from ..xla.lib import xla_client as xc
        return xc._xla.buffer_to_dlpack_managed_tensor(x, take_ownership=take_ownership)

    def execute(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        from ..tensor import Tensor
        from ..optimizer import Optimizer
        from ..traced_module.pytree import tree_flatten
        from ..utils.module_utils import get_expand_structure
        (inputs, _) = tree_flatten((args, kwargs))
        arrays = []
        cn = CompNode(get_default_device())
        stream = dict(self.xla_exec.backend.get_compute_compnode())
        (device_kind, device_id, stream_id) = cn.physical_locator
        xla_stream = stream[device_id]
        xla_comp_cn = 'gpu{}:{}'.format(device_id, xla_stream)
        self.optimizers = []
        for t in inputs:
            if isinstance(t, RawTensor):
                if not t._is_external_value():
                    assert cn == t.device
                    arrays.append(t.to(xla_comp_cn, _borrow=True))
                else:
                    arrays.append(t)
            if isinstance(t, Optimizer):
                self.optimizers.append(t)
        arrays = self.prepare_xla_inputs(arrays)
        outputs = self.xla_exec(*arrays)
        global xla_client_compute_stream
        xla_client_compute_stream = xla_stream
        return_vals = []
        for i in self.out_list:
            if i == -1:
                if not hasattr(self, 'outdef'):
                    return_vals.append(None)
            else:
                return_vals.append(outputs[self.outkey2idx[i]])
        if not self.out_list:
            return_vals = [None]
        keeped_features = []
        for i in self.keeped_activation:
            keeped_features.append(tensor(outputs[self.outkey2idx[i]], device=cn))
        out_tensors = []
        for array in return_vals:
            if array is not None:
                t = tensor(array, device=cn)
                out_tensors.append(t)
            else:
                out_tensors.append(array)
        if self.overall:
            for (attr, key) in self.update_param_dict.items():
                param = get_expand_structure(attr[0], attr[1])
                xla_array = outputs[self.outkey2idx[key]]
                t = tensor(xla_array, device=cn)
                param._reset(t)
            for (state, key) in self.update_opt_param_dict.items():
                xla_array = outputs[self.outkey2idx[key]]
                t = tensor(xla_array, device=cn)
                state._reset(t)
        elif hasattr(self, 'input_need_update_dict'):
            for (index, out_mark) in self.input_need_update_dict.items():
                inputs[index]._reset(outputs[self.outkey2idx[out_mark]])
        rst = self.outdef.unflatten(out_tensors) if hasattr(self, 'outdef') else out_tensors
        if self.has_randomstate:
            self.random_seed = tensor(outputs[-1], device=cn)
        if keeped_features:
            return (rst, keeped_features)
        else:
            return rst