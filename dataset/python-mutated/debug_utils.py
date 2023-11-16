import copy
import functools
import getpass
import itertools
import logging
import os
import subprocess
import tempfile
import textwrap
from collections import Counter
from importlib import import_module
from typing import Callable, Optional, TypeVar
import torch
import torch._prims_common as utils
import torch._subclasses.meta_utils
from torch._dynamo.testing import rand_strided
from torch._prims_common import is_float_dtype
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._content_store import ContentStoreReader, ContentStoreWriter
from . import config
from .utils import clone_inputs, get_debug_dir
log = logging.getLogger(__name__)
T = TypeVar('T')
inductor_config = import_module('torch._inductor.config')
use_buck = inductor_config.is_fbcode()
if use_buck:
    import libfb.py.build_info
extra_deps = []
extra_imports = ''
if use_buck:
    extra_deps = ['//caffe2/torch/fb/sparsenn:sparsenn_operators_gpu', '//caffe2/torch/fb/sparsenn:sparsenn_operators', '//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu', '//deeplearning/fbgemm/fbgemm_gpu:sparse_ops']
    cur_target = libfb.py.build_info.BuildInfo.get_build_rule().replace('fbcode:', '//')
    extra_imports = '\n'.join([f'torch.ops.load_library("{x}")' for x in extra_deps])
BUCK_CMD_PREFIX = ['buck2', 'run', '@mode/dev-nosan']

class BuckTargetWriter:

    def __init__(self, filename):
        if False:
            for i in range(10):
                print('nop')
        (self.subdir, self.py_file) = os.path.split(os.path.abspath(filename))
        self.target = self.py_file.replace('.py', '')
        self.path = f"{self.subdir.replace('/', '.')}.{self.target}"
        self.path = self.path[self.path.find('fbcode.'):]
        self.path = self.path[7:]
        tmp = self.subdir
        tmp = tmp[tmp.find('fbcode/'):][7:]
        self.cmd_line_path = f'//{tmp}:{self.target}'

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        extra_cpp_deps = '\n'.join([f'        "{x}",' for x in extra_deps])
        return textwrap.dedent(f'\nload("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")\n\npython_binary(\n    name="{self.target}",\n    srcs = ["{self.py_file}"],\n    compile = False,\n    deps = [\n        "//caffe2:torch",\n        "//caffe2/functorch:functorch",\n        "//triton:triton",\n        "{cur_target}",\n    ],\n    cpp_deps = [\n{extra_cpp_deps}\n    ],\n    main_module = "{self.path}",\n)\n')

    def write(self, print_msg=True):
        if False:
            for i in range(10):
                print('nop')
        target_file = os.path.join(self.subdir, 'TARGETS')
        with open(target_file, 'w') as fd:
            fd.write(self.build())
        cmd_split = BUCK_CMD_PREFIX + [self.cmd_line_path]
        if print_msg:
            log.warning('Found an example that reproduces the error. Run this cmd to repro - %s', ' '.join(cmd_split))
        return cmd_split

def minifier_dir():
    if False:
        for i in range(10):
            print('nop')
    path = os.path.join(get_debug_dir(), 'minifier')
    if path is None:
        path = f'{tempfile.gettempdir()}/minifier_{getpass.getuser()}'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path
MAX_CONSTANT_NUMEL_INLINE = 4

class NNModuleToString:
    safe_reprs = [torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.LayerNorm, torch.nn.Dropout, torch.nn.Softmax, torch.nn.ReLU, torch.nn.GELU, torch.nn.Identity, torch.nn.MaxPool2d, torch.nn.Embedding, torch.nn.Tanh, torch.nn.ConvTranspose1d, torch.nn.GLU, torch.nn.LSTM, torch.nn.Flatten, torch.nn.AdaptiveAvgPool2d]

    @staticmethod
    def can_convert_to_string(gm):
        if False:
            i = 10
            return i + 15
        cant_convert = set()
        for (_, module) in gm.named_children():
            if type(module) not in NNModuleToString.safe_reprs:
                cant_convert.add(module)
        if len(cant_convert) > 0:
            log.warning('We have not tested reprs of some modules - %s', cant_convert)
        return True

    @staticmethod
    def convert(gm):
        if False:
            for i in range(10):
                print('nop')
        from torch.nn.modules.module import _addindent
        tab = ' ' * 4
        model_str = textwrap.dedent('\n            from torch.nn import *\n            class Repro(torch.nn.Module):\n                def __init__(self):\n                    super().__init__()\n            ')
        for (module_name, module) in gm.named_children():
            module_str = f'{module.__repr__()}'
            example_param = next(module.parameters(), None)
            if example_param is not None and example_param.is_cuda:
                module_str = f'{module_str}.cuda()'
            model_str += f'{tab * 2}self.{module_name} = {module_str}\n'
        for (buffer_name, buffer) in gm._buffers.items():
            if buffer is None:
                continue
            if buffer.numel() <= MAX_CONSTANT_NUMEL_INLINE:
                from torch._tensor_str import PRINT_OPTS
                assert PRINT_OPTS.threshold >= MAX_CONSTANT_NUMEL_INLINE
                tensor_str = repr(buffer)
            elif torch.is_floating_point(buffer):
                tensor_str = f'torch.randn({list(buffer.shape)}, dtype={buffer.dtype})'
            else:
                tensor_str = f'torch.randint(1, size={list(buffer.shape)}, dtype={buffer.dtype})'
            if buffer.is_cuda:
                tensor_str = f'{tensor_str}.cuda()'
            model_str += f"{tab * 2}self.register_buffer('{buffer_name}', {tensor_str})\n"
        for (param_name, param) in gm._parameters.items():
            if param is None:
                continue
            maybe_device = ''
            if param.is_cuda:
                maybe_device = ', device="cuda"'
            tensor_str = f'torch.nn.Parameter(torch.randn({list(param.shape)}, dtype={param.dtype}{maybe_device}))'
            model_str += f'{tab * 2}self.{param_name} = {tensor_str}\n'
        model_str += f'{_addindent(gm.code, 4)}\n'
        return model_str

@functools.lru_cache(None)
def _cuda_system_info_comment():
    if False:
        print('Hello World!')
    if not torch.cuda.is_available():
        return '# torch.cuda.is_available()==False, no GPU info collected\n'
    model_str = '# CUDA Info: \n'
    try:
        cuda_version_out = subprocess.check_output(['nvcc', '--version'])
        cuda_version_lines = cuda_version_out.decode().split('\n')
        comment = ''.join([f'# {s} \n' for s in cuda_version_lines if s not in ['']])
        model_str += f'{comment}\n'
    except FileNotFoundError:
        model_str += '# nvcc not found\n'
    gpu_names = Counter((torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))
    model_str += '# GPU Hardware Info: \n'
    for (name, count) in gpu_names.items():
        model_str += f'# {name} : {count} \n'
    model_str += '\n'
    return model_str

def generate_config_string(*, stable_output=False):
    if False:
        return 10
    import torch._functorch.config
    import torch._inductor.config
    if stable_output:
        return '# config omitted due to stable_output=True'
    return f'import torch._dynamo.config\nimport torch._inductor.config\nimport torch._functorch.config\nimport torch.fx.experimental._config\n{torch._dynamo.config.codegen_config()}\n{torch._inductor.config.codegen_config()}\n{torch._functorch.config.codegen_config()}\n{torch.fx.experimental._config.codegen_config()}\n'

def get_minifier_repro_path():
    if False:
        return 10
    return os.path.join(minifier_dir(), 'minifier_launcher.py')

def helper_for_dump_minify(contents):
    if False:
        i = 10
        return i + 15
    minified_repro_path = get_minifier_repro_path()
    log.warning('Writing minified repro to:\n%s', minified_repro_path)
    if use_buck:
        BuckTargetWriter(minified_repro_path).write()
    try:
        with open(minified_repro_path, 'w') as fd:
            fd.write(contents)
    except OSError as e:
        log.exception(e)
        raise NotImplementedError('Could not write to {minified_repro_path}') from e

class AccuracyError(Exception):
    pass

def clone_inputs_retaining_gradness(example_inputs):
    if False:
        return 10
    '\n    This clone inputs is different from utils clone_input. In case of minifier,\n    all the tensors are leaf tensors while creating a new graph. So, we set the\n    requires_grad field w/o checking the leafness of the tensor.\n    '
    cloned_inputs = clone_inputs(example_inputs)
    for idx in range(len(example_inputs)):
        if isinstance(cloned_inputs[idx], torch.Tensor):
            cloned_inputs[idx].requires_grad_(example_inputs[idx].requires_grad)
    return cloned_inputs

def run_fwd_maybe_bwd(gm, args, only_fwd=False, disable_clone=False):
    if False:
        while True:
            i = 10
    '\n    Runs a forward and possibly backward iteration for a given mod and args.\n\n    When disable_clone is True, we will use args as-is without cloning.\n    This is higher fidelity but we may destroy the args in the process.\n    '
    from torch._functorch.aot_autograd import make_boxed_func
    from .testing import collect_results, reduce_to_scalar_loss, requires_bwd_pass
    gm = copy.deepcopy(gm)
    if not disable_clone:
        args = clone_inputs_retaining_gradness(args)
    if hasattr(gm, 'zero_grad'):
        gm.zero_grad(True)
    orig_named_parameters = getattr(gm, 'named_parameters', None)
    orig_named_buffers = getattr(gm, 'named_buffers', None)
    if not hasattr(gm, '_boxed_call') and (orig_named_parameters is not None or orig_named_buffers is not None):
        gm = make_boxed_func(gm)
        if orig_named_parameters is not None:
            gm.named_parameters = orig_named_parameters
        if orig_named_buffers is not None:
            gm.named_buffers = orig_named_buffers
    out = gm(args)
    if only_fwd:
        return out
    if requires_bwd_pass(out):
        loss = reduce_to_scalar_loss(out)
        loss.backward()
    return collect_results(gm, out, None, args)

def same_two_models(gm, opt_gm, example_inputs, only_fwd=False, *, require_fp64=False, ignore_non_fp=False):
    if False:
        while True:
            i = 10
    '\n    Check two models have same accuracy.\n\n    require_fp64: if True, raise an error if we unable to calculate the fp64 reference\n    ignore_non_fp: if True, do not compare outputs which are not floating point.  This\n        is mostly useful for the minifier (which wants to avoid quantizing floating point\n        error into integer/boolean error)\n    '
    from .eval_frame import OptimizedModule
    from .testing import named_buffers_for_optimized_module, named_parameters_for_optimized_module
    from .utils import same
    if isinstance(gm, OptimizedModule):
        gm.named_parameters = named_parameters_for_optimized_module(gm)
        gm.named_buffers = named_buffers_for_optimized_module(gm)
    if isinstance(opt_gm, OptimizedModule):
        opt_gm.named_parameters = named_parameters_for_optimized_module(opt_gm)
        opt_gm.named_buffers = named_buffers_for_optimized_module(opt_gm)
    ref = run_fwd_maybe_bwd(gm, example_inputs, only_fwd)
    fp64_ref = None
    if config.same_two_models_use_fp64:
        try:
            (fp64_model, fp64_examples) = cast_to_fp64(copy.deepcopy(gm), clone_inputs_retaining_gradness(example_inputs))
            fp64_ref = run_fwd_maybe_bwd(fp64_model, fp64_examples, only_fwd)
        except Exception:
            if require_fp64:
                raise RuntimeError('Could not generate fp64 outputs')
            log.warning('Could not generate fp64 outputs')
    try:
        res = run_fwd_maybe_bwd(opt_gm, example_inputs, only_fwd)
    except Exception as e:
        log.exception('While minifying the program in accuracy minification mode, ran into a runtime exception which is likely an unrelated issue. Skipping this graph.')
        return True
    passing = same(ref, res, fp64_ref, tol=config.repro_tolerance, equal_nan=True, ignore_non_fp=ignore_non_fp)
    return passing

def cast_dtype_args_to_fp64(model):
    if False:
        for i in range(10):
            print('nop')
    for node in model.graph.nodes:
        if node.op == 'call_function' and node.target == torch.ops.prims.convert_element_type.default:
            assert len(node.args) == 2
            if is_float_dtype(node.args[1]) and node.args[1] != torch.float64:
                node.args = (node.args[0], torch.float64)
        if node.op == 'call_function':
            dtype = node.kwargs.get('dtype')
            if dtype is not None and is_float_dtype(dtype):
                new_kwargs = dict(node.kwargs)
                new_kwargs['dtype'] = torch.float64
                node.kwargs = new_kwargs
    model.graph.lint()
    model.recompile()
    return model

def cast_to(dtype, model, inputs):
    if False:
        return 10
    from torch.utils._pytree import tree_map
    model = model.to(dtype)
    if dtype == torch.float64:
        model = cast_dtype_args_to_fp64(model)
    inputs = tree_map(lambda x: x.to(dtype) if isinstance(x, torch.Tensor) and x.is_floating_point() else x, inputs)
    return (model, inputs)

def cast_to_fp64(model, inputs):
    if False:
        while True:
            i = 10
    return cast_to(torch.float64, model, inputs)

def backend_accuracy_fails(gm, example_inputs, compiler_fn, only_fwd=False, *, require_fp64=False, ignore_non_fp=False):
    if False:
        i = 10
        return i + 15
    try:
        compiled_gm = compiler_fn(copy.deepcopy(gm), clone_inputs_retaining_gradness(example_inputs))
        return not same_two_models(gm, compiled_gm, example_inputs, only_fwd, require_fp64=require_fp64, ignore_non_fp=ignore_non_fp)
    except Exception as e:
        log.exception('While minifying the program in accuracy minification mode, ran into a runtime exception which is likely an unrelated issue. Skipping this graph')
        return False

def _stride_or_default(stride: Optional['torch._prims_common.StrideType'], *, shape: 'torch._prims_common.ShapeType') -> 'torch._prims_common.StrideType':
    if False:
        i = 10
        return i + 15
    return stride if stride is not None else utils.make_contiguous_strides_for(shape)

def _mk_defaulter(d: T) -> Callable[[Optional[T]], T]:
    if False:
        for i in range(10):
            print('nop')
    return lambda x: x if x is not None else d
_dtype_or_default = _mk_defaulter(torch.float32)
_device_or_default = _mk_defaulter(torch.device('cpu'))
_storage_offset_or_default = _mk_defaulter(0)
_requires_grad_or_default = _mk_defaulter(False)
_is_leaf_or_default = _mk_defaulter(False)

class NopInputReader:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.total = 0

    def storage(self, storage_hash, nbytes, *, device=None, dtype_hint=None):
        if False:
            while True:
                i = 10
        self.total += 1

    def tensor(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        pass

    def symint(self, *args, **kwargs):
        if False:
            print('Hello World!')
        pass

class InputReader:

    def __init__(self, save_dir=None, *, pbar=None):
        if False:
            for i in range(10):
                print('nop')
        if save_dir is None:
            log.warning('no save_dir specified, will generate random data')
        self.store = ContentStoreReader(save_dir) if save_dir is not None else None
        self.args = []
        self.pbar = pbar

    def storage(self, storage_hash, nbytes, *, device=None, dtype_hint=None):
        if False:
            for i in range(10):
                print('nop')
        if self.pbar is not None:
            self.pbar.update(1)
        device = _device_or_default(device)
        dtype_hint = _dtype_or_default(dtype_hint)
        if self.store is not None and storage_hash is not None:
            try:
                storage = self.store.read_storage(storage_hash)
            except FileNotFoundError:
                pass
            else:
                if device != storage.device:
                    log.warning('device mismatch: %s != %s', device, storage.device)
                return storage
        log.warning('could not load %s, generating random data instead', storage_hash)
        shape = (nbytes // dtype_hint.itemsize,)
        stride = _stride_or_default(None, shape=shape)
        return rand_strided(shape, stride, dtype_hint, device).untyped_storage()

    def tensor(self, storage, shape, stride=None, *, storage_offset=None, dtype=None, requires_grad=None, is_leaf=None, **metadata):
        if False:
            print('Hello World!')
        stride = _stride_or_default(stride, shape=shape)
        storage_offset = _storage_offset_or_default(storage_offset)
        dtype = _dtype_or_default(dtype)
        is_leaf = _is_leaf_or_default(is_leaf)
        requires_grad = _requires_grad_or_default(requires_grad)
        t = torch.tensor([], dtype=dtype, device=storage.device, requires_grad=requires_grad)
        with torch.no_grad():
            t.set_(storage, storage_offset, shape, stride)
        if not is_leaf:
            with torch.enable_grad():
                t = t.clone(memory_format=torch.preserve_format)
            with torch.no_grad():
                t.set_(storage, storage_offset, shape, stride)
        assert torch._subclasses.meta_utils.safe_is_leaf(t) == is_leaf
        torch._utils.set_tensor_metadata(t, metadata)
        self.args.append(t)
        return t

    def symint(self, val):
        if False:
            i = 10
            return i + 15
        self.args.append(val)
        return val

class InputWriter:

    def __init__(self, save_dir, *, stable_hash=False):
        if False:
            print('Hello World!')
        self._lines = []
        self.storage_counter = itertools.count()
        self.save_dir = save_dir
        self.store = ContentStoreWriter(save_dir, stable_hash=stable_hash) if save_dir is not None else None
        self.seen_storages = {}

    def lines(self):
        if False:
            print('Hello World!')
        r = ['def load_args(reader):']
        r.extend((f'    {l}' for l in self._lines))
        r.append('load_args._version = 0')
        return r

    def storage(self, untyped_storage, *, dtype_hint=None, device_hint=None) -> str:
        if False:
            i = 10
            return i + 15
        ws = StorageWeakRef(untyped_storage)
        v = self.seen_storages.get(ws)
        if v is not None:
            return v
        v = f'buf{next(self.storage_counter)}'
        maybe_dtype_hint = ''
        if _dtype_or_default(None) != _dtype_or_default(dtype_hint):
            maybe_dtype_hint = f', dtype_hint={dtype_hint!r}'
        maybe_device = ''
        device = untyped_storage.device
        if device.type == 'meta':
            assert device_hint is not None
            device = device_hint
        if _device_or_default(None) != device:
            maybe_device = f', device={device!r}'
        nbytes = untyped_storage.nbytes()
        storage_hash = None
        if self.store is not None and untyped_storage.device.type != 'meta':
            storage_hash = self.store.write_storage(untyped_storage)
        self._lines.append(f'{v} = reader.storage({storage_hash!r}, {nbytes!r}{maybe_device}{maybe_dtype_hint})')
        self.seen_storages[ws] = v
        return v

    def tensor(self, name, t) -> None:
        if False:
            return 10
        storage = self.storage(t.untyped_storage(), dtype_hint=t.dtype, device_hint=t.device)
        args = []
        if _stride_or_default(None, shape=t.shape) != t.stride():
            args.append(str(tuple(t.stride())))
        if _dtype_or_default(None) != t.dtype:
            args.append(f'dtype={t.dtype!r}')
        if _storage_offset_or_default(None) != t.storage_offset():
            args.append(f'storage_offset={t.storage_offset()!r}')
        tensor_metadata = torch._utils.get_tensor_metadata(t)
        if tensor_metadata:
            args.extend((f'{k}={v!r}' for (k, v) in tensor_metadata.items()))
        if _requires_grad_or_default(None) != t.requires_grad:
            args.append(f'requires_grad={t.requires_grad!r}')
        is_leaf = torch._subclasses.meta_utils.safe_is_leaf(t)
        if _is_leaf_or_default(None) != is_leaf:
            args.append(f'is_leaf={is_leaf!r}')
        self._lines.append('reader.tensor(' + ', '.join([storage, str(tuple(t.shape)), *args]) + f')  # {name}')

    def symint(self, name, val) -> None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(val, torch.SymInt):
            val = val.node.hint
        self._lines.append(f'reader.symint({val!r})  # {name}')