import builtins
import copy
import functools
import hashlib
import inspect
import json
import logging
import math
import operator
import os
import os.path
import re
import threading
from enum import auto, Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import torch
import torch.autograd.profiler as autograd_profiler
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import dynamo_timed
from torch.utils._triton import has_triton, has_triton_package
from . import config
from .codecache import cache_dir, CudaKernelParamCache
from .coordinate_descent_tuner import CoordescTuner
from .ir import ReductionHint, TileHint
from .utils import ceildiv, conditional_product, create_bandwidth_info_str, do_bench, get_num_bytes, next_power_of_2, triton_config_to_hashable
log = logging.getLogger(__name__)
if has_triton_package():
    import triton
    from triton import Config
    from triton.runtime.autotuner import OutOfResources
    from triton.runtime.jit import KernelInterface
else:
    Config = object
    triton = None
    KernelInterface = object
    OutOfResources = object
if has_triton():
    from triton.runtime.jit import get_cuda_stream
else:
    get_cuda_stream = None
_NUM_THREADS_PER_WARP = 32

class HeuristicType(Enum):
    POINTWISE = auto()
    REDUCTION = auto()
    PERSISTENT_REDUCTION = auto()
    TEMPLATE = auto()
    USER_AUTOTUNE = auto()

class AutotuneHint(Enum):
    ELEMENTS_PER_WARP_32 = 0
    __repr__ = Enum.__str__

def autotune_hints_to_configs(hints: Set[AutotuneHint], size_hints, block_size) -> List[Config]:
    if False:
        print('Hello World!')
    '\n    AutotuneHints can be attached to the metadata of triton kernels for providing\n    suggestions about what to try for autotuning. One reason to do this is if there are\n    some configs that are only useful in specific scenarios, in which case we can avoid\n    wasting compile time on autotuning unless we know we are in one of those scenarios.\n\n    Based on those hints, this function will generate a list of additional autotuning\n    configs to try.\n    '
    xyz_options: Tuple[Tuple[Any, ...], ...]
    configs = []
    for hint in hints:
        if hint == AutotuneHint.ELEMENTS_PER_WARP_32:
            if len(size_hints) == 1:
                xyz_options = ((block_size // 4,),)
            elif len(size_hints) == 2:
                xyz_options = ((block_size // 4, 1), (1, block_size // 4))
            elif len(size_hints) == 3:
                xyz_options = ((block_size // 4, 1, 1), (1, block_size // 4, 1), (1, 1, block_size // 4))
            for xyz in xyz_options:
                configs.append(triton_config(size_hints, *xyz, num_elements_per_warp=32))
    return configs

def disable_pointwise_autotuning():
    if False:
        return 10
    if torch.are_deterministic_algorithms_enabled():
        return True
    return not config.triton.autotune_pointwise

class CachingAutotuner(KernelInterface):
    """
    Simplified version of Triton autotuner that has no invalidation
    key and caches the best config to disk to improve cold start times.
    Unlike the main triton Autotuner, this version can precompile all
    configs, and does not rely on the Triton JIT.
    """

    def __init__(self, fn, triton_meta, configs, save_cache_hook, mutated_arg_names, heuristic_type, size_hints=None, inductor_meta=None):
        if False:
            return 10
        super().__init__()
        self.fn = fn
        self.triton_meta = triton_meta
        self.inductor_meta = {} if inductor_meta is None else inductor_meta
        self.save_cache_hook = save_cache_hook
        self.mutated_arg_names = mutated_arg_names
        self.configs = configs
        self.heuristic_type = heuristic_type
        if log.isEnabledFor(logging.DEBUG):
            log.debug('CachingAutotuner gets %d configs', len(self.configs))
            for c in self.configs:
                log.debug(c)
        self.launchers = []
        self.lock = threading.Lock()
        if os.getenv('TRITON_CACHE_DIR') is None:
            os.environ['TRITON_CACHE_DIR'] = os.path.join(cache_dir(), 'triton', str(self.triton_meta.get('device', 0)))
        self.size_hints = size_hints
        self.coordesc_tuner = CoordescTuner(is_mm=False, name=self.fn.__name__, size_hints=size_hints)
        self.record_function_ctx = torch._C._profiler._RecordFunctionFast(self.inductor_meta.get('kernel_name', 'triton kernel'))

    def precompile(self, warm_cache_only_with_cc=None):
        if False:
            print('Hello World!')
        with self.lock:
            if self.launchers:
                return
            self.launchers = []
            compiled_binaries = []
            for c in self.configs:
                try:
                    (compiled_binary, launcher) = self._precompile_config(c, warm_cache_only_with_cc)
                except OutOfResources:
                    continue
                self.launchers.append(launcher)
                compiled_binaries.append(compiled_binary)
            if len(self.launchers) == 0:
                raise RuntimeError('No valid triton configs. Report a fatal compilation error')
            seen_configs = set(self.configs)
            device_interface = get_interface_for_device('cuda')
            device_prop = device_interface.Worker.get_device_properties(self.triton_meta['device'])
            if config.dynamic_scale_rblock and self.heuristic_type == HeuristicType.REDUCTION and (self.size_hints is not None) and (device_prop.major == 8):
                for (triton_config, compiled_binary) in zip(self.configs, compiled_binaries):
                    assert len(self.size_hints) == 2
                    xblock = triton_config.kwargs['XBLOCK']
                    rblock = triton_config.kwargs['RBLOCK']
                    total_block = (self.size_hints[0] + xblock - 1) // xblock
                    nreg = getattr(compiled_binary, 'n_regs', None)
                    if nreg is None:
                        continue
                    if rblock <= 64:
                        continue
                    if nreg <= 65536 // device_prop.max_threads_per_multi_processor:
                        continue
                    nreg_per_warp = nreg * 32
                    nreg_per_block = nreg_per_warp * triton_config.num_warps
                    max_blocks_per_sm = max(65536 // nreg_per_block, 1)
                    if total_block <= max_blocks_per_sm * device_prop.multi_processor_count:
                        continue
                    new_config = copy.deepcopy(triton_config)
                    new_config.kwargs['RBLOCK'] = rblock // 2
                    if new_config in seen_configs:
                        continue
                    seen_configs.add(new_config)
                    self.launchers.append(self._precompile_config(new_config, warm_cache_only_with_cc)[1])
            self.configs = None

    def _precompile_config(self, cfg: Config, warm_cache_only_with_cc: Optional[int]):
        if False:
            while True:
                i = 10
        'Ahead of time compile a given autotuner config.'
        compile_meta = copy.deepcopy(self.triton_meta)
        for (k, v) in cfg.kwargs.items():
            compile_meta['constants'][self.fn.arg_names.index(k)] = v
        compile_meta['num_warps'] = cfg.num_warps
        compile_meta['num_stages'] = cfg.num_stages
        compile_meta['debug'] = config.assert_indirect_indexing and torch.version.hip is None
        compile_meta['device_type'] = 'cuda' if torch.version.hip is None else 'hip'
        if warm_cache_only_with_cc:
            return (triton.compile(self.fn, warm_cache_only=True, cc=warm_cache_only_with_cc, **compile_meta), None)
        with torch.cuda.device(compile_meta['device']):
            torch.cuda.synchronize(torch.cuda.current_device())
            binary = triton.compile(self.fn, **compile_meta)
            binary._init_handles()
        call_args = [arg for (i, arg) in enumerate(self.fn.arg_names) if i not in self.fn.constexprs]
        def_args = list(self.fn.arg_names)
        while def_args and def_args[-1] in cfg.kwargs:
            def_args.pop()
        scope = {'grid_meta': cfg.kwargs, 'bin': binary, 'torch': torch, 'set_device': torch.cuda.set_device, 'current_device': torch.cuda.current_device}
        exec(f"""\n            def launcher({', '.join(def_args)}, grid, stream):\n                if callable(grid):\n                    grid_0, grid_1, grid_2 = grid(grid_meta)\n                else:\n                    grid_0, grid_1, grid_2 = grid\n\n                if hasattr(bin, "num_ctas"):\n                    bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps,\n                                bin.num_ctas, *bin.clusterDims, bin.shared,\n                                stream, bin.cu_function, None, None, None,\n                                {', '.join(call_args)})\n                else:\n                    bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps, bin.shared,\n                                stream, bin.cu_function, None, None, None,\n                                {', '.join(call_args)})\n                return bin\n            """.lstrip(), scope)
        launcher = scope['launcher']
        launcher.config = cfg
        launcher.n_regs = getattr(binary, 'n_regs', None)
        launcher.n_spills = getattr(binary, 'n_spills', None)
        launcher.shared = getattr(binary, 'shared', None)
        launcher.store_cubin = config.triton.store_cubin
        if launcher.store_cubin:
            launcher.fn = self.fn
            launcher.bin = binary
        return (binary, launcher)

    def bench(self, launcher, *args, grid, **kwargs):
        if False:
            i = 10
            return i + 15
        'Measure the performance of a given launcher'
        if launcher.n_spills > config.triton.spill_threshold:
            log.debug('Skip config %s because of register spilling: %d', launcher.config, launcher.n_spills)
            return float('inf')
        stream = get_cuda_stream(torch.cuda.current_device())

        def kernel_call():
            if False:
                for i in range(10):
                    print('nop')
            if launcher.config.pre_hook is not None:
                launcher.config.pre_hook({**dict(zip(self.arg_names, args)), **launcher.config.kwargs})
            (cloned_args, cloned_kwargs) = self.clone_args(*args, **kwargs)
            launcher(*cloned_args, **cloned_kwargs, grid=grid, stream=stream)
        return do_bench(kernel_call, rep=40, fast_flush=True)

    def clone_args(self, *args, **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        if False:
            while True:
                i = 10
        from .compile_fx import clone_preserve_strides
        cloned_args = []
        for (i, arg) in enumerate(args):
            if self.fn.arg_names[i] in self.mutated_arg_names:
                assert isinstance(arg, torch.Tensor)
                cloned_args.append(clone_preserve_strides(arg))
            else:
                cloned_args.append(arg)
        cloned_kwargs: Dict[str, Any] = {}
        for (name, arg) in kwargs.items():
            if name in self.mutated_arg_names:
                assert isinstance(arg, torch.Tensor)
                cloned_kwargs[name] = clone_preserve_strides(arg)
            else:
                cloned_kwargs[name] = arg
        return (cloned_args, cloned_kwargs)

    @dynamo_timed
    def benchmark_all_configs(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        timings = {launcher: self.bench(launcher, *args, **kwargs) for launcher in self.launchers}
        for (k, v) in timings.items():
            self.coordesc_tuner.cache_benchmark_result(k.config, v)
        if log.isEnabledFor(logging.DEBUG):
            log.debug('Benchmark all input configs get:')
            for (k, v) in timings.items():
                log.debug('%s: %f, nreg %d, nspill %d, #shared-mem %d', k.config, v, k.n_regs, k.n_spills, k.shared)
        return timings

    def autotune_to_one_config(self, *args, **kwargs):
        if False:
            return 10
        'Do the actual autotuning'
        timings = self.benchmark_all_configs(*args, **kwargs)
        self.launchers = [builtins.min(timings, key=timings.get)]
        if self.save_cache_hook:
            self.save_cache_hook(self.launchers[0].config)

    def save_cuda_kernel(self, grid, stream, launcher):
        if False:
            for i in range(10):
                print('nop')
        if callable(grid):
            (grid_x, grid_y, grid_z) = grid(launcher.config.kwargs)
        else:
            (grid_x, grid_y, grid_z) = grid
        key = self.inductor_meta.get('kernel_name', None)
        assert key is not None, 'kernel_name can not be None'
        params = {'mangled_name': launcher.bin.metadata['name'], 'grid_x': grid_x, 'grid_y': grid_y, 'grid_z': grid_z, 'x_block': launcher.config.kwargs.get('XBLOCK', 1), 'y_block': launcher.config.kwargs.get('YBLOCK', None), 'z_block': launcher.config.kwargs.get('ZBLOCK', None), 'num_warps': launcher.bin.num_warps, 'shared_mem': launcher.bin.shared, 'stream': stream, 'meta': launcher.config.kwargs}
        CudaKernelParamCache.set(key, params, launcher.bin.asm['cubin'])

    def coordinate_descent_tuning(self, launcher, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Coordinate descent tuning can be run with or without max-autotune.\n\n        The only difference between these two is the starting config for coordinate_descent tuning.\n        E.g., assuming regular autotune only get one config C1; while max-autotune get 4 configs C1, C2, C3, C4\n        and max-autotune figure out C3 is the best.\n\n        Then if coordinate descnt tuning is run with max-autotune disabled, it will start from C1;\n        while if coordinate descent tuning is run with max-autotune enabled, it will start from C3.\n        '
        if self.heuristic_type == HeuristicType.TEMPLATE or self.heuristic_type == HeuristicType.USER_AUTOTUNE:
            return launcher
        (cloned_args, _) = self.clone_args(*args)
        config2launcher = {launcher.config: launcher}

        def benchmark_one_config(config):
            if False:
                for i in range(10):
                    print('nop')
            with self.lock:
                (_, launcher) = self._precompile_config(config, None)
            config2launcher[config] = launcher
            out = self.bench(launcher, *cloned_args, **kwargs)
            log.debug('COORDESC: %s: %f, nreg %d, nspill %d, #shared-mem %d', launcher.config, out, launcher.n_regs, launcher.n_spills, launcher.shared)
            return out
        assert not (self.heuristic_type == HeuristicType.PERSISTENT_REDUCTION and 'RBLOCK' in launcher.config.kwargs), "Coordinate descent tuner relies on the assumption that persistent reduction's triton config does not have RBLOCK"
        best_config = self.coordesc_tuner.autotune(benchmark_one_config, launcher.config, None)
        best_config.found_by_coordesc = True
        if self.save_cache_hook:
            self.save_cache_hook(best_config, found_by_coordesc=True)
        return config2launcher.get(best_config)

    def run(self, *args, grid, stream, **kwargs):
        if False:
            return 10
        if len(self.launchers) != 1:
            if len(self.launchers) == 0:
                self.precompile()
            if len(self.launchers) > 1:
                self.autotune_to_one_config(*args, grid=grid, **kwargs)
        if not getattr(self.launchers[0].config, 'found_by_coordesc', False) and config.coordinate_descent_tuning:
            self.launchers = [self.coordinate_descent_tuning(self.launchers[0], *args, grid=grid, **kwargs)]
        (launcher,) = self.launchers
        if launcher.store_cubin:
            self.save_cuda_kernel(grid, stream, launcher)
        if launcher.config.pre_hook is not None:
            launcher.config.pre_hook({**dict(zip(self.arg_names, args)), **launcher.config.kwargs, **kwargs})
        if autograd_profiler._is_profiler_enabled:
            with self.record_function_ctx:
                return launcher(*args, **kwargs, grid=grid, stream=stream)
        else:
            return launcher(*args, **kwargs, grid=grid, stream=stream)

def _find_names(obj):
    if False:
        i = 10
        return i + 15
    import gc
    import inspect
    frame = inspect.currentframe()
    while frame is not None:
        frame.f_locals
        frame = frame.f_back
    obj_names = []
    for referrer in gc.get_referrers(obj):
        if isinstance(referrer, dict):
            for (k, v) in referrer.items():
                if v is obj:
                    obj_names.append(k)
    return obj_names
collected_calls: List[Any] = []

def start_graph():
    if False:
        return 10
    collected_calls.clear()

def end_graph():
    if False:
        i = 10
        return i + 15
    if len(collected_calls) == 0:
        return
    overall_time = sum((call[0] for call in collected_calls))
    overall_gb = sum((call[1] for call in collected_calls))
    cur_file = inspect.stack()[1].filename
    print(f'SUMMARY ({cur_file})')
    print(f'{overall_time:.2f}ms   \t {overall_gb:.2f} GB\t {overall_gb / (overall_time / 1000.0):.2f}GB/s')
    print()

class DebugAutotuner(CachingAutotuner):

    def __init__(self, *args, regex_filter='', **kwargs):
        if False:
            print('Hello World!')
        self.regex_filter = regex_filter
        super().__init__(*args, **kwargs)
        self.cached = None

    def run(self, *args, grid, stream):
        if False:
            return 10
        possible_names = _find_names(self)
        kernel_name = f'{max(possible_names, key=len)}'
        if not re.match(self.regex_filter, kernel_name):
            return
        super().run(*args, grid=grid, stream=stream)
        (launcher,) = self.launchers
        if self.cached is None:
            ms = self.bench(launcher, *args, grid=grid)
            num_in_out_ptrs = len([arg_name for arg_name in self.fn.arg_names if arg_name.startswith('in_out_ptr')])
            num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1000000000.0
            gb_per_s = num_gb / (ms / 1000.0)
            self.cached = (ms, num_gb, gb_per_s, kernel_name)
        else:
            (ms, num_gb, gb_per_s, kernel_name) = self.cached
        collected_calls.append((ms, num_gb, gb_per_s, kernel_name))
        print(create_bandwidth_info_str(ms, num_gb, gb_per_s, suffix=f' \t {kernel_name}'))

def hash_configs(configs: List[Config]):
    if False:
        while True:
            i = 10
    '\n    Hash used to check for changes in configurations\n    '
    hasher = hashlib.sha256()
    for cfg in configs:
        hasher.update(f'{sorted(cfg.kwargs.items())} {cfg.num_warps} {cfg.num_stages}\n'.encode())
    return hasher.hexdigest()

def load_cached_autotuning(cache_filename: str, configs_hash: str, configs: List[Config]):
    if False:
        return 10
    '\n    Read a cached autotuning result from disk\n    '
    if not os.path.exists(cache_filename):
        return None
    with open(cache_filename) as fd:
        best_config = json.loads(fd.read())
    if best_config.pop('configs_hash', None) != configs_hash:
        return None
    if config.coordinate_descent_tuning and best_config.pop('found_by_coordesc', False):
        num_warps = best_config.pop('num_warps')
        num_stages = best_config.pop('num_stages')
        triton_config = Config(best_config, num_warps=num_warps, num_stages=num_stages)
        triton_config.found_by_coordesc = True
        return triton_config
    matching_configs = [cfg for cfg in configs if all((val == best_config.get(key) for (key, val) in cfg.kwargs.items())) and cfg.num_warps == best_config.get('num_warps') and (cfg.num_stages == best_config.get('num_stages'))]
    if len(matching_configs) != 1:
        return None
    return matching_configs[0]

def cached_autotune(size_hints: Optional[List[int]], configs: List[Config], triton_meta, heuristic_type, filename=None, inductor_meta=None):
    if False:
        return 10
    '\n    A copy of triton.autotune that calls our subclass.  Our subclass\n    has additional debugging, error handling, and on-disk caching.\n    '
    configs = unique_configs(configs)
    assert len(configs) == 1 or filename
    save_cache_hook: Optional[Callable[[Any, Any], Any]]
    inductor_meta = {} if inductor_meta is None else inductor_meta
    if filename is not None and (len(configs) > 1 or config.coordinate_descent_tuning):
        cache_filename = os.path.splitext(filename)[0] + '.best_config'
        configs_hash = hash_configs(configs)
        best_config = load_cached_autotuning(cache_filename, configs_hash, configs)
        if best_config:
            configs = [best_config]

        def save_cache_hook(cfg, found_by_coordesc=False):
            if False:
                return 10
            with open(cache_filename, 'w') as fd:
                fd.write(json.dumps({**cfg.kwargs, 'num_warps': cfg.num_warps, 'num_stages': cfg.num_stages, 'configs_hash': configs_hash, 'found_by_coordesc': found_by_coordesc}))
            if log.isEnabledFor(logging.DEBUG):
                type_str = 'coordesc' if found_by_coordesc else 'heuristic'
                log.debug('Save %s tuning result to %s', type_str, cache_filename)
    else:
        save_cache_hook = None
    mutated_arg_names = inductor_meta.pop('mutated_arg_names', ())

    def decorator(fn):
        if False:
            for i in range(10):
                print('nop')
        import inspect
        if 'XBLOCK' not in inspect.signature(fn.fn).parameters:
            for tconfig in configs:
                if 'XBLOCK' in tconfig.kwargs:
                    assert tconfig.kwargs['XBLOCK'] == 1
                    tconfig.kwargs.pop('XBLOCK')
        if config.profile_bandwidth:
            return DebugAutotuner(fn, triton_meta=triton_meta, inductor_meta=inductor_meta, regex_filter=config.profile_bandwidth_regex, configs=configs, save_cache_hook=save_cache_hook, mutated_arg_names=mutated_arg_names, heuristic_type=heuristic_type, size_hints=size_hints)
        return CachingAutotuner(fn, triton_meta=triton_meta, inductor_meta=inductor_meta, configs=configs, save_cache_hook=save_cache_hook, mutated_arg_names=mutated_arg_names, heuristic_type=heuristic_type, size_hints=size_hints)
    return decorator

def unique_configs(configs: List[Config]):
    if False:
        while True:
            i = 10
    'Remove duplicate configurations'
    seen = set()
    pruned_configs = []
    for cfg in configs:
        key = triton_config_to_hashable(cfg)
        if key not in seen:
            seen.add(key)
            pruned_configs.append(cfg)
    return pruned_configs

def check_config(cfg, *, xnumel=None, ynumel=None, znumel=None):
    if False:
        i = 10
        return i + 15
    for (numel, label) in zip((xnumel, ynumel, znumel), 'XYZ'):
        if numel is None:
            continue
        block = cfg[f'{label}BLOCK']
        if numel == 1:
            assert block == 1, f'TritonKernel.indexing assumes numel == 1 => BLOCK == 1 but {label.lower()}numel=={numel} and {label}BLOCK={block} (cfg={cfg}).'
        max_block = config.triton.max_block[label]
        max_block_str = f'config.triton.max_block["{label}"]'
        assert max_block % block == 0, f'TritonKernel.indexing assumes {label}BLOCK divides {max_block_str} but {label}BLOCK={block} and {max_block_str}={max_block} (cfg={cfg}).'

def triton_config(size_hints, x, y=None, z=None, num_stages=1, num_elements_per_warp=256, min_elem_per_thread=0) -> Config:
    if False:
        print('Hello World!')
    "\n    Construct a pointwise triton config with some adjustment heuristics\n    based on size_hints. Size_hints is a tuple of numels in each tile\n    dimension and will be rounded up to the nearest power of 2.\n\n    num_elements_per_warp is a suggestion for controlling how many warps\n    the triton config should contain. e.g.: if x=16, y=8, z=4 then\n    num_elements = 16*8*4 = 512. Then if we set num_elements_per_warp=128,\n    we'll launch 512 (elem) / 128 (elem/warp) = 4 warps. Note that it's\n    just a suggestion, and sometimes other adjustment heuristics will\n    override the num_elements_per_warp.\n\n    min_elem_per_thread controls the minimum number of elements\n    processed by each thread. It's always enforced.\n    "
    size_hints = list(reversed(size_hints))
    maxGridSize = [2147483647, 65535, 65535]
    target = conditional_product(x, y, z)
    if conditional_product(*size_hints) < target:
        target //= 8
    x = min(x, size_hints[0])
    if y:
        y = min(y, size_hints[1])
    if z:
        z = min(z, size_hints[2])
    while x < min(size_hints[0], config.triton.max_block['X']) and (x * maxGridSize[0] < size_hints[0] or conditional_product(x, y, z) < target):
        x *= 2
    while y and y < min(size_hints[1], config.triton.max_block['Y']) and (y * maxGridSize[1] < size_hints[1] or conditional_product(x, y, z) < target):
        y *= 2
    while z and z < min(size_hints[2], config.triton.max_block['Z']) and (z * maxGridSize[2] < size_hints[2] or conditional_product(x, y, z) < target):
        z *= 2
    num_warps = next_power_of_2(min(max(conditional_product(x, y, z) // num_elements_per_warp, 1), 8))
    num_warps = max(num_warps, 4) if conditional_product(x, y, z) >= 128 else num_warps
    xnumel = size_hints[0]
    ynumel = size_hints[1] if y else None
    znumel = size_hints[2] if z else None
    block_size = max(conditional_product(x, y, z), min_elem_per_thread * _NUM_THREADS_PER_WARP * num_warps)
    x *= math.ceil(block_size / conditional_product(x, y, z))
    cfg = {'XBLOCK': x}
    if y:
        cfg['YBLOCK'] = y
    if z:
        cfg['ZBLOCK'] = z
    check_config(cfg, xnumel=xnumel, ynumel=ynumel, znumel=znumel)
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)

def triton_config_reduction(size_hints, x, r, num_stages=1, num_warps=None) -> Config:
    if False:
        while True:
            i = 10
    '\n    Construct a reduction triton config with some adjustment heuristics\n    based on size_hints. Size_hints is a tuple of numels in each tile\n    dimension and will be rounded up to the nearest power of 2.\n    '
    target = conditional_product(x, r)
    if conditional_product(*size_hints) < target:
        target //= 8
    x = min(x, size_hints[0])
    r = min(r, size_hints[1])
    while x < size_hints[0] and conditional_product(x, r) < target:
        x *= 2
    while r < size_hints[1] and conditional_product(x, r) < target:
        r *= 2
    cfg = {'XBLOCK': x, 'RBLOCK': r}
    if num_warps is None:
        num_warps = conditional_product(x, r) // 128
    num_warps = next_power_of_2(min(max(num_warps, 2), 8))
    check_config(cfg, xnumel=size_hints[0])
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)

def triton_config_tiled_reduction(size_hints, x, y, r, num_stages=1):
    if False:
        return 10
    '\n    Construct a tile reduction triton config with some adjustment\n    heuristics based on size_hints. Size_hints is a tuple of numels in\n    each tile dimension and will be rounded up to the nearest power of 2.\n    '
    target = conditional_product(x, y, r)
    if conditional_product(*size_hints) < target:
        target //= 8
    x = min(x, size_hints[0])
    y = min(y, size_hints[1])
    r = min(r, size_hints[2])
    while x < size_hints[0] and conditional_product(x, y, r) < target:
        x *= 2
    while r < size_hints[2] and conditional_product(x, y, r) < target:
        r *= 2
    while y < size_hints[1] and conditional_product(x, y, r) < target:
        y *= 2
    cfg = {'XBLOCK': x, 'YBLOCK': y, 'RBLOCK': r}
    num_warps = next_power_of_2(min(max(conditional_product(x, y, r) // 256, 1), 8))
    check_config(cfg, xnumel=size_hints[0], ynumel=size_hints[1])
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)

def pointwise(size_hints, triton_meta, tile_hint=None, filename=None, min_elem_per_thread=0, inductor_meta=None):
    if False:
        i = 10
        return i + 15
    '\n    Construct @triton.heuristics() based on size_hints.\n    '
    inductor_meta = {} if inductor_meta is None else inductor_meta
    numel = functools.reduce(operator.mul, size_hints)
    bs = max(256, min(numel // 128, 1024))
    hinted_configs = autotune_hints_to_configs(inductor_meta.get('autotune_hints', set()), size_hints, bs)
    triton_config_with_settings = functools.partial(triton_config, min_elem_per_thread=min_elem_per_thread)
    if len(size_hints) == 1:
        if disable_pointwise_autotuning() and (not (config.max_autotune or config.max_autotune_pointwise)):
            return cached_autotune(size_hints, [triton_config_with_settings(size_hints, bs)], triton_meta=triton_meta, inductor_meta=inductor_meta, heuristic_type=HeuristicType.POINTWISE, filename=filename)
        else:
            return cached_autotune(size_hints, [triton_config_with_settings(size_hints, bs, num_elements_per_warp=256), triton_config_with_settings(size_hints, bs // 2, num_elements_per_warp=64), *hinted_configs], triton_meta=triton_meta, inductor_meta=inductor_meta, heuristic_type=HeuristicType.POINTWISE, filename=filename)
    if len(size_hints) == 2:
        if (disable_pointwise_autotuning() or tile_hint == TileHint.SQUARE) and (not (config.max_autotune or config.max_autotune_pointwise)):
            return cached_autotune(size_hints, [triton_config_with_settings(size_hints, 32, 32)], triton_meta=triton_meta, inductor_meta=inductor_meta, heuristic_type=HeuristicType.POINTWISE, filename=filename)
        return cached_autotune(size_hints, [triton_config_with_settings(size_hints, 32, 32), triton_config_with_settings(size_hints, 64, 64), triton_config_with_settings(size_hints, 256, 16), triton_config_with_settings(size_hints, 16, 256), triton_config_with_settings(size_hints, bs, 1), triton_config_with_settings(size_hints, 1, bs), *hinted_configs], triton_meta=triton_meta, inductor_meta=inductor_meta, filename=filename, heuristic_type=HeuristicType.POINTWISE)
    if len(size_hints) == 3:
        if disable_pointwise_autotuning():
            return cached_autotune(size_hints, [triton_config_with_settings(size_hints, 16, 16, 16)], triton_meta=triton_meta, inductor_meta=inductor_meta, heuristic_type=HeuristicType.POINTWISE, filename=filename)
        return cached_autotune(size_hints, [triton_config_with_settings(size_hints, 16, 16, 16), triton_config_with_settings(size_hints, 64, 8, 8), triton_config_with_settings(size_hints, 8, 64, 8), triton_config_with_settings(size_hints, 8, 8, 64), triton_config_with_settings(size_hints, bs, 1, 1), triton_config_with_settings(size_hints, 1, bs, 1), triton_config_with_settings(size_hints, 1, 1, bs), *hinted_configs], triton_meta=triton_meta, inductor_meta=inductor_meta, filename=filename, heuristic_type=HeuristicType.POINTWISE)
    raise NotImplementedError(f'size_hints: {size_hints}')

def reduction(size_hints, reduction_hint=False, triton_meta=None, filename=None, inductor_meta=None):
    if False:
        i = 10
        return i + 15
    'args to @triton.heuristics()'
    inductor_meta = {} if inductor_meta is None else inductor_meta
    assert triton_meta is not None
    rnumel = size_hints[-1]
    if len(size_hints) == 2:
        contiguous_config = triton_config_reduction(size_hints, 1, rnumel if 256 <= rnumel < 2048 else 2048)
        outer_config = triton_config_reduction(size_hints, 128, 8)
        tiny_config = triton_config_reduction(size_hints, 2 * (256 // rnumel) if rnumel <= 256 else 1, min(rnumel, 2048))
        if config.max_autotune or config.max_autotune_pointwise:
            pass
        elif reduction_hint == ReductionHint.INNER:
            return cached_autotune(size_hints, [contiguous_config], triton_meta=triton_meta, inductor_meta=inductor_meta, heuristic_type=HeuristicType.REDUCTION, filename=filename)
        elif reduction_hint == ReductionHint.OUTER:
            return cached_autotune(size_hints, [outer_config], triton_meta=triton_meta, inductor_meta=inductor_meta, heuristic_type=HeuristicType.REDUCTION, filename=filename)
        elif reduction_hint == ReductionHint.OUTER_TINY:
            return cached_autotune(size_hints, [tiny_config], triton_meta=triton_meta, inductor_meta=inductor_meta, heuristic_type=HeuristicType.REDUCTION, filename=filename)
        if disable_pointwise_autotuning():
            return cached_autotune(size_hints, [triton_config_reduction(size_hints, 32, 128)], triton_meta=triton_meta, inductor_meta=inductor_meta, heuristic_type=HeuristicType.REDUCTION, filename=filename)
        return cached_autotune(size_hints, [contiguous_config, outer_config, tiny_config, triton_config_reduction(size_hints, 64, 64), triton_config_reduction(size_hints, 8, 512), triton_config_reduction(size_hints, 64, 4, num_warps=8)], triton_meta=triton_meta, inductor_meta=inductor_meta, filename=filename, heuristic_type=HeuristicType.REDUCTION)
    raise NotImplementedError(f'size_hints: {size_hints}')

def persistent_reduction(size_hints, reduction_hint=False, triton_meta=None, filename=None, inductor_meta=None):
    if False:
        while True:
            i = 10
    (xnumel, rnumel) = size_hints
    configs = [triton_config_reduction(size_hints, xblock, rnumel) for xblock in (1, 8, 32, 128) if rnumel * xblock <= 4096 and xblock <= xnumel]
    if reduction_hint == ReductionHint.INNER and rnumel >= 256:
        configs = configs[:1]
    elif reduction_hint == ReductionHint.OUTER:
        configs = configs[-1:]
    elif reduction_hint == ReductionHint.OUTER_TINY:
        configs = [triton_config_reduction(size_hints, 2 * (256 // rnumel) if rnumel <= 256 else 1, rnumel)]
    for c in configs:
        c.kwargs.pop('RBLOCK')
    if disable_pointwise_autotuning():
        configs = configs[:1]
    return cached_autotune(size_hints, configs, triton_meta=triton_meta, inductor_meta=inductor_meta, filename=filename, heuristic_type=HeuristicType.PERSISTENT_REDUCTION)

def template(num_stages, num_warps, triton_meta, filename=None, inductor_meta=None):
    if False:
        return 10
    '\n    Compile a triton template\n    '
    return cached_autotune(None, [triton.Config({}, num_stages=num_stages, num_warps=num_warps)], triton_meta=triton_meta, inductor_meta=inductor_meta, heuristic_type=HeuristicType.TEMPLATE, filename=filename)

def user_autotune(configs, triton_meta, filename=None, inductor_meta=None):
    if False:
        i = 10
        return i + 15
    '\n    Compile a user defined triton kernel\n    '
    defaults = inspect.signature(triton.Config).parameters
    default_num_stages = defaults['num_stages'].default
    default_num_warps = defaults['num_warps'].default
    if len(configs) == 0:
        configs = [triton.Config({}, num_stages=default_num_stages, num_warps=default_num_warps)]
    else:
        configs = [triton.Config(c.get('kwargs', {}), num_stages=c.get('num_stages', default_num_stages), num_warps=c.get('num_warps', default_num_warps)) for c in configs]
    return cached_autotune(None, configs, triton_meta=triton_meta, heuristic_type=HeuristicType.USER_AUTOTUNE, filename=filename, inductor_meta=inductor_meta)

def foreach(triton_meta, num_warps, filename=None, inductor_meta=None):
    if False:
        i = 10
        return i + 15
    '\n    Compile a triton foreach kernel\n    '
    return cached_autotune(None, [triton.Config({}, num_stages=1, num_warps=num_warps)], triton_meta=triton_meta, inductor_meta=inductor_meta, heuristic_type=HeuristicType.TEMPLATE, filename=filename)

def grid(*numels):
    if False:
        while True:
            i = 10
    'Helper function to compute triton grids'
    if len(numels) == 1:
        (xnumel, ynumel, znumel) = (numels[0], None, None)
    elif len(numels) == 2:
        (xnumel, ynumel, znumel) = (numels[1], numels[0], None)
    elif len(numels) == 3:
        (xnumel, ynumel, znumel) = (numels[2], numels[1], numels[0])
    else:
        raise AssertionError(f'invalid size for numels {len(numels)}')

    def get_grid_dim(numel, block):
        if False:
            for i in range(10):
                print('nop')
        if numel is None:
            return 1
        if block is None:
            return numel
        return ceildiv(numel, block)

    def grid_fn(meta):
        if False:
            while True:
                i = 10
        return (get_grid_dim(xnumel, meta.get('XBLOCK', 1)), get_grid_dim(ynumel, meta.get('YBLOCK', None)), get_grid_dim(znumel, meta.get('ZBLOCK', None)))
    return grid_fn