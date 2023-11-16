from __future__ import annotations
import base64
import copyreg
import dataclasses
import functools
import hashlib
import importlib
import io
import json
import logging
import multiprocessing
import os
import pathlib
import pickle
import pkgutil
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import warnings
import weakref
from bisect import bisect_right
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy
from ctypes import c_void_p, cdll, CDLL
from dataclasses import field
from functools import partial
from importlib import abc
from pathlib import Path
from threading import Thread
from time import sleep, time
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union
import torch
from torch._dynamo.device_interface import get_interface_for_device, get_registered_device_interfaces
from torch._dynamo.utils import counters
from torch._inductor import config, exc
from torch._inductor.codegen.cuda import cuda_env
from torch._inductor.utils import cache_dir, developer_warning, is_linux
from torch._prims_common import suggest_memory_format
from torch.fx.experimental.symbolic_shapes import has_hint, hint_int, ShapeEnv
if TYPE_CHECKING:
    from torch._inductor.graph import GraphLowering
    from torch._inductor.select_algorithm import ChoiceCaller
from torch.hub import _Faketqdm, tqdm
_HERE = os.path.abspath(__file__)
_TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))
if config.is_fbcode():
    from triton.fb import build_paths
    from triton.fb.build import _run_build_command
    from torch._inductor.fb.utils import log_global_cache_errors, log_global_cache_stats, log_global_cache_vals, use_global_cache
else:

    def log_global_cache_errors(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def log_global_cache_stats(*args, **kwargs):
        if False:
            while True:
                i = 10
        pass

    def log_global_cache_vals(*args, **kwargs):
        if False:
            return 10
        pass

    def use_global_cache() -> bool:
        if False:
            for i in range(10):
                print('nop')
        return False
LOCK_TIMEOUT = 600
_cumulative_compile_time = 0.0
_t0 = None

def _compile_start() -> None:
    if False:
        i = 10
        return i + 15
    global _t0
    if _t0 is None:
        _t0 = time()

def _compile_end() -> None:
    if False:
        return 10
    global _cumulative_compile_time, _t0
    if _t0 is not None:
        t1 = time()
        _cumulative_compile_time += t1 - _t0
        _t0 = None
log = logging.getLogger(__name__)

def cpp_wrapper_cache_dir(name: str) -> str:
    if False:
        i = 10
        return i + 15
    cu_str = 'cpu' if torch.version.cuda is None else f"cu{torch.version.cuda.replace('.', '')}"
    python_version = f'py{sys.version_info.major}{sys.version_info.minor}'
    build_folder = f'{python_version}_{cu_str}'
    cpp_wrapper_dir = os.path.join(cache_dir(), build_folder)
    cpp_wrapper_build_directory = os.path.join(cpp_wrapper_dir, name)
    os.makedirs(cpp_wrapper_build_directory, exist_ok=True)
    return cpp_wrapper_build_directory

class CacheBase:

    @staticmethod
    @functools.lru_cache(None)
    def get_system() -> Dict[str, Any]:
        if False:
            return 10
        try:
            import triton
            triton_version = triton.__version__
        except ModuleNotFoundError:
            triton_version = None
        try:
            system: Dict[str, Any] = {'device': {'name': torch.cuda.get_device_properties(torch.cuda.current_device()).name}, 'version': {'cuda': torch.version.cuda, 'triton': triton_version}, 'other': {'allow_tf32': torch.backends.cuda.matmul.allow_tf32}}
        except (AssertionError, RuntimeError):
            system = {}
        system['hash'] = hashlib.sha256(json.dumps(system, sort_keys=True).encode('utf-8')).hexdigest()
        return system

    @staticmethod
    @functools.lru_cache(None)
    def get_local_cache_path() -> Path:
        if False:
            return 10
        return Path(os.path.join(cache_dir(), 'cache', CacheBase.get_system()['hash']))

    @staticmethod
    @functools.lru_cache(None)
    def get_global_cache_path() -> Optional[Path]:
        if False:
            return 10
        return Path(os.path.join(config.global_cache_dir, CacheBase.get_system()['hash'])) if config.global_cache_dir is not None else None

    def __init__(self) -> None:
        if False:
            return 10
        if not torch.cuda.is_available():
            return
        self.system = CacheBase.get_system()
        self.local_cache_path = CacheBase.get_local_cache_path()
        self.global_cache_path = CacheBase.get_global_cache_path()

    def get_local_cache(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        if not self.local_cache_path.is_file():
            return {}
        with open(self.local_cache_path) as local_cache_fp:
            local_cache = json.load(local_cache_fp)
        return local_cache['cache']

    def update_local_cache(self, local_cache: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        if not os.path.exists(self.local_cache_path.parent):
            os.makedirs(self.local_cache_path.parent, exist_ok=True)
        write_atomic(str(self.local_cache_path), json.dumps({'system': self.system, 'cache': local_cache}, indent=4))

class LocalCache(CacheBase):

    def lookup(self, *keys: Tuple[str]) -> Optional[Dict[str, Any]]:
        if False:
            while True:
                i = 10
        cache = self.get_local_cache()
        sub_cache = cache
        for key in keys:
            if key in cache:
                sub_cache = cache[key]
            else:
                return None
        return sub_cache

    def set_value(self, *keys: List[str], value: Any) -> None:
        if False:
            return 10
        cache = self.get_local_cache()
        sub_cache: Dict = cache
        for key in keys[0:-1]:
            sub_cache.setdefault(key, {})
            sub_cache = sub_cache[key]
        sub_cache[keys[-1]] = value
        self.update_local_cache(cache)

class PersistentCache(CacheBase):

    @functools.lru_cache(None)
    def get_global_cache(self):
        if False:
            print('Hello World!')
        if self.global_cache_path is None or not self.global_cache_path.is_file():
            return {}
        with open(self.global_cache_path) as global_cache_fp:
            global_cache = json.load(global_cache_fp)
        return global_cache['cache']

    def lookup(self, choices: List[ChoiceCaller], name: str, inputs: str, benchmark: Callable[[Any], Dict[ChoiceCaller, float]]) -> Dict[ChoiceCaller, float]:
        if False:
            return 10
        "\n        Check to see if we have benchmarked the given choice callers. For each\n        choice caller:\n\n            1. Check global_cache[name][inputs][choice], return benchmark if cached.\n            2. Check local_cache[name][inputs][choice], return benchmark if cached.\n            3.\n                a. `max_autotune_gemm=True`: benchmark the choice, update\n                    local_cache[name][inputs][choice], and return the benchmark.\n                b. `max_autotune_gemm=False`: don't benchmark the choice, return nothing.\n        "
        log_stats = partial(log_global_cache_stats, self.system, name, inputs)
        log_vals = partial(log_global_cache_vals, self.system, name, inputs)
        log_errors = partial(log_global_cache_errors, self.system, name, inputs)
        timings = {}

        def check_cache(cache, callback=None) -> bool:
            if False:
                i = 10
                return i + 15
            'Check if `cache` contains data for all the choices'
            hit = True
            for choice in choices:
                choice_hash = choice.hash_key()
                if choice_hash in cache.get(name, {}).get(inputs, {}):
                    timings[choice] = cache[name][inputs][choice_hash]
                else:
                    hit = False
                    break
            if callback:
                callback(cached=hit)
            return hit
        if config.max_autotune or config.max_autotune_gemm:
            local_cache = self.get_local_cache()
            if not check_cache(local_cache) and (not (use_global_cache() and check_cache(self.get_global_cache(), callback=log_stats))):
                try:
                    timings = benchmark(choices)
                    assert all((choice in timings for choice in choices))
                    local_cache.setdefault(name, {})
                    local_cache[name].setdefault(inputs, {})
                    for (choice, timing) in timings.items():
                        local_cache[name][inputs][choice.hash_key()] = timing
                except RuntimeError as e:
                    log_errors(e)
                    raise e
                self.update_local_cache(local_cache)
                timings_to_log = {choice.hash_key(): timings[choice] for choice in choices}
                log_vals(timings_to_log)
        elif use_global_cache():
            check_cache(self.get_global_cache(), callback=log_stats)
        return timings

def get_lock_dir() -> str:
    if False:
        return 10
    lock_dir = os.path.join(cache_dir(), 'locks')
    if not os.path.exists(lock_dir):
        os.makedirs(lock_dir, exist_ok=True)
    return lock_dir

def sha256_hash(data: bytes) -> str:
    if False:
        for i in range(10):
            print('nop')
    return base64.b32encode(hashlib.sha256(data).digest())[:51].decode('utf-8').lower()

def code_hash(code: Union[str, bytes], extra: str=''):
    if False:
        for i in range(10):
            print('nop')
    hashing_str = code if isinstance(code, bytes) else code.encode('utf-8')
    if extra != '':
        hashing_str = hashing_str + b'||' + extra.encode('utf-8')
    return 'c' + sha256_hash(hashing_str)

def get_path(basename: str, extension: str, specified_dir: str='') -> Tuple[str, str, str]:
    if False:
        print('Hello World!')
    if specified_dir:
        if os.path.isabs(specified_dir):
            subdir = specified_dir
        else:
            subdir = os.path.join(cache_dir(), specified_dir)
    else:
        subdir = os.path.join(cache_dir(), basename[1:3])
    path = os.path.join(subdir, f'{basename}.{extension}')
    return (basename, subdir, path)

def get_hash(content: Union[str, bytes], extra: str='', hash_type: str='code'):
    if False:
        return 10
    if hash_type == 'code':
        return code_hash(content, extra)
    if hash_type == 'cubin':
        return code_hash(repr(content))
    raise AssertionError(f'Unknown hash type {hash_type}')

def write(content: Union[str, bytes], extension: str, extra: str='', hash_type: str='code', specified_dir: str='') -> Tuple[str, str]:
    if False:
        print('Hello World!')
    key: str = get_hash(content, extra, hash_type)
    (basename, subdir, path) = get_path(key, extension, specified_dir)
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    if not os.path.exists(path):
        write_atomic(path, content)
    return (basename, path)

def write_atomic(path: str, content: Union[str, bytes]) -> None:
    if False:
        return 10
    assert isinstance(content, (str, bytes)), 'Only strings and byte arrays can be saved in the cache'
    path = pathlib.Path(path)
    tmp_path = path.parent / f'.{os.getpid()}.{threading.get_ident()}.tmp'
    write_mode = 'w' if isinstance(content, str) else 'wb'
    with tmp_path.open(write_mode) as f:
        f.write(content)
    tmp_path.rename(path)

@dataclasses.dataclass
class TensorMetadata:
    """
    The Tensor metadata relevant when hashing FxGraph cache keys.
    """
    dtype: torch.dtype
    shape: torch.Size
    stride: Tuple[Any, ...]
    device: torch.device
    layout: torch.layout
    memory_format: Optional[torch.memory_format]
    storage_offset: int
    requires_grad: bool
    is_quantized: bool
    is_conj: bool
    is_neg: bool
    is_coalesced: bool
    dense_dim: int
    sparse_dim: int

@dataclasses.dataclass
class TensorMetadataAndValues:
    """
    TensorMetadata plus the elements as a list of raw values.
    Used for hashing inlined constants.
    """
    tensor_metadata: TensorMetadata
    values: List[Any]

def extract_tensor_metadata(t: torch.Tensor) -> TensorMetadata:
    if False:
        i = 10
        return i + 15
    '\n    Extract the TensorMetadata of a tensor.\n    '
    memory_format: Optional[torch.memory_format] = suggest_memory_format(t)
    if not t.is_contiguous(memory_format=memory_format):
        memory_format = None
    return TensorMetadata(dtype=t.dtype, shape=t.shape, stride=t.stride() if t.layout == torch.strided else (), device=t.device, layout=t.layout, memory_format=memory_format, storage_offset=t.storage_offset(), requires_grad=t.requires_grad, is_quantized=t.is_quantized, is_conj=t.is_conj(), is_neg=t.is_neg(), is_coalesced=t.is_coalesced() if t.is_sparse else False, dense_dim=t.dense_dim() if t.is_sparse else False, sparse_dim=t.sparse_dim() if t.is_sparse else False)

def _ident(x: Any) -> Any:
    if False:
        return 10
    return x

def _reduce_fake_tensor(t):
    if False:
        while True:
            i = 10
    '\n    See FxGraphCachePickler. Custom reducer to pickle FakeTensors.\n    '
    metadata = extract_tensor_metadata(t)
    return (_ident, (metadata,))

def _reduce_tensor(t):
    if False:
        for i in range(10):
            print('nop')
    '\n    See FxGraphCachePickler. Custom reducer to pickle Tensors.\n    '
    metadata = extract_tensor_metadata(t)
    if len(t.shape) == 0 or torch._inductor.graph.GraphLowering.can_inline_constant(t):
        return (_ident, (TensorMetadataAndValues(metadata, t.tolist()),))
    else:
        return (_ident, (metadata,))

def _reduce_symint(s):
    if False:
        for i in range(10):
            print('nop')
    '\n    See FxGraphCachePickler. Custom reducer to pickle SymInts.\n    '
    return (_ident, (str(s),))

class FxGraphCachePickler(pickle.Pickler):
    """
    Custom pickler to customize the pickling of some objects (Tensors), only for the
    purpose of computing a hash for keying into the FxGraphCache. Tensors contain
    objects that don't pickle and/or vary between runs, and we want to capture the
    data that allow us to compute a stable, but safe hash.
    """
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[torch._subclasses.fake_tensor.FakeTensor] = _reduce_fake_tensor
    dispatch_table[torch.Tensor] = _reduce_tensor
    dispatch_table[torch.SymInt] = _reduce_symint

    @staticmethod
    def dumps(obj) -> bytes:
        if False:
            i = 10
            return i + 15
        '\n        Pickle an object using the FxGraphCachePickler.\n        '
        with io.BytesIO() as stream:
            pickler = FxGraphCachePickler(stream)
            pickler.dump(obj)
            return stream.getvalue()

    @staticmethod
    def get_hash(obj: Any) -> str:
        if False:
            while True:
                i = 10
        '\n        Serialize an object using the FxGraphCachePickler and return a hash\n        of the pickled object.\n        '
        serialized_data = FxGraphCachePickler.dumps(obj)
        return sha256_hash(serialized_data)

@functools.lru_cache(None)
def get_inductor_code_hash() -> bytes:
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute a hash of all inductor code modules. Used by the FxGraph cache\n    so any inductor code changes would result in new cache keys.\n    '
    inductor_root = os.path.dirname(__file__)
    contents: Dict[str, bytes] = {}
    for lib in pkgutil.iter_modules([inductor_root]):
        spec = lib.module_finder.find_spec(lib.name, None)
        assert spec is not None
        module = spec.origin
        assert module is not None
        with open(module, 'rb') as f:
            contents[module] = f.read()
    return hashlib.sha256(pickle.dumps(contents)).digest()

@dataclasses.dataclass
class OrderedSetHolder:
    """
    See FxGraphHashDetails. Holds a sorted list to support stable hashing
    of set kwargs.
    """
    items: List[Any]

class FxGraphHashDetails:
    """
    Object to capture all the details for a compiled FX graph relevant to computing
    a safe and stable cache key.
    """
    EXCLUDED_KWARGS = ['graph_id']

    def __init__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], fx_kwargs: Dict[str, Any]):
        if False:
            return 10
        self.gm = gm
        self.example_inputs = example_inputs
        self.fx_kwargs = {}
        for k in sorted(fx_kwargs):
            if k not in self.EXCLUDED_KWARGS:
                if type(fx_kwargs[k]) is set:
                    self.fx_kwargs[k] = OrderedSetHolder(sorted(fx_kwargs[k]))
                else:
                    self.fx_kwargs[k] = fx_kwargs[k]
        self.torch_version = torch.__version__
        self.system_info = CacheBase.get_system()
        self.inductor_config = config.save_config()
        self.inductor_code_hash = get_inductor_code_hash()

    def debug_str(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get a printable string describing in more detail all the attributes\n        comprising this object. Useful for debugging when one graph hashes\n        to a different value than another.\n        '

        def get_str(obj) -> str:
            if False:
                return 10
            if isinstance(obj, torch.Tensor):
                return str(extract_tensor_metadata(obj))
            elif isinstance(obj, bytes):
                return '<bytes>'
            else:
                return str(obj)
        lines = []
        for (attr, obj) in vars(self).items():
            if isinstance(obj, list):
                for ii in range(len(obj)):
                    h = FxGraphCachePickler.get_hash(obj[ii])
                    lines.append(f'[{h}] {attr}[{ii}]: {get_str(obj[ii])}')
            elif isinstance(obj, dict):
                for (k, v) in obj.items():
                    h = FxGraphCachePickler.get_hash(v)
                    lines.append(f'[{h}] {attr}[{k}]: {get_str(v)}')
            else:
                h = FxGraphCachePickler.get_hash(obj)
                lines.append(f'[{h}] {attr}: {get_str(obj)}')
        return '\n'.join(lines)

def compiled_fx_graph_hash(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], fx_kwargs: Dict[str, Any]) -> str:
    if False:
        print('Hello World!')
    '\n    Generate a unique hash of the FX graph for caching.\n    '
    details = FxGraphHashDetails(gm, example_inputs, fx_kwargs)
    key = 'f' + FxGraphCachePickler.get_hash(details)
    log.debug('FX graph cache hash details for key %s:\n%s', key, details.debug_str())
    return key

class FxGraphCache:
    """
    Supports caching and reusing compiled Fx graphs.

    The overall strategy is as follows:
    - This cache stores entries on disk. When saving an entry, we can't
      serialize callables (that could be C++, Triton, etc.), so we serialize
      their own disk cache location. We then recreate the compiled artifact
      after fetching from disk.
    - For indexing the cache, we gather the fields relevant to identifying an
      FxGraph (the graph module, graph inputs, system settings etc.) into an
      FxGraphCacheDetails object, pickle it, and compute a hash for the key.
      See FxGraphCachePickler.
    - Among the metadata we store, we also include a guards expression that's
      appropriate for validating any symbols for Tensor arguments that have
      symbolic bounds. On cache lookup then, we evaluate those guards in the
      current context to validate that a cached entry can be served.
    - A given graph could have multiple compiled versions, corresponding to
      different sets of guards. Therefore, we store cache entries in the form:
          <temp dir>/<fx graph hash>/<serialized metatdata>
    - On lookup, we compute the key from the graph details, iterate over all
      leaf files in the corresponding subdirectory, deserialize the entry, and
      evaluate its guards expression. If the evaluation succeeds, we have a
      cache hit. If it fails, we compile the graph and store a new entry.
    - Finally, on a cache hit, we need to make sure any guards that would
      have been created during compilation are added to the current context.
    """

    @staticmethod
    def _get_tmp_dir() -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the toplevel temporary directory for storing compiled graphs.\n        '
        return os.path.join(cache_dir(), 'fxgraph')

    @staticmethod
    def _get_tmp_dir_for_key(key: str) -> str:
        if False:
            print('Hello World!')
        '\n        Return the disk location for a given cache key.\n        '
        return os.path.join(FxGraphCache._get_tmp_dir(), key[1:3], key)

    @staticmethod
    def _filter_symints(inputs: List[Any]) -> List[torch.SymInt]:
        if False:
            print('Hello World!')
        '\n        Get the SymInt objects from the input list.\n        '
        return [s for s in inputs if isinstance(s, torch.SymInt)]

    @staticmethod
    def _get_shape_env() -> ShapeEnv:
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper to get the shape env from the tracing context.\n        '
        return torch._guards.TracingContext.get().fake_mode.shape_env

    @staticmethod
    def _lookup_graph(key: str, example_inputs: List[torch.Tensor]) -> Optional[CompiledFxGraph]:
        if False:
            print('Hello World!')
        '\n        Lookup a compiled graph in the cache by key. On a hit, return the\n        deserialized CompiledFxGraph object. On a miss, return None.\n        '
        subdir = FxGraphCache._get_tmp_dir_for_key(key)
        if not os.path.exists(subdir):
            return None
        for path in sorted(os.listdir(subdir)):
            with open(os.path.join(subdir, path), 'rb') as f:
                graph: CompiledFxGraph = pickle.load(f)
            guards_expr = graph.guards_expr
            if not guards_expr:
                return graph
            shape_env = FxGraphCache._get_shape_env()
            symints = FxGraphCache._filter_symints(example_inputs)
            assert all((has_hint(s) for s in symints))
            hints = [hint_int(s) for s in symints]
            hit = bool(shape_env.evaluate_guards_expression(guards_expr, hints))
            log.debug('fx graph cache key %s evaluating guards for %s with values %s => %s', key, guards_expr, hints, hit)
            if hit:
                check = bool(shape_env.evaluate_guards_expression(guards_expr, symints))
                assert check is True
                log.debug('fx graph cache key %s post-load guards: %s', key, shape_env.guards)
                return graph
        return None

    @staticmethod
    def _save_graph(key: str, compiled_graph: CompiledFxGraph, example_inputs: List[torch.Tensor]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Store a serialized CompiledFxGraph on disk.\n        '
        disk_compiled_graph = copy(compiled_graph)
        disk_compiled_graph.compiled_artifact = None
        shape_env = FxGraphCache._get_shape_env()
        symints = FxGraphCache._filter_symints(example_inputs)
        disk_compiled_graph.guards_expr = shape_env.produce_guards_expression(symints)
        content = pickle.dumps(disk_compiled_graph)
        subdir = FxGraphCache._get_tmp_dir_for_key(key)
        if not os.path.exists(subdir):
            os.makedirs(subdir, exist_ok=True)
        path = os.path.join(subdir, sha256_hash(content))
        write_atomic(path, content)

    @staticmethod
    def load(compile_fx_fn: Callable[..., Any], gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], fx_kwargs: Dict[str, Any]):
        if False:
            while True:
                i = 10
        '\n        Load a compiled graph from the cache. If a cached entry does not exist,\n        compile the graph and save it to the cache.\n        '
        from filelock import FileLock
        key = compiled_fx_graph_hash(gm, example_inputs, fx_kwargs)
        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + '.lock'), timeout=LOCK_TIMEOUT)
        with lock:
            compiled_graph = FxGraphCache._lookup_graph(key, example_inputs)
            if compiled_graph is None:
                log.debug('fx graph cache miss for key %s', key)
                counters['inductor']['fxgraph_cache_miss'] += 1
                compiled_graph = compile_fx_fn(gm, example_inputs, **fx_kwargs)
                FxGraphCache._save_graph(key, compiled_graph, example_inputs)
            else:
                log.debug('fx graph cache hit for key %s', key)
                counters['inductor']['fxgraph_cache_hit'] += 1
            return compiled_graph

    @staticmethod
    def clear():
        if False:
            for i in range(10):
                print('nop')
        '\n        Clear out the on-disk cache.\n        '
        shutil.rmtree(FxGraphCache._get_tmp_dir())

@dataclasses.dataclass
class CompiledFxGraph:
    """
    Class holding a compiled FX graph. This is the object serialized on disk
    to support FxGraph caching.
    """
    compiled_artifact: Optional[Callable[..., Any]] = None
    current_callable: Optional[Callable[..., Any]] = None
    cache_key: Optional[str] = None
    artifact_path: Optional[str] = None
    cache_linemap: Optional[List[Tuple[int, str]]] = None
    device_types: Set[str] = field(default_factory=set)
    device_idxs: Set[int] = field(default_factory=set)
    mutated_inputs: Set[str] = field(default_factory=set)
    mutated_input_idxs: Set[int] = field(default_factory=set)
    constants: Dict[str, torch.Tensor] = field(default_factory=dict)
    output_strides: Optional[List[Optional[Tuple[int, ...]]]] = None
    guards_expr: Optional[str] = None
    _boxed_call: Optional[bool] = None

    def __init__(self, compiled_artifact: Optional[Callable[..., Any]], graph: GraphLowering, output_strides: List[Optional[Tuple[int, ...]]]):
        if False:
            while True:
                i = 10
        self.compiled_artifact = compiled_artifact
        self.cache_key = graph.cache_key
        self.artifact_path = graph.cache_path
        self.cache_linemap = graph.cache_linemap
        self.device_types = graph.device_types
        self.device_idxs = graph.device_idxs
        self.mutated_inputs = graph.mutated_inputs
        self.mutated_input_idxs = set(graph.mutated_input_idxs)
        self.constants = graph.constants
        self.output_strides = output_strides
        self.guards_expr = None

    def __call__(self, inputs: List[Any]) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return self.get_current_callable()(inputs)

    def get_current_callable(self) -> Callable[..., Any]:
        if False:
            print('Hello World!')
        if self.current_callable is None:
            return functools.partial(_run_from_cache, weakref.proxy(self))
        else:
            return self.current_callable

def _run_from_cache(compiled_graph: CompiledFxGraph, inputs: List[Any]) -> Any:
    if False:
        i = 10
        return i + 15
    if compiled_graph.compiled_artifact is None:
        from .codecache import PyCodeCache
        assert compiled_graph.cache_key
        assert compiled_graph.artifact_path
        compiled_graph.compiled_artifact = PyCodeCache.load_by_key_path(compiled_graph.cache_key, compiled_graph.artifact_path, compiled_graph.cache_linemap, compiled_graph.constants).call
    return compiled_graph.compiled_artifact(inputs)

def cpp_compiler() -> str:
    if False:
        return 10
    if config.is_fbcode():
        return build_paths.gcc()
    if isinstance(config.cpp.cxx, (list, tuple)):
        search = tuple(config.cpp.cxx)
    else:
        search = (config.cpp.cxx,)
    return cpp_compiler_search(search)

@functools.lru_cache(1)
def cpp_compiler_search(search: str) -> str:
    if False:
        return 10
    for cxx in search:
        try:
            if cxx is None:
                if sys.platform != 'linux':
                    continue
                if not os.getenv('TORCH_INDUCTOR_INSTALL_GXX'):
                    continue
                from filelock import FileLock
                lock_dir = get_lock_dir()
                lock = FileLock(os.path.join(lock_dir, 'g++.lock'), timeout=LOCK_TIMEOUT)
                with lock:
                    cxx = install_gcc_via_conda()
            subprocess.check_output([cxx, '--version'])
            return cxx
        except (subprocess.SubprocessError, FileNotFoundError, ImportError):
            continue
    raise exc.InvalidCxxCompiler()

def install_gcc_via_conda() -> str:
    if False:
        i = 10
        return i + 15
    'On older systems, this is a quick way to get a modern compiler'
    prefix = os.path.join(cache_dir(), 'gcc')
    cxx_path = os.path.join(prefix, 'bin', 'g++')
    if not os.path.exists(cxx_path):
        log.info('Downloading GCC via conda')
        conda = os.environ.get('CONDA_EXE', 'conda')
        if conda is None:
            conda = shutil.which('conda')
        if conda is not None:
            subprocess.check_call([conda, 'create', f'--prefix={prefix}', '--channel=conda-forge', '--quiet', '-y', 'python=3.8', 'gxx'], stdout=subprocess.PIPE)
    return cxx_path

def is_gcc() -> bool:
    if False:
        print('Hello World!')
    return bool(re.search('(gcc|g\\+\\+)', cpp_compiler()))

def is_clang() -> bool:
    if False:
        while True:
            i = 10
    return bool(re.search('(clang|clang\\+\\+)', cpp_compiler()))

@functools.lru_cache(None)
def is_apple_clang() -> bool:
    if False:
        for i in range(10):
            print('nop')
    cxx = cpp_compiler()
    version_string = subprocess.check_output([cxx, '--version']).decode('utf8')
    return 'Apple' in version_string.splitlines()[0]

class VecISA:
    _bit_width: int
    _macro: str
    _arch_flags: str
    _dtype_nelements: Dict[torch.dtype, int]
    _avx_code = '\n#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_ZVECTOR)\n#include <ATen/cpu/vec/functional.h>\n#include <ATen/cpu/vec/vec.h>\n#endif\n\n__attribute__((aligned(64))) float in_out_ptr0[16] = {0.0};\n\nextern "C" void __avx_chk_kernel() {\n    auto tmp0 = at::vec::Vectorized<float>(1);\n    auto tmp1 = tmp0.exp();\n    tmp1.store(in_out_ptr0);\n}\n'
    _avx_py_load = '\nimport torch\nfrom ctypes import cdll\ncdll.LoadLibrary("__lib_path__")\n'

    def bit_width(self) -> int:
        if False:
            while True:
                i = 10
        return self._bit_width

    def nelements(self, dtype: torch.dtype=torch.float) -> int:
        if False:
            while True:
                i = 10
        return self._dtype_nelements[dtype]

    def build_macro(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._macro

    def build_arch_flags(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._arch_flags

    def __hash__(self) -> int:
        if False:
            print('Hello World!')
        return hash(str(self))

    @functools.lru_cache(None)
    def __bool__(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if config.cpp.vec_isa_ok is not None:
            return config.cpp.vec_isa_ok
        (key, input_path) = write(VecISA._avx_code, 'cpp')
        from filelock import FileLock
        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + '.lock'), timeout=LOCK_TIMEOUT)
        with lock:
            output_path = input_path[:-3] + 'so'
            build_cmd = shlex.split(cpp_compile_command(input_path, output_path, warning_all=False, vec_isa=self))
            try:
                compile_file(input_path, output_path, build_cmd)
                subprocess.check_call([sys.executable, '-c', VecISA._avx_py_load.replace('__lib_path__', output_path)], stderr=subprocess.DEVNULL, env={**os.environ, 'PYTHONPATH': ':'.join(sys.path)})
            except Exception as e:
                return False
            return True

@dataclasses.dataclass
class VecAVX512(VecISA):
    _bit_width = 512
    _macro = '-DCPU_CAPABILITY_AVX512'
    _arch_flags = '-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma'
    _dtype_nelements = {torch.float: 16, torch.bfloat16: 32, torch.float16: 32}

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'avx512'
    __hash__: Callable[[VecISA], Any] = VecISA.__hash__

@dataclasses.dataclass
class VecAVX2(VecISA):
    _bit_width = 256
    _macro = '-DCPU_CAPABILITY_AVX2'
    _arch_flags = '-mavx2 -mfma'
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16, torch.float16: 16}

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return 'avx2'
    __hash__: Callable[[VecISA], Any] = VecISA.__hash__

@dataclasses.dataclass
class VecZVECTOR(VecISA):
    _bit_width = 256
    _macro = '-DCPU_CAPABILITY_ZVECTOR -DCPU_CAPABILITY=ZVECTOR -DHAVE_ZVECTOR_CPU_DEFINITION'
    _arch_flags = '-mvx -mzvector'
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16, torch.float16: 16}

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return 'zvector'
    __hash__: Callable[[VecISA], Any] = VecISA.__hash__

class InvalidVecISA(VecISA):
    _bit_width = 0
    _macro = ''
    _arch_flags = ''
    _dtype_nelements = {}

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'INVALID_VEC_ISA'

    def __bool__(self) -> bool:
        if False:
            while True:
                i = 10
        return False
    __hash__: Callable[[VecISA], Any] = VecISA.__hash__
invalid_vec_isa = InvalidVecISA()
supported_vec_isa_list = [VecAVX512(), VecAVX2()]

@functools.lru_cache(None)
def valid_vec_isa_list() -> List[VecISA]:
    if False:
        print('Hello World!')
    if sys.platform != 'linux':
        return []
    if platform.machine() == 's390x':
        return [VecZVECTOR()]
    isa_list = []
    with open('/proc/cpuinfo') as _cpu_info:
        _cpu_info_content = _cpu_info.read()
        for isa in supported_vec_isa_list:
            if str(isa) in _cpu_info_content and isa:
                isa_list.append(isa)
        return isa_list

def pick_vec_isa() -> VecISA:
    if False:
        return 10
    if config.is_fbcode():
        return VecAVX2()
    _valid_vec_isa_list: List[VecISA] = valid_vec_isa_list()
    if not _valid_vec_isa_list:
        return invalid_vec_isa
    if config.cpp.simdlen is None:
        assert _valid_vec_isa_list
        return _valid_vec_isa_list[0]
    for isa in _valid_vec_isa_list:
        if config.cpp.simdlen == isa.bit_width():
            return isa
    return invalid_vec_isa

def get_compile_only(compile_only: bool=True) -> str:
    if False:
        return 10
    return '-c' if compile_only else ''

def get_shared(shared: bool=True) -> str:
    if False:
        print('Hello World!')
    return '-shared -fPIC' if shared else ''

def get_warning_all_flag(warning_all: bool=True) -> str:
    if False:
        for i in range(10):
            print('nop')
    return '-Wall' if warning_all else ''

def get_glibcxx_abi_build_flags() -> str:
    if False:
        i = 10
        return i + 15
    return '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))

def cpp_flags() -> str:
    if False:
        return 10
    flags = ['-std=c++17', '-Wno-unused-variable', '-Wno-unknown-pragmas']
    if is_clang():
        flags.append('-Werror=ignored-optimization-argument')
    return ' '.join(flags)

def cpp_wrapper_flags() -> str:
    if False:
        print('Hello World!')
    return '-DTORCH_INDUCTOR_CPP_WRAPPER'

def optimization_flags() -> str:
    if False:
        print('Hello World!')
    base_flags = '-O0 -g' if config.aot_inductor.debug_compile else '-O3 -DNDEBUG'
    base_flags += ' -ffast-math -fno-finite-math-only'
    if config.is_fbcode():
        return base_flags
    if sys.platform == 'darwin':
        base_flags += ' -Xclang'
    elif platform.machine() == 'ppc64le':
        base_flags += ' -mcpu=native'
    else:
        base_flags += ' -march=native'
    if not config.is_fbcode():
        base_flags += ' -fopenmp'
    return base_flags

def use_custom_generated_macros() -> str:
    if False:
        i = 10
        return i + 15
    return '-D C10_USING_CUSTOM_GENERATED_MACROS'

def use_fb_internal_macros() -> str:
    if False:
        print('Hello World!')
    if config.is_fbcode():
        openmp_lib = build_paths.openmp_lib()
        preprocessor_flags = ' '.join(('-D C10_USE_GLOG', '-D C10_USE_MINIMAL_GLOG', '-D C10_DISABLE_TENSORIMPL_EXTENSIBILITY'))
        return f'-Wp,-fopenmp {openmp_lib} {preprocessor_flags}'
    else:
        return ''

def use_standard_sys_dir_headers() -> str:
    if False:
        return 10
    if config.is_fbcode():
        return '-nostdinc'
    else:
        return ''

@functools.lru_cache(None)
def is_conda_llvm_openmp_installed() -> bool:
    if False:
        print('Hello World!')
    try:
        command = 'conda list llvm-openmp --json'
        output = subprocess.check_output(command.split()).decode('utf8')
        return len(json.loads(output)) > 0
    except subprocess.SubprocessError:
        return False

@functools.lru_cache(None)
def homebrew_libomp() -> Tuple[bool, str]:
    if False:
        for i in range(10):
            print('nop')
    try:
        subprocess.check_output(['which', 'brew'])
        libomp_path = subprocess.check_output(['brew', '--prefix', 'libomp']).decode('utf8').strip()
        omp_available = os.path.exists(libomp_path)
        return (omp_available, libomp_path)
    except subprocess.SubprocessError:
        return (False, '')

def get_include_and_linking_paths(include_pytorch: bool=False, vec_isa: VecISA=invalid_vec_isa, cuda: bool=False, aot_mode: bool=False) -> Tuple[List[str], str, str, str, str]:
    if False:
        return 10
    if config.is_fbcode() and 'CUDA_HOME' not in os.environ and ('CUDA_PATH' not in os.environ):
        os.environ['CUDA_HOME'] = os.path.dirname(build_paths.cuda())
    from torch.utils import cpp_extension
    macros = ''
    build_arch_flags = ''
    if sys.platform == 'linux' and (include_pytorch or vec_isa != invalid_vec_isa or cuda or config.cpp.enable_kernel_profile):
        ipaths = cpp_extension.include_paths(cuda) + [sysconfig.get_path('include')]
        lpaths = cpp_extension.library_paths(cuda) + [sysconfig.get_config_var('LIBDIR')]
        libs = []
        if not config.is_fbcode():
            libs += ['torch', 'torch_cpu']
            libs += ['gomp']
            if not aot_mode:
                libs += ['torch_python']
        else:
            libs += ['omp']
            if aot_mode:
                ipaths += [os.path.dirname(cpp_prefix_path())]
                if cuda:
                    for (i, path) in enumerate(lpaths):
                        if path.startswith(os.environ['CUDA_HOME']) and (not os.path.exists(f'{path}/libcudart_static.a')):
                            for (root, dirs, files) in os.walk(path):
                                if 'libcudart_static.a' in files:
                                    lpaths[i] = os.path.join(path, root)
                                    lpaths.append(os.path.join(lpaths[i], 'stubs'))
                                    break
        macros = vec_isa.build_macro()
        if macros:
            if config.is_fbcode() and vec_isa != invalid_vec_isa:
                cap = str(vec_isa).upper()
                macros = ' '.join([vec_isa.build_arch_flags(), f'-D CPU_CAPABILITY={cap}', f'-D CPU_CAPABILITY_{cap}', f'-D HAVE_{cap}_CPU_DEFINITION'])
        if aot_mode and cuda:
            if macros is None:
                macros = ''
            macros += ' -D USE_CUDA'
        if cuda:
            if config.is_fbcode():
                libs += ['cuda']
            else:
                libs += ['c10_cuda', 'cuda', 'torch_cuda']
        build_arch_flags = vec_isa.build_arch_flags()
    else:
        ipaths = cpp_extension.include_paths(cuda) + [sysconfig.get_path('include')]
        if aot_mode:
            ipaths += [os.path.dirname(cpp_prefix_path())]
        lpaths = []
        if sys.platform == 'darwin':
            omp_available = not is_apple_clang()
            if os.getenv('OMP_PREFIX') is not None:
                header_path = os.path.join(os.getenv('OMP_PREFIX'), 'include', 'omp.h')
                valid_env = os.path.exists(header_path)
                if valid_env:
                    ipaths.append(os.path.join(os.getenv('OMP_PREFIX'), 'include'))
                    lpaths.append(os.path.join(os.getenv('OMP_PREFIX'), 'lib'))
                else:
                    warnings.warn('environment variable `OMP_PREFIX` is invalid.')
                omp_available = omp_available or valid_env
            libs = [] if omp_available else ['omp']
            if not omp_available and os.getenv('CONDA_PREFIX') is not None:
                omp_available = is_conda_llvm_openmp_installed()
                if omp_available:
                    conda_lib_path = os.path.join(os.getenv('CONDA_PREFIX'), 'lib')
                    ipaths.append(os.path.join(os.getenv('CONDA_PREFIX'), 'include'))
                    lpaths.append(conda_lib_path)
                    if os.uname().machine == 'x86_64' and os.path.exists(os.path.join(conda_lib_path, 'libiomp5.dylib')):
                        libs = ['iomp5']
            if not omp_available:
                (omp_available, libomp_path) = homebrew_libomp()
                if omp_available:
                    ipaths.append(os.path.join(libomp_path, 'include'))
                    lpaths.append(os.path.join(libomp_path, 'lib'))
        else:
            libs = ['omp'] if config.is_fbcode() else ['gomp']
    if not config.aot_inductor.abi_compatible:
        libs += ['c10']
        lpaths += [cpp_extension.TORCH_LIB_PATH]
    if config.is_fbcode():
        ipaths.append(build_paths.sleef())
        ipaths.append(build_paths.openmp())
        ipaths.append(build_paths.cc_include())
        ipaths.append(build_paths.libgcc())
        ipaths.append(build_paths.libgcc_arch())
        ipaths.append(build_paths.libgcc_backward())
        ipaths.append(build_paths.glibc())
        ipaths.append(build_paths.linux_kernel())
        ipaths.append(build_paths.cuda())
        ipaths.append('include')
    static_link_libs = []
    if aot_mode and cuda and config.is_fbcode():
        static_link_libs = ['-Wl,-Bstatic', '-lcudart_static', '-Wl,-Bdynamic']
    lpaths_str = ' '.join(['-L' + p for p in lpaths])
    libs_str = ' '.join(static_link_libs + ['-l' + p for p in libs])
    return (ipaths, lpaths_str, libs_str, macros, build_arch_flags)

def cpp_compile_command(input: Union[str, List[str]], output: str, warning_all: bool=True, shared: bool=True, include_pytorch: bool=False, vec_isa: VecISA=invalid_vec_isa, cuda: bool=False, aot_mode: bool=False, compile_only: bool=False, use_absolute_path: bool=False) -> str:
    if False:
        print('Hello World!')
    (ipaths, lpaths, libs, macros, build_arch_flags) = get_include_and_linking_paths(include_pytorch, vec_isa, cuda, aot_mode)
    if isinstance(input, str):
        input = [input]
    ipaths_str = ' '.join(['-I' + p for p in ipaths])
    if config.is_fbcode():
        if aot_mode and (not use_absolute_path):
            inp_name = input
            out_name = output
        else:
            inp_name = [os.path.basename(i) for i in input]
            out_name = os.path.basename(output)
        linker_paths = [os.path.dirname(build_paths.ld()), build_paths.glibc_lib()]
        linker_paths = ' '.join(['-B' + p for p in linker_paths])
    else:
        inp_name = input
        out_name = output
        linker_paths = ''
    inp_name_str = ' '.join(inp_name)
    return re.sub('[ \\n]+', ' ', f'\n            {cpp_compiler()} {inp_name_str} {get_shared(shared)}\n            {get_warning_all_flag(warning_all)} {cpp_flags()}\n            {get_glibcxx_abi_build_flags()}\n            {ipaths_str} {lpaths} {libs} {build_arch_flags}\n            {macros} {linker_paths}\n            {optimization_flags()}\n            {use_custom_generated_macros()}\n            {use_fb_internal_macros()}\n            {use_standard_sys_dir_headers()}\n            {get_compile_only(compile_only)}\n            -o {out_name}\n        ').strip()

def run_command_and_check(cmd: str):
    if False:
        print('Hello World!')
    cmd = shlex.split(cmd)
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise exc.CppCompileError(cmd, e.output) from e

@functools.lru_cache(None)
def split_aot_inductor_output_path(path: str) -> Tuple[str, str]:
    if False:
        while True:
            i = 10
    'Returns the path where the AOT Inductor compiled kernels are stored.'
    if path.endswith('.so'):
        return os.path.split(path)
    else:
        return (path, '')

class CudaKernelParamCache:
    cache: Dict[str, Dict[str, str]] = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def set(cls, key: str, params: Dict[str, str], cubin: str) -> None:
        if False:
            return 10
        (_, path) = write(cubin, 'cubin', hash_type='cubin', specified_dir=split_aot_inductor_output_path(config.aot_inductor.output_path)[0])
        params['cubin_path'] = path
        cls.cache[key] = params

    @classmethod
    def get(cls, key: str) -> Optional[Dict[str, str]]:
        if False:
            i = 10
            return i + 15
        return cls.cache.get(key, None)

class AotCodeCache:
    cache: Dict[str, str] = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def compile(cls, graph: GraphLowering, source_code: str, serialized_extern_kernel_nodes: Optional[str], cuda: bool) -> str:
        if False:
            for i in range(10):
                print('nop')
        picked_vec_isa = pick_vec_isa()
        cpp_command = repr(cpp_compile_command('i', 'o', vec_isa=picked_vec_isa, cuda=cuda, aot_mode=graph.aot_mode))
        fbcode_aot_cpu_re = False
        use_absolute_path = False
        if config.is_fbcode():
            ld_command = build_paths.ld()
            if not cuda and graph.aot_mode:
                objcopy_command = build_paths.objcopy_fallback()
                fbcode_aot_cpu_re = True
                use_absolute_path = True
            else:
                objcopy_command = build_paths.objcopy()
        else:
            ld_command = 'ld'
            objcopy_command = 'objcopy'
        (specified_output_path, specified_so_name) = split_aot_inductor_output_path(config.aot_inductor.output_path)
        (key, input_path) = write(source_code, 'cpp', extra=cpp_command, specified_dir=specified_output_path)
        if key not in cls.cache or (specified_output_path and os.path.dirname(cls.cache[key]) != specified_output_path or (specified_so_name and os.path.basename(cls.cache[key]) != specified_so_name)):
            from filelock import FileLock
            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + '.lock'), timeout=LOCK_TIMEOUT)
            with lock:
                if config.is_fbcode() and serialized_extern_kernel_nodes:
                    output_json = os.path.splitext(input_path)[0] + '.json'
                    with open(output_json, 'w') as f:
                        f.write(serialized_extern_kernel_nodes)
                output_so = config.aot_inductor.output_path if specified_so_name else os.path.splitext(input_path)[0] + '.so'
                if not os.path.exists(output_so):
                    output_o = os.path.splitext(input_path)[0] + '.o'
                    cmd = cpp_compile_command(input=input_path, output=output_o, vec_isa=picked_vec_isa, cuda=cuda, aot_mode=graph.aot_mode, compile_only=True, use_absolute_path=use_absolute_path)
                    log.debug('aot compilation command: %s', cmd)
                    if fbcode_aot_cpu_re:
                        compile_file(input_path, output_o, cmd.split())
                        os.chmod(output_o, 420)
                    else:
                        run_command_and_check(cmd)

                    def _to_bytes(t: torch.Tensor) -> bytes:
                        if False:
                            for i in range(10):
                                print('nop')
                        import ctypes
                        t_cpu = t.untyped_storage().cpu()
                        raw_array = ctypes.cast(t_cpu.data_ptr(), ctypes.POINTER(ctypes.c_ubyte * t_cpu.nbytes()))
                        return bytes(raw_array.contents)
                    aot_constants = b''.join((_to_bytes(tensor) for tensor in graph.constants.values()))
                    (consts_key, consts_path) = write(aot_constants, 'bin', specified_dir=specified_output_path)
                    consts_o = os.path.splitext(consts_path)[0] + '.o'
                    if fbcode_aot_cpu_re:
                        cmd = f'{ld_command} -r -b binary -o {os.path.basename(consts_o)} {os.path.basename(consts_path)}'
                        compile_file(consts_path, consts_o, cmd.split())
                        os.chmod(consts_o, 420)
                    else:
                        cmd = f'{ld_command} -r -b binary -o {consts_o} {consts_path}'
                        run_command_and_check(cmd)
                    log.debug('aot constant binary command: %s', cmd)
                    cmd = f'{objcopy_command} --rename-section .data=.lrodata,alloc,load,readonly,data,contents {consts_o} {consts_o}'
                    log.debug('aot constant obj command: %s', cmd)
                    run_command_and_check(cmd)
                    cmd = f'rm {consts_path}'
                    log.debug('aot constant bin removal command: %s', cmd)
                    run_command_and_check(cmd)
                    if fbcode_aot_cpu_re:
                        body = re.sub('[\\W]', '_', os.path.basename(consts_path))
                    else:
                        body = re.sub('[\\W]', '_', consts_path)
                    symbol_list = []
                    symbol_list.append(f'{objcopy_command} --redefine-sym _binary_{body}_start=_binary_constants_bin_start {consts_o}')
                    symbol_list.append(f'{objcopy_command} --redefine-sym _binary_{body}_size=_binary_constants_bin_size {consts_o}')
                    symbol_list.append(f'{objcopy_command} --redefine-sym _binary_{body}_end=_binary_constants_bin_end {consts_o}')
                    log.debug('aot constant binary redefine symbol: %s', ' '.join(symbol_list))
                    for cmd in symbol_list:
                        run_command_and_check(cmd)
                    cmd = cpp_compile_command(input=[output_o, consts_o], output=output_so, vec_isa=picked_vec_isa, cuda=cuda, aot_mode=graph.aot_mode, use_absolute_path=use_absolute_path)
                    log.debug('aot linkage command: %s', cmd)
                    if fbcode_aot_cpu_re:
                        compile_file([output_o, consts_o], output_so, cmd.split())
                        os.chmod(output_so, 493)
                    else:
                        run_command_and_check(cmd)
                else:
                    log.debug('aot_inductor dynamic library already exist: %s', output_so)
                cls.cache[key] = output_so
        return cls.cache[key]

@functools.lru_cache
def cpp_prefix_path() -> str:
    if False:
        i = 10
        return i + 15
    path = Path(__file__).parent / 'codegen/cpp_prefix.h'
    with path.open() as f:
        content = f.read()
        (_, filename) = write(content, 'h')
    return filename

def cpp_prefix() -> str:
    if False:
        for i in range(10):
            print('nop')
    filename = cpp_prefix_path()
    if config.is_fbcode():
        return f'#include "{os.path.basename(filename)}"'
    else:
        return f'#include "{filename}"'

def compile_file(input_path: Union[str, List[str]], output_path: str, cmd: List[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    input_paths = [input_path] if isinstance(input_path, str) else input_path
    input_files = [os.path.basename(ip) if config.is_fbcode() else ip for ip in input_paths]
    try:
        if config.is_fbcode():
            header_path = cpp_prefix_path()
            header_name = os.path.basename(header_path)
            output_name = os.path.basename(output_path)
            torch_includes_path = os.path.join(torch.utils.cpp_extension._TORCH_PATH, 'include')
            with tempfile.TemporaryDirectory() as tmp_dir:
                shutil.copy(header_path, os.path.join(tmp_dir, header_name))
                for (p, f) in zip(input_paths, input_files):
                    shutil.copy(p, os.path.join(tmp_dir, f))
                dest_include_path = os.path.join(tmp_dir, 'include')
                shutil.copytree(torch_includes_path, dest_include_path)
                output_file_path = _run_build_command(cmd, tmp_dir, output_name)
                if os.path.exists(output_path):
                    os.remove(output_path)
                shutil.copy(output_file_path, output_path)
        else:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        output = e.output.decode('utf-8')
        openmp_problem = "'omp.h' file not found" in output or 'libomp' in output
        if openmp_problem and sys.platform == 'darwin':
            instruction = '\n\nOpenMP support not found. Please try one of the following solutions:\n(1) Set the `CXX` environment variable to a compiler other than Apple clang++/g++ that has builtin OpenMP support;\n(2) install OpenMP via conda: `conda install llvm-openmp`;\n(3) install libomp via brew: `brew install libomp`;\n(4) manually setup OpenMP and set the `OMP_PREFIX` environment variable to point to a path with `include/omp.h` under it.'
            output += instruction
        raise exc.CppCompileError(cmd, output) from e
_libgomp: Optional[CDLL] = None

class CppCodeCache:
    cache: Dict[str, CDLL] = dict()
    clear = staticmethod(cache.clear)

    @staticmethod
    def _load_library(path: str) -> CDLL:
        if False:
            for i in range(10):
                print('nop')
        try:
            return cdll.LoadLibrary(path)
        except OSError as e:
            if 'gomp' in str(e) and os.path.exists('/usr/lib64/libgomp.so.1'):
                global _libgomp
                _libgomp = cdll.LoadLibrary('/usr/lib64/libgomp.so.1')
                return cdll.LoadLibrary(path)
            if 'failed to map segment from shared object' in str(e):
                raise OSError(f'{e}.  The most common reason this may occur is if the {tempfile.gettempdir()} folder is mounted with noexec (e.g., by default Docker mounts tmp file systems as noexec).  Please remount {tempfile.gettempdir()} with exec enabled, or set another temporary directory with TORCHINDUCTOR_CACHE_DIR environment variable.') from e
            raise

    @classmethod
    def load(cls, source_code: str) -> CDLL:
        if False:
            i = 10
            return i + 15
        picked_vec_isa = pick_vec_isa()
        cpp_command = repr(cpp_compile_command('i', 'o', vec_isa=picked_vec_isa))
        (key, input_path) = write(source_code, 'cpp', extra=cpp_command)
        if key not in cls.cache:
            from filelock import FileLock
            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + '.lock'), timeout=LOCK_TIMEOUT)
            with lock:
                output_path = input_path[:-3] + 'so'
                if not os.path.exists(output_path):
                    cmd = shlex.split(cpp_compile_command(input=input_path, output=output_path, vec_isa=picked_vec_isa))
                    compile_file(input_path, output_path, cmd)
                cls.cache[key] = cls._load_library(output_path)
                cls.cache[key].key = key
        return cls.cache[key]

class PyCodeCache:
    cache: Dict[str, ModuleType] = dict()
    linemaps: Dict[str, List[Tuple[Any, ...]]] = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def write(cls, source_code: str, extra: str='') -> Tuple[str, str]:
        if False:
            for i in range(10):
                print('nop')
        return write(source_code, 'py', extra=extra)

    @classmethod
    def load(cls, source_code: str, extra: str='', linemap: Optional[List[Tuple[int, str]]]=None, attrs: Optional[Dict[str, Any]]=None) -> ModuleType:
        if False:
            for i in range(10):
                print('nop')
        (key, path) = write(source_code, 'py', extra=extra)
        return cls.load_by_key_path(key, path, linemap, attrs)

    @classmethod
    def load_by_key_path(cls, key: str, path: str, linemap: Optional[List[Tuple[int, str]]]=None, attrs: Optional[Dict[str, Any]]=None) -> ModuleType:
        if False:
            i = 10
            return i + 15
        if linemap is None:
            linemap = []
        if key not in cls.cache:
            with open(path) as f:
                try:
                    code = compile(f.read(), path, 'exec')
                except Exception as e:
                    raise RuntimeError(f'Failed to import {path}\n{type(e).__name__}: {e}') from None
                mod = ModuleType(f'{__name__}.{key}')
                mod.__file__ = path
                mod.key = key
                exec(code, mod.__dict__, mod.__dict__)
                sys.modules[mod.__name__] = mod
                cls.cache.setdefault(key, mod)
                cls.linemaps[path] = list(zip(*linemap))
                if attrs is not None:
                    for (k, v) in attrs.items():
                        setattr(mod, k, v)
        return cls.cache[key]

    @classmethod
    @functools.lru_cache(None)
    def stack_frames_for_code(cls, path: str, lineno: int) -> Optional[List[Dict[str, Any]]]:
        if False:
            return 10
        if path not in cls.linemaps:
            return None
        (lines, nodes) = cls.linemaps[path]
        p = bisect_right(lines, lineno)
        if p == 0:
            return None
        entry = nodes[p - 1]
        if not entry:
            return None

        def parse_stack_trace(stack_trace: str) -> List[Dict[str, Any]]:
            if False:
                print('Hello World!')
            regex = 'File "(.+)", line (\\d+), in (.+)\\n'
            matches = re.findall(regex, stack_trace)
            return [{'filename': f, 'line': int(l), 'name': n} for (f, l, n) in reversed(matches)]
        return parse_stack_trace(entry)

class CppWrapperCodeCache:
    cache: Dict[str, CDLL] = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def load(cls, source_code: str, func_name: str, key: str, cuda: bool) -> CDLL:
        if False:
            i = 10
            return i + 15
        name = f'inline_extension_{key}'
        cpp_wrapper_dir = cpp_wrapper_cache_dir(name)
        if not os.path.exists(cpp_wrapper_dir):
            os.makedirs(cpp_wrapper_dir)
        ext = 'so'
        filepath = os.path.join(cpp_wrapper_dir, f'{name}.{ext}')
        log.debug('Cpp wrapper code path %s', filepath)
        if key not in cls.cache:
            log.debug('Cpp wrapper cache miss for %s', filepath)
            from filelock import FileLock
            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + '.lock'), timeout=LOCK_TIMEOUT)
            with lock:
                if not os.path.exists(filepath):
                    log.debug('Cpp wrapper building %s', filepath)
                    _cpp_flags = cpp_flags()
                    _opt_flags = optimization_flags()
                    _shared = get_shared()
                    _warning_all_flag = get_warning_all_flag()
                    (_ipaths, _lpaths, _libs, _macros, _build_arch_flags) = get_include_and_linking_paths(vec_isa=pick_vec_isa(), cuda=cuda)
                    _use_custom_generated_macros = use_custom_generated_macros()
                    _cpp_wrapper_flags = cpp_wrapper_flags()
                    extra_cflags = f'{_cpp_flags} {_opt_flags} {_warning_all_flag} {_build_arch_flags} {_macros}                     {_cpp_wrapper_flags} {_use_custom_generated_macros}'
                    extra_ldflags = f'{_shared} {_lpaths} {_libs} -ffast-math'
                    mod = torch.utils.cpp_extension.load_inline(name=name, build_directory=cpp_wrapper_dir, cpp_sources=[source_code], functions=[func_name], extra_cflags=[extra_cflags], extra_ldflags=[extra_ldflags], extra_include_paths=_ipaths, use_pch=True)
                    log.debug('Cpp wrapper done building %s', filepath)
                else:
                    log.debug('Found target .so, cpp wrapper loading %s', filepath)
                    spec = importlib.util.spec_from_file_location(name, filepath)
                    assert spec is not None
                    mod = importlib.util.module_from_spec(spec)
                    assert isinstance(spec.loader, abc.Loader)
                    spec.loader.exec_module(mod)
                    log.debug('Cpp wrapper done loading %s', filepath)
                cls.cache[key] = mod
        return cls.cache[key]

class TritonCodeCache:

    @classmethod
    def load(cls, kernel_name: str, source_code: str) -> ModuleType:
        if False:
            print('Hello World!')
        mod = PyCodeCache.load(source_code)
        return getattr(mod, kernel_name)

def _cuda_compiler() -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    if cuda_env.nvcc_exist(config.cuda.cuda_cxx):
        return config.cuda.cuda_cxx
    if cuda_env.nvcc_exist(os.getenv('CUDACXX')):
        return os.getenv('CUDACXX', '')
    if cuda_env.nvcc_exist(os.getenv('CUDA_HOME')):
        return os.path.join(os.getenv('CUDA_HOME', ''), 'bin/nvcc')
    return 'nvcc'

def _cutlass_include_paths() -> List[str]:
    if False:
        print('Hello World!')
    cutlass_path = config.cuda.cutlass_dir
    return [os.path.join(cutlass_path, 'include'), os.path.join(cutlass_path, 'tools/library/include'), os.path.join(cutlass_path, 'tools/library/src'), os.path.join(cutlass_path, 'tools/util/include')]

def _cuda_lib_options() -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    from torch.utils import cpp_extension
    extra_ldflags: List[str] = []
    if is_linux():
        extra_lib_dir = 'lib64'
        if not os.path.exists(cpp_extension._join_cuda_home(extra_lib_dir)) and os.path.exists(cpp_extension._join_cuda_home('lib')):
            extra_lib_dir = 'lib'
        extra_ldflags.append(f'-L{cpp_extension._join_cuda_home(extra_lib_dir)}')
        extra_ldflags.append(f"-L{cpp_extension._join_cuda_home(extra_lib_dir, 'stubs')}")
        extra_ldflags.append('-lcuda')
        extra_ldflags.append('-lcudart')
    else:
        raise NotImplementedError('Unsupported env, failed to find cuda libs! Currently only Linux is supported.')
    return extra_ldflags

def _nvcc_host_compiler_options() -> List[str]:
    if False:
        while True:
            i = 10
    return ['-fPIC', '-fno-strict-aliasing', '-fvisibility=hidden', '-Wconversion']

def _nvcc_compiler_options() -> List[str]:
    if False:
        i = 10
        return i + 15
    arch = cuda_env.get_cuda_arch()
    if arch == '90':
        arch = '90a'
    code = [f'sm_{arch}', f'compute_{arch}']
    if config.cuda.enable_cuda_lto:
        code += [f'lto_{arch}']
    options = ['-t=0', '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1', '-w', f"-gencode=arch=compute_{arch},code=[{','.join(code)}]", config.cuda.compile_opt_level, '-std=c++17', '--expt-relaxed-constexpr']
    if config.cuda.enable_debug_info:
        options.extend(['-lineinfo', '-g', '-DCUTLASS_DEBUG_TRACE_LEVEL=1'])
    if config.cuda.enable_ptxas_info:
        options.extend(['--keep', '--ptxas-options=--warn-on-local-memory-usage', '--ptxas-options=--warn-on-spills', '--resource-usage', '--source-in-ptx'])
    if config.cuda.use_fast_math:
        options.extend(['--use_fast_math', '-DCUTLASS_USE_TANH_FOR_SIGMOID=1'])
    return options

def cuda_compile_command(src_files: List[str], dst_file: str, dst_file_ext: str) -> str:
    if False:
        while True:
            i = 10
    include_paths = _cutlass_include_paths()
    cuda_lib_options = _cuda_lib_options()
    nvcc_host_compiler_options = _nvcc_host_compiler_options()
    nvcc_compiler_options = _nvcc_compiler_options()
    options = nvcc_compiler_options + [f'-Xcompiler {opt}' if '=' in opt else f'-Xcompiler={opt}' for opt in nvcc_host_compiler_options] + ['-I' + path for path in include_paths] + cuda_lib_options
    src_file = ' '.join(src_files)
    res = ''
    if dst_file_ext == 'o':
        res = f"{_cuda_compiler()} {' '.join(options)} -c -o {dst_file} {src_file}"
    elif dst_file_ext == 'so':
        options.append('-shared')
        res = f"{_cuda_compiler()} {' '.join(options)} -o {dst_file} {src_file}"
    else:
        raise NotImplementedError(f'Unsupported output file suffix {dst_file_ext}!')
    log.debug('CUDA command: %s', res)
    return res

class DLLWrapper:
    """A wrapper for a dynamic library."""

    def __init__(self, lib_path: str):
        if False:
            while True:
                i = 10
        self.lib_path = lib_path
        self.DLL = cdll.LoadLibrary(lib_path)
        self.is_open = True

    def close(self):
        if False:
            return 10
        if self.is_open:
            self._dlclose()
            self.is_open = False

    def _dlclose(self):
        if False:
            print('Hello World!')
        f_dlclose = None
        if is_linux():
            syms = CDLL(None)
            if not hasattr(syms, 'dlclose'):
                syms = CDLL('libc.so')
            if hasattr(syms, 'dlclose'):
                f_dlclose = syms.dlclose
        else:
            raise NotImplementedError('Unsupported env, failed to do dlclose!')
        if f_dlclose is not None:
            f_dlclose.argtypes = [c_void_p]
            f_dlclose(self.DLL._handle)
        else:
            log.warning('dll unloading function was not found, library may not be unloaded properly!')

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        if not self.is_open:
            raise RuntimeError(f'Cannot use closed DLL library: {self.lib_path}')
        method = getattr(self.DLL, name)

        def _wrapped_func(*args):
            if False:
                for i in range(10):
                    print('nop')
            err = method(*args)
            if err:
                raise RuntimeError(f'Error in function: {method.__name__}')
        return _wrapped_func

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

    def __exit__(self, *args):
        if False:
            return 10
        self.close()

    def __del__(self):
        if False:
            while True:
                i = 10
        self.close()

class CUDACodeCache:

    @dataclasses.dataclass
    class CacheEntry:
        input_path: str
        output_path: str
    cache: Dict[str, CacheEntry] = dict()
    clear = staticmethod(cache.clear)
    _SOURCE_CODE_SUFFIX = 'cu'

    @classmethod
    def write(cls, source_code, dst_file_ext) -> Tuple[str, str]:
        if False:
            print('Hello World!')
        '\n        Writes source code into a file with dst_file_ext as the file extension.\n        Returns the hash key of source code, and the path to the file.\n        '
        cuda_command = repr(cuda_compile_command(['dummy_input'], 'dummy_output', dst_file_ext))
        (key, input_path) = write(source_code, cls._SOURCE_CODE_SUFFIX, extra=cuda_command)
        return (key, input_path)

    @classmethod
    def compile(cls, source_code, dst_file_ext) -> Tuple[str, str, str]:
        if False:
            i = 10
            return i + 15
        '\n        Compiles CUDA source_code into a file with dst_file_ext extension.\n        Returns a tuple of dst_file_path, hash_key, source_code_path\n        '
        (key, input_path) = cls.write(source_code, dst_file_ext)
        if key not in cls.cache:
            from filelock import FileLock
            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + '.lock'), timeout=LOCK_TIMEOUT)
            with lock:
                output_path = input_path[:-len(cls._SOURCE_CODE_SUFFIX)] + dst_file_ext
                if not os.path.exists(output_path):
                    cmd = cuda_compile_command([input_path], output_path, dst_file_ext).split(' ')
                    try:
                        subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=os.environ)
                    except subprocess.CalledProcessError as error:
                        raise exc.CUDACompileError(cmd, error.output) from error
                cls.cache[key] = CUDACodeCache.CacheEntry(input_path, output_path)
        return (cls.cache[key].output_path, key, input_path)

    @classmethod
    def load(cls, source_code, dst_file_ext) -> Tuple[DLLWrapper, str, str]:
        if False:
            i = 10
            return i + 15
        '\n        Compiles source code and loads the generated .so file.\n        Returns a tuple of DLLWrapper, hash_key, source_code_path\n        '
        if dst_file_ext != 'so':
            raise RuntimeError(f'Only support loading a .so file for now. Requested file extension: {dst_file_ext}. Source code: {source_code}')
        (dst_file_path, hash_key, source_code_path) = cls.compile(source_code, dst_file_ext)
        return (DLLWrapper(dst_file_path), hash_key, source_code_path)

def caching_device_properties():
    if False:
        while True:
            i = 10
    for (_, device_interface) in get_registered_device_interfaces():
        if device_interface.is_available():
            device_interface.Worker.get_device_properties()

def _worker_compile(kernel_name: str, source_code: str, cc: int, device: torch.device) -> None:
    if False:
        i = 10
        return i + 15
    device_interface = get_interface_for_device(device.type)
    device_interface.Worker.set_device(device.index)
    kernel = TritonCodeCache.load(kernel_name, source_code)
    kernel.precompile(warm_cache_only_with_cc=cc)

def _load_kernel(kernel_name: str, source_code: str) -> ModuleType:
    if False:
        i = 10
        return i + 15
    kernel = TritonCodeCache.load(kernel_name, source_code)
    kernel.precompile()
    return kernel

class TritonFuture:

    def __init__(self, kernel_name: str, source_code: str, future: Future[Any]) -> None:
        if False:
            print('Hello World!')
        self.kernel_name = kernel_name
        self.source_code = source_code
        self.future = future

    def result(self) -> ModuleType:
        if False:
            return 10
        t0 = time()
        if hasattr(self, 'kernel'):
            return self.kernel
        self.future.result()
        kernel = self.kernel = _load_kernel(self.kernel_name, self.source_code)
        latency = time() - t0
        if latency > 50:
            developer_warning(f'Detected long compilation time of {latency} seconds for kernel name {self.kernel_name}')
            developer_warning(self.source_code)
        del self.kernel_name, self.source_code, self.future
        return kernel

def _async_compile_initializer(orig_ppid) -> None:
    if False:
        print('Hello World!')

    def run() -> None:
        if False:
            for i in range(10):
                print('nop')
        while True:
            sleep(1)
            if orig_ppid != os.getppid():
                os.kill(os.getpid(), signal.SIGKILL)
    global _watchdog_thread
    _watchdog_thread = Thread(target=run, daemon=True)
    _watchdog_thread.start()
_watchdog_thread: Optional[Thread] = None

class AsyncCompile:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

    @staticmethod
    @functools.lru_cache(1)
    def pool() -> ThreadPoolExecutor:
        if False:
            for i in range(10):
                print('nop')
        assert config.compile_threads > 1
        return ThreadPoolExecutor(config.compile_threads)

    @staticmethod
    @functools.lru_cache(1)
    def process_pool() -> ProcessPoolExecutor:
        if False:
            return 10
        caching_device_properties()
        assert config.compile_threads > 1
        orig_ppid = os.getpid()
        ctx = multiprocessing.get_context(config.worker_start_method)
        pool = ProcessPoolExecutor(config.compile_threads, mp_context=ctx, initializer=partial(_async_compile_initializer, orig_ppid))
        multiprocessing.util.Finalize(None, pool.shutdown, exitpriority=sys.maxsize)
        return pool

    @classmethod
    def warm_pool(cls) -> None:
        if False:
            while True:
                i = 10
        if config.compile_threads <= 1:
            return
        _compile_start()
        pool = cls.process_pool()
        if hasattr(pool, '_start_queue_management_thread'):
            pool._start_queue_management_thread()
        else:
            for _ in range(config.compile_threads):
                pool._adjust_process_count()
            if hasattr(pool, '_start_executor_manager_thread'):
                pool._start_executor_manager_thread()
        _compile_end()

    @classmethod
    def submit(cls, task: Callable[..., Any]) -> Any:
        if False:
            print('Hello World!')
        if config.compile_threads <= 1:
            return task()
        return cls.pool().submit(task)

    @classmethod
    def map(cls, fn: Callable[..., Any], seq: List[Any]) -> List[Any]:
        if False:
            for i in range(10):
                print('nop')
        if config.compile_threads <= 1 or len(seq) <= 1:
            return list(map(fn, seq))
        return [t.result() for t in [cls.pool().submit(fn, x) for x in seq]]

    def triton(self, kernel_name: str, source_code: str, device_str: str='cuda') -> Union[TritonFuture, ModuleType]:
        if False:
            for i in range(10):
                print('nop')
        _compile_start()
        if config.compile_threads > 1:
            device_interface = get_interface_for_device(device_str)
            device = torch.device(device_str, device_interface.current_device())
            cc = device_interface.get_compute_capability(device)
            future = self.process_pool().submit(_worker_compile, kernel_name, source_code, cc, device)
            return TritonFuture(kernel_name, source_code, future)
        else:
            return _load_kernel(kernel_name, source_code)

    def cpp(self, source_code: str) -> ModuleType:
        if False:
            while True:
                i = 10

        def task():
            if False:
                return 10
            return CppCodeCache.load(source_code).kernel
        return self.submit(task)

    def cuda(self, source_code, dst_file_ext):
        if False:
            while True:
                i = 10

        def task():
            if False:
                for i in range(10):
                    print('nop')
            return CUDACodeCache.load(source_code, dst_file_ext)[0]
        return self.submit(task)

    def wait(self, scope: Dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        num_kernels = len([value for (key, value) in scope.items() if isinstance(value, (Future, TritonFuture))])
        pbar = tqdm(total=num_kernels, desc='Inductor Compilation', disable=config.disable_progress, delay=0)
        if config.compile_threads > 1:
            for (key, result) in scope.items():
                if config.verbose_progress and (not isinstance(pbar, _Faketqdm)):
                    pbar.set_postfix_str(key)
                if isinstance(result, (Future, TritonFuture)):
                    scope[key] = result.result()
                    pbar.update(1)
        _compile_end()
AsyncCompile.warm_pool()