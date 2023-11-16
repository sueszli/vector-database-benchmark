"""JIT C++ strings into executables."""
import atexit
import os
import re
import shutil
import textwrap
import threading
from typing import Any, List, Optional
import torch
from torch.utils.benchmark.utils._stubs import CallgrindModuleType, TimeitModuleType
from torch.utils.benchmark.utils.common import _make_temp_dir
from torch.utils import cpp_extension
LOCK = threading.Lock()
SOURCE_ROOT = os.path.split(os.path.abspath(__file__))[0]
_BUILD_ROOT: Optional[str] = None

def _get_build_root() -> str:
    if False:
        print('Hello World!')
    global _BUILD_ROOT
    if _BUILD_ROOT is None:
        _BUILD_ROOT = _make_temp_dir(prefix='benchmark_utils_jit_build')
        atexit.register(shutil.rmtree, _BUILD_ROOT)
    return _BUILD_ROOT
CXX_FLAGS: Optional[List[str]]
if hasattr(torch.__config__, '_cxx_flags'):
    try:
        CXX_FLAGS = torch.__config__._cxx_flags().strip().split()
        if CXX_FLAGS is not None and '-g' not in CXX_FLAGS:
            CXX_FLAGS.append('-g')
        if CXX_FLAGS is not None:
            CXX_FLAGS = list(filter(lambda x: not x.startswith('-W'), CXX_FLAGS))
    except RuntimeError:
        CXX_FLAGS = None
else:
    CXX_FLAGS = ['-O2', '-fPIC', '-g']
EXTRA_INCLUDE_PATHS: List[str] = [os.path.join(SOURCE_ROOT, 'valgrind_wrapper')]
CONDA_PREFIX = os.getenv('CONDA_PREFIX')
if CONDA_PREFIX is not None:
    EXTRA_INCLUDE_PATHS.append(os.path.join(CONDA_PREFIX, 'include'))
COMPAT_CALLGRIND_BINDINGS: Optional[CallgrindModuleType] = None

def get_compat_bindings() -> CallgrindModuleType:
    if False:
        i = 10
        return i + 15
    with LOCK:
        global COMPAT_CALLGRIND_BINDINGS
        if COMPAT_CALLGRIND_BINDINGS is None:
            COMPAT_CALLGRIND_BINDINGS = cpp_extension.load(name='callgrind_bindings', sources=[os.path.join(SOURCE_ROOT, 'valgrind_wrapper', 'compat_bindings.cpp')], extra_cflags=CXX_FLAGS, extra_include_paths=EXTRA_INCLUDE_PATHS)
    return COMPAT_CALLGRIND_BINDINGS

def _compile_template(*, stmt: str, setup: str, global_setup: str, src: str, is_standalone: bool) -> Any:
    if False:
        while True:
            i = 10
    for (before, after, indentation) in (('// GLOBAL_SETUP_TEMPLATE_LOCATION', global_setup, 0), ('// SETUP_TEMPLATE_LOCATION', setup, 4), ('// STMT_TEMPLATE_LOCATION', stmt, 8)):
        src = re.sub(before, textwrap.indent(after, ' ' * indentation)[indentation:], src)
    with LOCK:
        name = f'timer_cpp_{abs(hash(src))}'
        build_dir = os.path.join(_get_build_root(), name)
        os.makedirs(build_dir, exist_ok=True)
        src_path = os.path.join(build_dir, 'timer_src.cpp')
        with open(src_path, 'w') as f:
            f.write(src)
    return cpp_extension.load(name=name, sources=[src_path], build_directory=build_dir, extra_cflags=CXX_FLAGS, extra_include_paths=EXTRA_INCLUDE_PATHS, is_python_module=not is_standalone, is_standalone=is_standalone)

def compile_timeit_template(*, stmt: str, setup: str, global_setup: str) -> TimeitModuleType:
    if False:
        for i in range(10):
            print('nop')
    template_path: str = os.path.join(SOURCE_ROOT, 'timeit_template.cpp')
    with open(template_path) as f:
        src: str = f.read()
    module = _compile_template(stmt=stmt, setup=setup, global_setup=global_setup, src=src, is_standalone=False)
    assert isinstance(module, TimeitModuleType)
    return module

def compile_callgrind_template(*, stmt: str, setup: str, global_setup: str) -> str:
    if False:
        i = 10
        return i + 15
    template_path: str = os.path.join(SOURCE_ROOT, 'valgrind_wrapper', 'timer_callgrind_template.cpp')
    with open(template_path) as f:
        src: str = f.read()
    target = _compile_template(stmt=stmt, setup=setup, global_setup=global_setup, src=src, is_standalone=True)
    assert isinstance(target, str)
    return target