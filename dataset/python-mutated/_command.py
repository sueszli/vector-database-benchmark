import json
import os
import os.path
import subprocess
import sys
from typing import Any, Dict, List, Tuple
import setuptools
import setuptools.command.build_ext
import cupy_builder
import cupy_builder.install_build as build
from cupy_builder._context import Context
from cupy_builder._compiler import DeviceCompilerUnix, DeviceCompilerWin32

def filter_files_by_extension(sources: List[str], extension: str) -> Tuple[List[str], List[str]]:
    if False:
        return 10
    sources_selected = []
    sources_others = []
    for src in sources:
        if os.path.splitext(src)[1] == extension:
            sources_selected.append(src)
        else:
            sources_others.append(src)
    return (sources_selected, sources_others)

def compile_device_code(ctx: Context, ext: setuptools.Extension) -> Tuple[List[str], List[str]]:
    if False:
        print('Hello World!')
    'Compiles device code ("*.cu").\n\n    This method invokes the device compiler (nvcc/hipcc) to build object\n    files from device code, then returns the tuple of:\n    - list of remaining (non-device) source files ("*.cpp")\n    - list of compiled object files for device code ("*.o")\n    '
    (sources_cu, sources_cpp) = filter_files_by_extension(ext.sources, '.cu')
    if len(sources_cu) == 0:
        return (ext.sources, [])
    if sys.platform == 'win32':
        compiler = DeviceCompilerWin32(ctx)
    else:
        compiler = DeviceCompilerUnix(ctx)
    objects = []
    for src in sources_cu:
        print(f'{ext.name}: Device code: {src}')
        obj_ext = 'obj' if sys.platform == 'win32' else 'o'
        obj = f'build/temp.device_objects/{src}.{obj_ext}'
        if os.path.exists(obj) and _get_timestamp(src) < _get_timestamp(obj):
            print(f'{ext.name}: Reusing cached object file: {obj}')
        else:
            os.makedirs(os.path.dirname(obj), exist_ok=True)
            print(f'{ext.name}: Building: {obj}')
            compiler.compile(obj, src, ext)
        objects.append(obj)
    return (sources_cpp, objects)

def _get_timestamp(path: str) -> float:
    if False:
        while True:
            i = 10
    stat = os.lstat(path)
    return max(stat.st_atime, stat.st_mtime, stat.st_ctime)

def dumpbin_dependents(dumpbin: str, path: str) -> List[str]:
    if False:
        i = 10
        return i + 15
    args = [dumpbin, '/nologo', '/dependents', path]
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE)
    except FileNotFoundError:
        print(f'*** DUMPBIN not found: {args}')
        return []
    if p.returncode != 0:
        print(f'*** DUMPBIN failed ({p.returncode}): {args}')
        return []
    sections = p.stdout.decode().split('\r\n\r\n')
    for (num, section) in enumerate(sections):
        if 'Image has the following dependencies:' in section:
            return [line.strip() for line in sections[num + 1].splitlines()]
    print(f'*** DUMPBIN output could not be parsed: {args}')
    return []

class custom_build_ext(setuptools.command.build_ext.build_ext):
    """Custom `build_ext` command to include CUDA C source files."""

    def _cythonize(self, nthreads: int) -> None:
        if False:
            i = 10
            return i + 15
        import Cython.Build
        ctx = cupy_builder.get_context()
        compiler_directives = {'linetrace': ctx.linetrace, 'profile': ctx.profile, 'embedsignature': True}
        compile_time_env: Dict[str, Any] = {}
        use_cuda_python = ctx.use_cuda_python
        compile_time_env['CUPY_USE_CUDA_PYTHON'] = use_cuda_python
        if use_cuda_python:
            print('Using CUDA Python')
        compile_time_env['CUPY_CUFFT_STATIC'] = False
        compile_time_env['CUPY_CYTHON_VERSION'] = Cython.__version__
        if ctx.use_stub:
            compile_time_env['CUPY_CUDA_VERSION'] = 0
            compile_time_env['CUPY_HIP_VERSION'] = 0
        elif ctx.use_hip:
            compile_time_env['CUPY_CUDA_VERSION'] = 0
            compile_time_env['CUPY_HIP_VERSION'] = build.get_hip_version()
        else:
            compile_time_env['CUPY_CUDA_VERSION'] = ctx.features['cuda'].get_version()
            compile_time_env['CUPY_HIP_VERSION'] = 0
        print('Compile-time constants: ' + json.dumps(compile_time_env, indent=4))
        if sys.platform == 'win32':
            nthreads = 0
        Cython.Build.cythonize(self.extensions, verbose=True, nthreads=nthreads, language_level=3, compiler_directives=compiler_directives, annotate=ctx.annotate, compile_time_env=compile_time_env)

    def build_extensions(self) -> None:
        if False:
            return 10
        num_jobs = int(os.environ.get('CUPY_NUM_BUILD_JOBS', '4'))
        if num_jobs > 1:
            self.parallel = num_jobs
            if hasattr(self.compiler, 'initialize'):
                self.compiler.initialize()
        print('Cythonizing...')
        self._cythonize(num_jobs)
        for ext in self.extensions:
            (sources_pyx, sources_others) = filter_files_by_extension(ext.sources, '.pyx')
            sources_cpp = ['{}.cpp'.format(os.path.splitext(src)[0]) for src in sources_pyx]
            ext.sources = sources_cpp + sources_others
            for src in ext.sources:
                if not os.path.isfile(src):
                    raise RuntimeError(f'Fatal error: missing file: {src}')
        print('Building extensions...')
        super().build_extensions()
        if sys.platform == 'win32':
            print('Generating DLL dependency list...')
            dumpbin = os.path.join(os.path.dirname(self.compiler.lib), 'dumpbin.exe')
            depends = sorted(set(sum([dumpbin_dependents(dumpbin, f) for f in self.get_outputs()], [])))
            depends_json = os.path.join(self.build_lib, 'cupy', '.data', '_depends.json')
            os.makedirs(os.path.dirname(depends_json), exist_ok=True)
            with open(depends_json, 'w') as f:
                json.dump({'depends': depends}, f)

    def build_extension(self, ext: setuptools.Extension) -> None:
        if False:
            i = 10
            return i + 15
        ctx = cupy_builder.get_context()
        (sources_cpp, extra_objects) = compile_device_code(ctx, ext)
        ext.sources = sources_cpp
        ext.extra_objects = extra_objects + ext.extra_objects
        super().build_extension(ext)