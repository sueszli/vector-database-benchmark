import copy
from distutils import ccompiler
from distutils import sysconfig
import os
import shutil
import sys
import setuptools
import cupy_builder.install_build as build
from cupy_builder._context import Context
from cupy_builder.install_build import PLATFORM_LINUX
from cupy_builder.install_build import PLATFORM_WIN32

def ensure_module_file(file):
    if False:
        print('Hello World!')
    if isinstance(file, tuple):
        return file
    else:
        return (file, [])

def module_extension_name(file):
    if False:
        return 10
    return ensure_module_file(file)[0]

def module_extension_sources(file, use_cython, no_cuda):
    if False:
        return 10
    (pyx, others) = ensure_module_file(file)
    base = os.path.join(*pyx.split('.'))
    pyx = base + ('.pyx' if use_cython else '.cpp')
    if no_cuda:
        others1 = []
        for source in others:
            (base, ext) = os.path.splitext(source)
            if ext == '.cu':
                continue
            others1.append(source)
        others = others1
    return [pyx] + others

def get_required_modules(MODULES):
    if False:
        for i in range(10):
            print('nop')
    return [m['name'] for m in MODULES if m.required]

def check_library(compiler, includes=(), libraries=(), include_dirs=(), library_dirs=(), define_macros=None, extra_compile_args=()):
    if False:
        while True:
            i = 10
    source = ''.join(['#include <%s>\n' % header for header in includes])
    source += 'int main() {return 0;}'
    try:
        build.build_shlib(compiler, source, libraries, include_dirs, library_dirs, define_macros, extra_compile_args)
    except Exception as e:
        print(e)
        sys.stdout.flush()
        return False
    return True

def canonicalize_hip_libraries(hip_version, libraries):
    if False:
        while True:
            i = 10

    def ensure_tuple(x):
        if False:
            i = 10
            return i + 15
        return x if isinstance(x, tuple) else (x, None)
    new_libraries = []
    for library in libraries:
        (lib_name, pred) = ensure_tuple(library)
        if pred is None:
            new_libraries.append(lib_name)
        elif pred(hip_version):
            new_libraries.append(lib_name)
    libraries.clear()
    libraries.extend(new_libraries)

def preconfigure_modules(ctx: Context, MODULES, compiler, settings):
    if False:
        while True:
            i = 10
    'Returns a list of modules buildable in given environment and settings.\n\n    For each module in MODULES list, this function checks if the module\n    can be built in the current environment and reports it.\n    Returns a list of module names available.\n    '
    nvcc_path = build.get_nvcc_path()
    hipcc_path = build.get_hipcc_path()
    summary = ['', '************************************************************', '* CuPy Configuration Summary                               *', '************************************************************', '', 'Build Environment:', '  Include directories: {}'.format(str(settings['include_dirs'])), '  Library directories: {}'.format(str(settings['library_dirs'])), '  nvcc command       : {}'.format(nvcc_path if nvcc_path else '(not found)'), '  hipcc command      : {}'.format(hipcc_path if hipcc_path else '(not found)'), '', 'Environment Variables:']
    for key in ['CFLAGS', 'LDFLAGS', 'LIBRARY_PATH', 'CUDA_PATH', 'NVCC', 'HIPCC', 'ROCM_HOME']:
        summary += ['  {:<16}: {}'.format(key, os.environ.get(key, '(none)'))]
    summary += ['', 'Modules:']
    ret = []
    for module in MODULES:
        installed = False
        status = 'No'
        errmsg = []
        if module['name'] == 'cutensor':
            cutensor_path = os.environ.get('CUTENSOR_PATH', '')
            inc_path = os.path.join(cutensor_path, 'include')
            if os.path.exists(inc_path):
                settings['include_dirs'].append(inc_path)
            cuda_version = ctx.features['cuda'].get_version()
            cuda_major = str(cuda_version // 1000)
            cuda_major_minor = cuda_major + '.' + str(cuda_version // 10 % 100)
            for cuda_ver in (cuda_major_minor, cuda_major):
                lib_path = os.path.join(cutensor_path, 'lib', cuda_ver)
                if os.path.exists(lib_path):
                    settings['library_dirs'].append(lib_path)
                    break
        if ctx.use_hip and module['name'] == 'cuda':
            if module.configure(compiler, settings):
                hip_version = module.get_version()
                if hip_version >= 401:
                    rocm_path = build.get_rocm_path()
                    inc_path = os.path.join(rocm_path, 'hipfft', 'include')
                    settings['include_dirs'].insert(0, inc_path)
                    lib_path = os.path.join(rocm_path, 'hipfft', 'lib')
                    settings['library_dirs'].insert(0, lib_path)
                canonicalize_hip_libraries(hip_version, module['libraries'])
        print('')
        print('-------- Configuring Module: {} --------'.format(module['name']))
        sys.stdout.flush()
        if not check_library(compiler, includes=module['include'], include_dirs=settings['include_dirs'], define_macros=settings['define_macros'], extra_compile_args=settings['extra_compile_args']):
            errmsg = ['Include files not found: %s' % module['include'], 'Check your CFLAGS environment variable.']
        elif not check_library(compiler, libraries=module['libraries'], library_dirs=settings['library_dirs'], define_macros=settings['define_macros'], extra_compile_args=settings['extra_compile_args']):
            errmsg = ['Cannot link libraries: %s' % module['libraries'], 'Check your LDFLAGS environment variable.']
        elif not module.configure(compiler, settings):
            installed = True
            errmsg = ['The library is installed but not supported.']
        elif module['name'] in ('thrust', 'cub', 'random') and (nvcc_path is None and hipcc_path is None):
            installed = True
            cmd = 'nvcc' if not ctx.use_hip else 'hipcc'
            errmsg = ['{} command could not be found in PATH.'.format(cmd), 'Check your PATH environment variable.']
        else:
            installed = True
            status = 'Yes'
            ret.append(module['name'])
        if installed:
            version = module.get_version()
            if version is not None:
                status += f' (version {version})'
        summary += ['  {:<10}: {}'.format(module['name'], status)]
        if len(errmsg) != 0:
            summary += ['    -> {}'.format(m) for m in errmsg]
            if module['name'] == 'cuda':
                break
    if not ctx.use_hip:
        build.check_compute_capabilities(compiler, settings)
    if len(ret) != len(MODULES):
        if 'cuda' in ret:
            lines = ['WARNING: Some modules could not be configured.', 'CuPy will be installed without these modules.']
        else:
            lines = ['ERROR: CUDA could not be found on your system.', '', 'HINT: You are trying to build CuPy from source, which is NOT recommended for general use.', '      Please consider using binary packages instead.', '']
        summary += [''] + lines + ['Please refer to the Installation Guide for details:', 'https://docs.cupy.dev/en/stable/install.html', '']
    summary += ['************************************************************', '']
    print('\n'.join(summary))
    return (ret, settings)

def _rpath_base():
    if False:
        i = 10
        return i + 15
    if PLATFORM_LINUX:
        return '$ORIGIN'
    else:
        raise Exception('not supported on this platform')

def _find_static_library(name: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    if PLATFORM_LINUX:
        filename = f'lib{name}.a'
        libdir = 'lib64'
    elif PLATFORM_WIN32:
        filename = f'{name}.lib'
        libdir = 'lib\\x64'
    else:
        raise Exception('not supported on this platform')
    cuda_path = build.get_cuda_path()
    if cuda_path is None:
        raise Exception(f'Could not find {filename}: CUDA path unavailable')
    path = os.path.join(cuda_path, libdir, filename)
    if not os.path.exists(path):
        raise Exception(f'Could not find {filename}: {path} does not exist')
    return path

def make_extensions(ctx: Context, compiler, use_cython):
    if False:
        for i in range(10):
            print('nop')
    'Produce a list of Extension instances which passed to cythonize().'
    MODULES = ctx.features.values()
    no_cuda = ctx.use_stub
    use_hip = not no_cuda and ctx.use_hip
    settings = build.get_compiler_setting(ctx, use_hip)
    include_dirs = settings['include_dirs']
    settings['include_dirs'] = [x for x in include_dirs if os.path.exists(x)]
    settings['library_dirs'] = [x for x in settings['library_dirs'] if os.path.exists(x)]
    use_wheel_libs_rpath = 0 < len(ctx.wheel_libs) and (not PLATFORM_WIN32)
    settings['define_macros'].append(('_FORCE_INLINES', '1'))
    if ctx.linetrace:
        settings['define_macros'].append(('CYTHON_TRACE', '1'))
        settings['define_macros'].append(('CYTHON_TRACE_NOGIL', '1'))
    if no_cuda:
        settings['define_macros'].append(('CUPY_NO_CUDA', '1'))
    if use_hip:
        settings['define_macros'].append(('CUPY_USE_HIP', '1'))
        settings['define_macros'].append(('__HIP_PLATFORM_AMD__', '1'))
        settings['define_macros'].append(('__HIP_PLATFORM_HCC__', '1'))
    available_modules = []
    if no_cuda:
        available_modules = [m['name'] for m in MODULES]
    else:
        (available_modules, settings) = preconfigure_modules(ctx, MODULES, compiler, settings)
        required_modules = get_required_modules(MODULES)
        if not set(required_modules) <= set(available_modules):
            raise Exception('Your CUDA environment is invalid. Please check above error log.')
    ret = []
    for module in MODULES:
        if module['name'] not in available_modules:
            continue
        s = copy.deepcopy(settings)
        if not no_cuda:
            s['libraries'] = module.libraries
            s['extra_objects'] = [_find_static_library(name) for name in module.static_libraries]
        compile_args = s.setdefault('extra_compile_args', [])
        link_args = s.setdefault('extra_link_args', [])
        if module['name'] == 'cusolver':
            compile_args.append('--std=c++11')
            if use_hip:
                pass
            elif compiler.compiler_type == 'unix':
                compile_args.append('-fopenmp')
                link_args.append('-fopenmp')
            elif compiler.compiler_type == 'msvc':
                compile_args.append('/openmp')
        if module['name'] == 'random':
            if compiler.compiler_type == 'msvc':
                compile_args.append('-D_USE_MATH_DEFINES')
        if module['name'] == 'jitify':
            compile_args.append('--std=c++11')
            s['depends'] = ['./cupy/_core/include/cupy/_jitify/jitify.hpp']
        if module['name'] == 'dlpack':
            s['depends'] = ['./cupy/_core/include/cupy/_dlpack/dlpack.h']
        for f in module['file']:
            s_file = copy.deepcopy(s)
            name = module_extension_name(f)
            if name.endswith('fft._callback') and (not PLATFORM_LINUX):
                continue
            rpath = []
            if not ctx.no_rpath:
                rpath += s_file['library_dirs']
            if use_wheel_libs_rpath:
                depth = name.count('.')
                rpath.append('{}{}/cupy/.data/lib'.format(_rpath_base(), '/..' * depth))
            if PLATFORM_LINUX and len(rpath) != 0:
                ldflag = '-Wl,'
                if PLATFORM_LINUX:
                    ldflag += '--disable-new-dtags,'
                ldflag += ','.join(('-rpath,' + p for p in rpath))
                args = s_file.setdefault('extra_link_args', [])
                args.append(ldflag)
            sources = module_extension_sources(f, use_cython, no_cuda)
            extension = setuptools.Extension(name, sources, **s_file)
            ret.append(extension)
    return ret

def prepare_wheel_libs(ctx: Context):
    if False:
        while True:
            i = 10
    'Prepare shared libraries and include files for wheels.\n\n    Shared libraries are placed under `cupy/.data/lib` and\n    RUNPATH will be set to this directory later (Linux only).\n    Include files are placed under `cupy/.data/include`.\n\n    Returns the list of files (path relative to `cupy` module) to add to\n    the sdist/wheel distribution.\n    '
    data_dir = os.path.abspath(os.path.join('cupy', '.data'))
    if os.path.exists(data_dir):
        print('Clearing directory: {}'.format(data_dir))
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)
    files_to_copy = []
    for srcpath in ctx.wheel_libs:
        relpath = os.path.basename(srcpath)
        dstpath = os.path.join(data_dir, 'lib', relpath)
        files_to_copy.append((srcpath, dstpath))
    for include_path_spec in ctx.wheel_includes:
        (srcpath, relpath) = include_path_spec.rsplit(':', 1)
        dstpath = os.path.join(data_dir, 'include', relpath)
        files_to_copy.append((srcpath, dstpath))
    if ctx.wheel_metadata_path:
        files_to_copy.append((ctx.wheel_metadata_path, os.path.join(data_dir, '_wheel.json')))
    for (srcpath, dstpath) in files_to_copy:
        print('Copying file for wheel: {}'.format(srcpath))
        dirpath = os.path.dirname(dstpath)
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        shutil.copy2(srcpath, dstpath)
    package_files = [x[1] for x in files_to_copy] + ['cupy/.data/_depends.json']
    return [os.path.relpath(f, 'cupy') for f in package_files]

def get_ext_modules(use_cython: bool, ctx: Context):
    if False:
        i = 10
        return i + 15
    sysconfig.get_config_vars()
    compiler = ccompiler.new_compiler()
    sysconfig.customize_compiler(compiler)
    extensions = make_extensions(ctx, compiler, use_cython)
    return extensions