import glob
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from sysconfig import get_config_var, get_config_vars, get_path
from .runners import CCompilerRunner, CppCompilerRunner, FortranCompilerRunner
from .util import get_abspath, make_dirs, copy, Glob, ArbitraryDepthGlob, glob_at_depth, import_module_from_file, pyx_is_cplus, sha256_of_string, sha256_of_file, CompileError
if os.name == 'posix':
    objext = '.o'
elif os.name == 'nt':
    objext = '.obj'
else:
    warnings.warn('Unknown os.name: {}'.format(os.name))
    objext = '.o'

def compile_sources(files, Runner=None, destdir=None, cwd=None, keep_dir_struct=False, per_file_kwargs=None, **kwargs):
    if False:
        i = 10
        return i + 15
    ' Compile source code files to object files.\n\n    Parameters\n    ==========\n\n    files : iterable of str\n        Paths to source files, if ``cwd`` is given, the paths are taken as relative.\n    Runner: CompilerRunner subclass (optional)\n        Could be e.g. ``FortranCompilerRunner``. Will be inferred from filename\n        extensions if missing.\n    destdir: str\n        Output directory, if cwd is given, the path is taken as relative.\n    cwd: str\n        Working directory. Specify to have compiler run in other directory.\n        also used as root of relative paths.\n    keep_dir_struct: bool\n        Reproduce directory structure in `destdir`. default: ``False``\n    per_file_kwargs: dict\n        Dict mapping instances in ``files`` to keyword arguments.\n    \\*\\*kwargs: dict\n        Default keyword arguments to pass to ``Runner``.\n\n    '
    _per_file_kwargs = {}
    if per_file_kwargs is not None:
        for (k, v) in per_file_kwargs.items():
            if isinstance(k, Glob):
                for path in glob.glob(k.pathname):
                    _per_file_kwargs[path] = v
            elif isinstance(k, ArbitraryDepthGlob):
                for path in glob_at_depth(k.filename, cwd):
                    _per_file_kwargs[path] = v
            else:
                _per_file_kwargs[k] = v
    destdir = destdir or '.'
    if not os.path.isdir(destdir):
        if os.path.exists(destdir):
            raise OSError('{} is not a directory'.format(destdir))
        else:
            make_dirs(destdir)
    if cwd is None:
        cwd = '.'
        for f in files:
            copy(f, destdir, only_update=True, dest_is_dir=True)
    dstpaths = []
    for f in files:
        if keep_dir_struct:
            (name, ext) = os.path.splitext(f)
        else:
            (name, ext) = os.path.splitext(os.path.basename(f))
        file_kwargs = kwargs.copy()
        file_kwargs.update(_per_file_kwargs.get(f, {}))
        dstpaths.append(src2obj(f, Runner, cwd=cwd, **file_kwargs))
    return dstpaths

def get_mixed_fort_c_linker(vendor=None, cplus=False, cwd=None):
    if False:
        while True:
            i = 10
    vendor = vendor or os.environ.get('SYMPY_COMPILER_VENDOR', 'gnu')
    if vendor.lower() == 'intel':
        if cplus:
            return (FortranCompilerRunner, {'flags': ['-nofor_main', '-cxxlib']}, vendor)
        else:
            return (FortranCompilerRunner, {'flags': ['-nofor_main']}, vendor)
    elif vendor.lower() == 'gnu' or 'llvm':
        if cplus:
            return (CppCompilerRunner, {'lib_options': ['fortran']}, vendor)
        else:
            return (FortranCompilerRunner, {}, vendor)
    else:
        raise ValueError('No vendor found.')

def link(obj_files, out_file=None, shared=False, Runner=None, cwd=None, cplus=False, fort=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    ' Link object files.\n\n    Parameters\n    ==========\n\n    obj_files: iterable of str\n        Paths to object files.\n    out_file: str (optional)\n        Path to executable/shared library, if ``None`` it will be\n        deduced from the last item in obj_files.\n    shared: bool\n        Generate a shared library?\n    Runner: CompilerRunner subclass (optional)\n        If not given the ``cplus`` and ``fort`` flags will be inspected\n        (fallback is the C compiler).\n    cwd: str\n        Path to the root of relative paths and working directory for compiler.\n    cplus: bool\n        C++ objects? default: ``False``.\n    fort: bool\n        Fortran objects? default: ``False``.\n    \\*\\*kwargs: dict\n        Keyword arguments passed to ``Runner``.\n\n    Returns\n    =======\n\n    The absolute path to the generated shared object / executable.\n\n    '
    if out_file is None:
        (out_file, ext) = os.path.splitext(os.path.basename(obj_files[-1]))
        if shared:
            out_file += get_config_var('EXT_SUFFIX')
    if not Runner:
        if fort:
            (Runner, extra_kwargs, vendor) = get_mixed_fort_c_linker(vendor=kwargs.get('vendor', None), cplus=cplus, cwd=cwd)
            for (k, v) in extra_kwargs.items():
                if k in kwargs:
                    kwargs[k].expand(v)
                else:
                    kwargs[k] = v
        elif cplus:
            Runner = CppCompilerRunner
        else:
            Runner = CCompilerRunner
    flags = kwargs.pop('flags', [])
    if shared:
        if '-shared' not in flags:
            flags.append('-shared')
    run_linker = kwargs.pop('run_linker', True)
    if not run_linker:
        raise ValueError('run_linker was set to False (nonsensical).')
    out_file = get_abspath(out_file, cwd=cwd)
    runner = Runner(obj_files, out_file, flags, cwd=cwd, **kwargs)
    runner.run()
    return out_file

def link_py_so(obj_files, so_file=None, cwd=None, libraries=None, cplus=False, fort=False, **kwargs):
    if False:
        while True:
            i = 10
    " Link Python extension module (shared object) for importing\n\n    Parameters\n    ==========\n\n    obj_files: iterable of str\n        Paths to object files to be linked.\n    so_file: str\n        Name (path) of shared object file to create. If not specified it will\n        have the basname of the last object file in `obj_files` but with the\n        extension '.so' (Unix).\n    cwd: path string\n        Root of relative paths and working directory of linker.\n    libraries: iterable of strings\n        Libraries to link against, e.g. ['m'].\n    cplus: bool\n        Any C++ objects? default: ``False``.\n    fort: bool\n        Any Fortran objects? default: ``False``.\n    kwargs**: dict\n        Keyword arguments passed to ``link(...)``.\n\n    Returns\n    =======\n\n    Absolute path to the generate shared object.\n    "
    libraries = libraries or []
    include_dirs = kwargs.pop('include_dirs', [])
    library_dirs = kwargs.pop('library_dirs', [])
    if sys.platform == 'win32':
        warnings.warn('Windows not yet supported.')
    elif sys.platform == 'darwin':
        cfgDict = get_config_vars()
        kwargs['linkline'] = kwargs.get('linkline', []) + [cfgDict['LDFLAGS']]
        library_dirs += [cfgDict['LIBDIR']]
        is_framework = False
        for opt in cfgDict['LIBS'].split():
            if is_framework:
                kwargs['linkline'] = kwargs.get('linkline', []) + ['-framework', opt]
                is_framework = False
            elif opt.startswith('-l'):
                libraries.append(opt[2:])
            elif opt.startswith('-framework'):
                is_framework = True
        libfile = cfgDict['LIBRARY']
        libname = '.'.join(libfile.split('.')[:-1])[3:]
        libraries.append(libname)
    elif sys.platform[:3] == 'aix':
        pass
    elif get_config_var('Py_ENABLE_SHARED'):
        cfgDict = get_config_vars()
        kwargs['linkline'] = kwargs.get('linkline', []) + [cfgDict['LDFLAGS']]
        library_dirs += [cfgDict['LIBDIR']]
        for opt in cfgDict['BLDLIBRARY'].split():
            if opt.startswith('-l'):
                libraries += [opt[2:]]
    else:
        pass
    flags = kwargs.pop('flags', [])
    needed_flags = ('-pthread',)
    for flag in needed_flags:
        if flag not in flags:
            flags.append(flag)
    return link(obj_files, shared=True, flags=flags, cwd=cwd, cplus=cplus, fort=fort, include_dirs=include_dirs, libraries=libraries, library_dirs=library_dirs, **kwargs)

def simple_cythonize(src, destdir=None, cwd=None, **cy_kwargs):
    if False:
        for i in range(10):
            print('nop')
    " Generates a C file from a Cython source file.\n\n    Parameters\n    ==========\n\n    src: str\n        Path to Cython source.\n    destdir: str (optional)\n        Path to output directory (default: '.').\n    cwd: path string (optional)\n        Root of relative paths (default: '.').\n    **cy_kwargs:\n        Second argument passed to cy_compile. Generates a .cpp file if ``cplus=True`` in ``cy_kwargs``,\n        else a .c file.\n    "
    from Cython.Compiler.Main import default_options, CompilationOptions
    from Cython.Compiler.Main import compile as cy_compile
    assert src.lower().endswith('.pyx') or src.lower().endswith('.py')
    cwd = cwd or '.'
    destdir = destdir or '.'
    ext = '.cpp' if cy_kwargs.get('cplus', False) else '.c'
    c_name = os.path.splitext(os.path.basename(src))[0] + ext
    dstfile = os.path.join(destdir, c_name)
    if cwd:
        ori_dir = os.getcwd()
    else:
        ori_dir = '.'
    os.chdir(cwd)
    try:
        cy_options = CompilationOptions(default_options)
        cy_options.__dict__.update(cy_kwargs)
        if 'language_level' not in cy_kwargs:
            cy_options.__dict__['language_level'] = 3
        cy_result = cy_compile([src], cy_options)
        if cy_result.num_errors > 0:
            raise ValueError('Cython compilation failed.')
        if os.path.realpath(os.path.dirname(src)) != os.path.realpath(destdir):
            if os.path.exists(dstfile):
                os.unlink(dstfile)
            shutil.move(os.path.join(os.path.dirname(src), c_name), destdir)
    finally:
        os.chdir(ori_dir)
    return dstfile
extension_mapping = {'.c': (CCompilerRunner, None), '.cpp': (CppCompilerRunner, None), '.cxx': (CppCompilerRunner, None), '.f': (FortranCompilerRunner, None), '.for': (FortranCompilerRunner, None), '.ftn': (FortranCompilerRunner, None), '.f90': (FortranCompilerRunner, None), '.f95': (FortranCompilerRunner, 'f95'), '.f03': (FortranCompilerRunner, 'f2003'), '.f08': (FortranCompilerRunner, 'f2008')}

def src2obj(srcpath, Runner=None, objpath=None, cwd=None, inc_py=False, **kwargs):
    if False:
        print('Hello World!')
    ' Compiles a source code file to an object file.\n\n    Files ending with \'.pyx\' assumed to be cython files and\n    are dispatched to pyx2obj.\n\n    Parameters\n    ==========\n\n    srcpath: str\n        Path to source file.\n    Runner: CompilerRunner subclass (optional)\n        If ``None``: deduced from extension of srcpath.\n    objpath : str (optional)\n        Path to generated object. If ``None``: deduced from ``srcpath``.\n    cwd: str (optional)\n        Working directory and root of relative paths. If ``None``: current dir.\n    inc_py: bool\n        Add Python include path to kwarg "include_dirs". Default: False\n    \\*\\*kwargs: dict\n        keyword arguments passed to Runner or pyx2obj\n\n    '
    (name, ext) = os.path.splitext(os.path.basename(srcpath))
    if objpath is None:
        if os.path.isabs(srcpath):
            objpath = '.'
        else:
            objpath = os.path.dirname(srcpath)
            objpath = objpath or '.'
    if os.path.isdir(objpath):
        objpath = os.path.join(objpath, name + objext)
    include_dirs = kwargs.pop('include_dirs', [])
    if inc_py:
        py_inc_dir = get_path('include')
        if py_inc_dir not in include_dirs:
            include_dirs.append(py_inc_dir)
    if ext.lower() == '.pyx':
        return pyx2obj(srcpath, objpath=objpath, include_dirs=include_dirs, cwd=cwd, **kwargs)
    if Runner is None:
        (Runner, std) = extension_mapping[ext.lower()]
        if 'std' not in kwargs:
            kwargs['std'] = std
    flags = kwargs.pop('flags', [])
    needed_flags = ('-fPIC',)
    for flag in needed_flags:
        if flag not in flags:
            flags.append(flag)
    run_linker = kwargs.pop('run_linker', False)
    if run_linker:
        raise CompileError('src2obj called with run_linker=True')
    runner = Runner([srcpath], objpath, include_dirs=include_dirs, run_linker=run_linker, cwd=cwd, flags=flags, **kwargs)
    runner.run()
    return objpath

def pyx2obj(pyxpath, objpath=None, destdir=None, cwd=None, include_dirs=None, cy_kwargs=None, cplus=None, **kwargs):
    if False:
        return 10
    "\n    Convenience function\n\n    If cwd is specified, pyxpath and dst are taken to be relative\n    If only_update is set to `True` the modification time is checked\n    and compilation is only run if the source is newer than the\n    destination\n\n    Parameters\n    ==========\n\n    pyxpath: str\n        Path to Cython source file.\n    objpath: str (optional)\n        Path to object file to generate.\n    destdir: str (optional)\n        Directory to put generated C file. When ``None``: directory of ``objpath``.\n    cwd: str (optional)\n        Working directory and root of relative paths.\n    include_dirs: iterable of path strings (optional)\n        Passed onto src2obj and via cy_kwargs['include_path']\n        to simple_cythonize.\n    cy_kwargs: dict (optional)\n        Keyword arguments passed onto `simple_cythonize`\n    cplus: bool (optional)\n        Indicate whether C++ is used. default: auto-detect using ``.util.pyx_is_cplus``.\n    compile_kwargs: dict\n        keyword arguments passed onto src2obj\n\n    Returns\n    =======\n\n    Absolute path of generated object file.\n\n    "
    assert pyxpath.endswith('.pyx')
    cwd = cwd or '.'
    objpath = objpath or '.'
    destdir = destdir or os.path.dirname(objpath)
    abs_objpath = get_abspath(objpath, cwd=cwd)
    if os.path.isdir(abs_objpath):
        pyx_fname = os.path.basename(pyxpath)
        (name, ext) = os.path.splitext(pyx_fname)
        objpath = os.path.join(objpath, name + objext)
    cy_kwargs = cy_kwargs or {}
    cy_kwargs['output_dir'] = cwd
    if cplus is None:
        cplus = pyx_is_cplus(pyxpath)
    cy_kwargs['cplus'] = cplus
    interm_c_file = simple_cythonize(pyxpath, destdir=destdir, cwd=cwd, **cy_kwargs)
    include_dirs = include_dirs or []
    flags = kwargs.pop('flags', [])
    needed_flags = ('-fwrapv', '-pthread', '-fPIC')
    for flag in needed_flags:
        if flag not in flags:
            flags.append(flag)
    options = kwargs.pop('options', [])
    if kwargs.pop('strict_aliasing', False):
        raise CompileError('Cython requires strict aliasing to be disabled.')
    if cplus:
        std = kwargs.pop('std', 'c++98')
    else:
        std = kwargs.pop('std', 'c99')
    return src2obj(interm_c_file, objpath=objpath, cwd=cwd, include_dirs=include_dirs, flags=flags, std=std, options=options, inc_py=True, strict_aliasing=False, **kwargs)

def _any_X(srcs, cls):
    if False:
        return 10
    for src in srcs:
        (name, ext) = os.path.splitext(src)
        key = ext.lower()
        if key in extension_mapping:
            if extension_mapping[key][0] == cls:
                return True
    return False

def any_fortran_src(srcs):
    if False:
        for i in range(10):
            print('nop')
    return _any_X(srcs, FortranCompilerRunner)

def any_cplus_src(srcs):
    if False:
        while True:
            i = 10
    return _any_X(srcs, CppCompilerRunner)

def compile_link_import_py_ext(sources, extname=None, build_dir='.', compile_kwargs=None, link_kwargs=None):
    if False:
        while True:
            i = 10
    ' Compiles sources to a shared object (Python extension) and imports it\n\n    Sources in ``sources`` which is imported. If shared object is newer than the sources, they\n    are not recompiled but instead it is imported.\n\n    Parameters\n    ==========\n\n    sources : string\n        List of paths to sources.\n    extname : string\n        Name of extension (default: ``None``).\n        If ``None``: taken from the last file in ``sources`` without extension.\n    build_dir: str\n        Path to directory in which objects files etc. are generated.\n    compile_kwargs: dict\n        keyword arguments passed to ``compile_sources``\n    link_kwargs: dict\n        keyword arguments passed to ``link_py_so``\n\n    Returns\n    =======\n\n    The imported module from of the Python extension.\n    '
    if extname is None:
        extname = os.path.splitext(os.path.basename(sources[-1]))[0]
    compile_kwargs = compile_kwargs or {}
    link_kwargs = link_kwargs or {}
    try:
        mod = import_module_from_file(os.path.join(build_dir, extname), sources)
    except ImportError:
        objs = compile_sources(list(map(get_abspath, sources)), destdir=build_dir, cwd=build_dir, **compile_kwargs)
        so = link_py_so(objs, cwd=build_dir, fort=any_fortran_src(sources), cplus=any_cplus_src(sources), **link_kwargs)
        mod = import_module_from_file(so)
    return mod

def _write_sources_to_build_dir(sources, build_dir):
    if False:
        return 10
    build_dir = build_dir or tempfile.mkdtemp()
    if not os.path.isdir(build_dir):
        raise OSError('Non-existent directory: ', build_dir)
    source_files = []
    for (name, src) in sources:
        dest = os.path.join(build_dir, name)
        differs = True
        sha256_in_mem = sha256_of_string(src.encode('utf-8')).hexdigest()
        if os.path.exists(dest):
            if os.path.exists(dest + '.sha256'):
                with open(dest + '.sha256') as fh:
                    sha256_on_disk = fh.read()
            else:
                sha256_on_disk = sha256_of_file(dest).hexdigest()
            differs = sha256_on_disk != sha256_in_mem
        if differs:
            with open(dest, 'wt') as fh:
                fh.write(src)
            with open(dest + '.sha256', 'wt') as fh:
                fh.write(sha256_in_mem)
        source_files.append(dest)
    return (source_files, build_dir)

def compile_link_import_strings(sources, build_dir=None, **kwargs):
    if False:
        return 10
    " Compiles, links and imports extension module from source.\n\n    Parameters\n    ==========\n\n    sources : iterable of name/source pair tuples\n    build_dir : string (default: None)\n        Path. ``None`` implies use a temporary directory.\n    **kwargs:\n        Keyword arguments passed onto `compile_link_import_py_ext`.\n\n    Returns\n    =======\n\n    mod : module\n        The compiled and imported extension module.\n    info : dict\n        Containing ``build_dir`` as 'build_dir'.\n\n    "
    (source_files, build_dir) = _write_sources_to_build_dir(sources, build_dir)
    mod = compile_link_import_py_ext(source_files, build_dir=build_dir, **kwargs)
    info = {'build_dir': build_dir}
    return (mod, info)

def compile_run_strings(sources, build_dir=None, clean=False, compile_kwargs=None, link_kwargs=None):
    if False:
        for i in range(10):
            print('nop')
    " Compiles, links and runs a program built from sources.\n\n    Parameters\n    ==========\n\n    sources : iterable of name/source pair tuples\n    build_dir : string (default: None)\n        Path. ``None`` implies use a temporary directory.\n    clean : bool\n        Whether to remove build_dir after use. This will only have an\n        effect if ``build_dir`` is ``None`` (which creates a temporary directory).\n        Passing ``clean == True`` and ``build_dir != None`` raises a ``ValueError``.\n        This will also set ``build_dir`` in returned info dictionary to ``None``.\n    compile_kwargs: dict\n        Keyword arguments passed onto ``compile_sources``\n    link_kwargs: dict\n        Keyword arguments passed onto ``link``\n\n    Returns\n    =======\n\n    (stdout, stderr): pair of strings\n    info: dict\n        Containing exit status as 'exit_status' and ``build_dir`` as 'build_dir'\n\n    "
    if clean and build_dir is not None:
        raise ValueError('Automatic removal of build_dir is only available for temporary directory.')
    try:
        (source_files, build_dir) = _write_sources_to_build_dir(sources, build_dir)
        objs = compile_sources(list(map(get_abspath, source_files)), destdir=build_dir, cwd=build_dir, **compile_kwargs or {})
        prog = link(objs, cwd=build_dir, fort=any_fortran_src(source_files), cplus=any_cplus_src(source_files), **link_kwargs or {})
        p = subprocess.Popen([prog], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        exit_status = p.wait()
        (stdout, stderr) = [txt.decode('utf-8') for txt in p.communicate()]
    finally:
        if clean and os.path.isdir(build_dir):
            shutil.rmtree(build_dir)
            build_dir = None
    info = {'exit_status': exit_status, 'build_dir': build_dir}
    return ((stdout, stderr), info)