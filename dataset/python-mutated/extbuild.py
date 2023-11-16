"""
Build a c-extension module on-the-fly in tests.
See build_and_import_extensions for usage hints

"""
import os
import pathlib
import subprocess
import sys
import sysconfig
import textwrap
__all__ = ['build_and_import_extension', 'compile_extension_module']

def build_and_import_extension(modname, functions, *, prologue='', build_dir=None, include_dirs=[], more_init=''):
    if False:
        return 10
    '\n    Build and imports a c-extension module `modname` from a list of function\n    fragments `functions`.\n\n\n    Parameters\n    ----------\n    functions : list of fragments\n        Each fragment is a sequence of func_name, calling convention, snippet.\n    prologue : string\n        Code to precede the rest, usually extra ``#include`` or ``#define``\n        macros.\n    build_dir : pathlib.Path\n        Where to build the module, usually a temporary directory\n    include_dirs : list\n        Extra directories to find include files when compiling\n    more_init : string\n        Code to appear in the module PyMODINIT_FUNC\n\n    Returns\n    -------\n    out: module\n        The module will have been loaded and is ready for use\n\n    Examples\n    --------\n    >>> functions = [("test_bytes", "METH_O", """\n        if ( !PyBytesCheck(args)) {\n            Py_RETURN_FALSE;\n        }\n        Py_RETURN_TRUE;\n    """)]\n    >>> mod = build_and_import_extension("testme", functions)\n    >>> assert not mod.test_bytes(u\'abc\')\n    >>> assert mod.test_bytes(b\'abc\')\n    '
    body = prologue + _make_methods(functions, modname)
    init = 'PyObject *mod = PyModule_Create(&moduledef);\n           '
    if not build_dir:
        build_dir = pathlib.Path('.')
    if more_init:
        init += '#define INITERROR return NULL\n                '
        init += more_init
    init += '\nreturn mod;'
    source_string = _make_source(modname, init, body)
    try:
        mod_so = compile_extension_module(modname, build_dir, include_dirs, source_string)
    except Exception as e:
        raise RuntimeError(f'could not compile in {build_dir}:') from e
    import importlib.util
    spec = importlib.util.spec_from_file_location(modname, mod_so)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo

def compile_extension_module(name, builddir, include_dirs, source_string, libraries=[], library_dirs=[]):
    if False:
        for i in range(10):
            print('nop')
    '\n    Build an extension module and return the filename of the resulting\n    native code file.\n\n    Parameters\n    ----------\n    name : string\n        name of the module, possibly including dots if it is a module inside a\n        package.\n    builddir : pathlib.Path\n        Where to build the module, usually a temporary directory\n    include_dirs : list\n        Extra directories to find include files when compiling\n    libraries : list\n        Libraries to link into the extension module\n    library_dirs: list\n        Where to find the libraries, ``-L`` passed to the linker\n    '
    modname = name.split('.')[-1]
    dirname = builddir / name
    dirname.mkdir(exist_ok=True)
    cfile = _convert_str_to_file(source_string, dirname)
    include_dirs = include_dirs + [sysconfig.get_config_var('INCLUDEPY')]
    return _c_compile(cfile, outputfilename=dirname / modname, include_dirs=include_dirs, libraries=[], library_dirs=[])

def _convert_str_to_file(source, dirname):
    if False:
        print('Hello World!')
    'Helper function to create a file ``source.c`` in `dirname` that contains\n    the string in `source`. Returns the file name\n    '
    filename = dirname / 'source.c'
    with filename.open('w') as f:
        f.write(str(source))
    return filename

def _make_methods(functions, modname):
    if False:
        while True:
            i = 10
    ' Turns the name, signature, code in functions into complete functions\n    and lists them in a methods_table. Then turns the methods_table into a\n    ``PyMethodDef`` structure and returns the resulting code fragment ready\n    for compilation\n    '
    methods_table = []
    codes = []
    for (funcname, flags, code) in functions:
        cfuncname = '%s_%s' % (modname, funcname)
        if 'METH_KEYWORDS' in flags:
            signature = '(PyObject *self, PyObject *args, PyObject *kwargs)'
        else:
            signature = '(PyObject *self, PyObject *args)'
        methods_table.append('{"%s", (PyCFunction)%s, %s},' % (funcname, cfuncname, flags))
        func_code = '\n        static PyObject* {cfuncname}{signature}\n        {{\n        {code}\n        }}\n        '.format(cfuncname=cfuncname, signature=signature, code=code)
        codes.append(func_code)
    body = '\n'.join(codes) + '\n    static PyMethodDef methods[] = {\n    %(methods)s\n    { NULL }\n    };\n    static struct PyModuleDef moduledef = {\n        PyModuleDef_HEAD_INIT,\n        "%(modname)s",  /* m_name */\n        NULL,           /* m_doc */\n        -1,             /* m_size */\n        methods,        /* m_methods */\n    };\n    ' % dict(methods='\n'.join(methods_table), modname=modname)
    return body

def _make_source(name, init, body):
    if False:
        for i in range(10):
            print('nop')
    ' Combines the code fragments into source code ready to be compiled\n    '
    code = '\n    #include <Python.h>\n\n    %(body)s\n\n    PyMODINIT_FUNC\n    PyInit_%(name)s(void) {\n    %(init)s\n    }\n    ' % dict(name=name, init=init, body=body)
    return code

def _c_compile(cfile, outputfilename, include_dirs=[], libraries=[], library_dirs=[]):
    if False:
        i = 10
        return i + 15
    if sys.platform == 'win32':
        compile_extra = ['/we4013']
        link_extra = ['/LIBPATH:' + os.path.join(sys.base_prefix, 'libs')]
    elif sys.platform.startswith('linux'):
        compile_extra = ['-O0', '-g', '-Werror=implicit-function-declaration', '-fPIC']
        link_extra = []
    else:
        compile_extra = link_extra = []
        pass
    if sys.platform == 'win32':
        link_extra = link_extra + ['/DEBUG']
    if sys.platform == 'darwin':
        for s in ('/sw/', '/opt/local/'):
            if s + 'include' not in include_dirs and os.path.exists(s + 'include'):
                include_dirs.append(s + 'include')
            if s + 'lib' not in library_dirs and os.path.exists(s + 'lib'):
                library_dirs.append(s + 'lib')
    outputfilename = outputfilename.with_suffix(get_so_suffix())
    build(cfile, outputfilename, compile_extra, link_extra, include_dirs, libraries, library_dirs)
    return outputfilename

def build(cfile, outputfilename, compile_extra, link_extra, include_dirs, libraries, library_dirs):
    if False:
        for i in range(10):
            print('nop')
    'use meson to build'
    build_dir = cfile.parent / 'build'
    os.makedirs(build_dir, exist_ok=True)
    so_name = outputfilename.parts[-1]
    with open(cfile.parent / 'meson.build', 'wt') as fid:
        includes = ['-I' + d for d in include_dirs]
        link_dirs = ['-L' + d for d in library_dirs]
        fid.write(textwrap.dedent(f"            project('foo', 'c')\n            shared_module('{so_name}', '{cfile.parts[-1]}',\n                c_args: {includes} + {compile_extra},\n                link_args: {link_dirs} + {link_extra},\n                link_with: {libraries},\n                name_prefix: '',\n                name_suffix: 'dummy',\n            )\n        "))
    if sys.platform == 'win32':
        subprocess.check_call(['meson', 'setup', '--buildtype=release', '--vsenv', '..'], cwd=build_dir)
    else:
        subprocess.check_call(['meson', 'setup', '--vsenv', '..'], cwd=build_dir)
    subprocess.check_call(['meson', 'compile'], cwd=build_dir)
    os.rename(str(build_dir / so_name) + '.dummy', cfile.parent / so_name)

def get_so_suffix():
    if False:
        while True:
            i = 10
    ret = sysconfig.get_config_var('EXT_SUFFIX')
    assert ret
    return ret