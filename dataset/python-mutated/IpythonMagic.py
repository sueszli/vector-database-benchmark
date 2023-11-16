"""
=====================
Cython related magics
=====================

Magic command interface for interactive work with Cython

.. note::

  The ``Cython`` package needs to be installed separately. It
  can be obtained using ``easy_install`` or ``pip``.

Usage
=====

To enable the magics below, execute ``%load_ext cython``.

``%%cython``

{CYTHON_DOC}

``%%cython_inline``

{CYTHON_INLINE_DOC}

``%%cython_pyximport``

{CYTHON_PYXIMPORT_DOC}

Author:
* Brian Granger

Code moved from IPython and adapted by:
* Martín Gaitán

Parts of this code were taken from Cython.inline.
"""
from __future__ import absolute_import, print_function
import io
import os
import re
import sys
import time
import copy
import distutils.log
import textwrap
IO_ENCODING = sys.getfilesystemencoding()
IS_PY2 = sys.version_info[0] < 3
import hashlib
from distutils.core import Distribution, Extension
from distutils.command.build_ext import build_ext
from IPython.core import display
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, cell_magic
try:
    from IPython.paths import get_ipython_cache_dir
except ImportError:
    from IPython.utils.path import get_ipython_cache_dir
from IPython.utils.text import dedent
from ..Shadow import __version__ as cython_version
from ..Compiler.Errors import CompileError
from .Inline import cython_inline, load_dynamic
from .Dependencies import cythonize
from ..Utils import captured_fd, print_captured
PGO_CONFIG = {'gcc': {'gen': ['-fprofile-generate', '-fprofile-dir={TEMPDIR}'], 'use': ['-fprofile-use', '-fprofile-correction', '-fprofile-dir={TEMPDIR}']}, 'icc': {'gen': ['-prof-gen'], 'use': ['-prof-use']}}
PGO_CONFIG['mingw32'] = PGO_CONFIG['gcc']
if IS_PY2:

    def encode_fs(name):
        if False:
            print('Hello World!')
        return name if isinstance(name, bytes) else name.encode(IO_ENCODING)
else:

    def encode_fs(name):
        if False:
            return 10
        return name

@magics_class
class CythonMagics(Magics):

    def __init__(self, shell):
        if False:
            while True:
                i = 10
        super(CythonMagics, self).__init__(shell)
        self._reloads = {}
        self._code_cache = {}
        self._pyximport_installed = False

    def _import_all(self, module):
        if False:
            i = 10
            return i + 15
        mdict = module.__dict__
        if '__all__' in mdict:
            keys = mdict['__all__']
        else:
            keys = [k for k in mdict if not k.startswith('_')]
        for k in keys:
            try:
                self.shell.push({k: mdict[k]})
            except KeyError:
                msg = "'module' object has no attribute '%s'" % k
                raise AttributeError(msg)

    @cell_magic
    def cython_inline(self, line, cell):
        if False:
            for i in range(10):
                print('nop')
        "Compile and run a Cython code cell using Cython.inline.\n\n        This magic simply passes the body of the cell to Cython.inline\n        and returns the result. If the variables `a` and `b` are defined\n        in the user's namespace, here is a simple example that returns\n        their sum::\n\n            %%cython_inline\n            return a+b\n\n        For most purposes, we recommend the usage of the `%%cython` magic.\n        "
        locs = self.shell.user_global_ns
        globs = self.shell.user_ns
        return cython_inline(cell, locals=locs, globals=globs)

    @cell_magic
    def cython_pyximport(self, line, cell):
        if False:
            return 10
        "Compile and import a Cython code cell using pyximport.\n\n        The contents of the cell are written to a `.pyx` file in the current\n        working directory, which is then imported using `pyximport`. This\n        magic requires a module name to be passed::\n\n            %%cython_pyximport modulename\n            def f(x):\n                return 2.0*x\n\n        The compiled module is then imported and all of its symbols are\n        injected into the user's namespace. For most purposes, we recommend\n        the usage of the `%%cython` magic.\n        "
        module_name = line.strip()
        if not module_name:
            raise ValueError('module name must be given')
        fname = module_name + '.pyx'
        with io.open(fname, 'w', encoding='utf-8') as f:
            f.write(cell)
        if 'pyximport' not in sys.modules or not self._pyximport_installed:
            import pyximport
            pyximport.install()
            self._pyximport_installed = True
        if module_name in self._reloads:
            module = self._reloads[module_name]
        else:
            __import__(module_name)
            module = sys.modules[module_name]
            self._reloads[module_name] = module
        self._import_all(module)

    @magic_arguments.magic_arguments()
    @magic_arguments.argument('-a', '--annotate', action='store_const', const='default', dest='annotate', help='Produce a colorized HTML version of the source.')
    @magic_arguments.argument('--annotate-fullc', action='store_const', const='fullc', dest='annotate', help='Produce a colorized HTML version of the source which includes entire generated C/C++-code.')
    @magic_arguments.argument('-+', '--cplus', action='store_true', default=False, help='Output a C++ rather than C file.')
    @magic_arguments.argument('-3', dest='language_level', action='store_const', const=3, default=None, help='Select Python 3 syntax.')
    @magic_arguments.argument('-2', dest='language_level', action='store_const', const=2, default=None, help='Select Python 2 syntax.')
    @magic_arguments.argument('-f', '--force', action='store_true', default=False, help='Force the compilation of a new module, even if the source has been previously compiled.')
    @magic_arguments.argument('-c', '--compile-args', action='append', default=[], help='Extra flags to pass to compiler via the `extra_compile_args` Extension flag (can be specified  multiple times).')
    @magic_arguments.argument('--link-args', action='append', default=[], help='Extra flags to pass to linker via the `extra_link_args` Extension flag (can be specified  multiple times).')
    @magic_arguments.argument('-l', '--lib', action='append', default=[], help='Add a library to link the extension against (can be specified multiple times).')
    @magic_arguments.argument('-n', '--name', help='Specify a name for the Cython module.')
    @magic_arguments.argument('-L', dest='library_dirs', metavar='dir', action='append', default=[], help='Add a path to the list of library directories (can be specified multiple times).')
    @magic_arguments.argument('-I', '--include', action='append', default=[], help='Add a path to the list of include directories (can be specified multiple times).')
    @magic_arguments.argument('-S', '--src', action='append', default=[], help='Add a path to the list of src files (can be specified multiple times).')
    @magic_arguments.argument('--pgo', dest='pgo', action='store_true', default=False, help='Enable profile guided optimisation in the C compiler. Compiles the cell twice and executes it in between to generate a runtime profile.')
    @magic_arguments.argument('--verbose', dest='quiet', action='store_false', default=True, help='Print debug information like generated .c/.cpp file location and exact gcc/g++ command invoked.')
    @cell_magic
    def cython(self, line, cell):
        if False:
            return 10
        'Compile and import everything from a Cython code cell.\n\n        The contents of the cell are written to a `.pyx` file in the\n        directory `IPYTHONDIR/cython` using a filename with the hash of the\n        code. This file is then cythonized and compiled. The resulting module\n        is imported and all of its symbols are injected into the user\'s\n        namespace. The usage is similar to that of `%%cython_pyximport` but\n        you don\'t have to pass a module name::\n\n            %%cython\n            def f(x):\n                return 2.0*x\n\n        To compile OpenMP codes, pass the required  `--compile-args`\n        and `--link-args`.  For example with gcc::\n\n            %%cython --compile-args=-fopenmp --link-args=-fopenmp\n            ...\n\n        To enable profile guided optimisation, pass the ``--pgo`` option.\n        Note that the cell itself needs to take care of establishing a suitable\n        profile when executed. This can be done by implementing the functions to\n        optimise, and then calling them directly in the same cell on some realistic\n        training data like this::\n\n            %%cython --pgo\n            def critical_function(data):\n                for item in data:\n                    ...\n\n            # execute function several times to build profile\n            from somewhere import some_typical_data\n            for _ in range(100):\n                critical_function(some_typical_data)\n\n        In Python 3.5 and later, you can distinguish between the profile and\n        non-profile runs as follows::\n\n            if "_pgo_" in __name__:\n                ...  # execute critical code here\n        '
        args = magic_arguments.parse_argstring(self.cython, line)
        code = cell if cell.endswith('\n') else cell + '\n'
        lib_dir = os.path.join(get_ipython_cache_dir(), 'cython')
        key = (code, line, sys.version_info, sys.executable, cython_version)
        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)
        if args.pgo:
            key += ('pgo',)
        if args.force:
            key += (time.time(),)
        if args.name:
            module_name = str(args.name)
        else:
            module_name = '_cython_magic_' + hashlib.sha1(str(key).encode('utf-8')).hexdigest()
        html_file = os.path.join(lib_dir, module_name + '.html')
        module_path = os.path.join(lib_dir, module_name + self.so_ext)
        have_module = os.path.isfile(module_path)
        need_cythonize = args.pgo or not have_module
        if args.annotate:
            if not os.path.isfile(html_file):
                need_cythonize = True
        extension = None
        if need_cythonize:
            extensions = self._cythonize(module_name, code, lib_dir, args, quiet=args.quiet)
            if extensions is None:
                return None
            assert len(extensions) == 1
            extension = extensions[0]
            self._code_cache[key] = module_name
            if args.pgo:
                self._profile_pgo_wrapper(extension, lib_dir)

        def print_compiler_output(stdout, stderr, where):
            if False:
                while True:
                    i = 10
            print_captured(stdout, where, u'Content of stdout:\n')
            print_captured(stderr, where, u'Content of stderr:\n')
        get_stderr = get_stdout = None
        try:
            with captured_fd(1) as get_stdout:
                with captured_fd(2) as get_stderr:
                    self._build_extension(extension, lib_dir, pgo_step_name='use' if args.pgo else None, quiet=args.quiet)
        except (distutils.errors.CompileError, distutils.errors.LinkError):
            print_compiler_output(get_stdout(), get_stderr(), sys.stderr)
            return None
        print_compiler_output(get_stdout(), get_stderr(), sys.stdout)
        module = load_dynamic(module_name, module_path)
        self._import_all(module)
        if args.annotate:
            try:
                with io.open(html_file, encoding='utf-8') as f:
                    annotated_html = f.read()
            except IOError as e:
                print('Cython completed successfully but the annotated source could not be read.', file=sys.stderr)
                print(e, file=sys.stderr)
            else:
                return display.HTML(self.clean_annotated_html(annotated_html))

    def _profile_pgo_wrapper(self, extension, lib_dir):
        if False:
            print('Hello World!')
        '\n        Generate a .c file for a separate extension module that calls the\n        module init function of the original module.  This makes sure that the\n        PGO profiler sees the correct .o file of the final module, but it still\n        allows us to import the module under a different name for profiling,\n        before recompiling it into the PGO optimised module.  Overwriting and\n        reimporting the same shared library is not portable.\n        '
        extension = copy.copy(extension)
        module_name = extension.name
        pgo_module_name = '_pgo_' + module_name
        pgo_wrapper_c_file = os.path.join(lib_dir, pgo_module_name + '.c')
        with io.open(pgo_wrapper_c_file, 'w', encoding='utf-8') as f:
            f.write(textwrap.dedent(u'\n            #include "Python.h"\n            #if PY_MAJOR_VERSION < 3\n            extern PyMODINIT_FUNC init%(module_name)s(void);\n            PyMODINIT_FUNC init%(pgo_module_name)s(void); /*proto*/\n            PyMODINIT_FUNC init%(pgo_module_name)s(void) {\n                PyObject *sys_modules;\n                init%(module_name)s();  if (PyErr_Occurred()) return;\n                sys_modules = PyImport_GetModuleDict();  /* borrowed, no exception, "never" fails */\n                if (sys_modules) {\n                    PyObject *module = PyDict_GetItemString(sys_modules, "%(module_name)s");  if (!module) return;\n                    PyDict_SetItemString(sys_modules, "%(pgo_module_name)s", module);\n                    Py_DECREF(module);\n                }\n            }\n            #else\n            extern PyMODINIT_FUNC PyInit_%(module_name)s(void);\n            PyMODINIT_FUNC PyInit_%(pgo_module_name)s(void); /*proto*/\n            PyMODINIT_FUNC PyInit_%(pgo_module_name)s(void) {\n                return PyInit_%(module_name)s();\n            }\n            #endif\n            ' % {'module_name': module_name, 'pgo_module_name': pgo_module_name}))
        extension.sources = extension.sources + [pgo_wrapper_c_file]
        extension.name = pgo_module_name
        self._build_extension(extension, lib_dir, pgo_step_name='gen')
        so_module_path = os.path.join(lib_dir, pgo_module_name + self.so_ext)
        load_dynamic(pgo_module_name, so_module_path)

    def _cythonize(self, module_name, code, lib_dir, args, quiet=True):
        if False:
            for i in range(10):
                print('nop')
        pyx_file = os.path.join(lib_dir, module_name + '.pyx')
        pyx_file = encode_fs(pyx_file)
        c_include_dirs = args.include
        c_src_files = list(map(str, args.src))
        if 'numpy' in code:
            import numpy
            c_include_dirs.append(numpy.get_include())
        with io.open(pyx_file, 'w', encoding='utf-8') as f:
            f.write(code)
        extension = Extension(name=module_name, sources=[pyx_file] + c_src_files, include_dirs=c_include_dirs, library_dirs=args.library_dirs, extra_compile_args=args.compile_args, extra_link_args=args.link_args, libraries=args.lib, language='c++' if args.cplus else 'c')
        try:
            opts = dict(quiet=quiet, annotate=args.annotate, force=True, language_level=min(3, sys.version_info[0]))
            if args.language_level is not None:
                assert args.language_level in (2, 3)
                opts['language_level'] = args.language_level
            return cythonize([extension], **opts)
        except CompileError:
            return None

    def _build_extension(self, extension, lib_dir, temp_dir=None, pgo_step_name=None, quiet=True):
        if False:
            print('Hello World!')
        build_extension = self._get_build_extension(extension, lib_dir=lib_dir, temp_dir=temp_dir, pgo_step_name=pgo_step_name)
        old_threshold = None
        try:
            if not quiet:
                old_threshold = distutils.log.set_threshold(distutils.log.DEBUG)
            build_extension.run()
        finally:
            if not quiet and old_threshold is not None:
                distutils.log.set_threshold(old_threshold)

    def _add_pgo_flags(self, build_extension, step_name, temp_dir):
        if False:
            print('Hello World!')
        compiler_type = build_extension.compiler.compiler_type
        if compiler_type == 'unix':
            compiler_cmd = build_extension.compiler.compiler_so
            if not compiler_cmd:
                pass
            elif 'clang' in compiler_cmd or 'clang' in compiler_cmd[0]:
                compiler_type = 'clang'
            elif 'icc' in compiler_cmd or 'icc' in compiler_cmd[0]:
                compiler_type = 'icc'
            elif 'gcc' in compiler_cmd or 'gcc' in compiler_cmd[0]:
                compiler_type = 'gcc'
            elif 'g++' in compiler_cmd or 'g++' in compiler_cmd[0]:
                compiler_type = 'gcc'
        config = PGO_CONFIG.get(compiler_type)
        orig_flags = []
        if config and step_name in config:
            flags = [f.format(TEMPDIR=temp_dir) for f in config[step_name]]
            for extension in build_extension.extensions:
                orig_flags.append((extension.extra_compile_args, extension.extra_link_args))
                extension.extra_compile_args = extension.extra_compile_args + flags
                extension.extra_link_args = extension.extra_link_args + flags
        else:
            print("No PGO %s configuration known for C compiler type '%s'" % (step_name, compiler_type), file=sys.stderr)
        return orig_flags

    @property
    def so_ext(self):
        if False:
            return 10
        'The extension suffix for compiled modules.'
        try:
            return self._so_ext
        except AttributeError:
            self._so_ext = self._get_build_extension().get_ext_filename('')
            return self._so_ext

    def _clear_distutils_mkpath_cache(self):
        if False:
            return 10
        'clear distutils mkpath cache\n\n        prevents distutils from skipping re-creation of dirs that have been removed\n        '
        try:
            from distutils.dir_util import _path_created
        except ImportError:
            pass
        else:
            _path_created.clear()

    def _get_build_extension(self, extension=None, lib_dir=None, temp_dir=None, pgo_step_name=None, _build_ext=build_ext):
        if False:
            print('Hello World!')
        self._clear_distutils_mkpath_cache()
        dist = Distribution()
        config_files = dist.find_config_files()
        try:
            config_files.remove('setup.cfg')
        except ValueError:
            pass
        dist.parse_config_files(config_files)
        if not temp_dir:
            temp_dir = lib_dir
        add_pgo_flags = self._add_pgo_flags
        if pgo_step_name:
            base_build_ext = _build_ext

            class _build_ext(_build_ext):

                def build_extensions(self):
                    if False:
                        while True:
                            i = 10
                    add_pgo_flags(self, pgo_step_name, temp_dir)
                    base_build_ext.build_extensions(self)
        build_extension = _build_ext(dist)
        build_extension.finalize_options()
        if temp_dir:
            temp_dir = encode_fs(temp_dir)
            build_extension.build_temp = temp_dir
        if lib_dir:
            lib_dir = encode_fs(lib_dir)
            build_extension.build_lib = lib_dir
        if extension is not None:
            build_extension.extensions = [extension]
        return build_extension

    @staticmethod
    def clean_annotated_html(html):
        if False:
            i = 10
            return i + 15
        'Clean up the annotated HTML source.\n\n        Strips the link to the generated C or C++ file, which we do not\n        present to the user.\n        '
        r = re.compile('<p>Raw output: <a href="(.*)">(.*)</a>')
        html = '\n'.join((l for l in html.splitlines() if not r.match(l)))
        return html
__doc__ = __doc__.format(CYTHON_DOC=dedent(CythonMagics.cython.__doc__.replace('-+, --cplus', '--cplus    ')), CYTHON_INLINE_DOC=dedent(CythonMagics.cython_inline.__doc__), CYTHON_PYXIMPORT_DOC=dedent(CythonMagics.cython_pyximport.__doc__))