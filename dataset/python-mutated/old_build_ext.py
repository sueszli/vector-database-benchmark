"""Cython.Distutils.old_build_ext

Implements a version of the Distutils 'build_ext' command, for
building Cython extension modules.

Note that this module is deprecated.  Use cythonize() instead.
"""
__revision__ = '$Id:$'
import sys
import os
from distutils.errors import DistutilsPlatformError
from distutils.dep_util import newer, newer_group
from distutils import log
from distutils.command import build_ext as _build_ext
from distutils import sysconfig
try:
    from __builtin__ import basestring
except ImportError:
    basestring = str
'\nimport inspect\nimport warnings\n\ndef _check_stack(path):\n    try:\n        for frame in inspect.getouterframes(inspect.currentframe(), 0):\n            if path in frame[1].replace(os.sep, \'/\'):\n                return True\n    except Exception:\n        pass\n    return False\n\n\nif (not _check_stack(\'setuptools/extensions.py\')\n        and not _check_stack(\'pyximport/pyxbuild.py\')\n        and not _check_stack(\'Cython/Distutils/build_ext.py\')):\n    warnings.warn(\n        "Cython.Distutils.old_build_ext does not properly handle dependencies "\n        "and is deprecated.")\n'
extension_name_re = _build_ext.extension_name_re
show_compilers = _build_ext.show_compilers

class Optimization(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.flags = ('OPT', 'CFLAGS', 'CPPFLAGS', 'EXTRA_CFLAGS', 'BASECFLAGS', 'PY_CFLAGS')
        self.state = sysconfig.get_config_vars(*self.flags)
        self.config_vars = sysconfig.get_config_vars()

    def disable_optimization(self):
        if False:
            print('Hello World!')
        'disable optimization for the C or C++ compiler'
        badoptions = ('-O1', '-O2', '-O3')
        for (flag, option) in zip(self.flags, self.state):
            if option is not None:
                L = [opt for opt in option.split() if opt not in badoptions]
                self.config_vars[flag] = ' '.join(L)

    def restore_state(self):
        if False:
            for i in range(10):
                print('nop')
        'restore the original state'
        for (flag, option) in zip(self.flags, self.state):
            if option is not None:
                self.config_vars[flag] = option
optimization = Optimization()

class old_build_ext(_build_ext.build_ext):
    description = 'build C/C++ and Cython extensions (compile/link to build directory)'
    sep_by = _build_ext.build_ext.sep_by
    user_options = _build_ext.build_ext.user_options[:]
    boolean_options = _build_ext.build_ext.boolean_options[:]
    help_options = _build_ext.build_ext.help_options[:]
    user_options.extend([('cython-cplus', None, 'generate C++ source files'), ('cython-create-listing', None, 'write errors to a listing file'), ('cython-line-directives', None, 'emit source line directives'), ('cython-include-dirs=', None, 'path to the Cython include files' + sep_by), ('cython-c-in-temp', None, 'put generated C files in temp directory'), ('cython-gen-pxi', None, 'generate .pxi file for public declarations'), ('cython-directives=', None, 'compiler directive overrides'), ('cython-gdb', None, 'generate debug information for cygdb'), ('cython-compile-time-env', None, 'cython compile time environment'), ('pyrex-cplus', None, 'generate C++ source files'), ('pyrex-create-listing', None, 'write errors to a listing file'), ('pyrex-line-directives', None, 'emit source line directives'), ('pyrex-include-dirs=', None, 'path to the Cython include files' + sep_by), ('pyrex-c-in-temp', None, 'put generated C files in temp directory'), ('pyrex-gen-pxi', None, 'generate .pxi file for public declarations'), ('pyrex-directives=', None, 'compiler directive overrides'), ('pyrex-gdb', None, 'generate debug information for cygdb')])
    boolean_options.extend(['cython-cplus', 'cython-create-listing', 'cython-line-directives', 'cython-c-in-temp', 'cython-gdb', 'pyrex-cplus', 'pyrex-create-listing', 'pyrex-line-directives', 'pyrex-c-in-temp', 'pyrex-gdb'])

    def initialize_options(self):
        if False:
            return 10
        _build_ext.build_ext.initialize_options(self)
        self.cython_cplus = 0
        self.cython_create_listing = 0
        self.cython_line_directives = 0
        self.cython_include_dirs = None
        self.cython_directives = None
        self.cython_c_in_temp = 0
        self.cython_gen_pxi = 0
        self.cython_gdb = False
        self.no_c_in_traceback = 0
        self.cython_compile_time_env = None

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        if name[:6] == 'pyrex_':
            return getattr(self, 'cython_' + name[6:])
        else:
            return _build_ext.build_ext.__getattr__(self, name)

    def __setattr__(self, name, value):
        if False:
            print('Hello World!')
        if name[:6] == 'pyrex_':
            return setattr(self, 'cython_' + name[6:], value)
        else:
            self.__dict__[name] = value

    def finalize_options(self):
        if False:
            i = 10
            return i + 15
        _build_ext.build_ext.finalize_options(self)
        if self.cython_include_dirs is None:
            self.cython_include_dirs = []
        elif isinstance(self.cython_include_dirs, basestring):
            self.cython_include_dirs = self.cython_include_dirs.split(os.pathsep)
        if self.cython_directives is None:
            self.cython_directives = {}

    def run(self):
        if False:
            return 10
        if self.cython_gdb or [1 for ext in self.extensions if getattr(ext, 'cython_gdb', False)]:
            optimization.disable_optimization()
        _build_ext.build_ext.run(self)

    def check_extensions_list(self, extensions):
        if False:
            for i in range(10):
                print('nop')
        _build_ext.build_ext.check_extensions_list(self, extensions)
        for ext in self.extensions:
            ext.sources = self.cython_sources(ext.sources, ext)

    def cython_sources(self, sources, extension):
        if False:
            while True:
                i = 10
        "\n        Walk the list of source files in 'sources', looking for Cython\n        source files (.pyx and .py).  Run Cython on all that are\n        found, and return a modified 'sources' list with Cython source\n        files replaced by the generated C (or C++) files.\n        "
        new_sources = []
        cython_sources = []
        cython_targets = {}
        create_listing = self.cython_create_listing or getattr(extension, 'cython_create_listing', 0)
        line_directives = self.cython_line_directives or getattr(extension, 'cython_line_directives', 0)
        no_c_in_traceback = self.no_c_in_traceback or getattr(extension, 'no_c_in_traceback', 0)
        cplus = self.cython_cplus or getattr(extension, 'cython_cplus', 0) or (extension.language and extension.language.lower() == 'c++')
        cython_gen_pxi = self.cython_gen_pxi or getattr(extension, 'cython_gen_pxi', 0)
        cython_gdb = self.cython_gdb or getattr(extension, 'cython_gdb', False)
        cython_compile_time_env = self.cython_compile_time_env or getattr(extension, 'cython_compile_time_env', None)
        includes = list(self.cython_include_dirs)
        try:
            for i in extension.cython_include_dirs:
                if i not in includes:
                    includes.append(i)
        except AttributeError:
            pass
        extension.include_dirs = list(extension.include_dirs)
        for i in extension.include_dirs:
            if i not in includes:
                includes.append(i)
        directives = dict(self.cython_directives)
        if hasattr(extension, 'cython_directives'):
            directives.update(extension.cython_directives)
        if cplus:
            target_ext = '.cpp'
        else:
            target_ext = '.c'
        if not self.inplace and (self.cython_c_in_temp or getattr(extension, 'cython_c_in_temp', 0)):
            target_dir = os.path.join(self.build_temp, 'pyrex')
            for package_name in extension.name.split('.')[:-1]:
                target_dir = os.path.join(target_dir, package_name)
        else:
            target_dir = None
        newest_dependency = None
        for source in sources:
            (base, ext) = os.path.splitext(os.path.basename(source))
            if ext == '.py':
                ext = '.pyx'
            if ext == '.pyx':
                output_dir = target_dir or os.path.dirname(source)
                new_sources.append(os.path.join(output_dir, base + target_ext))
                cython_sources.append(source)
                cython_targets[source] = new_sources[-1]
            elif ext == '.pxi' or ext == '.pxd':
                if newest_dependency is None or newer(source, newest_dependency):
                    newest_dependency = source
            else:
                new_sources.append(source)
        if not cython_sources:
            return new_sources
        try:
            from Cython.Compiler.Main import CompilationOptions, default_options as cython_default_options, compile as cython_compile
            from Cython.Compiler.Errors import PyrexError
        except ImportError:
            e = sys.exc_info()[1]
            print('failed to import Cython: %s' % e)
            raise DistutilsPlatformError('Cython does not appear to be installed')
        module_name = extension.name
        for source in cython_sources:
            target = cython_targets[source]
            depends = [source] + list(extension.depends or ())
            if source[-4:].lower() == '.pyx' and os.path.isfile(source[:-3] + 'pxd'):
                depends += [source[:-3] + 'pxd']
            rebuild = self.force or newer_group(depends, target, 'newer')
            if not rebuild and newest_dependency is not None:
                rebuild = newer(newest_dependency, target)
            if rebuild:
                log.info('cythoning %s to %s', source, target)
                self.mkpath(os.path.dirname(target))
                if self.inplace:
                    output_dir = os.curdir
                else:
                    output_dir = self.build_lib
                options = CompilationOptions(cython_default_options, use_listing_file=create_listing, include_path=includes, compiler_directives=directives, output_file=target, cplus=cplus, emit_linenums=line_directives, c_line_in_traceback=not no_c_in_traceback, generate_pxi=cython_gen_pxi, output_dir=output_dir, gdb_debug=cython_gdb, compile_time_env=cython_compile_time_env)
                result = cython_compile(source, options=options, full_module_name=module_name)
            else:
                log.info("skipping '%s' Cython extension (up-to-date)", target)
        return new_sources