"""
Import hooks; when installed with the install() function, these hooks
allow importing .pyx files as if they were Python modules.

If you want the hook installed every time you run Python
you can add it to your Python version by adding these lines to
sitecustomize.py (which you can create from scratch in site-packages
if it doesn't exist there or somewhere else on your python path)::

    import pyximport
    pyximport.install()

For instance on the Mac with a non-system Python 2.3, you could create
sitecustomize.py with only those two lines at
/usr/local/lib/python2.3/site-packages/sitecustomize.py .

A custom distutils.core.Extension instance and setup() args
(Distribution) for for the build can be defined by a <modulename>.pyxbld
file like:

# examplemod.pyxbld
def make_ext(modname, pyxfilename):
    from distutils.extension import Extension
    return Extension(name = modname,
                     sources=[pyxfilename, 'hello.c'],
                     include_dirs=['/myinclude'] )
def make_setup_args():
    return dict(script_args=["--compiler=mingw32"])

Extra dependencies can be defined by a <modulename>.pyxdep .
See README.

Since Cython 0.11, the :mod:`pyximport` module also has experimental
compilation support for normal Python modules.  This allows you to
automatically run Cython on every .pyx and .py module that Python
imports, including parts of the standard library and installed
packages.  Cython will still fail to compile a lot of Python modules,
in which case the import mechanism will fall back to loading the
Python source modules instead.  The .py import mechanism is installed
like this::

    pyximport.install(pyimport = True)

Running this module as a top-level script will run a test and then print
the documentation.

This code is based on the Py2.3+ import protocol as described in PEP 302.
"""
import glob
import imp
import os
import sys
from zipimport import zipimporter, ZipImportError
mod_name = 'pyximport'
PYX_EXT = '.pyx'
PYXDEP_EXT = '.pyxdep'
PYXBLD_EXT = '.pyxbld'
DEBUG_IMPORT = False

def _print(message, args):
    if False:
        i = 10
        return i + 15
    if args:
        message = message % args
    print(message)

def _debug(message, *args):
    if False:
        for i in range(10):
            print('nop')
    if DEBUG_IMPORT:
        _print(message, args)

def _info(message, *args):
    if False:
        i = 10
        return i + 15
    _print(message, args)

def _load_pyrex(name, filename):
    if False:
        i = 10
        return i + 15
    'Load a pyrex file given a name and filename.'

def get_distutils_extension(modname, pyxfilename, language_level=None):
    if False:
        return 10
    (extension_mod, setup_args) = handle_special_build(modname, pyxfilename)
    if not extension_mod:
        if not isinstance(pyxfilename, str):
            pyxfilename = pyxfilename.encode(sys.getfilesystemencoding())
        from distutils.extension import Extension
        extension_mod = Extension(name=modname, sources=[pyxfilename])
        if language_level is not None:
            extension_mod.cython_directives = {'language_level': language_level}
    return (extension_mod, setup_args)

def handle_special_build(modname, pyxfilename):
    if False:
        while True:
            i = 10
    special_build = os.path.splitext(pyxfilename)[0] + PYXBLD_EXT
    ext = None
    setup_args = {}
    if os.path.exists(special_build):
        with open(special_build) as fid:
            mod = imp.load_source('XXXX', special_build, fid)
        make_ext = getattr(mod, 'make_ext', None)
        if make_ext:
            ext = make_ext(modname, pyxfilename)
            assert ext and ext.sources, 'make_ext in %s did not return Extension' % special_build
        make_setup_args = getattr(mod, 'make_setup_args', None)
        if make_setup_args:
            setup_args = make_setup_args()
            assert isinstance(setup_args, dict), 'make_setup_args in %s did not return a dict' % special_build
        assert set or setup_args, 'neither make_ext nor make_setup_args %s' % special_build
        ext.sources = [os.path.join(os.path.dirname(special_build), source) for source in ext.sources]
    return (ext, setup_args)

def handle_dependencies(pyxfilename):
    if False:
        for i in range(10):
            print('nop')
    testing = '_test_files' in globals()
    dependfile = os.path.splitext(pyxfilename)[0] + PYXDEP_EXT
    if os.path.exists(dependfile):
        with open(dependfile) as fid:
            depends = fid.readlines()
        depends = [depend.strip() for depend in depends]
        files = [dependfile]
        for depend in depends:
            fullpath = os.path.join(os.path.dirname(dependfile), depend)
            files.extend(glob.glob(fullpath))
        if testing:
            _test_files[:] = []
        for file in files:
            from distutils.dep_util import newer
            if newer(file, pyxfilename):
                _debug('Rebuilding %s because of %s', pyxfilename, file)
                filetime = os.path.getmtime(file)
                os.utime(pyxfilename, (filetime, filetime))
                if testing:
                    _test_files.append(file)

def build_module(name, pyxfilename, pyxbuild_dir=None, inplace=False, language_level=None):
    if False:
        while True:
            i = 10
    assert os.path.exists(pyxfilename), 'Path does not exist: %s' % pyxfilename
    handle_dependencies(pyxfilename)
    (extension_mod, setup_args) = get_distutils_extension(name, pyxfilename, language_level)
    build_in_temp = pyxargs.build_in_temp
    sargs = pyxargs.setup_args.copy()
    sargs.update(setup_args)
    build_in_temp = sargs.pop('build_in_temp', build_in_temp)
    from . import pyxbuild
    olddir = os.getcwd()
    common = ''
    if pyxbuild_dir:
        common = os.path.commonprefix([pyxbuild_dir, pyxfilename])
    if len(common) > 30:
        pyxfilename = os.path.relpath(pyxfilename)
        pyxbuild_dir = os.path.relpath(pyxbuild_dir)
        os.chdir(common)
    try:
        so_path = pyxbuild.pyx_to_dll(pyxfilename, extension_mod, build_in_temp=build_in_temp, pyxbuild_dir=pyxbuild_dir, setup_args=sargs, inplace=inplace, reload_support=pyxargs.reload_support)
    finally:
        os.chdir(olddir)
    so_path = os.path.join(common, so_path)
    assert os.path.exists(so_path), 'Cannot find: %s' % so_path
    junkpath = os.path.join(os.path.dirname(so_path), name + '_*')
    junkstuff = glob.glob(junkpath)
    for path in junkstuff:
        if path != so_path:
            try:
                os.remove(path)
            except IOError:
                _info("Couldn't remove %s", path)
    return so_path

def load_module(name, pyxfilename, pyxbuild_dir=None, is_package=False, build_inplace=False, language_level=None, so_path=None):
    if False:
        i = 10
        return i + 15
    try:
        if so_path is None:
            if is_package:
                module_name = name + '.__init__'
            else:
                module_name = name
            so_path = build_module(module_name, pyxfilename, pyxbuild_dir, inplace=build_inplace, language_level=language_level)
        mod = imp.load_dynamic(name, so_path)
        if is_package and (not hasattr(mod, '__path__')):
            mod.__path__ = [os.path.dirname(so_path)]
        assert mod.__file__ == so_path, (mod.__file__, so_path)
    except Exception as failure_exc:
        _debug('Failed to load extension module: %r' % failure_exc)
        if pyxargs.load_py_module_on_import_failure and pyxfilename.endswith('.py'):
            mod = imp.load_source(name, pyxfilename)
            assert mod.__file__ in (pyxfilename, pyxfilename + 'c', pyxfilename + 'o'), (mod.__file__, pyxfilename)
        else:
            tb = sys.exc_info()[2]
            import traceback
            exc = ImportError('Building module %s failed: %s' % (name, traceback.format_exception_only(*sys.exc_info()[:2])))
            if sys.version_info[0] >= 3:
                raise exc.with_traceback(tb)
            else:
                exec('raise exc, None, tb', {'exc': exc, 'tb': tb})
    return mod

class PyxImporter(object):
    """A meta-path importer for .pyx files.
    """

    def __init__(self, extension=PYX_EXT, pyxbuild_dir=None, inplace=False, language_level=None):
        if False:
            return 10
        self.extension = extension
        self.pyxbuild_dir = pyxbuild_dir
        self.inplace = inplace
        self.language_level = language_level

    def find_module(self, fullname, package_path=None):
        if False:
            print('Hello World!')
        if fullname in sys.modules and (not pyxargs.reload_support):
            return None
        if package_path is not None and (not isinstance(package_path, list)):
            package_path = list(package_path)
        try:
            (fp, pathname, (ext, mode, ty)) = imp.find_module(fullname, package_path)
            if fp:
                fp.close()
            if pathname and ty == imp.PKG_DIRECTORY:
                pkg_file = os.path.join(pathname, '__init__' + self.extension)
                if os.path.isfile(pkg_file):
                    return PyxLoader(fullname, pathname, init_path=pkg_file, pyxbuild_dir=self.pyxbuild_dir, inplace=self.inplace, language_level=self.language_level)
            if pathname and pathname.endswith(self.extension):
                return PyxLoader(fullname, pathname, pyxbuild_dir=self.pyxbuild_dir, inplace=self.inplace, language_level=self.language_level)
            if ty != imp.C_EXTENSION:
                return None
            pyxpath = os.path.splitext(pathname)[0] + self.extension
            if os.path.isfile(pyxpath):
                return PyxLoader(fullname, pyxpath, pyxbuild_dir=self.pyxbuild_dir, inplace=self.inplace, language_level=self.language_level)
        except ImportError:
            pass
        mod_parts = fullname.split('.')
        module_name = mod_parts[-1]
        pyx_module_name = module_name + self.extension
        paths = package_path or sys.path
        for path in paths:
            pyx_data = None
            if not path:
                path = os.getcwd()
            elif os.path.isfile(path):
                try:
                    zi = zipimporter(path)
                    pyx_data = zi.get_data(pyx_module_name)
                except (ZipImportError, IOError, OSError):
                    continue
                path = self.pyxbuild_dir
            elif not os.path.isabs(path):
                path = os.path.abspath(path)
            pyx_module_path = os.path.join(path, pyx_module_name)
            if pyx_data is not None:
                if not os.path.exists(path):
                    try:
                        os.makedirs(path)
                    except OSError:
                        if not os.path.exists(path):
                            raise
                with open(pyx_module_path, 'wb') as f:
                    f.write(pyx_data)
            elif not os.path.isfile(pyx_module_path):
                continue
            return PyxLoader(fullname, pyx_module_path, pyxbuild_dir=self.pyxbuild_dir, inplace=self.inplace, language_level=self.language_level)
        _debug('%s not found' % fullname)
        return None

class PyImporter(PyxImporter):
    """A meta-path importer for normal .py files.
    """

    def __init__(self, pyxbuild_dir=None, inplace=False, language_level=None):
        if False:
            print('Hello World!')
        if language_level is None:
            language_level = sys.version_info[0]
        self.super = super(PyImporter, self)
        self.super.__init__(extension='.py', pyxbuild_dir=pyxbuild_dir, inplace=inplace, language_level=language_level)
        self.uncompilable_modules = {}
        self.blocked_modules = ['Cython', 'pyxbuild', 'pyximport.pyxbuild', 'distutils']
        self.blocked_packages = ['Cython.', 'distutils.']

    def find_module(self, fullname, package_path=None):
        if False:
            return 10
        if fullname in sys.modules:
            return None
        if any([fullname.startswith(pkg) for pkg in self.blocked_packages]):
            return None
        if fullname in self.blocked_modules:
            return None
        if _lib_loader.knows(fullname):
            return _lib_loader
        _debug("trying import of module '%s'", fullname)
        if fullname in self.uncompilable_modules:
            (path, last_modified) = self.uncompilable_modules[fullname]
            try:
                new_last_modified = os.stat(path).st_mtime
                if new_last_modified > last_modified:
                    return None
            except OSError:
                pass
        self.blocked_modules.append(fullname)
        try:
            importer = self.super.find_module(fullname, package_path)
            if importer is not None:
                if importer.init_path:
                    path = importer.init_path
                    real_name = fullname + '.__init__'
                else:
                    path = importer.path
                    real_name = fullname
                _debug('importer found path %s for module %s', path, real_name)
                try:
                    so_path = build_module(real_name, path, pyxbuild_dir=self.pyxbuild_dir, language_level=self.language_level, inplace=self.inplace)
                    _lib_loader.add_lib(fullname, path, so_path, is_package=bool(importer.init_path))
                    return _lib_loader
                except Exception:
                    if DEBUG_IMPORT:
                        import traceback
                        traceback.print_exc()
                    try:
                        last_modified = os.stat(path).st_mtime
                    except OSError:
                        last_modified = 0
                    self.uncompilable_modules[fullname] = (path, last_modified)
                    importer = None
        finally:
            self.blocked_modules.pop()
        return importer

class LibLoader(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._libs = {}

    def load_module(self, fullname):
        if False:
            return 10
        try:
            (source_path, so_path, is_package) = self._libs[fullname]
        except KeyError:
            raise ValueError('invalid module %s' % fullname)
        _debug("Loading shared library module '%s' from %s", fullname, so_path)
        return load_module(fullname, source_path, so_path=so_path, is_package=is_package)

    def add_lib(self, fullname, path, so_path, is_package):
        if False:
            for i in range(10):
                print('nop')
        self._libs[fullname] = (path, so_path, is_package)

    def knows(self, fullname):
        if False:
            for i in range(10):
                print('nop')
        return fullname in self._libs
_lib_loader = LibLoader()

class PyxLoader(object):

    def __init__(self, fullname, path, init_path=None, pyxbuild_dir=None, inplace=False, language_level=None):
        if False:
            while True:
                i = 10
        _debug('PyxLoader created for loading %s from %s (init path: %s)', fullname, path, init_path)
        self.fullname = fullname
        (self.path, self.init_path) = (path, init_path)
        self.pyxbuild_dir = pyxbuild_dir
        self.inplace = inplace
        self.language_level = language_level

    def load_module(self, fullname):
        if False:
            return 10
        assert self.fullname == fullname, 'invalid module, expected %s, got %s' % (self.fullname, fullname)
        if self.init_path:
            module = load_module(fullname, self.init_path, self.pyxbuild_dir, is_package=True, build_inplace=self.inplace, language_level=self.language_level)
            module.__path__ = [self.path]
        else:
            module = load_module(fullname, self.path, self.pyxbuild_dir, build_inplace=self.inplace, language_level=self.language_level)
        return module

class PyxArgs(object):
    build_dir = True
    build_in_temp = True
    setup_args = {}

def _have_importers():
    if False:
        print('Hello World!')
    has_py_importer = False
    has_pyx_importer = False
    for importer in sys.meta_path:
        if isinstance(importer, PyxImporter):
            if isinstance(importer, PyImporter):
                has_py_importer = True
            else:
                has_pyx_importer = True
    return (has_py_importer, has_pyx_importer)

def install(pyximport=True, pyimport=False, build_dir=None, build_in_temp=True, setup_args=None, reload_support=False, load_py_module_on_import_failure=False, inplace=False, language_level=None):
    if False:
        for i in range(10):
            print('nop')
    " Main entry point for pyxinstall.\n\n    Call this to install the ``.pyx`` import hook in\n    your meta-path for a single Python process.  If you want it to be\n    installed whenever you use Python, add it to your ``sitecustomize``\n    (as described above).\n\n    :param pyximport: If set to False, does not try to import ``.pyx`` files.\n\n    :param pyimport: You can pass ``pyimport=True`` to also\n        install the ``.py`` import hook\n        in your meta-path.  Note, however, that it is rather experimental,\n        will not work at all for some ``.py`` files and packages, and will\n        heavily slow down your imports due to search and compilation.\n        Use at your own risk.\n\n    :param build_dir: By default, compiled modules will end up in a ``.pyxbld``\n        directory in the user's home directory.  Passing a different path\n        as ``build_dir`` will override this.\n\n    :param build_in_temp: If ``False``, will produce the C files locally. Working\n        with complex dependencies and debugging becomes more easy. This\n        can principally interfere with existing files of the same name.\n\n    :param setup_args: Dict of arguments for Distribution.\n        See ``distutils.core.setup()``.\n\n    :param reload_support: Enables support for dynamic\n        ``reload(my_module)``, e.g. after a change in the Cython code.\n        Additional files ``<so_path>.reloadNN`` may arise on that account, when\n        the previously loaded module file cannot be overwritten.\n\n    :param load_py_module_on_import_failure: If the compilation of a ``.py``\n        file succeeds, but the subsequent import fails for some reason,\n        retry the import with the normal ``.py`` module instead of the\n        compiled module.  Note that this may lead to unpredictable results\n        for modules that change the system state during their import, as\n        the second import will rerun these modifications in whatever state\n        the system was left after the import of the compiled module\n        failed.\n\n    :param inplace: Install the compiled module\n        (``.so`` for Linux and Mac / ``.pyd`` for Windows)\n        next to the source file.\n\n    :param language_level: The source language level to use: 2 or 3.\n        The default is to use the language level of the current Python\n        runtime for .py files and Py2 for ``.pyx`` files.\n    "
    if setup_args is None:
        setup_args = {}
    if not build_dir:
        build_dir = os.path.join(os.path.expanduser('~'), '.pyxbld')
    global pyxargs
    pyxargs = PyxArgs()
    pyxargs.build_dir = build_dir
    pyxargs.build_in_temp = build_in_temp
    pyxargs.setup_args = (setup_args or {}).copy()
    pyxargs.reload_support = reload_support
    pyxargs.load_py_module_on_import_failure = load_py_module_on_import_failure
    (has_py_importer, has_pyx_importer) = _have_importers()
    (py_importer, pyx_importer) = (None, None)
    if pyimport and (not has_py_importer):
        py_importer = PyImporter(pyxbuild_dir=build_dir, inplace=inplace, language_level=language_level)
        import Cython.Compiler.Main, Cython.Compiler.Pipeline, Cython.Compiler.Optimize
        sys.meta_path.insert(0, py_importer)
    if pyximport and (not has_pyx_importer):
        pyx_importer = PyxImporter(pyxbuild_dir=build_dir, inplace=inplace, language_level=language_level)
        sys.meta_path.append(pyx_importer)
    return (py_importer, pyx_importer)

def uninstall(py_importer, pyx_importer):
    if False:
        return 10
    '\n    Uninstall an import hook.\n    '
    try:
        sys.meta_path.remove(py_importer)
    except ValueError:
        pass
    try:
        sys.meta_path.remove(pyx_importer)
    except ValueError:
        pass

def show_docs():
    if False:
        return 10
    import __main__
    __main__.__name__ = mod_name
    for name in dir(__main__):
        item = getattr(__main__, name)
        try:
            setattr(item, '__module__', mod_name)
        except (AttributeError, TypeError):
            pass
    help(__main__)
if __name__ == '__main__':
    show_docs()