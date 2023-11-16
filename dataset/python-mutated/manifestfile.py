from __future__ import print_function
import contextlib
import os
import sys
import glob
import tempfile
from collections import namedtuple
__all__ = ['ManifestFileError', 'ManifestFile']
MODE_FREEZE = 1
MODE_COMPILE = 2
MODE_PYPROJECT = 3
KIND_AUTO = 1
KIND_FREEZE_AUTO = 2
KIND_FREEZE_AS_STR = 3
KIND_FREEZE_AS_MPY = 4
KIND_FREEZE_MPY = 5
KIND_COMPILE_AS_MPY = 6
FILE_TYPE_LOCAL = 1
FILE_TYPE_HTTP = 2

class ManifestFileError(Exception):
    pass

class ManifestIgnoreException(Exception):
    pass

class ManifestUsePyPIException(Exception):

    def __init__(self, pypi_name):
        if False:
            while True:
                i = 10
        self.pypi_name = pypi_name
ManifestOutput = namedtuple('ManifestOutput', ['file_type', 'full_path', 'target_path', 'timestamp', 'kind', 'metadata', 'opt'])

class ManifestPackageMetadata:

    def __init__(self, is_require=False):
        if False:
            print('Hello World!')
        self._is_require = is_require
        self._initialised = False
        self.version = None
        self.description = None
        self.license = None
        self.author = None
        self.stdlib = False
        self.pypi = None
        self.pypi_publish = None

    def update(self, mode, description=None, version=None, license=None, author=None, stdlib=False, pypi=None, pypi_publish=None):
        if False:
            print('Hello World!')
        if self._initialised:
            raise ManifestFileError('Duplicate call to metadata().')
        if mode == MODE_PYPROJECT and self._is_require:
            if stdlib:
                raise ManifestIgnoreException
            if pypi_publish or pypi:
                raise ManifestUsePyPIException(pypi_publish or pypi)
        self.description = description
        self.version = version
        self.license = license
        self.author = author
        self.pypi = pypi
        self.pypi_publish = pypi_publish
        self._initialised = True

    def check_initialised(self, mode):
        if False:
            return 10
        if mode in (MODE_COMPILE, MODE_PYPROJECT):
            if not self._initialised:
                raise ManifestFileError('metadata() must be the first command in a manifest file.')

    def __str__(self):
        if False:
            print('Hello World!')
        return 'version={} description={} license={} author={} pypi={} pypi_publish={}'.format(self.version, self.description, self.license, self.author, self.pypi, self.pypi_publish)

class IncludeOptions:

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        self._kwargs = kwargs
        self._defaults = {}

    def defaults(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._defaults = kwargs

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        return self._kwargs.get(name, self._defaults.get(name, None))

class ManifestFile:

    def __init__(self, mode, path_vars=None):
        if False:
            return 10
        self._mode = mode
        self._path_vars = path_vars or {}
        self._manifest_files = []
        self._pypi_dependencies = []
        self._visited = set()
        self._metadata = [ManifestPackageMetadata()]

    def _resolve_path(self, path):
        if False:
            return 10
        for (name, value) in self._path_vars.items():
            if value is not None:
                path = path.replace('$({})'.format(name), value)
        return os.path.abspath(path)

    def _manifest_globals(self, kwargs):
        if False:
            i = 10
            return i + 15
        g = {'metadata': self.metadata, 'include': self.include, 'require': self.require, 'package': self.package, 'module': self.module, 'options': IncludeOptions(**kwargs)}
        if self._mode == MODE_FREEZE:
            g.update({'freeze': self.freeze, 'freeze_as_str': self.freeze_as_str, 'freeze_as_mpy': self.freeze_as_mpy, 'freeze_mpy': self.freeze_mpy})
        return g

    def files(self):
        if False:
            for i in range(10):
                print('nop')
        return self._manifest_files

    def pypi_dependencies(self):
        if False:
            while True:
                i = 10
        return self._pypi_dependencies

    def execute(self, manifest_file):
        if False:
            return 10
        if manifest_file.endswith('.py'):
            self.include(manifest_file)
        else:
            try:
                exec(manifest_file, self._manifest_globals({}))
            except Exception as er:
                raise ManifestFileError('Error in manifest: {}'.format(er))

    def _add_file(self, full_path, target_path, kind=KIND_AUTO, opt=None):
        if False:
            i = 10
            return i + 15
        try:
            stat = os.stat(full_path)
            timestamp = stat.st_mtime
        except OSError:
            raise ManifestFileError('Cannot stat {}'.format(full_path))
        (_, ext) = os.path.splitext(full_path)
        if self._mode == MODE_FREEZE:
            if kind in (KIND_AUTO, KIND_FREEZE_AUTO):
                if ext.lower() == '.py':
                    kind = KIND_FREEZE_AS_MPY
                elif ext.lower() == '.mpy':
                    kind = KIND_FREEZE_MPY
        else:
            if kind != KIND_AUTO:
                raise ManifestFileError('Not in freeze mode')
            if ext.lower() != '.py':
                raise ManifestFileError('Expected .py file')
            kind = KIND_COMPILE_AS_MPY
        self._manifest_files.append(ManifestOutput(FILE_TYPE_LOCAL, full_path, target_path, timestamp, kind, self._metadata[-1], opt))

    def _search(self, base_path, package_path, files, exts, kind, opt=None, strict=False):
        if False:
            return 10
        base_path = self._resolve_path(base_path)
        if files:
            for file in files:
                if package_path:
                    file = os.path.join(package_path, file)
                self._add_file(os.path.join(base_path, file), file, kind=kind, opt=opt)
        else:
            if base_path:
                prev_cwd = os.getcwd()
                os.chdir(self._resolve_path(base_path))
            for (dirpath, _, filenames) in os.walk(package_path or '.', followlinks=True):
                for file in filenames:
                    file = os.path.relpath(os.path.join(dirpath, file), '.')
                    (_, ext) = os.path.splitext(file)
                    if ext.lower() in exts:
                        self._add_file(os.path.join(base_path, file), file, kind=kind, opt=opt)
                    elif strict:
                        raise ManifestFileError('Unexpected file type')
            if base_path:
                os.chdir(prev_cwd)

    def metadata(self, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        From within a manifest file, use this to set the metadata for the\n        package described by current manifest.\n\n        After executing a manifest file (via execute()), call this\n        to obtain the metadata for the top-level manifest file.\n\n        See ManifestPackageMetadata.update() for valid kwargs.\n        '
        if kwargs:
            self._metadata[-1].update(self._mode, **kwargs)
        return self._metadata[-1]

    def include(self, manifest_path, is_require=False, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Include another manifest.\n\n        The manifest argument can be a string (filename) or an iterable of\n        strings.\n\n        Relative paths are resolved with respect to the current manifest file.\n\n        If the path is to a directory, then it implicitly includes the\n        manifest.py file inside that directory.\n\n        Optional kwargs can be provided which will be available to the\n        included script via the `options` variable.\n\n        e.g. include("path.py", extra_features=True)\n\n        in path.py:\n            options.defaults(standard_features=True)\n\n            # freeze minimal modules.\n            if options.standard_features:\n                # freeze standard modules.\n            if options.extra_features:\n                # freeze extra modules.\n        '
        if is_require:
            self._metadata[-1].check_initialised(self._mode)
        if not isinstance(manifest_path, str):
            for m in manifest_path:
                self.include(m, **kwargs)
        else:
            manifest_path = self._resolve_path(manifest_path)
            if os.path.isdir(manifest_path):
                manifest_path = os.path.join(manifest_path, 'manifest.py')
            if manifest_path in self._visited:
                return
            self._visited.add(manifest_path)
            if is_require:
                self._metadata.append(ManifestPackageMetadata(is_require=True))
            try:
                with open(manifest_path) as f:
                    prev_cwd = os.getcwd()
                    os.chdir(os.path.dirname(manifest_path))
                    try:
                        exec(f.read(), self._manifest_globals(kwargs))
                    finally:
                        os.chdir(prev_cwd)
            except ManifestIgnoreException:
                pass
            except ManifestUsePyPIException as e:
                self._pypi_dependencies.append(e.pypi_name)
            except Exception as e:
                raise ManifestFileError('Error in manifest file: {}: {}'.format(manifest_path, e))
            if is_require:
                self._metadata.pop()

    def require(self, name, version=None, unix_ffi=False, pypi=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Require a module by name from micropython-lib.\n\n        Optionally specify unix_ffi=True to use a module from the unix-ffi directory.\n\n        Optionally specify pipy="package-name" to indicate that this should\n        use the named package from PyPI when building for CPython.\n        '
        self._metadata[-1].check_initialised(self._mode)
        if self._mode == MODE_PYPROJECT and pypi:
            self._pypi_dependencies.append(pypi)
            return
        if self._path_vars['MPY_LIB_DIR']:
            lib_dirs = ['micropython', 'python-stdlib', 'python-ecosys']
            if unix_ffi:
                lib_dirs = ['unix-ffi'] + lib_dirs
            for lib_dir in lib_dirs:
                for (root, dirnames, filenames) in os.walk(os.path.join(self._path_vars['MPY_LIB_DIR'], lib_dir)):
                    if os.path.basename(root) == name and 'manifest.py' in filenames:
                        self.include(root, is_require=True, **kwargs)
                        return
            raise ValueError('Library not found in local micropython-lib: {}'.format(name))
        else:
            raise ValueError("micropython-lib not available for require('{}').", name)

    def package(self, package_path, files=None, base_path='.', opt=None):
        if False:
            print('Hello World!')
        '\n        Define a package, optionally restricting to a set of files.\n\n        Simple case, a package in the current directory:\n            package("foo")\n        will include all .py files in foo, and will be stored as foo/bar/baz.py.\n\n        If the package isn\'t in the current directory, use base_path:\n            package("foo", base_path="src")\n\n        To restrict to certain files in the package use files (note: paths should be relative to the package):\n            package("foo", files=["bar/baz.py"])\n        '
        self._metadata[-1].check_initialised(self._mode)
        self._search(base_path, package_path, files, exts=('.py',), kind=KIND_AUTO, opt=opt)

    def module(self, module_path, base_path='.', opt=None):
        if False:
            print('Hello World!')
        '\n        Include a single Python file as a module.\n\n        If the file is in the current directory:\n            module("foo.py")\n\n        Otherwise use base_path to locate the file:\n            module("foo.py", "src/drivers")\n        '
        self._metadata[-1].check_initialised(self._mode)
        base_path = self._resolve_path(base_path)
        (_, ext) = os.path.splitext(module_path)
        if ext.lower() != '.py':
            raise ManifestFileError('module must be .py file')
        self._add_file(os.path.join(base_path, module_path), module_path, opt=opt)

    def _freeze_internal(self, path, script, exts, kind, opt):
        if False:
            for i in range(10):
                print('nop')
        if script is None:
            self._search(path, None, None, exts=exts, kind=kind, opt=opt)
        elif isinstance(script, str) and os.path.isdir(os.path.join(path, script)):
            self._search(path, script, None, exts=exts, kind=kind, opt=opt)
        elif not isinstance(script, str):
            self._search(path, None, script, exts=exts, kind=kind, opt=opt)
        else:
            self._search(path, None, (script,), exts=exts, kind=kind, opt=opt)

    def freeze(self, path, script=None, opt=None):
        if False:
            while True:
                i = 10
        '\n        Freeze the input, automatically determining its type.  A .py script\n        will be compiled to a .mpy first then frozen, and a .mpy file will be\n        frozen directly.\n\n        `path` must be a directory, which is the base directory to _search for\n        files from.  When importing the resulting frozen modules, the name of\n        the module will start after `path`, ie `path` is excluded from the\n        module name.\n\n        If `path` is relative, it is resolved to the current manifest.py.\n        Use $(MPY_DIR), $(MPY_LIB_DIR), $(PORT_DIR), $(BOARD_DIR) if you need\n        to access specific paths.\n\n        If `script` is None all files in `path` will be frozen.\n\n        If `script` is an iterable then freeze() is called on all items of the\n        iterable (with the same `path` and `opt` passed through).\n\n        If `script` is a string then it specifies the file or directory to\n        freeze, and can include extra directories before the file or last\n        directory.  The file or directory will be _searched for in `path`.  If\n        `script` is a directory then all files in that directory will be frozen.\n\n        `opt` is the optimisation level to pass to mpy-cross when compiling .py\n        to .mpy.\n        '
        self._freeze_internal(path, script, exts=('.py', '.mpy'), kind=KIND_FREEZE_AUTO, opt=opt)

    def freeze_as_str(self, path):
        if False:
            while True:
                i = 10
        '\n        Freeze the given `path` and all .py scripts within it as a string,\n        which will be compiled upon import.\n        '
        self._search(path, None, None, exts=('.py',), kind=KIND_FREEZE_AS_STR)

    def freeze_as_mpy(self, path, script=None, opt=None):
        if False:
            return 10
        '\n        Freeze the input (see above) by first compiling the .py scripts to\n        .mpy files, then freezing the resulting .mpy files.\n        '
        self._freeze_internal(path, script, exts=('.py',), kind=KIND_FREEZE_AS_MPY, opt=opt)

    def freeze_mpy(self, path, script=None, opt=None):
        if False:
            return 10
        '\n        Freeze the input (see above), which must be .mpy files that are\n        frozen directly.\n        '
        self._freeze_internal(path, script, exts=('.mpy',), kind=KIND_FREEZE_MPY, opt=opt)

@contextlib.contextmanager
def tagged_py_file(path, metadata):
    if False:
        for i in range(10):
            print('nop')
    (dest_fd, dest_path) = tempfile.mkstemp(suffix='.py', text=True)
    try:
        with os.fdopen(dest_fd, 'w') as dest:
            with open(path, 'r') as src:
                contents = src.read()
                dest.write(contents)
                if metadata.version and '__version__ =' not in contents:
                    dest.write('\n\n__version__ = {}\n'.format(repr(metadata.version)))
        yield dest_path
    finally:
        os.unlink(dest_path)

def main():
    if False:
        while True:
            i = 10
    import argparse
    cmd_parser = argparse.ArgumentParser(description='List the files referenced by a manifest.')
    cmd_parser.add_argument('--freeze', action='store_true', help='freeze mode')
    cmd_parser.add_argument('--compile', action='store_true', help='compile mode')
    cmd_parser.add_argument('--pyproject', action='store_true', help='pyproject mode')
    cmd_parser.add_argument('--lib', default=os.path.join(os.path.dirname(__file__), '../lib/micropython-lib'), help='path to micropython-lib repo')
    cmd_parser.add_argument('--port', default=None, help='path to port dir')
    cmd_parser.add_argument('--board', default=None, help='path to board dir')
    cmd_parser.add_argument('--top', default=os.path.join(os.path.dirname(__file__), '..'), help='path to micropython repo')
    cmd_parser.add_argument('files', nargs='+', help='input manifest.py')
    args = cmd_parser.parse_args()
    path_vars = {'MPY_DIR': os.path.abspath(args.top) if args.top else None, 'BOARD_DIR': os.path.abspath(args.board) if args.board else None, 'PORT_DIR': os.path.abspath(args.port) if args.port else None, 'MPY_LIB_DIR': os.path.abspath(args.lib) if args.lib else None}
    mode = None
    if args.freeze:
        mode = MODE_FREEZE
    elif args.compile:
        mode = MODE_COMPILE
    elif args.pyproject:
        mode = MODE_PYPROJECT
    else:
        print('Error: No mode specified.', file=sys.stderr)
        exit(1)
    m = ManifestFile(mode, path_vars)
    for manifest_file in args.files:
        try:
            m.execute(manifest_file)
        except ManifestFileError as er:
            print(er, file=sys.stderr)
            exit(1)
    print(m.metadata())
    for f in m.files():
        print(f)
    if mode == MODE_PYPROJECT:
        for r in m.pypi_dependencies():
            print('pypi-require:', r)
if __name__ == '__main__':
    main()