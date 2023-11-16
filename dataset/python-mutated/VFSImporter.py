"""The VFS importer allows importing Python modules from Panda3D's virtual
file system, through Python's standard import mechanism.

Calling the :func:`register()` function to register the import hooks should be
sufficient to enable this functionality.
"""
__all__ = ['register', 'sharedPackages', 'reloadSharedPackage', 'reloadSharedPackages']
from panda3d.core import Filename, VirtualFileSystem, VirtualFileMountSystem, OFileStream, copyStream
from direct.stdpy.file import open
import sys
import marshal
import imp
import types
sharedPackages = {}
vfs = VirtualFileSystem.getGlobalPtr()
compiledExtensions = ['pyc', 'pyo']
if not __debug__:
    compiledExtensions = ['pyo', 'pyc']

class VFSImporter:
    """ This class serves as a Python importer to support loading
    Python .py and .pyc/.pyo files from Panda's Virtual File System,
    which allows loading Python source files from mounted .mf files
    (among other places). """

    def __init__(self, path):
        if False:
            print('Hello World!')
        if isinstance(path, Filename):
            self.dir_path = Filename(path)
        else:
            self.dir_path = Filename.fromOsSpecific(path)

    def find_module(self, fullname, path=None):
        if False:
            i = 10
            return i + 15
        if path is None:
            dir_path = self.dir_path
        else:
            dir_path = path
        basename = fullname.split('.')[-1]
        path = Filename(dir_path, basename)
        filename = Filename(path)
        filename.setExtension('py')
        vfile = vfs.getFile(filename, True)
        if vfile:
            return VFSLoader(dir_path, vfile, filename, desc=('.py', 'r', imp.PY_SOURCE))
        for ext in compiledExtensions:
            filename = Filename(path)
            filename.setExtension(ext)
            vfile = vfs.getFile(filename, True)
            if vfile:
                return VFSLoader(dir_path, vfile, filename, desc=('.' + ext, 'rb', imp.PY_COMPILED))
        for desc in imp.get_suffixes():
            if desc[2] != imp.C_EXTENSION:
                continue
            filename = Filename(path + desc[0])
            vfile = vfs.getFile(filename, True)
            if vfile:
                return VFSLoader(dir_path, vfile, filename, desc=desc)
        filename = Filename(path, '__init__.py')
        vfile = vfs.getFile(filename, True)
        if vfile:
            return VFSLoader(dir_path, vfile, filename, packagePath=path, desc=('.py', 'r', imp.PY_SOURCE))
        for ext in compiledExtensions:
            filename = Filename(path, '__init__.' + ext)
            vfile = vfs.getFile(filename, True)
            if vfile:
                return VFSLoader(dir_path, vfile, filename, packagePath=path, desc=('.' + ext, 'rb', imp.PY_COMPILED))
        return None

class VFSLoader:
    """ The second part of VFSImporter, this is created for a
    particular .py file or directory. """

    def __init__(self, dir_path, vfile, filename, desc, packagePath=None):
        if False:
            for i in range(10):
                print('nop')
        self.dir_path = dir_path
        self.timestamp = None
        if vfile:
            self.timestamp = vfile.getTimestamp()
        self.filename = filename
        self.desc = desc
        self.packagePath = packagePath

    def load_module(self, fullname, loadingShared=False):
        if False:
            for i in range(10):
                print('nop')
        if self.desc[2] == imp.PY_FROZEN:
            return self._import_frozen_module(fullname)
        if self.desc[2] == imp.C_EXTENSION:
            return self._import_extension_module(fullname)
        if not loadingShared and self.packagePath and ('.' in fullname):
            parentname = fullname.rsplit('.', 1)[0]
            if parentname in sharedPackages:
                parent = sys.modules[parentname]
                path = getattr(parent, '__path__', None)
                importer = VFSSharedImporter()
                sharedPackages[fullname] = True
                loader = importer.find_module(fullname, path=path)
                assert loader
                return loader.load_module(fullname)
        code = self._read_code()
        if not code:
            raise ImportError('No Python code in %s' % fullname)
        mod = sys.modules.setdefault(fullname, imp.new_module(fullname))
        mod.__file__ = self.filename.toOsSpecific()
        mod.__loader__ = self
        if self.packagePath:
            mod.__path__ = [self.packagePath.toOsSpecific()]
        exec(code, mod.__dict__)
        return sys.modules[fullname]

    def getdata(self, path):
        if False:
            return 10
        path = Filename(self.dir_path, Filename.fromOsSpecific(path))
        vfile = vfs.getFile(path)
        if not vfile:
            raise IOError("Could not find '%s'" % path)
        return vfile.readFile(True)

    def is_package(self, fullname):
        if False:
            print('Hello World!')
        return bool(self.packagePath)

    def get_code(self, fullname):
        if False:
            while True:
                i = 10
        return self._read_code()

    def get_source(self, fullname):
        if False:
            i = 10
            return i + 15
        return self._read_source()

    def get_filename(self, fullname):
        if False:
            return 10
        return self.filename.toOsSpecific()

    def _read_source(self):
        if False:
            i = 10
            return i + 15
        ' Returns the Python source for this file, if it is\n        available, or None if it is not.  May raise IOError. '
        if self.desc[2] == imp.PY_COMPILED or self.desc[2] == imp.C_EXTENSION:
            return None
        filename = Filename(self.filename)
        filename.setExtension('py')
        filename.setText()
        import tokenize
        fh = open(self.filename, 'rb')
        (encoding, lines) = tokenize.detect_encoding(fh.readline)
        return (b''.join(lines) + fh.read()).decode(encoding)

    def _import_extension_module(self, fullname):
        if False:
            while True:
                i = 10
        ' Loads the binary shared object as a Python module, and\n        returns it. '
        vfile = vfs.getFile(self.filename, False)
        if hasattr(vfile, 'getMount') and isinstance(vfile.getMount(), VirtualFileMountSystem):
            filename = self.filename
        elif self.filename.exists():
            filename = self.filename
        else:
            filename = Filename.temporary('', self.filename.getBasenameWoExtension(), '.' + self.filename.getExtension(), type=Filename.TDso)
            filename.setExtension(self.filename.getExtension())
            filename.setBinary()
            sin = vfile.openReadFile(True)
            sout = OFileStream()
            if not filename.openWrite(sout):
                raise IOError
            if not copyStream(sin, sout):
                raise IOError
            vfile.closeReadFile(sin)
            del sout
        module = imp.load_module(fullname, None, filename.toOsSpecific(), self.desc)
        module.__file__ = self.filename.toOsSpecific()
        return module

    def _import_frozen_module(self, fullname):
        if False:
            for i in range(10):
                print('nop')
        ' Imports the frozen module without messing around with\n        searching any more. '
        module = imp.load_module(fullname, None, fullname, ('', '', imp.PY_FROZEN))
        module.__path__ = []
        return module

    def _read_code(self):
        if False:
            print('Hello World!')
        ' Returns the Python compiled code object for this file, if\n        it is available, or None if it is not.  May raise IOError,\n        ValueError, SyntaxError, or a number of other errors generated\n        by the low-level system. '
        if self.desc[2] == imp.PY_COMPILED:
            pycVfile = vfs.getFile(self.filename, False)
            if pycVfile:
                return self._loadPyc(pycVfile, None)
            raise IOError('Could not read %s' % self.filename)
        elif self.desc[2] == imp.C_EXTENSION:
            return None
        t_pyc = None
        for ext in compiledExtensions:
            pycFilename = Filename(self.filename)
            pycFilename.setExtension(ext)
            pycVfile = vfs.getFile(pycFilename, False)
            if pycVfile:
                t_pyc = pycVfile.getTimestamp()
                break
        code = None
        if t_pyc and t_pyc >= self.timestamp:
            try:
                code = self._loadPyc(pycVfile, self.timestamp)
            except ValueError:
                code = None
        if not code:
            source = self._read_source()
            filename = Filename(self.filename)
            filename.setExtension('py')
            code = self._compile(filename, source)
        return code

    def _loadPyc(self, vfile, timestamp):
        if False:
            return 10
        ' Reads and returns the marshal data from a .pyc file.\n        Raises ValueError if there is a problem. '
        code = None
        data = vfile.readFile(True)
        if data[:4] != imp.get_magic():
            raise ValueError('Bad magic number in %s' % vfile)
        t = int.from_bytes(data[4:8], 'little')
        data = data[12:]
        if not timestamp or t == timestamp:
            return marshal.loads(data)
        else:
            raise ValueError('Timestamp wrong on %s' % vfile)

    def _compile(self, filename, source):
        if False:
            return 10
        ' Compiles the Python source code to a code object and\n        attempts to write it to an appropriate .pyc file.  May raise\n        SyntaxError or other errors generated by the compiler. '
        if source and source[-1] != '\n':
            source = source + '\n'
        code = compile(source, filename.toOsSpecific(), 'exec')
        pycFilename = Filename(filename)
        pycFilename.setExtension(compiledExtensions[0])
        try:
            f = open(pycFilename.toOsSpecific(), 'wb')
        except IOError:
            pass
        else:
            f.write(imp.get_magic())
            f.write((self.timestamp & 4294967295).to_bytes(4, 'little'))
            f.write(b'\x00\x00\x00\x00')
            f.write(marshal.dumps(code))
            f.close()
        return code

class VFSSharedImporter:
    """ This is a special importer that is added onto the meta_path
    list, so that it is called before sys.path is traversed.  It uses
    special logic to load one of the "shared" packages, by searching
    the entire sys.path for all instances of this shared package, and
    merging them. """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def find_module(self, fullname, path=None, reload=False):
        if False:
            i = 10
            return i + 15
        if fullname not in sharedPackages:
            return None
        if path is None:
            path = sys.path
        excludePaths = []
        if reload:
            mod = sys.modules[fullname]
            excludePaths = getattr(mod, '_vfs_shared_path', None)
            if excludePaths is None:
                d = self.getLoadedDirname(mod)
                excludePaths = [d]
        loaders = []
        for dir in path:
            if dir in excludePaths:
                continue
            importer = sys.path_importer_cache.get(dir, None)
            if importer is None:
                try:
                    importer = VFSImporter(dir)
                except ImportError:
                    continue
                sys.path_importer_cache[dir] = importer
            try:
                loader = importer.find_module(fullname)
                if not loader:
                    continue
            except ImportError:
                continue
            loaders.append(loader)
        if not loaders:
            return None
        return VFSSharedLoader(loaders, reload=reload)

    def getLoadedDirname(self, mod):
        if False:
            i = 10
            return i + 15
        ' Returns the directory name that the indicated\n        conventionally-loaded module must have been loaded from. '
        if not getattr(mod, '__file__', None):
            return None
        fullname = mod.__name__
        dirname = Filename.fromOsSpecific(mod.__file__).getDirname()
        parentname = None
        basename = fullname
        if '.' in fullname:
            (parentname, basename) = fullname.rsplit('.', 1)
        path = None
        if parentname:
            parent = sys.modules[parentname]
            path = parent.__path__
        if path is None:
            path = sys.path
        for dir in path:
            pdir = str(Filename.fromOsSpecific(dir))
            if pdir + '/' + basename == dirname:
                return dir
        return None

class VFSSharedLoader:
    """ The second part of VFSSharedImporter, this imports a list of
    packages and combines them. """

    def __init__(self, loaders, reload):
        if False:
            i = 10
            return i + 15
        self.loaders = loaders
        self.reload = reload

    def load_module(self, fullname):
        if False:
            print('Hello World!')
        mod = None
        message = None
        path = []
        vfs_shared_path = []
        if self.reload:
            mod = sys.modules[fullname]
            path = mod.__path__ or []
            if path == fullname:
                path = []
            vfs_shared_path = getattr(mod, '_vfs_shared_path', [])
        for loader in self.loaders:
            try:
                mod = loader.load_module(fullname, loadingShared=True)
            except ImportError:
                (etype, evalue, etraceback) = sys.exc_info()
                print('%s on %s: %s' % (etype.__name__, fullname, evalue))
                if not message:
                    message = '%s: %s' % (fullname, evalue)
                continue
            for dir in getattr(mod, '__path__', []):
                if dir not in path:
                    path.append(dir)
        if mod is None:
            raise ImportError(message)
        mod.__path__ = path
        mod.__package__ = fullname
        mod._vfs_shared_path = vfs_shared_path + [l.dir_path for l in self.loaders]
        return mod
_registered = False

def register():
    if False:
        while True:
            i = 10
    " Register the VFSImporter on the path_hooks, if it has not\n    already been registered, so that future Python import statements\n    will vector through here (and therefore will take advantage of\n    Panda's virtual file system). "
    global _registered
    if not _registered:
        _registered = True
        sys.path_hooks.insert(0, VFSImporter)
        sys.meta_path.insert(0, VFSSharedImporter())
        sys.path_importer_cache = {}

def reloadSharedPackage(mod):
    if False:
        for i in range(10):
            print('nop')
    ' Reloads the specific module as a shared package, adding any\n    new directories that might have appeared on the search path. '
    fullname = mod.__name__
    path = None
    if '.' in fullname:
        parentname = fullname.rsplit('.', 1)[0]
        parent = sys.modules[parentname]
        path = parent.__path__
    importer = VFSSharedImporter()
    loader = importer.find_module(fullname, path=path, reload=True)
    if loader:
        loader.load_module(fullname)
    for (basename, child) in list(mod.__dict__.items()):
        if isinstance(child, types.ModuleType):
            childname = child.__name__
            if childname == fullname + '.' + basename and hasattr(child, '__path__') and (childname not in sharedPackages):
                sharedPackages[childname] = True
                reloadSharedPackage(child)

def reloadSharedPackages():
    if False:
        return 10
    ' Walks through the sharedPackages list, and forces a reload of\n    any modules on that list that have already been loaded.  This\n    allows new directories to be added to the search path. '
    for fullname in sorted(sharedPackages.keys()):
        mod = sys.modules.get(fullname, None)
        if not mod:
            continue
        reloadSharedPackage(mod)