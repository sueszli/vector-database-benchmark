import os
import pathlib
import warnings
from PyInstaller import log as logging
from PyInstaller.building.utils import _check_guts_eq
from PyInstaller.utils import misc
logger = logging.getLogger(__name__)

def unique_name(entry):
    if False:
        return 10
    '\n    Return the filename used to enforce uniqueness for the given TOC entry.\n\n    Parameters\n    ----------\n    entry : tuple\n\n    Returns\n    -------\n    unique_name: str\n    '
    (name, path, typecode) = entry
    if typecode in ('BINARY', 'DATA', 'EXTENSION', 'DEPENDENCY'):
        name = os.path.normcase(name)
    return name

class TOC(list):
    """
    TOC (Table of Contents) class is a list of tuples of the form (name, path, typecode).

    typecode    name                   path                        description
    --------------------------------------------------------------------------------------
    EXTENSION   Python internal name.  Full path name in build.    Extension module.
    PYSOURCE    Python internal name.  Full path name in build.    Script.
    PYMODULE    Python internal name.  Full path name in build.    Pure Python module (including __init__ modules).
    PYZ         Runtime name.          Full path name in build.    A .pyz archive (ZlibArchive data structure).
    PKG         Runtime name.          Full path name in build.    A .pkg archive (Carchive data structure).
    BINARY      Runtime name.          Full path name in build.    Shared library.
    DATA        Runtime name.          Full path name in build.    Arbitrary files.
    OPTION      The option.            Unused.                     Python runtime option (frozen into executable).

    A TOC contains various types of files. A TOC contains no duplicates and preserves order.
    PyInstaller uses TOC data type to collect necessary files bundle them into an executable.
    """

    def __init__(self, initlist=None):
        if False:
            print('Hello World!')
        super().__init__()
        warnings.warn('TOC class is deprecated. Use a plain list of 3-element tuples instead.', DeprecationWarning, stacklevel=2)
        self.filenames = set()
        if initlist:
            for entry in initlist:
                self.append(entry)

    def append(self, entry):
        if False:
            while True:
                i = 10
        if not isinstance(entry, tuple):
            logger.info('TOC found a %s, not a tuple', entry)
            raise TypeError('Expected tuple, not %s.' % type(entry).__name__)
        unique = unique_name(entry)
        if unique not in self.filenames:
            self.filenames.add(unique)
            super().append(entry)

    def insert(self, pos, entry):
        if False:
            while True:
                i = 10
        if not isinstance(entry, tuple):
            logger.info('TOC found a %s, not a tuple', entry)
            raise TypeError('Expected tuple, not %s.' % type(entry).__name__)
        unique = unique_name(entry)
        if unique not in self.filenames:
            self.filenames.add(unique)
            super().insert(pos, entry)

    def __add__(self, other):
        if False:
            print('Hello World!')
        result = TOC(self)
        result.extend(other)
        return result

    def __radd__(self, other):
        if False:
            i = 10
            return i + 15
        result = TOC(other)
        result.extend(self)
        return result

    def __iadd__(self, other):
        if False:
            while True:
                i = 10
        for entry in other:
            self.append(entry)
        return self

    def extend(self, other):
        if False:
            for i in range(10):
                print('nop')
        for entry in other:
            self.append(entry)

    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        other = TOC(other)
        return TOC([entry for entry in self if unique_name(entry) not in other.filenames])

    def __rsub__(self, other):
        if False:
            while True:
                i = 10
        result = TOC(other)
        return result.__sub__(self)

    def __setitem__(self, key, value):
        if False:
            while True:
                i = 10
        if isinstance(key, slice):
            if key == slice(None, None, None):
                self.filenames = set()
                self.clear()
                self.extend(value)
                return
            else:
                raise KeyError("TOC.__setitem__ doesn't handle slices")
        else:
            old_value = self[key]
            old_name = unique_name(old_value)
            self.filenames.remove(old_name)
            new_name = unique_name(value)
            if new_name not in self.filenames:
                self.filenames.add(new_name)
                super(TOC, self).__setitem__(key, value)

class Target:
    invcnum = 0

    def __init__(self):
        if False:
            while True:
                i = 10
        from PyInstaller.config import CONF
        self.invcnum = self.__class__.invcnum
        self.__class__.invcnum += 1
        self.tocfilename = os.path.join(CONF['workpath'], '%s-%02d.toc' % (self.__class__.__name__, self.invcnum))
        self.tocbasename = os.path.basename(self.tocfilename)
        self.dependencies = []

    def __postinit__(self):
        if False:
            while True:
                i = 10
        '\n        Check if the target need to be rebuild and if so, re-assemble.\n\n        `__postinit__` is to be called at the end of `__init__` of every subclass of Target. `__init__` is meant to\n        setup the parameters and `__postinit__` is checking if rebuild is required and in case calls `assemble()`\n        '
        logger.info('checking %s', self.__class__.__name__)
        data = None
        last_build = misc.mtime(self.tocfilename)
        if last_build == 0:
            logger.info('Building %s because %s is non existent', self.__class__.__name__, self.tocbasename)
        else:
            try:
                data = misc.load_py_data_struct(self.tocfilename)
            except Exception:
                logger.info('Building because %s is bad', self.tocbasename)
            else:
                data = dict(zip((g[0] for g in self._GUTS), data))
        if not data or self._check_guts(data, last_build):
            self.assemble()
            self._save_guts()
    _GUTS = []

    def _check_guts(self, data, last_build):
        if False:
            return 10
        '\n        Returns True if rebuild/assemble is required.\n        '
        if len(data) != len(self._GUTS):
            logger.info('Building because %s is bad', self.tocbasename)
            return True
        for (attr, func) in self._GUTS:
            if func is None:
                continue
            if func(attr, data[attr], getattr(self, attr), last_build):
                return True
        return False

    def _save_guts(self):
        if False:
            i = 10
            return i + 15
        '\n        Save the input parameters and the work-product of this run to maybe avoid regenerating it later.\n        '
        data = tuple((getattr(self, g[0]) for g in self._GUTS))
        misc.save_py_data_struct(self.tocfilename, data)

class Tree(Target, list):
    """
    This class is a way of creating a TOC (Table of Contents) list that describes some or all of the files within a
    directory.
    """

    def __init__(self, root=None, prefix=None, excludes=None, typecode='DATA'):
        if False:
            while True:
                i = 10
        '\n        root\n                The root of the tree (on the build system).\n        prefix\n                Optional prefix to the names of the target system.\n        excludes\n                A list of names to exclude. Two forms are allowed:\n\n                    name\n                        Files with this basename will be excluded (do not include the path).\n                    *.ext\n                        Any file with the given extension will be excluded.\n        typecode\n                The typecode to be used for all files found in this tree. See the TOC class for for information about\n                the typcodes.\n        '
        Target.__init__(self)
        list.__init__(self)
        self.root = root
        self.prefix = prefix
        self.excludes = excludes
        self.typecode = typecode
        if excludes is None:
            self.excludes = []
        self.__postinit__()
    _GUTS = (('root', _check_guts_eq), ('prefix', _check_guts_eq), ('excludes', _check_guts_eq), ('typecode', _check_guts_eq), ('data', None))

    def _check_guts(self, data, last_build):
        if False:
            return 10
        if Target._check_guts(self, data, last_build):
            return True
        stack = [data['root']]
        while stack:
            d = stack.pop()
            if misc.mtime(d) > last_build:
                logger.info('Building %s because directory %s changed', self.tocbasename, d)
                return True
            for nm in os.listdir(d):
                path = os.path.join(d, nm)
                if os.path.isdir(path):
                    stack.append(path)
        self[:] = data['data']
        return False

    def _save_guts(self):
        if False:
            return 10
        self.data = self
        super()._save_guts()
        del self.data

    def assemble(self):
        if False:
            while True:
                i = 10
        logger.info('Building Tree %s', self.tocbasename)
        stack = [(self.root, self.prefix)]
        excludes = set()
        xexcludes = set()
        for name in self.excludes:
            if name.startswith('*'):
                xexcludes.add(name[1:])
            else:
                excludes.add(name)
        result = []
        while stack:
            (dir, prefix) = stack.pop()
            for filename in os.listdir(dir):
                if filename in excludes:
                    continue
                ext = os.path.splitext(filename)[1]
                if ext in xexcludes:
                    continue
                fullfilename = os.path.join(dir, filename)
                if prefix:
                    resfilename = os.path.join(prefix, filename)
                else:
                    resfilename = filename
                if os.path.isdir(fullfilename):
                    stack.append((fullfilename, resfilename))
                else:
                    result.append((resfilename, fullfilename, self.typecode))
        self[:] = result

def normalize_toc(toc):
    if False:
        print('Hello World!')
    _TOC_TYPE_PRIORITIES = {'DEPENDENCY': 3, 'SYMLINK': 2, 'BINARY': 1, 'EXTENSION': 1}

    def _type_case_normalization_fcn(typecode):
        if False:
            print('Hello World!')
        return typecode not in {'OPTION'}
    return _normalize_toc(toc, _TOC_TYPE_PRIORITIES, _type_case_normalization_fcn)

def normalize_pyz_toc(toc):
    if False:
        i = 10
        return i + 15
    _TOC_TYPE_PRIORITIES = {'PYMODULE': 1}
    return _normalize_toc(toc, _TOC_TYPE_PRIORITIES)

def _normalize_toc(toc, toc_type_priorities, type_case_normalization_fcn=lambda typecode: False):
    if False:
        while True:
            i = 10
    options_toc = []
    tmp_toc = dict()
    for (dest_name, src_name, typecode) in toc:
        if typecode == 'OPTION':
            options_toc.append((dest_name, src_name, typecode))
            continue
        dest_name = os.path.normpath(dest_name)
        if type_case_normalization_fcn(typecode):
            entry_key = pathlib.PurePath(dest_name)
        else:
            entry_key = dest_name
        existing_entry = tmp_toc.get(entry_key)
        if existing_entry is None:
            tmp_toc[entry_key] = (dest_name, src_name, typecode)
        else:
            (_, _, existing_typecode) = existing_entry
            if toc_type_priorities.get(typecode, 0) > toc_type_priorities.get(existing_typecode, 0):
                tmp_toc[entry_key] = (dest_name, src_name, typecode)
    return options_toc + list(tmp_toc.values())

def toc_process_symbolic_links(toc):
    if False:
        while True:
            i = 10
    '\n    Process TOC entries and replace entries whose files are symbolic links with SYMLINK entries (provided original file\n    is also being collected).\n    '
    all_dest_files = set([dest_name for (dest_name, src_name, typecode) in toc])
    new_toc = []
    for entry in toc:
        (dest_name, src_name, typecode) = entry
        if typecode == 'SYMLINK':
            new_toc.append(entry)
            continue
        if not src_name:
            new_toc.append(entry)
            continue
        if not os.path.islink(src_name):
            new_toc.append(entry)
            continue
        symlink_entry = _try_preserving_symbolic_link(dest_name, src_name, all_dest_files)
        if symlink_entry:
            new_toc.append(symlink_entry)
        else:
            new_toc.append(entry)
    return new_toc

def _try_preserving_symbolic_link(dest_name, src_name, all_dest_files):
    if False:
        print('Hello World!')
    seen_src_files = set()
    ref_src_file = src_name
    ref_dest_file = dest_name
    while True:
        if ref_src_file in seen_src_files:
            break
        seen_src_files.add(ref_src_file)
        if not os.path.islink(ref_src_file):
            break
        symlink_target = os.readlink(ref_src_file)
        if os.path.isabs(symlink_target):
            break
        ref_dest_file = os.path.join(os.path.dirname(ref_dest_file), symlink_target)
        ref_dest_file = os.path.normpath(ref_dest_file)
        ref_src_file = os.path.join(os.path.dirname(ref_src_file), symlink_target)
        ref_src_file = os.path.normpath(ref_src_file)
        if ref_dest_file in all_dest_files:
            if os.path.realpath(src_name) == os.path.realpath(ref_src_file):
                rel_link = os.path.relpath(ref_dest_file, os.path.dirname(dest_name))
                return (dest_name, rel_link, 'SYMLINK')
    return None