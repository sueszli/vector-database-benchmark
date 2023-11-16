"""scons.Node.FS

File system nodes.

These Nodes represent the canonical external objects that people think
of when they think of building software: files and directories.

This holds a "default_fs" variable that should be initialized with an FS
that can be used by scripts or modules looking for the canonical default.

"""
from __future__ import print_function
__revision__ = 'src/engine/SCons/Node/FS.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import fnmatch
import os
import re
import shutil
import stat
import sys
import time
import codecs
from itertools import chain
import SCons.Action
import SCons.Debug
from SCons.Debug import logInstanceCreation
import SCons.Errors
import SCons.Memoize
import SCons.Node
import SCons.Node.Alias
import SCons.Subst
import SCons.Util
import SCons.Warnings
from SCons.Debug import Trace
print_duplicate = 0
MD5_TIMESTAMP_DEBUG = False

def sconsign_none(node):
    if False:
        return 10
    raise NotImplementedError

def sconsign_dir(node):
    if False:
        while True:
            i = 10
    'Return the .sconsign file info for this directory,\n    creating it first if necessary.'
    if not node._sconsign:
        import SCons.SConsign
        node._sconsign = SCons.SConsign.ForDirectory(node)
    return node._sconsign
_sconsign_map = {0: sconsign_none, 1: sconsign_dir}

class FileBuildInfoFileToCsigMappingError(Exception):
    pass

class EntryProxyAttributeError(AttributeError):
    """
    An AttributeError subclass for recording and displaying the name
    of the underlying Entry involved in an AttributeError exception.
    """

    def __init__(self, entry_proxy, attribute):
        if False:
            for i in range(10):
                print('nop')
        AttributeError.__init__(self)
        self.entry_proxy = entry_proxy
        self.attribute = attribute

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        entry = self.entry_proxy.get()
        fmt = '%s instance %s has no attribute %s'
        return fmt % (entry.__class__.__name__, repr(entry.name), repr(self.attribute))
default_max_drift = 2 * 24 * 60 * 60
Save_Strings = None

def save_strings(val):
    if False:
        for i in range(10):
            print('nop')
    global Save_Strings
    Save_Strings = val
do_splitdrive = None
_my_splitdrive = None

def initialize_do_splitdrive():
    if False:
        for i in range(10):
            print('nop')
    global do_splitdrive
    global has_unc
    (drive, path) = os.path.splitdrive('X:/foo')
    has_unc = hasattr(os.path, 'splitunc') or os.path.splitdrive('\\\\split\\drive\\test')[0] == '\\\\split\\drive'
    do_splitdrive = not not drive or has_unc
    global _my_splitdrive
    if has_unc:

        def splitdrive(p):
            if False:
                for i in range(10):
                    print('nop')
            if p[1:2] == ':':
                return (p[:2], p[2:])
            if p[0:2] == '//':
                return ('//', p[1:])
            return ('', p)
    else:

        def splitdrive(p):
            if False:
                for i in range(10):
                    print('nop')
            if p[1:2] == ':':
                return (p[:2], p[2:])
            return ('', p)
    _my_splitdrive = splitdrive
    global OS_SEP
    global UNC_PREFIX
    global os_sep_is_slash
    OS_SEP = os.sep
    UNC_PREFIX = OS_SEP + OS_SEP
    os_sep_is_slash = OS_SEP == '/'
initialize_do_splitdrive()
needs_normpath_check = re.compile("\n      # We need to renormalize the path if it contains any consecutive\n      # '/' characters.\n      .*// |\n\n      # We need to renormalize the path if it contains a '..' directory.\n      # Note that we check for all the following cases:\n      #\n      #    a) The path is a single '..'\n      #    b) The path starts with '..'. E.g. '../' or '../moredirs'\n      #       but we not match '..abc/'.\n      #    c) The path ends with '..'. E.g. '/..' or 'dirs/..'\n      #    d) The path contains a '..' in the middle.\n      #       E.g. dirs/../moredirs\n\n      (.*/)?\\.\\.(?:/|$) |\n\n      # We need to renormalize the path if it contains a '.'\n      # directory, but NOT if it is a single '.'  '/' characters. We\n      # do not want to match a single '.' because this case is checked\n      # for explicitly since this is common enough case.\n      #\n      # Note that we check for all the following cases:\n      #\n      #    a) We don't match a single '.'\n      #    b) We match if the path starts with '.'. E.g. './' or\n      #       './moredirs' but we not match '.abc/'.\n      #    c) We match if the path ends with '.'. E.g. '/.' or\n      #    'dirs/.'\n      #    d) We match if the path contains a '.' in the middle.\n      #       E.g. dirs/./moredirs\n\n      \\./|.*/\\.(?:/|$)\n\n    ", re.VERBOSE)
needs_normpath_match = needs_normpath_check.match
if hasattr(os, 'link') and sys.platform != 'win32':

    def _hardlink_func(fs, src, dst):
        if False:
            while True:
                i = 10
        while fs.islink(src):
            link = fs.readlink(src)
            if not os.path.isabs(link):
                src = link
            else:
                src = os.path.join(os.path.dirname(src), link)
        fs.link(src, dst)
else:
    _hardlink_func = None
if hasattr(os, 'symlink') and sys.platform != 'win32':

    def _softlink_func(fs, src, dst):
        if False:
            return 10
        fs.symlink(src, dst)
else:
    _softlink_func = None

def _copy_func(fs, src, dest):
    if False:
        return 10
    shutil.copy2(src, dest)
    st = fs.stat(src)
    fs.chmod(dest, stat.S_IMODE(st[stat.ST_MODE]) | stat.S_IWRITE)
Valid_Duplicates = ['hard-soft-copy', 'soft-hard-copy', 'hard-copy', 'soft-copy', 'copy']
Link_Funcs = []

def set_duplicate(duplicate):
    if False:
        return 10
    link_dict = {'hard': _hardlink_func, 'soft': _softlink_func, 'copy': _copy_func}
    if duplicate not in Valid_Duplicates:
        raise SCons.Errors.InternalError('The argument of set_duplicate should be in Valid_Duplicates')
    global Link_Funcs
    Link_Funcs = []
    for func in duplicate.split('-'):
        if link_dict[func]:
            Link_Funcs.append(link_dict[func])

def LinkFunc(target, source, env):
    if False:
        i = 10
        return i + 15
    "\n    Relative paths cause problems with symbolic links, so\n    we use absolute paths, which may be a problem for people\n    who want to move their soft-linked src-trees around. Those\n    people should use the 'hard-copy' mode, softlinks cannot be\n    used for that; at least I have no idea how ...\n    "
    src = source[0].get_abspath()
    dest = target[0].get_abspath()
    (dir, file) = os.path.split(dest)
    if dir and (not target[0].fs.isdir(dir)):
        os.makedirs(dir)
    if not Link_Funcs:
        set_duplicate('hard-soft-copy')
    fs = source[0].fs
    for func in Link_Funcs:
        try:
            func(fs, src, dest)
            break
        except (IOError, OSError):
            if func == Link_Funcs[-1]:
                raise
    return 0
Link = SCons.Action.Action(LinkFunc, None)

def LocalString(target, source, env):
    if False:
        for i in range(10):
            print('nop')
    return 'Local copy of %s from %s' % (target[0], source[0])
LocalCopy = SCons.Action.Action(LinkFunc, LocalString)

def UnlinkFunc(target, source, env):
    if False:
        print('Hello World!')
    t = target[0]
    t.fs.unlink(t.get_abspath())
    return 0
Unlink = SCons.Action.Action(UnlinkFunc, None)

def MkdirFunc(target, source, env):
    if False:
        while True:
            i = 10
    t = target[0]
    if not t.exists() and (not os.path.exists(t.get_abspath())):
        t.fs.mkdir(t.get_abspath())
    return 0
Mkdir = SCons.Action.Action(MkdirFunc, None, presub=None)
MkdirBuilder = None

def get_MkdirBuilder():
    if False:
        return 10
    global MkdirBuilder
    if MkdirBuilder is None:
        import SCons.Builder
        import SCons.Defaults
        MkdirBuilder = SCons.Builder.Builder(action=Mkdir, env=None, explain=None, is_explicit=None, target_scanner=SCons.Defaults.DirEntryScanner, name='MkdirBuilder')
    return MkdirBuilder

class _Null(object):
    pass
_null = _Null()
_is_cygwin = sys.platform == 'cygwin'
if os.path.normcase('TeSt') == os.path.normpath('TeSt') and (not _is_cygwin):

    def _my_normcase(x):
        if False:
            while True:
                i = 10
        return x
else:

    def _my_normcase(x):
        if False:
            while True:
                i = 10
        return x.upper()

class DiskChecker(object):

    def __init__(self, type, do, ignore):
        if False:
            return 10
        self.type = type
        self.do = do
        self.ignore = ignore
        self.func = do

    def __call__(self, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        return self.func(*args, **kw)

    def set(self, list):
        if False:
            while True:
                i = 10
        if self.type in list:
            self.func = self.do
        else:
            self.func = self.ignore

def do_diskcheck_match(node, predicate, errorfmt):
    if False:
        i = 10
        return i + 15
    result = predicate()
    try:
        if node._memo['stat'] is None:
            del node._memo['stat']
    except (AttributeError, KeyError):
        pass
    if result:
        raise TypeError(errorfmt % node.get_abspath())

def ignore_diskcheck_match(node, predicate, errorfmt):
    if False:
        return 10
    pass
diskcheck_match = DiskChecker('match', do_diskcheck_match, ignore_diskcheck_match)
diskcheckers = [diskcheck_match]

def set_diskcheck(list):
    if False:
        while True:
            i = 10
    for dc in diskcheckers:
        dc.set(list)

def diskcheck_types():
    if False:
        while True:
            i = 10
    return [dc.type for dc in diskcheckers]

class EntryProxy(SCons.Util.Proxy):
    __str__ = SCons.Util.Delegate('__str__')
    __hash__ = SCons.Util.Delegate('__hash__')

    def __get_abspath(self):
        if False:
            i = 10
            return i + 15
        entry = self.get()
        return SCons.Subst.SpecialAttrWrapper(entry.get_abspath(), entry.name + '_abspath')

    def __get_filebase(self):
        if False:
            return 10
        name = self.get().name
        return SCons.Subst.SpecialAttrWrapper(SCons.Util.splitext(name)[0], name + '_filebase')

    def __get_suffix(self):
        if False:
            for i in range(10):
                print('nop')
        name = self.get().name
        return SCons.Subst.SpecialAttrWrapper(SCons.Util.splitext(name)[1], name + '_suffix')

    def __get_file(self):
        if False:
            return 10
        name = self.get().name
        return SCons.Subst.SpecialAttrWrapper(name, name + '_file')

    def __get_base_path(self):
        if False:
            print('Hello World!')
        "Return the file's directory and file name, with the\n        suffix stripped."
        entry = self.get()
        return SCons.Subst.SpecialAttrWrapper(SCons.Util.splitext(entry.get_path())[0], entry.name + '_base')

    def __get_posix_path(self):
        if False:
            print('Hello World!')
        'Return the path with / as the path separator,\n        regardless of platform.'
        if os_sep_is_slash:
            return self
        else:
            entry = self.get()
            r = entry.get_path().replace(OS_SEP, '/')
            return SCons.Subst.SpecialAttrWrapper(r, entry.name + '_posix')

    def __get_windows_path(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the path with \\ as the path separator,\n        regardless of platform.'
        if OS_SEP == '\\':
            return self
        else:
            entry = self.get()
            r = entry.get_path().replace(OS_SEP, '\\')
            return SCons.Subst.SpecialAttrWrapper(r, entry.name + '_windows')

    def __get_srcnode(self):
        if False:
            i = 10
            return i + 15
        return EntryProxy(self.get().srcnode())

    def __get_srcdir(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the directory containing the source node linked to this\n        node via VariantDir(), or the directory of this node if not linked.'
        return EntryProxy(self.get().srcnode().dir)

    def __get_rsrcnode(self):
        if False:
            print('Hello World!')
        return EntryProxy(self.get().srcnode().rfile())

    def __get_rsrcdir(self):
        if False:
            print('Hello World!')
        'Returns the directory containing the source node linked to this\n        node via VariantDir(), or the directory of this node if not linked.'
        return EntryProxy(self.get().srcnode().rfile().dir)

    def __get_dir(self):
        if False:
            for i in range(10):
                print('nop')
        return EntryProxy(self.get().dir)
    dictSpecialAttrs = {'base': __get_base_path, 'posix': __get_posix_path, 'windows': __get_windows_path, 'win32': __get_windows_path, 'srcpath': __get_srcnode, 'srcdir': __get_srcdir, 'dir': __get_dir, 'abspath': __get_abspath, 'filebase': __get_filebase, 'suffix': __get_suffix, 'file': __get_file, 'rsrcpath': __get_rsrcnode, 'rsrcdir': __get_rsrcdir}

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        try:
            attr_function = self.dictSpecialAttrs[name]
        except KeyError:
            try:
                attr = SCons.Util.Proxy.__getattr__(self, name)
            except AttributeError:
                raise EntryProxyAttributeError(self, name)
            return attr
        else:
            return attr_function(self)

class Base(SCons.Node.Node):
    """A generic class for file system entries.  This class is for
    when we don't know yet whether the entry being looked up is a file
    or a directory.  Instances of this class can morph into either
    Dir or File objects by a later, more precise lookup.

    Note: this class does not define __cmp__ and __hash__ for
    efficiency reasons.  SCons does a lot of comparing of
    Node.FS.{Base,Entry,File,Dir} objects, so those operations must be
    as fast as possible, which means we want to use Python's built-in
    object identity comparisons.
    """
    __slots__ = ['name', 'fs', '_abspath', '_labspath', '_path', '_tpath', '_path_elements', 'dir', 'cwd', 'duplicate', '_local', 'sbuilder', '_proxy', '_func_sconsign']

    def __init__(self, name, directory, fs):
        if False:
            print('Hello World!')
        'Initialize a generic Node.FS.Base object.\n\n        Call the superclass initialization, take care of setting up\n        our relative and absolute paths, identify our parent\n        directory, and indicate that this node should use\n        signatures.'
        if SCons.Debug.track_instances:
            logInstanceCreation(self, 'Node.FS.Base')
        SCons.Node.Node.__init__(self)
        self.name = SCons.Util.silent_intern(name)
        self.fs = fs
        assert directory, 'A directory must be provided'
        self._abspath = None
        self._labspath = None
        self._path = None
        self._tpath = None
        self._path_elements = None
        self.dir = directory
        self.cwd = None
        self.duplicate = directory.duplicate
        self.changed_since_last_build = 2
        self._func_sconsign = 0
        self._func_exists = 2
        self._func_rexists = 2
        self._func_get_contents = 0
        self._func_target_from_source = 1
        self.store_info = 1

    def str_for_display(self):
        if False:
            i = 10
            return i + 15
        return '"' + self.__str__() + '"'

    def must_be_same(self, klass):
        if False:
            i = 10
            return i + 15
        "\n        This node, which already existed, is being looked up as the\n        specified klass.  Raise an exception if it isn't.\n        "
        if isinstance(self, klass) or klass is Entry:
            return
        raise TypeError("Tried to lookup %s '%s' as a %s." % (self.__class__.__name__, self.get_internal_path(), klass.__name__))

    def get_dir(self):
        if False:
            return 10
        return self.dir

    def get_suffix(self):
        if False:
            while True:
                i = 10
        return SCons.Util.splitext(self.name)[1]

    def rfile(self):
        if False:
            while True:
                i = 10
        return self

    def __getattr__(self, attr):
        if False:
            return 10
        " Together with the node_bwcomp dict defined below,\n            this method provides a simple backward compatibility\n            layer for the Node attributes 'abspath', 'labspath',\n            'path', 'tpath', 'suffix' and 'path_elements'. These Node\n            attributes used to be directly available in v2.3 and earlier, but\n            have been replaced by getter methods that initialize the\n            single variables lazily when required, in order to save memory.\n            The redirection to the getters lets older Tools and\n            SConstruct continue to work without any additional changes,\n            fully transparent to the user.\n            Note, that __getattr__ is only called as fallback when the\n            requested attribute can't be found, so there should be no\n            speed performance penalty involved for standard builds.\n        "
        if attr in node_bwcomp:
            return node_bwcomp[attr](self)
        raise AttributeError('%r object has no attribute %r' % (self.__class__, attr))

    def __str__(self):
        if False:
            return 10
        "A Node.FS.Base object's string representation is its path\n        name."
        global Save_Strings
        if Save_Strings:
            return self._save_str()
        return self._get_str()

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        ' less than operator used by sorting on py3'
        return str(self) < str(other)

    @SCons.Memoize.CountMethodCall
    def _save_str(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._memo['_save_str']
        except KeyError:
            pass
        result = SCons.Util.silent_intern(self._get_str())
        self._memo['_save_str'] = result
        return result

    def _get_str(self):
        if False:
            while True:
                i = 10
        global Save_Strings
        if self.duplicate or self.is_derived():
            return self.get_path()
        srcnode = self.srcnode()
        if srcnode.stat() is None and self.stat() is not None:
            result = self.get_path()
        else:
            result = srcnode.get_path()
        if not Save_Strings:
            try:
                del self._memo['stat']
            except KeyError:
                pass
            if self is not srcnode:
                try:
                    del srcnode._memo['stat']
                except KeyError:
                    pass
        return result
    rstr = __str__

    @SCons.Memoize.CountMethodCall
    def stat(self):
        if False:
            print('Hello World!')
        try:
            return self._memo['stat']
        except KeyError:
            pass
        try:
            result = self.fs.stat(self.get_abspath())
        except os.error:
            result = None
        self._memo['stat'] = result
        return result

    def exists(self):
        if False:
            for i in range(10):
                print('nop')
        return SCons.Node._exists_map[self._func_exists](self)

    def rexists(self):
        if False:
            return 10
        return SCons.Node._rexists_map[self._func_rexists](self)

    def getmtime(self):
        if False:
            print('Hello World!')
        st = self.stat()
        if st:
            return st[stat.ST_MTIME]
        else:
            return None

    def getsize(self):
        if False:
            return 10
        st = self.stat()
        if st:
            return st[stat.ST_SIZE]
        else:
            return None

    def isdir(self):
        if False:
            i = 10
            return i + 15
        st = self.stat()
        return st is not None and stat.S_ISDIR(st[stat.ST_MODE])

    def isfile(self):
        if False:
            for i in range(10):
                print('nop')
        st = self.stat()
        return st is not None and stat.S_ISREG(st[stat.ST_MODE])
    if hasattr(os, 'symlink'):

        def islink(self):
            if False:
                return 10
            try:
                st = self.fs.lstat(self.get_abspath())
            except os.error:
                return 0
            return stat.S_ISLNK(st[stat.ST_MODE])
    else:

        def islink(self):
            if False:
                i = 10
                return i + 15
            return 0

    def is_under(self, dir):
        if False:
            while True:
                i = 10
        if self is dir:
            return 1
        else:
            return self.dir.is_under(dir)

    def set_local(self):
        if False:
            i = 10
            return i + 15
        self._local = 1

    def srcnode(self):
        if False:
            i = 10
            return i + 15
        'If this node is in a build path, return the node\n        corresponding to its source file.  Otherwise, return\n        ourself.\n        '
        srcdir_list = self.dir.srcdir_list()
        if srcdir_list:
            srcnode = srcdir_list[0].Entry(self.name)
            srcnode.must_be_same(self.__class__)
            return srcnode
        return self

    def get_path(self, dir=None):
        if False:
            return 10
        'Return path relative to the current working directory of the\n        Node.FS.Base object that owns us.'
        if not dir:
            dir = self.fs.getcwd()
        if self == dir:
            return '.'
        path_elems = self.get_path_elements()
        pathname = ''
        try:
            i = path_elems.index(dir)
        except ValueError:
            for p in path_elems[:-1]:
                pathname += p.dirname
        else:
            for p in path_elems[i + 1:-1]:
                pathname += p.dirname
        return pathname + path_elems[-1].name

    def set_src_builder(self, builder):
        if False:
            print('Hello World!')
        'Set the source code builder for this node.'
        self.sbuilder = builder
        if not self.has_builder():
            self.builder_set(builder)

    def src_builder(self):
        if False:
            for i in range(10):
                print('nop')
        "Fetch the source code builder for this node.\n\n        If there isn't one, we cache the source code builder specified\n        for the directory (which in turn will cache the value from its\n        parent directory, and so on up to the file system root).\n        "
        try:
            scb = self.sbuilder
        except AttributeError:
            scb = self.dir.src_builder()
            self.sbuilder = scb
        return scb

    def get_abspath(self):
        if False:
            print('Hello World!')
        'Get the absolute path of the file.'
        return self.dir.entry_abspath(self.name)

    def get_labspath(self):
        if False:
            return 10
        'Get the absolute path of the file.'
        return self.dir.entry_labspath(self.name)

    def get_internal_path(self):
        if False:
            for i in range(10):
                print('nop')
        if self.dir._path == '.':
            return self.name
        else:
            return self.dir.entry_path(self.name)

    def get_tpath(self):
        if False:
            print('Hello World!')
        if self.dir._tpath == '.':
            return self.name
        else:
            return self.dir.entry_tpath(self.name)

    def get_path_elements(self):
        if False:
            print('Hello World!')
        return self.dir._path_elements + [self]

    def for_signature(self):
        if False:
            while True:
                i = 10
        return self.name

    def get_subst_proxy(self):
        if False:
            return 10
        try:
            return self._proxy
        except AttributeError:
            ret = EntryProxy(self)
            self._proxy = ret
            return ret

    def target_from_source(self, prefix, suffix, splitext=SCons.Util.splitext):
        if False:
            while True:
                i = 10
        '\n\n        Generates a target entry that corresponds to this entry (usually\n        a source file) with the specified prefix and suffix.\n\n        Note that this method can be overridden dynamically for generated\n        files that need different behavior.  See Tool/swig.py for\n        an example.\n        '
        return SCons.Node._target_from_source_map[self._func_target_from_source](self, prefix, suffix, splitext)

    def _Rfindalldirs_key(self, pathlist):
        if False:
            print('Hello World!')
        return pathlist

    @SCons.Memoize.CountDictCall(_Rfindalldirs_key)
    def Rfindalldirs(self, pathlist):
        if False:
            i = 10
            return i + 15
        '\n        Return all of the directories for a given path list, including\n        corresponding "backing" directories in any repositories.\n\n        The Node lookups are relative to this Node (typically a\n        directory), so memoizing result saves cycles from looking\n        up the same path for each target in a given directory.\n        '
        try:
            memo_dict = self._memo['Rfindalldirs']
        except KeyError:
            memo_dict = {}
            self._memo['Rfindalldirs'] = memo_dict
        else:
            try:
                return memo_dict[pathlist]
            except KeyError:
                pass
        create_dir_relative_to_self = self.Dir
        result = []
        for path in pathlist:
            if isinstance(path, SCons.Node.Node):
                result.append(path)
            else:
                dir = create_dir_relative_to_self(path)
                result.extend(dir.get_all_rdirs())
        memo_dict[pathlist] = result
        return result

    def RDirs(self, pathlist):
        if False:
            for i in range(10):
                print('nop')
        'Search for a list of directories in the Repository list.'
        cwd = self.cwd or self.fs._cwd
        return cwd.Rfindalldirs(pathlist)

    @SCons.Memoize.CountMethodCall
    def rentry(self):
        if False:
            i = 10
            return i + 15
        try:
            return self._memo['rentry']
        except KeyError:
            pass
        result = self
        if not self.exists():
            norm_name = _my_normcase(self.name)
            for dir in self.dir.get_all_rdirs():
                try:
                    node = dir.entries[norm_name]
                except KeyError:
                    if dir.entry_exists_on_disk(self.name):
                        result = dir.Entry(self.name)
                        break
        self._memo['rentry'] = result
        return result

    def _glob1(self, pattern, ondisk=True, source=False, strings=False):
        if False:
            while True:
                i = 10
        return []
node_bwcomp = {'abspath': Base.get_abspath, 'labspath': Base.get_labspath, 'path': Base.get_internal_path, 'tpath': Base.get_tpath, 'path_elements': Base.get_path_elements, 'suffix': Base.get_suffix}

class Entry(Base):
    """This is the class for generic Node.FS entries--that is, things
    that could be a File or a Dir, but we're just not sure yet.
    Consequently, the methods in this class really exist just to
    transform their associated object into the right class when the
    time comes, and then call the same-named method in the transformed
    class."""
    __slots__ = ['scanner_paths', 'cachedir_csig', 'cachesig', 'repositories', 'srcdir', 'entries', 'searched', '_sconsign', 'variant_dirs', 'root', 'dirname', 'on_disk_entries', 'released_target_info', 'contentsig']

    def __init__(self, name, directory, fs):
        if False:
            for i in range(10):
                print('nop')
        Base.__init__(self, name, directory, fs)
        self._func_exists = 3
        self._func_get_contents = 1

    def diskcheck_match(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def disambiguate(self, must_exist=None):
        if False:
            while True:
                i = 10
        '\n        '
        if self.isfile():
            self.__class__ = File
            self._morph()
            self.clear()
        elif self.isdir():
            self.__class__ = Dir
            self._morph()
        else:
            srcdir = self.dir.srcnode()
            if srcdir != self.dir and srcdir.entry_exists_on_disk(self.name) and self.srcnode().isdir():
                self.__class__ = Dir
                self._morph()
            elif must_exist:
                msg = "No such file or directory: '%s'" % self.get_abspath()
                raise SCons.Errors.UserError(msg)
            else:
                self.__class__ = File
                self._morph()
                self.clear()
        return self

    def rfile(self):
        if False:
            i = 10
            return i + 15
        "We're a generic Entry, but the caller is actually looking for\n        a File at this point, so morph into one."
        self.__class__ = File
        self._morph()
        self.clear()
        return File.rfile(self)

    def scanner_key(self):
        if False:
            while True:
                i = 10
        return self.get_suffix()

    def get_contents(self):
        if False:
            while True:
                i = 10
        'Fetch the contents of the entry.  Returns the exact binary\n        contents of the file.'
        return SCons.Node._get_contents_map[self._func_get_contents](self)

    def get_text_contents(self):
        if False:
            for i in range(10):
                print('nop')
        'Fetch the decoded text contents of a Unicode encoded Entry.\n\n        Since this should return the text contents from the file\n        system, we check to see into what sort of subclass we should\n        morph this Entry.'
        try:
            self = self.disambiguate(must_exist=1)
        except SCons.Errors.UserError:
            return ''
        else:
            return self.get_text_contents()

    def must_be_same(self, klass):
        if False:
            while True:
                i = 10
        "Called to make sure a Node is a Dir.  Since we're an\n        Entry, we can morph into one."
        if self.__class__ is not klass:
            self.__class__ = klass
            self._morph()
            self.clear()

    def exists(self):
        if False:
            print('Hello World!')
        return SCons.Node._exists_map[self._func_exists](self)

    def rel_path(self, other):
        if False:
            return 10
        d = self.disambiguate()
        if d.__class__ is Entry:
            raise Exception('rel_path() could not disambiguate File/Dir')
        return d.rel_path(other)

    def new_ninfo(self):
        if False:
            i = 10
            return i + 15
        return self.disambiguate().new_ninfo()

    def _glob1(self, pattern, ondisk=True, source=False, strings=False):
        if False:
            while True:
                i = 10
        return self.disambiguate()._glob1(pattern, ondisk, source, strings)

    def get_subst_proxy(self):
        if False:
            return 10
        return self.disambiguate().get_subst_proxy()
_classEntry = Entry

class LocalFS(object):
    """
    This class implements an abstraction layer for operations involving
    a local file system.  Essentially, this wraps any function in
    the os, os.path or shutil modules that we use to actually go do
    anything with or to the local file system.

    Note that there's a very good chance we'll refactor this part of
    the architecture in some way as we really implement the interface(s)
    for remote file system Nodes.  For example, the right architecture
    might be to have this be a subclass instead of a base class.
    Nevertheless, we're using this as a first step in that direction.

    We're not using chdir() yet because the calling subclass method
    needs to use os.chdir() directly to avoid recursion.  Will we
    really need this one?
    """

    def chmod(self, path, mode):
        if False:
            i = 10
            return i + 15
        return os.chmod(path, mode)

    def copy(self, src, dst):
        if False:
            print('Hello World!')
        return shutil.copy(src, dst)

    def copy2(self, src, dst):
        if False:
            print('Hello World!')
        return shutil.copy2(src, dst)

    def exists(self, path):
        if False:
            i = 10
            return i + 15
        return os.path.exists(path)

    def getmtime(self, path):
        if False:
            return 10
        return os.path.getmtime(path)

    def getsize(self, path):
        if False:
            return 10
        return os.path.getsize(path)

    def isdir(self, path):
        if False:
            print('Hello World!')
        return os.path.isdir(path)

    def isfile(self, path):
        if False:
            print('Hello World!')
        return os.path.isfile(path)

    def link(self, src, dst):
        if False:
            while True:
                i = 10
        return os.link(src, dst)

    def lstat(self, path):
        if False:
            return 10
        return os.lstat(path)

    def listdir(self, path):
        if False:
            return 10
        return os.listdir(path)

    def makedirs(self, path):
        if False:
            i = 10
            return i + 15
        return os.makedirs(path)

    def mkdir(self, path):
        if False:
            print('Hello World!')
        return os.mkdir(path)

    def rename(self, old, new):
        if False:
            i = 10
            return i + 15
        return os.rename(old, new)

    def stat(self, path):
        if False:
            for i in range(10):
                print('nop')
        return os.stat(path)

    def symlink(self, src, dst):
        if False:
            for i in range(10):
                print('nop')
        return os.symlink(src, dst)

    def open(self, path):
        if False:
            return 10
        return open(path)

    def unlink(self, path):
        if False:
            return 10
        return os.unlink(path)
    if hasattr(os, 'symlink'):

        def islink(self, path):
            if False:
                while True:
                    i = 10
            return os.path.islink(path)
    else:

        def islink(self, path):
            if False:
                while True:
                    i = 10
            return 0
    if hasattr(os, 'readlink'):

        def readlink(self, file):
            if False:
                i = 10
                return i + 15
            return os.readlink(file)
    else:

        def readlink(self, file):
            if False:
                for i in range(10):
                    print('nop')
            return ''

class FS(LocalFS):

    def __init__(self, path=None):
        if False:
            print('Hello World!')
        'Initialize the Node.FS subsystem.\n\n        The supplied path is the top of the source tree, where we\n        expect to find the top-level build file.  If no path is\n        supplied, the current directory is the default.\n\n        The path argument must be a valid absolute path.\n        '
        if SCons.Debug.track_instances:
            logInstanceCreation(self, 'Node.FS')
        self._memo = {}
        self.Root = {}
        self.SConstruct_dir = None
        self.max_drift = default_max_drift
        self.Top = None
        if path is None:
            self.pathTop = os.getcwd()
        else:
            self.pathTop = path
        self.defaultDrive = _my_normcase(_my_splitdrive(self.pathTop)[0])
        self.Top = self.Dir(self.pathTop)
        self.Top._path = '.'
        self.Top._tpath = '.'
        self._cwd = self.Top
        DirNodeInfo.fs = self
        FileNodeInfo.fs = self

    def set_SConstruct_dir(self, dir):
        if False:
            i = 10
            return i + 15
        self.SConstruct_dir = dir

    def get_max_drift(self):
        if False:
            for i in range(10):
                print('nop')
        return self.max_drift

    def set_max_drift(self, max_drift):
        if False:
            i = 10
            return i + 15
        self.max_drift = max_drift

    def getcwd(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, '_cwd'):
            return self._cwd
        else:
            return '<no cwd>'

    def chdir(self, dir, change_os_dir=0):
        if False:
            for i in range(10):
                print('nop')
        'Change the current working directory for lookups.\n        If change_os_dir is true, we will also change the "real" cwd\n        to match.\n        '
        curr = self._cwd
        try:
            if dir is not None:
                self._cwd = dir
                if change_os_dir:
                    os.chdir(dir.get_abspath())
        except OSError:
            self._cwd = curr
            raise

    def get_root(self, drive):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the root directory for the specified drive, creating\n        it if necessary.\n        '
        drive = _my_normcase(drive)
        try:
            return self.Root[drive]
        except KeyError:
            root = RootDir(drive, self)
            self.Root[drive] = root
            if not drive:
                self.Root[self.defaultDrive] = root
            elif drive == self.defaultDrive:
                self.Root[''] = root
            return root

    def _lookup(self, p, directory, fsclass, create=1):
        if False:
            while True:
                i = 10
        "\n        The generic entry point for Node lookup with user-supplied data.\n\n        This translates arbitrary input into a canonical Node.FS object\n        of the specified fsclass.  The general approach for strings is\n        to turn it into a fully normalized absolute path and then call\n        the root directory's lookup_abs() method for the heavy lifting.\n\n        If the path name begins with '#', it is unconditionally\n        interpreted relative to the top-level directory of this FS.  '#'\n        is treated as a synonym for the top-level SConstruct directory,\n        much like '~' is treated as a synonym for the user's home\n        directory in a UNIX shell.  So both '#foo' and '#/foo' refer\n        to the 'foo' subdirectory underneath the top-level SConstruct\n        directory.\n\n        If the path name is relative, then the path is looked up relative\n        to the specified directory, or the current directory (self._cwd,\n        typically the SConscript directory) if the specified directory\n        is None.\n        "
        if isinstance(p, Base):
            p.must_be_same(fsclass)
            return p
        p = str(p)
        if not os_sep_is_slash:
            p = p.replace(OS_SEP, '/')
        if p[0:1] == '#':
            p = p[1:]
            directory = self.Top
            if do_splitdrive:
                (drive, p) = _my_splitdrive(p)
                if drive:
                    root = self.get_root(drive)
                else:
                    root = directory.root
            else:
                root = directory.root
            p = p.strip('/')
            needs_normpath = needs_normpath_match(p)
            if p in ('', '.'):
                p = directory.get_labspath()
            else:
                p = directory.get_labspath() + '/' + p
        else:
            if do_splitdrive:
                (drive, p) = _my_splitdrive(p)
                if drive and (not p):
                    p = '/'
            else:
                drive = ''
            if p != '/':
                p = p.rstrip('/')
            needs_normpath = needs_normpath_match(p)
            if p[0:1] == '/':
                root = self.get_root(drive)
            else:
                if directory:
                    if not isinstance(directory, Dir):
                        directory = self.Dir(directory)
                else:
                    directory = self._cwd
                if p in ('', '.'):
                    p = directory.get_labspath()
                else:
                    p = directory.get_labspath() + '/' + p
                if drive:
                    root = self.get_root(drive)
                else:
                    root = directory.root
        if needs_normpath is not None:
            ins = p.split('/')[1:]
            outs = []
            for d in ins:
                if d == '..':
                    try:
                        outs.pop()
                    except IndexError:
                        pass
                elif d not in ('', '.'):
                    outs.append(d)
            p = '/' + '/'.join(outs)
        return root._lookup_abs(p, fsclass, create)

    def Entry(self, name, directory=None, create=1):
        if False:
            return 10
        'Look up or create a generic Entry node with the specified name.\n        If the name is a relative path (begins with ./, ../, or a file\n        name), then it is looked up relative to the supplied directory\n        node, or to the top level directory of the FS (supplied at\n        construction time) if no directory is supplied.\n        '
        return self._lookup(name, directory, Entry, create)

    def File(self, name, directory=None, create=1):
        if False:
            return 10
        'Look up or create a File node with the specified name.  If\n        the name is a relative path (begins with ./, ../, or a file name),\n        then it is looked up relative to the supplied directory node,\n        or to the top level directory of the FS (supplied at construction\n        time) if no directory is supplied.\n\n        This method will raise TypeError if a directory is found at the\n        specified path.\n        '
        return self._lookup(name, directory, File, create)

    def Dir(self, name, directory=None, create=True):
        if False:
            while True:
                i = 10
        'Look up or create a Dir node with the specified name.  If\n        the name is a relative path (begins with ./, ../, or a file name),\n        then it is looked up relative to the supplied directory node,\n        or to the top level directory of the FS (supplied at construction\n        time) if no directory is supplied.\n\n        This method will raise TypeError if a normal file is found at the\n        specified path.\n        '
        return self._lookup(name, directory, Dir, create)

    def VariantDir(self, variant_dir, src_dir, duplicate=1):
        if False:
            for i in range(10):
                print('nop')
        'Link the supplied variant directory to the source directory\n        for purposes of building files.'
        if not isinstance(src_dir, SCons.Node.Node):
            src_dir = self.Dir(src_dir)
        if not isinstance(variant_dir, SCons.Node.Node):
            variant_dir = self.Dir(variant_dir)
        if src_dir.is_under(variant_dir):
            raise SCons.Errors.UserError('Source directory cannot be under variant directory.')
        if variant_dir.srcdir:
            if variant_dir.srcdir == src_dir:
                return
            raise SCons.Errors.UserError("'%s' already has a source directory: '%s'." % (variant_dir, variant_dir.srcdir))
        variant_dir.link(src_dir, duplicate)

    def Repository(self, *dirs):
        if False:
            print('Hello World!')
        'Specify Repository directories to search.'
        for d in dirs:
            if not isinstance(d, SCons.Node.Node):
                d = self.Dir(d)
            self.Top.addRepository(d)

    def PyPackageDir(self, modulename):
        if False:
            print('Hello World!')
        'Locate the directory of a given python module name\n\n        For example scons might resolve to\n        Windows: C:\\Python27\\Lib\\site-packages\\scons-2.5.1\n        Linux: /usr/lib/scons\n\n        This can be useful when we want to determine a toolpath based on a python module name'
        dirpath = ''
        if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] in (0, 1, 2, 3, 4)):
            import imp
            splitname = modulename.split('.')
            srchpths = sys.path
            for item in splitname:
                (file, path, desc) = imp.find_module(item, srchpths)
                if file is not None:
                    path = os.path.dirname(path)
                srchpths = [path]
            dirpath = path
        else:
            import importlib.util
            modspec = importlib.util.find_spec(modulename)
            dirpath = os.path.dirname(modspec.origin)
        return self._lookup(dirpath, None, Dir, True)

    def variant_dir_target_climb(self, orig, dir, tail):
        if False:
            print('Hello World!')
        "Create targets in corresponding variant directories\n\n        Climb the directory tree, and look up path names\n        relative to any linked variant directories we find.\n\n        Even though this loops and walks up the tree, we don't memoize\n        the return value because this is really only used to process\n        the command-line targets.\n        "
        targets = []
        message = None
        fmt = 'building associated VariantDir targets: %s'
        start_dir = dir
        while dir:
            for bd in dir.variant_dirs:
                if start_dir.is_under(bd):
                    return ([orig], fmt % str(orig))
                p = os.path.join(bd._path, *tail)
                targets.append(self.Entry(p))
            tail = [dir.name] + tail
            dir = dir.up()
        if targets:
            message = fmt % ' '.join(map(str, targets))
        return (targets, message)

    def Glob(self, pathname, ondisk=True, source=True, strings=False, exclude=None, cwd=None):
        if False:
            i = 10
            return i + 15
        '\n        Globs\n\n        This is mainly a shim layer\n        '
        if cwd is None:
            cwd = self.getcwd()
        return cwd.glob(pathname, ondisk, source, strings, exclude)

class DirNodeInfo(SCons.Node.NodeInfoBase):
    __slots__ = ()
    current_version_id = 2
    fs = None

    def str_to_node(self, s):
        if False:
            print('Hello World!')
        top = self.fs.Top
        root = top.root
        if do_splitdrive:
            (drive, s) = _my_splitdrive(s)
            if drive:
                root = self.fs.get_root(drive)
        if not os.path.isabs(s):
            s = top.get_labspath() + '/' + s
        return root._lookup_abs(s, Entry)

class DirBuildInfo(SCons.Node.BuildInfoBase):
    __slots__ = ()
    current_version_id = 2
glob_magic_check = re.compile('[*?[]')

def has_glob_magic(s):
    if False:
        for i in range(10):
            print('nop')
    return glob_magic_check.search(s) is not None

class Dir(Base):
    """A class for directories in a file system.
    """
    __slots__ = ['scanner_paths', 'cachedir_csig', 'cachesig', 'repositories', 'srcdir', 'entries', 'searched', '_sconsign', 'variant_dirs', 'root', 'dirname', 'on_disk_entries', 'released_target_info', 'contentsig']
    NodeInfo = DirNodeInfo
    BuildInfo = DirBuildInfo

    def __init__(self, name, directory, fs):
        if False:
            print('Hello World!')
        if SCons.Debug.track_instances:
            logInstanceCreation(self, 'Node.FS.Dir')
        Base.__init__(self, name, directory, fs)
        self._morph()

    def _morph(self):
        if False:
            print('Hello World!')
        "Turn a file system Node (either a freshly initialized directory\n        object or a separate Entry object) into a proper directory object.\n\n        Set up this directory's entries and hook it into the file\n        system tree.  Specify that directories (this Node) don't use\n        signatures for calculating whether they're current.\n        "
        self.repositories = []
        self.srcdir = None
        self.entries = {}
        self.entries['.'] = self
        self.entries['..'] = self.dir
        self.cwd = self
        self.searched = 0
        self._sconsign = None
        self.variant_dirs = []
        self.root = self.dir.root
        self.changed_since_last_build = 3
        self._func_sconsign = 1
        self._func_exists = 2
        self._func_get_contents = 2
        self._abspath = SCons.Util.silent_intern(self.dir.entry_abspath(self.name))
        self._labspath = SCons.Util.silent_intern(self.dir.entry_labspath(self.name))
        if self.dir._path == '.':
            self._path = SCons.Util.silent_intern(self.name)
        else:
            self._path = SCons.Util.silent_intern(self.dir.entry_path(self.name))
        if self.dir._tpath == '.':
            self._tpath = SCons.Util.silent_intern(self.name)
        else:
            self._tpath = SCons.Util.silent_intern(self.dir.entry_tpath(self.name))
        self._path_elements = self.dir._path_elements + [self]
        self.dirname = self.name + OS_SEP
        if not hasattr(self, 'executor'):
            self.builder = get_MkdirBuilder()
            self.get_executor().set_action_list(self.builder.action)
        else:
            l = self.get_executor().action_list
            a = get_MkdirBuilder().action
            l.insert(0, a)
            self.get_executor().set_action_list(l)

    def diskcheck_match(self):
        if False:
            i = 10
            return i + 15
        if os.name == 'nt' and str is bytes:
            return
        diskcheck_match(self, self.isfile, 'File %s found where directory expected.')

    def __clearRepositoryCache(self, duplicate=None):
        if False:
            print('Hello World!')
        'Called when we change the repository(ies) for a directory.\n        This clears any cached information that is invalidated by changing\n        the repository.'
        for node in list(self.entries.values()):
            if node != self.dir:
                if node != self and isinstance(node, Dir):
                    node.__clearRepositoryCache(duplicate)
                else:
                    node.clear()
                    try:
                        del node._srcreps
                    except AttributeError:
                        pass
                    if duplicate is not None:
                        node.duplicate = duplicate

    def __resetDuplicate(self, node):
        if False:
            i = 10
            return i + 15
        if node != self:
            node.duplicate = node.get_dir().duplicate

    def Entry(self, name):
        if False:
            while True:
                i = 10
        "\n        Looks up or creates an entry node named 'name' relative to\n        this directory.\n        "
        return self.fs.Entry(name, self)

    def Dir(self, name, create=True):
        if False:
            i = 10
            return i + 15
        "\n        Looks up or creates a directory node named 'name' relative to\n        this directory.\n        "
        return self.fs.Dir(name, self, create)

    def File(self, name):
        if False:
            return 10
        "\n        Looks up or creates a file node named 'name' relative to\n        this directory.\n        "
        return self.fs.File(name, self)

    def link(self, srcdir, duplicate):
        if False:
            while True:
                i = 10
        'Set this directory as the variant directory for the\n        supplied source directory.'
        self.srcdir = srcdir
        self.duplicate = duplicate
        self.__clearRepositoryCache(duplicate)
        srcdir.variant_dirs.append(self)

    def getRepositories(self):
        if False:
            while True:
                i = 10
        'Returns a list of repositories for this directory.\n        '
        if self.srcdir and (not self.duplicate):
            return self.srcdir.get_all_rdirs() + self.repositories
        return self.repositories

    @SCons.Memoize.CountMethodCall
    def get_all_rdirs(self):
        if False:
            i = 10
            return i + 15
        try:
            return list(self._memo['get_all_rdirs'])
        except KeyError:
            pass
        result = [self]
        fname = '.'
        dir = self
        while dir:
            for rep in dir.getRepositories():
                result.append(rep.Dir(fname))
            if fname == '.':
                fname = dir.name
            else:
                fname = dir.name + OS_SEP + fname
            dir = dir.up()
        self._memo['get_all_rdirs'] = list(result)
        return result

    def addRepository(self, dir):
        if False:
            print('Hello World!')
        if dir != self and dir not in self.repositories:
            self.repositories.append(dir)
            dir._tpath = '.'
            self.__clearRepositoryCache()

    def up(self):
        if False:
            i = 10
            return i + 15
        return self.dir

    def _rel_path_key(self, other):
        if False:
            for i in range(10):
                print('nop')
        return str(other)

    @SCons.Memoize.CountDictCall(_rel_path_key)
    def rel_path(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Return a path to "other" relative to this directory.\n        '
        try:
            memo_dict = self._memo['rel_path']
        except KeyError:
            memo_dict = {}
            self._memo['rel_path'] = memo_dict
        else:
            try:
                return memo_dict[other]
            except KeyError:
                pass
        if self is other:
            result = '.'
        elif other not in self._path_elements:
            try:
                other_dir = other.get_dir()
            except AttributeError:
                result = str(other)
            else:
                if other_dir is None:
                    result = other.name
                else:
                    dir_rel_path = self.rel_path(other_dir)
                    if dir_rel_path == '.':
                        result = other.name
                    else:
                        result = dir_rel_path + OS_SEP + other.name
        else:
            i = self._path_elements.index(other) + 1
            path_elems = ['..'] * (len(self._path_elements) - i) + [n.name for n in other._path_elements[i:]]
            result = OS_SEP.join(path_elems)
        memo_dict[other] = result
        return result

    def get_env_scanner(self, env, kw={}):
        if False:
            while True:
                i = 10
        import SCons.Defaults
        return SCons.Defaults.DirEntryScanner

    def get_target_scanner(self):
        if False:
            print('Hello World!')
        import SCons.Defaults
        return SCons.Defaults.DirEntryScanner

    def get_found_includes(self, env, scanner, path):
        if False:
            for i in range(10):
                print('nop')
        "Return this directory's implicit dependencies.\n\n        We don't bother caching the results because the scan typically\n        shouldn't be requested more than once (as opposed to scanning\n        .h file contents, which can be requested as many times as the\n        files is #included by other files).\n        "
        if not scanner:
            return []
        self.clear()
        return scanner(self, env, path)

    def prepare(self):
        if False:
            return 10
        pass

    def build(self, **kw):
        if False:
            return 10
        'A null "builder" for directories.'
        global MkdirBuilder
        if self.builder is not MkdirBuilder:
            SCons.Node.Node.build(self, **kw)

    def _create(self):
        if False:
            i = 10
            return i + 15
        'Create this directory, silently and without worrying about\n        whether the builder is the default or not.'
        listDirs = []
        parent = self
        while parent:
            if parent.exists():
                break
            listDirs.append(parent)
            p = parent.up()
            if p is None:
                raise SCons.Errors.StopError(parent._path)
            parent = p
        listDirs.reverse()
        for dirnode in listDirs:
            try:
                SCons.Node.Node.build(dirnode)
                dirnode.get_executor().nullify()
                dirnode.clear()
            except OSError:
                pass

    def multiple_side_effect_has_builder(self):
        if False:
            print('Hello World!')
        global MkdirBuilder
        return self.builder is not MkdirBuilder and self.has_builder()

    def alter_targets(self):
        if False:
            while True:
                i = 10
        'Return any corresponding targets in a variant directory.\n        '
        return self.fs.variant_dir_target_climb(self, self, [])

    def scanner_key(self):
        if False:
            while True:
                i = 10
        'A directory does not get scanned.'
        return None

    def get_text_contents(self):
        if False:
            print('Hello World!')
        'We already emit things in text, so just return the binary\n        version.'
        return self.get_contents()

    def get_contents(self):
        if False:
            print('Hello World!')
        'Return content signatures and names of all our children\n        separated by new-lines. Ensure that the nodes are sorted.'
        return SCons.Node._get_contents_map[self._func_get_contents](self)

    def get_csig(self):
        if False:
            for i in range(10):
                print('nop')
        'Compute the content signature for Directory nodes. In\n        general, this is not needed and the content signature is not\n        stored in the DirNodeInfo. However, if get_contents on a Dir\n        node is called which has a child directory, the child\n        directory should return the hash of its contents.'
        contents = self.get_contents()
        return SCons.Util.MD5signature(contents)

    def do_duplicate(self, src):
        if False:
            print('Hello World!')
        pass

    def is_up_to_date(self):
        if False:
            return 10
        "If any child is not up-to-date, then this directory isn't,\n        either."
        if self.builder is not MkdirBuilder and (not self.exists()):
            return 0
        up_to_date = SCons.Node.up_to_date
        for kid in self.children():
            if kid.get_state() > up_to_date:
                return 0
        return 1

    def rdir(self):
        if False:
            i = 10
            return i + 15
        if not self.exists():
            norm_name = _my_normcase(self.name)
            for dir in self.dir.get_all_rdirs():
                try:
                    node = dir.entries[norm_name]
                except KeyError:
                    node = dir.dir_on_disk(self.name)
                if node and node.exists() and (isinstance(dir, Dir) or isinstance(dir, Entry)):
                    return node
        return self

    def sconsign(self):
        if False:
            return 10
        'Return the .sconsign file info for this directory. '
        return _sconsign_map[self._func_sconsign](self)

    def srcnode(self):
        if False:
            i = 10
            return i + 15
        'Dir has a special need for srcnode()...if we\n        have a srcdir attribute set, then that *is* our srcnode.'
        if self.srcdir:
            return self.srcdir
        return Base.srcnode(self)

    def get_timestamp(self):
        if False:
            i = 10
            return i + 15
        'Return the latest timestamp from among our children'
        stamp = 0
        for kid in self.children():
            if kid.get_timestamp() > stamp:
                stamp = kid.get_timestamp()
        return stamp

    def get_abspath(self):
        if False:
            i = 10
            return i + 15
        'Get the absolute path of the file.'
        return self._abspath

    def get_labspath(self):
        if False:
            print('Hello World!')
        'Get the absolute path of the file.'
        return self._labspath

    def get_internal_path(self):
        if False:
            return 10
        return self._path

    def get_tpath(self):
        if False:
            return 10
        return self._tpath

    def get_path_elements(self):
        if False:
            while True:
                i = 10
        return self._path_elements

    def entry_abspath(self, name):
        if False:
            i = 10
            return i + 15
        return self._abspath + OS_SEP + name

    def entry_labspath(self, name):
        if False:
            return 10
        return self._labspath + '/' + name

    def entry_path(self, name):
        if False:
            print('Hello World!')
        return self._path + OS_SEP + name

    def entry_tpath(self, name):
        if False:
            return 10
        return self._tpath + OS_SEP + name

    def entry_exists_on_disk(self, name):
        if False:
            while True:
                i = 10
        ' Searches through the file/dir entries of the current\n            directory, and returns True if a physical entry with the given\n            name could be found.\n\n            @see rentry_exists_on_disk\n        '
        try:
            d = self.on_disk_entries
        except AttributeError:
            d = {}
            try:
                entries = os.listdir(self._abspath)
            except OSError:
                pass
            else:
                for entry in map(_my_normcase, entries):
                    d[entry] = True
            self.on_disk_entries = d
        if sys.platform == 'win32' or sys.platform == 'cygwin':
            name = _my_normcase(name)
            result = d.get(name)
            if result is None:
                result = os.path.exists(self._abspath + OS_SEP + name)
                d[name] = result
            return result
        else:
            return name in d

    def rentry_exists_on_disk(self, name):
        if False:
            while True:
                i = 10
        ' Searches through the file/dir entries of the current\n            *and* all its remote directories (repos), and returns\n            True if a physical entry with the given name could be found.\n            The local directory (self) gets searched first, so\n            repositories take a lower precedence regarding the\n            searching order.\n\n            @see entry_exists_on_disk\n        '
        rentry_exists = self.entry_exists_on_disk(name)
        if not rentry_exists:
            norm_name = _my_normcase(name)
            for rdir in self.get_all_rdirs():
                try:
                    node = rdir.entries[norm_name]
                    if node:
                        rentry_exists = True
                        break
                except KeyError:
                    if rdir.entry_exists_on_disk(name):
                        rentry_exists = True
                        break
        return rentry_exists

    @SCons.Memoize.CountMethodCall
    def srcdir_list(self):
        if False:
            i = 10
            return i + 15
        try:
            return self._memo['srcdir_list']
        except KeyError:
            pass
        result = []
        dirname = '.'
        dir = self
        while dir:
            if dir.srcdir:
                result.append(dir.srcdir.Dir(dirname))
            dirname = dir.name + OS_SEP + dirname
            dir = dir.up()
        self._memo['srcdir_list'] = result
        return result

    def srcdir_duplicate(self, name):
        if False:
            return 10
        for dir in self.srcdir_list():
            if self.is_under(dir):
                break
            if dir.entry_exists_on_disk(name):
                srcnode = dir.Entry(name).disambiguate()
                if self.duplicate:
                    node = self.Entry(name).disambiguate()
                    node.do_duplicate(srcnode)
                    return node
                else:
                    return srcnode
        return None

    def _srcdir_find_file_key(self, filename):
        if False:
            i = 10
            return i + 15
        return filename

    @SCons.Memoize.CountDictCall(_srcdir_find_file_key)
    def srcdir_find_file(self, filename):
        if False:
            for i in range(10):
                print('nop')
        try:
            memo_dict = self._memo['srcdir_find_file']
        except KeyError:
            memo_dict = {}
            self._memo['srcdir_find_file'] = memo_dict
        else:
            try:
                return memo_dict[filename]
            except KeyError:
                pass

        def func(node):
            if False:
                return 10
            if (isinstance(node, File) or isinstance(node, Entry)) and (node.is_derived() or node.exists()):
                return node
            return None
        norm_name = _my_normcase(filename)
        for rdir in self.get_all_rdirs():
            try:
                node = rdir.entries[norm_name]
            except KeyError:
                node = rdir.file_on_disk(filename)
            else:
                node = func(node)
            if node:
                result = (node, self)
                memo_dict[filename] = result
                return result
        for srcdir in self.srcdir_list():
            for rdir in srcdir.get_all_rdirs():
                try:
                    node = rdir.entries[norm_name]
                except KeyError:
                    node = rdir.file_on_disk(filename)
                else:
                    node = func(node)
                if node:
                    result = (File(filename, self, self.fs), srcdir)
                    memo_dict[filename] = result
                    return result
        result = (None, None)
        memo_dict[filename] = result
        return result

    def dir_on_disk(self, name):
        if False:
            i = 10
            return i + 15
        if self.entry_exists_on_disk(name):
            try:
                return self.Dir(name)
            except TypeError:
                pass
        node = self.srcdir_duplicate(name)
        if isinstance(node, File):
            return None
        return node

    def file_on_disk(self, name):
        if False:
            i = 10
            return i + 15
        if self.entry_exists_on_disk(name):
            try:
                return self.File(name)
            except TypeError:
                pass
        node = self.srcdir_duplicate(name)
        if isinstance(node, Dir):
            return None
        return node

    def walk(self, func, arg):
        if False:
            i = 10
            return i + 15
        '\n        Walk this directory tree by calling the specified function\n        for each directory in the tree.\n\n        This behaves like the os.path.walk() function, but for in-memory\n        Node.FS.Dir objects.  The function takes the same arguments as\n        the functions passed to os.path.walk():\n\n                func(arg, dirname, fnames)\n\n        Except that "dirname" will actually be the directory *Node*,\n        not the string.  The \'.\' and \'..\' entries are excluded from\n        fnames.  The fnames list may be modified in-place to filter the\n        subdirectories visited or otherwise impose a specific order.\n        The "arg" argument is always passed to func() and may be used\n        in any way (or ignored, passing None is common).\n        '
        entries = self.entries
        names = list(entries.keys())
        names.remove('.')
        names.remove('..')
        func(arg, self, names)
        for dirname in [n for n in names if isinstance(entries[n], Dir)]:
            entries[dirname].walk(func, arg)

    def glob(self, pathname, ondisk=True, source=False, strings=False, exclude=None):
        if False:
            while True:
                i = 10
        '\n        Returns a list of Nodes (or strings) matching a specified\n        pathname pattern.\n\n        Pathname patterns follow UNIX shell semantics:  * matches\n        any-length strings of any characters, ? matches any character,\n        and [] can enclose lists or ranges of characters.  Matches do\n        not span directory separators.\n\n        The matches take into account Repositories, returning local\n        Nodes if a corresponding entry exists in a Repository (either\n        an in-memory Node or something on disk).\n\n        By defafult, the glob() function matches entries that exist\n        on-disk, in addition to in-memory Nodes.  Setting the "ondisk"\n        argument to False (or some other non-true value) causes the glob()\n        function to only match in-memory Nodes.  The default behavior is\n        to return both the on-disk and in-memory Nodes.\n\n        The "source" argument, when true, specifies that corresponding\n        source Nodes must be returned if you\'re globbing in a build\n        directory (initialized with VariantDir()).  The default behavior\n        is to return Nodes local to the VariantDir().\n\n        The "strings" argument, when true, returns the matches as strings,\n        not Nodes.  The strings are path names relative to this directory.\n\n        The "exclude" argument, if not None, must be a pattern or a list\n        of patterns following the same UNIX shell semantics.\n        Elements matching a least one pattern of this list will be excluded\n        from the result.\n\n        The underlying algorithm is adapted from the glob.glob() function\n        in the Python library (but heavily modified), and uses fnmatch()\n        under the covers.\n        '
        (dirname, basename) = os.path.split(pathname)
        if not dirname:
            result = self._glob1(basename, ondisk, source, strings)
        else:
            if has_glob_magic(dirname):
                list = self.glob(dirname, ondisk, source, False, exclude)
            else:
                list = [self.Dir(dirname, create=True)]
            result = []
            for dir in list:
                r = dir._glob1(basename, ondisk, source, strings)
                if strings:
                    r = [os.path.join(str(dir), x) for x in r]
                result.extend(r)
        if exclude:
            excludes = []
            excludeList = SCons.Util.flatten(exclude)
            for x in excludeList:
                r = self.glob(x, ondisk, source, strings)
                excludes.extend(r)
            result = [x for x in result if not any((fnmatch.fnmatch(str(x), str(e)) for e in SCons.Util.flatten(excludes)))]
        return sorted(result, key=lambda a: str(a))

    def _glob1(self, pattern, ondisk=True, source=False, strings=False):
        if False:
            print('Hello World!')
        '\n        Globs for and returns a list of entry names matching a single\n        pattern in this directory.\n\n        This searches any repositories and source directories for\n        corresponding entries and returns a Node (or string) relative\n        to the current directory if an entry is found anywhere.\n\n        TODO: handle pattern with no wildcard\n        '
        search_dir_list = self.get_all_rdirs()
        for srcdir in self.srcdir_list():
            search_dir_list.extend(srcdir.get_all_rdirs())
        selfEntry = self.Entry
        names = []
        for dir in search_dir_list:
            node_names = [v.name for (k, v) in dir.entries.items() if k not in ('.', '..')]
            names.extend(node_names)
            if not strings:
                for name in node_names:
                    selfEntry(name)
            if ondisk:
                try:
                    disk_names = os.listdir(dir._abspath)
                except os.error:
                    continue
                names.extend(disk_names)
                if not strings:
                    if pattern[0] != '.':
                        disk_names = [x for x in disk_names if x[0] != '.']
                    disk_names = fnmatch.filter(disk_names, pattern)
                    dirEntry = dir.Entry
                    for name in disk_names:
                        name = './' + name
                        node = dirEntry(name).disambiguate()
                        n = selfEntry(name)
                        if n.__class__ != node.__class__:
                            n.__class__ = node.__class__
                            n._morph()
        names = set(names)
        if pattern[0] != '.':
            names = [x for x in names if x[0] != '.']
        names = fnmatch.filter(names, pattern)
        if strings:
            return names
        return [self.entries[_my_normcase(n)] for n in names]

class RootDir(Dir):
    """A class for the root directory of a file system.

    This is the same as a Dir class, except that the path separator
    ('/' or '\\') is actually part of the name, so we don't need to
    add a separator when creating the path names of entries within
    this directory.
    """
    __slots__ = ('_lookupDict',)

    def __init__(self, drive, fs):
        if False:
            while True:
                i = 10
        if SCons.Debug.track_instances:
            logInstanceCreation(self, 'Node.FS.RootDir')
        SCons.Node.Node.__init__(self)
        if drive == '':
            name = OS_SEP
            dirname = OS_SEP
        elif drive == '//':
            name = UNC_PREFIX
            dirname = UNC_PREFIX
        else:
            name = drive
            dirname = drive + OS_SEP
        self.name = SCons.Util.silent_intern(name)
        self.fs = fs
        self._path_elements = [self]
        self.dir = self
        self._func_rexists = 2
        self._func_target_from_source = 1
        self.store_info = 1
        self._abspath = dirname
        self._labspath = ''
        self._path = dirname
        self._tpath = dirname
        self.dirname = dirname
        self._morph()
        self.duplicate = 0
        self._lookupDict = {}
        self._lookupDict[''] = self
        self._lookupDict['/'] = self
        self.root = self
        if not has_unc:
            self._lookupDict['//'] = self

    def _morph(self):
        if False:
            while True:
                i = 10
        "Turn a file system Node (either a freshly initialized directory\n        object or a separate Entry object) into a proper directory object.\n\n        Set up this directory's entries and hook it into the file\n        system tree.  Specify that directories (this Node) don't use\n        signatures for calculating whether they're current.\n        "
        self.repositories = []
        self.srcdir = None
        self.entries = {}
        self.entries['.'] = self
        self.entries['..'] = self.dir
        self.cwd = self
        self.searched = 0
        self._sconsign = None
        self.variant_dirs = []
        self.changed_since_last_build = 3
        self._func_sconsign = 1
        self._func_exists = 2
        self._func_get_contents = 2
        if not hasattr(self, 'executor'):
            self.builder = get_MkdirBuilder()
            self.get_executor().set_action_list(self.builder.action)
        else:
            l = self.get_executor().action_list
            a = get_MkdirBuilder().action
            l.insert(0, a)
            self.get_executor().set_action_list(l)

    def must_be_same(self, klass):
        if False:
            i = 10
            return i + 15
        if klass is Dir:
            return
        Base.must_be_same(self, klass)

    def _lookup_abs(self, p, klass, create=1):
        if False:
            print('Hello World!')
        '\n        Fast (?) lookup of a *normalized* absolute path.\n\n        This method is intended for use by internal lookups with\n        already-normalized path data.  For general-purpose lookups,\n        use the FS.Entry(), FS.Dir() or FS.File() methods.\n\n        The caller is responsible for making sure we\'re passed a\n        normalized absolute path; we merely let Python\'s dictionary look\n        up and return the One True Node.FS object for the path.\n\n        If a Node for the specified "p" doesn\'t already exist, and\n        "create" is specified, the Node may be created after recursive\n        invocation to find or create the parent directory or directories.\n        '
        k = _my_normcase(p)
        try:
            result = self._lookupDict[k]
        except KeyError:
            if not create:
                msg = "No such file or directory: '%s' in '%s' (and create is False)" % (p, str(self))
                raise SCons.Errors.UserError(msg)
            (dir_name, file_name) = p.rsplit('/', 1)
            dir_node = self._lookup_abs(dir_name, Dir)
            result = klass(file_name, dir_node, self.fs)
            result.diskcheck_match()
            self._lookupDict[k] = result
            dir_node.entries[_my_normcase(file_name)] = result
            dir_node.implicit = None
        else:
            result.must_be_same(klass)
        return result

    def __str__(self):
        if False:
            return 10
        return self._abspath

    def entry_abspath(self, name):
        if False:
            return 10
        return self._abspath + name

    def entry_labspath(self, name):
        if False:
            print('Hello World!')
        return '/' + name

    def entry_path(self, name):
        if False:
            i = 10
            return i + 15
        return self._path + name

    def entry_tpath(self, name):
        if False:
            while True:
                i = 10
        return self._tpath + name

    def is_under(self, dir):
        if False:
            print('Hello World!')
        if self is dir:
            return 1
        else:
            return 0

    def up(self):
        if False:
            for i in range(10):
                print('nop')
        return None

    def get_dir(self):
        if False:
            return 10
        return None

    def src_builder(self):
        if False:
            for i in range(10):
                print('nop')
        return _null

class FileNodeInfo(SCons.Node.NodeInfoBase):
    __slots__ = ('csig', 'timestamp', 'size')
    current_version_id = 2
    field_list = ['csig', 'timestamp', 'size']
    fs = None

    def str_to_node(self, s):
        if False:
            while True:
                i = 10
        top = self.fs.Top
        root = top.root
        if do_splitdrive:
            (drive, s) = _my_splitdrive(s)
            if drive:
                root = self.fs.get_root(drive)
        if not os.path.isabs(s):
            s = top.get_labspath() + '/' + s
        return root._lookup_abs(s, Entry)

    def __getstate__(self):
        if False:
            print('Hello World!')
        "\n        Return all fields that shall be pickled. Walk the slots in the class\n        hierarchy and add those to the state dictionary. If a '__dict__' slot is\n        available, copy all entries to the dictionary. Also include the version\n        id, which is fixed for all instances of a class.\n        "
        state = getattr(self, '__dict__', {}).copy()
        for obj in type(self).mro():
            for name in getattr(obj, '__slots__', ()):
                if hasattr(self, name):
                    state[name] = getattr(self, name)
        state['_version_id'] = self.current_version_id
        try:
            del state['__weakref__']
        except KeyError:
            pass
        return state

    def __setstate__(self, state):
        if False:
            i = 10
            return i + 15
        '\n        Restore the attributes from a pickled state.\n        '
        del state['_version_id']
        for (key, value) in state.items():
            if key not in ('__weakref__',):
                setattr(self, key, value)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return self.csig == other.csig and self.timestamp == other.timestamp and (self.size == other.size)

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self.__eq__(other)

class FileBuildInfo(SCons.Node.BuildInfoBase):
    """
    This is info loaded from sconsign.

    Attributes unique to FileBuildInfo:
        dependency_map : Caches file->csig mapping
                    for all dependencies.  Currently this is only used when using
                    MD5-timestamp decider.
                    It's used to ensure that we copy the correct
                    csig from previous build to be written to .sconsign when current build
                    is done. Previously the matching of csig to file was strictly by order
                    they appeared in bdepends, bsources, or bimplicit, and so a change in order
                    or count of any of these could yield writing wrong csig, and then false positive
                    rebuilds
    """
    __slots__ = ['dependency_map']
    current_version_id = 2

    def __setattr__(self, key, value):
        if False:
            while True:
                i = 10
        if key != 'dependency_map' and hasattr(self, 'dependency_map'):
            del self.dependency_map
        return super(FileBuildInfo, self).__setattr__(key, value)

    def convert_to_sconsign(self):
        if False:
            print('Hello World!')
        "\n        Converts this FileBuildInfo object for writing to a .sconsign file\n\n        This replaces each Node in our various dependency lists with its\n        usual string representation: relative to the top-level SConstruct\n        directory, or an absolute path if it's outside.\n        "
        if os_sep_is_slash:
            node_to_str = str
        else:

            def node_to_str(n):
                if False:
                    i = 10
                    return i + 15
                try:
                    s = n.get_internal_path()
                except AttributeError:
                    s = str(n)
                else:
                    s = s.replace(OS_SEP, '/')
                return s
        for attr in ['bsources', 'bdepends', 'bimplicit']:
            try:
                val = getattr(self, attr)
            except AttributeError:
                pass
            else:
                setattr(self, attr, list(map(node_to_str, val)))

    def convert_from_sconsign(self, dir, name):
        if False:
            for i in range(10):
                print('nop')
        "\n        Converts a newly-read FileBuildInfo object for in-SCons use\n\n        For normal up-to-date checking, we don't have any conversion to\n        perform--but we're leaving this method here to make that clear.\n        "
        pass

    def prepare_dependencies(self):
        if False:
            while True:
                i = 10
        '\n        Prepares a FileBuildInfo object for explaining what changed\n\n        The bsources, bdepends and bimplicit lists have all been\n        stored on disk as paths relative to the top-level SConstruct\n        directory.  Convert the strings to actual Nodes (for use by the\n        --debug=explain code and --implicit-cache).\n        '
        attrs = [('bsources', 'bsourcesigs'), ('bdepends', 'bdependsigs'), ('bimplicit', 'bimplicitsigs')]
        for (nattr, sattr) in attrs:
            try:
                strings = getattr(self, nattr)
                nodeinfos = getattr(self, sattr)
            except AttributeError:
                continue
            if strings is None or nodeinfos is None:
                continue
            nodes = []
            for (s, ni) in zip(strings, nodeinfos):
                if not isinstance(s, SCons.Node.Node):
                    s = ni.str_to_node(s)
                nodes.append(s)
            setattr(self, nattr, nodes)

    def format(self, names=0):
        if False:
            for i in range(10):
                print('nop')
        result = []
        bkids = self.bsources + self.bdepends + self.bimplicit
        bkidsigs = self.bsourcesigs + self.bdependsigs + self.bimplicitsigs
        for (bkid, bkidsig) in zip(bkids, bkidsigs):
            result.append(str(bkid) + ': ' + ' '.join(bkidsig.format(names=names)))
        if not hasattr(self, 'bact'):
            self.bact = 'none'
        result.append('%s [%s]' % (self.bactsig, self.bact))
        return '\n'.join(result)

class File(Base):
    """A class for files in a file system.
    """
    __slots__ = ['scanner_paths', 'cachedir_csig', 'cachesig', 'repositories', 'srcdir', 'entries', 'searched', '_sconsign', 'variant_dirs', 'root', 'dirname', 'on_disk_entries', 'released_target_info', 'contentsig']
    NodeInfo = FileNodeInfo
    BuildInfo = FileBuildInfo
    md5_chunksize = 64

    def diskcheck_match(self):
        if False:
            while True:
                i = 10
        diskcheck_match(self, self.isdir, 'Directory %s found where file expected.')

    def __init__(self, name, directory, fs):
        if False:
            while True:
                i = 10
        if SCons.Debug.track_instances:
            logInstanceCreation(self, 'Node.FS.File')
        Base.__init__(self, name, directory, fs)
        self._morph()

    def Entry(self, name):
        if False:
            while True:
                i = 10
        "Create an entry node named 'name' relative to\n        the directory of this file."
        return self.dir.Entry(name)

    def Dir(self, name, create=True):
        if False:
            for i in range(10):
                print('nop')
        "Create a directory node named 'name' relative to\n        the directory of this file."
        return self.dir.Dir(name, create=create)

    def Dirs(self, pathlist):
        if False:
            return 10
        'Create a list of directories relative to the SConscript\n        directory of this file.'
        return [self.Dir(p) for p in pathlist]

    def File(self, name):
        if False:
            return 10
        "Create a file node named 'name' relative to\n        the directory of this file."
        return self.dir.File(name)

    def _morph(self):
        if False:
            print('Hello World!')
        'Turn a file system node into a File object.'
        self.scanner_paths = {}
        if not hasattr(self, '_local'):
            self._local = 0
        if not hasattr(self, 'released_target_info'):
            self.released_target_info = False
        self.store_info = 1
        self._func_exists = 4
        self._func_get_contents = 3
        self.changed_since_last_build = 4
        if self.has_builder():
            self.changed_since_last_build = 5

    def scanner_key(self):
        if False:
            return 10
        return self.get_suffix()

    def get_contents(self):
        if False:
            return 10
        return SCons.Node._get_contents_map[self._func_get_contents](self)

    def get_text_contents(self):
        if False:
            return 10
        "\n        This attempts to figure out what the encoding of the text is\n        based upon the BOM bytes, and then decodes the contents so that\n        it's a valid python string.\n        "
        contents = self.get_contents()
        if contents[:len(codecs.BOM_UTF8)] == codecs.BOM_UTF8:
            return contents[len(codecs.BOM_UTF8):].decode('utf-8')
        if contents[:len(codecs.BOM_UTF16_LE)] == codecs.BOM_UTF16_LE:
            return contents[len(codecs.BOM_UTF16_LE):].decode('utf-16-le')
        if contents[:len(codecs.BOM_UTF16_BE)] == codecs.BOM_UTF16_BE:
            return contents[len(codecs.BOM_UTF16_BE):].decode('utf-16-be')
        try:
            return contents.decode('utf-8')
        except UnicodeDecodeError as e:
            try:
                return contents.decode('latin-1')
            except UnicodeDecodeError as e:
                return contents.decode('utf-8', error='backslashreplace')

    def get_content_hash(self):
        if False:
            return 10
        '\n        Compute and return the MD5 hash for this file.\n        '
        if not self.rexists():
            return SCons.Util.MD5signature('')
        fname = self.rfile().get_abspath()
        try:
            cs = SCons.Util.MD5filesignature(fname, chunksize=SCons.Node.FS.File.md5_chunksize * 1024)
        except EnvironmentError as e:
            if not e.filename:
                e.filename = fname
            raise
        return cs

    @SCons.Memoize.CountMethodCall
    def get_size(self):
        if False:
            return 10
        try:
            return self._memo['get_size']
        except KeyError:
            pass
        if self.rexists():
            size = self.rfile().getsize()
        else:
            size = 0
        self._memo['get_size'] = size
        return size

    @SCons.Memoize.CountMethodCall
    def get_timestamp(self):
        if False:
            i = 10
            return i + 15
        try:
            return self._memo['get_timestamp']
        except KeyError:
            pass
        if self.rexists():
            timestamp = self.rfile().getmtime()
        else:
            timestamp = 0
        self._memo['get_timestamp'] = timestamp
        return timestamp
    convert_copy_attrs = ['bsources', 'bimplicit', 'bdepends', 'bact', 'bactsig', 'ninfo']
    convert_sig_attrs = ['bsourcesigs', 'bimplicitsigs', 'bdependsigs']

    def convert_old_entry(self, old_entry):
        if False:
            print('Hello World!')
        import SCons.SConsign
        new_entry = SCons.SConsign.SConsignEntry()
        new_entry.binfo = self.new_binfo()
        binfo = new_entry.binfo
        for attr in self.convert_copy_attrs:
            try:
                value = getattr(old_entry, attr)
            except AttributeError:
                continue
            setattr(binfo, attr, value)
            delattr(old_entry, attr)
        for attr in self.convert_sig_attrs:
            try:
                sig_list = getattr(old_entry, attr)
            except AttributeError:
                continue
            value = []
            for sig in sig_list:
                ninfo = self.new_ninfo()
                if len(sig) == 32:
                    ninfo.csig = sig
                else:
                    ninfo.timestamp = sig
                value.append(ninfo)
            setattr(binfo, attr, value)
            delattr(old_entry, attr)
        return new_entry

    @SCons.Memoize.CountMethodCall
    def get_stored_info(self):
        if False:
            return 10
        try:
            return self._memo['get_stored_info']
        except KeyError:
            pass
        try:
            sconsign_entry = self.dir.sconsign().get_entry(self.name)
        except (KeyError, EnvironmentError):
            import SCons.SConsign
            sconsign_entry = SCons.SConsign.SConsignEntry()
            sconsign_entry.binfo = self.new_binfo()
            sconsign_entry.ninfo = self.new_ninfo()
        else:
            if isinstance(sconsign_entry, FileBuildInfo):
                sconsign_entry = self.convert_old_entry(sconsign_entry)
            try:
                delattr(sconsign_entry.ninfo, 'bsig')
            except AttributeError:
                pass
        self._memo['get_stored_info'] = sconsign_entry
        return sconsign_entry

    def get_stored_implicit(self):
        if False:
            print('Hello World!')
        binfo = self.get_stored_info().binfo
        binfo.prepare_dependencies()
        try:
            return binfo.bimplicit
        except AttributeError:
            return None

    def rel_path(self, other):
        if False:
            print('Hello World!')
        return self.dir.rel_path(other)

    def _get_found_includes_key(self, env, scanner, path):
        if False:
            while True:
                i = 10
        return (id(env), id(scanner), path)

    @SCons.Memoize.CountDictCall(_get_found_includes_key)
    def get_found_includes(self, env, scanner, path):
        if False:
            print('Hello World!')
        'Return the included implicit dependencies in this file.\n        Cache results so we only scan the file once per path\n        regardless of how many times this information is requested.\n        '
        memo_key = (id(env), id(scanner), path)
        try:
            memo_dict = self._memo['get_found_includes']
        except KeyError:
            memo_dict = {}
            self._memo['get_found_includes'] = memo_dict
        else:
            try:
                return memo_dict[memo_key]
            except KeyError:
                pass
        if scanner:
            result = [n.disambiguate() for n in scanner(self, env, path)]
        else:
            result = []
        memo_dict[memo_key] = result
        return result

    def _createDir(self):
        if False:
            for i in range(10):
                print('nop')
        self.dir._create()

    def push_to_cache(self):
        if False:
            while True:
                i = 10
        'Try to push the node into a cache\n        '
        if self.nocache:
            return
        self.clear_memoized_values()
        if self.exists():
            self.get_build_env().get_CacheDir().push(self)

    def retrieve_from_cache(self):
        if False:
            print('Hello World!')
        "Try to retrieve the node's content from a cache\n\n        This method is called from multiple threads in a parallel build,\n        so only do thread safe stuff here. Do thread unsafe stuff in\n        built().\n\n        Returns true if the node was successfully retrieved.\n        "
        if self.nocache:
            return None
        if not self.is_derived():
            return None
        return self.get_build_env().get_CacheDir().retrieve(self)

    def visited(self):
        if False:
            for i in range(10):
                print('nop')
        if self.exists() and self.executor is not None:
            self.get_build_env().get_CacheDir().push_if_forced(self)
        ninfo = self.get_ninfo()
        csig = self.get_max_drift_csig()
        if csig:
            ninfo.csig = csig
        ninfo.timestamp = self.get_timestamp()
        ninfo.size = self.get_size()
        if not self.has_builder():
            old = self.get_stored_info()
            self.get_binfo().merge(old.binfo)
        SCons.Node.store_info_map[self.store_info](self)

    def release_target_info(self):
        if False:
            while True:
                i = 10
        "Called just after this node has been marked\n         up-to-date or was built completely.\n\n         This is where we try to release as many target node infos\n         as possible for clean builds and update runs, in order\n         to minimize the overall memory consumption.\n\n         We'd like to remove a lot more attributes like self.sources\n         and self.sources_set, but they might get used\n         in a next build step. For example, during configuration\n         the source files for a built E{*}.o file are used to figure out\n         which linker to use for the resulting Program (gcc vs. g++)!\n         That's why we check for the 'keep_targetinfo' attribute,\n         config Nodes and the Interactive mode just don't allow\n         an early release of most variables.\n\n         In the same manner, we can't simply remove the self.attributes\n         here. The smart linking relies on the shared flag, and some\n         parts of the java Tool use it to transport information\n         about nodes...\n\n         @see: built() and Node.release_target_info()\n         "
        if self.released_target_info or SCons.Node.interactive:
            return
        if not hasattr(self.attributes, 'keep_targetinfo'):
            self.changed(allowcache=True)
            self.get_contents_sig()
            self.get_build_env()
            self.executor = None
            self._memo.pop('rfile', None)
            self.prerequisites = None
            if not len(self.ignore_set):
                self.ignore_set = None
            if not len(self.implicit_set):
                self.implicit_set = None
            if not len(self.depends_set):
                self.depends_set = None
            if not len(self.ignore):
                self.ignore = None
            if not len(self.depends):
                self.depends = None
            self.released_target_info = True

    def find_src_builder(self):
        if False:
            return 10
        if self.rexists():
            return None
        scb = self.dir.src_builder()
        if scb is _null:
            scb = None
        if scb is not None:
            try:
                b = self.builder
            except AttributeError:
                b = None
            if b is None:
                self.builder_set(scb)
        return scb

    def has_src_builder(self):
        if False:
            return 10
        "Return whether this Node has a source builder or not.\n\n        If this Node doesn't have an explicit source code builder, this\n        is where we figure out, on the fly, if there's a transparent\n        source code builder for it.\n\n        Note that if we found a source builder, we also set the\n        self.builder attribute, so that all of the methods that actually\n        *build* this file don't have to do anything different.\n        "
        try:
            scb = self.sbuilder
        except AttributeError:
            scb = self.sbuilder = self.find_src_builder()
        return scb is not None

    def alter_targets(self):
        if False:
            print('Hello World!')
        'Return any corresponding targets in a variant directory.\n        '
        if self.is_derived():
            return ([], None)
        return self.fs.variant_dir_target_climb(self, self.dir, [self.name])

    def _rmv_existing(self):
        if False:
            print('Hello World!')
        self.clear_memoized_values()
        if SCons.Node.print_duplicate:
            print('dup: removing existing target {}'.format(self))
        e = Unlink(self, [], None)
        if isinstance(e, SCons.Errors.BuildError):
            raise e

    def make_ready(self):
        if False:
            for i in range(10):
                print('nop')
        self.has_src_builder()
        self.get_binfo()

    def prepare(self):
        if False:
            while True:
                i = 10
        'Prepare for this file to be created.'
        SCons.Node.Node.prepare(self)
        if self.get_state() != SCons.Node.up_to_date:
            if self.exists():
                if self.is_derived() and (not self.precious):
                    self._rmv_existing()
            else:
                try:
                    self._createDir()
                except SCons.Errors.StopError as drive:
                    raise SCons.Errors.StopError("No drive `{}' for target `{}'.".format(drive, self))

    def remove(self):
        if False:
            print('Hello World!')
        'Remove this file.'
        if self.exists() or self.islink():
            self.fs.unlink(self.get_internal_path())
            return 1
        return None

    def do_duplicate(self, src):
        if False:
            for i in range(10):
                print('nop')
        self._createDir()
        if SCons.Node.print_duplicate:
            print("dup: relinking variant '{}' from '{}'".format(self, src))
        Unlink(self, None, None)
        e = Link(self, src, None)
        if isinstance(e, SCons.Errors.BuildError):
            raise SCons.Errors.StopError("Cannot duplicate `{}' in `{}': {}.".format(src.get_internal_path(), self.dir._path, e.errstr))
        self.linked = 1
        self.clear()

    @SCons.Memoize.CountMethodCall
    def exists(self):
        if False:
            return 10
        try:
            return self._memo['exists']
        except KeyError:
            pass
        result = SCons.Node._exists_map[self._func_exists](self)
        self._memo['exists'] = result
        return result

    def get_max_drift_csig(self):
        if False:
            while True:
                i = 10
        "\n        Returns the content signature currently stored for this node\n        if it's been unmodified longer than the max_drift value, or the\n        max_drift value is 0.  Returns None otherwise.\n        "
        old = self.get_stored_info()
        mtime = self.get_timestamp()
        max_drift = self.fs.max_drift
        if max_drift > 0:
            if time.time() - mtime > max_drift:
                try:
                    n = old.ninfo
                    if n.timestamp and n.csig and (n.timestamp == mtime):
                        return n.csig
                except AttributeError:
                    pass
        elif max_drift == 0:
            try:
                return old.ninfo.csig
            except AttributeError:
                pass
        return None

    def get_csig(self):
        if False:
            i = 10
            return i + 15
        "\n        Generate a node's content signature, the digested signature\n        of its content.\n\n        node - the node\n        cache - alternate node to use for the signature cache\n        returns - the content signature\n        "
        ninfo = self.get_ninfo()
        try:
            return ninfo.csig
        except AttributeError:
            pass
        csig = self.get_max_drift_csig()
        if csig is None:
            try:
                if self.get_size() < SCons.Node.FS.File.md5_chunksize:
                    contents = self.get_contents()
                else:
                    csig = self.get_content_hash()
            except IOError:
                csig = ''
            else:
                if not csig:
                    csig = SCons.Util.MD5signature(contents)
        ninfo.csig = csig
        return csig

    def builder_set(self, builder):
        if False:
            print('Hello World!')
        SCons.Node.Node.builder_set(self, builder)
        self.changed_since_last_build = 5

    def built(self):
        if False:
            print('Hello World!')
        "Called just after this File node is successfully built.\n\n         Just like for 'release_target_info' we try to release\n         some more target node attributes in order to minimize the\n         overall memory consumption.\n\n         @see: release_target_info\n        "
        SCons.Node.Node.built(self)
        if not SCons.Node.interactive and (not hasattr(self.attributes, 'keep_targetinfo')):
            SCons.Node.store_info_map[self.store_info](self)
            self._specific_sources = False
            self._labspath = None
            self._save_str()
            self.cwd = None
            self.scanner_paths = None

    def changed(self, node=None, allowcache=False):
        if False:
            print('Hello World!')
        '\n        Returns if the node is up-to-date with respect to the BuildInfo\n        stored last time it was built.\n\n        For File nodes this is basically a wrapper around Node.changed(),\n        but we allow the return value to get cached after the reference\n        to the Executor got released in release_target_info().\n\n        @see: Node.changed()\n        '
        if node is None:
            try:
                return self._memo['changed']
            except KeyError:
                pass
        has_changed = SCons.Node.Node.changed(self, node)
        if allowcache:
            self._memo['changed'] = has_changed
        return has_changed

    def changed_content(self, target, prev_ni, repo_node=None):
        if False:
            while True:
                i = 10
        cur_csig = self.get_csig()
        try:
            return cur_csig != prev_ni.csig
        except AttributeError:
            return 1

    def changed_state(self, target, prev_ni, repo_node=None):
        if False:
            return 10
        return self.state != SCons.Node.up_to_date
    __dmap_cache = {}
    __dmap_sig_cache = {}

    def _build_dependency_map(self, binfo):
        if False:
            return 10
        '\n        Build mapping from file -> signature\n\n        Args:\n            self - self\n            binfo - buildinfo from node being considered\n\n        Returns:\n            dictionary of file->signature mappings\n        '
        if len(binfo.bsourcesigs) + len(binfo.bdependsigs) + len(binfo.bimplicitsigs) == 0:
            return {}
        binfo.dependency_map = {child: signature for (child, signature) in zip(chain(binfo.bsources, binfo.bdepends, binfo.bimplicit), chain(binfo.bsourcesigs, binfo.bdependsigs, binfo.bimplicitsigs))}
        return binfo.dependency_map

    def _add_strings_to_dependency_map(self, dmap):
        if False:
            while True:
                i = 10
        "\n        In the case comparing node objects isn't sufficient, we'll add the strings for the nodes to the dependency map\n        :return:\n        "
        first_string = str(next(iter(dmap)))
        if first_string not in dmap:
            string_dict = {str(child): signature for (child, signature) in dmap.items()}
            dmap.update(string_dict)
        return dmap

    def _get_previous_signatures(self, dmap):
        if False:
            i = 10
            return i + 15
        '\n        Return a list of corresponding csigs from previous\n        build in order of the node/files in children.\n\n        Args:\n            self - self\n            dmap - Dictionary of file -> csig\n\n        Returns:\n            List of csigs for provided list of children\n        '
        prev = []
        if len(dmap) == 0:
            if MD5_TIMESTAMP_DEBUG:
                print('Nothing dmap shortcutting')
            return None
        elif MD5_TIMESTAMP_DEBUG:
            print('len(dmap):%d' % len(dmap))
        if MD5_TIMESTAMP_DEBUG:
            print('Checking if self is in  map:%s id:%s type:%s' % (str(self), id(self), type(self)))
        df = dmap.get(self, False)
        if df:
            return df
        rf = self.rfile()
        if MD5_TIMESTAMP_DEBUG:
            print('Checking if self.rfile  is in  map:%s id:%s type:%s' % (str(rf), id(rf), type(rf)))
        rfm = dmap.get(rf, False)
        if rfm:
            return rfm
        c_strs = [str(self)]
        if os.altsep:
            c_strs.append(c_strs[0].replace(os.sep, os.altsep))
        for s in c_strs:
            if MD5_TIMESTAMP_DEBUG:
                print('Checking if str(self) is in map  :%s' % s)
            df = dmap.get(s, False)
            if df:
                return df
        dmap = self._add_strings_to_dependency_map(dmap)
        for s in c_strs:
            if MD5_TIMESTAMP_DEBUG:
                print('Checking if str(self) is in map (now with strings)  :%s' % s)
            df = dmap.get(s, False)
            if df:
                return df
        if not df:
            try:
                c_str = self.get_path()
                if os.altsep:
                    c_str = c_str.replace(os.sep, os.altsep)
                if MD5_TIMESTAMP_DEBUG:
                    print('Checking if self.get_path is in map (now with strings)  :%s' % s)
                df = dmap.get(c_str, None)
            except AttributeError as e:
                raise FileBuildInfoFileToCsigMappingError('No mapping from file name to content signature for :%s' % c_str)
        return df

    def changed_timestamp_then_content(self, target, prev_ni, node=None):
        if False:
            while True:
                i = 10
        "\n        Used when decider for file is Timestamp-MD5\n\n        NOTE: If the timestamp hasn't changed this will skip md5'ing the\n              file and just copy the prev_ni provided.  If the prev_ni\n              is wrong. It will propagate it.\n              See: https://github.com/SCons/scons/issues/2980\n\n        Args:\n            self - dependency\n            target - target\n            prev_ni - The NodeInfo object loaded from previous builds .sconsign\n            node - Node instance.  Check this node for file existence/timestamp\n                   if specified.\n\n        Returns:\n            Boolean - Indicates if node(File) has changed.\n        "
        if node is None:
            node = self
        bi = node.get_stored_info().binfo
        rebuilt = False
        try:
            dependency_map = bi.dependency_map
        except AttributeError as e:
            dependency_map = self._build_dependency_map(bi)
            rebuilt = True
        if len(dependency_map) == 0:
            if MD5_TIMESTAMP_DEBUG:
                print('Skipping checks len(dmap)=0')
            self.get_csig()
            return True
        new_prev_ni = self._get_previous_signatures(dependency_map)
        new = self.changed_timestamp_match(target, new_prev_ni)
        if MD5_TIMESTAMP_DEBUG:
            old = self.changed_timestamp_match(target, prev_ni)
            if old != new:
                print('Mismatch self.changed_timestamp_match(%s, prev_ni) old:%s new:%s' % (str(target), old, new))
                new_prev_ni = self._get_previous_signatures(dependency_map)
        if not new:
            try:
                self.get_ninfo().csig = new_prev_ni.csig
            except AttributeError:
                pass
            return False
        return self.changed_content(target, new_prev_ni)

    def changed_timestamp_newer(self, target, prev_ni, repo_node=None):
        if False:
            return 10
        try:
            return self.get_timestamp() > target.get_timestamp()
        except AttributeError:
            return 1

    def changed_timestamp_match(self, target, prev_ni, repo_node=None):
        if False:
            return 10
        "\n        Return True if the timestamps don't match or if there is no previous timestamp\n        :param target:\n        :param prev_ni: Information about the node from the previous build\n        :return:\n        "
        try:
            return self.get_timestamp() != prev_ni.timestamp
        except AttributeError:
            return 1

    def is_up_to_date(self):
        if False:
            print('Hello World!')
        "Check for whether the Node is current\n           In all cases self is the target we're checking to see if it's up to date\n        "
        T = 0
        if T:
            Trace('is_up_to_date(%s):' % self)
        if not self.exists():
            if T:
                Trace(' not self.exists():')
            r = self.rfile()
            if r != self:
                if not self.changed(r):
                    if T:
                        Trace(' changed(%s):' % r)
                    if self._local:
                        e = LocalCopy(self, r, None)
                        if isinstance(e, SCons.Errors.BuildError):
                            raise e
                        SCons.Node.store_info_map[self.store_info](self)
                    if T:
                        Trace(' 1\n')
                    return 1
            self.changed()
            if T:
                Trace(' None\n')
            return None
        else:
            r = self.changed()
            if T:
                Trace(' self.exists():  %s\n' % r)
            return not r

    @SCons.Memoize.CountMethodCall
    def rfile(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._memo['rfile']
        except KeyError:
            pass
        result = self
        if not self.exists():
            norm_name = _my_normcase(self.name)
            for repo_dir in self.dir.get_all_rdirs():
                try:
                    node = repo_dir.entries[norm_name]
                except KeyError:
                    node = repo_dir.file_on_disk(self.name)
                if node and node.exists() and (isinstance(node, File) or isinstance(node, Entry) or (not node.is_derived())):
                    result = node
                    result.attributes = self.attributes
                    break
        self._memo['rfile'] = result
        return result

    def find_repo_file(self):
        if False:
            while True:
                i = 10
        '\n        For this node, find if there exists a corresponding file in one or more repositories\n        :return: list of corresponding files in repositories\n        '
        retvals = []
        norm_name = _my_normcase(self.name)
        for repo_dir in self.dir.get_all_rdirs():
            try:
                node = repo_dir.entries[norm_name]
            except KeyError:
                node = repo_dir.file_on_disk(self.name)
            if node and node.exists() and (isinstance(node, File) or isinstance(node, Entry) or (not node.is_derived())):
                retvals.append(node)
        return retvals

    def rstr(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.rfile())

    def get_cachedir_csig(self):
        if False:
            i = 10
            return i + 15
        '\n        Fetch a Node\'s content signature for purposes of computing\n        another Node\'s cachesig.\n\n        This is a wrapper around the normal get_csig() method that handles\n        the somewhat obscure case of using CacheDir with the -n option.\n        Any files that don\'t exist would normally be "built" by fetching\n        them from the cache, but the normal get_csig() method will try\n        to open up the local file, which doesn\'t exist because the -n\n        option meant we didn\'t actually pull the file from cachedir.\n        But since the file *does* actually exist in the cachedir, we\n        can use its contents for the csig.\n        '
        try:
            return self.cachedir_csig
        except AttributeError:
            pass
        (cachedir, cachefile) = self.get_build_env().get_CacheDir().cachepath(self)
        if not self.exists() and cachefile and os.path.exists(cachefile):
            self.cachedir_csig = SCons.Util.MD5filesignature(cachefile, SCons.Node.FS.File.md5_chunksize * 1024)
        else:
            self.cachedir_csig = self.get_csig()
        return self.cachedir_csig

    def get_contents_sig(self):
        if False:
            print('Hello World!')
        "\n        A helper method for get_cachedir_bsig.\n\n        It computes and returns the signature for this\n        node's contents.\n        "
        try:
            return self.contentsig
        except AttributeError:
            pass
        executor = self.get_executor()
        result = self.contentsig = SCons.Util.MD5signature(executor.get_contents())
        return result

    def get_cachedir_bsig(self):
        if False:
            while True:
                i = 10
        '\n        Return the signature for a cached file, including\n        its children.\n\n        It adds the path of the cached file to the cache signature,\n        because multiple targets built by the same action will all\n        have the same build signature, and we have to differentiate\n        them somehow.\n\n        Signature should normally be string of hex digits.\n        '
        try:
            return self.cachesig
        except AttributeError:
            pass
        children = self.children()
        sigs = [n.get_cachedir_csig() for n in children]
        sigs.append(self.get_contents_sig())
        sigs.append(self.get_internal_path())
        result = self.cachesig = SCons.Util.MD5collect(sigs)
        return result
default_fs = None

def get_default_fs():
    if False:
        i = 10
        return i + 15
    global default_fs
    if not default_fs:
        default_fs = FS()
    return default_fs

class FileFinder(object):
    """
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self._memo = {}

    def filedir_lookup(self, p, fd=None):
        if False:
            return 10
        "\n        A helper method for find_file() that looks up a directory for\n        a file we're trying to find.  This only creates the Dir Node if\n        it exists on-disk, since if the directory doesn't exist we know\n        we won't find any files in it...  :-)\n\n        It would be more compact to just use this as a nested function\n        with a default keyword argument (see the commented-out version\n        below), but that doesn't work unless you have nested scopes,\n        so we define it here just so this work under Python 1.5.2.\n        "
        if fd is None:
            fd = self.default_filedir
        (dir, name) = os.path.split(fd)
        (drive, d) = _my_splitdrive(dir)
        if not name and d[:1] in ('/', OS_SEP):
            return p.fs.get_root(drive)
        if dir:
            p = self.filedir_lookup(p, dir)
            if not p:
                return None
        norm_name = _my_normcase(name)
        try:
            node = p.entries[norm_name]
        except KeyError:
            return p.dir_on_disk(name)
        if isinstance(node, Dir):
            return node
        if isinstance(node, Entry):
            node.must_be_same(Dir)
            return node
        return None

    def _find_file_key(self, filename, paths, verbose=None):
        if False:
            while True:
                i = 10
        return (filename, paths)

    @SCons.Memoize.CountDictCall(_find_file_key)
    def find_file(self, filename, paths, verbose=None):
        if False:
            return 10
        '\n        Find a node corresponding to either a derived file or a file that exists already.\n\n        Only the first file found is returned, and none is returned if no file is found.\n\n        filename: A filename to find\n        paths: A list of directory path *nodes* to search in.  Can be represented as a list, a tuple, or a callable that is called with no arguments and returns the list or tuple.\n\n        returns The node created from the found file.\n\n        '
        memo_key = self._find_file_key(filename, paths)
        try:
            memo_dict = self._memo['find_file']
        except KeyError:
            memo_dict = {}
            self._memo['find_file'] = memo_dict
        else:
            try:
                return memo_dict[memo_key]
            except KeyError:
                pass
        if verbose and (not callable(verbose)):
            if not SCons.Util.is_String(verbose):
                verbose = 'find_file'
            _verbose = u'  %s: ' % verbose
            verbose = lambda s: sys.stdout.write(_verbose + s)
        (filedir, filename) = os.path.split(filename)
        if filedir:
            self.default_filedir = filedir
            paths = [_f for _f in map(self.filedir_lookup, paths) if _f]
        result = None
        for dir in paths:
            if verbose:
                verbose("looking for '%s' in '%s' ...\n" % (filename, dir))
            (node, d) = dir.srcdir_find_file(filename)
            if node:
                if verbose:
                    verbose("... FOUND '%s' in '%s'\n" % (filename, d))
                result = node
                break
        memo_dict[memo_key] = result
        return result
find_file = FileFinder().find_file

def invalidate_node_memos(targets):
    if False:
        return 10
    '\n    Invalidate the memoized values of all Nodes (files or directories)\n    that are associated with the given entries. Has been added to\n    clear the cache of nodes affected by a direct execution of an\n    action (e.g.  Delete/Copy/Chmod). Existing Node caches become\n    inconsistent if the action is run through Execute().  The argument\n    `targets` can be a single Node object or filename, or a sequence\n    of Nodes/filenames.\n    '
    from traceback import extract_stack
    for f in extract_stack():
        if f[2] == 'Execute' and f[0][-14:] == 'Environment.py':
            break
    else:
        return
    if not SCons.Util.is_List(targets):
        targets = [targets]
    for entry in targets:
        try:
            entry.clear_memoized_values()
        except AttributeError:
            node = get_default_fs().Entry(entry)
            if node:
                node.clear_memoized_values()