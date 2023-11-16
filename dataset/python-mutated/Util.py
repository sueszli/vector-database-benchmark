"""SCons.Util

Various utility functions go here.
"""
__revision__ = 'src/engine/SCons/Util.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import os
import sys
import copy
import re
import types
import codecs
import pprint
import hashlib
PY3 = sys.version_info[0] == 3
try:
    from collections import UserDict, UserList, UserString
except ImportError:
    from UserDict import UserDict
    from UserList import UserList
    from UserString import UserString
try:
    from collections.abc import Iterable, MappingView
except ImportError:
    from collections import Iterable
from collections import OrderedDict
MethodType = types.MethodType
FunctionType = types.FunctionType
try:
    _ = type(unicode)
except NameError:
    UnicodeType = str
else:
    UnicodeType = unicode

def dictify(keys, values, result={}):
    if False:
        print('Hello World!')
    for (k, v) in zip(keys, values):
        result[k] = v
    return result
_altsep = os.altsep
if _altsep is None and sys.platform == 'win32':
    _altsep = '/'
if _altsep:

    def rightmost_separator(path, sep):
        if False:
            print('Hello World!')
        return max(path.rfind(sep), path.rfind(_altsep))
else:

    def rightmost_separator(path, sep):
        if False:
            while True:
                i = 10
        return path.rfind(sep)

def containsAny(str, set):
    if False:
        for i in range(10):
            print('nop')
    'Check whether sequence str contains ANY of the items in set.'
    for c in set:
        if c in str:
            return 1
    return 0

def containsAll(str, set):
    if False:
        return 10
    'Check whether sequence str contains ALL of the items in set.'
    for c in set:
        if c not in str:
            return 0
    return 1

def containsOnly(str, set):
    if False:
        for i in range(10):
            print('nop')
    'Check whether sequence str contains ONLY items in set.'
    for c in str:
        if c not in set:
            return 0
    return 1

def splitext(path):
    if False:
        print('Hello World!')
    'Same as os.path.splitext() but faster.'
    sep = rightmost_separator(path, os.sep)
    dot = path.rfind('.')
    if dot > sep and (not containsOnly(path[dot:], '0123456789.')):
        return (path[:dot], path[dot:])
    else:
        return (path, '')

def updrive(path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Make the drive letter (if any) upper case.\n    This is useful because Windows is inconsistent on the case\n    of the drive letter, which can cause inconsistencies when\n    calculating command signatures.\n    '
    (drive, rest) = os.path.splitdrive(path)
    if drive:
        path = drive.upper() + rest
    return path

class NodeList(UserList):
    """This class is almost exactly like a regular list of Nodes
    (actually it can hold any object), with one important difference.
    If you try to get an attribute from this list, it will return that
    attribute from every item in the list.  For example:

    >>> someList = NodeList([ '  foo  ', '  bar  ' ])
    >>> someList.strip()
    [ 'foo', 'bar' ]
    """

    def __nonzero__(self):
        if False:
            while True:
                i = 10
        return len(self.data) != 0

    def __bool__(self):
        if False:
            while True:
                i = 10
        return self.__nonzero__()

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return ' '.join(map(str, self.data))

    def __iter__(self):
        if False:
            return 10
        return iter(self.data)

    def __call__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        result = [x(*args, **kwargs) for x in self.data]
        return self.__class__(result)

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        result = [getattr(x, name) for x in self.data]
        return self.__class__(result)

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        '\n        This comes for free on py2,\n        but py3 slices of NodeList are returning a list\n        breaking slicing nodelist and refering to\n        properties and methods on contained object\n        '
        if isinstance(index, slice):
            indices = index.indices(len(self.data))
            return self.__class__([self[x] for x in range(*indices)])
        else:
            return self.data[index]
_get_env_var = re.compile('^\\$([_a-zA-Z]\\w*|{[_a-zA-Z]\\w*})$')

def get_environment_var(varstr):
    if False:
        return 10
    'Given a string, first determine if it looks like a reference\n    to a single environment variable, like "$FOO" or "${FOO}".\n    If so, return that variable with no decorations ("FOO").\n    If not, return None.'
    mo = _get_env_var.match(to_String(varstr))
    if mo:
        var = mo.group(1)
        if var[0] == '{':
            return var[1:-1]
        else:
            return var
    else:
        return None

class DisplayEngine(object):
    print_it = True

    def __call__(self, text, append_newline=1):
        if False:
            return 10
        if not self.print_it:
            return
        if append_newline:
            text = text + '\n'
        try:
            sys.stdout.write(UnicodeType(text))
        except IOError:
            pass

    def set_mode(self, mode):
        if False:
            for i in range(10):
                print('nop')
        self.print_it = mode

def render_tree(root, child_func, prune=0, margin=[0], visited=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Render a tree of nodes into an ASCII tree view.\n\n    :Parameters:\n        - `root`:       the root node of the tree\n        - `child_func`: the function called to get the children of a node\n        - `prune`:      don't visit the same node twice\n        - `margin`:     the format of the left margin to use for children of root. 1 results in a pipe, and 0 results in no pipe.\n        - `visited`:    a dictionary of visited nodes in the current branch if not prune, or in the whole tree if prune.\n    "
    rname = str(root)
    if visited is None:
        visited = {}
    children = child_func(root)
    retval = ''
    for pipe in margin[:-1]:
        if pipe:
            retval = retval + '| '
        else:
            retval = retval + '  '
    if rname in visited:
        return retval + '+-[' + rname + ']\n'
    retval = retval + '+-' + rname + '\n'
    if not prune:
        visited = copy.copy(visited)
    visited[rname] = 1
    for i in range(len(children)):
        margin.append(i < len(children) - 1)
        retval = retval + render_tree(children[i], child_func, prune, margin, visited)
        margin.pop()
    return retval
IDX = lambda N: N and 1 or 0

def print_tree(root, child_func, prune=0, showtags=0, margin=[0], visited=None):
    if False:
        while True:
            i = 10
    "\n    Print a tree of nodes.  This is like render_tree, except it prints\n    lines directly instead of creating a string representation in memory,\n    so that huge trees can be printed.\n\n    :Parameters:\n        - `root`       - the root node of the tree\n        - `child_func` - the function called to get the children of a node\n        - `prune`      - don't visit the same node twice\n        - `showtags`   - print status information to the left of each node line\n        - `margin`     - the format of the left margin to use for children of root. 1 results in a pipe, and 0 results in no pipe.\n        - `visited`    - a dictionary of visited nodes in the current branch if not prune, or in the whole tree if prune.\n    "
    rname = str(root)
    if visited is None:
        visited = {}
    if showtags:
        if showtags == 2:
            legend = ' E         = exists\n' + '  R        = exists in repository only\n' + '   b       = implicit builder\n' + '   B       = explicit builder\n' + '    S      = side effect\n' + '     P     = precious\n' + '      A    = always build\n' + '       C   = current\n' + '        N  = no clean\n' + '         H = no cache\n' + '\n'
            sys.stdout.write(legend)
        tags = ['[']
        tags.append(' E'[IDX(root.exists())])
        tags.append(' R'[IDX(root.rexists() and (not root.exists()))])
        tags.append(' BbB'[[0, 1][IDX(root.has_explicit_builder())] + [0, 2][IDX(root.has_builder())]])
        tags.append(' S'[IDX(root.side_effect)])
        tags.append(' P'[IDX(root.precious)])
        tags.append(' A'[IDX(root.always_build)])
        tags.append(' C'[IDX(root.is_up_to_date())])
        tags.append(' N'[IDX(root.noclean)])
        tags.append(' H'[IDX(root.nocache)])
        tags.append(']')
    else:
        tags = []

    def MMM(m):
        if False:
            for i in range(10):
                print('nop')
        return ['  ', '| '][m]
    margins = list(map(MMM, margin[:-1]))
    children = child_func(root)
    if prune and rname in visited and children:
        sys.stdout.write(''.join(tags + margins + ['+-[', rname, ']']) + '\n')
        return
    sys.stdout.write(''.join(tags + margins + ['+-', rname]) + '\n')
    visited[rname] = 1
    if children:
        margin.append(1)
        idx = IDX(showtags)
        for C in children[:-1]:
            print_tree(C, child_func, prune, idx, margin, visited)
        margin[-1] = 0
        print_tree(children[-1], child_func, prune, idx, margin, visited)
        margin.pop()
DictTypes = (dict, UserDict)
ListTypes = (list, UserList)
try:
    SequenceTypes = (list, tuple, UserList, MappingView)
except NameError:
    SequenceTypes = (list, tuple, UserList)
try:
    StringTypes = (str, unicode, UserString)
except NameError:
    StringTypes = (str, UserString)
try:
    BaseStringTypes = (str, unicode)
except NameError:
    BaseStringTypes = str

def is_Dict(obj, isinstance=isinstance, DictTypes=DictTypes):
    if False:
        return 10
    return isinstance(obj, DictTypes)

def is_List(obj, isinstance=isinstance, ListTypes=ListTypes):
    if False:
        i = 10
        return i + 15
    return isinstance(obj, ListTypes)

def is_Sequence(obj, isinstance=isinstance, SequenceTypes=SequenceTypes):
    if False:
        for i in range(10):
            print('nop')
    return isinstance(obj, SequenceTypes)

def is_Tuple(obj, isinstance=isinstance, tuple=tuple):
    if False:
        i = 10
        return i + 15
    return isinstance(obj, tuple)

def is_String(obj, isinstance=isinstance, StringTypes=StringTypes):
    if False:
        return 10
    return isinstance(obj, StringTypes)

def is_Scalar(obj, isinstance=isinstance, StringTypes=StringTypes, SequenceTypes=SequenceTypes):
    if False:
        i = 10
        return i + 15
    return isinstance(obj, StringTypes) or not isinstance(obj, SequenceTypes)

def do_flatten(sequence, result, isinstance=isinstance, StringTypes=StringTypes, SequenceTypes=SequenceTypes):
    if False:
        return 10
    for item in sequence:
        if isinstance(item, StringTypes) or not isinstance(item, SequenceTypes):
            result.append(item)
        else:
            do_flatten(item, result)

def flatten(obj, isinstance=isinstance, StringTypes=StringTypes, SequenceTypes=SequenceTypes, do_flatten=do_flatten):
    if False:
        while True:
            i = 10
    'Flatten a sequence to a non-nested list.\n\n    Flatten() converts either a single scalar or a nested sequence\n    to a non-nested list. Note that flatten() considers strings\n    to be scalars instead of sequences like Python would.\n    '
    if isinstance(obj, StringTypes) or not isinstance(obj, SequenceTypes):
        return [obj]
    result = []
    for item in obj:
        if isinstance(item, StringTypes) or not isinstance(item, SequenceTypes):
            result.append(item)
        else:
            do_flatten(item, result)
    return result

def flatten_sequence(sequence, isinstance=isinstance, StringTypes=StringTypes, SequenceTypes=SequenceTypes, do_flatten=do_flatten):
    if False:
        print('Hello World!')
    'Flatten a sequence to a non-nested list.\n\n    Same as flatten(), but it does not handle the single scalar\n    case. This is slightly more efficient when one knows that\n    the sequence to flatten can not be a scalar.\n    '
    result = []
    for item in sequence:
        if isinstance(item, StringTypes) or not isinstance(item, SequenceTypes):
            result.append(item)
        else:
            do_flatten(item, result)
    return result

def to_String(s, isinstance=isinstance, str=str, UserString=UserString, BaseStringTypes=BaseStringTypes):
    if False:
        print('Hello World!')
    if isinstance(s, BaseStringTypes):
        return s
    elif isinstance(s, UserString):
        return s.data
    else:
        return str(s)

def to_String_for_subst(s, isinstance=isinstance, str=str, to_String=to_String, BaseStringTypes=BaseStringTypes, SequenceTypes=SequenceTypes, UserString=UserString):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(s, BaseStringTypes):
        return s
    elif isinstance(s, SequenceTypes):
        return ' '.join([to_String_for_subst(e) for e in s])
    elif isinstance(s, UserString):
        return s.data
    else:
        return str(s)

def to_String_for_signature(obj, to_String_for_subst=to_String_for_subst, AttributeError=AttributeError):
    if False:
        while True:
            i = 10
    try:
        f = obj.for_signature
    except AttributeError:
        if isinstance(obj, dict):
            return pprint.pformat(obj, width=1000000)
        else:
            return to_String_for_subst(obj)
    else:
        return f()
_semi_deepcopy_dispatch = d = {}

def semi_deepcopy_dict(x, exclude=[]):
    if False:
        while True:
            i = 10
    copy = {}
    for (key, val) in x.items():
        if key not in exclude:
            copy[key] = semi_deepcopy(val)
    return copy
d[dict] = semi_deepcopy_dict

def _semi_deepcopy_list(x):
    if False:
        print('Hello World!')
    return list(map(semi_deepcopy, x))
d[list] = _semi_deepcopy_list

def _semi_deepcopy_tuple(x):
    if False:
        print('Hello World!')
    return tuple(map(semi_deepcopy, x))
d[tuple] = _semi_deepcopy_tuple

def semi_deepcopy(x):
    if False:
        print('Hello World!')
    copier = _semi_deepcopy_dispatch.get(type(x))
    if copier:
        return copier(x)
    else:
        if hasattr(x, '__semi_deepcopy__') and callable(x.__semi_deepcopy__):
            return x.__semi_deepcopy__()
        elif isinstance(x, UserDict):
            return x.__class__(semi_deepcopy_dict(x))
        elif isinstance(x, UserList):
            return x.__class__(_semi_deepcopy_list(x))
        return x

class Proxy(object):
    """A simple generic Proxy class, forwarding all calls to
    subject.  So, for the benefit of the python newbie, what does
    this really mean?  Well, it means that you can take an object, let's
    call it 'objA', and wrap it in this Proxy class, with a statement
    like this

                 proxyObj = Proxy(objA),

    Then, if in the future, you do something like this

                 x = proxyObj.var1,

    since Proxy does not have a 'var1' attribute (but presumably objA does),
    the request actually is equivalent to saying

                 x = objA.var1

    Inherit from this class to create a Proxy.

    Note that, with new-style classes, this does *not* work transparently
    for Proxy subclasses that use special .__*__() method names, because
    those names are now bound to the class, not the individual instances.
    You now need to know in advance which .__*__() method names you want
    to pass on to the underlying Proxy object, and specifically delegate
    their calls like this:

        class Foo(Proxy):
            __str__ = Delegate('__str__')
    """

    def __init__(self, subject):
        if False:
            while True:
                i = 10
        'Wrap an object as a Proxy object'
        self._subject = subject

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        "Retrieve an attribute from the wrapped object.  If the named\n           attribute doesn't exist, AttributeError is raised"
        return getattr(self._subject, name)

    def get(self):
        if False:
            return 10
        'Retrieve the entire wrapped object'
        return self._subject

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if issubclass(other.__class__, self._subject.__class__):
            return self._subject == other
        return self.__dict__ == other.__dict__

class Delegate(object):
    """A Python Descriptor class that delegates attribute fetches
    to an underlying wrapped subject of a Proxy.  Typical use:

        class Foo(Proxy):
            __str__ = Delegate('__str__')
    """

    def __init__(self, attribute):
        if False:
            return 10
        self.attribute = attribute

    def __get__(self, obj, cls):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(obj, cls):
            return getattr(obj._subject, self.attribute)
        else:
            return self
can_read_reg = 0
try:
    import winreg
    can_read_reg = 1
    hkey_mod = winreg
    RegOpenKeyEx = winreg.OpenKeyEx
    RegEnumKey = winreg.EnumKey
    RegEnumValue = winreg.EnumValue
    RegQueryValueEx = winreg.QueryValueEx
    RegError = winreg.error
except ImportError:
    try:
        import win32api
        import win32con
        can_read_reg = 1
        hkey_mod = win32con
        RegOpenKeyEx = win32api.RegOpenKeyEx
        RegEnumKey = win32api.RegEnumKey
        RegEnumValue = win32api.RegEnumValue
        RegQueryValueEx = win32api.RegQueryValueEx
        RegError = win32api.error
    except ImportError:

        class _NoError(Exception):
            pass
        RegError = _NoError

class PlainWindowsError(OSError):
    pass
try:
    WinError = WindowsError
except NameError:
    WinError = PlainWindowsError
if can_read_reg:
    HKEY_CLASSES_ROOT = hkey_mod.HKEY_CLASSES_ROOT
    HKEY_LOCAL_MACHINE = hkey_mod.HKEY_LOCAL_MACHINE
    HKEY_CURRENT_USER = hkey_mod.HKEY_CURRENT_USER
    HKEY_USERS = hkey_mod.HKEY_USERS

    def RegGetValue(root, key):
        if False:
            print('Hello World!')
        "This utility function returns a value in the registry\n        without having to open the key first.  Only available on\n        Windows platforms with a version of Python that can read the\n        registry.  Returns the same thing as\n        SCons.Util.RegQueryValueEx, except you just specify the entire\n        path to the value, and don't have to bother opening the key\n        first.  So:\n\n        Instead of:\n          k = SCons.Util.RegOpenKeyEx(SCons.Util.HKEY_LOCAL_MACHINE,\n                r'SOFTWARE\\Microsoft\\Windows\\CurrentVersion')\n          out = SCons.Util.RegQueryValueEx(k,\n                'ProgramFilesDir')\n\n        You can write:\n          out = SCons.Util.RegGetValue(SCons.Util.HKEY_LOCAL_MACHINE,\n                r'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\ProgramFilesDir')\n        "
        p = key.rfind('\\') + 1
        keyp = key[:p - 1]
        val = key[p:]
        k = RegOpenKeyEx(root, keyp)
        return RegQueryValueEx(k, val)
else:
    HKEY_CLASSES_ROOT = None
    HKEY_LOCAL_MACHINE = None
    HKEY_CURRENT_USER = None
    HKEY_USERS = None

    def RegGetValue(root, key):
        if False:
            return 10
        raise WinError

    def RegOpenKeyEx(root, key):
        if False:
            print('Hello World!')
        raise WinError
if sys.platform == 'win32':

    def WhereIs(file, path=None, pathext=None, reject=[]):
        if False:
            while True:
                i = 10
        if path is None:
            try:
                path = os.environ['PATH']
            except KeyError:
                return None
        if is_String(path):
            path = path.split(os.pathsep)
        if pathext is None:
            try:
                pathext = os.environ['PATHEXT']
            except KeyError:
                pathext = '.COM;.EXE;.BAT;.CMD'
        if is_String(pathext):
            pathext = pathext.split(os.pathsep)
        for ext in pathext:
            if ext.lower() == file[-len(ext):].lower():
                pathext = ['']
                break
        if not is_List(reject) and (not is_Tuple(reject)):
            reject = [reject]
        for dir in path:
            f = os.path.join(dir, file)
            for ext in pathext:
                fext = f + ext
                if os.path.isfile(fext):
                    try:
                        reject.index(fext)
                    except ValueError:
                        return os.path.normpath(fext)
                    continue
        return None
elif os.name == 'os2':

    def WhereIs(file, path=None, pathext=None, reject=[]):
        if False:
            return 10
        if path is None:
            try:
                path = os.environ['PATH']
            except KeyError:
                return None
        if is_String(path):
            path = path.split(os.pathsep)
        if pathext is None:
            pathext = ['.exe', '.cmd']
        for ext in pathext:
            if ext.lower() == file[-len(ext):].lower():
                pathext = ['']
                break
        if not is_List(reject) and (not is_Tuple(reject)):
            reject = [reject]
        for dir in path:
            f = os.path.join(dir, file)
            for ext in pathext:
                fext = f + ext
                if os.path.isfile(fext):
                    try:
                        reject.index(fext)
                    except ValueError:
                        return os.path.normpath(fext)
                    continue
        return None
else:

    def WhereIs(file, path=None, pathext=None, reject=[]):
        if False:
            return 10
        import stat
        if path is None:
            try:
                path = os.environ['PATH']
            except KeyError:
                return None
        if is_String(path):
            path = path.split(os.pathsep)
        if not is_List(reject) and (not is_Tuple(reject)):
            reject = [reject]
        for d in path:
            f = os.path.join(d, file)
            if os.path.isfile(f):
                try:
                    st = os.stat(f)
                except OSError:
                    continue
                if stat.S_IMODE(st[stat.ST_MODE]) & 73:
                    try:
                        reject.index(f)
                    except ValueError:
                        return os.path.normpath(f)
                    continue
        return None

def PrependPath(oldpath, newpath, sep=os.pathsep, delete_existing=1, canonicalize=None):
    if False:
        i = 10
        return i + 15
    'This prepends newpath elements to the given oldpath.  Will only\n    add any particular path once (leaving the first one it encounters\n    and ignoring the rest, to preserve path order), and will\n    os.path.normpath and os.path.normcase all paths to help assure\n    this.  This can also handle the case where the given old path\n    variable is a list instead of a string, in which case a list will\n    be returned instead of a string.\n\n    Example:\n      Old Path: "/foo/bar:/foo"\n      New Path: "/biz/boom:/foo"\n      Result:   "/biz/boom:/foo:/foo/bar"\n\n    If delete_existing is 0, then adding a path that exists will\n    not move it to the beginning; it will stay where it is in the\n    list.\n\n    If canonicalize is not None, it is applied to each element of\n    newpath before use.\n    '
    orig = oldpath
    is_list = 1
    paths = orig
    if not is_List(orig) and (not is_Tuple(orig)):
        paths = paths.split(sep)
        is_list = 0
    if is_String(newpath):
        newpaths = newpath.split(sep)
    elif not is_List(newpath) and (not is_Tuple(newpath)):
        newpaths = [newpath]
    else:
        newpaths = newpath
    if canonicalize:
        newpaths = list(map(canonicalize, newpaths))
    if not delete_existing:
        result = []
        normpaths = []
        for path in paths:
            if not path:
                continue
            normpath = os.path.normpath(os.path.normcase(path))
            if normpath not in normpaths:
                result.append(path)
                normpaths.append(normpath)
        newpaths.reverse()
        for path in newpaths:
            if not path:
                continue
            normpath = os.path.normpath(os.path.normcase(path))
            if normpath not in normpaths:
                result.insert(0, path)
                normpaths.append(normpath)
        paths = result
    else:
        newpaths = newpaths + paths
        normpaths = []
        paths = []
        for path in newpaths:
            normpath = os.path.normpath(os.path.normcase(path))
            if path and normpath not in normpaths:
                paths.append(path)
                normpaths.append(normpath)
    if is_list:
        return paths
    else:
        return sep.join(paths)

def AppendPath(oldpath, newpath, sep=os.pathsep, delete_existing=1, canonicalize=None):
    if False:
        i = 10
        return i + 15
    'This appends new path elements to the given old path.  Will\n    only add any particular path once (leaving the last one it\n    encounters and ignoring the rest, to preserve path order), and\n    will os.path.normpath and os.path.normcase all paths to help\n    assure this.  This can also handle the case where the given old\n    path variable is a list instead of a string, in which case a list\n    will be returned instead of a string.\n\n    Example:\n      Old Path: "/foo/bar:/foo"\n      New Path: "/biz/boom:/foo"\n      Result:   "/foo/bar:/biz/boom:/foo"\n\n    If delete_existing is 0, then adding a path that exists\n    will not move it to the end; it will stay where it is in the list.\n\n    If canonicalize is not None, it is applied to each element of\n    newpath before use.\n    '
    orig = oldpath
    is_list = 1
    paths = orig
    if not is_List(orig) and (not is_Tuple(orig)):
        paths = paths.split(sep)
        is_list = 0
    if is_String(newpath):
        newpaths = newpath.split(sep)
    elif not is_List(newpath) and (not is_Tuple(newpath)):
        newpaths = [newpath]
    else:
        newpaths = newpath
    if canonicalize:
        newpaths = list(map(canonicalize, newpaths))
    if not delete_existing:
        result = []
        normpaths = []
        for path in paths:
            if not path:
                continue
            result.append(path)
            normpaths.append(os.path.normpath(os.path.normcase(path)))
        for path in newpaths:
            if not path:
                continue
            normpath = os.path.normpath(os.path.normcase(path))
            if normpath not in normpaths:
                result.append(path)
                normpaths.append(normpath)
        paths = result
    else:
        newpaths = paths + newpaths
        newpaths.reverse()
        normpaths = []
        paths = []
        for path in newpaths:
            normpath = os.path.normpath(os.path.normcase(path))
            if path and normpath not in normpaths:
                paths.append(path)
                normpaths.append(normpath)
        paths.reverse()
    if is_list:
        return paths
    else:
        return sep.join(paths)

def AddPathIfNotExists(env_dict, key, path, sep=os.pathsep):
    if False:
        for i in range(10):
            print('nop')
    "This function will take 'key' out of the dictionary\n    'env_dict', then add the path 'path' to that key if it is not\n    already there.  This treats the value of env_dict[key] as if it\n    has a similar format to the PATH variable...a list of paths\n    separated by tokens.  The 'path' will get added to the list if it\n    is not already there."
    try:
        is_list = 1
        paths = env_dict[key]
        if not is_List(env_dict[key]):
            paths = paths.split(sep)
            is_list = 0
        if os.path.normcase(path) not in list(map(os.path.normcase, paths)):
            paths = [path] + paths
        if is_list:
            env_dict[key] = paths
        else:
            env_dict[key] = sep.join(paths)
    except KeyError:
        env_dict[key] = path
if sys.platform == 'cygwin':

    def get_native_path(path):
        if False:
            while True:
                i = 10
        'Transforms an absolute path into a native path for the system.  In\n        Cygwin, this converts from a Cygwin path to a Windows one.'
        with os.popen('cygpath -w ' + path) as p:
            npath = p.read().replace('\n', '')
        return npath
else:

    def get_native_path(path):
        if False:
            i = 10
            return i + 15
        'Transforms an absolute path into a native path for the system.\n        Non-Cygwin version, just leave the path alone.'
        return path
display = DisplayEngine()

def Split(arg):
    if False:
        return 10
    if is_List(arg) or is_Tuple(arg):
        return arg
    elif is_String(arg):
        return arg.split()
    else:
        return [arg]

class CLVar(UserList):
    """A class for command-line construction variables.

    This is a list that uses Split() to split an initial string along
    white-space arguments, and similarly to split any strings that get
    added.  This allows us to Do the Right Thing with Append() and
    Prepend() (as well as straight Python foo = env['VAR'] + 'arg1
    arg2') regardless of whether a user adds a list or a string to a
    command-line construction variable.
    """

    def __init__(self, seq=[]):
        if False:
            while True:
                i = 10
        UserList.__init__(self, Split(seq))

    def __add__(self, other):
        if False:
            i = 10
            return i + 15
        return UserList.__add__(self, CLVar(other))

    def __radd__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return UserList.__radd__(self, CLVar(other))

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return ' '.join(self.data)

class Selector(OrderedDict):
    """A callable ordered dictionary that maps file suffixes to
    dictionary values.  We preserve the order in which items are added
    so that get_suffix() calls always return the first suffix added."""

    def __call__(self, env, source, ext=None):
        if False:
            print('Hello World!')
        if ext is None:
            try:
                ext = source[0].get_suffix()
            except IndexError:
                ext = ''
        try:
            return self[ext]
        except KeyError:
            s_dict = {}
            for (k, v) in self.items():
                if k is not None:
                    s_k = env.subst(k)
                    if s_k in s_dict:
                        raise KeyError(s_dict[s_k][0], k, s_k)
                    s_dict[s_k] = (k, v)
            try:
                return s_dict[ext][1]
            except KeyError:
                try:
                    return self[None]
                except KeyError:
                    return None
if sys.platform == 'cygwin':

    def case_sensitive_suffixes(s1, s2):
        if False:
            print('Hello World!')
        return 0
else:

    def case_sensitive_suffixes(s1, s2):
        if False:
            print('Hello World!')
        return os.path.normcase(s1) != os.path.normcase(s2)

def adjustixes(fname, pre, suf, ensure_suffix=False):
    if False:
        i = 10
        return i + 15
    if pre:
        (path, fn) = os.path.split(os.path.normpath(fname))
        if fn[:len(pre)] != pre:
            fname = os.path.join(path, pre + fn)
    if suf and fname[-len(suf):] != suf and (ensure_suffix or not splitext(fname)[1]):
        fname = fname + suf
    return fname

def unique(s):
    if False:
        for i in range(10):
            print('nop')
    'Return a list of the elements in s, but without duplicates.\n\n    For example, unique([1,2,3,1,2,3]) is some permutation of [1,2,3],\n    unique("abcabc") some permutation of ["a", "b", "c"], and\n    unique(([1, 2], [2, 3], [1, 2])) some permutation of\n    [[2, 3], [1, 2]].\n\n    For best speed, all sequence elements should be hashable.  Then\n    unique() will usually work in linear time.\n\n    If not possible, the sequence elements should enjoy a total\n    ordering, and if list(s).sort() doesn\'t raise TypeError it\'s\n    assumed that they do enjoy a total ordering.  Then unique() will\n    usually work in O(N*log2(N)) time.\n\n    If that\'s not possible either, the sequence elements must support\n    equality-testing.  Then unique() will usually work in quadratic\n    time.\n    '
    n = len(s)
    if n == 0:
        return []
    u = {}
    try:
        for x in s:
            u[x] = 1
    except TypeError:
        pass
    else:
        return list(u.keys())
    del u
    try:
        t = sorted(s)
    except TypeError:
        pass
    else:
        assert n > 0
        last = t[0]
        lasti = i = 1
        while i < n:
            if t[i] != last:
                t[lasti] = last = t[i]
                lasti = lasti + 1
            i = i + 1
        return t[:lasti]
    del t
    u = []
    for x in s:
        if x not in u:
            u.append(x)
    return u

def uniquer(seq, idfun=None):
    if False:
        while True:
            i = 10

    def default_idfun(x):
        if False:
            while True:
                i = 10
        return x
    if not idfun:
        idfun = default_idfun
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result

def uniquer_hashables(seq):
    if False:
        print('Hello World!')
    seen = {}
    result = []
    for item in seq:
        if item not in seen:
            seen[item] = 1
            result.append(item)
    return result

def logical_lines(physical_lines, joiner=''.join):
    if False:
        i = 10
        return i + 15
    logical_line = []
    for line in physical_lines:
        stripped = line.rstrip()
        if stripped.endswith('\\'):
            logical_line.append(stripped[:-1])
        else:
            logical_line.append(line)
            yield joiner(logical_line)
            logical_line = []
    if logical_line:
        yield joiner(logical_line)

class LogicalLines(object):
    """ Wrapper class for the logical_lines method.

        Allows us to read all "logical" lines at once from a
        given file object.
    """

    def __init__(self, fileobj):
        if False:
            print('Hello World!')
        self.fileobj = fileobj

    def readlines(self):
        if False:
            while True:
                i = 10
        result = [l for l in logical_lines(self.fileobj)]
        return result

class UniqueList(UserList):

    def __init__(self, seq=[]):
        if False:
            while True:
                i = 10
        UserList.__init__(self, seq)
        self.unique = True

    def __make_unique(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.unique:
            self.data = uniquer_hashables(self.data)
            self.unique = True

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        self.__make_unique()
        return UserList.__lt__(self, other)

    def __le__(self, other):
        if False:
            i = 10
            return i + 15
        self.__make_unique()
        return UserList.__le__(self, other)

    def __eq__(self, other):
        if False:
            return 10
        self.__make_unique()
        return UserList.__eq__(self, other)

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        self.__make_unique()
        return UserList.__ne__(self, other)

    def __gt__(self, other):
        if False:
            i = 10
            return i + 15
        self.__make_unique()
        return UserList.__gt__(self, other)

    def __ge__(self, other):
        if False:
            while True:
                i = 10
        self.__make_unique()
        return UserList.__ge__(self, other)

    def __cmp__(self, other):
        if False:
            while True:
                i = 10
        self.__make_unique()
        return UserList.__cmp__(self, other)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        self.__make_unique()
        return UserList.__len__(self)

    def __getitem__(self, i):
        if False:
            for i in range(10):
                print('nop')
        self.__make_unique()
        return UserList.__getitem__(self, i)

    def __setitem__(self, i, item):
        if False:
            while True:
                i = 10
        UserList.__setitem__(self, i, item)
        self.unique = False

    def __getslice__(self, i, j):
        if False:
            while True:
                i = 10
        self.__make_unique()
        return UserList.__getslice__(self, i, j)

    def __setslice__(self, i, j, other):
        if False:
            for i in range(10):
                print('nop')
        UserList.__setslice__(self, i, j, other)
        self.unique = False

    def __add__(self, other):
        if False:
            print('Hello World!')
        result = UserList.__add__(self, other)
        result.unique = False
        return result

    def __radd__(self, other):
        if False:
            return 10
        result = UserList.__radd__(self, other)
        result.unique = False
        return result

    def __iadd__(self, other):
        if False:
            while True:
                i = 10
        result = UserList.__iadd__(self, other)
        result.unique = False
        return result

    def __mul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        result = UserList.__mul__(self, other)
        result.unique = False
        return result

    def __rmul__(self, other):
        if False:
            return 10
        result = UserList.__rmul__(self, other)
        result.unique = False
        return result

    def __imul__(self, other):
        if False:
            i = 10
            return i + 15
        result = UserList.__imul__(self, other)
        result.unique = False
        return result

    def append(self, item):
        if False:
            i = 10
            return i + 15
        UserList.append(self, item)
        self.unique = False

    def insert(self, i):
        if False:
            for i in range(10):
                print('nop')
        UserList.insert(self, i)
        self.unique = False

    def count(self, item):
        if False:
            while True:
                i = 10
        self.__make_unique()
        return UserList.count(self, item)

    def index(self, item):
        if False:
            for i in range(10):
                print('nop')
        self.__make_unique()
        return UserList.index(self, item)

    def reverse(self):
        if False:
            for i in range(10):
                print('nop')
        self.__make_unique()
        UserList.reverse(self)

    def sort(self, *args, **kwds):
        if False:
            return 10
        self.__make_unique()
        return UserList.sort(self, *args, **kwds)

    def extend(self, other):
        if False:
            return 10
        UserList.extend(self, other)
        self.unique = False

class Unbuffered(object):
    """
    A proxy class that wraps a file object, flushing after every write,
    and delegating everything else to the wrapped object.
    """

    def __init__(self, file):
        if False:
            while True:
                i = 10
        self.file = file
        self.softspace = 0

    def write(self, arg):
        if False:
            i = 10
            return i + 15
        try:
            self.file.write(arg)
            self.file.flush()
        except IOError:
            pass

    def __getattr__(self, attr):
        if False:
            return 10
        return getattr(self.file, attr)

def make_path_relative(path):
    if False:
        i = 10
        return i + 15
    ' makes an absolute path name to a relative pathname.\n    '
    if os.path.isabs(path):
        (drive_s, path) = os.path.splitdrive(path)
        import re
        if not drive_s:
            path = re.compile('/*(.*)').findall(path)[0]
        else:
            path = path[1:]
    assert not os.path.isabs(path), path
    return path

def AddMethod(obj, function, name=None):
    if False:
        while True:
            i = 10
    '\n    Adds either a bound method to an instance or the function itself (or an unbound method in Python 2) to a class.\n    If name is ommited the name of the specified function\n    is used by default.\n\n    Example::\n\n        a = A()\n        def f(self, x, y):\n        self.z = x + y\n        AddMethod(f, A, "add")\n        a.add(2, 4)\n        print(a.z)\n        AddMethod(lambda self, i: self.l[i], a, "listIndex")\n        print(a.listIndex(5))\n    '
    if name is None:
        name = function.__name__
    else:
        function = RenameFunction(function, name)
    if hasattr(obj, '__class__') and obj.__class__ is not type:
        if sys.version_info[:2] > (3, 2):
            method = MethodType(function, obj)
        else:
            method = MethodType(function, obj, obj.__class__)
    else:
        method = function
    setattr(obj, name, method)

def RenameFunction(function, name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a function identical to the specified function, but with\n    the specified name.\n    '
    return FunctionType(function.__code__, function.__globals__, name, function.__defaults__)
if hasattr(hashlib, 'md5'):
    md5 = True

    def MD5signature(s):
        if False:
            i = 10
            return i + 15
        '\n        Generate md5 signature of a string\n\n        :param s: either string or bytes. Normally should be bytes\n        :return: String of hex digits representing the signature\n        '
        m = hashlib.md5()
        try:
            m.update(to_bytes(s))
        except TypeError as e:
            m.update(to_bytes(str(s)))
        return m.hexdigest()

    def MD5filesignature(fname, chunksize=65536):
        if False:
            return 10
        '\n        Generate the md5 signature of a file\n\n        :param fname: file to hash\n        :param chunksize: chunk size to read\n        :return: String of Hex digits representing the signature\n        '
        m = hashlib.md5()
        with open(fname, 'rb') as f:
            while True:
                blck = f.read(chunksize)
                if not blck:
                    break
                m.update(to_bytes(blck))
        return m.hexdigest()
else:
    md5 = False

    def MD5signature(s):
        if False:
            i = 10
            return i + 15
        return str(s)

    def MD5filesignature(fname, chunksize=65536):
        if False:
            while True:
                i = 10
        with open(fname, 'rb') as f:
            result = f.read()
        return result

def MD5collect(signatures):
    if False:
        for i in range(10):
            print('nop')
    '\n    Collects a list of signatures into an aggregate signature.\n\n    signatures - a list of signatures\n    returns - the aggregate signature\n    '
    if len(signatures) == 1:
        return signatures[0]
    else:
        return MD5signature(', '.join(signatures))

def silent_intern(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Perform sys.intern() on the passed argument and return the result.\n    If the input is ineligible (e.g. a unicode string) the original argument is\n    returned and no exception is thrown.\n    '
    try:
        return sys.intern(x)
    except TypeError:
        return x

class Null(object):
    """ Null objects always and reliably "do nothing." """

    def __new__(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if '_instance' not in vars(cls):
            cls._instance = super(Null, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        pass

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Null(0x%08X)' % id(self)

    def __nonzero__(self):
        if False:
            return 10
        return False

    def __bool__(self):
        if False:
            return 10
        return False

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        return self

    def __setattr__(self, name, value):
        if False:
            print('Hello World!')
        return self

    def __delattr__(self, name):
        if False:
            print('Hello World!')
        return self

class NullSeq(Null):

    def __len__(self):
        if False:
            while True:
                i = 10
        return 0

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(())

    def __getitem__(self, i):
        if False:
            print('Hello World!')
        return self

    def __delitem__(self, i):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __setitem__(self, i, v):
        if False:
            print('Hello World!')
        return self
del __revision__

def to_bytes(s):
    if False:
        return 10
    if s is None:
        return b'None'
    if not PY3 and isinstance(s, UnicodeType):
        return bytearray(s, 'utf-8')
    if isinstance(s, (bytes, bytearray)) or bytes is str:
        return s
    return bytes(s, 'utf-8')

def to_str(s):
    if False:
        i = 10
        return i + 15
    if s is None:
        return 'None'
    if bytes is str or is_String(s):
        return s
    return str(s, 'utf-8')

def cmp(a, b):
    if False:
        while True:
            i = 10
    "\n    Define cmp because it's no longer available in python3\n    Works under python 2 as well\n    "
    return (a > b) - (a < b)

def get_env_bool(env, name, default=False):
    if False:
        return 10
    "Get a value of env[name] converted to boolean. The value of env[name] is\n    interpreted as follows: 'true', 'yes', 'y', 'on' (case insensitive) and\n    anything convertible to int that yields non-zero integer are True values;\n    '0', 'false', 'no', 'n' and 'off' (case insensitive) are False values. For\n    all other cases, default value is returned.\n\n    :Parameters:\n        - `env`     - dict or dict-like object, a convainer with variables\n        - `name`    - name of the variable in env to be returned\n        - `default` - returned when env[name] does not exist or can't be converted to bool\n    "
    try:
        var = env[name]
    except KeyError:
        return default
    try:
        return bool(int(var))
    except ValueError:
        if str(var).lower() in ('true', 'yes', 'y', 'on'):
            return True
        elif str(var).lower() in ('false', 'no', 'n', 'off'):
            return False
        else:
            return default

def get_os_env_bool(name, default=False):
    if False:
        while True:
            i = 10
    'Same as get_env_bool(os.environ, name, default).'
    return get_env_bool(os.environ, name, default)