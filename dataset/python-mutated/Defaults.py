"""SCons.Defaults

Builders and other things for the local site.  Here's where we'll
duplicate the functionality of autoconf until we move it into the
installation procedure or use something like qmconf.

The code that reads the registry to find MSVC components was borrowed
from distutils.msvccompiler.

"""
from __future__ import division
__revision__ = 'src/engine/SCons/Defaults.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import os
import errno
import shutil
import stat
import time
import sys
import SCons.Action
import SCons.Builder
import SCons.CacheDir
import SCons.Environment
import SCons.PathList
import SCons.Subst
import SCons.Tool
_default_env = None

def _fetch_DefaultEnvironment(*args, **kw):
    if False:
        i = 10
        return i + 15
    '\n    Returns the already-created default construction environment.\n    '
    global _default_env
    return _default_env

def DefaultEnvironment(*args, **kw):
    if False:
        print('Hello World!')
    "\n    Initial public entry point for creating the default construction\n    Environment.\n\n    After creating the environment, we overwrite our name\n    (DefaultEnvironment) with the _fetch_DefaultEnvironment() function,\n    which more efficiently returns the initialized default construction\n    environment without checking for its existence.\n\n    (This function still exists with its _default_check because someone\n    else (*cough* Script/__init__.py *cough*) may keep a reference\n    to this function.  So we can't use the fully functional idiom of\n    having the name originally be a something that *only* creates the\n    construction environment and then overwrites the name.)\n    "
    global _default_env
    if not _default_env:
        import SCons.Util
        _default_env = SCons.Environment.Environment(*args, **kw)
        if SCons.Util.md5:
            _default_env.Decider('MD5')
        else:
            _default_env.Decider('timestamp-match')
        global DefaultEnvironment
        DefaultEnvironment = _fetch_DefaultEnvironment
        _default_env._CacheDir_path = None
    return _default_env

def StaticObjectEmitter(target, source, env):
    if False:
        for i in range(10):
            print('nop')
    for tgt in target:
        tgt.attributes.shared = None
    return (target, source)

def SharedObjectEmitter(target, source, env):
    if False:
        for i in range(10):
            print('nop')
    for tgt in target:
        tgt.attributes.shared = 1
    return (target, source)

def SharedFlagChecker(source, target, env):
    if False:
        print('Hello World!')
    same = env.subst('$STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME')
    if same == '0' or same == '' or same == 'False':
        for src in source:
            try:
                shared = src.attributes.shared
            except AttributeError:
                shared = None
            if not shared:
                raise SCons.Errors.UserError('Source file: %s is static and is not compatible with shared target: %s' % (src, target[0]))
SharedCheck = SCons.Action.Action(SharedFlagChecker, None)
CScan = SCons.Tool.CScanner
ProgScan = SCons.Tool.ProgramScanner
import SCons.Scanner.Dir
DirScanner = SCons.Scanner.Dir.DirScanner()
DirEntryScanner = SCons.Scanner.Dir.DirEntryScanner()
CAction = SCons.Action.Action('$CCCOM', '$CCCOMSTR')
ShCAction = SCons.Action.Action('$SHCCCOM', '$SHCCCOMSTR')
CXXAction = SCons.Action.Action('$CXXCOM', '$CXXCOMSTR')
ShCXXAction = SCons.Action.Action('$SHCXXCOM', '$SHCXXCOMSTR')
DAction = SCons.Action.Action('$DCOM', '$DCOMSTR')
ShDAction = SCons.Action.Action('$SHDCOM', '$SHDCOMSTR')
ASAction = SCons.Action.Action('$ASCOM', '$ASCOMSTR')
ASPPAction = SCons.Action.Action('$ASPPCOM', '$ASPPCOMSTR')
LinkAction = SCons.Action.Action('$LINKCOM', '$LINKCOMSTR')
ShLinkAction = SCons.Action.Action('$SHLINKCOM', '$SHLINKCOMSTR')
LdModuleLinkAction = SCons.Action.Action('$LDMODULECOM', '$LDMODULECOMSTR')
ActionFactory = SCons.Action.ActionFactory

def get_paths_str(dest):
    if False:
        for i in range(10):
            print('nop')
    if SCons.Util.is_List(dest):
        elem_strs = []
        for element in dest:
            elem_strs.append('"' + str(element) + '"')
        return '[' + ', '.join(elem_strs) + ']'
    else:
        return '"' + str(dest) + '"'
permission_dic = {'u': {'r': stat.S_IRUSR, 'w': stat.S_IWUSR, 'x': stat.S_IXUSR}, 'g': {'r': stat.S_IRGRP, 'w': stat.S_IWGRP, 'x': stat.S_IXGRP}, 'o': {'r': stat.S_IROTH, 'w': stat.S_IWOTH, 'x': stat.S_IXOTH}}

def chmod_func(dest, mode):
    if False:
        while True:
            i = 10
    import SCons.Util
    from string import digits
    SCons.Node.FS.invalidate_node_memos(dest)
    if not SCons.Util.is_List(dest):
        dest = [dest]
    if SCons.Util.is_String(mode) and 0 not in [i in digits for i in mode]:
        mode = int(mode, 8)
    if not SCons.Util.is_String(mode):
        for element in dest:
            os.chmod(str(element), mode)
    else:
        mode = str(mode)
        for operation in mode.split(','):
            if '=' in operation:
                operator = '='
            elif '+' in operation:
                operator = '+'
            elif '-' in operation:
                operator = '-'
            else:
                raise SyntaxError('Could not find +, - or =')
            operation_list = operation.split(operator)
            if len(operation_list) != 2:
                raise SyntaxError('More than one operator found')
            user = operation_list[0].strip().replace('a', 'ugo')
            permission = operation_list[1].strip()
            new_perm = 0
            for u in user:
                for p in permission:
                    try:
                        new_perm = new_perm | permission_dic[u][p]
                    except KeyError:
                        raise SyntaxError('Unrecognized user or permission format')
            for element in dest:
                curr_perm = os.stat(str(element)).st_mode
                if operator == '=':
                    os.chmod(str(element), new_perm)
                elif operator == '+':
                    os.chmod(str(element), curr_perm | new_perm)
                elif operator == '-':
                    os.chmod(str(element), curr_perm & ~new_perm)

def chmod_strfunc(dest, mode):
    if False:
        while True:
            i = 10
    import SCons.Util
    if not SCons.Util.is_String(mode):
        return 'Chmod(%s, 0%o)' % (get_paths_str(dest), mode)
    else:
        return 'Chmod(%s, "%s")' % (get_paths_str(dest), str(mode))
Chmod = ActionFactory(chmod_func, chmod_strfunc)

def copy_func(dest, src, symlinks=True):
    if False:
        print('Hello World!')
    "\n    If symlinks (is true), then a symbolic link will be\n    shallow copied and recreated as a symbolic link; otherwise, copying\n    a symbolic link will be equivalent to copying the symbolic link's\n    final target regardless of symbolic link depth.\n    "
    dest = str(dest)
    src = str(src)
    SCons.Node.FS.invalidate_node_memos(dest)
    if SCons.Util.is_List(src) and os.path.isdir(dest):
        for file in src:
            shutil.copy2(file, dest)
        return 0
    elif os.path.islink(src):
        if symlinks:
            return os.symlink(os.readlink(src), dest)
        else:
            return copy_func(dest, os.path.realpath(src))
    elif os.path.isfile(src):
        shutil.copy2(src, dest)
        return 0
    else:
        shutil.copytree(src, dest, symlinks)
        return 0
Copy = ActionFactory(copy_func, lambda dest, src, symlinks=True: 'Copy("%s", "%s")' % (dest, src))

def delete_func(dest, must_exist=0):
    if False:
        for i in range(10):
            print('nop')
    SCons.Node.FS.invalidate_node_memos(dest)
    if not SCons.Util.is_List(dest):
        dest = [dest]
    for entry in dest:
        entry = str(entry)
        entry_exists = os.path.exists(entry) or os.path.islink(entry)
        if not entry_exists and (not must_exist):
            continue
        if os.path.isdir(entry) and (not os.path.islink(entry)):
            shutil.rmtree(entry, 1)
            continue
        os.unlink(entry)

def delete_strfunc(dest, must_exist=0):
    if False:
        for i in range(10):
            print('nop')
    return 'Delete(%s)' % get_paths_str(dest)
Delete = ActionFactory(delete_func, delete_strfunc)

def mkdir_func(dest):
    if False:
        for i in range(10):
            print('nop')
    SCons.Node.FS.invalidate_node_memos(dest)
    if not SCons.Util.is_List(dest):
        dest = [dest]
    for entry in dest:
        try:
            os.makedirs(str(entry))
        except os.error as e:
            p = str(entry)
            if (e.args[0] == errno.EEXIST or (sys.platform == 'win32' and e.args[0] == 183)) and os.path.isdir(str(entry)):
                pass
            else:
                raise
Mkdir = ActionFactory(mkdir_func, lambda dir: 'Mkdir(%s)' % get_paths_str(dir))

def move_func(dest, src):
    if False:
        for i in range(10):
            print('nop')
    SCons.Node.FS.invalidate_node_memos(dest)
    SCons.Node.FS.invalidate_node_memos(src)
    shutil.move(src, dest)
Move = ActionFactory(move_func, lambda dest, src: 'Move("%s", "%s")' % (dest, src), convert=str)

def touch_func(dest):
    if False:
        i = 10
        return i + 15
    SCons.Node.FS.invalidate_node_memos(dest)
    if not SCons.Util.is_List(dest):
        dest = [dest]
    for file in dest:
        file = str(file)
        mtime = int(time.time())
        if os.path.exists(file):
            atime = os.path.getatime(file)
        else:
            with open(file, 'w'):
                atime = mtime
        os.utime(file, (atime, mtime))
Touch = ActionFactory(touch_func, lambda file: 'Touch(%s)' % get_paths_str(file))

def _concat(prefix, list, suffix, env, f=lambda x: x, target=None, source=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Creates a new list from 'list' by first interpolating each element\n    in the list using the 'env' dictionary and then calling f on the\n    list, and finally calling _concat_ixes to concatenate 'prefix' and\n    'suffix' onto each element of the list.\n    "
    if not list:
        return list
    l = f(SCons.PathList.PathList(list).subst_path(env, target, source))
    if l is not None:
        list = l
    return _concat_ixes(prefix, list, suffix, env)

def _concat_ixes(prefix, list, suffix, env):
    if False:
        while True:
            i = 10
    "\n    Creates a new list from 'list' by concatenating the 'prefix' and\n    'suffix' arguments onto each element of the list.  A trailing space\n    on 'prefix' or leading space on 'suffix' will cause them to be put\n    into separate list elements rather than being concatenated.\n    "
    result = []
    prefix = str(env.subst(prefix, SCons.Subst.SUBST_RAW))
    suffix = str(env.subst(suffix, SCons.Subst.SUBST_RAW))
    for x in list:
        if isinstance(x, SCons.Node.FS.File):
            result.append(x)
            continue
        x = str(x)
        if x:
            if prefix:
                if prefix[-1] == ' ':
                    result.append(prefix[:-1])
                elif x[:len(prefix)] != prefix:
                    x = prefix + x
            result.append(x)
            if suffix:
                if suffix[0] == ' ':
                    result.append(suffix[1:])
                elif x[-len(suffix):] != suffix:
                    result[-1] = result[-1] + suffix
    return result

def _stripixes(prefix, itms, suffix, stripprefixes, stripsuffixes, env, c=None):
    if False:
        i = 10
        return i + 15
    "\n    This is a wrapper around _concat()/_concat_ixes() that checks for\n    the existence of prefixes or suffixes on list items and strips them\n    where it finds them.  This is used by tools (like the GNU linker)\n    that need to turn something like 'libfoo.a' into '-lfoo'.\n    "
    if not itms:
        return itms
    if not callable(c):
        env_c = env['_concat']
        if env_c != _concat and callable(env_c):
            c = env_c
        else:
            c = _concat_ixes
    stripprefixes = list(map(env.subst, SCons.Util.flatten(stripprefixes)))
    stripsuffixes = list(map(env.subst, SCons.Util.flatten(stripsuffixes)))
    stripped = []
    for l in SCons.PathList.PathList(itms).subst_path(env, None, None):
        if isinstance(l, SCons.Node.FS.File):
            stripped.append(l)
            continue
        if not SCons.Util.is_String(l):
            l = str(l)
        for stripprefix in stripprefixes:
            lsp = len(stripprefix)
            if l[:lsp] == stripprefix:
                l = l[lsp:]
                break
        for stripsuffix in stripsuffixes:
            lss = len(stripsuffix)
            if l[-lss:] == stripsuffix:
                l = l[:-lss]
                break
        stripped.append(l)
    return c(prefix, stripped, suffix, env)

def processDefines(defs):
    if False:
        i = 10
        return i + 15
    'process defines, resolving strings, lists, dictionaries, into a list of\n    strings\n    '
    if SCons.Util.is_List(defs):
        l = []
        for d in defs:
            if d is None:
                continue
            elif SCons.Util.is_List(d) or isinstance(d, tuple):
                if len(d) >= 2:
                    l.append(str(d[0]) + '=' + str(d[1]))
                else:
                    l.append(str(d[0]))
            elif SCons.Util.is_Dict(d):
                for (macro, value) in d.items():
                    if value is not None:
                        l.append(str(macro) + '=' + str(value))
                    else:
                        l.append(str(macro))
            elif SCons.Util.is_String(d):
                l.append(str(d))
            else:
                raise SCons.Errors.UserError('DEFINE %s is not a list, dict, string or None.' % repr(d))
    elif SCons.Util.is_Dict(defs):
        l = []
        for (k, v) in sorted(defs.items()):
            if v is None:
                l.append(str(k))
            else:
                l.append(str(k) + '=' + str(v))
    else:
        l = [str(defs)]
    return l

def _defines(prefix, defs, suffix, env, c=_concat_ixes):
    if False:
        for i in range(10):
            print('nop')
    'A wrapper around _concat_ixes that turns a list or string\n    into a list of C preprocessor command-line definitions.\n    '
    return c(prefix, env.subst_path(processDefines(defs)), suffix, env)

class NullCmdGenerator(object):
    """This is a callable class that can be used in place of other
    command generators if you don't want them to do anything.

    The __call__ method for this class simply returns the thing
    you instantiated it with.

    Example usage:
    env["DO_NOTHING"] = NullCmdGenerator
    env["LINKCOM"] = "${DO_NOTHING('$LINK $SOURCES $TARGET')}"
    """

    def __init__(self, cmd):
        if False:
            while True:
                i = 10
        self.cmd = cmd

    def __call__(self, target, source, env, for_signature=None):
        if False:
            print('Hello World!')
        return self.cmd

class Variable_Method_Caller(object):
    """A class for finding a construction variable on the stack and
    calling one of its methods.

    We use this to support "construction variables" in our string
    eval()s that actually stand in for methods--specifically, use
    of "RDirs" in call to _concat that should actually execute the
    "TARGET.RDirs" method.  (We used to support this by creating a little
    "build dictionary" that mapped RDirs to the method, but this got in
    the way of Memoizing construction environments, because we had to
    create new environment objects to hold the variables.)
    """

    def __init__(self, variable, method):
        if False:
            while True:
                i = 10
        self.variable = variable
        self.method = method

    def __call__(self, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        try:
            1 // 0
        except ZeroDivisionError:
            frame = sys.exc_info()[2].tb_frame.f_back
        variable = self.variable
        while frame:
            if variable in frame.f_locals:
                v = frame.f_locals[variable]
                if v:
                    method = getattr(v, self.method)
                    return method(*args, **kw)
            frame = frame.f_back
        return None

def __libversionflags(env, version_var, flags_var):
    if False:
        while True:
            i = 10
    try:
        if env.subst('$' + version_var):
            return env[flags_var]
    except KeyError:
        pass
    return None
ConstructionEnvironment = {'BUILDERS': {}, 'SCANNERS': [SCons.Tool.SourceFileScanner], 'CONFIGUREDIR': '#/.sconf_temp', 'CONFIGURELOG': '#/config.log', 'CPPSUFFIXES': SCons.Tool.CSuffixes, 'DSUFFIXES': SCons.Tool.DSuffixes, 'ENV': {}, 'IDLSUFFIXES': SCons.Tool.IDLSuffixes, '_concat': _concat, '_defines': _defines, '_stripixes': _stripixes, '_LIBFLAGS': '${_concat(LIBLINKPREFIX, LIBS, LIBLINKSUFFIX, __env__)}', '_LIBDIRFLAGS': '$( ${_concat(LIBDIRPREFIX, LIBPATH, LIBDIRSUFFIX, __env__, RDirs, TARGET, SOURCE)} $)', '_CPPINCFLAGS': '$( ${_concat(INCPREFIX, CPPPATH, INCSUFFIX, __env__, RDirs, TARGET, SOURCE)} $)', '_CPPDEFFLAGS': '${_defines(CPPDEFPREFIX, CPPDEFINES, CPPDEFSUFFIX, __env__)}', '__libversionflags': __libversionflags, '__SHLIBVERSIONFLAGS': '${__libversionflags(__env__,"SHLIBVERSION","_SHLIBVERSIONFLAGS")}', '__LDMODULEVERSIONFLAGS': '${__libversionflags(__env__,"LDMODULEVERSION","_LDMODULEVERSIONFLAGS")}', '__DSHLIBVERSIONFLAGS': '${__libversionflags(__env__,"DSHLIBVERSION","_DSHLIBVERSIONFLAGS")}', 'TEMPFILE': NullCmdGenerator, 'TEMPFILEARGJOIN': ' ', 'Dir': Variable_Method_Caller('TARGET', 'Dir'), 'Dirs': Variable_Method_Caller('TARGET', 'Dirs'), 'File': Variable_Method_Caller('TARGET', 'File'), 'RDirs': Variable_Method_Caller('TARGET', 'RDirs')}