"""SCons.Script.SConscript

This module defines the Python API provided to SConscript and SConstruct
files.

"""
__revision__ = 'src/engine/SCons/Script/SConscript.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons
import SCons.Action
import SCons.Builder
import SCons.Defaults
import SCons.Environment
import SCons.Errors
import SCons.Node
import SCons.Node.Alias
import SCons.Node.FS
import SCons.Platform
import SCons.SConf
import SCons.Script.Main
import SCons.Tool
from SCons.Util import is_List, is_String, is_Dict, flatten
from . import Main
import collections
import os
import os.path
import re
import sys
import traceback
import time

class SConscriptReturn(Exception):
    pass
launch_dir = os.path.abspath(os.curdir)
GlobalDict = None
global_exports = {}
sconscript_chdir = 1

def get_calling_namespaces():
    if False:
        while True:
            i = 10
    'Return the locals and globals for the function that called\n    into this module in the current call stack.'
    try:
        1 // 0
    except ZeroDivisionError:
        frame = sys.exc_info()[2].tb_frame.f_back
    while frame.f_globals.get('__name__') == __name__:
        frame = frame.f_back
    return (frame.f_locals, frame.f_globals)

def compute_exports(exports):
    if False:
        i = 10
        return i + 15
    'Compute a dictionary of exports given one of the parameters\n    to the Export() function or the exports argument to SConscript().'
    (loc, glob) = get_calling_namespaces()
    retval = {}
    try:
        for export in exports:
            if is_Dict(export):
                retval.update(export)
            else:
                try:
                    retval[export] = loc[export]
                except KeyError:
                    retval[export] = glob[export]
    except KeyError as x:
        raise SCons.Errors.UserError("Export of non-existent variable '%s'" % x)
    return retval

class Frame(object):
    """A frame on the SConstruct/SConscript call stack"""

    def __init__(self, fs, exports, sconscript):
        if False:
            while True:
                i = 10
        self.globals = BuildDefaultGlobals()
        self.retval = None
        self.prev_dir = fs.getcwd()
        self.exports = compute_exports(exports)
        if isinstance(sconscript, SCons.Node.Node):
            self.sconscript = sconscript
        elif sconscript == '-':
            self.sconscript = None
        else:
            self.sconscript = fs.File(str(sconscript))
call_stack = []

def Return(*vars, **kw):
    if False:
        i = 10
        return i + 15
    retval = []
    try:
        fvars = flatten(vars)
        for var in fvars:
            for v in var.split():
                retval.append(call_stack[-1].globals[v])
    except KeyError as x:
        raise SCons.Errors.UserError("Return of non-existent variable '%s'" % x)
    if len(retval) == 1:
        call_stack[-1].retval = retval[0]
    else:
        call_stack[-1].retval = tuple(retval)
    stop = kw.get('stop', True)
    if stop:
        raise SConscriptReturn
stack_bottom = '% Stack boTTom %'

def handle_missing_SConscript(f, must_exist=None):
    if False:
        for i in range(10):
            print('nop')
    "Take appropriate action on missing file in SConscript() call.\n\n    Print a warning or raise an exception on missing file.\n    On first warning, print a deprecation message.\n\n    Args:\n        f (str): path of missing configuration file\n        must_exist (bool): raise exception if file does not exist\n\n    Raises:\n        UserError if 'must_exist' is True or if global\n          SCons.Script._no_missing_sconscript is True.\n    "
    if must_exist or (SCons.Script._no_missing_sconscript and must_exist is not False):
        msg = "Fatal: missing SConscript '%s'" % f.get_internal_path()
        raise SCons.Errors.UserError(msg)
    if SCons.Script._warn_missing_sconscript_deprecated:
        msg = 'Calling missing SConscript without error is deprecated.\n' + 'Transition by adding must_exist=0 to SConscript calls.\n' + "Missing SConscript '%s'" % f.get_internal_path()
        SCons.Warnings.warn(SCons.Warnings.MissingSConscriptWarning, msg)
        SCons.Script._warn_missing_sconscript_deprecated = False
    else:
        msg = "Ignoring missing SConscript '%s'" % f.get_internal_path()
        SCons.Warnings.warn(SCons.Warnings.MissingSConscriptWarning, msg)

def _SConscript(fs, *files, **kw):
    if False:
        print('Hello World!')
    top = fs.Top
    sd = fs.SConstruct_dir.rdir()
    exports = kw.get('exports', [])
    results = []
    for fn in files:
        call_stack.append(Frame(fs, exports, fn))
        old_sys_path = sys.path
        try:
            SCons.Script.sconscript_reading = SCons.Script.sconscript_reading + 1
            if fn == '-':
                exec(sys.stdin.read(), call_stack[-1].globals)
            else:
                if isinstance(fn, SCons.Node.Node):
                    f = fn
                else:
                    f = fs.File(str(fn))
                _file_ = None
                fs.chdir(top, change_os_dir=1)
                if f.rexists():
                    actual = f.rfile()
                    _file_ = open(actual.get_abspath(), 'rb')
                elif f.srcnode().rexists():
                    actual = f.srcnode().rfile()
                    _file_ = open(actual.get_abspath(), 'rb')
                elif f.has_src_builder():
                    f.build()
                    f.built()
                    f.builder_set(None)
                    if f.exists():
                        _file_ = open(f.get_abspath(), 'rb')
                if _file_:
                    try:
                        src_dir = kw['src_dir']
                    except KeyError:
                        ldir = fs.Dir(f.dir.get_path(sd))
                    else:
                        ldir = fs.Dir(src_dir)
                        if not ldir.is_under(f.dir):
                            ldir = fs.Dir(f.dir.get_path(sd))
                    try:
                        fs.chdir(ldir, change_os_dir=sconscript_chdir)
                    except OSError:
                        fs.chdir(ldir, change_os_dir=0)
                        os.chdir(actual.dir.get_abspath())
                    sys.path = [f.dir.get_abspath()] + sys.path
                    call_stack[-1].globals.update({stack_bottom: 1})
                    old_file = call_stack[-1].globals.get('__file__')
                    try:
                        del call_stack[-1].globals['__file__']
                    except KeyError:
                        pass
                    try:
                        try:
                            if Main.print_time:
                                time1 = time.time()
                            scriptdata = _file_.read()
                            scriptname = _file_.name
                            _file_.close()
                            exec(compile(scriptdata, scriptname, 'exec'), call_stack[-1].globals)
                        except SConscriptReturn:
                            pass
                    finally:
                        if Main.print_time:
                            time2 = time.time()
                            print('SConscript:%s  took %0.3f ms' % (f.get_abspath(), (time2 - time1) * 1000.0))
                        if old_file is not None:
                            call_stack[-1].globals.update({__file__: old_file})
                else:
                    handle_missing_SConscript(f, kw.get('must_exist', None))
        finally:
            SCons.Script.sconscript_reading = SCons.Script.sconscript_reading - 1
            sys.path = old_sys_path
            frame = call_stack.pop()
            try:
                fs.chdir(frame.prev_dir, change_os_dir=sconscript_chdir)
            except OSError:
                fs.chdir(frame.prev_dir, change_os_dir=0)
                rdir = frame.prev_dir.rdir()
                rdir._create()
                try:
                    os.chdir(rdir.get_abspath())
                except OSError as e:
                    if SCons.Action.execute_actions:
                        raise e
            results.append(frame.retval)
    if len(results) == 1:
        return results[0]
    else:
        return tuple(results)

def SConscript_exception(file=sys.stderr):
    if False:
        for i in range(10):
            print('nop')
    'Print an exception stack trace just for the SConscript file(s).\n    This will show users who have Python errors where the problem is,\n    without cluttering the output with all of the internal calls leading\n    up to where we exec the SConscript.'
    (exc_type, exc_value, exc_tb) = sys.exc_info()
    tb = exc_tb
    while tb and stack_bottom not in tb.tb_frame.f_locals:
        tb = tb.tb_next
    if not tb:
        tb = exc_tb
    stack = traceback.extract_tb(tb)
    try:
        type = exc_type.__name__
    except AttributeError:
        type = str(exc_type)
        if type[:11] == 'exceptions.':
            type = type[11:]
    file.write('%s: %s:\n' % (type, exc_value))
    for (fname, line, func, text) in stack:
        file.write('  File "%s", line %d:\n' % (fname, line))
        file.write('    %s\n' % text)

def annotate(node):
    if False:
        return 10
    'Annotate a node with the stack frame describing the\n    SConscript file and line number that created it.'
    tb = sys.exc_info()[2]
    while tb and stack_bottom not in tb.tb_frame.f_locals:
        tb = tb.tb_next
    if not tb:
        raise SCons.Errors.InternalError('could not find SConscript stack frame')
    node.creator = traceback.extract_stack(tb)[0]

class SConsEnvironment(SCons.Environment.Base):
    """An Environment subclass that contains all of the methods that
    are particular to the wrapper SCons interface and which aren't
    (or shouldn't be) part of the build engine itself.

    Note that not all of the methods of this class have corresponding
    global functions, there are some private methods.
    """

    def _exceeds_version(self, major, minor, v_major, v_minor):
        if False:
            print('Hello World!')
        "Return 1 if 'major' and 'minor' are greater than the version\n        in 'v_major' and 'v_minor', and 0 otherwise."
        return major > v_major or (major == v_major and minor > v_minor)

    def _get_major_minor_revision(self, version_string):
        if False:
            while True:
                i = 10
        'Split a version string into major, minor and (optionally)\n        revision parts.\n\n        This is complicated by the fact that a version string can be\n        something like 3.2b1.'
        version = version_string.split(' ')[0].split('.')
        v_major = int(version[0])
        v_minor = int(re.match('\\d+', version[1]).group())
        if len(version) >= 3:
            v_revision = int(re.match('\\d+', version[2]).group())
        else:
            v_revision = 0
        return (v_major, v_minor, v_revision)

    def _get_SConscript_filenames(self, ls, kw):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert the parameters passed to SConscript() calls into a list\n        of files and export variables.  If the parameters are invalid,\n        throws SCons.Errors.UserError. Returns a tuple (l, e) where l\n        is a list of SConscript filenames and e is a list of exports.\n        '
        exports = []
        if len(ls) == 0:
            try:
                dirs = kw['dirs']
            except KeyError:
                raise SCons.Errors.UserError('Invalid SConscript usage - no parameters')
            if not is_List(dirs):
                dirs = [dirs]
            dirs = list(map(str, dirs))
            name = kw.get('name', 'SConscript')
            files = [os.path.join(n, name) for n in dirs]
        elif len(ls) == 1:
            files = ls[0]
        elif len(ls) == 2:
            files = ls[0]
            exports = self.Split(ls[1])
        else:
            raise SCons.Errors.UserError('Invalid SConscript() usage - too many arguments')
        if not is_List(files):
            files = [files]
        if kw.get('exports'):
            exports.extend(self.Split(kw['exports']))
        variant_dir = kw.get('variant_dir')
        if variant_dir:
            if len(files) != 1:
                raise SCons.Errors.UserError('Invalid SConscript() usage - can only specify one SConscript with a variant_dir')
            duplicate = kw.get('duplicate', 1)
            src_dir = kw.get('src_dir')
            if not src_dir:
                (src_dir, fname) = os.path.split(str(files[0]))
                files = [os.path.join(str(variant_dir), fname)]
            else:
                if not isinstance(src_dir, SCons.Node.Node):
                    src_dir = self.fs.Dir(src_dir)
                fn = files[0]
                if not isinstance(fn, SCons.Node.Node):
                    fn = self.fs.File(fn)
                if fn.is_under(src_dir):
                    fname = fn.get_path(src_dir)
                    files = [os.path.join(str(variant_dir), fname)]
                else:
                    files = [fn.get_abspath()]
                kw['src_dir'] = variant_dir
            self.fs.VariantDir(variant_dir, src_dir, duplicate)
        return (files, exports)

    def Configure(self, *args, **kw):
        if False:
            i = 10
            return i + 15
        if not SCons.Script.sconscript_reading:
            raise SCons.Errors.UserError('Calling Configure from Builders is not supported.')
        kw['_depth'] = kw.get('_depth', 0) + 1
        return SCons.Environment.Base.Configure(self, *args, **kw)

    def Default(self, *targets):
        if False:
            return 10
        SCons.Script._Set_Default_Targets(self, targets)

    def EnsureSConsVersion(self, major, minor, revision=0):
        if False:
            return 10
        'Exit abnormally if the SCons version is not late enough.'
        if SCons.__version__ == '__' + 'VERSION__':
            SCons.Warnings.warn(SCons.Warnings.DevelopmentVersionWarning, 'EnsureSConsVersion is ignored for development version')
            return
        scons_ver = self._get_major_minor_revision(SCons.__version__)
        if scons_ver < (major, minor, revision):
            if revision:
                scons_ver_string = '%d.%d.%d' % (major, minor, revision)
            else:
                scons_ver_string = '%d.%d' % (major, minor)
            print('SCons %s or greater required, but you have SCons %s' % (scons_ver_string, SCons.__version__))
            sys.exit(2)

    def EnsurePythonVersion(self, major, minor):
        if False:
            while True:
                i = 10
        'Exit abnormally if the Python version is not late enough.'
        if sys.version_info < (major, minor):
            v = sys.version.split()[0]
            print('Python %d.%d or greater required, but you have Python %s' % (major, minor, v))
            sys.exit(2)

    def Exit(self, value=0):
        if False:
            print('Hello World!')
        sys.exit(value)

    def Export(self, *vars, **kw):
        if False:
            for i in range(10):
                print('nop')
        for var in vars:
            global_exports.update(compute_exports(self.Split(var)))
        global_exports.update(kw)

    def GetLaunchDir(self):
        if False:
            return 10
        global launch_dir
        return launch_dir

    def GetOption(self, name):
        if False:
            i = 10
            return i + 15
        name = self.subst(name)
        return SCons.Script.Main.GetOption(name)

    def Help(self, text, append=False):
        if False:
            i = 10
            return i + 15
        text = self.subst(text, raw=1)
        SCons.Script.HelpFunction(text, append=append)

    def Import(self, *vars):
        if False:
            i = 10
            return i + 15
        try:
            frame = call_stack[-1]
            globals = frame.globals
            exports = frame.exports
            for var in vars:
                var = self.Split(var)
                for v in var:
                    if v == '*':
                        globals.update(global_exports)
                        globals.update(exports)
                    elif v in exports:
                        globals[v] = exports[v]
                    else:
                        globals[v] = global_exports[v]
        except KeyError as x:
            raise SCons.Errors.UserError("Import of non-existent variable '%s'" % x)

    def SConscript(self, *ls, **kw):
        if False:
            for i in range(10):
                print('nop')
        "Execute SCons configuration files.\n\n        Parameters:\n            *ls (str or list): configuration file(s) to execute.\n\n        Keyword arguments:\n            dirs (list): execute SConscript in each listed directory.\n            name (str): execute script 'name' (used only with 'dirs').\n            exports (list or dict): locally export variables the\n              called script(s) can import.\n            variant_dir (str): mirror sources needed for the build in\n             a variant directory to allow building in it.\n            duplicate (bool): physically duplicate sources instead of just\n              adjusting paths of derived files (used only with 'variant_dir')\n              (default is True).\n            must_exist (bool): fail if a requested script is missing\n              (default is False, default is deprecated).\n\n        Returns:\n            list of variables returned by the called script\n\n        Raises:\n            UserError: a script is not found and such exceptions are enabled.\n        "

        def subst_element(x, subst=self.subst):
            if False:
                while True:
                    i = 10
            if SCons.Util.is_List(x):
                x = list(map(subst, x))
            else:
                x = subst(x)
            return x
        ls = list(map(subst_element, ls))
        subst_kw = {}
        for (key, val) in kw.items():
            if is_String(val):
                val = self.subst(val)
            elif SCons.Util.is_List(val):
                val = [self.subst(v) if is_String(v) else v for v in val]
            subst_kw[key] = val
        (files, exports) = self._get_SConscript_filenames(ls, subst_kw)
        subst_kw['exports'] = exports
        return _SConscript(self.fs, *files, **subst_kw)

    def SConscriptChdir(self, flag):
        if False:
            return 10
        global sconscript_chdir
        sconscript_chdir = flag

    def SetOption(self, name, value):
        if False:
            while True:
                i = 10
        name = self.subst(name)
        SCons.Script.Main.SetOption(name, value)
SCons.Environment.Environment = SConsEnvironment

def Configure(*args, **kw):
    if False:
        for i in range(10):
            print('nop')
    if not SCons.Script.sconscript_reading:
        raise SCons.Errors.UserError('Calling Configure from Builders is not supported.')
    kw['_depth'] = 1
    return SCons.SConf.SConf(*args, **kw)
_DefaultEnvironmentProxy = None

def get_DefaultEnvironmentProxy():
    if False:
        i = 10
        return i + 15
    global _DefaultEnvironmentProxy
    if not _DefaultEnvironmentProxy:
        default_env = SCons.Defaults.DefaultEnvironment()
        _DefaultEnvironmentProxy = SCons.Environment.NoSubstitutionProxy(default_env)
    return _DefaultEnvironmentProxy

class DefaultEnvironmentCall(object):
    """A class that implements "global function" calls of
    Environment methods by fetching the specified method from the
    DefaultEnvironment's class.  Note that this uses an intermediate
    proxy class instead of calling the DefaultEnvironment method
    directly so that the proxy can override the subst() method and
    thereby prevent expansion of construction variables (since from
    the user's point of view this was called as a global function,
    with no associated construction environment)."""

    def __init__(self, method_name, subst=0):
        if False:
            print('Hello World!')
        self.method_name = method_name
        if subst:
            self.factory = SCons.Defaults.DefaultEnvironment
        else:
            self.factory = get_DefaultEnvironmentProxy

    def __call__(self, *args, **kw):
        if False:
            return 10
        env = self.factory()
        method = getattr(env, self.method_name)
        return method(*args, **kw)

def BuildDefaultGlobals():
    if False:
        return 10
    '\n    Create a dictionary containing all the default globals for\n    SConstruct and SConscript files.\n    '
    global GlobalDict
    if GlobalDict is None:
        GlobalDict = {}
        import SCons.Script
        d = SCons.Script.__dict__

        def not_a_module(m, d=d, mtype=type(SCons.Script)):
            if False:
                for i in range(10):
                    print('nop')
            return not isinstance(d[m], mtype)
        for m in filter(not_a_module, dir(SCons.Script)):
            GlobalDict[m] = d[m]
    return GlobalDict.copy()