"""The xonsh built-ins.

Note that this module is named 'built_ins' so as not to be confused with the
special Python builtins module.
"""
import atexit
import builtins
import collections.abc as cabc
import contextlib
import inspect
import itertools
import os
import pathlib
import re
import signal
import sys
import types
import warnings
from ast import AST
from xonsh.inspectors import Inspector
from xonsh.lazyasd import lazyobject
from xonsh.platform import ON_POSIX
from xonsh.tools import XonshCalledProcessError, XonshError, expand_path, globpath, print_color
INSPECTOR = Inspector()
warnings.filterwarnings('once', category=DeprecationWarning)

@lazyobject
def AT_EXIT_SIGNALS():
    if False:
        while True:
            i = 10
    sigs = (signal.SIGABRT, signal.SIGFPE, signal.SIGILL, signal.SIGSEGV, signal.SIGTERM)
    if ON_POSIX:
        sigs += (signal.SIGTSTP, signal.SIGQUIT, signal.SIGHUP)
    return sigs

def resetting_signal_handle(sig, f):
    if False:
        print('Hello World!')
    'Sets a new signal handle that will automatically restore the old value\n    once the new handle is finished.\n    '
    oldh = signal.getsignal(sig)

    def newh(s=None, frame=None):
        if False:
            while True:
                i = 10
        f(s, frame)
        signal.signal(sig, oldh)
        if sig != 0:
            sys.exit(sig)
    signal.signal(sig, newh)

def helper(x, name=''):
    if False:
        print('Hello World!')
    'Prints help about, and then returns that variable.'
    name = name or getattr(x, '__name__', '')
    INSPECTOR.pinfo(x, oname=name, detail_level=0)
    return x

def superhelper(x, name=''):
    if False:
        while True:
            i = 10
    'Prints help about, and then returns that variable.'
    name = name or getattr(x, '__name__', '')
    INSPECTOR.pinfo(x, oname=name, detail_level=1)
    return x

def reglob(path, parts=None, i=None):
    if False:
        while True:
            i = 10
    'Regular expression-based globbing.'
    if parts is None:
        path = os.path.normpath(path)
        (drive, tail) = os.path.splitdrive(path)
        parts = tail.split(os.sep)
        d = os.sep if os.path.isabs(path) else '.'
        d = os.path.join(drive, d)
        return reglob(d, parts, i=0)
    base = subdir = path
    if i == 0:
        if not os.path.isabs(base):
            base = ''
        elif len(parts) > 1:
            i += 1
    try:
        regex = re.compile(parts[i])
    except Exception as e:
        if isinstance(e, re.error) and str(e) == 'nothing to repeat at position 0':
            raise XonshError("Consider adding a leading '.' to your glob regex pattern.") from e
        else:
            raise e
    files = os.listdir(subdir)
    files.sort()
    paths = []
    i1 = i + 1
    if i1 == len(parts):
        for f in files:
            p = os.path.join(base, f)
            if regex.fullmatch(f) is not None:
                paths.append(p)
    else:
        for f in files:
            p = os.path.join(base, f)
            if regex.fullmatch(f) is None or not os.path.isdir(p):
                continue
            paths += reglob(p, parts=parts, i=i1)
    return paths

def path_literal(s):
    if False:
        print('Hello World!')
    s = expand_path(s)
    return pathlib.Path(s)

def regexsearch(s):
    if False:
        i = 10
        return i + 15
    s = expand_path(s)
    return reglob(s)

def globsearch(s):
    if False:
        while True:
            i = 10
    csc = XSH.env.get('CASE_SENSITIVE_COMPLETIONS')
    glob_sorted = XSH.env.get('GLOB_SORTED')
    dotglob = XSH.env.get('DOTGLOB')
    return globpath(s, ignore_case=not csc, return_empty=True, sort_result=glob_sorted, include_dotfiles=dotglob)

def pathsearch(func, s, pymode=False, pathobj=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Takes a string and returns a list of file paths that match (regex, glob,\n    or arbitrary search function). If pathobj=True, the return is a list of\n    pathlib.Path objects instead of strings.\n    '
    if not callable(func) or len(inspect.signature(func).parameters) != 1:
        error = '%r is not a known path search function'
        raise XonshError(error % func)
    o = func(s)
    if pathobj and pymode:
        o = list(map(pathlib.Path, o))
    no_match = [] if pymode else [s]
    return o if len(o) != 0 else no_match

def subproc_captured_stdout(*cmds, envs=None):
    if False:
        for i in range(10):
            print('nop')
    'Runs a subprocess, capturing the output. Returns the stdout\n    that was produced as a str.\n    '
    import xonsh.procs.specs
    return xonsh.procs.specs.run_subproc(cmds, captured='stdout', envs=envs)

def subproc_captured_inject(*cmds, envs=None):
    if False:
        print('Hello World!')
    "Runs a subprocess, capturing the output. Returns a list of\n    whitespace-separated strings of the stdout that was produced.\n    The string is split using xonsh's lexer, rather than Python's str.split()\n    or shlex.split().\n    "
    import xonsh.procs.specs
    o = xonsh.procs.specs.run_subproc(cmds, captured='object', envs=envs)
    o.end()
    toks = []
    for line in o:
        line = line.rstrip(os.linesep)
        toks.extend(XSH.execer.parser.lexer.split(line))
    return toks

def subproc_captured_object(*cmds, envs=None):
    if False:
        i = 10
        return i + 15
    '\n    Runs a subprocess, capturing the output. Returns an instance of\n    CommandPipeline representing the completed command.\n    '
    import xonsh.procs.specs
    return xonsh.procs.specs.run_subproc(cmds, captured='object', envs=envs)

def subproc_captured_hiddenobject(*cmds, envs=None):
    if False:
        print('Hello World!')
    'Runs a subprocess, capturing the output. Returns an instance of\n    HiddenCommandPipeline representing the completed command.\n    '
    import xonsh.procs.specs
    return xonsh.procs.specs.run_subproc(cmds, captured='hiddenobject', envs=envs)

def subproc_uncaptured(*cmds, envs=None):
    if False:
        i = 10
        return i + 15
    'Runs a subprocess, without capturing the output. Returns the stdout\n    that was produced as a str.\n    '
    import xonsh.procs.specs
    return xonsh.procs.specs.run_subproc(cmds, captured=False, envs=envs)

def ensure_list_of_strs(x):
    if False:
        print('Hello World!')
    'Ensures that x is a list of strings.'
    if isinstance(x, str):
        rtn = [x]
    elif isinstance(x, cabc.Sequence):
        rtn = [i if isinstance(i, str) else str(i) for i in x]
    else:
        rtn = [str(x)]
    return rtn

def ensure_str_or_callable(x):
    if False:
        return 10
    'Ensures that x is single string or function.'
    if isinstance(x, str) or callable(x):
        return x
    if isinstance(x, bytes):
        return os.fsdecode(x)
    return str(x)

def list_of_strs_or_callables(x):
    if False:
        while True:
            i = 10
    "\n    Ensures that x is a list of strings or functions.\n    This is called when using the ``@()`` operator to expand it's content.\n    "
    if isinstance(x, (str, bytes)) or callable(x):
        rtn = [ensure_str_or_callable(x)]
    elif isinstance(x, cabc.Iterable):
        rtn = list(map(ensure_str_or_callable, x))
    else:
        rtn = [ensure_str_or_callable(x)]
    return rtn

def list_of_list_of_strs_outer_product(x):
    if False:
        while True:
            i = 10
    'Takes an outer product of a list of strings'
    lolos = map(ensure_list_of_strs, x)
    rtn = []
    for los in itertools.product(*lolos):
        s = ''.join(los)
        if '*' in s:
            rtn.extend(XSH.glob(s))
        else:
            rtn.append(XSH.expand_path(s))
    return rtn

def eval_fstring_field(field):
    if False:
        while True:
            i = 10
    'Evaluates the argument in Xonsh context.'
    res = XSH.execer.eval(field[0].strip(), glbs=globals(), locs=XSH.ctx, filename=field[1])
    return res

@lazyobject
def MACRO_FLAG_KINDS():
    if False:
        print('Hello World!')
    return {'s': str, 'str': str, 'string': str, 'a': AST, 'ast': AST, 'c': types.CodeType, 'code': types.CodeType, 'compile': types.CodeType, 'v': eval, 'eval': eval, 'x': exec, 'exec': exec, 't': type, 'type': type}

def _convert_kind_flag(x):
    if False:
        for i in range(10):
            print('nop')
    'Puts a kind flag (string) a canonical form.'
    x = x.lower()
    kind = MACRO_FLAG_KINDS.get(x, None)
    if kind is None:
        raise TypeError(f'{x!r} not a recognized macro type.')
    return kind

def convert_macro_arg(raw_arg, kind, glbs, locs, *, name='<arg>', macroname='<macro>'):
    if False:
        return 10
    'Converts a string macro argument based on the requested kind.\n\n    Parameters\n    ----------\n    raw_arg : str\n        The str representation of the macro argument.\n    kind : object\n        A flag or type representing how to convert the argument.\n    glbs : Mapping\n        The globals from the call site.\n    locs : Mapping or None\n        The locals from the call site.\n    name : str, optional\n        The macro argument name.\n    macroname : str, optional\n        The name of the macro itself.\n\n    Returns\n    -------\n    The converted argument.\n    '
    mode = None
    if isinstance(kind, cabc.Sequence) and (not isinstance(kind, str)):
        (kind, mode) = kind
    if isinstance(kind, str):
        kind = _convert_kind_flag(kind)
    if kind is str or kind is None:
        return raw_arg
    execer = XSH.execer
    filename = macroname + '(' + name + ')'
    if kind is AST:
        ctx = set(dir(builtins)) | set(glbs.keys())
        if locs is not None:
            ctx |= set(locs.keys())
        mode = mode or 'eval'
        if mode != 'eval' and (not raw_arg.endswith('\n')):
            raw_arg += '\n'
        arg = execer.parse(raw_arg, ctx, mode=mode, filename=filename)
    elif kind is types.CodeType or kind is compile:
        mode = mode or 'eval'
        arg = execer.compile(raw_arg, mode=mode, glbs=glbs, locs=locs, filename=filename)
    elif kind is eval:
        arg = execer.eval(raw_arg, glbs=glbs, locs=locs, filename=filename)
    elif kind is exec:
        mode = mode or 'exec'
        if not raw_arg.endswith('\n'):
            raw_arg += '\n'
        arg = execer.exec(raw_arg, mode=mode, glbs=glbs, locs=locs, filename=filename)
    elif kind is type:
        arg = type(execer.eval(raw_arg, glbs=glbs, locs=locs, filename=filename))
    else:
        msg = 'kind={0!r} and mode={1!r} was not recognized for macro argument {2!r}'
        raise TypeError(msg.format(kind, mode, name))
    return arg

@contextlib.contextmanager
def in_macro_call(f, glbs, locs):
    if False:
        print('Hello World!')
    'Attaches macro globals and locals temporarily to function as a\n    context manager.\n\n    Parameters\n    ----------\n    f : callable object\n        The function that is called as ``f(*args)``.\n    glbs : Mapping\n        The globals from the call site.\n    locs : Mapping or None\n        The locals from the call site.\n    '
    prev_glbs = getattr(f, 'macro_globals', None)
    prev_locs = getattr(f, 'macro_locals', None)
    f.macro_globals = glbs
    f.macro_locals = locs
    yield
    if prev_glbs is None:
        del f.macro_globals
    else:
        f.macro_globals = prev_glbs
    if prev_locs is None:
        del f.macro_locals
    else:
        f.macro_locals = prev_locs

def call_macro(f, raw_args, glbs, locs):
    if False:
        i = 10
        return i + 15
    'Calls a function as a macro, returning its result.\n\n    Parameters\n    ----------\n    f : callable object\n        The function that is called as ``f(*args)``.\n    raw_args : tuple of str\n        The str representation of arguments of that were passed into the\n        macro. These strings will be parsed, compiled, evaled, or left as\n        a string depending on the annotations of f.\n    glbs : Mapping\n        The globals from the call site.\n    locs : Mapping or None\n        The locals from the call site.\n    '
    sig = inspect.signature(f)
    empty = inspect.Parameter.empty
    macroname = f.__name__
    i = 0
    args = []
    for ((key, param), raw_arg) in zip(sig.parameters.items(), raw_args):
        i += 1
        if raw_arg == '*':
            break
        kind = param.annotation
        if kind is empty or kind is None:
            kind = str
        arg = convert_macro_arg(raw_arg, kind, glbs, locs, name=key, macroname=macroname)
        args.append(arg)
    (reg_args, kwargs) = _eval_regular_args(raw_args[i:], glbs, locs)
    args += reg_args
    with in_macro_call(f, glbs, locs):
        rtn = f(*args, **kwargs)
    return rtn

@lazyobject
def KWARG_RE():
    if False:
        return 10
    return re.compile('([A-Za-z_]\\w*=|\\*\\*)')

def _starts_as_arg(s):
    if False:
        for i in range(10):
            print('nop')
    'Tests if a string starts as a non-kwarg string would.'
    return KWARG_RE.match(s) is None

def _eval_regular_args(raw_args, glbs, locs):
    if False:
        return 10
    if not raw_args:
        return ([], {})
    arglist = list(itertools.takewhile(_starts_as_arg, raw_args))
    kwarglist = raw_args[len(arglist):]
    execer = XSH.execer
    if not arglist:
        args = arglist
        kwargstr = 'dict({})'.format(', '.join(kwarglist))
        kwargs = execer.eval(kwargstr, glbs=glbs, locs=locs)
    elif not kwarglist:
        argstr = '({},)'.format(', '.join(arglist))
        args = execer.eval(argstr, glbs=glbs, locs=locs)
        kwargs = {}
    else:
        argstr = '({},)'.format(', '.join(arglist))
        kwargstr = 'dict({})'.format(', '.join(kwarglist))
        both = f'({argstr}, {kwargstr})'
        (args, kwargs) = execer.eval(both, glbs=glbs, locs=locs)
    return (args, kwargs)

def enter_macro(obj, raw_block, glbs, locs):
    if False:
        for i in range(10):
            print('nop')
    'Prepares to enter a context manager macro by attaching the contents\n    of the macro block, globals, and locals to the object. These modifications\n    are made in-place and the original object is returned.\n\n    Parameters\n    ----------\n    obj : context manager\n        The object that is about to be entered via a with-statement.\n    raw_block : str\n        The str of the block that is the context body.\n        This string will be parsed, compiled, evaled, or left as\n        a string depending on the return annotation of obj.__enter__.\n    glbs : Mapping\n        The globals from the context site.\n    locs : Mapping or None\n        The locals from the context site.\n\n    Returns\n    -------\n    obj : context manager\n        The same context manager but with the new macro information applied.\n    '
    if isinstance(obj, cabc.Sequence):
        for x in obj:
            enter_macro(x, raw_block, glbs, locs)
        return obj
    kind = getattr(obj, '__xonsh_block__', str)
    macroname = getattr(obj, '__name__', '<context>')
    block = convert_macro_arg(raw_block, kind, glbs, locs, name='<with!>', macroname=macroname)
    obj.macro_globals = glbs
    obj.macro_locals = locs
    obj.macro_block = block
    return obj

@contextlib.contextmanager
def xonsh_builtins(execer=None):
    if False:
        for i in range(10):
            print('nop')
    'A context manager for using the xonsh builtins only in a limited\n    scope. Likely useful in testing.\n    '
    XSH.load(execer=execer)
    yield
    XSH.unload()

class XonshSession:
    """All components defining a xonsh session."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.execer = None
        self.ctx = {}
        self.builtins_loaded = False
        self.history = None
        self.shell = None
        self.env = None
        self.rc_files = None
        self.help = helper
        self.superhelp = superhelper
        self.pathsearch = pathsearch
        self.globsearch = globsearch
        self.regexsearch = regexsearch
        self.glob = globpath
        self.expand_path = expand_path
        self.subproc_captured_stdout = subproc_captured_stdout
        self.subproc_captured_inject = subproc_captured_inject
        self.subproc_captured_object = subproc_captured_object
        self.subproc_captured_hiddenobject = subproc_captured_hiddenobject
        self.subproc_uncaptured = subproc_uncaptured
        self.call_macro = call_macro
        self.enter_macro = enter_macro
        self.path_literal = path_literal
        self.list_of_strs_or_callables = list_of_strs_or_callables
        self.list_of_list_of_strs_outer_product = list_of_list_of_strs_outer_product
        self.eval_fstring_field = eval_fstring_field
        self.exit = None
        self.stdout_uncaptured = None
        self.stderr_uncaptured = None
        self._py_exit = None
        self._py_quit = None
        self.commands_cache = None
        self.modules_cache = None
        self.all_jobs = None
        self._completers = None
        self.builtins = None
        self._initial_builtin_names = None

    @property
    def aliases(self):
        if False:
            return 10
        if self.commands_cache is None:
            return
        return self.commands_cache.aliases

    @property
    def completers(self):
        if False:
            while True:
                i = 10
        'Returns a list of all available completers. Init when first accessing the attribute'
        if self._completers is None:
            from xonsh.completers.init import default_completers
            self._completers = default_completers(self.commands_cache)
        return self._completers

    def _disable_python_exit(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(builtins, 'exit'):
            self._py_exit = builtins.exit
            del builtins.exit
        if hasattr(builtins, 'quit'):
            self._py_quit = builtins.quit
            del builtins.quit

    def _restore_python_exit(self):
        if False:
            while True:
                i = 10
        if self._py_exit is not None:
            builtins.exit = self._py_exit
        if self._py_quit is not None:
            builtins.quit = self._py_quit

    def load(self, execer=None, ctx=None, **kwargs):
        if False:
            while True:
                i = 10
        'Loads the session with default values.\n\n        Parameters\n        ----------\n        execer : Execer, optional\n            Xonsh execution object, may be None to start\n        ctx : Mapping, optional\n            Context to start xonsh session with.\n        '
        from xonsh.commands_cache import CommandsCache
        from xonsh.environ import Env, default_env
        if not hasattr(builtins, '__xonsh__'):
            builtins.__xonsh__ = self
        if ctx is not None:
            self.ctx = ctx
        self.env = kwargs.pop('env') if 'env' in kwargs else Env(default_env())
        self.exit = False
        self.stdout_uncaptured = None
        self.stderr_uncaptured = None
        self._disable_python_exit()
        self.execer = execer
        self.modules_cache = {}
        self.all_jobs = {}
        self.builtins = get_default_builtins(execer)
        self._initial_builtin_names = frozenset(vars(self.builtins))
        aliases_given = kwargs.pop('aliases', None)
        for (attr, value) in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)
        self.commands_cache = kwargs.pop('commands_cache') if 'commands_cache' in kwargs else CommandsCache(self.env, aliases_given)
        self.link_builtins()
        self.builtins_loaded = True

        def flush_on_exit(s=None, f=None):
            if False:
                for i in range(10):
                    print('nop')
            if self.history is not None:
                self.history.flush(at_exit=True)
        atexit.register(flush_on_exit)
        for sig in AT_EXIT_SIGNALS:
            resetting_signal_handle(sig, flush_on_exit)

    def link_builtins(self):
        if False:
            while True:
                i = 10
        for refname in self._initial_builtin_names:
            objname = f'__xonsh__.builtins.{refname}'
            proxy = DynamicAccessProxy(refname, objname)
            setattr(builtins, refname, proxy)
        builtins.default_aliases = builtins.aliases = self.aliases

    def unlink_builtins(self):
        if False:
            return 10
        for name in self._initial_builtin_names:
            if hasattr(builtins, name):
                delattr(builtins, name)

    def unload(self):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(builtins, '__xonsh__'):
            self.builtins_loaded = False
            return
        if hasattr(self.env, 'undo_replace_env'):
            self.env.undo_replace_env()
        self._restore_python_exit()
        if not self.builtins_loaded:
            return
        self.unlink_builtins()
        delattr(builtins, '__xonsh__')
        self.builtins_loaded = False
        self._completers = None

def get_default_builtins(execer=None):
    if False:
        i = 10
        return i + 15
    from xonsh.events import events
    return types.SimpleNamespace(XonshError=XonshError, XonshCalledProcessError=XonshCalledProcessError, evalx=None if execer is None else execer.eval, execx=None if execer is None else execer.exec, compilex=None if execer is None else execer.compile, events=events, print_color=print_color, printx=print_color)

class DynamicAccessProxy:
    """Proxies access dynamically."""

    def __init__(self, refname, objname):
        if False:
            while True:
                i = 10
        "\n        Parameters\n        ----------\n        refname : str\n            '.'-separated string that represents the new, reference name that\n            the user will access.\n        objname : str\n            '.'-separated string that represents the name where the target\n            object actually lives that refname points to.\n        "
        super().__setattr__('refname', refname)
        super().__setattr__('objname', objname)

    @property
    def obj(self):
        if False:
            i = 10
            return i + 15
        'Dynamically grabs object'
        names = self.objname.split('.')
        obj = builtins
        for name in names:
            obj = getattr(obj, name)
        return obj

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        return getattr(self.obj, name)

    def __setattr__(self, name, value):
        if False:
            return 10
        return super().__setattr__(self.obj, name, value)

    def __delattr__(self, name):
        if False:
            i = 10
            return i + 15
        return delattr(self.obj, name)

    def __getitem__(self, item):
        if False:
            for i in range(10):
                print('nop')
        return self.obj.__getitem__(item)

    def __setitem__(self, item, value):
        if False:
            while True:
                i = 10
        return self.obj.__setitem__(item, value)

    def __delitem__(self, item):
        if False:
            print('Hello World!')
        del self.obj[item]

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        return self.obj.__call__(*args, **kwargs)

    def __dir__(self):
        if False:
            while True:
                i = 10
        return self.obj.__dir__()
XSH = XonshSession()