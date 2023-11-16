"""SCons.Action

This encapsulates information about executing any sort of action that
can build one or more target Nodes (typically files) from one or more
source Nodes (also typically files) given a specific Environment.

The base class here is ActionBase.  The base class supplies just a few
OO utility methods and some generic methods for displaying information
about an Action in response to the various commands that control printing.

A second-level base class is _ActionAction.  This extends ActionBase
by providing the methods that can be used to show and perform an
action.  True Action objects will subclass _ActionAction; Action
factory class objects will subclass ActionBase.

The heavy lifting is handled by subclasses for the different types of
actions we might execute:

    CommandAction
    CommandGeneratorAction
    FunctionAction
    ListAction

The subclasses supply the following public interface methods used by
other modules:

    __call__()
        THE public interface, "calling" an Action object executes the
        command or Python function.  This also takes care of printing
        a pre-substitution command for debugging purposes.

    get_contents()
        Fetches the "contents" of an Action for signature calculation
        plus the varlist.  This is what gets MD5 checksummed to decide
        if a target needs to be rebuilt because its action changed.

    genstring()
        Returns a string representation of the Action *without*
        command substitution, but allows a CommandGeneratorAction to
        generate the right action based on the specified target,
        source and env.  This is used by the Signature subsystem
        (through the Executor) to obtain an (imprecise) representation
        of the Action operation for informative purposes.


Subclasses also supply the following methods for internal use within
this module:

    __str__()
        Returns a string approximation of the Action; no variable
        substitution is performed.

    execute()
        The internal method that really, truly, actually handles the
        execution of a command or Python function.  This is used so
        that the __call__() methods can take care of displaying any
        pre-substitution representations, and *then* execute an action
        without worrying about the specific Actions involved.

    get_presig()
        Fetches the "contents" of a subclass for signature calculation.
        The varlist is added to this to produce the Action's contents.
        TODO(?): Change this to always return ascii/bytes and not unicode (or py3 strings)

    strfunction()
        Returns a substituted string representation of the Action.
        This is used by the _ActionAction.show() command to display the
        command/function that will be executed to generate the target(s).

There is a related independent ActionCaller class that looks like a
regular Action, and which serves as a wrapper for arbitrary functions
that we want to let the user specify the arguments to now, but actually
execute later (when an out-of-date check determines that it's needed to
be executed, for example).  Objects of this class are returned by an
ActionFactory class that provides a __call__() method as a convenient
way for wrapping up the functions.

"""
__revision__ = 'src/engine/SCons/Action.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import os
import pickle
import re
import sys
import subprocess
import itertools
import inspect
from collections import OrderedDict
import SCons.Debug
from SCons.Debug import logInstanceCreation
import SCons.Errors
import SCons.Util
import SCons.Subst
from SCons.Util import is_String, is_List

class _null(object):
    pass
print_actions = 1
execute_actions = 1
print_actions_presub = 0
ACTION_SIGNATURE_PICKLE_PROTOCOL = 1

def rfile(n):
    if False:
        i = 10
        return i + 15
    try:
        return n.rfile()
    except AttributeError:
        return n

def default_exitstatfunc(s):
    if False:
        while True:
            i = 10
    return s
strip_quotes = re.compile('^[\'"](.*)[\'"]$')

def _callable_contents(obj):
    if False:
        while True:
            i = 10
    'Return the signature contents of a callable Python object.\n    '
    try:
        return _function_contents(obj.__func__)
    except AttributeError:
        try:
            return _function_contents(obj.__call__.__func__)
        except AttributeError:
            try:
                return _code_contents(obj)
            except AttributeError:
                return _function_contents(obj)

def _object_contents(obj):
    if False:
        i = 10
        return i + 15
    'Return the signature contents of any Python object.\n\n    We have to handle the case where object contains a code object\n    since it can be pickled directly.\n    '
    try:
        return _function_contents(obj.__func__)
    except AttributeError:
        try:
            return _function_contents(obj.__call__.__func__)
        except AttributeError:
            try:
                return _code_contents(obj)
            except AttributeError:
                try:
                    return _function_contents(obj)
                except AttributeError as ae:
                    try:
                        return _object_instance_content(obj)
                    except (pickle.PicklingError, TypeError, AttributeError) as ex:
                        return bytearray(repr(obj), 'utf-8')

def _code_contents(code, docstring=None):
    if False:
        return 10
    'Return the signature contents of a code object.\n\n    By providing direct access to the code object of the\n    function, Python makes this extremely easy.  Hooray!\n\n    Unfortunately, older versions of Python include line\n    number indications in the compiled byte code.  Boo!\n    So we remove the line number byte codes to prevent\n    recompilations from moving a Python function.\n\n    See:\n      - https://docs.python.org/2/library/inspect.html\n      - http://python-reference.readthedocs.io/en/latest/docs/code/index.html\n\n    For info on what each co\\_ variable provides\n\n    The signature is as follows (should be byte/chars):\n    co_argcount, len(co_varnames), len(co_cellvars), len(co_freevars),\n    ( comma separated signature for each object in co_consts ),\n    ( comma separated signature for each object in co_names ),\n    ( The bytecode with line number bytecodes removed from  co_code )\n\n    co_argcount - Returns the number of positional arguments (including arguments with default values).\n    co_varnames - Returns a tuple containing the names of the local variables (starting with the argument names).\n    co_cellvars - Returns a tuple containing the names of local variables that are referenced by nested functions.\n    co_freevars - Returns a tuple containing the names of free variables. (?)\n    co_consts   - Returns a tuple containing the literals used by the bytecode.\n    co_names    - Returns a tuple containing the names used by the bytecode.\n    co_code     - Returns a string representing the sequence of bytecode instructions.\n\n    '
    contents = bytearray('{}, {}'.format(code.co_argcount, len(code.co_varnames)), 'utf-8')
    contents.extend(b', ')
    contents.extend(bytearray(str(len(code.co_cellvars)), 'utf-8'))
    contents.extend(b', ')
    contents.extend(bytearray(str(len(code.co_freevars)), 'utf-8'))
    z = [_object_contents(cc) for cc in code.co_consts if cc != docstring]
    contents.extend(b',(')
    contents.extend(bytearray(',', 'utf-8').join(z))
    contents.extend(b')')
    z = [bytearray(_object_contents(cc)) for cc in code.co_names]
    contents.extend(b',(')
    contents.extend(bytearray(',', 'utf-8').join(z))
    contents.extend(b')')
    contents.extend(b',(')
    contents.extend(code.co_code)
    contents.extend(b')')
    return contents

def _function_contents(func):
    if False:
        for i in range(10):
            print('nop')
    "\n    The signature is as follows (should be byte/chars):\n    < _code_contents (see above) from func.__code__ >\n    ,( comma separated _object_contents for function argument defaults)\n    ,( comma separated _object_contents for any closure contents )\n\n\n    See also: https://docs.python.org/3/reference/datamodel.html\n      - func.__code__     - The code object representing the compiled function body.\n      - func.__defaults__ - A tuple containing default argument values for those arguments that have defaults, or None if no arguments have a default value\n      - func.__closure__  - None or a tuple of cells that contain bindings for the function's free variables.\n\n    :Returns:\n      Signature contents of a function. (in bytes)\n    "
    contents = [_code_contents(func.__code__, func.__doc__)]
    if func.__defaults__:
        function_defaults_contents = [_object_contents(cc) for cc in func.__defaults__]
        defaults = bytearray(b',(')
        defaults.extend(bytearray(b',').join(function_defaults_contents))
        defaults.extend(b')')
        contents.append(defaults)
    else:
        contents.append(b',()')
    closure = func.__closure__ or []
    try:
        closure_contents = [_object_contents(x.cell_contents) for x in closure]
    except AttributeError:
        closure_contents = []
    contents.append(b',(')
    contents.append(bytearray(b',').join(closure_contents))
    contents.append(b')')
    retval = bytearray(b'').join(contents)
    return retval

def _object_instance_content(obj):
    if False:
        return 10
    '\n    Returns consistant content for a action class or an instance thereof\n\n    :Parameters:\n      - `obj` Should be either and action class or an instance thereof\n\n    :Returns:\n      bytearray or bytes representing the obj suitable for generating a signature from.\n    '
    retval = bytearray()
    if obj is None:
        return b'N.'
    if isinstance(obj, SCons.Util.BaseStringTypes):
        return SCons.Util.to_bytes(obj)
    inst_class = obj.__class__
    inst_class_name = bytearray(obj.__class__.__name__, 'utf-8')
    inst_class_module = bytearray(obj.__class__.__module__, 'utf-8')
    inst_class_hierarchy = bytearray(repr(inspect.getclasstree([obj.__class__])), 'utf-8')
    properties = [(p, getattr(obj, p, 'None')) for p in dir(obj) if not (p[:2] == '__' or inspect.ismethod(getattr(obj, p)) or inspect.isbuiltin(getattr(obj, p)))]
    properties.sort()
    properties_str = ','.join(['%s=%s' % (p[0], p[1]) for p in properties])
    properties_bytes = bytearray(properties_str, 'utf-8')
    methods = [p for p in dir(obj) if inspect.ismethod(getattr(obj, p))]
    methods.sort()
    method_contents = []
    for m in methods:
        v = _function_contents(getattr(obj, m))
        method_contents.append(v)
    retval = bytearray(b'{')
    retval.extend(inst_class_name)
    retval.extend(b':')
    retval.extend(inst_class_module)
    retval.extend(b'}[[')
    retval.extend(inst_class_hierarchy)
    retval.extend(b']]{{')
    retval.extend(bytearray(b',').join(method_contents))
    retval.extend(b'}}{{{')
    retval.extend(properties_bytes)
    retval.extend(b'}}}')
    return retval

def _actionAppend(act1, act2):
    if False:
        print('Hello World!')
    a1 = Action(act1)
    a2 = Action(act2)
    if a1 is None:
        return a2
    if a2 is None:
        return a1
    if isinstance(a1, ListAction):
        if isinstance(a2, ListAction):
            return ListAction(a1.list + a2.list)
        else:
            return ListAction(a1.list + [a2])
    elif isinstance(a2, ListAction):
        return ListAction([a1] + a2.list)
    else:
        return ListAction([a1, a2])

def _do_create_keywords(args, kw):
    if False:
        for i in range(10):
            print('nop')
    'This converts any arguments after the action argument into\n    their equivalent keywords and adds them to the kw argument.\n    '
    v = kw.get('varlist', ())
    if is_String(v):
        v = (v,)
    kw['varlist'] = tuple(v)
    if args:
        cmdstrfunc = args[0]
        if cmdstrfunc is None or is_String(cmdstrfunc):
            kw['cmdstr'] = cmdstrfunc
        elif callable(cmdstrfunc):
            kw['strfunction'] = cmdstrfunc
        else:
            raise SCons.Errors.UserError('Invalid command display variable type. You must either pass a string or a callback which accepts (target, source, env) as parameters.')
        if len(args) > 1:
            kw['varlist'] = tuple(SCons.Util.flatten(args[1:])) + kw['varlist']
    if kw.get('strfunction', _null) is not _null and kw.get('cmdstr', _null) is not _null:
        raise SCons.Errors.UserError('Cannot have both strfunction and cmdstr args to Action()')

def _do_create_action(act, kw):
    if False:
        print('Hello World!')
    'This is the actual "implementation" for the\n    Action factory method, below.  This handles the\n    fact that passing lists to Action() itself has\n    different semantics than passing lists as elements\n    of lists.\n\n    The former will create a ListAction, the latter\n    will create a CommandAction by converting the inner\n    list elements to strings.'
    if isinstance(act, ActionBase):
        return act
    if is_String(act):
        var = SCons.Util.get_environment_var(act)
        if var:
            return LazyAction(var, kw)
        commands = str(act).split('\n')
        if len(commands) == 1:
            return CommandAction(commands[0], **kw)
        return _do_create_list_action(commands, kw)
    if is_List(act):
        return CommandAction(act, **kw)
    if callable(act):
        try:
            gen = kw['generator']
            del kw['generator']
        except KeyError:
            gen = 0
        if gen:
            action_type = CommandGeneratorAction
        else:
            action_type = FunctionAction
        return action_type(act, kw)
    if isinstance(act, int) or isinstance(act, float):
        raise TypeError("Don't know how to create an Action from a number (%s)" % act)
    return None

def _do_create_list_action(act, kw):
    if False:
        return 10
    'A factory for list actions.  Convert the input list into Actions\n    and then wrap them in a ListAction.'
    acts = []
    for a in act:
        aa = _do_create_action(a, kw)
        if aa is not None:
            acts.append(aa)
    if not acts:
        return ListAction([])
    elif len(acts) == 1:
        return acts[0]
    else:
        return ListAction(acts)

def Action(act, *args, **kw):
    if False:
        while True:
            i = 10
    'A factory for action objects.'
    _do_create_keywords(args, kw)
    if is_List(act):
        return _do_create_list_action(act, kw)
    return _do_create_action(act, kw)

class ActionBase(object):
    """Base class for all types of action objects that can be held by
    other objects (Builders, Executors, etc.)  This provides the
    common methods for manipulating and combining those actions."""

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return self.__dict__ == other

    def no_batch_key(self, env, target, source):
        if False:
            for i in range(10):
                print('nop')
        return None
    batch_key = no_batch_key

    def genstring(self, target, source, env):
        if False:
            i = 10
            return i + 15
        return str(self)

    def get_contents(self, target, source, env):
        if False:
            print('Hello World!')
        result = self.get_presig(target, source, env)
        if not isinstance(result, (bytes, bytearray)):
            result = bytearray(result, 'utf-8')
        else:
            result = bytearray(result)
        vl = self.get_varlist(target, source, env)
        if is_String(vl):
            vl = (vl,)
        for v in vl:
            if isinstance(result, bytearray):
                result.extend(SCons.Util.to_bytes(env.subst_target_source('${' + v + '}', SCons.Subst.SUBST_SIG, target, source)))
            else:
                raise Exception('WE SHOULD NEVER GET HERE result should be bytearray not:%s' % type(result))
        if isinstance(result, (bytes, bytearray)):
            return result
        else:
            raise Exception('WE SHOULD NEVER GET HERE - #2 result should be bytearray not:%s' % type(result))

    def __add__(self, other):
        if False:
            return 10
        return _actionAppend(self, other)

    def __radd__(self, other):
        if False:
            i = 10
            return i + 15
        return _actionAppend(other, self)

    def presub_lines(self, env):
        if False:
            print('Hello World!')
        self.presub_env = env
        lines = str(self).split('\n')
        self.presub_env = None
        return lines

    def get_varlist(self, target, source, env, executor=None):
        if False:
            while True:
                i = 10
        return self.varlist

    def get_targets(self, env, executor):
        if False:
            return 10
        '\n        Returns the type of targets ($TARGETS, $CHANGED_TARGETS) used\n        by this action.\n        '
        return self.targets

class _ActionAction(ActionBase):
    """Base class for actions that create output objects."""

    def __init__(self, cmdstr=_null, strfunction=_null, varlist=(), presub=_null, chdir=None, exitstatfunc=None, batch_key=None, targets='$TARGETS', **kw):
        if False:
            for i in range(10):
                print('nop')
        self.cmdstr = cmdstr
        if strfunction is not _null:
            if strfunction is None:
                self.cmdstr = None
            else:
                self.strfunction = strfunction
        self.varlist = varlist
        self.presub = presub
        self.chdir = chdir
        if not exitstatfunc:
            exitstatfunc = default_exitstatfunc
        self.exitstatfunc = exitstatfunc
        self.targets = targets
        if batch_key:
            if not callable(batch_key):

                def default_batch_key(self, env, target, source):
                    if False:
                        return 10
                    return (id(self), id(env))
                batch_key = default_batch_key
            SCons.Util.AddMethod(self, batch_key, 'batch_key')

    def print_cmd_line(self, s, target, source, env):
        if False:
            print('Hello World!')
        "\n        In python 3, and in some of our tests, sys.stdout is\n        a String io object, and it takes unicode strings only\n        In other cases it's a regular Python 2.x file object\n        which takes strings (bytes), and if you pass those a\n        unicode object they try to decode with 'ascii' codec\n        which fails if the cmd line has any hi-bit-set chars.\n        This code assumes s is a regular string, but should\n        work if it's unicode too.\n        "
        try:
            sys.stdout.write(s + u'\n')
        except UnicodeDecodeError:
            sys.stdout.write(s + '\n')

    def __call__(self, target, source, env, exitstatfunc=_null, presub=_null, show=_null, execute=_null, chdir=_null, executor=None):
        if False:
            return 10
        if not is_List(target):
            target = [target]
        if not is_List(source):
            source = [source]
        if presub is _null:
            presub = self.presub
            if presub is _null:
                presub = print_actions_presub
        if exitstatfunc is _null:
            exitstatfunc = self.exitstatfunc
        if show is _null:
            show = print_actions
        if execute is _null:
            execute = execute_actions
        if chdir is _null:
            chdir = self.chdir
        save_cwd = None
        if chdir:
            save_cwd = os.getcwd()
            try:
                chdir = str(chdir.get_abspath())
            except AttributeError:
                if not is_String(chdir):
                    if executor:
                        chdir = str(executor.batches[0].targets[0].dir)
                    else:
                        chdir = str(target[0].dir)
        if presub:
            if executor:
                target = executor.get_all_targets()
                source = executor.get_all_sources()
            t = ' and '.join(map(str, target))
            l = '\n  '.join(self.presub_lines(env))
            out = u'Building %s with action:\n  %s\n' % (t, l)
            sys.stdout.write(out)
        cmd = None
        if show and self.strfunction:
            if executor:
                target = executor.get_all_targets()
                source = executor.get_all_sources()
            try:
                cmd = self.strfunction(target, source, env, executor)
            except TypeError:
                cmd = self.strfunction(target, source, env)
            if cmd:
                if chdir:
                    cmd = 'os.chdir(%s)\n' % repr(chdir) + cmd
                try:
                    get = env.get
                except AttributeError:
                    print_func = self.print_cmd_line
                else:
                    print_func = get('PRINT_CMD_LINE_FUNC')
                    if not print_func:
                        print_func = self.print_cmd_line
                print_func(cmd, target, source, env)
        stat = 0
        if execute:
            if chdir:
                os.chdir(chdir)
            try:
                stat = self.execute(target, source, env, executor=executor)
                if isinstance(stat, SCons.Errors.BuildError):
                    s = exitstatfunc(stat.status)
                    if s:
                        stat.status = s
                    else:
                        stat = s
                else:
                    stat = exitstatfunc(stat)
            finally:
                if save_cwd:
                    os.chdir(save_cwd)
        if cmd and save_cwd:
            print_func('os.chdir(%s)' % repr(save_cwd), target, source, env)
        return stat

def _string_from_cmd_list(cmd_list):
    if False:
        print('Hello World!')
    'Takes a list of command line arguments and returns a pretty\n    representation for printing.'
    cl = []
    for arg in map(str, cmd_list):
        if ' ' in arg or '\t' in arg:
            arg = '"' + arg + '"'
        cl.append(arg)
    return ' '.join(cl)
default_ENV = None

def get_default_ENV(env):
    if False:
        return 10
    "\n    A fiddlin' little function that has an 'import SCons.Environment' which\n    can't be moved to the top level without creating an import loop.  Since\n    this import creates a local variable named 'SCons', it blocks access to\n    the global variable, so we move it here to prevent complaints about local\n    variables being used uninitialized.\n    "
    global default_ENV
    try:
        return env['ENV']
    except KeyError:
        if not default_ENV:
            import SCons.Environment
            default_ENV = SCons.Environment.Environment()['ENV']
        return default_ENV

def _subproc(scons_env, cmd, error='ignore', **kw):
    if False:
        print('Hello World!')
    "Do common setup for a subprocess.Popen() call\n\n    This function is still in draft mode.  We're going to need something like\n    it in the long run as more and more places use subprocess, but I'm sure\n    it'll have to be tweaked to get the full desired functionality.\n    one special arg (so far?), 'error', to tell what to do with exceptions.\n    "
    try:
        from subprocess import DEVNULL
    except ImportError:
        DEVNULL = None
    for stream in ('stdin', 'stdout', 'stderr'):
        io = kw.get(stream)
        if is_String(io) and io == 'devnull':
            if DEVNULL:
                kw[stream] = DEVNULL
            else:
                kw[stream] = open(os.devnull, 'r+')
    ENV = kw.get('env', None)
    if ENV is None:
        ENV = get_default_ENV(scons_env)
    new_env = {}
    for (key, value) in ENV.items():
        if is_List(value):
            value = SCons.Util.flatten_sequence(value)
            new_env[key] = os.pathsep.join(map(str, value))
        else:
            new_env[key] = str(value)
    kw['env'] = new_env
    try:
        pobj = subprocess.Popen(cmd, **kw)
    except EnvironmentError as e:
        if error == 'raise':
            raise

        class dummyPopen(object):

            def __init__(self, e):
                if False:
                    for i in range(10):
                        print('nop')
                self.exception = e

            def communicate(self, input=None):
                if False:
                    for i in range(10):
                        print('nop')
                return ('', '')

            def wait(self):
                if False:
                    while True:
                        i = 10
                return -self.exception.errno
            stdin = None

            class f(object):

                def read(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    return ''

                def readline(self):
                    if False:
                        print('Hello World!')
                    return ''

                def __iter__(self):
                    if False:
                        while True:
                            i = 10
                    return iter(())
            stdout = stderr = f()
        pobj = dummyPopen(e)
    finally:
        for (k, v) in kw.items():
            if inspect.ismethod(getattr(v, 'close', None)):
                v.close()
    return pobj

class CommandAction(_ActionAction):
    """Class for command-execution actions."""

    def __init__(self, cmd, **kw):
        if False:
            print('Hello World!')
        if SCons.Debug.track_instances:
            logInstanceCreation(self, 'Action.CommandAction')
        _ActionAction.__init__(self, **kw)
        if is_List(cmd):
            if [c for c in cmd if is_List(c)]:
                raise TypeError('CommandAction should be given only a single command')
        self.cmd_list = cmd

    def __str__(self):
        if False:
            while True:
                i = 10
        if is_List(self.cmd_list):
            return ' '.join(map(str, self.cmd_list))
        return str(self.cmd_list)

    def process(self, target, source, env, executor=None):
        if False:
            return 10
        if executor:
            result = env.subst_list(self.cmd_list, 0, executor=executor)
        else:
            result = env.subst_list(self.cmd_list, 0, target, source)
        silent = None
        ignore = None
        while True:
            try:
                c = result[0][0][0]
            except IndexError:
                c = None
            if c == '@':
                silent = 1
            elif c == '-':
                ignore = 1
            else:
                break
            result[0][0] = result[0][0][1:]
        try:
            if not result[0][0]:
                result[0] = result[0][1:]
        except IndexError:
            pass
        return (result, ignore, silent)

    def strfunction(self, target, source, env, executor=None):
        if False:
            i = 10
            return i + 15
        if self.cmdstr is None:
            return None
        if self.cmdstr is not _null:
            from SCons.Subst import SUBST_RAW
            if executor:
                c = env.subst(self.cmdstr, SUBST_RAW, executor=executor)
            else:
                c = env.subst(self.cmdstr, SUBST_RAW, target, source)
            if c:
                return c
        (cmd_list, ignore, silent) = self.process(target, source, env, executor)
        if silent:
            return ''
        return _string_from_cmd_list(cmd_list[0])

    def execute(self, target, source, env, executor=None):
        if False:
            print('Hello World!')
        'Execute a command action.\n\n        This will handle lists of commands as well as individual commands,\n        because construction variable substitution may turn a single\n        "command" into a list.  This means that this class can actually\n        handle lists of commands, even though that\'s not how we use it\n        externally.\n        '
        escape_list = SCons.Subst.escape_list
        flatten_sequence = SCons.Util.flatten_sequence
        try:
            shell = env['SHELL']
        except KeyError:
            raise SCons.Errors.UserError('Missing SHELL construction variable.')
        try:
            spawn = env['SPAWN']
        except KeyError:
            raise SCons.Errors.UserError('Missing SPAWN construction variable.')
        else:
            if is_String(spawn):
                spawn = env.subst(spawn, raw=1, conv=lambda x: x)
        escape = env.get('ESCAPE', lambda x: x)
        ENV = get_default_ENV(env)
        for (key, value) in ENV.items():
            if not is_String(value):
                if is_List(value):
                    value = flatten_sequence(value)
                    ENV[key] = os.pathsep.join(map(str, value))
                else:
                    ENV[key] = str(value)
        if executor:
            target = executor.get_all_targets()
            source = executor.get_all_sources()
        (cmd_list, ignore, silent) = self.process(target, list(map(rfile, source)), env, executor)
        for cmd_line in filter(len, cmd_list):
            cmd_line = escape_list(cmd_line, escape)
            result = spawn(shell, escape, cmd_line[0], cmd_line, ENV)
            if not ignore and result:
                msg = 'Error %s' % result
                return SCons.Errors.BuildError(errstr=msg, status=result, action=self, command=cmd_line)
        return 0

    def get_presig(self, target, source, env, executor=None):
        if False:
            print('Hello World!')
        "Return the signature contents of this action's command line.\n\n        This strips $(-$) and everything in between the string,\n        since those parts don't affect signatures.\n        "
        from SCons.Subst import SUBST_SIG
        cmd = self.cmd_list
        if is_List(cmd):
            cmd = ' '.join(map(str, cmd))
        else:
            cmd = str(cmd)
        if executor:
            return env.subst_target_source(cmd, SUBST_SIG, executor=executor)
        else:
            return env.subst_target_source(cmd, SUBST_SIG, target, source)

    def get_implicit_deps(self, target, source, env, executor=None):
        if False:
            for i in range(10):
                print('nop')
        icd = env.get('IMPLICIT_COMMAND_DEPENDENCIES', True)
        if is_String(icd) and icd[:1] == '$':
            icd = env.subst(icd)
        if not icd or icd in ('0', 'None'):
            return []
        from SCons.Subst import SUBST_SIG
        if executor:
            cmd_list = env.subst_list(self.cmd_list, SUBST_SIG, executor=executor)
        else:
            cmd_list = env.subst_list(self.cmd_list, SUBST_SIG, target, source)
        res = []
        for cmd_line in cmd_list:
            if cmd_line:
                d = str(cmd_line[0])
                m = strip_quotes.match(d)
                if m:
                    d = m.group(1)
                d = env.WhereIs(d)
                if d:
                    res.append(env.fs.File(d))
        return res

class CommandGeneratorAction(ActionBase):
    """Class for command-generator actions."""

    def __init__(self, generator, kw):
        if False:
            print('Hello World!')
        if SCons.Debug.track_instances:
            logInstanceCreation(self, 'Action.CommandGeneratorAction')
        self.generator = generator
        self.gen_kw = kw
        self.varlist = kw.get('varlist', ())
        self.targets = kw.get('targets', '$TARGETS')

    def _generate(self, target, source, env, for_signature, executor=None):
        if False:
            for i in range(10):
                print('nop')
        if not is_List(target):
            target = [target]
        if executor:
            target = executor.get_all_targets()
            source = executor.get_all_sources()
        ret = self.generator(target=target, source=source, env=env, for_signature=for_signature)
        gen_cmd = Action(ret, **self.gen_kw)
        if not gen_cmd:
            raise SCons.Errors.UserError('Object returned from command generator: %s cannot be used to create an Action.' % repr(ret))
        return gen_cmd

    def __str__(self):
        if False:
            return 10
        try:
            env = self.presub_env
        except AttributeError:
            env = None
        if env is None:
            env = SCons.Defaults.DefaultEnvironment()
        act = self._generate([], [], env, 1)
        return str(act)

    def batch_key(self, env, target, source):
        if False:
            while True:
                i = 10
        return self._generate(target, source, env, 1).batch_key(env, target, source)

    def genstring(self, target, source, env, executor=None):
        if False:
            for i in range(10):
                print('nop')
        return self._generate(target, source, env, 1, executor).genstring(target, source, env)

    def __call__(self, target, source, env, exitstatfunc=_null, presub=_null, show=_null, execute=_null, chdir=_null, executor=None):
        if False:
            i = 10
            return i + 15
        act = self._generate(target, source, env, 0, executor)
        if act is None:
            raise SCons.Errors.UserError("While building `%s': Cannot deduce file extension from source files: %s" % (repr(list(map(str, target))), repr(list(map(str, source)))))
        return act(target, source, env, exitstatfunc, presub, show, execute, chdir, executor)

    def get_presig(self, target, source, env, executor=None):
        if False:
            while True:
                i = 10
        "Return the signature contents of this action's command line.\n\n        This strips $(-$) and everything in between the string,\n        since those parts don't affect signatures.\n        "
        return self._generate(target, source, env, 1, executor).get_presig(target, source, env)

    def get_implicit_deps(self, target, source, env, executor=None):
        if False:
            while True:
                i = 10
        return self._generate(target, source, env, 1, executor).get_implicit_deps(target, source, env)

    def get_varlist(self, target, source, env, executor=None):
        if False:
            i = 10
            return i + 15
        return self._generate(target, source, env, 1, executor).get_varlist(target, source, env, executor)

    def get_targets(self, env, executor):
        if False:
            for i in range(10):
                print('nop')
        return self._generate(None, None, env, 1, executor).get_targets(env, executor)

class LazyAction(CommandGeneratorAction, CommandAction):
    """
    A LazyAction is a kind of hybrid generator and command action for
    strings of the form "$VAR".  These strings normally expand to other
    strings (think "$CCCOM" to "$CC -c -o $TARGET $SOURCE"), but we also
    want to be able to replace them with functions in the construction
    environment.  Consequently, we want lazy evaluation and creation of
    an Action in the case of the function, but that's overkill in the more
    normal case of expansion to other strings.

    So we do this with a subclass that's both a generator *and*
    a command action.  The overridden methods all do a quick check
    of the construction variable, and if it's a string we just call
    the corresponding CommandAction method to do the heavy lifting.
    If not, then we call the same-named CommandGeneratorAction method.
    The CommandGeneratorAction methods work by using the overridden
    _generate() method, that is, our own way of handling "generation" of
    an action based on what's in the construction variable.
    """

    def __init__(self, var, kw):
        if False:
            i = 10
            return i + 15
        if SCons.Debug.track_instances:
            logInstanceCreation(self, 'Action.LazyAction')
        CommandAction.__init__(self, '${' + var + '}', **kw)
        self.var = SCons.Util.to_String(var)
        self.gen_kw = kw

    def get_parent_class(self, env):
        if False:
            for i in range(10):
                print('nop')
        c = env.get(self.var)
        if is_String(c) and '\n' not in c:
            return CommandAction
        return CommandGeneratorAction

    def _generate_cache(self, env):
        if False:
            while True:
                i = 10
        if env:
            c = env.get(self.var, '')
        else:
            c = ''
        gen_cmd = Action(c, **self.gen_kw)
        if not gen_cmd:
            raise SCons.Errors.UserError('$%s value %s cannot be used to create an Action.' % (self.var, repr(c)))
        return gen_cmd

    def _generate(self, target, source, env, for_signature, executor=None):
        if False:
            for i in range(10):
                print('nop')
        return self._generate_cache(env)

    def __call__(self, target, source, env, *args, **kw):
        if False:
            return 10
        c = self.get_parent_class(env)
        return c.__call__(self, target, source, env, *args, **kw)

    def get_presig(self, target, source, env):
        if False:
            print('Hello World!')
        c = self.get_parent_class(env)
        return c.get_presig(self, target, source, env)

    def get_varlist(self, target, source, env, executor=None):
        if False:
            i = 10
            return i + 15
        c = self.get_parent_class(env)
        return c.get_varlist(self, target, source, env, executor)

class FunctionAction(_ActionAction):
    """Class for Python function actions."""

    def __init__(self, execfunction, kw):
        if False:
            for i in range(10):
                print('nop')
        if SCons.Debug.track_instances:
            logInstanceCreation(self, 'Action.FunctionAction')
        self.execfunction = execfunction
        try:
            self.funccontents = _callable_contents(execfunction)
        except AttributeError:
            try:
                self.gc = execfunction.get_contents
            except AttributeError:
                self.funccontents = _object_contents(execfunction)
        _ActionAction.__init__(self, **kw)

    def function_name(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.execfunction.__name__
        except AttributeError:
            try:
                return self.execfunction.__class__.__name__
            except AttributeError:
                return 'unknown_python_function'

    def strfunction(self, target, source, env, executor=None):
        if False:
            print('Hello World!')
        if self.cmdstr is None:
            return None
        if self.cmdstr is not _null:
            from SCons.Subst import SUBST_RAW
            if executor:
                c = env.subst(self.cmdstr, SUBST_RAW, executor=executor)
            else:
                c = env.subst(self.cmdstr, SUBST_RAW, target, source)
            if c:
                return c

        def array(a):
            if False:
                for i in range(10):
                    print('nop')

            def quote(s):
                if False:
                    i = 10
                    return i + 15
                try:
                    str_for_display = s.str_for_display
                except AttributeError:
                    s = repr(s)
                else:
                    s = str_for_display()
                return s
            return '[' + ', '.join(map(quote, a)) + ']'
        try:
            strfunc = self.execfunction.strfunction
        except AttributeError:
            pass
        else:
            if strfunc is None:
                return None
            if callable(strfunc):
                return strfunc(target, source, env)
        name = self.function_name()
        tstr = array(target)
        sstr = array(source)
        return '%s(%s, %s)' % (name, tstr, sstr)

    def __str__(self):
        if False:
            while True:
                i = 10
        name = self.function_name()
        if name == 'ActionCaller':
            return str(self.execfunction)
        return '%s(target, source, env)' % name

    def execute(self, target, source, env, executor=None):
        if False:
            while True:
                i = 10
        exc_info = (None, None, None)
        try:
            if executor:
                target = executor.get_all_targets()
                source = executor.get_all_sources()
            rsources = list(map(rfile, source))
            try:
                result = self.execfunction(target=target, source=rsources, env=env)
            except KeyboardInterrupt as e:
                raise
            except SystemExit as e:
                raise
            except Exception as e:
                result = e
                exc_info = sys.exc_info()
            if result:
                result = SCons.Errors.convert_to_BuildError(result, exc_info)
                result.node = target
                result.action = self
                try:
                    result.command = self.strfunction(target, source, env, executor)
                except TypeError:
                    result.command = self.strfunction(target, source, env)
                if exc_info[1] and (not isinstance(exc_info[1], EnvironmentError)):
                    raise result
            return result
        finally:
            del exc_info

    def get_presig(self, target, source, env):
        if False:
            for i in range(10):
                print('nop')
        'Return the signature contents of this callable action.'
        try:
            return self.gc(target, source, env)
        except AttributeError:
            return self.funccontents

    def get_implicit_deps(self, target, source, env):
        if False:
            print('Hello World!')
        return []

class ListAction(ActionBase):
    """Class for lists of other actions."""

    def __init__(self, actionlist):
        if False:
            return 10
        if SCons.Debug.track_instances:
            logInstanceCreation(self, 'Action.ListAction')

        def list_of_actions(x):
            if False:
                return 10
            if isinstance(x, ActionBase):
                return x
            return Action(x)
        self.list = list(map(list_of_actions, actionlist))
        self.varlist = ()
        self.targets = '$TARGETS'

    def genstring(self, target, source, env):
        if False:
            print('Hello World!')
        return '\n'.join([a.genstring(target, source, env) for a in self.list])

    def __str__(self):
        if False:
            return 10
        return '\n'.join(map(str, self.list))

    def presub_lines(self, env):
        if False:
            for i in range(10):
                print('nop')
        return SCons.Util.flatten_sequence([a.presub_lines(env) for a in self.list])

    def get_presig(self, target, source, env):
        if False:
            return 10
        'Return the signature contents of this action list.\n\n        Simple concatenation of the signatures of the elements.\n        '
        return b''.join([bytes(x.get_contents(target, source, env)) for x in self.list])

    def __call__(self, target, source, env, exitstatfunc=_null, presub=_null, show=_null, execute=_null, chdir=_null, executor=None):
        if False:
            return 10
        if executor:
            target = executor.get_all_targets()
            source = executor.get_all_sources()
        for act in self.list:
            stat = act(target, source, env, exitstatfunc, presub, show, execute, chdir, executor)
            if stat:
                return stat
        return 0

    def get_implicit_deps(self, target, source, env):
        if False:
            print('Hello World!')
        result = []
        for act in self.list:
            result.extend(act.get_implicit_deps(target, source, env))
        return result

    def get_varlist(self, target, source, env, executor=None):
        if False:
            i = 10
            return i + 15
        result = OrderedDict()
        for act in self.list:
            for var in act.get_varlist(target, source, env, executor):
                result[var] = True
        return list(result.keys())

class ActionCaller(object):
    """A class for delaying calling an Action function with specific
    (positional and keyword) arguments until the Action is actually
    executed.

    This class looks to the rest of the world like a normal Action object,
    but what it's really doing is hanging on to the arguments until we
    have a target, source and env to use for the expansion.
    """

    def __init__(self, parent, args, kw):
        if False:
            for i in range(10):
                print('nop')
        self.parent = parent
        self.args = args
        self.kw = kw

    def get_contents(self, target, source, env):
        if False:
            i = 10
            return i + 15
        actfunc = self.parent.actfunc
        try:
            contents = actfunc.__code__.co_code
        except AttributeError:
            try:
                contents = actfunc.__call__.__func__.__code__.co_code
            except AttributeError:
                contents = repr(actfunc)
        return contents

    def subst(self, s, target, source, env):
        if False:
            print('Hello World!')
        if is_List(s):
            result = []
            for elem in s:
                result.append(self.subst(elem, target, source, env))
            return self.parent.convert(result)
        if s == '$__env__':
            return env
        elif is_String(s):
            return env.subst(s, 1, target, source)
        return self.parent.convert(s)

    def subst_args(self, target, source, env):
        if False:
            return 10
        return [self.subst(x, target, source, env) for x in self.args]

    def subst_kw(self, target, source, env):
        if False:
            return 10
        kw = {}
        for key in list(self.kw.keys()):
            kw[key] = self.subst(self.kw[key], target, source, env)
        return kw

    def __call__(self, target, source, env, executor=None):
        if False:
            print('Hello World!')
        args = self.subst_args(target, source, env)
        kw = self.subst_kw(target, source, env)
        return self.parent.actfunc(*args, **kw)

    def strfunction(self, target, source, env):
        if False:
            i = 10
            return i + 15
        args = self.subst_args(target, source, env)
        kw = self.subst_kw(target, source, env)
        return self.parent.strfunc(*args, **kw)

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.parent.strfunc(*self.args, **self.kw)

class ActionFactory(object):
    """A factory class that will wrap up an arbitrary function
    as an SCons-executable Action object.

    The real heavy lifting here is done by the ActionCaller class.
    We just collect the (positional and keyword) arguments that we're
    called with and give them to the ActionCaller object we create,
    so it can hang onto them until it needs them.
    """

    def __init__(self, actfunc, strfunc, convert=lambda x: x):
        if False:
            return 10
        self.actfunc = actfunc
        self.strfunc = strfunc
        self.convert = convert

    def __call__(self, *args, **kw):
        if False:
            while True:
                i = 10
        ac = ActionCaller(self, args, kw)
        action = Action(ac, strfunction=ac.strfunction)
        return action