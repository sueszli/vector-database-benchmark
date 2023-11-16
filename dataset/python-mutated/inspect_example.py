"""Get useful information from live Python objects.

This module encapsulates the interface provided by the internal special
attributes (co_*, im_*, tb_*, etc.) in a friendlier fashion.
It also provides some help for examining source code and class layout.

Here are some of the useful functions provided by this module:

    ismodule(), isclass(), ismethod(), isfunction(), isgeneratorfunction(),
        isgenerator(), istraceback(), isframe(), iscode(), isbuiltin(),
        isroutine() - check object types
    getmembers() - get members of an object that satisfy a given condition

    getfile(), getsourcefile(), getsource() - find an object's source code
    getdoc(), getcomments() - get documentation on an object
    getmodule() - determine the module that an object came from
    getclasstree() - arrange classes so as to represent their hierarchy

    getargspec(), getargvalues(), getcallargs() - get info about function arguments
    getfullargspec() - same, with support for Python-3000 features
    formatargspec(), formatargvalues() - format an argument spec
    getouterframes(), getinnerframes() - get info about frames
    currentframe() - get the current stack frame
    stack(), trace() - get info about frames on the stack or in a traceback

    signature() - get a Signature object for the callable
"""
__author__ = ('Ka-Ping Yee <ping@lfw.org>', 'Yury Selivanov <yselivanov@sprymix.com>')
import ast
import importlib.machinery
import itertools
import linecache
import os
import re
import sys
import tokenize
import token
import types
import warnings
import functools
import builtins
from operator import attrgetter
from collections import namedtuple, OrderedDict
try:
    from dis import COMPILER_FLAG_NAMES as _flag_names
except ImportError:
    (CO_OPTIMIZED, CO_NEWLOCALS) = (1, 2)
    (CO_VARARGS, CO_VARKEYWORDS) = (4, 8)
    (CO_NESTED, CO_GENERATOR, CO_NOFREE) = (16, 32, 64)
else:
    mod_dict = globals()
    for (k, v) in _flag_names.items():
        mod_dict['CO_' + v] = k
TPFLAGS_IS_ABSTRACT = 1 << 20

def ismodule(object):
    if False:
        for i in range(10):
            print('nop')
    'Return true if the object is a module.\n\n    Module objects provide these attributes:\n        __cached__      pathname to byte compiled file\n        __doc__         documentation string\n        __file__        filename (missing for built-in modules)'
    return isinstance(object, types.ModuleType)

def isclass(object):
    if False:
        return 10
    'Return true if the object is a class.\n\n    Class objects provide these attributes:\n        __doc__         documentation string\n        __module__      name of module in which this class was defined'
    return isinstance(object, type)

def ismethod(object):
    if False:
        print('Hello World!')
    'Return true if the object is an instance method.\n\n    Instance method objects provide these attributes:\n        __doc__         documentation string\n        __name__        name with which this method was defined\n        __func__        function object containing implementation of method\n        __self__        instance to which this method is bound'
    return isinstance(object, types.MethodType)

def ismethoddescriptor(object):
    if False:
        while True:
            i = 10
    'Return true if the object is a method descriptor.\n\n    But not if ismethod() or isclass() or isfunction() are true.\n\n    This is new in Python 2.2, and, for example, is true of int.__add__.\n    An object passing this test has a __get__ attribute but not a __set__\n    attribute, but beyond that the set of attributes varies.  __name__ is\n    usually sensible, and __doc__ often is.\n\n    Methods implemented via descriptors that also pass one of the other\n    tests return false from the ismethoddescriptor() test, simply because\n    the other tests promise more -- you can, e.g., count on having the\n    __func__ attribute (etc) when an object passes ismethod().'
    if isclass(object) or ismethod(object) or isfunction(object):
        return False
    tp = type(object)
    return hasattr(tp, '__get__') and (not hasattr(tp, '__set__'))

def isdatadescriptor(object):
    if False:
        i = 10
        return i + 15
    'Return true if the object is a data descriptor.\n\n    Data descriptors have both a __get__ and a __set__ attribute.  Examples are\n    properties (defined in Python) and getsets and members (defined in C).\n    Typically, data descriptors will also have __name__ and __doc__ attributes\n    (properties, getsets, and members have both of these attributes), but this\n    is not guaranteed.'
    if isclass(object) or ismethod(object) or isfunction(object):
        return False
    tp = type(object)
    return hasattr(tp, '__set__') and hasattr(tp, '__get__')
if hasattr(types, 'MemberDescriptorType'):

    def ismemberdescriptor(object):
        if False:
            i = 10
            return i + 15
        'Return true if the object is a member descriptor.\n\n        Member descriptors are specialized descriptors defined in extension\n        modules.'
        return isinstance(object, types.MemberDescriptorType)
else:

    def ismemberdescriptor(object):
        if False:
            while True:
                i = 10
        'Return true if the object is a member descriptor.\n\n        Member descriptors are specialized descriptors defined in extension\n        modules.'
        return False
if hasattr(types, 'GetSetDescriptorType'):

    def isgetsetdescriptor(object):
        if False:
            return 10
        'Return true if the object is a getset descriptor.\n\n        getset descriptors are specialized descriptors defined in extension\n        modules.'
        return isinstance(object, types.GetSetDescriptorType)
else:

    def isgetsetdescriptor(object):
        if False:
            for i in range(10):
                print('nop')
        'Return true if the object is a getset descriptor.\n\n        getset descriptors are specialized descriptors defined in extension\n        modules.'
        return False

def isfunction(object):
    if False:
        while True:
            i = 10
    'Return true if the object is a user-defined function.\n\n    Function objects provide these attributes:\n        __doc__         documentation string\n        __name__        name with which this function was defined\n        __code__        code object containing compiled function bytecode\n        __defaults__    tuple of any default values for arguments\n        __globals__     global namespace in which this function was defined\n        __annotations__ dict of parameter annotations\n        __kwdefaults__  dict of keyword only parameters with defaults'
    return isinstance(object, types.FunctionType)

def isgeneratorfunction(object):
    if False:
        return 10
    'Return true if the object is a user-defined generator function.\n\n    Generator function objects provides same attributes as functions.\n\n    See help(isfunction) for attributes listing.'
    return bool((isfunction(object) or ismethod(object)) and object.__code__.co_flags & CO_GENERATOR)

def isgenerator(object):
    if False:
        i = 10
        return i + 15
    'Return true if the object is a generator.\n\n    Generator objects provide these attributes:\n        __iter__        defined to support iteration over container\n        close           raises a new GeneratorExit exception inside the\n                        generator to terminate the iteration\n        gi_code         code object\n        gi_frame        frame object or possibly None once the generator has\n                        been exhausted\n        gi_running      set to 1 when generator is executing, 0 otherwise\n        next            return the next item from the container\n        send            resumes the generator and "sends" a value that becomes\n                        the result of the current yield-expression\n        throw           used to raise an exception inside the generator'
    return isinstance(object, types.GeneratorType)

def istraceback(object):
    if False:
        print('Hello World!')
    'Return true if the object is a traceback.\n\n    Traceback objects provide these attributes:\n        tb_frame        frame object at this level\n        tb_lasti        index of last attempted instruction in bytecode\n        tb_lineno       current line number in Python source code\n        tb_next         next inner traceback object (called by this level)'
    return isinstance(object, types.TracebackType)

def isframe(object):
    if False:
        while True:
            i = 10
    "Return true if the object is a frame object.\n\n    Frame objects provide these attributes:\n        f_back          next outer frame object (this frame's caller)\n        f_builtins      built-in namespace seen by this frame\n        f_code          code object being executed in this frame\n        f_globals       global namespace seen by this frame\n        f_lasti         index of last attempted instruction in bytecode\n        f_lineno        current line number in Python source code\n        f_locals        local namespace seen by this frame\n        f_trace         tracing function for this frame, or None"
    return isinstance(object, types.FrameType)

def iscode(object):
    if False:
        for i in range(10):
            print('nop')
    'Return true if the object is a code object.\n\n    Code objects provide these attributes:\n        co_argcount     number of arguments (not including * or ** args)\n        co_code         string of raw compiled bytecode\n        co_consts       tuple of constants used in the bytecode\n        co_filename     name of file in which this code object was created\n        co_firstlineno  number of first line in Python source code\n        co_flags        bitmap: 1=optimized | 2=newlocals | 4=*arg | 8=**arg\n        co_lnotab       encoded mapping of line numbers to bytecode indices\n        co_name         name with which this code object was defined\n        co_names        tuple of names of local variables\n        co_nlocals      number of local variables\n        co_stacksize    virtual machine stack space required\n        co_varnames     tuple of names of arguments and local variables'
    return isinstance(object, types.CodeType)

def isbuiltin(object):
    if False:
        while True:
            i = 10
    'Return true if the object is a built-in function or method.\n\n    Built-in functions and methods provide these attributes:\n        __doc__         documentation string\n        __name__        original name of this function or method\n        __self__        instance to which a method is bound, or None'
    return isinstance(object, types.BuiltinFunctionType)

def isroutine(object):
    if False:
        return 10
    'Return true if the object is any kind of function or method.'
    return isbuiltin(object) or isfunction(object) or ismethod(object) or ismethoddescriptor(object)

def isabstract(object):
    if False:
        return 10
    'Return true if the object is an abstract base class (ABC).'
    return bool(isinstance(object, type) and object.__flags__ & TPFLAGS_IS_ABSTRACT)

def getmembers(object, predicate=None):
    if False:
        print('Hello World!')
    'Return all members of an object as (name, value) pairs sorted by name.\n    Optionally, only return members that satisfy a given predicate.'
    if isclass(object):
        mro = (object,) + getmro(object)
    else:
        mro = ()
    results = []
    processed = set()
    names = dir(object)
    try:
        for base in object.__bases__:
            for (k, v) in base.__dict__.items():
                if isinstance(v, types.DynamicClassAttribute):
                    names.append(k)
    except AttributeError:
        pass
    for key in names:
        try:
            value = getattr(object, key)
            if key in processed:
                raise AttributeError
        except AttributeError:
            for base in mro:
                if key in base.__dict__:
                    value = base.__dict__[key]
                    break
            else:
                continue
        if not predicate or predicate(value):
            results.append((key, value))
        processed.add(key)
    results.sort(key=lambda pair: pair[0])
    return results
Attribute = namedtuple('Attribute', 'name kind defining_class object')

def classify_class_attrs(cls):
    if False:
        for i in range(10):
            print('nop')
    "Return list of attribute-descriptor tuples.\n\n    For each name in dir(cls), the return list contains a 4-tuple\n    with these elements:\n\n        0. The name (a string).\n\n        1. The kind of attribute this is, one of these strings:\n               'class method'    created via classmethod()\n               'static method'   created via staticmethod()\n               'property'        created via property()\n               'method'          any other flavor of method or descriptor\n               'data'            not a method\n\n        2. The class which defined this attribute (a class).\n\n        3. The object as obtained by calling getattr; if this fails, or if the\n           resulting object does not live anywhere in the class' mro (including\n           metaclasses) then the object is looked up in the defining class's\n           dict (found by walking the mro).\n\n    If one of the items in dir(cls) is stored in the metaclass it will now\n    be discovered and not have None be listed as the class in which it was\n    defined.  Any items whose home class cannot be discovered are skipped.\n    "
    mro = getmro(cls)
    metamro = getmro(type(cls))
    metamro = tuple([cls for cls in metamro if cls not in (type, object)])
    class_bases = (cls,) + mro
    all_bases = class_bases + metamro
    names = dir(cls)
    for base in mro:
        for (k, v) in base.__dict__.items():
            if isinstance(v, types.DynamicClassAttribute):
                names.append(k)
    result = []
    processed = set()
    for name in names:
        homecls = None
        get_obj = None
        dict_obj = None
        if name not in processed:
            try:
                if name == '__dict__':
                    raise Exception("__dict__ is special, don't want the proxy")
                get_obj = getattr(cls, name)
            except Exception as exc:
                pass
            else:
                homecls = getattr(get_obj, '__objclass__', homecls)
                if homecls not in class_bases:
                    homecls = None
                    last_cls = None
                    for srch_cls in class_bases:
                        srch_obj = getattr(srch_cls, name, None)
                        if srch_obj == get_obj:
                            last_cls = srch_cls
                    for srch_cls in metamro:
                        try:
                            srch_obj = srch_cls.__getattr__(cls, name)
                        except AttributeError:
                            continue
                        if srch_obj == get_obj:
                            last_cls = srch_cls
                    if last_cls is not None:
                        homecls = last_cls
        for base in all_bases:
            if name in base.__dict__:
                dict_obj = base.__dict__[name]
                if homecls not in metamro:
                    homecls = base
                break
        if homecls is None:
            continue
        obj = get_obj or dict_obj
        if isinstance(dict_obj, staticmethod):
            kind = 'static method'
            obj = dict_obj
        elif isinstance(dict_obj, classmethod):
            kind = 'class method'
            obj = dict_obj
        elif isinstance(dict_obj, property):
            kind = 'property'
            obj = dict_obj
        elif isroutine(obj):
            kind = 'method'
        else:
            kind = 'data'
        result.append(Attribute(name, kind, homecls, obj))
        processed.add(name)
    return result

def getmro(cls):
    if False:
        while True:
            i = 10
    'Return tuple of base classes (including cls) in method resolution order.'
    return cls.__mro__

def unwrap(func, *, stop=None):
    if False:
        while True:
            i = 10
    'Get the object wrapped by *func*.\n\n   Follows the chain of :attr:`__wrapped__` attributes returning the last\n   object in the chain.\n\n   *stop* is an optional callback accepting an object in the wrapper chain\n   as its sole argument that allows the unwrapping to be terminated early if\n   the callback returns a true value. If the callback never returns a true\n   value, the last object in the chain is returned as usual. For example,\n   :func:`signature` uses this to stop unwrapping if any object in the\n   chain has a ``__signature__`` attribute defined.\n\n   :exc:`ValueError` is raised if a cycle is encountered.\n\n    '
    if stop is None:

        def _is_wrapper(f):
            if False:
                print('Hello World!')
            return hasattr(f, '__wrapped__')
    else:

        def _is_wrapper(f):
            if False:
                i = 10
                return i + 15
            return hasattr(f, '__wrapped__') and (not stop(f))
    f = func
    memo = {id(f)}
    while _is_wrapper(func):
        func = func.__wrapped__
        id_func = id(func)
        if id_func in memo:
            raise ValueError('wrapper loop when unwrapping {!r}'.format(f))
        memo.add(id_func)
    return func

def indentsize(line):
    if False:
        for i in range(10):
            print('nop')
    'Return the indent size, in spaces, at the start of a line of text.'
    expline = line.expandtabs()
    return len(expline) - len(expline.lstrip())

def getdoc(object):
    if False:
        i = 10
        return i + 15
    'Get the documentation string for an object.\n\n    All tabs are expanded to spaces.  To clean up docstrings that are\n    indented to line up with blocks of code, any whitespace than can be\n    uniformly removed from the second line onwards is removed.'
    try:
        doc = object.__doc__
    except AttributeError:
        return None
    if not isinstance(doc, str):
        return None
    return cleandoc(doc)

def cleandoc(doc):
    if False:
        print('Hello World!')
    'Clean up indentation from docstrings.\n\n    Any whitespace that can be uniformly removed from the second line\n    onwards is removed.'
    try:
        lines = doc.expandtabs().split('\n')
    except UnicodeError:
        return None
    else:
        margin = sys.maxsize
        for line in lines[1:]:
            content = len(line.lstrip())
            if content:
                indent = len(line) - content
                margin = min(margin, indent)
        if lines:
            lines[0] = lines[0].lstrip()
        if margin < sys.maxsize:
            for i in range(1, len(lines)):
                lines[i] = lines[i][margin:]
        while lines and (not lines[-1]):
            lines.pop()
        while lines and (not lines[0]):
            lines.pop(0)
        return '\n'.join(lines)

def getfile(object):
    if False:
        while True:
            i = 10
    'Work out which source or compiled file an object was defined in.'
    if ismodule(object):
        if hasattr(object, '__file__'):
            return object.__file__
        raise TypeError('{!r} is a built-in module'.format(object))
    if isclass(object):
        if hasattr(object, '__module__'):
            object = sys.modules.get(object.__module__)
            if hasattr(object, '__file__'):
                return object.__file__
        raise TypeError('{!r} is a built-in class'.format(object))
    if ismethod(object):
        object = object.__func__
    if isfunction(object):
        object = object.__code__
    if istraceback(object):
        object = object.tb_frame
    if isframe(object):
        object = object.f_code
    if iscode(object):
        return object.co_filename
    raise TypeError('{!r} is not a module, class, method, function, traceback, frame, or code object'.format(object))
ModuleInfo = namedtuple('ModuleInfo', 'name suffix mode module_type')

def getmodulename(path):
    if False:
        i = 10
        return i + 15
    'Return the module name for a given file, or None.'
    fname = os.path.basename(path)
    suffixes = [(-len(suffix), suffix) for suffix in importlib.machinery.all_suffixes()]
    suffixes.sort()
    for (neglen, suffix) in suffixes:
        if fname.endswith(suffix):
            return fname[:neglen]
    return None

def getsourcefile(object):
    if False:
        print('Hello World!')
    "Return the filename that can be used to locate an object's source.\n    Return None if no way can be identified to get the source.\n    "
    filename = getfile(object)
    all_bytecode_suffixes = importlib.machinery.DEBUG_BYTECODE_SUFFIXES[:]
    all_bytecode_suffixes += importlib.machinery.OPTIMIZED_BYTECODE_SUFFIXES[:]
    if any((filename.endswith(s) for s in all_bytecode_suffixes)):
        filename = os.path.splitext(filename)[0] + importlib.machinery.SOURCE_SUFFIXES[0]
    elif any((filename.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES)):
        return None
    if os.path.exists(filename):
        return filename
    if getattr(getmodule(object, filename), '__loader__', None) is not None:
        return filename
    if filename in linecache.cache:
        return filename

def getabsfile(object, _filename=None):
    if False:
        print('Hello World!')
    'Return an absolute path to the source or compiled file for an object.\n\n    The idea is for each object to have a unique origin, so this routine\n    normalizes the result as much as possible.'
    if _filename is None:
        _filename = getsourcefile(object) or getfile(object)
    return os.path.normcase(os.path.abspath(_filename))
modulesbyfile = {}
_filesbymodname = {}

def getmodule(object, _filename=None):
    if False:
        while True:
            i = 10
    'Return the module an object was defined in, or None if not found.'
    if ismodule(object):
        return object
    if hasattr(object, '__module__'):
        return sys.modules.get(object.__module__)
    if _filename is not None and _filename in modulesbyfile:
        return sys.modules.get(modulesbyfile[_filename])
    try:
        file = getabsfile(object, _filename)
    except TypeError:
        return None
    if file in modulesbyfile:
        return sys.modules.get(modulesbyfile[file])
    for (modname, module) in list(sys.modules.items()):
        if ismodule(module) and hasattr(module, '__file__'):
            f = module.__file__
            if f == _filesbymodname.get(modname, None):
                continue
            _filesbymodname[modname] = f
            f = getabsfile(module)
            modulesbyfile[f] = modulesbyfile[os.path.realpath(f)] = module.__name__
    if file in modulesbyfile:
        return sys.modules.get(modulesbyfile[file])
    main = sys.modules['__main__']
    if not hasattr(object, '__name__'):
        return None
    if hasattr(main, object.__name__):
        mainobject = getattr(main, object.__name__)
        if mainobject is object:
            return main
    builtin = sys.modules['builtins']
    if hasattr(builtin, object.__name__):
        builtinobject = getattr(builtin, object.__name__)
        if builtinobject is object:
            return builtin

def findsource(object):
    if False:
        return 10
    'Return the entire source file and starting line number for an object.\n\n    The argument may be a module, class, method, function, traceback, frame,\n    or code object.  The source code is returned as a list of all the lines\n    in the file and the line number indexes a line in that list.  An OSError\n    is raised if the source code cannot be retrieved.'
    file = getsourcefile(object)
    if file:
        linecache.checkcache(file)
    else:
        file = getfile(object)
        if not (file.startswith('<') and file.endswith('>')):
            raise OSError('source code not available')
    module = getmodule(object, file)
    if module:
        lines = linecache.getlines(file, module.__dict__)
    else:
        lines = linecache.getlines(file)
    if not lines:
        raise OSError('could not get source code')
    if ismodule(object):
        return (lines, 0)
    if isclass(object):
        name = object.__name__
        pat = re.compile('^(\\s*)class\\s*' + name + '\\b')
        candidates = []
        for i in range(len(lines)):
            match = pat.match(lines[i])
            if match:
                if lines[i][0] == 'c':
                    return (lines, i)
                candidates.append((match.group(1), i))
        if candidates:
            candidates.sort()
            return (lines, candidates[0][1])
        else:
            raise OSError('could not find class definition')
    if ismethod(object):
        object = object.__func__
    if isfunction(object):
        object = object.__code__
    if istraceback(object):
        object = object.tb_frame
    if isframe(object):
        object = object.f_code
    if iscode(object):
        if not hasattr(object, 'co_firstlineno'):
            raise OSError('could not find function definition')
        lnum = object.co_firstlineno - 1
        pat = re.compile('^(\\s*def\\s)|(.*(?<!\\w)lambda(:|\\s))|^(\\s*@)')
        while lnum > 0:
            if pat.match(lines[lnum]):
                break
            lnum = lnum - 1
        return (lines, lnum)
    raise OSError('could not find code object')

def getcomments(object):
    if False:
        while True:
            i = 10
    "Get lines of comments immediately preceding an object's source code.\n\n    Returns None when source can't be found.\n    "
    try:
        (lines, lnum) = findsource(object)
    except (OSError, TypeError):
        return None
    if ismodule(object):
        start = 0
        if lines and lines[0][:2] == '#!':
            start = 1
        while start < len(lines) and lines[start].strip() in ('', '#'):
            start = start + 1
        if start < len(lines) and lines[start][:1] == '#':
            comments = []
            end = start
            while end < len(lines) and lines[end][:1] == '#':
                comments.append(lines[end].expandtabs())
                end = end + 1
            return ''.join(comments)
    elif lnum > 0:
        indent = indentsize(lines[lnum])
        end = lnum - 1
        if end >= 0 and lines[end].lstrip()[:1] == '#' and (indentsize(lines[end]) == indent):
            comments = [lines[end].expandtabs().lstrip()]
            if end > 0:
                end = end - 1
                comment = lines[end].expandtabs().lstrip()
                while comment[:1] == '#' and indentsize(lines[end]) == indent:
                    comments[:0] = [comment]
                    end = end - 1
                    if end < 0:
                        break
                    comment = lines[end].expandtabs().lstrip()
            while comments and comments[0].strip() == '#':
                comments[:1] = []
            while comments and comments[-1].strip() == '#':
                comments[-1:] = []
            return ''.join(comments)

class EndOfBlock(Exception):
    pass

class BlockFinder:
    """Provide a tokeneater() method to detect the end of a code block."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.indent = 0
        self.islambda = False
        self.started = False
        self.passline = False
        self.last = 1

    def tokeneater(self, type, token, srowcol, erowcol, line):
        if False:
            print('Hello World!')
        if not self.started:
            if token in ('def', 'class', 'lambda'):
                if token == 'lambda':
                    self.islambda = True
                self.started = True
            self.passline = True
        elif type == tokenize.NEWLINE:
            self.passline = False
            self.last = srowcol[0]
            if self.islambda:
                raise EndOfBlock
        elif self.passline:
            pass
        elif type == tokenize.INDENT:
            self.indent = self.indent + 1
            self.passline = True
        elif type == tokenize.DEDENT:
            self.indent = self.indent - 1
            if self.indent <= 0:
                raise EndOfBlock
        elif self.indent == 0 and type not in (tokenize.COMMENT, tokenize.NL):
            raise EndOfBlock

def getblock(lines):
    if False:
        for i in range(10):
            print('nop')
    'Extract the block of code at the top of the given list of lines.'
    blockfinder = BlockFinder()
    try:
        tokens = tokenize.generate_tokens(iter(lines).__next__)
        for _token in tokens:
            blockfinder.tokeneater(*_token)
    except (EndOfBlock, IndentationError):
        pass
    return lines[:blockfinder.last]

def getsourcelines(object):
    if False:
        for i in range(10):
            print('nop')
    'Return a list of source lines and starting line number for an object.\n\n    The argument may be a module, class, method, function, traceback, frame,\n    or code object.  The source code is returned as a list of the lines\n    corresponding to the object and the line number indicates where in the\n    original source file the first line of code was found.  An OSError is\n    raised if the source code cannot be retrieved.'
    (lines, lnum) = findsource(object)
    if ismodule(object):
        return (lines, 0)
    else:
        return (getblock(lines[lnum:]), lnum + 1)

def getsource(object):
    if False:
        return 10
    'Return the text of the source code for an object.\n\n    The argument may be a module, class, method, function, traceback, frame,\n    or code object.  The source code is returned as a single string.  An\n    OSError is raised if the source code cannot be retrieved.'
    (lines, lnum) = getsourcelines(object)
    return ''.join(lines)

def walktree(classes, children, parent):
    if False:
        i = 10
        return i + 15
    'Recursive helper function for getclasstree().'
    results = []
    classes.sort(key=attrgetter('__module__', '__name__'))
    for c in classes:
        results.append((c, c.__bases__))
        if c in children:
            results.append(walktree(children[c], children, c))
    return results

def getclasstree(classes, unique=False):
    if False:
        while True:
            i = 10
    "Arrange the given list of classes into a hierarchy of nested lists.\n\n    Where a nested list appears, it contains classes derived from the class\n    whose entry immediately precedes the list.  Each entry is a 2-tuple\n    containing a class and a tuple of its base classes.  If the 'unique'\n    argument is true, exactly one entry appears in the returned structure\n    for each class in the given list.  Otherwise, classes using multiple\n    inheritance and their descendants will appear multiple times."
    children = {}
    roots = []
    for c in classes:
        if c.__bases__:
            for parent in c.__bases__:
                if not parent in children:
                    children[parent] = []
                if c not in children[parent]:
                    children[parent].append(c)
                if unique and parent in classes:
                    break
        elif c not in roots:
            roots.append(c)
    for parent in children:
        if parent not in classes:
            roots.append(parent)
    return walktree(roots, children, None)
Arguments = namedtuple('Arguments', 'args, varargs, varkw')

def getargs(co):
    if False:
        print('Hello World!')
    "Get information about the arguments accepted by a code object.\n\n    Three things are returned: (args, varargs, varkw), where\n    'args' is the list of argument names. Keyword-only arguments are\n    appended. 'varargs' and 'varkw' are the names of the * and **\n    arguments or None."
    (args, varargs, kwonlyargs, varkw) = _getfullargs(co)
    return Arguments(args + kwonlyargs, varargs, varkw)

def _getfullargs(co):
    if False:
        i = 10
        return i + 15
    "Get information about the arguments accepted by a code object.\n\n    Four things are returned: (args, varargs, kwonlyargs, varkw), where\n    'args' and 'kwonlyargs' are lists of argument names, and 'varargs'\n    and 'varkw' are the names of the * and ** arguments or None."
    if not iscode(co):
        raise TypeError('{!r} is not a code object'.format(co))
    nargs = co.co_argcount
    names = co.co_varnames
    nkwargs = co.co_kwonlyargcount
    args = list(names[:nargs])
    kwonlyargs = list(names[nargs:nargs + nkwargs])
    step = 0
    nargs += nkwargs
    varargs = None
    if co.co_flags & CO_VARARGS:
        varargs = co.co_varnames[nargs]
        nargs = nargs + 1
    varkw = None
    if co.co_flags & CO_VARKEYWORDS:
        varkw = co.co_varnames[nargs]
    return (args, varargs, kwonlyargs, varkw)
ArgSpec = namedtuple('ArgSpec', 'args varargs keywords defaults')

def getargspec(func):
    if False:
        print('Hello World!')
    "Get the names and default values of a function's arguments.\n\n    A tuple of four things is returned: (args, varargs, varkw, defaults).\n    'args' is a list of the argument names.\n    'args' will include keyword-only argument names.\n    'varargs' and 'varkw' are the names of the * and ** arguments or None.\n    'defaults' is an n-tuple of the default values of the last n arguments.\n\n    Use the getfullargspec() API for Python-3000 code, as annotations\n    and keyword arguments are supported. getargspec() will raise ValueError\n    if the func has either annotations or keyword arguments.\n    "
    (args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, ann) = getfullargspec(func)
    if kwonlyargs or ann:
        raise ValueError('Function has keyword-only arguments or annotations, use getfullargspec() API which can support them')
    return ArgSpec(args, varargs, varkw, defaults)
FullArgSpec = namedtuple('FullArgSpec', 'args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations')

def getfullargspec(func):
    if False:
        i = 10
        return i + 15
    "Get the names and default values of a callable object's arguments.\n\n    A tuple of seven things is returned:\n    (args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults annotations).\n    'args' is a list of the argument names.\n    'varargs' and 'varkw' are the names of the * and ** arguments or None.\n    'defaults' is an n-tuple of the default values of the last n arguments.\n    'kwonlyargs' is a list of keyword-only argument names.\n    'kwonlydefaults' is a dictionary mapping names from kwonlyargs to defaults.\n    'annotations' is a dictionary mapping argument names to annotations.\n\n    The first four items in the tuple correspond to getargspec().\n    "
    try:
        sig = _signature_internal(func, follow_wrapper_chains=False, skip_bound_arg=False)
    except Exception as ex:
        raise TypeError('unsupported callable') from ex
    args = []
    varargs = None
    varkw = None
    kwonlyargs = []
    defaults = ()
    annotations = {}
    defaults = ()
    kwdefaults = {}
    if sig.return_annotation is not sig.empty:
        annotations['return'] = sig.return_annotation
    for param in sig.parameters.values():
        kind = param.kind
        name = param.name
        if kind is _POSITIONAL_ONLY:
            args.append(name)
        elif kind is _POSITIONAL_OR_KEYWORD:
            args.append(name)
            if param.default is not param.empty:
                defaults += (param.default,)
        elif kind is _VAR_POSITIONAL:
            varargs = name
        elif kind is _KEYWORD_ONLY:
            kwonlyargs.append(name)
            if param.default is not param.empty:
                kwdefaults[name] = param.default
        elif kind is _VAR_KEYWORD:
            varkw = name
        if param.annotation is not param.empty:
            annotations[name] = param.annotation
    if not kwdefaults:
        kwdefaults = None
    if not defaults:
        defaults = None
    return FullArgSpec(args, varargs, varkw, defaults, kwonlyargs, kwdefaults, annotations)
ArgInfo = namedtuple('ArgInfo', 'args varargs keywords locals')

def getargvalues(frame):
    if False:
        return 10
    "Get information about arguments passed into a particular frame.\n\n    A tuple of four things is returned: (args, varargs, varkw, locals).\n    'args' is a list of the argument names.\n    'varargs' and 'varkw' are the names of the * and ** arguments or None.\n    'locals' is the locals dictionary of the given frame."
    (args, varargs, varkw) = getargs(frame.f_code)
    return ArgInfo(args, varargs, varkw, frame.f_locals)

def formatannotation(annotation, base_module=None):
    if False:
        print('Hello World!')
    if isinstance(annotation, type):
        if annotation.__module__ in ('builtins', base_module):
            return annotation.__name__
        return annotation.__module__ + '.' + annotation.__name__
    return repr(annotation)

def formatannotationrelativeto(object):
    if False:
        i = 10
        return i + 15
    module = getattr(object, '__module__', None)

    def _formatannotation(annotation):
        if False:
            i = 10
            return i + 15
        return formatannotation(annotation, module)
    return _formatannotation

def formatargspec(args, varargs=None, varkw=None, defaults=None, kwonlyargs=(), kwonlydefaults={}, annotations={}, formatarg=str, formatvarargs=lambda name: '*' + name, formatvarkw=lambda name: '**' + name, formatvalue=lambda value: '=' + repr(value), formatreturns=lambda text: ' -> ' + text, formatannotation=formatannotation):
    if False:
        while True:
            i = 10
    'Format an argument spec from the values returned by getargspec\n    or getfullargspec.\n\n    The first seven arguments are (args, varargs, varkw, defaults,\n    kwonlyargs, kwonlydefaults, annotations).  The other five arguments\n    are the corresponding optional formatting functions that are called to\n    turn names and values into strings.  The last argument is an optional\n    function to format the sequence of arguments.'

    def formatargandannotation(arg):
        if False:
            print('Hello World!')
        result = formatarg(arg)
        if arg in annotations:
            result += ': ' + formatannotation(annotations[arg])
        return result
    specs = []
    if defaults:
        firstdefault = len(args) - len(defaults)
    for (i, arg) in enumerate(args):
        spec = formatargandannotation(arg)
        if defaults and i >= firstdefault:
            spec = spec + formatvalue(defaults[i - firstdefault])
        specs.append(spec)
    if varargs is not None:
        specs.append(formatvarargs(formatargandannotation(varargs)))
    elif kwonlyargs:
        specs.append('*')
    if kwonlyargs:
        for kwonlyarg in kwonlyargs:
            spec = formatargandannotation(kwonlyarg)
            if kwonlydefaults and kwonlyarg in kwonlydefaults:
                spec += formatvalue(kwonlydefaults[kwonlyarg])
            specs.append(spec)
    if varkw is not None:
        specs.append(formatvarkw(formatargandannotation(varkw)))
    result = '(' + ', '.join(specs) + ')'
    if 'return' in annotations:
        result += formatreturns(formatannotation(annotations['return']))
    return result

def formatargvalues(args, varargs, varkw, locals, formatarg=str, formatvarargs=lambda name: '*' + name, formatvarkw=lambda name: '**' + name, formatvalue=lambda value: '=' + repr(value)):
    if False:
        while True:
            i = 10
    'Format an argument spec from the 4 values returned by getargvalues.\n\n    The first four arguments are (args, varargs, varkw, locals).  The\n    next four arguments are the corresponding optional formatting functions\n    that are called to turn names and values into strings.  The ninth\n    argument is an optional function to format the sequence of arguments.'

    def convert(name, locals=locals, formatarg=formatarg, formatvalue=formatvalue):
        if False:
            while True:
                i = 10
        return formatarg(name) + formatvalue(locals[name])
    specs = []
    for i in range(len(args)):
        specs.append(convert(args[i]))
    if varargs:
        specs.append(formatvarargs(varargs) + formatvalue(locals[varargs]))
    if varkw:
        specs.append(formatvarkw(varkw) + formatvalue(locals[varkw]))
    return '(' + ', '.join(specs) + ')'

def _missing_arguments(f_name, argnames, pos, values):
    if False:
        for i in range(10):
            print('nop')
    names = [repr(name) for name in argnames if name not in values]
    missing = len(names)
    if missing == 1:
        s = names[0]
    elif missing == 2:
        s = '{} and {}'.format(*names)
    else:
        tail = ', {} and {}'.format(*names[-2:])
        del names[-2:]
        s = ', '.join(names) + tail
    raise TypeError('%s() missing %i required %s argument%s: %s' % (f_name, missing, 'positional' if pos else 'keyword-only', '' if missing == 1 else 's', s))

def _too_many(f_name, args, kwonly, varargs, defcount, given, values):
    if False:
        i = 10
        return i + 15
    atleast = len(args) - defcount
    kwonly_given = len([arg for arg in kwonly if arg in values])
    if varargs:
        plural = atleast != 1
        sig = 'at least %d' % (atleast,)
    elif defcount:
        plural = True
        sig = 'from %d to %d' % (atleast, len(args))
    else:
        plural = len(args) != 1
        sig = str(len(args))
    kwonly_sig = ''
    if kwonly_given:
        msg = ' positional argument%s (and %d keyword-only argument%s)'
        kwonly_sig = msg % ('s' if given != 1 else '', kwonly_given, 's' if kwonly_given != 1 else '')
    raise TypeError('%s() takes %s positional argument%s but %d%s %s given' % (f_name, sig, 's' if plural else '', given, kwonly_sig, 'was' if given == 1 and (not kwonly_given) else 'were'))

def getcallargs(*func_and_positional, **named):
    if False:
        while True:
            i = 10
    "Get the mapping of arguments to values.\n\n    A dict is returned, with keys the function argument names (including the\n    names of the * and ** arguments, if any), and values the respective bound\n    values from 'positional' and 'named'."
    func = func_and_positional[0]
    positional = func_and_positional[1:]
    spec = getfullargspec(func)
    (args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, ann) = spec
    f_name = func.__name__
    arg2value = {}
    if ismethod(func) and func.__self__ is not None:
        positional = (func.__self__,) + positional
    num_pos = len(positional)
    num_args = len(args)
    num_defaults = len(defaults) if defaults else 0
    n = min(num_pos, num_args)
    for i in range(n):
        arg2value[args[i]] = positional[i]
    if varargs:
        arg2value[varargs] = tuple(positional[n:])
    possible_kwargs = set(args + kwonlyargs)
    if varkw:
        arg2value[varkw] = {}
    for (kw, value) in named.items():
        if kw not in possible_kwargs:
            if not varkw:
                raise TypeError('%s() got an unexpected keyword argument %r' % (f_name, kw))
            arg2value[varkw][kw] = value
            continue
        if kw in arg2value:
            raise TypeError('%s() got multiple values for argument %r' % (f_name, kw))
        arg2value[kw] = value
    if num_pos > num_args and (not varargs):
        _too_many(f_name, args, kwonlyargs, varargs, num_defaults, num_pos, arg2value)
    if num_pos < num_args:
        req = args[:num_args - num_defaults]
        for arg in req:
            if arg not in arg2value:
                _missing_arguments(f_name, req, True, arg2value)
        for (i, arg) in enumerate(args[num_args - num_defaults:]):
            if arg not in arg2value:
                arg2value[arg] = defaults[i]
    missing = 0
    for kwarg in kwonlyargs:
        if kwarg not in arg2value:
            if kwonlydefaults and kwarg in kwonlydefaults:
                arg2value[kwarg] = kwonlydefaults[kwarg]
            else:
                missing += 1
    if missing:
        _missing_arguments(f_name, kwonlyargs, False, arg2value)
    return arg2value
ClosureVars = namedtuple('ClosureVars', 'nonlocals globals builtins unbound')

def getclosurevars(func):
    if False:
        i = 10
        return i + 15
    '\n    Get the mapping of free variables to their current values.\n\n    Returns a named tuple of dicts mapping the current nonlocal, global\n    and builtin references as seen by the body of the function. A final\n    set of unbound names that could not be resolved is also provided.\n    '
    if ismethod(func):
        func = func.__func__
    if not isfunction(func):
        raise TypeError("'{!r}' is not a Python function".format(func))
    code = func.__code__
    if func.__closure__ is None:
        nonlocal_vars = {}
    else:
        nonlocal_vars = {var: cell.cell_contents for (var, cell) in zip(code.co_freevars, func.__closure__)}
    global_ns = func.__globals__
    builtin_ns = global_ns.get('__builtins__', builtins.__dict__)
    if ismodule(builtin_ns):
        builtin_ns = builtin_ns.__dict__
    global_vars = {}
    builtin_vars = {}
    unbound_names = set()
    for name in code.co_names:
        if name in ('None', 'True', 'False'):
            continue
        try:
            global_vars[name] = global_ns[name]
        except KeyError:
            try:
                builtin_vars[name] = builtin_ns[name]
            except KeyError:
                unbound_names.add(name)
    return ClosureVars(nonlocal_vars, global_vars, builtin_vars, unbound_names)
Traceback = namedtuple('Traceback', 'filename lineno function code_context index')

def getframeinfo(frame, context=1):
    if False:
        print('Hello World!')
    'Get information about a frame or traceback object.\n\n    A tuple of five things is returned: the filename, the line number of\n    the current line, the function name, a list of lines of context from\n    the source code, and the index of the current line within that list.\n    The optional second argument specifies the number of lines of context\n    to return, which are centered around the current line.'
    if istraceback(frame):
        lineno = frame.tb_lineno
        frame = frame.tb_frame
    else:
        lineno = frame.f_lineno
    if not isframe(frame):
        raise TypeError('{!r} is not a frame or traceback object'.format(frame))
    filename = getsourcefile(frame) or getfile(frame)
    if context > 0:
        start = lineno - 1 - context // 2
        try:
            (lines, lnum) = findsource(frame)
        except OSError:
            lines = index = None
        else:
            start = max(start, 1)
            start = max(0, min(start, len(lines) - context))
            lines = lines[start:start + context]
            index = lineno - 1 - start
    else:
        lines = index = None
    return Traceback(filename, lineno, frame.f_code.co_name, lines, index)

def getlineno(frame):
    if False:
        i = 10
        return i + 15
    'Get the line number from a frame object, allowing for optimization.'
    return frame.f_lineno

def getouterframes(frame, context=1):
    if False:
        return 10
    'Get a list of records for a frame and all higher (calling) frames.\n\n    Each record contains a frame object, filename, line number, function\n    name, a list of lines of context, and index within the context.'
    framelist = []
    while frame:
        framelist.append((frame,) + getframeinfo(frame, context))
        frame = frame.f_back
    return framelist

def getinnerframes(tb, context=1):
    if False:
        return 10
    "Get a list of records for a traceback's frame and all lower frames.\n\n    Each record contains a frame object, filename, line number, function\n    name, a list of lines of context, and index within the context."
    framelist = []
    while tb:
        framelist.append((tb.tb_frame,) + getframeinfo(tb, context))
        tb = tb.tb_next
    return framelist

def currentframe():
    if False:
        i = 10
        return i + 15
    'Return the frame of the caller or None if this is not possible.'
    return sys._getframe(1) if hasattr(sys, '_getframe') else None

def stack(context=1):
    if False:
        print('Hello World!')
    "Return a list of records for the stack above the caller's frame."
    return getouterframes(sys._getframe(1), context)

def trace(context=1):
    if False:
        while True:
            i = 10
    'Return a list of records for the stack below the current exception.'
    return getinnerframes(sys.exc_info()[2], context)
_sentinel = object()

def _static_getmro(klass):
    if False:
        return 10
    return type.__dict__['__mro__'].__get__(klass)

def _check_instance(obj, attr):
    if False:
        while True:
            i = 10
    instance_dict = {}
    try:
        instance_dict = object.__getattribute__(obj, '__dict__')
    except AttributeError:
        pass
    return dict.get(instance_dict, attr, _sentinel)

def _check_class(klass, attr):
    if False:
        print('Hello World!')
    for entry in _static_getmro(klass):
        if _shadowed_dict(type(entry)) is _sentinel:
            try:
                return entry.__dict__[attr]
            except KeyError:
                pass
    return _sentinel

def _is_type(obj):
    if False:
        print('Hello World!')
    try:
        _static_getmro(obj)
    except TypeError:
        return False
    return True

def _shadowed_dict(klass):
    if False:
        return 10
    dict_attr = type.__dict__['__dict__']
    for entry in _static_getmro(klass):
        try:
            class_dict = dict_attr.__get__(entry)['__dict__']
        except KeyError:
            pass
        else:
            if not (type(class_dict) is types.GetSetDescriptorType and class_dict.__name__ == '__dict__' and (class_dict.__objclass__ is entry)):
                return class_dict
    return _sentinel

def getattr_static(obj, attr, default=_sentinel):
    if False:
        while True:
            i = 10
    "Retrieve attributes without triggering dynamic lookup via the\n       descriptor protocol,  __getattr__ or __getattribute__.\n\n       Note: this function may not be able to retrieve all attributes\n       that getattr can fetch (like dynamically created attributes)\n       and may find attributes that getattr can't (like descriptors\n       that raise AttributeError). It can also return descriptor objects\n       instead of instance members in some cases. See the\n       documentation for details.\n    "
    instance_result = _sentinel
    if not _is_type(obj):
        klass = type(obj)
        dict_attr = _shadowed_dict(klass)
        if dict_attr is _sentinel or type(dict_attr) is types.MemberDescriptorType:
            instance_result = _check_instance(obj, attr)
    else:
        klass = obj
    klass_result = _check_class(klass, attr)
    if instance_result is not _sentinel and klass_result is not _sentinel:
        if _check_class(type(klass_result), '__get__') is not _sentinel and _check_class(type(klass_result), '__set__') is not _sentinel:
            return klass_result
    if instance_result is not _sentinel:
        return instance_result
    if klass_result is not _sentinel:
        return klass_result
    if obj is klass:
        for entry in _static_getmro(type(klass)):
            if _shadowed_dict(type(entry)) is _sentinel:
                try:
                    return entry.__dict__[attr]
                except KeyError:
                    pass
    if default is not _sentinel:
        return default
    raise AttributeError(attr)
GEN_CREATED = 'GEN_CREATED'
GEN_RUNNING = 'GEN_RUNNING'
GEN_SUSPENDED = 'GEN_SUSPENDED'
GEN_CLOSED = 'GEN_CLOSED'

def getgeneratorstate(generator):
    if False:
        i = 10
        return i + 15
    'Get current state of a generator-iterator.\n\n    Possible states are:\n      GEN_CREATED: Waiting to start execution.\n      GEN_RUNNING: Currently being executed by the interpreter.\n      GEN_SUSPENDED: Currently suspended at a yield expression.\n      GEN_CLOSED: Execution has completed.\n    '
    if generator.gi_running:
        return GEN_RUNNING
    if generator.gi_frame is None:
        return GEN_CLOSED
    if generator.gi_frame.f_lasti == -1:
        return GEN_CREATED
    return GEN_SUSPENDED

def getgeneratorlocals(generator):
    if False:
        print('Hello World!')
    '\n    Get the mapping of generator local variables to their current values.\n\n    A dict is returned, with the keys the local variable names and values the\n    bound values.'
    if not isgenerator(generator):
        raise TypeError("'{!r}' is not a Python generator".format(generator))
    frame = getattr(generator, 'gi_frame', None)
    if frame is not None:
        return generator.gi_frame.f_locals
    else:
        return {}
_WrapperDescriptor = type(type.__call__)
_MethodWrapper = type(all.__call__)
_ClassMethodWrapper = type(int.__dict__['from_bytes'])
_NonUserDefinedCallables = (_WrapperDescriptor, _MethodWrapper, _ClassMethodWrapper, types.BuiltinFunctionType)

def _signature_get_user_defined_method(cls, method_name):
    if False:
        for i in range(10):
            print('nop')
    try:
        meth = getattr(cls, method_name)
    except AttributeError:
        return
    else:
        if not isinstance(meth, _NonUserDefinedCallables):
            return meth

def _signature_get_partial(wrapped_sig, partial, extra_args=()):
    if False:
        print('Hello World!')
    old_params = wrapped_sig.parameters
    new_params = OrderedDict(old_params.items())
    partial_args = partial.args or ()
    partial_keywords = partial.keywords or {}
    if extra_args:
        partial_args = extra_args + partial_args
    try:
        ba = wrapped_sig.bind_partial(*partial_args, **partial_keywords)
    except TypeError as ex:
        msg = 'partial object {!r} has incorrect arguments'.format(partial)
        raise ValueError(msg) from ex
    transform_to_kwonly = False
    for (param_name, param) in old_params.items():
        try:
            arg_value = ba.arguments[param_name]
        except KeyError:
            pass
        else:
            if param.kind is _POSITIONAL_ONLY:
                new_params.pop(param_name)
                continue
            if param.kind is _POSITIONAL_OR_KEYWORD:
                if param_name in partial_keywords:
                    transform_to_kwonly = True
                    new_params[param_name] = param.replace(default=arg_value)
                else:
                    new_params.pop(param.name)
                    continue
            if param.kind is _KEYWORD_ONLY:
                new_params[param_name] = param.replace(default=arg_value)
        if transform_to_kwonly:
            assert param.kind is not _POSITIONAL_ONLY
            if param.kind is _POSITIONAL_OR_KEYWORD:
                new_param = new_params[param_name].replace(kind=_KEYWORD_ONLY)
                new_params[param_name] = new_param
                new_params.move_to_end(param_name)
            elif param.kind in (_KEYWORD_ONLY, _VAR_KEYWORD):
                new_params.move_to_end(param_name)
            elif param.kind is _VAR_POSITIONAL:
                new_params.pop(param.name)
    return wrapped_sig.replace(parameters=new_params.values())

def _signature_bound_method(sig):
    if False:
        print('Hello World!')
    params = tuple(sig.parameters.values())
    if not params or params[0].kind in (_VAR_KEYWORD, _KEYWORD_ONLY):
        raise ValueError('invalid method signature')
    kind = params[0].kind
    if kind in (_POSITIONAL_OR_KEYWORD, _POSITIONAL_ONLY):
        params = params[1:]
    elif kind is not _VAR_POSITIONAL:
        raise ValueError('invalid argument type')
    return sig.replace(parameters=params)

def _signature_is_builtin(obj):
    if False:
        print('Hello World!')
    return isbuiltin(obj) or ismethoddescriptor(obj) or isinstance(obj, _NonUserDefinedCallables) or (obj in (type, object))

def _signature_is_functionlike(obj):
    if False:
        print('Hello World!')
    if not callable(obj) or isclass(obj):
        return False
    name = getattr(obj, '__name__', None)
    code = getattr(obj, '__code__', None)
    defaults = getattr(obj, '__defaults__', _void)
    kwdefaults = getattr(obj, '__kwdefaults__', _void)
    annotations = getattr(obj, '__annotations__', None)
    return isinstance(code, types.CodeType) and isinstance(name, str) and (defaults is None or isinstance(defaults, tuple)) and (kwdefaults is None or isinstance(kwdefaults, dict)) and isinstance(annotations, dict)

def _signature_get_bound_param(spec):
    if False:
        print('Hello World!')
    assert spec.startswith('($')
    pos = spec.find(',')
    if pos == -1:
        pos = spec.find(')')
    cpos = spec.find(':')
    assert cpos == -1 or cpos > pos
    cpos = spec.find('=')
    assert cpos == -1 or cpos > pos
    return spec[2:pos]

def _signature_strip_non_python_syntax(signature):
    if False:
        i = 10
        return i + 15
    '\n    Takes a signature in Argument Clinic\'s extended signature format.\n    Returns a tuple of three things:\n      * that signature re-rendered in standard Python syntax,\n      * the index of the "self" parameter (generally 0), or None if\n        the function does not have a "self" parameter, and\n      * the index of the last "positional only" parameter,\n        or None if the signature has no positional-only parameters.\n    '
    if not signature:
        return (signature, None, None)
    self_parameter = None
    last_positional_only = None
    lines = [l.encode('ascii') for l in signature.split('\n')]
    generator = iter(lines).__next__
    token_stream = tokenize.tokenize(generator)
    delayed_comma = False
    skip_next_comma = False
    text = []
    add = text.append
    current_parameter = 0
    OP = token.OP
    ERRORTOKEN = token.ERRORTOKEN
    t = next(token_stream)
    assert t.type == tokenize.ENCODING
    for t in token_stream:
        (type, string) = (t.type, t.string)
        if type == OP:
            if string == ',':
                if skip_next_comma:
                    skip_next_comma = False
                else:
                    assert not delayed_comma
                    delayed_comma = True
                    current_parameter += 1
                continue
            if string == '/':
                assert not skip_next_comma
                assert last_positional_only is None
                skip_next_comma = True
                last_positional_only = current_parameter - 1
                continue
        if type == ERRORTOKEN and string == '$':
            assert self_parameter is None
            self_parameter = current_parameter
            continue
        if delayed_comma:
            delayed_comma = False
            if not (type == OP and string == ')'):
                add(', ')
        add(string)
        if string == ',':
            add(' ')
    clean_signature = ''.join(text)
    return (clean_signature, self_parameter, last_positional_only)

def _signature_fromstr(cls, obj, s, skip_bound_arg=True):
    if False:
        print('Hello World!')
    Parameter = cls._parameter_cls
    (clean_signature, self_parameter, last_positional_only) = _signature_strip_non_python_syntax(s)
    program = 'def foo' + clean_signature + ': pass'
    try:
        module = ast.parse(program)
    except SyntaxError:
        module = None
    if not isinstance(module, ast.Module):
        raise ValueError('{!r} builtin has invalid signature'.format(obj))
    f = module.body[0]
    parameters = []
    empty = Parameter.empty
    invalid = object()
    module = None
    module_dict = {}
    module_name = getattr(obj, '__module__', None)
    if module_name:
        module = sys.modules.get(module_name, None)
        if module:
            module_dict = module.__dict__
    sys_module_dict = sys.modules

    def parse_name(node):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(node, ast.arg)
        if node.annotation != None:
            raise ValueError('Annotations are not currently supported')
        return node.arg

    def wrap_value(s):
        if False:
            return 10
        try:
            value = eval(s, module_dict)
        except NameError:
            try:
                value = eval(s, sys_module_dict)
            except NameError:
                raise RuntimeError()
        if isinstance(value, str):
            return ast.Str(value)
        if isinstance(value, (int, float)):
            return ast.Num(value)
        if isinstance(value, bytes):
            return ast.Bytes(value)
        if value in (True, False, None):
            return ast.NameConstant(value)
        raise RuntimeError()

    class RewriteSymbolics(ast.NodeTransformer):

        def visit_Attribute(self, node):
            if False:
                return 10
            a = []
            n = node
            while isinstance(n, ast.Attribute):
                a.append(n.attr)
                n = n.value
            if not isinstance(n, ast.Name):
                raise RuntimeError()
            a.append(n.id)
            value = '.'.join(reversed(a))
            return wrap_value(value)

        def visit_Name(self, node):
            if False:
                for i in range(10):
                    print('nop')
            if not isinstance(node.ctx, ast.Load):
                raise ValueError()
            return wrap_value(node.id)

    def p(name_node, default_node, default=empty):
        if False:
            for i in range(10):
                print('nop')
        name = parse_name(name_node)
        if name is invalid:
            return None
        if default_node and default_node is not _empty:
            try:
                default_node = RewriteSymbolics().visit(default_node)
                o = ast.literal_eval(default_node)
            except ValueError:
                o = invalid
            if o is invalid:
                return None
            default = o if o is not invalid else default
        parameters.append(Parameter(name, kind, default=default, annotation=empty))
    args = reversed(f.args.args)
    defaults = reversed(f.args.defaults)
    iter = itertools.zip_longest(args, defaults, fillvalue=None)
    if last_positional_only is not None:
        kind = Parameter.POSITIONAL_ONLY
    else:
        kind = Parameter.POSITIONAL_OR_KEYWORD
    for (i, (name, default)) in enumerate(reversed(list(iter))):
        p(name, default)
        if i == last_positional_only:
            kind = Parameter.POSITIONAL_OR_KEYWORD
    if f.args.vararg:
        kind = Parameter.VAR_POSITIONAL
        p(f.args.vararg, empty)
    kind = Parameter.KEYWORD_ONLY
    for (name, default) in zip(f.args.kwonlyargs, f.args.kw_defaults):
        p(name, default)
    if f.args.kwarg:
        kind = Parameter.VAR_KEYWORD
        p(f.args.kwarg, empty)
    if self_parameter is not None:
        assert parameters
        _self = getattr(obj, '__self__', None)
        self_isbound = _self is not None
        self_ismodule = ismodule(_self)
        if self_isbound and (self_ismodule or skip_bound_arg):
            parameters.pop(0)
        else:
            p = parameters[0].replace(kind=Parameter.POSITIONAL_ONLY)
            parameters[0] = p
    return cls(parameters, return_annotation=cls.empty)

def _signature_from_builtin(cls, func, skip_bound_arg=True):
    if False:
        for i in range(10):
            print('nop')
    if not _signature_is_builtin(func):
        raise TypeError('{!r} is not a Python builtin function'.format(func))
    s = getattr(func, '__text_signature__', None)
    if not s:
        raise ValueError('no signature found for builtin {!r}'.format(func))
    return _signature_fromstr(cls, func, s, skip_bound_arg)

def _signature_internal(obj, follow_wrapper_chains=True, skip_bound_arg=True):
    if False:
        return 10
    if not callable(obj):
        raise TypeError('{!r} is not a callable object'.format(obj))
    if isinstance(obj, types.MethodType):
        sig = _signature_internal(obj.__func__, follow_wrapper_chains, skip_bound_arg)
        if skip_bound_arg:
            return _signature_bound_method(sig)
        else:
            return sig
    if follow_wrapper_chains:
        obj = unwrap(obj, stop=lambda f: hasattr(f, '__signature__'))
    try:
        sig = obj.__signature__
    except AttributeError:
        pass
    else:
        if sig is not None:
            if not isinstance(sig, Signature):
                raise TypeError('unexpected object {!r} in __signature__ attribute'.format(sig))
            return sig
    try:
        partialmethod = obj._partialmethod
    except AttributeError:
        pass
    else:
        if isinstance(partialmethod, functools.partialmethod):
            wrapped_sig = _signature_internal(partialmethod.func, follow_wrapper_chains, skip_bound_arg)
            sig = _signature_get_partial(wrapped_sig, partialmethod, (None,))
            first_wrapped_param = tuple(wrapped_sig.parameters.values())[0]
            new_params = (first_wrapped_param,) + tuple(sig.parameters.values())
            return sig.replace(parameters=new_params)
    if isfunction(obj) or _signature_is_functionlike(obj):
        return Signature.from_function(obj)
    if _signature_is_builtin(obj):
        return _signature_from_builtin(Signature, obj, skip_bound_arg=skip_bound_arg)
    if isinstance(obj, functools.partial):
        wrapped_sig = _signature_internal(obj.func, follow_wrapper_chains, skip_bound_arg)
        return _signature_get_partial(wrapped_sig, obj)
    sig = None
    if isinstance(obj, type):
        call = _signature_get_user_defined_method(type(obj), '__call__')
        if call is not None:
            sig = _signature_internal(call, follow_wrapper_chains, skip_bound_arg)
        else:
            new = _signature_get_user_defined_method(obj, '__new__')
            if new is not None:
                sig = _signature_internal(new, follow_wrapper_chains, skip_bound_arg)
            else:
                init = _signature_get_user_defined_method(obj, '__init__')
                if init is not None:
                    sig = _signature_internal(init, follow_wrapper_chains, skip_bound_arg)
        if sig is None:
            for base in obj.__mro__[:-1]:
                try:
                    text_sig = base.__text_signature__
                except AttributeError:
                    pass
                else:
                    if text_sig:
                        return _signature_fromstr(Signature, obj, text_sig)
            if type not in obj.__mro__:
                if obj.__init__ is object.__init__:
                    return signature(object)
    elif not isinstance(obj, _NonUserDefinedCallables):
        call = _signature_get_user_defined_method(type(obj), '__call__')
        if call is not None:
            try:
                sig = _signature_internal(call, follow_wrapper_chains, skip_bound_arg)
            except ValueError as ex:
                msg = 'no signature found for {!r}'.format(obj)
                raise ValueError(msg) from ex
    if sig is not None:
        if skip_bound_arg:
            return _signature_bound_method(sig)
        else:
            return sig
    if isinstance(obj, types.BuiltinFunctionType):
        msg = 'no signature found for builtin function {!r}'.format(obj)
        raise ValueError(msg)
    raise ValueError('callable {!r} is not supported by signature'.format(obj))

def signature(obj):
    if False:
        while True:
            i = 10
    'Get a signature object for the passed callable.'
    return _signature_internal(obj)

class _void:
    """A private marker - used in Parameter & Signature"""

class _empty:
    pass

class _ParameterKind(int):

    def __new__(self, *args, name):
        if False:
            i = 10
            return i + 15
        obj = int.__new__(self, *args)
        obj._name = name
        return obj

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self._name

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<_ParameterKind: {!r}>'.format(self._name)
_POSITIONAL_ONLY = _ParameterKind(0, name='POSITIONAL_ONLY')
_POSITIONAL_OR_KEYWORD = _ParameterKind(1, name='POSITIONAL_OR_KEYWORD')
_VAR_POSITIONAL = _ParameterKind(2, name='VAR_POSITIONAL')
_KEYWORD_ONLY = _ParameterKind(3, name='KEYWORD_ONLY')
_VAR_KEYWORD = _ParameterKind(4, name='VAR_KEYWORD')

class Parameter:
    """Represents a parameter in a function signature.

    Has the following public attributes:

    * name : str
        The name of the parameter as a string.
    * default : object
        The default value for the parameter if specified.  If the
        parameter has no default value, this attribute is set to
        `Parameter.empty`.
    * annotation
        The annotation for the parameter if specified.  If the
        parameter has no annotation, this attribute is set to
        `Parameter.empty`.
    * kind : str
        Describes how argument values are bound to the parameter.
        Possible values: `Parameter.POSITIONAL_ONLY`,
        `Parameter.POSITIONAL_OR_KEYWORD`, `Parameter.VAR_POSITIONAL`,
        `Parameter.KEYWORD_ONLY`, `Parameter.VAR_KEYWORD`.
    """
    __slots__ = ('_name', '_kind', '_default', '_annotation')
    POSITIONAL_ONLY = _POSITIONAL_ONLY
    POSITIONAL_OR_KEYWORD = _POSITIONAL_OR_KEYWORD
    VAR_POSITIONAL = _VAR_POSITIONAL
    KEYWORD_ONLY = _KEYWORD_ONLY
    VAR_KEYWORD = _VAR_KEYWORD
    empty = _empty

    def __init__(self, name, kind, *, default=_empty, annotation=_empty):
        if False:
            while True:
                i = 10
        if kind not in (_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD, _VAR_POSITIONAL, _KEYWORD_ONLY, _VAR_KEYWORD):
            raise ValueError("invalid value for 'Parameter.kind' attribute")
        self._kind = kind
        if default is not _empty:
            if kind in (_VAR_POSITIONAL, _VAR_KEYWORD):
                msg = '{} parameters cannot have default values'.format(kind)
                raise ValueError(msg)
        self._default = default
        self._annotation = annotation
        if name is _empty:
            raise ValueError('name is a required attribute for Parameter')
        if not isinstance(name, str):
            raise TypeError('name must be a str, not a {!r}'.format(name))
        if not name.isidentifier():
            raise ValueError('{!r} is not a valid parameter name'.format(name))
        self._name = name

    @property
    def name(self):
        if False:
            while True:
                i = 10
        return self._name

    @property
    def default(self):
        if False:
            return 10
        return self._default

    @property
    def annotation(self):
        if False:
            print('Hello World!')
        return self._annotation

    @property
    def kind(self):
        if False:
            i = 10
            return i + 15
        return self._kind

    def replace(self, *, name=_void, kind=_void, annotation=_void, default=_void):
        if False:
            for i in range(10):
                print('nop')
        'Creates a customized copy of the Parameter.'
        if name is _void:
            name = self._name
        if kind is _void:
            kind = self._kind
        if annotation is _void:
            annotation = self._annotation
        if default is _void:
            default = self._default
        return type(self)(name, kind, default=default, annotation=annotation)

    def __str__(self):
        if False:
            while True:
                i = 10
        kind = self.kind
        formatted = self._name
        if self._annotation is not _empty:
            formatted = '{}:{}'.format(formatted, formatannotation(self._annotation))
        if self._default is not _empty:
            formatted = '{}={}'.format(formatted, repr(self._default))
        if kind == _VAR_POSITIONAL:
            formatted = '*' + formatted
        elif kind == _VAR_KEYWORD:
            formatted = '**' + formatted
        return formatted

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<{} at {:#x} {!r}>'.format(self.__class__.__name__, id(self), self.name)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return issubclass(other.__class__, Parameter) and self._name == other._name and (self._kind == other._kind) and (self._default == other._default) and (self._annotation == other._annotation)

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self.__eq__(other)

class BoundArguments:
    """Result of `Signature.bind` call.  Holds the mapping of arguments
    to the function's parameters.

    Has the following public attributes:

    * arguments : OrderedDict
        An ordered mutable mapping of parameters' names to arguments' values.
        Does not contain arguments' default values.
    * signature : Signature
        The Signature object that created this instance.
    * args : tuple
        Tuple of positional arguments values.
    * kwargs : dict
        Dict of keyword arguments values.
    """

    def __init__(self, signature, arguments):
        if False:
            print('Hello World!')
        self.arguments = arguments
        self._signature = signature

    @property
    def signature(self):
        if False:
            while True:
                i = 10
        return self._signature

    @property
    def args(self):
        if False:
            for i in range(10):
                print('nop')
        args = []
        for (param_name, param) in self._signature.parameters.items():
            if param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY):
                break
            try:
                arg = self.arguments[param_name]
            except KeyError:
                break
            else:
                if param.kind == _VAR_POSITIONAL:
                    args.extend(arg)
                else:
                    args.append(arg)
        return tuple(args)

    @property
    def kwargs(self):
        if False:
            while True:
                i = 10
        kwargs = {}
        kwargs_started = False
        for (param_name, param) in self._signature.parameters.items():
            if not kwargs_started:
                if param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY):
                    kwargs_started = True
                elif param_name not in self.arguments:
                    kwargs_started = True
                    continue
            if not kwargs_started:
                continue
            try:
                arg = self.arguments[param_name]
            except KeyError:
                pass
            else:
                if param.kind == _VAR_KEYWORD:
                    kwargs.update(arg)
                else:
                    kwargs[param_name] = arg
        return kwargs

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return issubclass(other.__class__, BoundArguments) and self.signature == other.signature and (self.arguments == other.arguments)

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self.__eq__(other)

class Signature:
    """A Signature object represents the overall signature of a function.
    It stores a Parameter object for each parameter accepted by the
    function, as well as information specific to the function itself.

    A Signature object has the following public attributes and methods:

    * parameters : OrderedDict
        An ordered mapping of parameters' names to the corresponding
        Parameter objects (keyword-only arguments are in the same order
        as listed in `code.co_varnames`).
    * return_annotation : object
        The annotation for the return type of the function if specified.
        If the function has no annotation for its return type, this
        attribute is set to `Signature.empty`.
    * bind(*args, **kwargs) -> BoundArguments
        Creates a mapping from positional and keyword arguments to
        parameters.
    * bind_partial(*args, **kwargs) -> BoundArguments
        Creates a partial mapping from positional and keyword arguments
        to parameters (simulating 'functools.partial' behavior.)
    """
    __slots__ = ('_return_annotation', '_parameters')
    _parameter_cls = Parameter
    _bound_arguments_cls = BoundArguments
    empty = _empty

    def __init__(self, parameters=None, *, return_annotation=_empty, __validate_parameters__=True):
        if False:
            return 10
        "Constructs Signature from the given list of Parameter\n        objects and 'return_annotation'.  All arguments are optional.\n        "
        if parameters is None:
            params = OrderedDict()
        elif __validate_parameters__:
            params = OrderedDict()
            top_kind = _POSITIONAL_ONLY
            kind_defaults = False
            for (idx, param) in enumerate(parameters):
                kind = param.kind
                name = param.name
                if kind < top_kind:
                    msg = 'wrong parameter order: {!r} before {!r}'
                    msg = msg.format(top_kind, kind)
                    raise ValueError(msg)
                elif kind > top_kind:
                    kind_defaults = False
                    top_kind = kind
                if kind in (_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD):
                    if param.default is _empty:
                        if kind_defaults:
                            msg = 'non-default argument follows default argument'
                            raise ValueError(msg)
                    else:
                        kind_defaults = True
                if name in params:
                    msg = 'duplicate parameter name: {!r}'.format(name)
                    raise ValueError(msg)
                params[name] = param
        else:
            params = OrderedDict(((param.name, param) for param in parameters))
        self._parameters = types.MappingProxyType(params)
        self._return_annotation = return_annotation

    @classmethod
    def from_function(cls, func):
        if False:
            for i in range(10):
                print('nop')
        'Constructs Signature for the given python function'
        is_duck_function = False
        if not isfunction(func):
            if _signature_is_functionlike(func):
                is_duck_function = True
            else:
                raise TypeError('{!r} is not a Python function'.format(func))
        Parameter = cls._parameter_cls
        func_code = func.__code__
        pos_count = func_code.co_argcount
        arg_names = func_code.co_varnames
        positional = tuple(arg_names[:pos_count])
        keyword_only_count = func_code.co_kwonlyargcount
        keyword_only = arg_names[pos_count:pos_count + keyword_only_count]
        annotations = func.__annotations__
        defaults = func.__defaults__
        kwdefaults = func.__kwdefaults__
        if defaults:
            pos_default_count = len(defaults)
        else:
            pos_default_count = 0
        parameters = []
        non_default_count = pos_count - pos_default_count
        for name in positional[:non_default_count]:
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation, kind=_POSITIONAL_OR_KEYWORD))
        for (offset, name) in enumerate(positional[non_default_count:]):
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation, kind=_POSITIONAL_OR_KEYWORD, default=defaults[offset]))
        if func_code.co_flags & CO_VARARGS:
            name = arg_names[pos_count + keyword_only_count]
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation, kind=_VAR_POSITIONAL))
        for name in keyword_only:
            default = _empty
            if kwdefaults is not None:
                default = kwdefaults.get(name, _empty)
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation, kind=_KEYWORD_ONLY, default=default))
        if func_code.co_flags & CO_VARKEYWORDS:
            index = pos_count + keyword_only_count
            if func_code.co_flags & CO_VARARGS:
                index += 1
            name = arg_names[index]
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation, kind=_VAR_KEYWORD))
        return cls(parameters, return_annotation=annotations.get('return', _empty), __validate_parameters__=is_duck_function)

    @classmethod
    def from_builtin(cls, func):
        if False:
            for i in range(10):
                print('nop')
        return _signature_from_builtin(cls, func)

    @property
    def parameters(self):
        if False:
            return 10
        return self._parameters

    @property
    def return_annotation(self):
        if False:
            i = 10
            return i + 15
        return self._return_annotation

    def replace(self, *, parameters=_void, return_annotation=_void):
        if False:
            i = 10
            return i + 15
        "Creates a customized copy of the Signature.\n        Pass 'parameters' and/or 'return_annotation' arguments\n        to override them in the new copy.\n        "
        if parameters is _void:
            parameters = self.parameters.values()
        if return_annotation is _void:
            return_annotation = self._return_annotation
        return type(self)(parameters, return_annotation=return_annotation)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not issubclass(type(other), Signature) or self.return_annotation != other.return_annotation or len(self.parameters) != len(other.parameters):
            return False
        other_positions = {param: idx for (idx, param) in enumerate(other.parameters.keys())}
        for (idx, (param_name, param)) in enumerate(self.parameters.items()):
            if param.kind == _KEYWORD_ONLY:
                try:
                    other_param = other.parameters[param_name]
                except KeyError:
                    return False
                else:
                    if param != other_param:
                        return False
            else:
                try:
                    other_idx = other_positions[param_name]
                except KeyError:
                    return False
                else:
                    if idx != other_idx or param != other.parameters[param_name]:
                        return False
        return True

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        return not self.__eq__(other)

    def _bind(self, args, kwargs, *, partial=False):
        if False:
            i = 10
            return i + 15
        "Private method.  Don't use directly."
        arguments = OrderedDict()
        parameters = iter(self.parameters.values())
        parameters_ex = ()
        arg_vals = iter(args)
        while True:
            try:
                arg_val = next(arg_vals)
            except StopIteration:
                try:
                    param = next(parameters)
                except StopIteration:
                    break
                else:
                    if param.kind == _VAR_POSITIONAL:
                        break
                    elif param.name in kwargs:
                        if param.kind == _POSITIONAL_ONLY:
                            msg = '{arg!r} parameter is positional only, but was passed as a keyword'
                            msg = msg.format(arg=param.name)
                            raise TypeError(msg) from None
                        parameters_ex = (param,)
                        break
                    elif param.kind == _VAR_KEYWORD or param.default is not _empty:
                        parameters_ex = (param,)
                        break
                    elif partial:
                        parameters_ex = (param,)
                        break
                    else:
                        msg = '{arg!r} parameter lacking default value'
                        msg = msg.format(arg=param.name)
                        raise TypeError(msg) from None
            else:
                try:
                    param = next(parameters)
                except StopIteration:
                    raise TypeError('too many positional arguments') from None
                else:
                    if param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY):
                        raise TypeError('too many positional arguments')
                    if param.kind == _VAR_POSITIONAL:
                        values = [arg_val]
                        values.extend(arg_vals)
                        arguments[param.name] = tuple(values)
                        break
                    if param.name in kwargs:
                        raise TypeError('multiple values for argument {arg!r}'.format(arg=param.name))
                    arguments[param.name] = arg_val
        kwargs_param = None
        for param in itertools.chain(parameters_ex, parameters):
            if param.kind == _VAR_KEYWORD:
                kwargs_param = param
                continue
            if param.kind == _VAR_POSITIONAL:
                continue
            param_name = param.name
            try:
                arg_val = kwargs.pop(param_name)
            except KeyError:
                if not partial and param.kind != _VAR_POSITIONAL and (param.default is _empty):
                    raise TypeError('{arg!r} parameter lacking default value'.format(arg=param_name)) from None
            else:
                if param.kind == _POSITIONAL_ONLY:
                    raise TypeError('{arg!r} parameter is positional only, but was passed as a keyword'.format(arg=param.name))
                arguments[param_name] = arg_val
        if kwargs:
            if kwargs_param is not None:
                arguments[kwargs_param.name] = kwargs
            else:
                raise TypeError('too many keyword arguments')
        return self._bound_arguments_cls(self, arguments)

    def bind(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Get a BoundArguments object, that maps the passed `args`\n        and `kwargs` to the function's signature.  Raises `TypeError`\n        if the passed arguments can not be bound.\n        "
        return args[0]._bind(args[1:], kwargs)

    def bind_partial(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        "Get a BoundArguments object, that partially maps the\n        passed `args` and `kwargs` to the function's signature.\n        Raises `TypeError` if the passed arguments can not be bound.\n        "
        return args[0]._bind(args[1:], kwargs, partial=True)

    def __str__(self):
        if False:
            while True:
                i = 10
        result = []
        render_pos_only_separator = False
        render_kw_only_separator = True
        for param in self.parameters.values():
            formatted = str(param)
            kind = param.kind
            if kind == _POSITIONAL_ONLY:
                render_pos_only_separator = True
            elif render_pos_only_separator:
                result.append('/')
                render_pos_only_separator = False
            if kind == _VAR_POSITIONAL:
                render_kw_only_separator = False
            elif kind == _KEYWORD_ONLY and render_kw_only_separator:
                result.append('*')
                render_kw_only_separator = False
            result.append(formatted)
        if render_pos_only_separator:
            result.append('/')
        rendered = '({})'.format(', '.join(result))
        if self.return_annotation is not _empty:
            anno = formatannotation(self.return_annotation)
            rendered += ' -> {}'.format(anno)
        return rendered

def _main():
    if False:
        for i in range(10):
            print('nop')
    ' Logic for inspecting an object given at command line '
    import argparse
    import importlib
    parser = argparse.ArgumentParser()
    parser.add_argument('object', help="The object to be analysed. It supports the 'module:qualname' syntax")
    parser.add_argument('-d', '--details', action='store_true', help='Display info about the module rather than its source code')
    args = parser.parse_args()
    target = args.object
    (mod_name, has_attrs, attrs) = target.partition(':')
    try:
        obj = module = importlib.import_module(mod_name)
    except Exception as exc:
        msg = 'Failed to import {} ({}: {})'.format(mod_name, type(exc).__name__, exc)
        print(msg, file=sys.stderr)
        exit(2)
    if has_attrs:
        parts = attrs.split('.')
        obj = module
        for part in parts:
            obj = getattr(obj, part)
    if module.__name__ in sys.builtin_module_names:
        print("Can't get info for builtin modules.", file=sys.stderr)
        exit(1)
    if args.details:
        print('Target: {}'.format(target))
        print('Origin: {}'.format(getsourcefile(module)))
        print('Cached: {}'.format(module.__cached__))
        if obj is module:
            print('Loader: {}'.format(repr(module.__loader__)))
            if hasattr(module, '__path__'):
                print('Submodule search path: {}'.format(module.__path__))
        else:
            try:
                (__, lineno) = findsource(obj)
            except Exception:
                pass
            else:
                print('Line: {}'.format(lineno))
        print('\n')
    else:
        print(getsource(obj))
if __name__ == '__main__':
    _main()