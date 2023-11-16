"""Live entity inspection utilities.

This module contains whatever inspect doesn't offer out of the box.
"""
import inspect
import itertools
import linecache
import sys
import threading
import types
import six
_linecache_lock = threading.Lock()
SPECIAL_BUILTINS = {'dict': dict, 'enumerate': enumerate, 'float': float, 'int': int, 'len': len, 'list': list, 'print': print, 'range': range, 'tuple': tuple, 'type': type, 'zip': zip}

def islambda(f):
    if False:
        for i in range(10):
            print('nop')
    if not inspect.isfunction(f):
        return False
    if not (hasattr(f, '__name__') and hasattr(f, '__code__')):
        return False
    return f.__name__ == '<lambda>' or f.__code__.co_name == '<lambda>'

def isnamedtuple(f):
    if False:
        print('Hello World!')
    'Returns True if the argument is a namedtuple-like.'
    if not (inspect.isclass(f) and issubclass(f, tuple)):
        return False
    if not hasattr(f, '_fields'):
        return False
    fields = getattr(f, '_fields')
    if not isinstance(fields, tuple):
        return False
    if not all((isinstance(f, str) for f in fields)):
        return False
    return True

def isbuiltin(f):
    if False:
        i = 10
        return i + 15
    'Returns True if the argument is a built-in function.'
    if any((f is builtin for builtin in six.moves.builtins.__dict__.values())):
        return True
    elif isinstance(f, types.BuiltinFunctionType):
        return True
    elif inspect.isbuiltin(f):
        return True
    elif f is eval:
        return True
    else:
        return False

def isconstructor(cls):
    if False:
        while True:
            i = 10
    'Returns True if the argument is an object constructor.\n\n  In general, any object of type class is a constructor, with the exception\n  of classes created using a callable metaclass.\n  See below for why a callable metaclass is not a trivial combination:\n  https://docs.python.org/2.7/reference/datamodel.html#customizing-class-creation\n\n  Args:\n    cls: Any\n\n  Returns:\n    Bool\n  '
    return inspect.isclass(cls) and (not (issubclass(cls.__class__, type) and hasattr(cls.__class__, '__call__') and (cls.__class__.__call__ is not type.__call__)))

def _fix_linecache_record(obj):
    if False:
        for i in range(10):
            print('nop')
    "Fixes potential corruption of linecache in the presence of functools.wraps.\n\n  functools.wraps modifies the target object's __module__ field, which seems\n  to confuse linecache in special instances, for example when the source is\n  loaded from a .par file (see https://google.github.io/subpar/subpar.html).\n\n  This function simply triggers a call to linecache.updatecache when a mismatch\n  was detected between the object's __module__ property and the object's source\n  file.\n\n  Args:\n    obj: Any\n  "
    if hasattr(obj, '__module__'):
        obj_file = inspect.getfile(obj)
        obj_module = obj.__module__
        loaded_modules = tuple(sys.modules.values())
        for m in loaded_modules:
            if hasattr(m, '__file__') and m.__file__ == obj_file:
                if obj_module is not m:
                    linecache.updatecache(obj_file, m.__dict__)

def getimmediatesource(obj):
    if False:
        for i in range(10):
            print('nop')
    'A variant of inspect.getsource that ignores the __wrapped__ property.'
    with _linecache_lock:
        _fix_linecache_record(obj)
        (lines, lnum) = inspect.findsource(obj)
        return ''.join(inspect.getblock(lines[lnum:]))

def getnamespace(f):
    if False:
        return 10
    'Returns the complete namespace of a function.\n\n  Namespace is defined here as the mapping of all non-local variables to values.\n  This includes the globals and the closure variables. Note that this captures\n  the entire globals collection of the function, and may contain extra symbols\n  that it does not actually use.\n\n  Args:\n    f: User defined function.\n\n  Returns:\n    A dict mapping symbol names to values.\n  '
    namespace = dict(six.get_function_globals(f))
    closure = six.get_function_closure(f)
    freevars = six.get_function_code(f).co_freevars
    if freevars and closure:
        for (name, cell) in zip(freevars, closure):
            try:
                namespace[name] = cell.cell_contents
            except ValueError:
                pass
    return namespace

def getqualifiedname(namespace, object_, max_depth=5, visited=None):
    if False:
        print('Hello World!')
    "Returns the name by which a value can be referred to in a given namespace.\n\n  If the object defines a parent module, the function attempts to use it to\n  locate the object.\n\n  This function will recurse inside modules, but it will not search objects for\n  attributes. The recursion depth is controlled by max_depth.\n\n  Args:\n    namespace: Dict[str, Any], the namespace to search into.\n    object_: Any, the value to search.\n    max_depth: Optional[int], a limit to the recursion depth when searching\n      inside modules.\n    visited: Optional[Set[int]], ID of modules to avoid visiting.\n  Returns: Union[str, None], the fully-qualified name that resolves to the value\n    o, or None if it couldn't be found.\n  "
    if visited is None:
        visited = set()
    namespace = dict(namespace)
    for name in namespace:
        if object_ is namespace[name]:
            return name
    parent = inspect.getmodule(object_)
    if parent is not None and parent is not object_ and (parent is not namespace):
        parent_name = getqualifiedname(namespace, parent, max_depth=0, visited=visited)
        if parent_name is not None:
            name_in_parent = getqualifiedname(parent.__dict__, object_, max_depth=0, visited=visited)
            assert name_in_parent is not None, 'An object should always be found in its owner module'
            return '{}.{}'.format(parent_name, name_in_parent)
    if max_depth:
        for name in namespace.keys():
            value = namespace[name]
            if inspect.ismodule(value) and id(value) not in visited:
                visited.add(id(value))
                name_in_module = getqualifiedname(value.__dict__, object_, max_depth - 1, visited)
                if name_in_module is not None:
                    return '{}.{}'.format(name, name_in_module)
    return None

def _get_unbound_function(m):
    if False:
        for i in range(10):
            print('nop')
    if hasattr(m, '__func__'):
        return m.__func__
    if hasattr(m, 'im_func'):
        return m.im_func
    return m

def getdefiningclass(m, owner_class):
    if False:
        while True:
            i = 10
    'Resolves the class (e.g. one of the superclasses) that defined a method.'
    m = _get_unbound_function(m)
    for superclass in reversed(inspect.getmro(owner_class)):
        if hasattr(superclass, m.__name__):
            superclass_m = getattr(superclass, m.__name__)
            if _get_unbound_function(superclass_m) is m:
                return superclass
            elif hasattr(m, '__self__') and m.__self__ == owner_class:
                return superclass
    return owner_class

def getmethodclass(m):
    if False:
        return 10
    "Resolves a function's owner, e.g.\n\n  a method's class.\n\n  Note that this returns the object that the function was retrieved from, not\n  necessarily the class where it was defined.\n\n  This function relies on Python stack frame support in the interpreter, and\n  has the same limitations that inspect.currentframe.\n\n  Limitations. This function will only work correctly if the owned class is\n  visible in the caller's global or local variables.\n\n  Args:\n    m: A user defined function\n\n  Returns:\n    The class that this function was retrieved from, or None if the function\n    is not an object or class method, or the class that owns the object or\n    method is not visible to m.\n\n  Raises:\n    ValueError: if the class could not be resolved for any unexpected reason.\n  "
    if not hasattr(m, '__name__') and hasattr(m, '__class__') and hasattr(m, '__call__'):
        if isinstance(m.__class__, six.class_types):
            return m.__class__
    m_self = getattr(m, '__self__', None)
    if m_self is not None:
        if inspect.isclass(m_self):
            return m_self
        return m_self.__class__
    owners = []
    caller_frame = inspect.currentframe().f_back
    try:
        for v in itertools.chain(caller_frame.f_locals.values(), caller_frame.f_globals.values()):
            if hasattr(v, m.__name__):
                candidate = getattr(v, m.__name__)
                if hasattr(candidate, 'im_func'):
                    candidate = candidate.im_func
                if hasattr(m, 'im_func'):
                    m = m.im_func
                if candidate is m:
                    owners.append(v)
    finally:
        del caller_frame
    if owners:
        if len(owners) == 1:
            return owners[0]
        owner_types = tuple((o if inspect.isclass(o) else type(o) for o in owners))
        for o in owner_types:
            if inspect.isclass(o) and issubclass(o, tuple(owner_types)):
                return o
        raise ValueError('Found too many owners of %s: %s' % (m, owners))
    return None

def getfutureimports(entity):
    if False:
        return 10
    'Detects what future imports are necessary to safely execute entity source.\n\n  Args:\n    entity: Any object\n\n  Returns:\n    A tuple of future strings\n  '
    if not (inspect.isfunction(entity) or inspect.ismethod(entity)):
        return tuple()
    return tuple(sorted((name for (name, value) in entity.__globals__.items() if getattr(value, '__module__', None) == '__future__')))