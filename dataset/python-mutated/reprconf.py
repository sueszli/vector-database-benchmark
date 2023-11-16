"""Generic configuration system using unrepr.

Configuration data may be supplied as a Python dictionary, as a filename,
or as an open file object. When you supply a filename or file, Python's
builtin ConfigParser is used (with some extensions).

Namespaces
----------

Configuration keys are separated into namespaces by the first "." in the key.

The only key that cannot exist in a namespace is the "environment" entry.
This special entry 'imports' other config entries from a template stored in
the Config.environments dict.

You can define your own namespaces to be called when new config is merged
by adding a named handler to Config.namespaces. The name can be any string,
and the handler must be either a callable or a context manager.
"""
import builtins
import configparser
import operator
import sys
from cherrypy._cpcompat import text_or_bytes

class NamespaceSet(dict):
    """A dict of config namespace names and handlers.

    Each config entry should begin with a namespace name; the corresponding
    namespace handler will be called once for each config entry in that
    namespace, and will be passed two arguments: the config key (with the
    namespace removed) and the config value.

    Namespace handlers may be any Python callable; they may also be
    context managers, in which case their __enter__
    method should return a callable to be used as the handler.
    See cherrypy.tools (the Toolbox class) for an example.
    """

    def __call__(self, config):
        if False:
            for i in range(10):
                print('nop')
        "Iterate through config and pass it to each namespace handler.\n\n        config\n            A flat dict, where keys use dots to separate\n            namespaces, and values are arbitrary.\n\n        The first name in each config key is used to look up the corresponding\n        namespace handler. For example, a config entry of {'tools.gzip.on': v}\n        will call the 'tools' namespace handler with the args: ('gzip.on', v)\n        "
        ns_confs = {}
        for k in config:
            if '.' in k:
                (ns, name) = k.split('.', 1)
                bucket = ns_confs.setdefault(ns, {})
                bucket[name] = config[k]
        for (ns, handler) in self.items():
            exit = getattr(handler, '__exit__', None)
            if exit:
                callable = handler.__enter__()
                no_exc = True
                try:
                    try:
                        for (k, v) in ns_confs.get(ns, {}).items():
                            callable(k, v)
                    except Exception:
                        no_exc = False
                        if exit is None:
                            raise
                        if not exit(*sys.exc_info()):
                            raise
                finally:
                    if no_exc and exit:
                        exit(None, None, None)
            else:
                for (k, v) in ns_confs.get(ns, {}).items():
                    handler(k, v)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '%s.%s(%s)' % (self.__module__, self.__class__.__name__, dict.__repr__(self))

    def __copy__(self):
        if False:
            while True:
                i = 10
        newobj = self.__class__()
        newobj.update(self)
        return newobj
    copy = __copy__

class Config(dict):
    """A dict-like set of configuration data, with defaults and namespaces.

    May take a file, filename, or dict.
    """
    defaults = {}
    environments = {}
    namespaces = NamespaceSet()

    def __init__(self, file=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.reset()
        if file is not None:
            self.update(file)
        if kwargs:
            self.update(kwargs)

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        'Reset self to default values.'
        self.clear()
        dict.update(self, self.defaults)

    def update(self, config):
        if False:
            print('Hello World!')
        'Update self from a dict, file, or filename.'
        self._apply(Parser.load(config))

    def _apply(self, config):
        if False:
            print('Hello World!')
        'Update self from a dict.'
        which_env = config.get('environment')
        if which_env:
            env = self.environments[which_env]
            for k in env:
                if k not in config:
                    config[k] = env[k]
        dict.update(self, config)
        self.namespaces(config)

    def __setitem__(self, k, v):
        if False:
            return 10
        dict.__setitem__(self, k, v)
        self.namespaces({k: v})

class Parser(configparser.ConfigParser):
    """Sub-class of ConfigParser that keeps the case of options and that
    raises an exception if the file cannot be read.
    """

    def optionxform(self, optionstr):
        if False:
            print('Hello World!')
        return optionstr

    def read(self, filenames):
        if False:
            return 10
        if isinstance(filenames, text_or_bytes):
            filenames = [filenames]
        for filename in filenames:
            with open(filename) as fp:
                self._read(fp, filename)

    def as_dict(self, raw=False, vars=None):
        if False:
            return 10
        'Convert an INI file to a dictionary'
        result = {}
        for section in self.sections():
            if section not in result:
                result[section] = {}
            for option in self.options(section):
                value = self.get(section, option, raw=raw, vars=vars)
                try:
                    value = unrepr(value)
                except Exception:
                    x = sys.exc_info()[1]
                    msg = 'Config error in section: %r, option: %r, value: %r. Config values must be valid Python.' % (section, option, value)
                    raise ValueError(msg, x.__class__.__name__, x.args)
                result[section][option] = value
        return result

    def dict_from_file(self, file):
        if False:
            i = 10
            return i + 15
        if hasattr(file, 'read'):
            self.readfp(file)
        else:
            self.read(file)
        return self.as_dict()

    @classmethod
    def load(self, input):
        if False:
            print('Hello World!')
        "Resolve 'input' to dict from a dict, file, or filename."
        is_file = isinstance(input, text_or_bytes) or hasattr(input, 'read')
        return Parser().dict_from_file(input) if is_file else input.copy()

class _Builder:

    def build(self, o):
        if False:
            return 10
        m = getattr(self, 'build_' + o.__class__.__name__, None)
        if m is None:
            raise TypeError('unrepr does not recognize %s' % repr(o.__class__.__name__))
        return m(o)

    def astnode(self, s):
        if False:
            for i in range(10):
                print('nop')
        'Return a Python3 ast Node compiled from a string.'
        try:
            import ast
        except ImportError:
            return eval(s)
        p = ast.parse('__tempvalue__ = ' + s)
        return p.body[0].value

    def build_Subscript(self, o):
        if False:
            print('Hello World!')
        return self.build(o.value)[self.build(o.slice)]

    def build_Index(self, o):
        if False:
            print('Hello World!')
        return self.build(o.value)

    def _build_call35(self, o):
        if False:
            for i in range(10):
                print('nop')
        '\n        Workaround for python 3.5 _ast.Call signature, docs found here\n        https://greentreesnakes.readthedocs.org/en/latest/nodes.html\n        '
        import ast
        callee = self.build(o.func)
        args = []
        if o.args is not None:
            for a in o.args:
                if isinstance(a, ast.Starred):
                    args.append(self.build(a.value))
                else:
                    args.append(self.build(a))
        kwargs = {}
        for kw in o.keywords:
            if kw.arg is None:
                rst = self.build(kw.value)
                if not isinstance(rst, dict):
                    raise TypeError('Invalid argument for call.Must be a mapping object.')
                for (k, v) in rst.items():
                    if k not in kwargs:
                        kwargs[k] = v
            else:
                kwargs[kw.arg] = self.build(kw.value)
        return callee(*args, **kwargs)

    def build_Call(self, o):
        if False:
            while True:
                i = 10
        if sys.version_info >= (3, 5):
            return self._build_call35(o)
        callee = self.build(o.func)
        if o.args is None:
            args = ()
        else:
            args = tuple([self.build(a) for a in o.args])
        if o.starargs is None:
            starargs = ()
        else:
            starargs = tuple(self.build(o.starargs))
        if o.kwargs is None:
            kwargs = {}
        else:
            kwargs = self.build(o.kwargs)
        if o.keywords is not None:
            for kw in o.keywords:
                kwargs[kw.arg] = self.build(kw.value)
        return callee(*args + starargs, **kwargs)

    def build_List(self, o):
        if False:
            return 10
        return list(map(self.build, o.elts))

    def build_Str(self, o):
        if False:
            for i in range(10):
                print('nop')
        return o.s

    def build_Num(self, o):
        if False:
            return 10
        return o.n

    def build_Dict(self, o):
        if False:
            for i in range(10):
                print('nop')
        return dict([(self.build(k), self.build(v)) for (k, v) in zip(o.keys, o.values)])

    def build_Tuple(self, o):
        if False:
            print('Hello World!')
        return tuple(self.build_List(o))

    def build_Name(self, o):
        if False:
            while True:
                i = 10
        name = o.id
        if name == 'None':
            return None
        if name == 'True':
            return True
        if name == 'False':
            return False
        try:
            return modules(name)
        except ImportError:
            pass
        try:
            return getattr(builtins, name)
        except AttributeError:
            pass
        raise TypeError('unrepr could not resolve the name %s' % repr(name))

    def build_NameConstant(self, o):
        if False:
            i = 10
            return i + 15
        return o.value
    build_Constant = build_NameConstant

    def build_UnaryOp(self, o):
        if False:
            i = 10
            return i + 15
        (op, operand) = map(self.build, [o.op, o.operand])
        return op(operand)

    def build_BinOp(self, o):
        if False:
            print('Hello World!')
        (left, op, right) = map(self.build, [o.left, o.op, o.right])
        return op(left, right)

    def build_Add(self, o):
        if False:
            print('Hello World!')
        return operator.add

    def build_Mult(self, o):
        if False:
            return 10
        return operator.mul

    def build_USub(self, o):
        if False:
            i = 10
            return i + 15
        return operator.neg

    def build_Attribute(self, o):
        if False:
            while True:
                i = 10
        parent = self.build(o.value)
        return getattr(parent, o.attr)

    def build_NoneType(self, o):
        if False:
            print('Hello World!')
        return None

def unrepr(s):
    if False:
        for i in range(10):
            print('nop')
    'Return a Python object compiled from a string.'
    if not s:
        return s
    b = _Builder()
    obj = b.astnode(s)
    return b.build(obj)

def modules(modulePath):
    if False:
        return 10
    'Load a module and retrieve a reference to that module.'
    __import__(modulePath)
    return sys.modules[modulePath]

def attributes(full_attribute_name):
    if False:
        return 10
    'Load a module and retrieve an attribute of that module.'
    last_dot = full_attribute_name.rfind('.')
    attr_name = full_attribute_name[last_dot + 1:]
    mod_path = full_attribute_name[:last_dot]
    mod = modules(mod_path)
    try:
        attr = getattr(mod, attr_name)
    except AttributeError:
        raise AttributeError("'%s' object has no attribute '%s'" % (mod_path, attr_name))
    return attr