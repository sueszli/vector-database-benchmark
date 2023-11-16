from __future__ import absolute_import
import cython
cython.declare(os=object, re=object, operator=object, textwrap=object, Template=object, Naming=object, Options=object, StringEncoding=object, Utils=object, SourceDescriptor=object, StringIOTree=object, DebugFlags=object, basestring=object, defaultdict=object, closing=object, partial=object)
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
try:
    from __builtin__ import basestring
except ImportError:
    from builtins import str as basestring
non_portable_builtins_map = {'bytes': ('PY_MAJOR_VERSION < 3', 'str'), 'unicode': ('PY_MAJOR_VERSION >= 3', 'str'), 'basestring': ('PY_MAJOR_VERSION >= 3', 'str'), 'xrange': ('PY_MAJOR_VERSION >= 3', 'range'), 'raw_input': ('PY_MAJOR_VERSION >= 3', 'input')}
ctypedef_builtins_map = {'py_int': '&PyInt_Type', 'py_long': '&PyLong_Type', 'py_float': '&PyFloat_Type', 'wrapper_descriptor': '&PyWrapperDescr_Type'}
basicsize_builtins_map = {'PyTypeObject': 'PyHeapTypeObject'}
uncachable_builtins = ['breakpoint', '__loader__', '__spec__', 'BlockingIOError', 'BrokenPipeError', 'ChildProcessError', 'ConnectionAbortedError', 'ConnectionError', 'ConnectionRefusedError', 'ConnectionResetError', 'FileExistsError', 'FileNotFoundError', 'InterruptedError', 'IsADirectoryError', 'ModuleNotFoundError', 'NotADirectoryError', 'PermissionError', 'ProcessLookupError', 'RecursionError', 'ResourceWarning', 'TimeoutError', '__build_class__', 'ascii', 'WindowsError', '_']
special_py_methods = cython.declare(frozenset, frozenset(('__cinit__', '__dealloc__', '__richcmp__', '__next__', '__await__', '__aiter__', '__anext__', '__getreadbuffer__', '__getwritebuffer__', '__getsegcount__', '__getcharbuffer__', '__getbuffer__', '__releasebuffer__')))
modifier_output_mapper = {'inline': 'CYTHON_INLINE'}.get

class IncludeCode(object):
    """
    An include file and/or verbatim C code to be included in the
    generated sources.
    """
    INITIAL = 0
    EARLY = 1
    LATE = 2
    counter = 1

    def __init__(self, include=None, verbatim=None, late=True, initial=False):
        if False:
            while True:
                i = 10
        self.order = self.counter
        type(self).counter += 1
        self.pieces = {}
        if include:
            if include[0] == '<' and include[-1] == '>':
                self.pieces[0] = u'#include {0}'.format(include)
                late = False
            else:
                self.pieces[0] = u'#include "{0}"'.format(include)
        if verbatim:
            self.pieces[self.order] = verbatim
        if initial:
            self.location = self.INITIAL
        elif late:
            self.location = self.LATE
        else:
            self.location = self.EARLY

    def dict_update(self, d, key):
        if False:
            return 10
        '\n        Insert `self` in dict `d` with key `key`. If that key already\n        exists, update the attributes of the existing value with `self`.\n        '
        if key in d:
            other = d[key]
            other.location = min(self.location, other.location)
            other.pieces.update(self.pieces)
        else:
            d[key] = self

    def sortkey(self):
        if False:
            i = 10
            return i + 15
        return self.order

    def mainpiece(self):
        if False:
            return 10
        '\n        Return the main piece of C code, corresponding to the include\n        file. If there was no include file, return None.\n        '
        return self.pieces.get(0)

    def write(self, code):
        if False:
            return 10
        for k in sorted(self.pieces):
            code.putln(self.pieces[k])

def get_utility_dir():
    if False:
        i = 10
        return i + 15
    Cython_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(Cython_dir, 'Utility')
read_utilities_hook = None
'\nOverride the hook for reading a utilities file that contains code fragments used\nby the codegen.\n\nThe hook functions takes the path of the utilities file, and returns a list\nof strings, one per line.\n\nThe default behavior is to open a file relative to get_utility_dir().\n'

def read_utilities_from_utility_dir(path):
    if False:
        i = 10
        return i + 15
    '\n    Read all lines of the file at the provided path from a path relative\n    to get_utility_dir().\n    '
    filename = os.path.join(get_utility_dir(), path)
    with closing(Utils.open_source_file(filename, encoding='UTF-8')) as f:
        return f.readlines()
read_utilities_hook = read_utilities_from_utility_dir

class UtilityCodeBase(object):
    """
    Support for loading utility code from a file.

    Code sections in the file can be specified as follows:

        ##### MyUtility.proto #####

        [proto declarations]

        ##### MyUtility.init #####

        [code run at module initialization]

        ##### MyUtility #####
        #@requires: MyOtherUtility
        #@substitute: naming

        [definitions]

        ##### MyUtility #####
        #@substitute: tempita

        [requires tempita substitution
         - context can't be specified here though so only
           tempita utility that requires no external context
           will benefit from this tag
         - only necessary when @required from non-tempita code]

    for prototypes and implementation respectively.  For non-python or
    -cython files backslashes should be used instead.  5 to 30 comment
    characters may be used on either side.

    If the @cname decorator is not used and this is a CythonUtilityCode,
    one should pass in the 'name' keyword argument to be used for name
    mangling of such entries.
    """
    is_cython_utility = False
    _utility_cache = {}

    @classmethod
    def _add_utility(cls, utility, type, lines, begin_lineno, tags=None):
        if False:
            return 10
        if utility is None:
            return
        code = '\n'.join(lines)
        if tags and 'substitute' in tags and ('naming' in tags['substitute']):
            try:
                code = Template(code).substitute(vars(Naming))
            except (KeyError, ValueError) as e:
                raise RuntimeError("Error parsing templated utility code of type '%s' at line %d: %s" % (type, begin_lineno, e))
        code = '\n' * begin_lineno + code
        if type == 'proto':
            utility[0] = code
        elif type == 'impl':
            utility[1] = code
        else:
            all_tags = utility[2]
            all_tags[type] = code
        if tags:
            all_tags = utility[2]
            for (name, values) in tags.items():
                all_tags.setdefault(name, set()).update(values)

    @classmethod
    def load_utilities_from_file(cls, path):
        if False:
            while True:
                i = 10
        utilities = cls._utility_cache.get(path)
        if utilities:
            return utilities
        (_, ext) = os.path.splitext(path)
        if ext in ('.pyx', '.py', '.pxd', '.pxi'):
            comment = '#'
            strip_comments = partial(re.compile('^\\s*#(?!\\s*cython\\s*:).*').sub, '')
            rstrip = StringEncoding._unicode.rstrip
        else:
            comment = '/'
            strip_comments = partial(re.compile('^\\s*//.*|/\\*[^*]*\\*/').sub, '')
            rstrip = partial(re.compile('\\s+(\\\\?)$').sub, '\\1')
        match_special = re.compile('^%(C)s{5,30}\\s*(?P<name>(?:\\w|\\.)+)\\s*%(C)s{5,30}|^%(C)s+@(?P<tag>\\w+)\\s*:\\s*(?P<value>(?:\\w|[.:])+)' % {'C': comment}).match
        match_type = re.compile('(.+)[.](proto(?:[.]\\S+)?|impl|init|cleanup)$').match
        all_lines = read_utilities_hook(path)
        utilities = defaultdict(lambda : [None, None, {}])
        lines = []
        tags = defaultdict(set)
        utility = type = None
        begin_lineno = 0
        for (lineno, line) in enumerate(all_lines):
            m = match_special(line)
            if m:
                if m.group('name'):
                    cls._add_utility(utility, type, lines, begin_lineno, tags)
                    begin_lineno = lineno + 1
                    del lines[:]
                    tags.clear()
                    name = m.group('name')
                    mtype = match_type(name)
                    if mtype:
                        (name, type) = mtype.groups()
                    else:
                        type = 'impl'
                    utility = utilities[name]
                else:
                    tags[m.group('tag')].add(m.group('value'))
                    lines.append('')
            else:
                lines.append(rstrip(strip_comments(line)))
        if utility is None:
            raise ValueError('Empty utility code file')
        cls._add_utility(utility, type, lines, begin_lineno, tags)
        utilities = dict(utilities)
        cls._utility_cache[path] = utilities
        return utilities

    @classmethod
    def load(cls, util_code_name, from_file, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Load utility code from a file specified by from_file (relative to\n        Cython/Utility) and name util_code_name.\n        '
        if '::' in util_code_name:
            (from_file, util_code_name) = util_code_name.rsplit('::', 1)
        assert from_file
        utilities = cls.load_utilities_from_file(from_file)
        (proto, impl, tags) = utilities[util_code_name]
        if tags:
            if 'substitute' in tags and 'tempita' in tags['substitute']:
                if not issubclass(cls, TempitaUtilityCode):
                    return TempitaUtilityCode.load(util_code_name, from_file, **kwargs)
            orig_kwargs = kwargs.copy()
            for (name, values) in tags.items():
                if name in kwargs:
                    continue
                if name == 'requires':
                    if orig_kwargs:
                        values = [cls.load(dep, from_file, **orig_kwargs) for dep in sorted(values)]
                    else:
                        values = [cls.load_cached(dep, from_file) for dep in sorted(values)]
                elif name == 'substitute':
                    values = values - {'naming', 'tempita'}
                    if not values:
                        continue
                elif not values:
                    values = None
                elif len(values) == 1:
                    values = list(values)[0]
                kwargs[name] = values
        if proto is not None:
            kwargs['proto'] = proto
        if impl is not None:
            kwargs['impl'] = impl
        if 'name' not in kwargs:
            kwargs['name'] = util_code_name
        if 'file' not in kwargs and from_file:
            kwargs['file'] = from_file
        return cls(**kwargs)

    @classmethod
    def load_cached(cls, utility_code_name, from_file, __cache={}):
        if False:
            while True:
                i = 10
        '\n        Calls .load(), but using a per-type cache based on utility name and file name.\n        '
        key = (utility_code_name, from_file, cls)
        try:
            return __cache[key]
        except KeyError:
            pass
        code = __cache[key] = cls.load(utility_code_name, from_file)
        return code

    @classmethod
    def load_as_string(cls, util_code_name, from_file, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Load a utility code as a string. Returns (proto, implementation)\n        '
        util = cls.load(util_code_name, from_file, **kwargs)
        (proto, impl) = (util.proto, util.impl)
        return (util.format_code(proto), util.format_code(impl))

    def format_code(self, code_string, replace_empty_lines=re.compile('\\n\\n+').sub):
        if False:
            while True:
                i = 10
        '\n        Format a code section for output.\n        '
        if code_string:
            code_string = replace_empty_lines('\n', code_string.strip()) + '\n\n'
        return code_string

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '<%s(%s)>' % (type(self).__name__, self.name)

    def get_tree(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return None

    def __deepcopy__(self, memodict=None):
        if False:
            while True:
                i = 10
        return self

class UtilityCode(UtilityCodeBase):
    """
    Stores utility code to add during code generation.

    See GlobalState.put_utility_code.

    hashes/equals by instance

    proto           C prototypes
    impl            implementation code
    init            code to call on module initialization
    requires        utility code dependencies
    proto_block     the place in the resulting file where the prototype should
                    end up
    name            name of the utility code (or None)
    file            filename of the utility code file this utility was loaded
                    from (or None)
    """

    def __init__(self, proto=None, impl=None, init=None, cleanup=None, requires=None, proto_block='utility_code_proto', name=None, file=None):
        if False:
            for i in range(10):
                print('nop')
        self.proto = proto
        self.impl = impl
        self.init = init
        self.cleanup = cleanup
        self.requires = requires
        self._cache = {}
        self.specialize_list = []
        self.proto_block = proto_block
        self.name = name
        self.file = file

    def __hash__(self):
        if False:
            return 10
        return hash((self.proto, self.impl))

    def __eq__(self, other):
        if False:
            return 10
        if self is other:
            return True
        (self_type, other_type) = (type(self), type(other))
        if self_type is not other_type and (not (isinstance(other, self_type) or isinstance(self, other_type))):
            return False
        self_proto = getattr(self, 'proto', None)
        other_proto = getattr(other, 'proto', None)
        return (self_proto, self.impl) == (other_proto, other.impl)

    def none_or_sub(self, s, context):
        if False:
            i = 10
            return i + 15
        '\n        Format a string in this utility code with context. If None, do nothing.\n        '
        if s is None:
            return None
        return s % context

    def specialize(self, pyrex_type=None, **data):
        if False:
            while True:
                i = 10
        name = self.name
        if pyrex_type is not None:
            data['type'] = pyrex_type.empty_declaration_code()
            data['type_name'] = pyrex_type.specialization_name()
            name = '%s[%s]' % (name, data['type_name'])
        key = tuple(sorted(data.items()))
        try:
            return self._cache[key]
        except KeyError:
            if self.requires is None:
                requires = None
            else:
                requires = [r.specialize(data) for r in self.requires]
            s = self._cache[key] = UtilityCode(self.none_or_sub(self.proto, data), self.none_or_sub(self.impl, data), self.none_or_sub(self.init, data), self.none_or_sub(self.cleanup, data), requires, self.proto_block, name)
            self.specialize_list.append(s)
            return s

    def inject_string_constants(self, impl, output):
        if False:
            print('Hello World!')
        'Replace \'PYIDENT("xyz")\' by a constant Python identifier cname.\n        '
        if 'PYIDENT(' not in impl and 'PYUNICODE(' not in impl:
            return (False, impl)
        replacements = {}

        def externalise(matchobj):
            if False:
                i = 10
                return i + 15
            key = matchobj.groups()
            try:
                cname = replacements[key]
            except KeyError:
                (str_type, name) = key
                cname = replacements[key] = output.get_py_string_const(StringEncoding.EncodedString(name), identifier=str_type == 'IDENT').cname
            return cname
        impl = re.sub('PY(IDENT|UNICODE)\\("([^"]+)"\\)', externalise, impl)
        assert 'PYIDENT(' not in impl and 'PYUNICODE(' not in impl
        return (True, impl)

    def inject_unbound_methods(self, impl, output):
        if False:
            for i in range(10):
                print('nop')
        'Replace \'UNBOUND_METHOD(type, "name")\' by a constant Python identifier cname.\n        '
        if 'CALL_UNBOUND_METHOD(' not in impl:
            return (False, impl)

        def externalise(matchobj):
            if False:
                for i in range(10):
                    print('nop')
            (type_cname, method_name, obj_cname, args) = matchobj.groups()
            type_cname = '&%s' % type_cname
            args = [arg.strip() for arg in args[1:].split(',')] if args else []
            assert len(args) < 3, 'CALL_UNBOUND_METHOD() does not support %d call arguments' % len(args)
            return output.cached_unbound_method_call_code(obj_cname, type_cname, method_name, args)
        impl = re.sub('CALL_UNBOUND_METHOD\\(([a-zA-Z_]+),\\s*"([^"]+)",\\s*([^),]+)((?:,[^),]+)*)\\)', externalise, impl)
        assert 'CALL_UNBOUND_METHOD(' not in impl
        return (True, impl)

    def wrap_c_strings(self, impl):
        if False:
            i = 10
            return i + 15
        "Replace CSTRING('''xyz''') by a C compatible string\n        "
        if 'CSTRING(' not in impl:
            return impl

        def split_string(matchobj):
            if False:
                while True:
                    i = 10
            content = matchobj.group(1).replace('"', '"')
            return ''.join(('"%s\\n"\n' % line if not line.endswith('\\') or line.endswith('\\\\') else '"%s"\n' % line[:-1] for line in content.splitlines()))
        impl = re.sub('CSTRING\\(\\s*"""([^"]*(?:"[^"]+)*)"""\\s*\\)', split_string, impl)
        assert 'CSTRING(' not in impl
        return impl

    def put_code(self, output):
        if False:
            for i in range(10):
                print('nop')
        if self.requires:
            for dependency in self.requires:
                output.use_utility_code(dependency)
        if self.proto:
            writer = output[self.proto_block]
            writer.putln('/* %s.proto */' % self.name)
            writer.put_or_include(self.format_code(self.proto), '%s_proto' % self.name)
        if self.impl:
            impl = self.format_code(self.wrap_c_strings(self.impl))
            (is_specialised1, impl) = self.inject_string_constants(impl, output)
            (is_specialised2, impl) = self.inject_unbound_methods(impl, output)
            writer = output['utility_code_def']
            writer.putln('/* %s */' % self.name)
            if not (is_specialised1 or is_specialised2):
                writer.put_or_include(impl, '%s_impl' % self.name)
            else:
                writer.put(impl)
        if self.init:
            writer = output['init_globals']
            writer.putln('/* %s.init */' % self.name)
            if isinstance(self.init, basestring):
                writer.put(self.format_code(self.init))
            else:
                self.init(writer, output.module_pos)
            writer.putln(writer.error_goto_if_PyErr(output.module_pos))
            writer.putln()
        if self.cleanup and Options.generate_cleanup_code:
            writer = output['cleanup_globals']
            writer.putln('/* %s.cleanup */' % self.name)
            if isinstance(self.cleanup, basestring):
                writer.put_or_include(self.format_code(self.cleanup), '%s_cleanup' % self.name)
            else:
                self.cleanup(writer, output.module_pos)

def sub_tempita(s, context, file=None, name=None):
    if False:
        print('Hello World!')
    'Run tempita on string s with given context.'
    if not s:
        return None
    if file:
        context['__name'] = '%s:%s' % (file, name)
    elif name:
        context['__name'] = name
    from ..Tempita import sub
    return sub(s, **context)

class TempitaUtilityCode(UtilityCode):

    def __init__(self, name=None, proto=None, impl=None, init=None, file=None, context=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if context is None:
            context = {}
        proto = sub_tempita(proto, context, file, name)
        impl = sub_tempita(impl, context, file, name)
        init = sub_tempita(init, context, file, name)
        super(TempitaUtilityCode, self).__init__(proto, impl, init=init, name=name, file=file, **kwargs)

    @classmethod
    def load_cached(cls, utility_code_name, from_file=None, context=None, __cache={}):
        if False:
            i = 10
            return i + 15
        context_key = tuple(sorted(context.items())) if context else None
        assert hash(context_key) is not None
        key = (cls, from_file, utility_code_name, context_key)
        try:
            return __cache[key]
        except KeyError:
            pass
        code = __cache[key] = cls.load(utility_code_name, from_file, context=context)
        return code

    def none_or_sub(self, s, context):
        if False:
            return 10
        '\n        Format a string in this utility code with context. If None, do nothing.\n        '
        if s is None:
            return None
        return sub_tempita(s, context, self.file, self.name)

class LazyUtilityCode(UtilityCodeBase):
    """
    Utility code that calls a callback with the root code writer when
    available. Useful when you only have 'env' but not 'code'.
    """
    __name__ = '<lazy>'
    requires = None

    def __init__(self, callback):
        if False:
            return 10
        self.callback = callback

    def put_code(self, globalstate):
        if False:
            i = 10
            return i + 15
        utility = self.callback(globalstate.rootwriter)
        globalstate.use_utility_code(utility)

class FunctionState(object):

    def __init__(self, owner, names_taken=set(), scope=None):
        if False:
            i = 10
            return i + 15
        self.names_taken = names_taken
        self.owner = owner
        self.scope = scope
        self.error_label = None
        self.label_counter = 0
        self.labels_used = set()
        self.return_label = self.new_label()
        self.new_error_label()
        self.continue_label = None
        self.break_label = None
        self.yield_labels = []
        self.in_try_finally = 0
        self.exc_vars = None
        self.current_except = None
        self.can_trace = False
        self.gil_owned = True
        self.temps_allocated = []
        self.temps_free = {}
        self.temps_used_type = {}
        self.zombie_temps = set()
        self.temp_counter = 0
        self.closure_temps = None
        self.collect_temps_stack = []
        self.should_declare_error_indicator = False
        self.uses_error_indicator = False
        self.error_without_exception = False
        self.needs_refnanny = False

    def validate_exit(self):
        if False:
            i = 10
            return i + 15
        if self.temps_allocated:
            leftovers = self.temps_in_use()
            if leftovers:
                msg = "TEMPGUARD: Temps left over at end of '%s': %s" % (self.scope.name, ', '.join(['%s [%s]' % (name, ctype) for (name, ctype, is_pytemp) in sorted(leftovers)]))
                raise RuntimeError(msg)

    def new_label(self, name=None):
        if False:
            while True:
                i = 10
        n = self.label_counter
        self.label_counter = n + 1
        label = '%s%d' % (Naming.label_prefix, n)
        if name is not None:
            label += '_' + name
        return label

    def new_yield_label(self, expr_type='yield'):
        if False:
            print('Hello World!')
        label = self.new_label('resume_from_%s' % expr_type)
        num_and_label = (len(self.yield_labels) + 1, label)
        self.yield_labels.append(num_and_label)
        return num_and_label

    def new_error_label(self, prefix=''):
        if False:
            print('Hello World!')
        old_err_lbl = self.error_label
        self.error_label = self.new_label(prefix + 'error')
        return old_err_lbl

    def get_loop_labels(self):
        if False:
            while True:
                i = 10
        return (self.continue_label, self.break_label)

    def set_loop_labels(self, labels):
        if False:
            print('Hello World!')
        (self.continue_label, self.break_label) = labels

    def new_loop_labels(self, prefix=''):
        if False:
            i = 10
            return i + 15
        old_labels = self.get_loop_labels()
        self.set_loop_labels((self.new_label(prefix + 'continue'), self.new_label(prefix + 'break')))
        return old_labels

    def get_all_labels(self):
        if False:
            return 10
        return (self.continue_label, self.break_label, self.return_label, self.error_label)

    def set_all_labels(self, labels):
        if False:
            while True:
                i = 10
        (self.continue_label, self.break_label, self.return_label, self.error_label) = labels

    def all_new_labels(self):
        if False:
            return 10
        old_labels = self.get_all_labels()
        new_labels = []
        for (old_label, name) in zip(old_labels, ['continue', 'break', 'return', 'error']):
            if old_label:
                new_labels.append(self.new_label(name))
            else:
                new_labels.append(old_label)
        self.set_all_labels(new_labels)
        return old_labels

    def use_label(self, lbl):
        if False:
            return 10
        self.labels_used.add(lbl)

    def label_used(self, lbl):
        if False:
            return 10
        return lbl in self.labels_used

    def allocate_temp(self, type, manage_ref, static=False, reusable=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Allocates a temporary (which may create a new one or get a previously\n        allocated and released one of the same type). Type is simply registered\n        and handed back, but will usually be a PyrexType.\n\n        If type.needs_refcounting, manage_ref comes into play. If manage_ref is set to\n        True, the temp will be decref-ed on return statements and in exception\n        handling clauses. Otherwise the caller has to deal with any reference\n        counting of the variable.\n\n        If not type.needs_refcounting, then manage_ref will be ignored, but it\n        still has to be passed. It is recommended to pass False by convention\n        if it is known that type will never be a reference counted type.\n\n        static=True marks the temporary declaration with "static".\n        This is only used when allocating backing store for a module-level\n        C array literals.\n\n        if reusable=False, the temp will not be reused after release.\n\n        A C string referring to the variable is returned.\n        '
        if type.is_cv_qualified and (not type.is_reference):
            type = type.cv_base_type
        elif type.is_reference and (not type.is_fake_reference):
            type = type.ref_base_type
        elif type.is_cfunction:
            from . import PyrexTypes
            type = PyrexTypes.c_ptr_type(type)
        elif type.is_cpp_class and (not type.is_fake_reference) and self.scope.directives['cpp_locals']:
            self.scope.use_utility_code(UtilityCode.load_cached('OptionalLocals', 'CppSupport.cpp'))
        if not type.needs_refcounting:
            manage_ref = False
        freelist = self.temps_free.get((type, manage_ref))
        if reusable and freelist is not None and freelist[0]:
            result = freelist[0].pop()
            freelist[1].remove(result)
        else:
            while True:
                self.temp_counter += 1
                result = '%s%d' % (Naming.codewriter_temp_prefix, self.temp_counter)
                if result not in self.names_taken:
                    break
            self.temps_allocated.append((result, type, manage_ref, static))
            if not reusable:
                self.zombie_temps.add(result)
        self.temps_used_type[result] = (type, manage_ref)
        if DebugFlags.debug_temp_code_comments:
            self.owner.putln('/* %s allocated (%s)%s */' % (result, type, '' if reusable else ' - zombie'))
        if self.collect_temps_stack:
            self.collect_temps_stack[-1].add((result, type))
        return result

    def release_temp(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Releases a temporary so that it can be reused by other code needing\n        a temp of the same type.\n        '
        (type, manage_ref) = self.temps_used_type[name]
        freelist = self.temps_free.get((type, manage_ref))
        if freelist is None:
            freelist = ([], set())
            self.temps_free[type, manage_ref] = freelist
        if name in freelist[1]:
            raise RuntimeError('Temp %s freed twice!' % name)
        if name not in self.zombie_temps:
            freelist[0].append(name)
        freelist[1].add(name)
        if DebugFlags.debug_temp_code_comments:
            self.owner.putln('/* %s released %s*/' % (name, ' - zombie' if name in self.zombie_temps else ''))

    def temps_in_use(self):
        if False:
            i = 10
            return i + 15
        'Return a list of (cname,type,manage_ref) tuples of temp names and their type\n        that are currently in use.\n        '
        used = []
        for (name, type, manage_ref, static) in self.temps_allocated:
            freelist = self.temps_free.get((type, manage_ref))
            if freelist is None or name not in freelist[1]:
                used.append((name, type, manage_ref and type.needs_refcounting))
        return used

    def temps_holding_reference(self):
        if False:
            i = 10
            return i + 15
        'Return a list of (cname,type) tuples of temp names and their type\n        that are currently in use. This includes only temps\n        with a reference counted type which owns its reference.\n        '
        return [(name, type) for (name, type, manage_ref) in self.temps_in_use() if manage_ref and type.needs_refcounting]

    def all_managed_temps(self):
        if False:
            print('Hello World!')
        'Return a list of (cname, type) tuples of refcount-managed Python objects.\n        '
        return [(cname, type) for (cname, type, manage_ref, static) in self.temps_allocated if manage_ref]

    def all_free_managed_temps(self):
        if False:
            return 10
        'Return a list of (cname, type) tuples of refcount-managed Python\n        objects that are not currently in use.  This is used by\n        try-except and try-finally blocks to clean up temps in the\n        error case.\n        '
        return sorted([(cname, type) for ((type, manage_ref), freelist) in self.temps_free.items() if manage_ref for cname in freelist[0]])

    def start_collecting_temps(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Useful to find out which temps were used in a code block\n        '
        self.collect_temps_stack.append(set())

    def stop_collecting_temps(self):
        if False:
            print('Hello World!')
        return self.collect_temps_stack.pop()

    def init_closure_temps(self, scope):
        if False:
            for i in range(10):
                print('nop')
        self.closure_temps = ClosureTempAllocator(scope)

class NumConst(object):
    """Global info about a Python number constant held by GlobalState.

    cname       string
    value       string
    py_type     string     int, long, float
    value_code  string     evaluation code if different from value
    """

    def __init__(self, cname, value, py_type, value_code=None):
        if False:
            i = 10
            return i + 15
        self.cname = cname
        self.value = value
        self.py_type = py_type
        self.value_code = value_code or value

class PyObjectConst(object):
    """Global info about a generic constant held by GlobalState.
    """

    def __init__(self, cname, type):
        if False:
            print('Hello World!')
        self.cname = cname
        self.type = type
cython.declare(possible_unicode_identifier=object, possible_bytes_identifier=object, replace_identifier=object, find_alphanums=object)
possible_unicode_identifier = re.compile(b'(?![0-9])\\w+$'.decode('ascii'), re.U).match
possible_bytes_identifier = re.compile('(?![0-9])\\w+$'.encode('ASCII')).match
replace_identifier = re.compile('[^a-zA-Z0-9_]+').sub
find_alphanums = re.compile('([a-zA-Z0-9]+)').findall

class StringConst(object):
    """Global info about a C string constant held by GlobalState.
    """

    def __init__(self, cname, text, byte_string):
        if False:
            while True:
                i = 10
        self.cname = cname
        self.text = text
        self.escaped_value = StringEncoding.escape_byte_string(byte_string)
        self.py_strings = None
        self.py_versions = []

    def add_py_version(self, version):
        if False:
            return 10
        if not version:
            self.py_versions = [2, 3]
        elif version not in self.py_versions:
            self.py_versions.append(version)

    def get_py_string_const(self, encoding, identifier=None, is_str=False, py3str_cstring=None):
        if False:
            for i in range(10):
                print('nop')
        py_strings = self.py_strings
        text = self.text
        is_str = bool(identifier or is_str)
        is_unicode = encoding is None and (not is_str)
        if encoding is None:
            encoding_key = None
        else:
            encoding = encoding.lower()
            if encoding in ('utf8', 'utf-8', 'ascii', 'usascii', 'us-ascii'):
                encoding = None
                encoding_key = None
            else:
                encoding_key = ''.join(find_alphanums(encoding))
        key = (is_str, is_unicode, encoding_key, py3str_cstring)
        if py_strings is not None:
            try:
                return py_strings[key]
            except KeyError:
                pass
        else:
            self.py_strings = {}
        if identifier:
            intern = True
        elif identifier is None:
            if isinstance(text, bytes):
                intern = bool(possible_bytes_identifier(text))
            else:
                intern = bool(possible_unicode_identifier(text))
        else:
            intern = False
        if intern:
            prefix = Naming.interned_prefixes['str']
        else:
            prefix = Naming.py_const_prefix
        if encoding_key:
            encoding_prefix = '_%s' % encoding_key
        else:
            encoding_prefix = ''
        pystring_cname = '%s%s%s_%s' % (prefix, is_str and 's' or (is_unicode and 'u') or 'b', encoding_prefix, self.cname[len(Naming.const_prefix):])
        py_string = PyStringConst(pystring_cname, encoding, is_unicode, is_str, py3str_cstring, intern)
        self.py_strings[key] = py_string
        return py_string

class PyStringConst(object):
    """Global info about a Python string constant held by GlobalState.
    """

    def __init__(self, cname, encoding, is_unicode, is_str=False, py3str_cstring=None, intern=False):
        if False:
            print('Hello World!')
        self.cname = cname
        self.py3str_cstring = py3str_cstring
        self.encoding = encoding
        self.is_str = is_str
        self.is_unicode = is_unicode
        self.intern = intern

    def __lt__(self, other):
        if False:
            return 10
        return self.cname < other.cname

class GlobalState(object):
    directives = {}
    code_layout = ['h_code', 'filename_table', 'utility_code_proto_before_types', 'numeric_typedefs', 'complex_type_declarations', 'type_declarations', 'utility_code_proto', 'module_declarations', 'typeinfo', 'before_global_var', 'global_var', 'string_decls', 'decls', 'late_includes', 'module_state', 'module_state_clear', 'module_state_traverse', 'module_state_defines', 'module_code', 'pystring_table', 'cached_builtins', 'cached_constants', 'init_constants', 'init_globals', 'init_module', 'cleanup_globals', 'cleanup_module', 'main_method', 'utility_code_pragmas', 'utility_code_def', 'utility_code_pragmas_end', 'end']
    h_code_layout = ['h_code', 'utility_code_proto_before_types', 'type_declarations', 'utility_code_proto', 'end']

    def __init__(self, writer, module_node, code_config, common_utility_include_dir=None):
        if False:
            print('Hello World!')
        self.filename_table = {}
        self.filename_list = []
        self.input_file_contents = {}
        self.utility_codes = set()
        self.declared_cnames = {}
        self.in_utility_code_generation = False
        self.code_config = code_config
        self.common_utility_include_dir = common_utility_include_dir
        self.parts = {}
        self.module_node = module_node
        self.const_cnames_used = {}
        self.string_const_index = {}
        self.dedup_const_index = {}
        self.pyunicode_ptr_const_index = {}
        self.num_const_index = {}
        self.py_constants = []
        self.cached_cmethods = {}
        self.initialised_constants = set()
        writer.set_global_state(self)
        self.rootwriter = writer

    def initialize_main_c_code(self):
        if False:
            for i in range(10):
                print('nop')
        rootwriter = self.rootwriter
        for (i, part) in enumerate(self.code_layout):
            w = self.parts[part] = rootwriter.insertion_point()
            if i > 0:
                w.putln('/* #### Code section: %s ### */' % part)
        if not Options.cache_builtins:
            del self.parts['cached_builtins']
        else:
            w = self.parts['cached_builtins']
            w.enter_cfunc_scope()
            w.putln('static CYTHON_SMALL_CODE int __Pyx_InitCachedBuiltins(void) {')
        w = self.parts['cached_constants']
        w.enter_cfunc_scope()
        w.putln('')
        w.putln('static CYTHON_SMALL_CODE int __Pyx_InitCachedConstants(void) {')
        w.put_declare_refcount_context()
        w.put_setup_refcount_context(StringEncoding.EncodedString('__Pyx_InitCachedConstants'))
        w = self.parts['init_globals']
        w.enter_cfunc_scope()
        w.putln('')
        w.putln('static CYTHON_SMALL_CODE int __Pyx_InitGlobals(void) {')
        w = self.parts['init_constants']
        w.enter_cfunc_scope()
        w.putln('')
        w.putln('static CYTHON_SMALL_CODE int __Pyx_InitConstants(void) {')
        if not Options.generate_cleanup_code:
            del self.parts['cleanup_globals']
        else:
            w = self.parts['cleanup_globals']
            w.enter_cfunc_scope()
            w.putln('')
            w.putln('static CYTHON_SMALL_CODE void __Pyx_CleanupGlobals(void) {')
        code = self.parts['utility_code_proto']
        code.putln('')
        code.putln('/* --- Runtime support code (head) --- */')
        code = self.parts['utility_code_def']
        if self.code_config.emit_linenums:
            code.write('\n#line 1 "cython_utility"\n')
        code.putln('')
        code.putln('/* --- Runtime support code --- */')

    def initialize_main_h_code(self):
        if False:
            return 10
        rootwriter = self.rootwriter
        for part in self.h_code_layout:
            self.parts[part] = rootwriter.insertion_point()

    def finalize_main_c_code(self):
        if False:
            print('Hello World!')
        self.close_global_decls()
        code = self.parts['utility_code_def']
        util = TempitaUtilityCode.load_cached('TypeConversions', 'TypeConversion.c')
        code.put(util.format_code(util.impl))
        code.putln('')
        code = self.parts['utility_code_pragmas']
        util = UtilityCode.load_cached('UtilityCodePragmas', 'ModuleSetupCode.c')
        code.putln(util.format_code(util.impl))
        code.putln('')
        code = self.parts['utility_code_pragmas_end']
        util = UtilityCode.load_cached('UtilityCodePragmasEnd', 'ModuleSetupCode.c')
        code.putln(util.format_code(util.impl))
        code.putln('')

    def __getitem__(self, key):
        if False:
            return 10
        return self.parts[key]

    def close_global_decls(self):
        if False:
            return 10
        self.generate_const_declarations()
        if Options.cache_builtins:
            w = self.parts['cached_builtins']
            w.putln('return 0;')
            if w.label_used(w.error_label):
                w.put_label(w.error_label)
                w.putln('return -1;')
            w.putln('}')
            w.exit_cfunc_scope()
        w = self.parts['cached_constants']
        w.put_finish_refcount_context()
        w.putln('return 0;')
        if w.label_used(w.error_label):
            w.put_label(w.error_label)
            w.put_finish_refcount_context()
            w.putln('return -1;')
        w.putln('}')
        w.exit_cfunc_scope()
        for part in ['init_globals', 'init_constants']:
            w = self.parts[part]
            w.putln('return 0;')
            if w.label_used(w.error_label):
                w.put_label(w.error_label)
                w.putln('return -1;')
            w.putln('}')
            w.exit_cfunc_scope()
        if Options.generate_cleanup_code:
            w = self.parts['cleanup_globals']
            w.putln('}')
            w.exit_cfunc_scope()
        if Options.generate_cleanup_code:
            w = self.parts['cleanup_module']
            w.putln('}')
            w.exit_cfunc_scope()

    def put_pyobject_decl(self, entry):
        if False:
            i = 10
            return i + 15
        self['global_var'].putln('static PyObject *%s;' % entry.cname)

    def get_cached_constants_writer(self, target=None):
        if False:
            while True:
                i = 10
        if target is not None:
            if target in self.initialised_constants:
                return None
            self.initialised_constants.add(target)
        return self.parts['cached_constants']

    def get_int_const(self, str_value, longness=False):
        if False:
            i = 10
            return i + 15
        py_type = longness and 'long' or 'int'
        try:
            c = self.num_const_index[str_value, py_type]
        except KeyError:
            c = self.new_num_const(str_value, py_type)
        return c

    def get_float_const(self, str_value, value_code):
        if False:
            while True:
                i = 10
        try:
            c = self.num_const_index[str_value, 'float']
        except KeyError:
            c = self.new_num_const(str_value, 'float', value_code)
        return c

    def get_py_const(self, type, prefix='', cleanup_level=None, dedup_key=None):
        if False:
            while True:
                i = 10
        if dedup_key is not None:
            const = self.dedup_const_index.get(dedup_key)
            if const is not None:
                return const
        const = self.new_py_const(type, prefix)
        if cleanup_level is not None and cleanup_level <= Options.generate_cleanup_code and type.needs_refcounting:
            cleanup_writer = self.parts['cleanup_globals']
            cleanup_writer.putln('Py_CLEAR(%s);' % const.cname)
        if dedup_key is not None:
            self.dedup_const_index[dedup_key] = const
        return const

    def get_string_const(self, text, py_version=None):
        if False:
            return 10
        if text.is_unicode:
            byte_string = text.utf8encode()
        else:
            byte_string = text.byteencode()
        try:
            c = self.string_const_index[byte_string]
        except KeyError:
            c = self.new_string_const(text, byte_string)
        c.add_py_version(py_version)
        return c

    def get_pyunicode_ptr_const(self, text):
        if False:
            for i in range(10):
                print('nop')
        assert text.is_unicode
        try:
            c = self.pyunicode_ptr_const_index[text]
        except KeyError:
            c = self.pyunicode_ptr_const_index[text] = self.new_const_cname()
        return c

    def get_py_string_const(self, text, identifier=None, is_str=False, unicode_value=None):
        if False:
            for i in range(10):
                print('nop')
        py3str_cstring = None
        if is_str and unicode_value is not None and (unicode_value.utf8encode() != text.byteencode()):
            py3str_cstring = self.get_string_const(unicode_value, py_version=3)
            c_string = self.get_string_const(text, py_version=2)
        else:
            c_string = self.get_string_const(text)
        py_string = c_string.get_py_string_const(text.encoding, identifier, is_str, py3str_cstring)
        return py_string

    def get_interned_identifier(self, text):
        if False:
            print('Hello World!')
        return self.get_py_string_const(text, identifier=True)

    def new_string_const(self, text, byte_string):
        if False:
            for i in range(10):
                print('nop')
        cname = self.new_string_const_cname(byte_string)
        c = StringConst(cname, text, byte_string)
        self.string_const_index[byte_string] = c
        return c

    def new_num_const(self, value, py_type, value_code=None):
        if False:
            print('Hello World!')
        cname = self.new_num_const_cname(value, py_type)
        c = NumConst(cname, value, py_type, value_code)
        self.num_const_index[value, py_type] = c
        return c

    def new_py_const(self, type, prefix=''):
        if False:
            for i in range(10):
                print('nop')
        cname = self.new_const_cname(prefix)
        c = PyObjectConst(cname, type)
        self.py_constants.append(c)
        return c

    def new_string_const_cname(self, bytes_value):
        if False:
            return 10
        value = bytes_value.decode('ASCII', 'ignore')
        return self.new_const_cname(value=value)

    def unique_const_cname(self, format_str):
        if False:
            print('Hello World!')
        used = self.const_cnames_used
        cname = value = format_str.format(sep='', counter='')
        while cname in used:
            counter = used[value] = used[value] + 1
            cname = format_str.format(sep='_', counter=counter)
        used[cname] = 1
        return cname

    def new_num_const_cname(self, value, py_type):
        if False:
            print('Hello World!')
        if py_type == 'long':
            value += 'L'
            py_type = 'int'
        prefix = Naming.interned_prefixes[py_type]
        value = value.replace('.', '_').replace('+', '_').replace('-', 'neg_')
        if len(value) > 42:
            cname = self.unique_const_cname(prefix + 'large{counter}_' + value[:18] + '_xxx_' + value[-18:])
        else:
            cname = '%s%s' % (prefix, value)
        return cname

    def new_const_cname(self, prefix='', value=''):
        if False:
            i = 10
            return i + 15
        value = replace_identifier('_', value)[:32].strip('_')
        name_suffix = self.unique_const_cname(value + '{sep}{counter}')
        if prefix:
            prefix = Naming.interned_prefixes[prefix]
        else:
            prefix = Naming.const_prefix
        return '%s%s' % (prefix, name_suffix)

    def get_cached_unbound_method(self, type_cname, method_name):
        if False:
            return 10
        key = (type_cname, method_name)
        try:
            cname = self.cached_cmethods[key]
        except KeyError:
            cname = self.cached_cmethods[key] = self.new_const_cname('umethod', '%s_%s' % (type_cname, method_name))
        return cname

    def cached_unbound_method_call_code(self, obj_cname, type_cname, method_name, arg_cnames):
        if False:
            while True:
                i = 10
        utility_code_name = 'CallUnboundCMethod%d' % len(arg_cnames)
        self.use_utility_code(UtilityCode.load_cached(utility_code_name, 'ObjectHandling.c'))
        cache_cname = self.get_cached_unbound_method(type_cname, method_name)
        args = [obj_cname] + arg_cnames
        return '__Pyx_%s(&%s, %s)' % (utility_code_name, cache_cname, ', '.join(args))

    def add_cached_builtin_decl(self, entry):
        if False:
            while True:
                i = 10
        if entry.is_builtin and entry.is_const:
            if self.should_declare(entry.cname, entry):
                self.put_pyobject_decl(entry)
                w = self.parts['cached_builtins']
                condition = None
                if entry.name in non_portable_builtins_map:
                    (condition, replacement) = non_portable_builtins_map[entry.name]
                    w.putln('#if %s' % condition)
                    self.put_cached_builtin_init(entry.pos, StringEncoding.EncodedString(replacement), entry.cname)
                    w.putln('#else')
                self.put_cached_builtin_init(entry.pos, StringEncoding.EncodedString(entry.name), entry.cname)
                if condition:
                    w.putln('#endif')

    def put_cached_builtin_init(self, pos, name, cname):
        if False:
            return 10
        w = self.parts['cached_builtins']
        interned_cname = self.get_interned_identifier(name).cname
        self.use_utility_code(UtilityCode.load_cached('GetBuiltinName', 'ObjectHandling.c'))
        w.putln('%s = __Pyx_GetBuiltinName(%s); if (!%s) %s' % (cname, interned_cname, cname, w.error_goto(pos)))

    def generate_const_declarations(self):
        if False:
            for i in range(10):
                print('nop')
        self.generate_cached_methods_decls()
        self.generate_string_constants()
        self.generate_num_constants()
        self.generate_object_constant_decls()

    def generate_object_constant_decls(self):
        if False:
            while True:
                i = 10
        consts = [(len(c.cname), c.cname, c) for c in self.py_constants]
        consts.sort()
        for (_, cname, c) in consts:
            self.parts['module_state'].putln('%s;' % c.type.declaration_code(cname))
            self.parts['module_state_defines'].putln('#define %s %s->%s' % (cname, Naming.modulestateglobal_cname, cname))
            if not c.type.needs_refcounting:
                continue
            self.parts['module_state_clear'].putln('Py_CLEAR(clear_module_state->%s);' % cname)
            self.parts['module_state_traverse'].putln('Py_VISIT(traverse_module_state->%s);' % cname)

    def generate_cached_methods_decls(self):
        if False:
            return 10
        if not self.cached_cmethods:
            return
        decl = self.parts['decls']
        init = self.parts['init_constants']
        cnames = []
        for ((type_cname, method_name), cname) in sorted(self.cached_cmethods.items()):
            cnames.append(cname)
            method_name_cname = self.get_interned_identifier(StringEncoding.EncodedString(method_name)).cname
            decl.putln('static __Pyx_CachedCFunction %s = {0, 0, 0, 0, 0};' % cname)
            init.putln('%s.type = (PyObject*)%s;' % (cname, type_cname))
            init.putln('%s.method_name = &%s;' % (cname, method_name_cname))
        if Options.generate_cleanup_code:
            cleanup = self.parts['cleanup_globals']
            for cname in cnames:
                cleanup.putln('Py_CLEAR(%s.method);' % cname)

    def generate_string_constants(self):
        if False:
            i = 10
            return i + 15
        c_consts = [(len(c.cname), c.cname, c) for c in self.string_const_index.values()]
        c_consts.sort()
        py_strings = []
        decls_writer = self.parts['string_decls']
        for (_, cname, c) in c_consts:
            conditional = False
            if c.py_versions and (2 not in c.py_versions or 3 not in c.py_versions):
                conditional = True
                decls_writer.putln('#if PY_MAJOR_VERSION %s 3' % (2 in c.py_versions and '<' or '>='))
            decls_writer.putln('static const char %s[] = "%s";' % (cname, StringEncoding.split_string_literal(c.escaped_value)))
            if conditional:
                decls_writer.putln('#endif')
            if c.py_strings is not None:
                for py_string in c.py_strings.values():
                    py_strings.append((c.cname, len(py_string.cname), py_string))
        for (c, cname) in sorted(self.pyunicode_ptr_const_index.items()):
            (utf16_array, utf32_array) = StringEncoding.encode_pyunicode_string(c)
            if utf16_array:
                decls_writer.putln('#ifdef Py_UNICODE_WIDE')
            decls_writer.putln('static Py_UNICODE %s[] = { %s };' % (cname, utf32_array))
            if utf16_array:
                decls_writer.putln('#else')
                decls_writer.putln('static Py_UNICODE %s[] = { %s };' % (cname, utf16_array))
                decls_writer.putln('#endif')
        init_constants = self.parts['init_constants']
        if py_strings:
            self.use_utility_code(UtilityCode.load_cached('InitStrings', 'StringTools.c'))
            py_strings.sort()
            w = self.parts['pystring_table']
            w.putln('')
            w.putln('static int __Pyx_CreateStringTabAndInitStrings(void) {')
            w.putln('__Pyx_StringTabEntry %s[] = {' % Naming.stringtab_cname)
            for py_string_args in py_strings:
                (c_cname, _, py_string) = py_string_args
                if not py_string.is_str or not py_string.encoding or py_string.encoding in ('ASCII', 'USASCII', 'US-ASCII', 'UTF8', 'UTF-8'):
                    encoding = '0'
                else:
                    encoding = '"%s"' % py_string.encoding.lower()
                self.parts['module_state'].putln('PyObject *%s;' % py_string.cname)
                self.parts['module_state_defines'].putln('#define %s %s->%s' % (py_string.cname, Naming.modulestateglobal_cname, py_string.cname))
                self.parts['module_state_clear'].putln('Py_CLEAR(clear_module_state->%s);' % py_string.cname)
                self.parts['module_state_traverse'].putln('Py_VISIT(traverse_module_state->%s);' % py_string.cname)
                if py_string.py3str_cstring:
                    w.putln('#if PY_MAJOR_VERSION >= 3')
                    w.putln('{&%s, %s, sizeof(%s), %s, %d, %d, %d},' % (py_string.cname, py_string.py3str_cstring.cname, py_string.py3str_cstring.cname, '0', 1, 0, py_string.intern))
                    w.putln('#else')
                w.putln('{&%s, %s, sizeof(%s), %s, %d, %d, %d},' % (py_string.cname, c_cname, c_cname, encoding, py_string.is_unicode, py_string.is_str, py_string.intern))
                if py_string.py3str_cstring:
                    w.putln('#endif')
            w.putln('{0, 0, 0, 0, 0, 0, 0}')
            w.putln('};')
            w.putln('return __Pyx_InitStrings(%s);' % Naming.stringtab_cname)
            w.putln('}')
            init_constants.putln('if (__Pyx_CreateStringTabAndInitStrings() < 0) %s;' % init_constants.error_goto(self.module_pos))

    def generate_num_constants(self):
        if False:
            i = 10
            return i + 15
        consts = [(c.py_type, c.value[0] == '-', len(c.value), c.value, c.value_code, c) for c in self.num_const_index.values()]
        consts.sort()
        init_constants = self.parts['init_constants']
        for (py_type, _, _, value, value_code, c) in consts:
            cname = c.cname
            self.parts['module_state'].putln('PyObject *%s;' % cname)
            self.parts['module_state_defines'].putln('#define %s %s->%s' % (cname, Naming.modulestateglobal_cname, cname))
            self.parts['module_state_clear'].putln('Py_CLEAR(clear_module_state->%s);' % cname)
            self.parts['module_state_traverse'].putln('Py_VISIT(traverse_module_state->%s);' % cname)
            if py_type == 'float':
                function = 'PyFloat_FromDouble(%s)'
            elif py_type == 'long':
                function = 'PyLong_FromString("%s", 0, 0)'
            elif Utils.long_literal(value):
                function = 'PyInt_FromString("%s", 0, 0)'
            elif len(value.lstrip('-')) > 4:
                function = 'PyInt_FromLong(%sL)'
            else:
                function = 'PyInt_FromLong(%s)'
            init_constants.putln('%s = %s; %s' % (cname, function % value_code, init_constants.error_goto_if_null(cname, self.module_pos)))

    def should_declare(self, cname, entry):
        if False:
            for i in range(10):
                print('nop')
        if cname in self.declared_cnames:
            other = self.declared_cnames[cname]
            assert str(entry.type) == str(other.type)
            assert entry.init == other.init
            return False
        else:
            self.declared_cnames[cname] = entry
            return True

    def lookup_filename(self, source_desc):
        if False:
            return 10
        entry = source_desc.get_filenametable_entry()
        try:
            index = self.filename_table[entry]
        except KeyError:
            index = len(self.filename_list)
            self.filename_list.append(source_desc)
            self.filename_table[entry] = index
        return index

    def commented_file_contents(self, source_desc):
        if False:
            while True:
                i = 10
        try:
            return self.input_file_contents[source_desc]
        except KeyError:
            pass
        source_file = source_desc.get_lines(encoding='ASCII', error_handling='ignore')
        try:
            F = [u' * ' + line.rstrip().replace(u'*/', u'*[inserted by cython to avoid comment closer]/').replace(u'/*', u'/[inserted by cython to avoid comment start]*') for line in source_file]
        finally:
            if hasattr(source_file, 'close'):
                source_file.close()
        if not F:
            F.append(u'')
        self.input_file_contents[source_desc] = F
        return F

    def use_utility_code(self, utility_code):
        if False:
            return 10
        '\n        Adds code to the C file. utility_code should\n        a) implement __eq__/__hash__ for the purpose of knowing whether the same\n           code has already been included\n        b) implement put_code, which takes a globalstate instance\n\n        See UtilityCode.\n        '
        if utility_code and utility_code not in self.utility_codes:
            self.utility_codes.add(utility_code)
            utility_code.put_code(self)

    def use_entry_utility_code(self, entry):
        if False:
            while True:
                i = 10
        if entry is None:
            return
        if entry.utility_code:
            self.use_utility_code(entry.utility_code)
        if entry.utility_code_definition:
            self.use_utility_code(entry.utility_code_definition)

def funccontext_property(func):
    if False:
        print('Hello World!')
    name = func.__name__
    attribute_of = operator.attrgetter(name)

    def get(self):
        if False:
            while True:
                i = 10
        return attribute_of(self.funcstate)

    def set(self, value):
        if False:
            while True:
                i = 10
        setattr(self.funcstate, name, value)
    return property(get, set)

class CCodeConfig(object):

    def __init__(self, emit_linenums=True, emit_code_comments=True, c_line_in_traceback=True):
        if False:
            while True:
                i = 10
        self.emit_code_comments = emit_code_comments
        self.emit_linenums = emit_linenums
        self.c_line_in_traceback = c_line_in_traceback

class CCodeWriter(object):
    """
    Utility class to output C code.

    When creating an insertion point one must care about the state that is
    kept:
    - formatting state (level, bol) is cloned and used in insertion points
      as well
    - labels, temps, exc_vars: One must construct a scope in which these can
      exist by calling enter_cfunc_scope/exit_cfunc_scope (these are for
      sanity checking and forward compatibility). Created insertion points
      looses this scope and cannot access it.
    - marker: Not copied to insertion point
    - filename_table, filename_list, input_file_contents: All codewriters
      coming from the same root share the same instances simultaneously.
    """

    @cython.locals(create_from='CCodeWriter')
    def __init__(self, create_from=None, buffer=None, copy_formatting=False):
        if False:
            print('Hello World!')
        if buffer is None:
            buffer = StringIOTree()
        self.buffer = buffer
        self.last_pos = None
        self.last_marked_pos = None
        self.pyclass_stack = []
        self.funcstate = None
        self.globalstate = None
        self.code_config = None
        self.level = 0
        self.call_level = 0
        self.bol = 1
        if create_from is not None:
            self.set_global_state(create_from.globalstate)
            self.funcstate = create_from.funcstate
            if copy_formatting:
                self.level = create_from.level
                self.bol = create_from.bol
                self.call_level = create_from.call_level
            self.last_pos = create_from.last_pos
            self.last_marked_pos = create_from.last_marked_pos

    def create_new(self, create_from, buffer, copy_formatting):
        if False:
            i = 10
            return i + 15
        result = CCodeWriter(create_from, buffer, copy_formatting)
        return result

    def set_global_state(self, global_state):
        if False:
            for i in range(10):
                print('nop')
        assert self.globalstate is None
        self.globalstate = global_state
        self.code_config = global_state.code_config

    def copyto(self, f):
        if False:
            return 10
        self.buffer.copyto(f)

    def getvalue(self):
        if False:
            while True:
                i = 10
        return self.buffer.getvalue()

    def write(self, s):
        if False:
            print('Hello World!')
        if '\n' in s:
            self._write_lines(s)
        else:
            self._write_to_buffer(s)

    def _write_lines(self, s):
        if False:
            while True:
                i = 10
        filename_line = self.last_marked_pos[:2] if self.last_marked_pos else (None, 0)
        self.buffer.markers.extend([filename_line] * s.count('\n'))
        self._write_to_buffer(s)

    def _write_to_buffer(self, s):
        if False:
            print('Hello World!')
        self.buffer.write(s)

    def insertion_point(self):
        if False:
            print('Hello World!')
        other = self.create_new(create_from=self, buffer=self.buffer.insertion_point(), copy_formatting=True)
        return other

    def new_writer(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new CCodeWriter connected to the same global state, which\n        can later be inserted using insert.\n        '
        return CCodeWriter(create_from=self)

    def insert(self, writer):
        if False:
            print('Hello World!')
        '\n        Inserts the contents of another code writer (created with\n        the same global state) in the current location.\n\n        It is ok to write to the inserted writer also after insertion.\n        '
        assert writer.globalstate is self.globalstate
        self.buffer.insert(writer.buffer)

    @funccontext_property
    def label_counter(self):
        if False:
            i = 10
            return i + 15
        pass

    @funccontext_property
    def return_label(self):
        if False:
            i = 10
            return i + 15
        pass

    @funccontext_property
    def error_label(self):
        if False:
            return 10
        pass

    @funccontext_property
    def labels_used(self):
        if False:
            i = 10
            return i + 15
        pass

    @funccontext_property
    def continue_label(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @funccontext_property
    def break_label(self):
        if False:
            while True:
                i = 10
        pass

    @funccontext_property
    def return_from_error_cleanup_label(self):
        if False:
            while True:
                i = 10
        pass

    @funccontext_property
    def yield_labels(self):
        if False:
            return 10
        pass

    def label_interceptor(self, new_labels, orig_labels, skip_to_label=None, pos=None, trace=True):
        if False:
            print('Hello World!')
        '\n        Helper for generating multiple label interceptor code blocks.\n\n        @param new_labels: the new labels that should be intercepted\n        @param orig_labels: the original labels that we should dispatch to after the interception\n        @param skip_to_label: a label to skip to before starting the code blocks\n        @param pos: the node position to mark for each interceptor block\n        @param trace: add a trace line for the pos marker or not\n        '
        for (label, orig_label) in zip(new_labels, orig_labels):
            if not self.label_used(label):
                continue
            if skip_to_label:
                self.put_goto(skip_to_label)
                skip_to_label = None
            if pos is not None:
                self.mark_pos(pos, trace=trace)
            self.put_label(label)
            yield (label, orig_label)
            self.put_goto(orig_label)

    def new_label(self, name=None):
        if False:
            while True:
                i = 10
        return self.funcstate.new_label(name)

    def new_error_label(self, *args):
        if False:
            i = 10
            return i + 15
        return self.funcstate.new_error_label(*args)

    def new_yield_label(self, *args):
        if False:
            while True:
                i = 10
        return self.funcstate.new_yield_label(*args)

    def get_loop_labels(self):
        if False:
            i = 10
            return i + 15
        return self.funcstate.get_loop_labels()

    def set_loop_labels(self, labels):
        if False:
            print('Hello World!')
        return self.funcstate.set_loop_labels(labels)

    def new_loop_labels(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self.funcstate.new_loop_labels(*args)

    def get_all_labels(self):
        if False:
            while True:
                i = 10
        return self.funcstate.get_all_labels()

    def set_all_labels(self, labels):
        if False:
            for i in range(10):
                print('nop')
        return self.funcstate.set_all_labels(labels)

    def all_new_labels(self):
        if False:
            for i in range(10):
                print('nop')
        return self.funcstate.all_new_labels()

    def use_label(self, lbl):
        if False:
            i = 10
            return i + 15
        return self.funcstate.use_label(lbl)

    def label_used(self, lbl):
        if False:
            return 10
        return self.funcstate.label_used(lbl)

    def enter_cfunc_scope(self, scope=None):
        if False:
            return 10
        self.funcstate = FunctionState(self, scope=scope)

    def exit_cfunc_scope(self):
        if False:
            return 10
        self.funcstate.validate_exit()
        self.funcstate = None

    def get_py_int(self, str_value, longness):
        if False:
            i = 10
            return i + 15
        return self.globalstate.get_int_const(str_value, longness).cname

    def get_py_float(self, str_value, value_code):
        if False:
            for i in range(10):
                print('nop')
        return self.globalstate.get_float_const(str_value, value_code).cname

    def get_py_const(self, type, prefix='', cleanup_level=None, dedup_key=None):
        if False:
            for i in range(10):
                print('nop')
        return self.globalstate.get_py_const(type, prefix, cleanup_level, dedup_key).cname

    def get_string_const(self, text):
        if False:
            print('Hello World!')
        return self.globalstate.get_string_const(text).cname

    def get_pyunicode_ptr_const(self, text):
        if False:
            print('Hello World!')
        return self.globalstate.get_pyunicode_ptr_const(text)

    def get_py_string_const(self, text, identifier=None, is_str=False, unicode_value=None):
        if False:
            while True:
                i = 10
        return self.globalstate.get_py_string_const(text, identifier, is_str, unicode_value).cname

    def get_argument_default_const(self, type):
        if False:
            while True:
                i = 10
        return self.globalstate.get_py_const(type).cname

    def intern(self, text):
        if False:
            i = 10
            return i + 15
        return self.get_py_string_const(text)

    def intern_identifier(self, text):
        if False:
            return 10
        return self.get_py_string_const(text, identifier=True)

    def get_cached_constants_writer(self, target=None):
        if False:
            for i in range(10):
                print('nop')
        return self.globalstate.get_cached_constants_writer(target)

    def putln(self, code='', safe=False):
        if False:
            return 10
        if self.last_pos and self.bol:
            self.emit_marker()
        if self.code_config.emit_linenums and self.last_marked_pos:
            (source_desc, line, _) = self.last_marked_pos
            self._write_lines('\n#line %s "%s"\n' % (line, source_desc.get_escaped_description()))
        if code:
            if safe:
                self.put_safe(code)
            else:
                self.put(code)
        self._write_lines('\n')
        self.bol = 1

    def mark_pos(self, pos, trace=True):
        if False:
            return 10
        if pos is None:
            return
        if self.last_marked_pos and self.last_marked_pos[:2] == pos[:2]:
            return
        self.last_pos = (pos, trace)

    def emit_marker(self):
        if False:
            return 10
        (pos, trace) = self.last_pos
        self.last_marked_pos = pos
        self.last_pos = None
        self._write_lines('\n')
        if self.code_config.emit_code_comments:
            self.indent()
            self._write_lines('/* %s */\n' % self._build_marker(pos))
        if trace and self.funcstate and self.funcstate.can_trace and self.globalstate.directives['linetrace']:
            self.indent()
            self._write_lines('__Pyx_TraceLine(%d,%d,%s)\n' % (pos[1], not self.funcstate.gil_owned, self.error_goto(pos)))

    def _build_marker(self, pos):
        if False:
            i = 10
            return i + 15
        (source_desc, line, col) = pos
        assert isinstance(source_desc, SourceDescriptor)
        contents = self.globalstate.commented_file_contents(source_desc)
        lines = contents[max(0, line - 3):line]
        lines[-1] += u'             # <<<<<<<<<<<<<<'
        lines += contents[line:line + 2]
        return u'"%s":%d\n%s\n' % (source_desc.get_escaped_description(), line, u'\n'.join(lines))

    def put_safe(self, code):
        if False:
            while True:
                i = 10
        self.write(code)
        self.bol = 0

    def put_or_include(self, code, name):
        if False:
            return 10
        include_dir = self.globalstate.common_utility_include_dir
        if include_dir and len(code) > 1024:
            include_file = '%s_%s.h' % (name, hashlib.sha1(code.encode('utf8')).hexdigest())
            path = os.path.join(include_dir, include_file)
            if not os.path.exists(path):
                tmp_path = '%s.tmp%s' % (path, os.getpid())
                with closing(Utils.open_new_file(tmp_path)) as f:
                    f.write(code)
                shutil.move(tmp_path, path)
            code = '#include "%s"\n' % path
        self.put(code)

    def put(self, code):
        if False:
            for i in range(10):
                print('nop')
        fix_indent = False
        if '{' in code:
            dl = code.count('{')
        else:
            dl = 0
        if '}' in code:
            dl -= code.count('}')
            if dl < 0:
                self.level += dl
            elif dl == 0 and code[0] == '}':
                fix_indent = True
                self.level -= 1
        if self.bol:
            self.indent()
        self.write(code)
        self.bol = 0
        if dl > 0:
            self.level += dl
        elif fix_indent:
            self.level += 1

    def putln_tempita(self, code, **context):
        if False:
            for i in range(10):
                print('nop')
        from ..Tempita import sub
        self.putln(sub(code, **context))

    def put_tempita(self, code, **context):
        if False:
            for i in range(10):
                print('nop')
        from ..Tempita import sub
        self.put(sub(code, **context))

    def increase_indent(self):
        if False:
            return 10
        self.level += 1

    def decrease_indent(self):
        if False:
            return 10
        self.level -= 1

    def begin_block(self):
        if False:
            print('Hello World!')
        self.putln('{')
        self.increase_indent()

    def end_block(self):
        if False:
            print('Hello World!')
        self.decrease_indent()
        self.putln('}')

    def indent(self):
        if False:
            i = 10
            return i + 15
        self._write_to_buffer('  ' * self.level)

    def get_py_version_hex(self, pyversion):
        if False:
            while True:
                i = 10
        return '0x%02X%02X%02X%02X' % (tuple(pyversion) + (0, 0, 0, 0))[:4]

    def put_label(self, lbl):
        if False:
            for i in range(10):
                print('nop')
        if lbl in self.funcstate.labels_used:
            self.putln('%s:;' % lbl)

    def put_goto(self, lbl):
        if False:
            print('Hello World!')
        self.funcstate.use_label(lbl)
        self.putln('goto %s;' % lbl)

    def put_var_declaration(self, entry, storage_class='', dll_linkage=None, definition=True):
        if False:
            print('Hello World!')
        if entry.visibility == 'private' and (not (definition or entry.defined_in_pxd)):
            return
        if entry.visibility == 'private' and (not entry.used):
            return
        if not entry.cf_used:
            self.put('CYTHON_UNUSED ')
        if storage_class:
            self.put('%s ' % storage_class)
        if entry.is_cpp_optional:
            self.put(entry.type.cpp_optional_declaration_code(entry.cname, dll_linkage=dll_linkage))
        else:
            self.put(entry.type.declaration_code(entry.cname, dll_linkage=dll_linkage))
        if entry.init is not None:
            self.put_safe(' = %s' % entry.type.literal_code(entry.init))
        elif entry.type.is_pyobject:
            self.put(' = NULL')
        self.putln(';')
        self.funcstate.scope.use_entry_utility_code(entry)

    def put_temp_declarations(self, func_context):
        if False:
            for i in range(10):
                print('nop')
        for (name, type, manage_ref, static) in func_context.temps_allocated:
            if type.is_cpp_class and (not type.is_fake_reference) and func_context.scope.directives['cpp_locals']:
                decl = type.cpp_optional_declaration_code(name)
            else:
                decl = type.declaration_code(name)
            if type.is_pyobject:
                self.putln('%s = NULL;' % decl)
            elif type.is_memoryviewslice:
                self.putln('%s = %s;' % (decl, type.literal_code(type.default_value)))
            else:
                self.putln('%s%s;' % (static and 'static ' or '', decl))
        if func_context.should_declare_error_indicator:
            if self.funcstate.uses_error_indicator:
                unused = ''
            else:
                unused = 'CYTHON_UNUSED '
            self.putln('%sint %s = 0;' % (unused, Naming.lineno_cname))
            self.putln('%sconst char *%s = NULL;' % (unused, Naming.filename_cname))
            self.putln('%sint %s = 0;' % (unused, Naming.clineno_cname))

    def put_generated_by(self):
        if False:
            while True:
                i = 10
        self.putln(Utils.GENERATED_BY_MARKER)
        self.putln('')

    def put_h_guard(self, guard):
        if False:
            print('Hello World!')
        self.putln('#ifndef %s' % guard)
        self.putln('#define %s' % guard)

    def unlikely(self, cond):
        if False:
            i = 10
            return i + 15
        if Options.gcc_branch_hints:
            return 'unlikely(%s)' % cond
        else:
            return cond

    def build_function_modifiers(self, modifiers, mapper=modifier_output_mapper):
        if False:
            print('Hello World!')
        if not modifiers:
            return ''
        return '%s ' % ' '.join([mapper(m, m) for m in modifiers])

    def entry_as_pyobject(self, entry):
        if False:
            print('Hello World!')
        type = entry.type
        if not entry.is_self_arg and (not entry.type.is_complete()) or entry.type.is_extension_type:
            return '(PyObject *)' + entry.cname
        else:
            return entry.cname

    def as_pyobject(self, cname, type):
        if False:
            for i in range(10):
                print('nop')
        from .PyrexTypes import py_object_type, typecast
        return typecast(py_object_type, type, cname)

    def put_gotref(self, cname, type):
        if False:
            return 10
        type.generate_gotref(self, cname)

    def put_giveref(self, cname, type):
        if False:
            print('Hello World!')
        type.generate_giveref(self, cname)

    def put_xgiveref(self, cname, type):
        if False:
            while True:
                i = 10
        type.generate_xgiveref(self, cname)

    def put_xgotref(self, cname, type):
        if False:
            while True:
                i = 10
        type.generate_xgotref(self, cname)

    def put_incref(self, cname, type, nanny=True):
        if False:
            print('Hello World!')
        type.generate_incref(self, cname, nanny=nanny)

    def put_xincref(self, cname, type, nanny=True):
        if False:
            print('Hello World!')
        type.generate_xincref(self, cname, nanny=nanny)

    def put_decref(self, cname, type, nanny=True, have_gil=True):
        if False:
            i = 10
            return i + 15
        type.generate_decref(self, cname, nanny=nanny, have_gil=have_gil)

    def put_xdecref(self, cname, type, nanny=True, have_gil=True):
        if False:
            while True:
                i = 10
        type.generate_xdecref(self, cname, nanny=nanny, have_gil=have_gil)

    def put_decref_clear(self, cname, type, clear_before_decref=False, nanny=True, have_gil=True):
        if False:
            i = 10
            return i + 15
        type.generate_decref_clear(self, cname, clear_before_decref=clear_before_decref, nanny=nanny, have_gil=have_gil)

    def put_xdecref_clear(self, cname, type, clear_before_decref=False, nanny=True, have_gil=True):
        if False:
            print('Hello World!')
        type.generate_xdecref_clear(self, cname, clear_before_decref=clear_before_decref, nanny=nanny, have_gil=have_gil)

    def put_decref_set(self, cname, type, rhs_cname):
        if False:
            while True:
                i = 10
        type.generate_decref_set(self, cname, rhs_cname)

    def put_xdecref_set(self, cname, type, rhs_cname):
        if False:
            i = 10
            return i + 15
        type.generate_xdecref_set(self, cname, rhs_cname)

    def put_incref_memoryviewslice(self, slice_cname, type, have_gil):
        if False:
            while True:
                i = 10
        type.generate_incref_memoryviewslice(self, slice_cname, have_gil=have_gil)

    def put_var_incref_memoryviewslice(self, entry, have_gil):
        if False:
            i = 10
            return i + 15
        self.put_incref_memoryviewslice(entry.cname, entry.type, have_gil=have_gil)

    def put_var_gotref(self, entry):
        if False:
            print('Hello World!')
        self.put_gotref(entry.cname, entry.type)

    def put_var_giveref(self, entry):
        if False:
            while True:
                i = 10
        self.put_giveref(entry.cname, entry.type)

    def put_var_xgotref(self, entry):
        if False:
            return 10
        self.put_xgotref(entry.cname, entry.type)

    def put_var_xgiveref(self, entry):
        if False:
            return 10
        self.put_xgiveref(entry.cname, entry.type)

    def put_var_incref(self, entry, **kwds):
        if False:
            while True:
                i = 10
        self.put_incref(entry.cname, entry.type, **kwds)

    def put_var_xincref(self, entry, **kwds):
        if False:
            i = 10
            return i + 15
        self.put_xincref(entry.cname, entry.type, **kwds)

    def put_var_decref(self, entry, **kwds):
        if False:
            print('Hello World!')
        self.put_decref(entry.cname, entry.type, **kwds)

    def put_var_xdecref(self, entry, **kwds):
        if False:
            while True:
                i = 10
        self.put_xdecref(entry.cname, entry.type, **kwds)

    def put_var_decref_clear(self, entry, **kwds):
        if False:
            i = 10
            return i + 15
        self.put_decref_clear(entry.cname, entry.type, clear_before_decref=entry.in_closure, **kwds)

    def put_var_decref_set(self, entry, rhs_cname, **kwds):
        if False:
            while True:
                i = 10
        self.put_decref_set(entry.cname, entry.type, rhs_cname, **kwds)

    def put_var_xdecref_set(self, entry, rhs_cname, **kwds):
        if False:
            i = 10
            return i + 15
        self.put_xdecref_set(entry.cname, entry.type, rhs_cname, **kwds)

    def put_var_xdecref_clear(self, entry, **kwds):
        if False:
            while True:
                i = 10
        self.put_xdecref_clear(entry.cname, entry.type, clear_before_decref=entry.in_closure, **kwds)

    def put_var_decrefs(self, entries, used_only=0):
        if False:
            while True:
                i = 10
        for entry in entries:
            if not used_only or entry.used:
                if entry.xdecref_cleanup:
                    self.put_var_xdecref(entry)
                else:
                    self.put_var_decref(entry)

    def put_var_xdecrefs(self, entries):
        if False:
            for i in range(10):
                print('nop')
        for entry in entries:
            self.put_var_xdecref(entry)

    def put_var_xdecrefs_clear(self, entries):
        if False:
            for i in range(10):
                print('nop')
        for entry in entries:
            self.put_var_xdecref_clear(entry)

    def put_init_to_py_none(self, cname, type, nanny=True):
        if False:
            for i in range(10):
                print('nop')
        from .PyrexTypes import py_object_type, typecast
        py_none = typecast(type, py_object_type, 'Py_None')
        if nanny:
            self.putln('%s = %s; __Pyx_INCREF(Py_None);' % (cname, py_none))
        else:
            self.putln('%s = %s; Py_INCREF(Py_None);' % (cname, py_none))

    def put_init_var_to_py_none(self, entry, template='%s', nanny=True):
        if False:
            for i in range(10):
                print('nop')
        code = template % entry.cname
        self.put_init_to_py_none(code, entry.type, nanny)
        if entry.in_closure:
            self.put_giveref('Py_None')

    def put_pymethoddef(self, entry, term, allow_skip=True, wrapper_code_writer=None):
        if False:
            while True:
                i = 10
        is_reverse_number_slot = False
        if entry.is_special or entry.name == '__getattribute__':
            from . import TypeSlots
            is_reverse_number_slot = True
            if entry.name not in special_py_methods and (not TypeSlots.is_reverse_number_slot(entry.name)):
                if entry.name == '__getattr__' and (not self.globalstate.directives['fast_getattr']):
                    pass
                elif allow_skip:
                    return
        method_flags = entry.signature.method_flags()
        if not method_flags:
            return
        if entry.is_special:
            method_flags += [TypeSlots.method_coexist]
        func_ptr = wrapper_code_writer.put_pymethoddef_wrapper(entry) if wrapper_code_writer else entry.func_cname
        cast = entry.signature.method_function_type()
        if cast != 'PyCFunction':
            func_ptr = '(void*)(%s)%s' % (cast, func_ptr)
        entry_name = entry.name.as_c_string_literal()
        if is_reverse_number_slot:
            slot = TypeSlots.get_slot_table(self.globalstate.directives).get_slot_by_method_name(entry.name)
            preproc_guard = slot.preprocessor_guard_code()
            if preproc_guard:
                self.putln(preproc_guard)
        self.putln('{%s, (PyCFunction)%s, %s, %s}%s' % (entry_name, func_ptr, '|'.join(method_flags), entry.doc_cname if entry.doc else '0', term))
        if is_reverse_number_slot and preproc_guard:
            self.putln('#endif')

    def put_pymethoddef_wrapper(self, entry):
        if False:
            return 10
        func_cname = entry.func_cname
        if entry.is_special:
            method_flags = entry.signature.method_flags() or []
            from .TypeSlots import method_noargs
            if method_noargs in method_flags:
                func_cname = Naming.method_wrapper_prefix + func_cname
                self.putln('static PyObject *%s(PyObject *self, CYTHON_UNUSED PyObject *arg) {' % func_cname)
                func_call = '%s(self)' % entry.func_cname
                if entry.name == '__next__':
                    self.putln('PyObject *res = %s;' % func_call)
                    self.putln('if (!res && !PyErr_Occurred()) { PyErr_SetNone(PyExc_StopIteration); }')
                    self.putln('return res;')
                else:
                    self.putln('return %s;' % func_call)
                self.putln('}')
        return func_cname

    def use_fast_gil_utility_code(self):
        if False:
            while True:
                i = 10
        if self.globalstate.directives['fast_gil']:
            self.globalstate.use_utility_code(UtilityCode.load_cached('FastGil', 'ModuleSetupCode.c'))
        else:
            self.globalstate.use_utility_code(UtilityCode.load_cached('NoFastGil', 'ModuleSetupCode.c'))

    def put_ensure_gil(self, declare_gilstate=True, variable=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Acquire the GIL. The generated code is safe even when no PyThreadState\n        has been allocated for this thread (for threads not initialized by\n        using the Python API). Additionally, the code generated by this method\n        may be called recursively.\n        '
        self.globalstate.use_utility_code(UtilityCode.load_cached('ForceInitThreads', 'ModuleSetupCode.c'))
        self.use_fast_gil_utility_code()
        self.putln('#ifdef WITH_THREAD')
        if not variable:
            variable = '__pyx_gilstate_save'
            if declare_gilstate:
                self.put('PyGILState_STATE ')
        self.putln('%s = __Pyx_PyGILState_Ensure();' % variable)
        self.putln('#endif')

    def put_release_ensured_gil(self, variable=None):
        if False:
            return 10
        '\n        Releases the GIL, corresponds to `put_ensure_gil`.\n        '
        self.use_fast_gil_utility_code()
        if not variable:
            variable = '__pyx_gilstate_save'
        self.putln('#ifdef WITH_THREAD')
        self.putln('__Pyx_PyGILState_Release(%s);' % variable)
        self.putln('#endif')

    def put_acquire_gil(self, variable=None, unknown_gil_state=True):
        if False:
            return 10
        "\n        Acquire the GIL. The thread's thread state must have been initialized\n        by a previous `put_release_gil`\n        "
        self.use_fast_gil_utility_code()
        self.putln('#ifdef WITH_THREAD')
        self.putln('__Pyx_FastGIL_Forget();')
        if variable:
            self.putln('_save = %s;' % variable)
        if unknown_gil_state:
            self.putln('if (_save) {')
        self.putln('Py_BLOCK_THREADS')
        if unknown_gil_state:
            self.putln('}')
        self.putln('#endif')

    def put_release_gil(self, variable=None, unknown_gil_state=True):
        if False:
            while True:
                i = 10
        'Release the GIL, corresponds to `put_acquire_gil`.'
        self.use_fast_gil_utility_code()
        self.putln('#ifdef WITH_THREAD')
        self.putln('PyThreadState *_save;')
        self.putln('_save = NULL;')
        if unknown_gil_state:
            self.putln('if (PyGILState_Check()) {')
        self.putln('Py_UNBLOCK_THREADS')
        if unknown_gil_state:
            self.putln('}')
        if variable:
            self.putln('%s = _save;' % variable)
        self.putln('__Pyx_FastGIL_Remember();')
        self.putln('#endif')

    def declare_gilstate(self):
        if False:
            return 10
        self.putln('#ifdef WITH_THREAD')
        self.putln('PyGILState_STATE __pyx_gilstate_save;')
        self.putln('#endif')

    def put_error_if_neg(self, pos, value):
        if False:
            print('Hello World!')
        return self.putln('if (%s < 0) %s' % (value, self.error_goto(pos)))

    def put_error_if_unbound(self, pos, entry, in_nogil_context=False, unbound_check_code=None):
        if False:
            for i in range(10):
                print('nop')
        if entry.from_closure:
            func = '__Pyx_RaiseClosureNameError'
            self.globalstate.use_utility_code(UtilityCode.load_cached('RaiseClosureNameError', 'ObjectHandling.c'))
        elif entry.type.is_memoryviewslice and in_nogil_context:
            func = '__Pyx_RaiseUnboundMemoryviewSliceNogil'
            self.globalstate.use_utility_code(UtilityCode.load_cached('RaiseUnboundMemoryviewSliceNogil', 'ObjectHandling.c'))
        elif entry.type.is_cpp_class and entry.is_cglobal:
            func = '__Pyx_RaiseCppGlobalNameError'
            self.globalstate.use_utility_code(UtilityCode.load_cached('RaiseCppGlobalNameError', 'ObjectHandling.c'))
        elif entry.type.is_cpp_class and entry.is_variable and (not entry.is_member) and entry.scope.is_c_class_scope:
            func = '__Pyx_RaiseCppAttributeError'
            self.globalstate.use_utility_code(UtilityCode.load_cached('RaiseCppAttributeError', 'ObjectHandling.c'))
        else:
            func = '__Pyx_RaiseUnboundLocalError'
            self.globalstate.use_utility_code(UtilityCode.load_cached('RaiseUnboundLocalError', 'ObjectHandling.c'))
        if not unbound_check_code:
            unbound_check_code = entry.type.check_for_null_code(entry.cname)
        self.putln('if (unlikely(!%s)) { %s("%s"); %s }' % (unbound_check_code, func, entry.name, self.error_goto(pos)))

    def set_error_info(self, pos, used=False):
        if False:
            print('Hello World!')
        self.funcstate.should_declare_error_indicator = True
        if used:
            self.funcstate.uses_error_indicator = True
        return '__PYX_MARK_ERR_POS(%s, %s)' % (self.lookup_filename(pos[0]), pos[1])

    def error_goto(self, pos, used=True):
        if False:
            return 10
        lbl = self.funcstate.error_label
        self.funcstate.use_label(lbl)
        if pos is None:
            return 'goto %s;' % lbl
        self.funcstate.should_declare_error_indicator = True
        if used:
            self.funcstate.uses_error_indicator = True
        return '__PYX_ERR(%s, %s, %s)' % (self.lookup_filename(pos[0]), pos[1], lbl)

    def error_goto_if(self, cond, pos):
        if False:
            for i in range(10):
                print('nop')
        return 'if (%s) %s' % (self.unlikely(cond), self.error_goto(pos))

    def error_goto_if_null(self, cname, pos):
        if False:
            return 10
        return self.error_goto_if('!%s' % cname, pos)

    def error_goto_if_neg(self, cname, pos):
        if False:
            i = 10
            return i + 15
        return self.error_goto_if('(%s < 0)' % cname, pos)

    def error_goto_if_PyErr(self, pos):
        if False:
            for i in range(10):
                print('nop')
        return self.error_goto_if('PyErr_Occurred()', pos)

    def lookup_filename(self, filename):
        if False:
            while True:
                i = 10
        return self.globalstate.lookup_filename(filename)

    def put_declare_refcount_context(self):
        if False:
            print('Hello World!')
        self.putln('__Pyx_RefNannyDeclarations')

    def put_setup_refcount_context(self, name, acquire_gil=False):
        if False:
            i = 10
            return i + 15
        name = name.as_c_string_literal()
        if acquire_gil:
            self.globalstate.use_utility_code(UtilityCode.load_cached('ForceInitThreads', 'ModuleSetupCode.c'))
        self.putln('__Pyx_RefNannySetupContext(%s, %d);' % (name, acquire_gil and 1 or 0))

    def put_finish_refcount_context(self, nogil=False):
        if False:
            i = 10
            return i + 15
        self.putln('__Pyx_RefNannyFinishContextNogil()' if nogil else '__Pyx_RefNannyFinishContext();')

    def put_add_traceback(self, qualified_name, include_cline=True):
        if False:
            print('Hello World!')
        '\n        Build a Python traceback for propagating exceptions.\n\n        qualified_name should be the qualified name of the function.\n        '
        qualified_name = qualified_name.as_c_string_literal()
        format_tuple = (qualified_name, Naming.clineno_cname if include_cline else 0, Naming.lineno_cname, Naming.filename_cname)
        self.funcstate.uses_error_indicator = True
        self.putln('__Pyx_AddTraceback(%s, %s, %s, %s);' % format_tuple)

    def put_unraisable(self, qualified_name, nogil=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate code to print a Python warning for an unraisable exception.\n\n        qualified_name should be the qualified name of the function.\n        '
        format_tuple = (qualified_name, Naming.clineno_cname, Naming.lineno_cname, Naming.filename_cname, self.globalstate.directives['unraisable_tracebacks'], nogil)
        self.funcstate.uses_error_indicator = True
        self.putln('__Pyx_WriteUnraisable("%s", %s, %s, %s, %d, %d);' % format_tuple)
        self.globalstate.use_utility_code(UtilityCode.load_cached('WriteUnraisableException', 'Exceptions.c'))

    def put_trace_declarations(self):
        if False:
            return 10
        self.putln('__Pyx_TraceDeclarations')

    def put_trace_frame_init(self, codeobj=None):
        if False:
            while True:
                i = 10
        if codeobj:
            self.putln('__Pyx_TraceFrameInit(%s)' % codeobj)

    def put_trace_call(self, name, pos, nogil=False):
        if False:
            while True:
                i = 10
        self.putln('__Pyx_TraceCall("%s", %s[%s], %s, %d, %s);' % (name, Naming.filetable_cname, self.lookup_filename(pos[0]), pos[1], nogil, self.error_goto(pos)))

    def put_trace_exception(self):
        if False:
            for i in range(10):
                print('nop')
        self.putln('__Pyx_TraceException();')

    def put_trace_return(self, retvalue_cname, nogil=False):
        if False:
            i = 10
            return i + 15
        self.putln('__Pyx_TraceReturn(%s, %d);' % (retvalue_cname, nogil))

    def putln_openmp(self, string):
        if False:
            return 10
        self.putln('#ifdef _OPENMP')
        self.putln(string)
        self.putln('#endif /* _OPENMP */')

    def undef_builtin_expect(self, cond):
        if False:
            i = 10
            return i + 15
        "\n        Redefine the macros likely() and unlikely to no-ops, depending on\n        condition 'cond'\n        "
        self.putln('#if %s' % cond)
        self.putln('    #undef likely')
        self.putln('    #undef unlikely')
        self.putln('    #define likely(x)   (x)')
        self.putln('    #define unlikely(x) (x)')
        self.putln('#endif')

    def redef_builtin_expect(self, cond):
        if False:
            for i in range(10):
                print('nop')
        self.putln('#if %s' % cond)
        self.putln('    #undef likely')
        self.putln('    #undef unlikely')
        self.putln('    #define likely(x)   __builtin_expect(!!(x), 1)')
        self.putln('    #define unlikely(x) __builtin_expect(!!(x), 0)')
        self.putln('#endif')

class PyrexCodeWriter(object):

    def __init__(self, outfile_name):
        if False:
            while True:
                i = 10
        self.f = Utils.open_new_file(outfile_name)
        self.level = 0

    def putln(self, code):
        if False:
            return 10
        self.f.write('%s%s\n' % (' ' * self.level, code))

    def indent(self):
        if False:
            return 10
        self.level += 1

    def dedent(self):
        if False:
            i = 10
            return i + 15
        self.level -= 1

class PyxCodeWriter(object):
    """
    Can be used for writing out some Cython code.
    """

    def __init__(self, buffer=None, indent_level=0, context=None, encoding='ascii'):
        if False:
            for i in range(10):
                print('nop')
        self.buffer = buffer or StringIOTree()
        self.level = indent_level
        self.original_level = indent_level
        self.context = context
        self.encoding = encoding

    def indent(self, levels=1):
        if False:
            return 10
        self.level += levels
        return True

    def dedent(self, levels=1):
        if False:
            while True:
                i = 10
        self.level -= levels

    @contextmanager
    def indenter(self, line):
        if False:
            i = 10
            return i + 15
        '\n        with pyx_code.indenter("for i in range(10):"):\n            pyx_code.putln("print i")\n        '
        self.putln(line)
        self.indent()
        yield
        self.dedent()

    def empty(self):
        if False:
            for i in range(10):
                print('nop')
        return self.buffer.empty()

    def getvalue(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.buffer.getvalue()
        if isinstance(result, bytes):
            result = result.decode(self.encoding)
        return result

    def putln(self, line, context=None):
        if False:
            for i in range(10):
                print('nop')
        context = context or self.context
        if context:
            line = sub_tempita(line, context)
        self._putln(line)

    def _putln(self, line):
        if False:
            while True:
                i = 10
        self.buffer.write(u'%s%s\n' % (self.level * u'    ', line))

    def put_chunk(self, chunk, context=None):
        if False:
            print('Hello World!')
        context = context or self.context
        if context:
            chunk = sub_tempita(chunk, context)
        chunk = textwrap.dedent(chunk)
        for line in chunk.splitlines():
            self._putln(line)

    def insertion_point(self):
        if False:
            return 10
        return type(self)(self.buffer.insertion_point(), self.level, self.context)

    def reset(self):
        if False:
            while True:
                i = 10
        self.buffer.reset()
        self.level = self.original_level

    def named_insertion_point(self, name):
        if False:
            i = 10
            return i + 15
        setattr(self, name, self.insertion_point())

class ClosureTempAllocator(object):

    def __init__(self, klass):
        if False:
            while True:
                i = 10
        self.klass = klass
        self.temps_allocated = {}
        self.temps_free = {}
        self.temps_count = 0

    def reset(self):
        if False:
            print('Hello World!')
        for (type, cnames) in self.temps_allocated.items():
            self.temps_free[type] = list(cnames)

    def allocate_temp(self, type):
        if False:
            while True:
                i = 10
        if type not in self.temps_allocated:
            self.temps_allocated[type] = []
            self.temps_free[type] = []
        elif self.temps_free[type]:
            return self.temps_free[type].pop(0)
        cname = '%s%d' % (Naming.codewriter_temp_prefix, self.temps_count)
        self.klass.declare_var(pos=None, name=cname, cname=cname, type=type, is_cdef=True)
        self.temps_allocated[type].append(cname)
        self.temps_count += 1
        return cname