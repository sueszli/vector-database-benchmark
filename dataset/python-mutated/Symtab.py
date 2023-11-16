from __future__ import absolute_import
import re
import copy
import operator
try:
    import __builtin__ as builtins
except ImportError:
    import builtins
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import pyfunction_signature, pymethod_signature, richcmp_special_methods, get_slot_table, get_property_accessor_signature
from . import Future
from . import Code
iso_c99_keywords = {'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if', 'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while', '_Bool', '_Complex, _Imaginary', 'inline', 'restrict'}

def c_safe_identifier(cname):
    if False:
        while True:
            i = 10
    if cname[:2] == '__' and (not (cname.startswith(Naming.pyrex_prefix) or cname in ('__weakref__', '__dict__'))) or cname in iso_c99_keywords:
        cname = Naming.pyrex_prefix + cname
    return cname

def punycodify_name(cname, mangle_with=None):
    if False:
        print('Hello World!')
    try:
        cname.encode('ascii')
    except UnicodeEncodeError:
        cname = cname.encode('punycode').replace(b'-', b'_').decode('ascii')
        if mangle_with:
            cname = '%s_%s' % (mangle_with, cname)
        elif cname.startswith(Naming.pyrex_prefix):
            cname = cname.replace(Naming.pyrex_prefix, Naming.pyunicode_identifier_prefix, 1)
    return cname

class BufferAux(object):
    writable_needed = False

    def __init__(self, buflocal_nd_var, rcbuf_var):
        if False:
            return 10
        self.buflocal_nd_var = buflocal_nd_var
        self.rcbuf_var = rcbuf_var

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<BufferAux %r>' % self.__dict__

class Entry(object):
    inline_func_in_pxd = False
    borrowed = 0
    init = ''
    annotation = None
    pep563_annotation = None
    visibility = 'private'
    is_builtin = 0
    is_cglobal = 0
    is_pyglobal = 0
    is_member = 0
    is_pyclass_attr = 0
    is_variable = 0
    is_cfunction = 0
    is_cmethod = 0
    is_builtin_cmethod = False
    is_unbound_cmethod = 0
    is_final_cmethod = 0
    is_inline_cmethod = 0
    is_anonymous = 0
    is_type = 0
    is_cclass = 0
    is_cpp_class = 0
    is_const = 0
    is_property = 0
    is_cproperty = 0
    doc_cname = None
    getter_cname = None
    setter_cname = None
    is_self_arg = 0
    is_arg = 0
    is_local = 0
    in_closure = 0
    from_closure = 0
    in_subscope = 0
    is_declared_generic = 0
    is_readonly = 0
    pyfunc_cname = None
    func_cname = None
    func_modifiers = []
    final_func_cname = None
    doc = None
    as_variable = None
    xdecref_cleanup = 0
    in_cinclude = 0
    as_module = None
    is_inherited = 0
    pystring_cname = None
    is_identifier = 0
    is_interned = 0
    used = 0
    is_special = 0
    defined_in_pxd = 0
    is_implemented = 0
    api = 0
    utility_code = None
    is_overridable = 0
    buffer_aux = None
    prev_entry = None
    might_overflow = 0
    fused_cfunction = None
    is_fused_specialized = False
    utility_code_definition = None
    needs_property = False
    in_with_gil_block = 0
    from_cython_utility_code = None
    error_on_uninitialized = False
    cf_used = True
    outer_entry = None
    is_cgetter = False
    is_cpp_optional = False
    known_standard_library_import = None
    pytyping_modifiers = None
    enum_int_value = None

    def __init__(self, name, cname, type, pos=None, init=None):
        if False:
            while True:
                i = 10
        self.name = name
        self.cname = cname
        self.type = type
        self.pos = pos
        self.init = init
        self.overloaded_alternatives = []
        self.cf_assignments = []
        self.cf_references = []
        self.inner_entries = []
        self.defining_entry = self

    def __repr__(self):
        if False:
            return 10
        return '%s(<%x>, name=%s, type=%s)' % (type(self).__name__, id(self), self.name, self.type)

    def already_declared_here(self):
        if False:
            for i in range(10):
                print('nop')
        error(self.pos, 'Previous declaration is here')

    def redeclared(self, pos):
        if False:
            while True:
                i = 10
        error(pos, "'%s' does not match previous declaration" % self.name)
        self.already_declared_here()

    def all_alternatives(self):
        if False:
            for i in range(10):
                print('nop')
        return [self] + self.overloaded_alternatives

    def all_entries(self):
        if False:
            while True:
                i = 10
        return [self] + self.inner_entries

    def __lt__(left, right):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(left, Entry) and isinstance(right, Entry):
            return (left.name, left.cname) < (right.name, right.cname)
        else:
            return NotImplemented

    @property
    def cf_is_reassigned(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.cf_assignments) > 1

    def make_cpp_optional(self):
        if False:
            return 10
        assert self.type.is_cpp_class
        self.is_cpp_optional = True
        assert not self.utility_code
        self.utility_code_definition = Code.UtilityCode.load_cached('OptionalLocals', 'CppSupport.cpp')

    def declared_with_pytyping_modifier(self, modifier_name):
        if False:
            return 10
        return modifier_name in self.pytyping_modifiers if self.pytyping_modifiers else False

class InnerEntry(Entry):
    """
    An entry in a closure scope that represents the real outer Entry.
    """
    from_closure = True

    def __init__(self, outer_entry, scope):
        if False:
            while True:
                i = 10
        Entry.__init__(self, outer_entry.name, outer_entry.cname, outer_entry.type, outer_entry.pos)
        self.outer_entry = outer_entry
        self.scope = scope
        outermost_entry = outer_entry
        while outermost_entry.outer_entry:
            outermost_entry = outermost_entry.outer_entry
        self.defining_entry = outermost_entry
        self.inner_entries = outermost_entry.inner_entries
        self.cf_assignments = outermost_entry.cf_assignments
        self.cf_references = outermost_entry.cf_references
        self.overloaded_alternatives = outermost_entry.overloaded_alternatives
        self.is_cpp_optional = outermost_entry.is_cpp_optional
        self.inner_entries.append(self)

    def __getattr__(self, name):
        if False:
            return 10
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self.defining_entry, name)

    def all_entries(self):
        if False:
            i = 10
            return i + 15
        return self.defining_entry.all_entries()

class Scope(object):
    is_builtin_scope = 0
    is_py_class_scope = 0
    is_c_class_scope = 0
    is_closure_scope = 0
    is_local_scope = False
    is_generator_expression_scope = 0
    is_comprehension_scope = 0
    is_passthrough = 0
    is_cpp_class_scope = 0
    is_property_scope = 0
    is_module_scope = 0
    is_c_dataclass_scope = False
    is_internal = 0
    scope_prefix = ''
    in_cinclude = 0
    nogil = 0
    fused_to_specific = None
    return_type = None
    scope_predefined_names = []
    in_c_type_context = True

    def __init__(self, name, outer_scope, parent_scope):
        if False:
            print('Hello World!')
        self.name = name
        self.outer_scope = outer_scope
        self.parent_scope = parent_scope
        mangled_name = '%d%s_' % (len(name), name.replace('.', '_dot_'))
        qual_scope = self.qualifying_scope()
        if qual_scope:
            self.qualified_name = qual_scope.qualify_name(name)
            self.scope_prefix = qual_scope.scope_prefix + mangled_name
        else:
            self.qualified_name = EncodedString(name)
            self.scope_prefix = mangled_name
        self.entries = {}
        self.subscopes = set()
        self.const_entries = []
        self.type_entries = []
        self.sue_entries = []
        self.arg_entries = []
        self.var_entries = []
        self.pyfunc_entries = []
        self.cfunc_entries = []
        self.c_class_entries = []
        self.defined_c_classes = []
        self.imported_c_classes = {}
        self.cname_to_entry = {}
        self.identifier_to_entry = {}
        self.num_to_entry = {}
        self.obj_to_entry = {}
        self.buffer_entries = []
        self.lambda_defs = []
        self.id_counters = {}
        for var_name in self.scope_predefined_names:
            self.declare_var(EncodedString(var_name), py_object_type, pos=None)

    def __deepcopy__(self, memo):
        if False:
            for i in range(10):
                print('nop')
        return self

    def merge_in(self, other, merge_unused=True, allowlist=None):
        if False:
            while True:
                i = 10
        entries = []
        for (name, entry) in other.entries.items():
            if not allowlist or name in allowlist:
                if entry.used or merge_unused:
                    entries.append((name, entry))
        self.entries.update(entries)
        for attr in ('const_entries', 'type_entries', 'sue_entries', 'arg_entries', 'var_entries', 'pyfunc_entries', 'cfunc_entries', 'c_class_entries'):
            self_entries = getattr(self, attr)
            names = set((e.name for e in self_entries))
            for entry in getattr(other, attr):
                if (entry.used or merge_unused) and entry.name not in names:
                    self_entries.append(entry)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '<%s %s>' % (self.__class__.__name__, self.qualified_name)

    def qualifying_scope(self):
        if False:
            return 10
        return self.parent_scope

    def mangle(self, prefix, name=None):
        if False:
            i = 10
            return i + 15
        if name:
            return punycodify_name('%s%s%s' % (prefix, self.scope_prefix, name))
        else:
            return self.parent_scope.mangle(prefix, self.name)

    def mangle_internal(self, name):
        if False:
            return 10
        prefix = '%s%s_' % (Naming.pyrex_prefix, name)
        return self.mangle(prefix)

    def mangle_class_private_name(self, name):
        if False:
            return 10
        if self.parent_scope:
            return self.parent_scope.mangle_class_private_name(name)
        return name

    def next_id(self, name=None):
        if False:
            while True:
                i = 10
        counters = self.global_scope().id_counters
        try:
            count = counters[name] + 1
        except KeyError:
            count = 0
        counters[name] = count
        if name:
            if not count:
                return name
            return '%s%d' % (name, count)
        else:
            return '%d' % count

    def global_scope(self):
        if False:
            for i in range(10):
                print('nop')
        ' Return the module-level scope containing this scope. '
        return self.outer_scope.global_scope()

    def builtin_scope(self):
        if False:
            for i in range(10):
                print('nop')
        ' Return the module-level scope containing this scope. '
        return self.outer_scope.builtin_scope()

    def iter_local_scopes(self):
        if False:
            while True:
                i = 10
        yield self
        if self.subscopes:
            for scope in sorted(self.subscopes, key=operator.attrgetter('scope_prefix')):
                yield scope

    @try_finally_contextmanager
    def new_c_type_context(self, in_c_type_context=None):
        if False:
            return 10
        old_c_type_context = self.in_c_type_context
        if in_c_type_context is not None:
            self.in_c_type_context = in_c_type_context
        yield
        self.in_c_type_context = old_c_type_context

    def declare(self, name, cname, type, pos, visibility, shadow=0, is_type=0, create_wrapper=0):
        if False:
            while True:
                i = 10
        if type.is_buffer and (not isinstance(self, LocalScope)):
            error(pos, 'Buffer types only allowed as function local variables')
        if not self.in_cinclude and cname and re.match('^_[_A-Z]+$', cname):
            warning(pos, "'%s' is a reserved name in C." % cname, -1)
        entries = self.entries
        if name and name in entries and (not shadow) and (not self.is_builtin_scope):
            old_entry = entries[name]
            cpp_override_allowed = False
            if type.is_cfunction and old_entry.type.is_cfunction and self.is_cpp():
                for alt_entry in old_entry.all_alternatives():
                    if type == alt_entry.type:
                        if name == '<init>' and (not type.args):
                            cpp_override_allowed = True
                        break
                else:
                    cpp_override_allowed = True
            if cpp_override_allowed:
                pass
            elif self.is_cpp_class_scope and entries[name].is_inherited:
                pass
            elif visibility == 'extern':
                warning(pos, "'%s' redeclared " % name, 1 if self.in_cinclude else 0)
            elif visibility != 'ignore':
                error(pos, "'%s' redeclared " % name)
                entries[name].already_declared_here()
        entry = Entry(name, cname, type, pos=pos)
        entry.in_cinclude = self.in_cinclude
        entry.create_wrapper = create_wrapper
        if name:
            entry.qualified_name = self.qualify_name(name)
            if not shadow:
                entries[name] = entry
        if type.is_memoryviewslice:
            entry.init = type.default_value
        entry.scope = self
        entry.visibility = visibility
        return entry

    def qualify_name(self, name):
        if False:
            for i in range(10):
                print('nop')
        return EncodedString('%s.%s' % (self.qualified_name, name))

    def declare_const(self, name, type, value, pos, cname=None, visibility='private', api=0, create_wrapper=0):
        if False:
            print('Hello World!')
        if not cname:
            if self.in_cinclude or (visibility == 'public' or api):
                cname = name
            else:
                cname = self.mangle(Naming.enum_prefix, name)
        entry = self.declare(name, cname, type, pos, visibility, create_wrapper=create_wrapper)
        entry.is_const = 1
        entry.value_node = value
        return entry

    def declare_type(self, name, type, pos, cname=None, visibility='private', api=0, defining=1, shadow=0, template=0):
        if False:
            while True:
                i = 10
        if not cname:
            cname = name
        entry = self.declare(name, cname, type, pos, visibility, shadow, is_type=True)
        entry.is_type = 1
        entry.api = api
        if defining:
            self.type_entries.append(entry)
        if not template and getattr(type, 'entry', None) is None:
            type.entry = entry
        return entry

    def declare_typedef(self, name, base_type, pos, cname=None, visibility='private', api=0):
        if False:
            for i in range(10):
                print('nop')
        if not cname:
            if self.in_cinclude or (visibility != 'private' or api):
                cname = name
            else:
                cname = self.mangle(Naming.type_prefix, name)
        try:
            if self.is_cpp_class_scope:
                namespace = self.outer_scope.lookup(self.name).type
            else:
                namespace = None
            type = PyrexTypes.create_typedef_type(name, base_type, cname, visibility == 'extern', namespace)
        except ValueError as e:
            error(pos, e.args[0])
            type = PyrexTypes.error_type
        entry = self.declare_type(name, type, pos, cname, visibility=visibility, api=api)
        type.qualified_name = entry.qualified_name
        return entry

    def declare_struct_or_union(self, name, kind, scope, typedef_flag, pos, cname=None, visibility='private', api=0, packed=False):
        if False:
            while True:
                i = 10
        if not cname:
            if self.in_cinclude or (visibility == 'public' or api):
                cname = name
            else:
                cname = self.mangle(Naming.type_prefix, name)
        entry = self.lookup_here(name)
        if not entry:
            in_cpp = self.is_cpp()
            type = PyrexTypes.CStructOrUnionType(name, kind, scope, typedef_flag, cname, packed, in_cpp=in_cpp)
            entry = self.declare_type(name, type, pos, cname, visibility=visibility, api=api, defining=scope is not None)
            self.sue_entries.append(entry)
            type.entry = entry
        elif not (entry.is_type and entry.type.is_struct_or_union and (entry.type.kind == kind)):
            warning(pos, "'%s' redeclared  " % name, 0)
        elif scope and entry.type.scope:
            warning(pos, "'%s' already defined  (ignoring second definition)" % name, 0)
        else:
            self.check_previous_typedef_flag(entry, typedef_flag, pos)
            self.check_previous_visibility(entry, visibility, pos)
            if scope:
                entry.type.scope = scope
                self.type_entries.append(entry)
        if self.is_cpp_class_scope:
            entry.type.namespace = self.outer_scope.lookup(self.name).type
        return entry

    def declare_cpp_class(self, name, scope, pos, cname=None, base_classes=(), visibility='extern', templates=None):
        if False:
            return 10
        if cname is None:
            if self.in_cinclude or visibility != 'private':
                cname = name
            else:
                cname = self.mangle(Naming.type_prefix, name)
        base_classes = list(base_classes)
        entry = self.lookup_here(name)
        if not entry:
            type = PyrexTypes.CppClassType(name, scope, cname, base_classes, templates=templates)
            entry = self.declare_type(name, type, pos, cname, visibility=visibility, defining=scope is not None)
            self.sue_entries.append(entry)
        else:
            if not (entry.is_type and entry.type.is_cpp_class):
                error(pos, "'%s' redeclared " % name)
                entry.already_declared_here()
                return None
            elif scope and entry.type.scope:
                warning(pos, "'%s' already defined  (ignoring second definition)" % name, 0)
            elif scope:
                entry.type.scope = scope
                self.type_entries.append(entry)
            if base_classes:
                if entry.type.base_classes and entry.type.base_classes != base_classes:
                    error(pos, 'Base type does not match previous declaration')
                    entry.already_declared_here()
                else:
                    entry.type.base_classes = base_classes
            if templates or entry.type.templates:
                if templates != entry.type.templates:
                    error(pos, 'Template parameters do not match previous declaration')
                    entry.already_declared_here()

        def declare_inherited_attributes(entry, base_classes):
            if False:
                return 10
            for base_class in base_classes:
                if base_class is PyrexTypes.error_type:
                    continue
                if base_class.scope is None:
                    error(pos, 'Cannot inherit from incomplete type')
                else:
                    declare_inherited_attributes(entry, base_class.base_classes)
                    entry.type.scope.declare_inherited_cpp_attributes(base_class)
        if scope:
            declare_inherited_attributes(entry, base_classes)
            scope.declare_var(name='this', cname='this', type=PyrexTypes.CPtrType(entry.type), pos=entry.pos)
        if self.is_cpp_class_scope:
            entry.type.namespace = self.outer_scope.lookup(self.name).type
        return entry

    def check_previous_typedef_flag(self, entry, typedef_flag, pos):
        if False:
            for i in range(10):
                print('nop')
        if typedef_flag != entry.type.typedef_flag:
            error(pos, "'%s' previously declared using '%s'" % (entry.name, ('cdef', 'ctypedef')[entry.type.typedef_flag]))

    def check_previous_visibility(self, entry, visibility, pos):
        if False:
            for i in range(10):
                print('nop')
        if entry.visibility != visibility:
            error(pos, "'%s' previously declared as '%s'" % (entry.name, entry.visibility))

    def declare_enum(self, name, pos, cname, scoped, typedef_flag, visibility='private', api=0, create_wrapper=0, doc=None):
        if False:
            print('Hello World!')
        if name:
            if not cname:
                if self.in_cinclude or visibility == 'public' or visibility == 'extern' or api:
                    cname = name
                else:
                    cname = self.mangle(Naming.type_prefix, name)
            if self.is_cpp_class_scope:
                namespace = self.outer_scope.lookup(self.name).type
            else:
                namespace = None
            if scoped:
                type = PyrexTypes.CppScopedEnumType(name, cname, namespace, doc=doc)
            else:
                type = PyrexTypes.CEnumType(name, cname, typedef_flag, namespace, doc=doc)
        else:
            type = PyrexTypes.c_anon_enum_type
        entry = self.declare_type(name, type, pos, cname=cname, visibility=visibility, api=api)
        if scoped:
            entry.utility_code = Code.UtilityCode.load_cached('EnumClassDecl', 'CppSupport.cpp')
            self.use_entry_utility_code(entry)
        entry.create_wrapper = create_wrapper
        entry.enum_values = []
        self.sue_entries.append(entry)
        return entry

    def declare_tuple_type(self, pos, components):
        if False:
            print('Hello World!')
        return self.outer_scope.declare_tuple_type(pos, components)

    def declare_var(self, name, type, pos, cname=None, visibility='private', api=False, in_pxd=False, is_cdef=False, pytyping_modifiers=None):
        if False:
            for i in range(10):
                print('nop')
        if not cname:
            if visibility != 'private' or api:
                cname = name
            else:
                cname = self.mangle(Naming.var_prefix, name)
        entry = self.declare(name, cname, type, pos, visibility)
        entry.is_variable = 1
        if type.is_cpp_class and visibility != 'extern':
            if self.directives['cpp_locals']:
                entry.make_cpp_optional()
            else:
                type.check_nullary_constructor(pos)
        if in_pxd and visibility != 'extern':
            entry.defined_in_pxd = 1
            entry.used = 1
        if api:
            entry.api = 1
            entry.used = 1
        if pytyping_modifiers:
            entry.pytyping_modifiers = pytyping_modifiers
        return entry

    def _reject_pytyping_modifiers(self, pos, modifiers, allowed=()):
        if False:
            i = 10
            return i + 15
        if not modifiers:
            return
        for modifier in modifiers:
            if modifier not in allowed:
                error(pos, "Modifier '%s' is not allowed here." % modifier)

    def declare_assignment_expression_target(self, name, type, pos):
        if False:
            i = 10
            return i + 15
        return self.declare_var(name, type, pos)

    def declare_builtin(self, name, pos):
        if False:
            while True:
                i = 10
        name = self.mangle_class_private_name(name)
        return self.outer_scope.declare_builtin(name, pos)

    def _declare_pyfunction(self, name, pos, visibility='extern', entry=None):
        if False:
            return 10
        if entry and (not entry.type.is_cfunction):
            error(pos, "'%s' already declared" % name)
            error(entry.pos, 'Previous declaration is here')
        entry = self.declare_var(name, py_object_type, pos, visibility=visibility)
        entry.signature = pyfunction_signature
        self.pyfunc_entries.append(entry)
        return entry

    def declare_pyfunction(self, name, pos, allow_redefine=False, visibility='extern'):
        if False:
            print('Hello World!')
        entry = self.lookup_here(name)
        if not allow_redefine:
            return self._declare_pyfunction(name, pos, visibility=visibility, entry=entry)
        if entry:
            if entry.type.is_unspecified:
                entry.type = py_object_type
            elif entry.type is not py_object_type:
                return self._declare_pyfunction(name, pos, visibility=visibility, entry=entry)
        else:
            self.declare_var(name, py_object_type, pos, visibility=visibility)
        entry = self.declare_var(None, py_object_type, pos, cname=name, visibility='private')
        entry.name = EncodedString(name)
        entry.qualified_name = self.qualify_name(name)
        entry.signature = pyfunction_signature
        entry.is_anonymous = True
        return entry

    def declare_lambda_function(self, lambda_name, pos):
        if False:
            i = 10
            return i + 15
        func_cname = self.mangle(Naming.lambda_func_prefix + u'funcdef_', lambda_name)
        pymethdef_cname = self.mangle(Naming.lambda_func_prefix + u'methdef_', lambda_name)
        qualified_name = self.qualify_name(lambda_name)
        entry = self.declare(None, func_cname, py_object_type, pos, 'private')
        entry.name = EncodedString(lambda_name)
        entry.qualified_name = qualified_name
        entry.pymethdef_cname = pymethdef_cname
        entry.func_cname = func_cname
        entry.signature = pyfunction_signature
        entry.is_anonymous = True
        return entry

    def add_lambda_def(self, def_node):
        if False:
            for i in range(10):
                print('nop')
        self.lambda_defs.append(def_node)

    def register_pyfunction(self, entry):
        if False:
            return 10
        self.pyfunc_entries.append(entry)

    def declare_cfunction(self, name, type, pos, cname=None, visibility='private', api=0, in_pxd=0, defining=0, modifiers=(), utility_code=None, overridable=False):
        if False:
            print('Hello World!')
        if not cname:
            if visibility != 'private' or api:
                cname = name
            else:
                cname = self.mangle(Naming.func_prefix, name)
        inline_in_pxd = 'inline' in modifiers and in_pxd and defining
        if inline_in_pxd:
            in_pxd = False
        entry = self.lookup_here(name)
        if entry:
            if not in_pxd and visibility != entry.visibility and (visibility == 'extern'):
                defining = True
                visibility = entry.visibility
                entry.cname = cname
                entry.func_cname = cname
            if visibility != 'private' and visibility != entry.visibility:
                warning(pos, "Function '%s' previously declared as '%s', now as '%s'" % (name, entry.visibility, visibility), 1)
            if overridable != entry.is_overridable:
                warning(pos, "Function '%s' previously declared as '%s'" % (name, 'cpdef' if overridable else 'cdef'), 1)
            if entry.type.same_as(type):
                entry.type = entry.type.with_with_gil(type.with_gil)
            elif visibility == 'extern' and entry.visibility == 'extern':
                can_override = self.is_builtin_scope
                if self.is_cpp():
                    can_override = True
                elif cname and (not can_override):
                    for alt_entry in entry.all_alternatives():
                        if not alt_entry.cname or cname == alt_entry.cname:
                            break
                    else:
                        can_override = True
                if can_override:
                    temp = self.add_cfunction(name, type, pos, cname, visibility, modifiers)
                    temp.overloaded_alternatives = entry.all_alternatives()
                    entry = temp
                else:
                    warning(pos, 'Function signature does not match previous declaration', 1)
                    entry.type = type
            elif not in_pxd and entry.defined_in_pxd and type.compatible_signature_with(entry.type):
                entry.type = type
            else:
                error(pos, 'Function signature does not match previous declaration')
        else:
            entry = self.add_cfunction(name, type, pos, cname, visibility, modifiers)
            entry.func_cname = cname
            entry.is_overridable = overridable
        if inline_in_pxd:
            entry.inline_func_in_pxd = True
        if in_pxd and visibility != 'extern':
            entry.defined_in_pxd = 1
        if api:
            entry.api = 1
        if not defining and (not in_pxd) and (visibility != 'extern'):
            error(pos, "Non-extern C function '%s' declared but not defined" % name)
        if defining:
            entry.is_implemented = True
        if modifiers:
            entry.func_modifiers = modifiers
        if utility_code:
            assert not entry.utility_code, 'duplicate utility code definition in entry %s (%s)' % (name, cname)
            entry.utility_code = utility_code
        if overridable:
            var_entry = Entry(name, cname, py_object_type)
            var_entry.qualified_name = self.qualify_name(name)
            var_entry.is_variable = 1
            var_entry.is_pyglobal = 1
            var_entry.scope = entry.scope
            entry.as_variable = var_entry
        type.entry = entry
        if type.exception_check and type.exception_value is None and type.nogil and (not pos[0].in_utility_code) and defining and (not in_pxd) and (not inline_in_pxd):
            PyrexTypes.write_noexcept_performance_hint(pos, self, function_name=name, void_return=type.return_type.is_void)
        return entry

    def declare_cgetter(self, name, return_type, pos=None, cname=None, visibility='private', modifiers=(), defining=False, **cfunc_type_config):
        if False:
            for i in range(10):
                print('nop')
        assert all((k in ('exception_value', 'exception_check', 'nogil', 'with_gil', 'is_const_method', 'is_static_method') for k in cfunc_type_config))
        cfunc_type = PyrexTypes.CFuncType(return_type, [PyrexTypes.CFuncTypeArg('self', self.parent_type, None)], **cfunc_type_config)
        entry = self.declare_cfunction(name, cfunc_type, pos, cname=None, visibility=visibility, modifiers=modifiers, defining=defining)
        entry.is_cgetter = True
        if cname is not None:
            entry.func_cname = cname
        return entry

    def add_cfunction(self, name, type, pos, cname, visibility, modifiers, inherited=False):
        if False:
            for i in range(10):
                print('nop')
        entry = self.declare(name, cname, type, pos, visibility)
        entry.is_cfunction = 1
        if modifiers:
            entry.func_modifiers = modifiers
        if inherited or type.is_fused:
            self.cfunc_entries.append(entry)
        else:
            i = len(self.cfunc_entries)
            for cfunc_entry in reversed(self.cfunc_entries):
                if cfunc_entry.is_inherited or not cfunc_entry.type.is_fused:
                    break
                i -= 1
            self.cfunc_entries.insert(i, entry)
        return entry

    def find(self, name, pos):
        if False:
            for i in range(10):
                print('nop')
        entry = self.lookup(name)
        if entry:
            return entry
        else:
            error(pos, "'%s' is not declared" % name)

    def find_imported_module(self, path, pos):
        if False:
            print('Hello World!')
        scope = self
        for name in path:
            entry = scope.find(name, pos)
            if not entry:
                return None
            if entry.as_module:
                scope = entry.as_module
            else:
                error(pos, "'%s' is not a cimported module" % '.'.join(path))
                return None
        return scope

    def lookup(self, name):
        if False:
            while True:
                i = 10
        mangled_name = self.mangle_class_private_name(name)
        entry = self.lookup_here(name) or (self.outer_scope and self.outer_scope.lookup(mangled_name)) or None
        if entry:
            return entry
        entry = self.outer_scope and self.outer_scope.lookup(name) or None
        if entry and entry.is_pyglobal:
            self._emit_class_private_warning(entry.pos, name)
        return entry

    def lookup_here(self, name):
        if False:
            i = 10
            return i + 15
        entry = self.entries.get(self.mangle_class_private_name(name), None)
        if entry:
            return entry
        return self.entries.get(name, None)

    def lookup_here_unmangled(self, name):
        if False:
            print('Hello World!')
        return self.entries.get(name, None)

    def lookup_assignment_expression_target(self, name):
        if False:
            return 10
        return self.lookup_here(name)

    def lookup_target(self, name):
        if False:
            i = 10
            return i + 15
        entry = self.lookup_here(name)
        if not entry:
            entry = self.lookup_here_unmangled(name)
            if entry and entry.is_pyglobal:
                self._emit_class_private_warning(entry.pos, name)
        if not entry:
            entry = self.declare_var(name, py_object_type, None)
        return entry

    def _type_or_specialized_type_from_entry(self, entry):
        if False:
            return 10
        if entry and entry.is_type:
            if entry.type.is_fused and self.fused_to_specific:
                return entry.type.specialize(self.fused_to_specific)
            return entry.type

    def lookup_type(self, name):
        if False:
            for i in range(10):
                print('nop')
        entry = self.lookup(name)
        tp = self._type_or_specialized_type_from_entry(entry)
        if tp:
            return tp
        if entry and entry.known_standard_library_import:
            from .Builtin import get_known_standard_library_entry
            entry = get_known_standard_library_entry(entry.known_standard_library_import)
        return self._type_or_specialized_type_from_entry(entry)

    def lookup_operator(self, operator, operands):
        if False:
            for i in range(10):
                print('nop')
        if operands[0].type.is_cpp_class:
            obj_type = operands[0].type
            method = obj_type.scope.lookup('operator%s' % operator)
            if method is not None:
                arg_types = [arg.type for arg in operands[1:]]
                res = PyrexTypes.best_match(arg_types, method.all_alternatives())
                if res is not None:
                    return res
        function = self.lookup('operator%s' % operator)
        function_alternatives = []
        if function is not None:
            function_alternatives = function.all_alternatives()
        method_alternatives = []
        if len(operands) == 2:
            for n in range(2):
                if operands[n].type.is_cpp_class:
                    obj_type = operands[n].type
                    method = obj_type.scope.lookup('operator%s' % operator)
                    if method is not None:
                        method_alternatives += method.all_alternatives()
        if not method_alternatives and (not function_alternatives):
            return None
        all_alternatives = list(set(method_alternatives + function_alternatives))
        return PyrexTypes.best_match([arg.type for arg in operands], all_alternatives)

    def lookup_operator_for_types(self, pos, operator, types):
        if False:
            for i in range(10):
                print('nop')
        from .Nodes import Node

        class FakeOperand(Node):
            pass
        operands = [FakeOperand(pos, type=type) for type in types]
        return self.lookup_operator(operator, operands)

    def _emit_class_private_warning(self, pos, name):
        if False:
            i = 10
            return i + 15
        warning(pos, "Global name %s matched from within class scope in contradiction to to Python 'class private name' rules. This may change in a future release." % name, 1)

    def use_utility_code(self, new_code):
        if False:
            return 10
        self.global_scope().use_utility_code(new_code)

    def use_entry_utility_code(self, entry):
        if False:
            print('Hello World!')
        self.global_scope().use_entry_utility_code(entry)

    def defines_any(self, names):
        if False:
            return 10
        for name in names:
            if name in self.entries:
                return 1
        return 0

    def defines_any_special(self, names):
        if False:
            while True:
                i = 10
        for name in names:
            if name in self.entries and self.entries[name].is_special:
                return 1
        return 0

    def infer_types(self):
        if False:
            i = 10
            return i + 15
        from .TypeInference import get_type_inferer
        get_type_inferer().infer_types(self)

    def is_cpp(self):
        if False:
            while True:
                i = 10
        outer = self.outer_scope
        if outer is None:
            return False
        else:
            return outer.is_cpp()

    def add_include_file(self, filename, verbatim_include=None, late=False):
        if False:
            i = 10
            return i + 15
        self.outer_scope.add_include_file(filename, verbatim_include, late)

class PreImportScope(Scope):
    namespace_cname = Naming.preimport_cname

    def __init__(self):
        if False:
            return 10
        Scope.__init__(self, Options.pre_import, None, None)

    def declare_builtin(self, name, pos):
        if False:
            return 10
        entry = self.declare(name, name, py_object_type, pos, 'private')
        entry.is_variable = True
        entry.is_pyglobal = True
        return entry

class BuiltinScope(Scope):
    is_builtin_scope = True

    def __init__(self):
        if False:
            return 10
        if Options.pre_import is None:
            Scope.__init__(self, '__builtin__', None, None)
        else:
            Scope.__init__(self, '__builtin__', PreImportScope(), None)
        self.type_names = {}
        self.declare_var('bool', py_object_type, None, '((PyObject*)&PyBool_Type)')

    def lookup(self, name, language_level=None, str_is_str=None):
        if False:
            while True:
                i = 10
        if name == 'str':
            if str_is_str is None:
                str_is_str = language_level in (None, 2)
            if not str_is_str:
                name = 'unicode'
        return Scope.lookup(self, name)

    def declare_builtin(self, name, pos):
        if False:
            return 10
        if not hasattr(builtins, name):
            if self.outer_scope is not None:
                return self.outer_scope.declare_builtin(name, pos)
            elif Options.error_on_unknown_names:
                error(pos, 'undeclared name not builtin: %s' % name)
            else:
                warning(pos, 'undeclared name not builtin: %s' % name, 2)

    def declare_builtin_cfunction(self, name, type, cname, python_equiv=None, utility_code=None):
        if False:
            while True:
                i = 10
        name = EncodedString(name)
        entry = self.declare_cfunction(name, type, None, cname, visibility='extern', utility_code=utility_code)
        if python_equiv:
            if python_equiv == '*':
                python_equiv = name
            else:
                python_equiv = EncodedString(python_equiv)
            var_entry = Entry(python_equiv, python_equiv, py_object_type)
            var_entry.qualified_name = self.qualify_name(name)
            var_entry.is_variable = 1
            var_entry.is_builtin = 1
            var_entry.utility_code = utility_code
            var_entry.scope = entry.scope
            entry.as_variable = var_entry
        return entry

    def declare_builtin_type(self, name, cname, utility_code=None, objstruct_cname=None, type_class=PyrexTypes.BuiltinObjectType):
        if False:
            for i in range(10):
                print('nop')
        name = EncodedString(name)
        type = type_class(name, cname, objstruct_cname)
        scope = CClassScope(name, outer_scope=None, visibility='extern', parent_type=type)
        scope.directives = {}
        if name == 'bool':
            type.is_final_type = True
        type.set_scope(scope)
        self.type_names[name] = 1
        entry = self.declare_type(name, type, None, visibility='extern')
        entry.utility_code = utility_code
        var_entry = Entry(name=entry.name, type=self.lookup('type').type, pos=entry.pos, cname=entry.type.typeptr_cname)
        var_entry.qualified_name = self.qualify_name(name)
        var_entry.is_variable = 1
        var_entry.is_cglobal = 1
        var_entry.is_readonly = 1
        var_entry.is_builtin = 1
        var_entry.utility_code = utility_code
        var_entry.scope = self
        if Options.cache_builtins:
            var_entry.is_const = True
        entry.as_variable = var_entry
        return type

    def builtin_scope(self):
        if False:
            return 10
        return self
const_counter = 1

class ModuleScope(Scope):
    is_module_scope = 1
    has_import_star = 0
    is_cython_builtin = 0
    old_style_globals = 0
    scope_predefined_names = ['__builtins__', '__name__', '__file__', '__doc__', '__path__', '__spec__', '__loader__', '__package__', '__cached__']

    def __init__(self, name, parent_module, context, is_package=False):
        if False:
            i = 10
            return i + 15
        from . import Builtin
        self.parent_module = parent_module
        outer_scope = Builtin.builtin_scope
        Scope.__init__(self, name, outer_scope, parent_module)
        self.is_package = is_package
        self.module_name = name
        self.module_name = EncodedString(self.module_name)
        self.context = context
        self.module_cname = Naming.module_cname
        self.module_dict_cname = Naming.moddict_cname
        self.method_table_cname = Naming.methtable_cname
        self.doc = ''
        self.doc_cname = Naming.moddoc_cname
        self.utility_code_list = []
        self.module_entries = {}
        self.c_includes = {}
        self.type_names = dict(outer_scope.type_names)
        self.pxd_file_loaded = 0
        self.cimported_modules = []
        self.types_imported = set()
        self.included_files = []
        self.has_extern_class = 0
        self.cached_builtins = []
        self.undeclared_cached_builtins = []
        self.namespace_cname = self.module_cname
        self._cached_tuple_types = {}
        self.process_include(Code.IncludeCode('Python.h', initial=True))

    def qualifying_scope(self):
        if False:
            print('Hello World!')
        return self.parent_module

    def global_scope(self):
        if False:
            return 10
        return self

    def lookup(self, name, language_level=None, str_is_str=None):
        if False:
            i = 10
            return i + 15
        entry = self.lookup_here(name)
        if entry is not None:
            return entry
        if language_level is None:
            language_level = self.context.language_level if self.context is not None else 3
        if str_is_str is None:
            str_is_str = language_level == 2 or (self.context is not None and Future.unicode_literals not in self.context.future_directives)
        return self.outer_scope.lookup(name, language_level=language_level, str_is_str=str_is_str)

    def declare_tuple_type(self, pos, components):
        if False:
            for i in range(10):
                print('nop')
        components = tuple(components)
        try:
            ttype = self._cached_tuple_types[components]
        except KeyError:
            ttype = self._cached_tuple_types[components] = PyrexTypes.c_tuple_type(components)
        cname = ttype.cname
        entry = self.lookup_here(cname)
        if not entry:
            scope = StructOrUnionScope(cname)
            for (ix, component) in enumerate(components):
                scope.declare_var(name='f%s' % ix, type=component, pos=pos)
            struct_entry = self.declare_struct_or_union(cname + '_struct', 'struct', scope, typedef_flag=True, pos=pos, cname=cname)
            self.type_entries.remove(struct_entry)
            ttype.struct_entry = struct_entry
            entry = self.declare_type(cname, ttype, pos, cname)
        ttype.entry = entry
        return entry

    def declare_builtin(self, name, pos):
        if False:
            print('Hello World!')
        if not hasattr(builtins, name) and name not in Code.non_portable_builtins_map and (name not in Code.uncachable_builtins):
            if self.has_import_star:
                entry = self.declare_var(name, py_object_type, pos)
                return entry
            else:
                if Options.error_on_unknown_names:
                    error(pos, 'undeclared name not builtin: %s' % name)
                else:
                    warning(pos, 'undeclared name not builtin: %s' % name, 2)
                entry = self.declare(name, None, py_object_type, pos, 'private')
                entry.is_builtin = 1
                return entry
        if Options.cache_builtins:
            for entry in self.cached_builtins:
                if entry.name == name:
                    return entry
        if name == 'globals' and (not self.old_style_globals):
            return self.outer_scope.lookup('__Pyx_Globals')
        else:
            entry = self.declare(None, None, py_object_type, pos, 'private')
        if Options.cache_builtins and name not in Code.uncachable_builtins:
            entry.is_builtin = 1
            entry.is_const = 1
            entry.name = name
            entry.cname = Naming.builtin_prefix + name
            self.cached_builtins.append(entry)
            self.undeclared_cached_builtins.append(entry)
        else:
            entry.is_builtin = 1
            entry.name = name
        entry.qualified_name = self.builtin_scope().qualify_name(name)
        return entry

    def find_module(self, module_name, pos, relative_level=-1):
        if False:
            for i in range(10):
                print('nop')
        is_relative_import = relative_level is not None and relative_level > 0
        from_module = None
        absolute_fallback = False
        if relative_level is not None and relative_level > 0:
            from_module = self
            top_level = 1 if self.is_package else 0
            while relative_level > top_level and from_module:
                from_module = from_module.parent_module
                relative_level -= 1
        elif relative_level != 0:
            from_module = self.parent_module
            absolute_fallback = True
        module_scope = self.global_scope()
        return module_scope.context.find_module(module_name, from_module=from_module, pos=pos, absolute_fallback=absolute_fallback, relative_import=is_relative_import)

    def find_submodule(self, name, as_package=False):
        if False:
            for i in range(10):
                print('nop')
        if '.' in name:
            (name, submodule) = name.split('.', 1)
        else:
            submodule = None
        scope = self.lookup_submodule(name)
        if not scope:
            scope = ModuleScope(name, parent_module=self, context=self.context, is_package=True if submodule else as_package)
            self.module_entries[name] = scope
        if submodule:
            scope = scope.find_submodule(submodule, as_package=as_package)
        return scope

    def lookup_submodule(self, name):
        if False:
            for i in range(10):
                print('nop')
        if '.' in name:
            (name, submodule) = name.split('.', 1)
        else:
            submodule = None
        module = self.module_entries.get(name, None)
        if submodule and module is not None:
            module = module.lookup_submodule(submodule)
        return module

    def add_include_file(self, filename, verbatim_include=None, late=False):
        if False:
            print('Hello World!')
        '\n        Add `filename` as include file. Add `verbatim_include` as\n        verbatim text in the C file.\n        Both `filename` and `verbatim_include` can be `None` or empty.\n        '
        inc = Code.IncludeCode(filename, verbatim_include, late=late)
        self.process_include(inc)

    def process_include(self, inc):
        if False:
            while True:
                i = 10
        '\n        Add `inc`, which is an instance of `IncludeCode`, to this\n        `ModuleScope`. This either adds a new element to the\n        `c_includes` dict or it updates an existing entry.\n\n        In detail: the values of the dict `self.c_includes` are\n        instances of `IncludeCode` containing the code to be put in the\n        generated C file. The keys of the dict are needed to ensure\n        uniqueness in two ways: if an include file is specified in\n        multiple "cdef extern" blocks, only one `#include` statement is\n        generated. Second, the same include might occur multiple times\n        if we find it through multiple "cimport" paths. So we use the\n        generated code (of the form `#include "header.h"`) as dict key.\n\n        If verbatim code does not belong to any include file (i.e. it\n        was put in a `cdef extern from *` block), then we use a unique\n        dict key: namely, the `sortkey()`.\n\n        One `IncludeCode` object can contain multiple pieces of C code:\n        one optional "main piece" for the include file and several other\n        pieces for the verbatim code. The `IncludeCode.dict_update`\n        method merges the pieces of two different `IncludeCode` objects\n        if needed.\n        '
        key = inc.mainpiece()
        if key is None:
            key = inc.sortkey()
        inc.dict_update(self.c_includes, key)
        inc = self.c_includes[key]

    def add_imported_module(self, scope):
        if False:
            while True:
                i = 10
        if scope not in self.cimported_modules:
            for inc in scope.c_includes.values():
                self.process_include(inc)
            self.cimported_modules.append(scope)
            for m in scope.cimported_modules:
                self.add_imported_module(m)

    def add_imported_entry(self, name, entry, pos):
        if False:
            while True:
                i = 10
        if entry.is_pyglobal:
            entry.is_variable = True
        if entry not in self.entries:
            self.entries[name] = entry
        else:
            warning(pos, "'%s' redeclared  " % name, 0)

    def declare_module(self, name, scope, pos):
        if False:
            return 10
        entry = self.lookup_here(name)
        if entry:
            if entry.is_pyglobal and entry.as_module is scope:
                return entry
            if not (entry.is_pyglobal and (not entry.as_module)):
                return entry
        else:
            entry = self.declare_var(name, py_object_type, pos)
            entry.is_variable = 0
        entry.as_module = scope
        self.add_imported_module(scope)
        return entry

    def declare_var(self, name, type, pos, cname=None, visibility='private', api=False, in_pxd=False, is_cdef=False, pytyping_modifiers=None):
        if False:
            print('Hello World!')
        if visibility not in ('private', 'public', 'extern'):
            error(pos, 'Module-level variable cannot be declared %s' % visibility)
        self._reject_pytyping_modifiers(pos, pytyping_modifiers, ('typing.Optional',))
        if not is_cdef:
            if type is unspecified_type:
                type = py_object_type
            if not (type.is_pyobject and (not type.is_extension_type)):
                raise InternalError('Non-cdef global variable is not a generic Python object')
        if not cname:
            defining = not in_pxd
            if visibility == 'extern' or (visibility == 'public' and defining):
                cname = name
            else:
                cname = self.mangle(Naming.var_prefix, name)
        entry = self.lookup_here(name)
        if entry and entry.defined_in_pxd:
            if not entry.type.same_as(type):
                if visibility == 'extern' and entry.visibility == 'extern':
                    warning(pos, "Variable '%s' type does not match previous declaration" % name, 1)
                    entry.type = type
            if entry.visibility != 'private':
                mangled_cname = self.mangle(Naming.var_prefix, name)
                if entry.cname == mangled_cname:
                    cname = name
                    entry.cname = name
            if not entry.is_implemented:
                entry.is_implemented = True
                return entry
        entry = Scope.declare_var(self, name, type, pos, cname=cname, visibility=visibility, api=api, in_pxd=in_pxd, is_cdef=is_cdef, pytyping_modifiers=pytyping_modifiers)
        if is_cdef:
            entry.is_cglobal = 1
            if entry.type.declaration_value:
                entry.init = entry.type.declaration_value
            self.var_entries.append(entry)
        else:
            entry.is_pyglobal = 1
        if Options.cimport_from_pyx:
            entry.used = 1
        return entry

    def declare_cfunction(self, name, type, pos, cname=None, visibility='private', api=0, in_pxd=0, defining=0, modifiers=(), utility_code=None, overridable=False):
        if False:
            while True:
                i = 10
        if not defining and 'inline' in modifiers:
            warning(pos, 'Declarations should not be declared inline.', 1)
        if not cname:
            if visibility == 'extern' or (visibility == 'public' and defining):
                cname = name
            else:
                cname = self.mangle(Naming.func_prefix, name)
        if visibility == 'extern' and type.optional_arg_count:
            error(pos, 'Extern functions cannot have default arguments values.')
        entry = self.lookup_here(name)
        if entry and entry.defined_in_pxd:
            if entry.visibility != 'private':
                mangled_cname = self.mangle(Naming.func_prefix, name)
                if entry.cname == mangled_cname:
                    cname = name
                    entry.cname = cname
                    entry.func_cname = cname
        entry = Scope.declare_cfunction(self, name, type, pos, cname=cname, visibility=visibility, api=api, in_pxd=in_pxd, defining=defining, modifiers=modifiers, utility_code=utility_code, overridable=overridable)
        return entry

    def declare_global(self, name, pos):
        if False:
            while True:
                i = 10
        entry = self.lookup_here(name)
        if not entry:
            self.declare_var(name, py_object_type, pos)

    def use_utility_code(self, new_code):
        if False:
            i = 10
            return i + 15
        if new_code is not None:
            self.utility_code_list.append(new_code)

    def use_entry_utility_code(self, entry):
        if False:
            for i in range(10):
                print('nop')
        if entry is None:
            return
        if entry.utility_code:
            self.utility_code_list.append(entry.utility_code)
        if entry.utility_code_definition:
            self.utility_code_list.append(entry.utility_code_definition)

    def declare_c_class(self, name, pos, defining=0, implementing=0, module_name=None, base_type=None, objstruct_cname=None, typeobj_cname=None, typeptr_cname=None, visibility='private', typedef_flag=0, api=0, check_size=None, buffer_defaults=None, shadow=0):
        if False:
            i = 10
            return i + 15
        if typedef_flag and visibility != 'extern':
            if not (visibility == 'public' or api):
                warning(pos, "ctypedef only valid for 'extern' , 'public', and 'api'", 2)
            objtypedef_cname = objstruct_cname
            typedef_flag = 0
        else:
            objtypedef_cname = None
        entry = self.lookup_here(name)
        if entry and (not shadow):
            type = entry.type
            if not (entry.is_type and type.is_extension_type):
                entry = None
            else:
                scope = type.scope
                if typedef_flag and (not scope or scope.defined):
                    self.check_previous_typedef_flag(entry, typedef_flag, pos)
                if scope and scope.defined or (base_type and type.base_type):
                    if base_type and base_type is not type.base_type:
                        error(pos, 'Base type does not match previous declaration')
                if base_type and (not type.base_type):
                    type.base_type = base_type
        if not entry or shadow:
            type = PyrexTypes.PyExtensionType(name, typedef_flag, base_type, visibility == 'extern', check_size=check_size)
            type.pos = pos
            type.buffer_defaults = buffer_defaults
            if objtypedef_cname is not None:
                type.objtypedef_cname = objtypedef_cname
            if visibility == 'extern':
                type.module_name = module_name
            else:
                type.module_name = self.qualified_name
            if typeptr_cname:
                type.typeptr_cname = typeptr_cname
            else:
                type.typeptr_cname = self.mangle(Naming.typeptr_prefix, name)
            entry = self.declare_type(name, type, pos, visibility=visibility, defining=0, shadow=shadow)
            entry.is_cclass = True
            if objstruct_cname:
                type.objstruct_cname = objstruct_cname
            elif not entry.in_cinclude:
                type.objstruct_cname = self.mangle(Naming.objstruct_prefix, name)
            else:
                error(entry.pos, "Object name required for 'public' or 'extern' C class")
            self.attach_var_entry_to_c_class(entry)
            self.c_class_entries.append(entry)
        if not type.scope:
            if defining or implementing:
                scope = CClassScope(name=name, outer_scope=self, visibility=visibility, parent_type=type)
                scope.directives = self.directives.copy()
                if base_type and base_type.scope:
                    scope.declare_inherited_c_attributes(base_type.scope)
                type.set_scope(scope)
                self.type_entries.append(entry)
        elif defining and type.scope.defined:
            error(pos, "C class '%s' already defined" % name)
        elif implementing and type.scope.implemented:
            error(pos, "C class '%s' already implemented" % name)
        if defining:
            entry.defined_in_pxd = 1
        if implementing:
            entry.pos = pos
        if visibility != 'private' and entry.visibility != visibility:
            error(pos, "Class '%s' previously declared as '%s'" % (name, entry.visibility))
        if api:
            entry.api = 1
        if objstruct_cname:
            if type.objstruct_cname and type.objstruct_cname != objstruct_cname:
                error(pos, 'Object struct name differs from previous declaration')
            type.objstruct_cname = objstruct_cname
        if typeobj_cname:
            if type.typeobj_cname and type.typeobj_cname != typeobj_cname:
                error(pos, 'Type object name differs from previous declaration')
            type.typeobj_cname = typeobj_cname
        if self.directives.get('final'):
            entry.type.is_final_type = True
        collection_type = self.directives.get('collection_type')
        if collection_type:
            from .UtilityCode import NonManglingModuleScope
            if not isinstance(self, NonManglingModuleScope):
                error(pos, "'collection_type' is not a public cython directive")
        if collection_type == 'sequence':
            entry.type.has_sequence_flag = True
        entry.used = True
        return entry

    def allocate_vtable_names(self, entry):
        if False:
            return 10
        type = entry.type
        if type.base_type and type.base_type.vtabslot_cname:
            type.vtabslot_cname = '%s.%s' % (Naming.obj_base_cname, type.base_type.vtabslot_cname)
        elif type.scope and type.scope.cfunc_entries:
            entry_count = len(type.scope.cfunc_entries)
            base_type = type.base_type
            while base_type:
                if not base_type.scope or entry_count > len(base_type.scope.cfunc_entries):
                    break
                if base_type.is_builtin_type:
                    return
                base_type = base_type.base_type
            type.vtabslot_cname = Naming.vtabslot_cname
        if type.vtabslot_cname:
            type.vtabstruct_cname = self.mangle(Naming.vtabstruct_prefix, entry.name)
            type.vtabptr_cname = self.mangle(Naming.vtabptr_prefix, entry.name)

    def check_c_classes_pxd(self):
        if False:
            for i in range(10):
                print('nop')
        for entry in self.c_class_entries:
            if not entry.type.scope:
                error(entry.pos, "C class '%s' is declared but not defined" % entry.name)

    def check_c_class(self, entry):
        if False:
            print('Hello World!')
        type = entry.type
        name = entry.name
        visibility = entry.visibility
        if not type.scope:
            error(entry.pos, "C class '%s' is declared but not defined" % name)
        if visibility != 'extern' and (not type.typeobj_cname):
            type.typeobj_cname = self.mangle(Naming.typeobj_prefix, name)
        if type.scope:
            for method_entry in type.scope.cfunc_entries:
                if not method_entry.is_inherited and (not method_entry.func_cname):
                    error(method_entry.pos, "C method '%s' is declared but not defined" % method_entry.name)
        if type.vtabslot_cname:
            type.vtable_cname = self.mangle(Naming.vtable_prefix, entry.name)

    def check_c_classes(self):
        if False:
            return 10
        debug_check_c_classes = 0
        if debug_check_c_classes:
            print('Scope.check_c_classes: checking scope ' + self.qualified_name)
        for entry in self.c_class_entries:
            if debug_check_c_classes:
                print('...entry %s %s' % (entry.name, entry))
                print('......type = ', entry.type)
                print('......visibility = ', entry.visibility)
            self.check_c_class(entry)

    def check_c_functions(self):
        if False:
            print('Hello World!')
        for (name, entry) in self.entries.items():
            if entry.is_cfunction:
                if entry.defined_in_pxd and entry.scope is self and (entry.visibility != 'extern') and (not entry.in_cinclude) and (not entry.is_implemented):
                    error(entry.pos, "Non-extern C function '%s' declared but not defined" % name)

    def attach_var_entry_to_c_class(self, entry):
        if False:
            for i in range(10):
                print('nop')
        from . import Builtin
        var_entry = Entry(name=entry.name, type=Builtin.type_type, pos=entry.pos, cname=entry.type.typeptr_cname)
        var_entry.qualified_name = entry.qualified_name
        var_entry.is_variable = 1
        var_entry.is_cglobal = 1
        var_entry.is_readonly = 1
        var_entry.scope = entry.scope
        entry.as_variable = var_entry

    def is_cpp(self):
        if False:
            i = 10
            return i + 15
        return self.cpp

    def infer_types(self):
        if False:
            while True:
                i = 10
        from .TypeInference import PyObjectTypeInferer
        PyObjectTypeInferer().infer_types(self)

class LocalScope(Scope):
    is_local_scope = True
    has_with_gil_block = False
    _in_with_gil_block = False

    def __init__(self, name, outer_scope, parent_scope=None):
        if False:
            for i in range(10):
                print('nop')
        if parent_scope is None:
            parent_scope = outer_scope
        Scope.__init__(self, name, outer_scope, parent_scope)

    def mangle(self, prefix, name):
        if False:
            while True:
                i = 10
        return punycodify_name(prefix + name)

    def declare_arg(self, name, type, pos):
        if False:
            print('Hello World!')
        name = self.mangle_class_private_name(name)
        cname = self.mangle(Naming.var_prefix, name)
        entry = self.declare(name, cname, type, pos, 'private')
        entry.is_variable = 1
        if type.is_pyobject:
            entry.init = '0'
        entry.is_arg = 1
        self.arg_entries.append(entry)
        return entry

    def declare_var(self, name, type, pos, cname=None, visibility='private', api=False, in_pxd=False, is_cdef=False, pytyping_modifiers=None):
        if False:
            i = 10
            return i + 15
        name = self.mangle_class_private_name(name)
        if visibility in ('public', 'readonly'):
            error(pos, 'Local variable cannot be declared %s' % visibility)
        entry = Scope.declare_var(self, name, type, pos, cname=cname, visibility=visibility, api=api, in_pxd=in_pxd, is_cdef=is_cdef, pytyping_modifiers=pytyping_modifiers)
        if entry.type.declaration_value:
            entry.init = entry.type.declaration_value
        entry.is_local = 1
        entry.in_with_gil_block = self._in_with_gil_block
        self.var_entries.append(entry)
        return entry

    def declare_global(self, name, pos):
        if False:
            while True:
                i = 10
        if self.lookup_here(name):
            warning(pos, "'%s' redeclared  ", 0)
        else:
            entry = self.global_scope().lookup_target(name)
            self.entries[name] = entry

    def declare_nonlocal(self, name, pos):
        if False:
            i = 10
            return i + 15
        orig_entry = self.lookup_here(name)
        if orig_entry and orig_entry.scope is self and (not orig_entry.from_closure):
            error(pos, "'%s' redeclared as nonlocal" % name)
            orig_entry.already_declared_here()
        else:
            entry = self.lookup(name)
            if entry is None or not entry.from_closure:
                error(pos, "no binding for nonlocal '%s' found" % name)

    def _create_inner_entry_for_closure(self, name, entry):
        if False:
            return 10
        entry.in_closure = True
        inner_entry = InnerEntry(entry, self)
        inner_entry.is_variable = True
        self.entries[name] = inner_entry
        return inner_entry

    def lookup(self, name):
        if False:
            while True:
                i = 10
        entry = Scope.lookup(self, name)
        if entry is not None:
            entry_scope = entry.scope
            while entry_scope.is_comprehension_scope:
                entry_scope = entry_scope.outer_scope
            if entry_scope is not self and entry_scope.is_closure_scope:
                if hasattr(entry.scope, 'scope_class'):
                    raise InternalError('lookup() after scope class created.')
                return self._create_inner_entry_for_closure(name, entry)
        return entry

    def mangle_closure_cnames(self, outer_scope_cname):
        if False:
            i = 10
            return i + 15
        for scope in self.iter_local_scopes():
            for entry in scope.entries.values():
                if entry.from_closure:
                    cname = entry.outer_entry.cname
                    if self.is_passthrough:
                        entry.cname = cname
                    else:
                        if cname.startswith(Naming.cur_scope_cname):
                            cname = cname[len(Naming.cur_scope_cname) + 2:]
                        entry.cname = '%s->%s' % (outer_scope_cname, cname)
                elif entry.in_closure:
                    entry.original_cname = entry.cname
                    entry.cname = '%s->%s' % (Naming.cur_scope_cname, entry.cname)
                    if entry.type.is_cpp_class and entry.scope.directives['cpp_locals']:
                        entry.make_cpp_optional()

class ComprehensionScope(Scope):
    """Scope for comprehensions (but not generator expressions, which use ClosureScope).
    As opposed to generators, these can be easily inlined in some cases, so all
    we really need is a scope that holds the loop variable(s).
    """
    is_comprehension_scope = True

    def __init__(self, outer_scope):
        if False:
            for i in range(10):
                print('nop')
        parent_scope = outer_scope
        while parent_scope.is_comprehension_scope:
            parent_scope = parent_scope.parent_scope
        name = parent_scope.global_scope().next_id(Naming.genexpr_id_ref)
        Scope.__init__(self, name, outer_scope, parent_scope)
        self.directives = outer_scope.directives
        self.genexp_prefix = '%s%d%s' % (Naming.pyrex_prefix, len(name), name)
        while outer_scope.is_comprehension_scope or outer_scope.is_c_class_scope or outer_scope.is_py_class_scope:
            outer_scope = outer_scope.outer_scope
        self.var_entries = outer_scope.var_entries
        outer_scope.subscopes.add(self)

    def mangle(self, prefix, name):
        if False:
            while True:
                i = 10
        return '%s%s' % (self.genexp_prefix, self.parent_scope.mangle(prefix, name))

    def declare_var(self, name, type, pos, cname=None, visibility='private', api=False, in_pxd=False, is_cdef=True, pytyping_modifiers=None):
        if False:
            return 10
        if type is unspecified_type:
            outer_entry = self.outer_scope.lookup(name)
            if outer_entry and outer_entry.is_variable:
                type = outer_entry.type
        self._reject_pytyping_modifiers(pos, pytyping_modifiers)
        cname = '%s%s' % (self.genexp_prefix, self.parent_scope.mangle(Naming.var_prefix, name or self.next_id()))
        entry = self.declare(name, cname, type, pos, visibility)
        entry.is_variable = True
        if self.parent_scope.is_module_scope:
            entry.is_cglobal = True
        else:
            entry.is_local = True
        entry.in_subscope = True
        self.var_entries.append(entry)
        self.entries[name] = entry
        return entry

    def declare_assignment_expression_target(self, name, type, pos):
        if False:
            print('Hello World!')
        return self.parent_scope.declare_var(name, type, pos)

    def declare_pyfunction(self, name, pos, allow_redefine=False):
        if False:
            return 10
        return self.outer_scope.declare_pyfunction(name, pos, allow_redefine)

    def declare_lambda_function(self, func_cname, pos):
        if False:
            return 10
        return self.outer_scope.declare_lambda_function(func_cname, pos)

    def add_lambda_def(self, def_node):
        if False:
            i = 10
            return i + 15
        return self.outer_scope.add_lambda_def(def_node)

    def lookup_assignment_expression_target(self, name):
        if False:
            for i in range(10):
                print('nop')
        entry = self.lookup_here(name)
        if not entry:
            entry = self.parent_scope.lookup_assignment_expression_target(name)
        return entry

class ClosureScope(LocalScope):
    is_closure_scope = True

    def __init__(self, name, scope_name, outer_scope, parent_scope=None):
        if False:
            i = 10
            return i + 15
        LocalScope.__init__(self, name, outer_scope, parent_scope)
        self.closure_cname = '%s%s' % (Naming.closure_scope_prefix, scope_name)

    def declare_pyfunction(self, name, pos, allow_redefine=False):
        if False:
            return 10
        return LocalScope.declare_pyfunction(self, name, pos, allow_redefine, visibility='private')

    def declare_assignment_expression_target(self, name, type, pos):
        if False:
            for i in range(10):
                print('nop')
        return self.declare_var(name, type, pos)

class GeneratorExpressionScope(ClosureScope):
    is_generator_expression_scope = True

    def declare_assignment_expression_target(self, name, type, pos):
        if False:
            print('Hello World!')
        entry = self.parent_scope.declare_var(name, type, pos)
        return self._create_inner_entry_for_closure(name, entry)

    def lookup_assignment_expression_target(self, name):
        if False:
            return 10
        entry = self.lookup_here(name)
        if not entry:
            entry = self.parent_scope.lookup_assignment_expression_target(name)
            if entry:
                return self._create_inner_entry_for_closure(name, entry)
        return entry

class StructOrUnionScope(Scope):

    def __init__(self, name='?'):
        if False:
            i = 10
            return i + 15
        Scope.__init__(self, name, outer_scope=None, parent_scope=None)

    def declare_var(self, name, type, pos, cname=None, visibility='private', api=False, in_pxd=False, is_cdef=False, pytyping_modifiers=None, allow_pyobject=False, allow_memoryview=False, allow_refcounted=False):
        if False:
            print('Hello World!')
        if not cname:
            cname = name
            if visibility == 'private':
                cname = c_safe_identifier(cname)
        if type.is_cfunction:
            type = PyrexTypes.CPtrType(type)
        self._reject_pytyping_modifiers(pos, pytyping_modifiers)
        entry = self.declare(name, cname, type, pos, visibility)
        entry.is_variable = 1
        self.var_entries.append(entry)
        if type.is_pyobject:
            if not allow_pyobject:
                error(pos, 'C struct/union member cannot be a Python object')
        elif type.is_memoryviewslice:
            if not allow_memoryview:
                error(pos, 'C struct/union member cannot be a memory view')
        elif type.needs_refcounting:
            if not allow_refcounted:
                error(pos, "C struct/union member cannot be reference-counted type '%s'" % type)
        return entry

    def declare_cfunction(self, name, type, pos, cname=None, visibility='private', api=0, in_pxd=0, defining=0, modifiers=(), overridable=False):
        if False:
            print('Hello World!')
        if overridable:
            error(pos, "C struct/union member cannot be declared 'cpdef'")
        return self.declare_var(name, type, pos, cname=cname, visibility=visibility)

class ClassScope(Scope):
    scope_predefined_names = ['__module__', '__qualname__']

    def mangle_class_private_name(self, name):
        if False:
            i = 10
            return i + 15
        if name and name.lower().startswith('__pyx_'):
            return name
        if name and name.startswith('__') and (not name.endswith('__')):
            name = EncodedString('_%s%s' % (self.class_name.lstrip('_'), name))
        return name

    def __init__(self, name, outer_scope):
        if False:
            print('Hello World!')
        Scope.__init__(self, name, outer_scope, outer_scope)
        self.class_name = name
        self.doc = None

    def lookup(self, name):
        if False:
            i = 10
            return i + 15
        entry = Scope.lookup(self, name)
        if entry:
            return entry
        if name == 'classmethod':
            entry = Entry('classmethod', '__Pyx_Method_ClassMethod', PyrexTypes.CFuncType(py_object_type, [PyrexTypes.CFuncTypeArg('', py_object_type, None)], 0, 0))
            entry.utility_code_definition = Code.UtilityCode.load_cached('ClassMethod', 'CythonFunction.c')
            self.use_entry_utility_code(entry)
            entry.is_cfunction = 1
        return entry

class PyClassScope(ClassScope):
    is_py_class_scope = 1

    def declare_var(self, name, type, pos, cname=None, visibility='private', api=False, in_pxd=False, is_cdef=False, pytyping_modifiers=None):
        if False:
            print('Hello World!')
        name = self.mangle_class_private_name(name)
        if type is unspecified_type:
            type = py_object_type
        entry = Scope.declare_var(self, name, type, pos, cname=cname, visibility=visibility, api=api, in_pxd=in_pxd, is_cdef=is_cdef, pytyping_modifiers=pytyping_modifiers)
        entry.is_pyglobal = 1
        entry.is_pyclass_attr = 1
        return entry

    def declare_nonlocal(self, name, pos):
        if False:
            while True:
                i = 10
        orig_entry = self.lookup_here(name)
        if orig_entry and orig_entry.scope is self and (not orig_entry.from_closure):
            error(pos, "'%s' redeclared as nonlocal" % name)
            orig_entry.already_declared_here()
        else:
            entry = self.lookup(name)
            if entry is None:
                error(pos, "no binding for nonlocal '%s' found" % name)
            else:
                self.entries[name] = entry

    def declare_global(self, name, pos):
        if False:
            for i in range(10):
                print('nop')
        if self.lookup_here(name):
            warning(pos, "'%s' redeclared  ", 0)
        else:
            entry = self.global_scope().lookup_target(name)
            self.entries[name] = entry

    def add_default_value(self, type):
        if False:
            print('Hello World!')
        return self.outer_scope.add_default_value(type)

class CClassScope(ClassScope):
    is_c_class_scope = 1
    is_closure_class_scope = False
    has_pyobject_attrs = False
    has_memoryview_attrs = False
    has_cpp_constructable_attrs = False
    has_cyclic_pyobject_attrs = False
    defined = False
    implemented = False

    def __init__(self, name, outer_scope, visibility, parent_type):
        if False:
            for i in range(10):
                print('nop')
        ClassScope.__init__(self, name, outer_scope)
        if visibility != 'extern':
            self.method_table_cname = outer_scope.mangle(Naming.methtab_prefix, name)
            self.getset_table_cname = outer_scope.mangle(Naming.gstab_prefix, name)
        self.property_entries = []
        self.inherited_var_entries = []
        self.parent_type = parent_type
        if (parent_type.is_builtin_type or parent_type.is_extension_type) and parent_type.typeptr_cname:
            self.namespace_cname = '(PyObject *)%s' % parent_type.typeptr_cname

    def needs_gc(self):
        if False:
            return 10
        if self.has_cyclic_pyobject_attrs and (not self.directives.get('no_gc', False)):
            return True
        base_type = self.parent_type.base_type
        if base_type and base_type.scope is not None:
            return base_type.scope.needs_gc()
        elif self.parent_type.is_builtin_type:
            return not self.parent_type.is_gc_simple
        return False

    def needs_trashcan(self):
        if False:
            while True:
                i = 10
        directive = self.directives.get('trashcan')
        if directive is False:
            return False
        if directive and self.has_cyclic_pyobject_attrs:
            return True
        base_type = self.parent_type.base_type
        if base_type and base_type.scope is not None:
            return base_type.scope.needs_trashcan()
        return self.parent_type.builtin_trashcan

    def needs_tp_clear(self):
        if False:
            i = 10
            return i + 15
        '\n        Do we need to generate an implementation for the tp_clear slot? Can\n        be disabled to keep references for the __dealloc__ cleanup function.\n        '
        return self.needs_gc() and (not self.directives.get('no_gc_clear', False))

    def may_have_finalize(self):
        if False:
            print('Hello World!')
        "\n        This covers cases where we definitely have a __del__ function\n        and also cases where one of the base classes could have a __del__\n        function but we don't know.\n        "
        current_type_scope = self
        while current_type_scope:
            del_entry = current_type_scope.lookup_here('__del__')
            if del_entry and del_entry.is_special:
                return True
            if current_type_scope.parent_type.is_extern or not current_type_scope.implemented or current_type_scope.parent_type.multiple_bases:
                return True
            current_base_type = current_type_scope.parent_type.base_type
            current_type_scope = current_base_type.scope if current_base_type else None
        return False

    def get_refcounted_entries(self, include_weakref=False, include_gc_simple=True):
        if False:
            i = 10
            return i + 15
        py_attrs = []
        py_buffers = []
        memoryview_slices = []
        for entry in self.var_entries:
            if entry.type.is_pyobject:
                if include_weakref or (self.is_closure_class_scope or entry.name != '__weakref__'):
                    if include_gc_simple or not entry.type.is_gc_simple:
                        py_attrs.append(entry)
            elif entry.type == PyrexTypes.c_py_buffer_type:
                py_buffers.append(entry)
            elif entry.type.is_memoryviewslice:
                memoryview_slices.append(entry)
        have_entries = py_attrs or py_buffers or memoryview_slices
        return (have_entries, (py_attrs, py_buffers, memoryview_slices))

    def declare_var(self, name, type, pos, cname=None, visibility='private', api=False, in_pxd=False, is_cdef=False, pytyping_modifiers=None):
        if False:
            i = 10
            return i + 15
        name = self.mangle_class_private_name(name)
        if pytyping_modifiers:
            if 'typing.ClassVar' in pytyping_modifiers:
                is_cdef = 0
                if not type.is_pyobject:
                    if not type.equivalent_type:
                        warning(pos, "ClassVar[] requires the type to be a Python object type. Found '%s', using object instead." % type)
                        type = py_object_type
                    else:
                        type = type.equivalent_type
            if 'dataclasses.InitVar' in pytyping_modifiers and (not self.is_c_dataclass_scope):
                error(pos, 'Use of cython.dataclasses.InitVar does not make sense outside a dataclass')
        if is_cdef:
            if self.defined:
                error(pos, 'C attributes cannot be added in implementation part of extension type defined in a pxd')
            if not self.is_closure_class_scope and get_slot_table(self.directives).get_special_method_signature(name):
                error(pos, "The name '%s' is reserved for a special method." % name)
            if not cname:
                cname = name
                if visibility == 'private':
                    cname = c_safe_identifier(cname)
                cname = punycodify_name(cname, Naming.unicode_structmember_prefix)
            entry = self.declare(name, cname, type, pos, visibility)
            entry.is_variable = 1
            self.var_entries.append(entry)
            entry.pytyping_modifiers = pytyping_modifiers
            if type.is_cpp_class and visibility != 'extern':
                if self.directives['cpp_locals']:
                    entry.make_cpp_optional()
                else:
                    type.check_nullary_constructor(pos)
            if type.is_memoryviewslice:
                self.has_memoryview_attrs = True
            elif type.needs_cpp_construction:
                self.use_utility_code(Code.UtilityCode('#include <new>'))
                self.has_cpp_constructable_attrs = True
            elif type.is_pyobject and (self.is_closure_class_scope or name != '__weakref__'):
                self.has_pyobject_attrs = True
                if not type.is_builtin_type or not type.scope or type.scope.needs_gc():
                    self.has_cyclic_pyobject_attrs = True
            if visibility not in ('private', 'public', 'readonly'):
                error(pos, 'Attribute of extension type cannot be declared %s' % visibility)
            if visibility in ('public', 'readonly'):
                entry.needs_property = True
                if not self.is_closure_class_scope and name == '__weakref__':
                    error(pos, 'Special attribute __weakref__ cannot be exposed to Python')
                if not (type.is_pyobject or type.can_coerce_to_pyobject(self)):
                    error(pos, "C attribute of type '%s' cannot be accessed from Python" % type)
            else:
                entry.needs_property = False
            return entry
        else:
            if type is unspecified_type:
                type = py_object_type
            entry = Scope.declare_var(self, name, type, pos, cname=cname, visibility=visibility, api=api, in_pxd=in_pxd, is_cdef=is_cdef, pytyping_modifiers=pytyping_modifiers)
            entry.is_member = 1
            entry.is_pyglobal = 1
            return entry

    def declare_pyfunction(self, name, pos, allow_redefine=False):
        if False:
            for i in range(10):
                print('nop')
        if name in richcmp_special_methods:
            if self.lookup_here('__richcmp__'):
                error(pos, 'Cannot define both % and __richcmp__' % name)
        elif name == '__richcmp__':
            for n in richcmp_special_methods:
                if self.lookup_here(n):
                    error(pos, 'Cannot define both % and __richcmp__' % n)
        if name == '__new__':
            error(pos, '__new__ method of extension type will change semantics in a future version of Pyrex and Cython. Use __cinit__ instead.')
        entry = self.declare_var(name, py_object_type, pos, visibility='extern')
        special_sig = get_slot_table(self.directives).get_special_method_signature(name)
        if special_sig:
            entry.signature = special_sig
            entry.is_special = 1
        else:
            entry.signature = pymethod_signature
            entry.is_special = 0
        self.pyfunc_entries.append(entry)
        return entry

    def lookup_here(self, name):
        if False:
            print('Hello World!')
        if not self.is_closure_class_scope and name == '__new__':
            name = EncodedString('__cinit__')
        entry = ClassScope.lookup_here(self, name)
        if entry and entry.is_builtin_cmethod:
            if not self.parent_type.is_builtin_type:
                if not self.parent_type.is_final_type:
                    return None
        return entry

    def declare_cfunction(self, name, type, pos, cname=None, visibility='private', api=0, in_pxd=0, defining=0, modifiers=(), utility_code=None, overridable=False):
        if False:
            for i in range(10):
                print('nop')
        name = self.mangle_class_private_name(name)
        if get_slot_table(self.directives).get_special_method_signature(name) and (not self.parent_type.is_builtin_type):
            error(pos, "Special methods must be declared with 'def', not 'cdef'")
        args = type.args
        if not type.is_static_method:
            if not args:
                error(pos, 'C method has no self argument')
            elif not self.parent_type.assignable_from(args[0].type):
                error(pos, "Self argument (%s) of C method '%s' does not match parent type (%s)" % (args[0].type, name, self.parent_type))
        entry = self.lookup_here(name)
        if cname is None:
            cname = punycodify_name(c_safe_identifier(name), Naming.unicode_vtabentry_prefix)
        if entry:
            if not entry.is_cfunction:
                error(pos, "'%s' redeclared " % name)
                entry.already_declared_here()
            else:
                if defining and entry.func_cname:
                    error(pos, "'%s' already defined" % name)
                if entry.is_final_cmethod and entry.is_inherited:
                    error(pos, 'Overriding final methods is not allowed')
                elif type.same_c_signature_as(entry.type, as_cmethod=1) and type.nogil == entry.type.nogil:
                    entry.type = entry.type.with_with_gil(type.with_gil)
                elif type.compatible_signature_with(entry.type, as_cmethod=1) and type.nogil == entry.type.nogil:
                    if self.defined and (not in_pxd) and (not type.same_c_signature_as_resolved_type(entry.type, as_cmethod=1, as_pxd_definition=1)):
                        warning(pos, "Compatible but non-identical C method '%s' not redeclared in definition part of extension type '%s'.  This may cause incorrect vtables to be generated." % (name, self.class_name), 2)
                        warning(entry.pos, 'Previous declaration is here', 2)
                    entry = self.add_cfunction(name, type, pos, cname, visibility='ignore', modifiers=modifiers)
                else:
                    error(pos, 'Signature not compatible with previous declaration')
                    error(entry.pos, 'Previous declaration is here')
        else:
            if self.defined:
                error(pos, "C method '%s' not previously declared in definition part of extension type '%s'" % (name, self.class_name))
            entry = self.add_cfunction(name, type, pos, cname, visibility, modifiers)
        if defining:
            entry.func_cname = self.mangle(Naming.func_prefix, name)
        entry.utility_code = utility_code
        type.entry = entry
        if u'inline' in modifiers:
            entry.is_inline_cmethod = True
        if self.parent_type.is_final_type or entry.is_inline_cmethod or self.directives.get('final'):
            entry.is_final_cmethod = True
            entry.final_func_cname = entry.func_cname
        return entry

    def add_cfunction(self, name, type, pos, cname, visibility, modifiers, inherited=False):
        if False:
            return 10
        prev_entry = self.lookup_here(name)
        entry = ClassScope.add_cfunction(self, name, type, pos, cname, visibility, modifiers, inherited=inherited)
        entry.is_cmethod = 1
        entry.prev_entry = prev_entry
        return entry

    def declare_builtin_cfunction(self, name, type, cname, utility_code=None):
        if False:
            while True:
                i = 10
        name = EncodedString(name)
        entry = self.declare_cfunction(name, type, pos=None, cname=cname, visibility='extern', utility_code=utility_code)
        var_entry = Entry(name, name, py_object_type)
        var_entry.qualified_name = name
        var_entry.is_variable = 1
        var_entry.is_builtin = 1
        var_entry.utility_code = utility_code
        var_entry.scope = entry.scope
        entry.as_variable = var_entry
        return entry

    def declare_property(self, name, doc, pos, ctype=None, property_scope=None):
        if False:
            print('Hello World!')
        entry = self.lookup_here(name)
        if entry is None:
            entry = self.declare(name, name, py_object_type if ctype is None else ctype, pos, 'private')
        entry.is_property = True
        if ctype is not None:
            entry.is_cproperty = True
        entry.doc = doc
        if property_scope is None:
            entry.scope = PropertyScope(name, class_scope=self)
        else:
            entry.scope = property_scope
        self.property_entries.append(entry)
        return entry

    def declare_cproperty(self, name, type, cfunc_name, doc=None, pos=None, visibility='extern', nogil=False, with_gil=False, exception_value=None, exception_check=False, utility_code=None):
        if False:
            print('Hello World!')
        'Internal convenience method to declare a C property function in one go.\n        '
        property_entry = self.declare_property(name, doc=doc, ctype=type, pos=pos)
        cfunc_entry = property_entry.scope.declare_cfunction(name=name, type=PyrexTypes.CFuncType(type, [PyrexTypes.CFuncTypeArg('self', self.parent_type, pos=None)], nogil=nogil, with_gil=with_gil, exception_value=exception_value, exception_check=exception_check), cname=cfunc_name, utility_code=utility_code, visibility=visibility, pos=pos)
        return (property_entry, cfunc_entry)

    def declare_inherited_c_attributes(self, base_scope):
        if False:
            for i in range(10):
                print('nop')

        def adapt(cname):
            if False:
                return 10
            return '%s.%s' % (Naming.obj_base_cname, base_entry.cname)
        entries = base_scope.inherited_var_entries + base_scope.var_entries
        for base_entry in entries:
            entry = self.declare(base_entry.name, adapt(base_entry.cname), base_entry.type, None, 'private')
            entry.is_variable = 1
            entry.is_inherited = True
            entry.annotation = base_entry.annotation
            self.inherited_var_entries.append(entry)
        for base_entry in base_scope.cfunc_entries[:]:
            if base_entry.type.is_fused:
                base_entry.type.get_all_specialized_function_types()
        for base_entry in base_scope.cfunc_entries:
            cname = base_entry.cname
            var_entry = base_entry.as_variable
            is_builtin = var_entry and var_entry.is_builtin
            if not is_builtin:
                cname = adapt(cname)
            entry = self.add_cfunction(base_entry.name, base_entry.type, base_entry.pos, cname, base_entry.visibility, base_entry.func_modifiers, inherited=True)
            entry.is_inherited = 1
            if base_entry.is_final_cmethod:
                entry.is_final_cmethod = True
                entry.is_inline_cmethod = base_entry.is_inline_cmethod
                if self.parent_scope == base_scope.parent_scope or entry.is_inline_cmethod:
                    entry.final_func_cname = base_entry.final_func_cname
            if is_builtin:
                entry.is_builtin_cmethod = True
                entry.as_variable = var_entry
            if base_entry.utility_code:
                entry.utility_code = base_entry.utility_code

class CppClassScope(Scope):
    is_cpp_class_scope = 1
    default_constructor = None
    type = None

    def __init__(self, name, outer_scope, templates=None):
        if False:
            return 10
        Scope.__init__(self, name, outer_scope, None)
        self.directives = outer_scope.directives
        self.inherited_var_entries = []
        if templates is not None:
            for T in templates:
                template_entry = self.declare(T, T, PyrexTypes.TemplatePlaceholderType(T), None, 'extern')
                template_entry.is_type = 1

    def declare_var(self, name, type, pos, cname=None, visibility='extern', api=False, in_pxd=False, is_cdef=False, defining=False, pytyping_modifiers=None):
        if False:
            for i in range(10):
                print('nop')
        if not cname:
            cname = name
        self._reject_pytyping_modifiers(pos, pytyping_modifiers)
        entry = self.lookup_here(name)
        if defining and entry is not None:
            if entry.type.same_as(type):
                entry.type = entry.type.with_with_gil(type.with_gil)
            elif type.is_cfunction and type.compatible_signature_with(entry.type):
                entry.type = type
            else:
                error(pos, 'Function signature does not match previous declaration')
        else:
            entry = self.declare(name, cname, type, pos, visibility)
        entry.is_variable = 1
        if type.is_cfunction and self.type:
            if not self.type.get_fused_types():
                entry.func_cname = '%s::%s' % (self.type.empty_declaration_code(), cname)
        if name != 'this' and (defining or name != '<init>'):
            self.var_entries.append(entry)
        return entry

    def declare_cfunction(self, name, type, pos, cname=None, visibility='extern', api=0, in_pxd=0, defining=0, modifiers=(), utility_code=None, overridable=False):
        if False:
            return 10
        class_name = self.name.split('::')[-1]
        if name in (class_name, '__init__') and cname is None:
            cname = '%s__init__%s' % (Naming.func_prefix, class_name)
            name = EncodedString('<init>')
            type.return_type = PyrexTypes.CVoidType()
            type.original_args = type.args

            def maybe_ref(arg):
                if False:
                    for i in range(10):
                        print('nop')
                if arg.type.is_cpp_class and (not arg.type.is_reference):
                    return PyrexTypes.CFuncTypeArg(arg.name, PyrexTypes.c_ref_type(arg.type), arg.pos)
                else:
                    return arg
            type.args = [maybe_ref(arg) for arg in type.args]
        elif name == '__dealloc__' and cname is None:
            cname = '%s__dealloc__%s' % (Naming.func_prefix, class_name)
            name = EncodedString('<del>')
            type.return_type = PyrexTypes.CVoidType()
        if name in ('<init>', '<del>') and type.nogil:
            for base in self.type.base_classes:
                base_entry = base.scope.lookup(name)
                if base_entry and (not base_entry.type.nogil):
                    error(pos, 'Constructor cannot be called without GIL unless all base constructors can also be called without GIL')
                    error(base_entry.pos, 'Base constructor defined here.')
        prev_entry = self.lookup_here(name)
        entry = self.declare_var(name, type, pos, defining=defining, cname=cname, visibility=visibility)
        if prev_entry and (not defining):
            entry.overloaded_alternatives = prev_entry.all_alternatives()
        entry.utility_code = utility_code
        type.entry = entry
        return entry

    def declare_inherited_cpp_attributes(self, base_class):
        if False:
            i = 10
            return i + 15
        base_scope = base_class.scope
        template_type = base_class
        while getattr(template_type, 'template_type', None):
            template_type = template_type.template_type
        if getattr(template_type, 'templates', None):
            base_templates = [T.name for T in template_type.templates]
        else:
            base_templates = ()
        for base_entry in base_scope.inherited_var_entries + base_scope.var_entries:
            if base_entry.name in ('<init>', '<del>'):
                continue
            if base_entry.name in self.entries:
                base_entry.name
            entry = self.declare(base_entry.name, base_entry.cname, base_entry.type, None, 'extern')
            entry.is_variable = 1
            entry.is_inherited = 1
            self.inherited_var_entries.append(entry)
        for base_entry in base_scope.cfunc_entries:
            entry = self.declare_cfunction(base_entry.name, base_entry.type, base_entry.pos, base_entry.cname, base_entry.visibility, api=0, modifiers=base_entry.func_modifiers, utility_code=base_entry.utility_code)
            entry.is_inherited = 1
        for base_entry in base_scope.type_entries:
            if base_entry.name not in base_templates:
                entry = self.declare_type(base_entry.name, base_entry.type, base_entry.pos, base_entry.cname, base_entry.visibility, defining=False)
                entry.is_inherited = 1

    def specialize(self, values, type_entry):
        if False:
            i = 10
            return i + 15
        scope = CppClassScope(self.name, self.outer_scope)
        scope.type = type_entry
        for entry in self.entries.values():
            if entry.is_type:
                scope.declare_type(entry.name, entry.type.specialize(values), entry.pos, entry.cname, template=1)
            elif entry.type.is_cfunction:
                for e in entry.all_alternatives():
                    scope.declare_cfunction(e.name, e.type.specialize(values), e.pos, e.cname, utility_code=e.utility_code)
            else:
                scope.declare_var(entry.name, entry.type.specialize(values), entry.pos, entry.cname, entry.visibility)
        return scope

class CppScopedEnumScope(Scope):

    def __init__(self, name, outer_scope):
        if False:
            print('Hello World!')
        Scope.__init__(self, name, outer_scope, None)

    def declare_var(self, name, type, pos, cname=None, visibility='extern', pytyping_modifiers=None):
        if False:
            for i in range(10):
                print('nop')
        if not cname:
            cname = name
        self._reject_pytyping_modifiers(pos, pytyping_modifiers)
        entry = self.declare(name, cname, type, pos, visibility)
        entry.is_variable = True
        return entry

class PropertyScope(Scope):
    is_property_scope = 1

    def __init__(self, name, class_scope):
        if False:
            return 10
        outer_scope = class_scope.global_scope() if class_scope.outer_scope else None
        Scope.__init__(self, name, outer_scope, parent_scope=class_scope)
        self.parent_type = class_scope.parent_type
        self.directives = class_scope.directives

    def declare_cfunction(self, name, type, pos, *args, **kwargs):
        if False:
            return 10
        'Declare a C property function.\n        '
        if type.return_type.is_void:
            error(pos, "C property method cannot return 'void'")
        if type.args and type.args[0].type is py_object_type:
            type.args[0].type = self.parent_scope.parent_type
        elif len(type.args) != 1:
            error(pos, 'C property method must have a single (self) argument')
        elif not (type.args[0].type.is_pyobject or type.args[0].type is self.parent_scope.parent_type):
            error(pos, 'C property method must have a single (object) argument')
        entry = Scope.declare_cfunction(self, name, type, pos, *args, **kwargs)
        entry.is_cproperty = True
        return entry

    def declare_pyfunction(self, name, pos, allow_redefine=False):
        if False:
            for i in range(10):
                print('nop')
        signature = get_property_accessor_signature(name)
        if signature:
            entry = self.declare(name, name, py_object_type, pos, 'private')
            entry.is_special = 1
            entry.signature = signature
            return entry
        else:
            error(pos, 'Only __get__, __set__ and __del__ methods allowed in a property declaration')
            return None

class CConstOrVolatileScope(Scope):

    def __init__(self, base_type_scope, is_const=0, is_volatile=0):
        if False:
            return 10
        Scope.__init__(self, 'cv_' + base_type_scope.name, base_type_scope.outer_scope, base_type_scope.parent_scope)
        self.base_type_scope = base_type_scope
        self.is_const = is_const
        self.is_volatile = is_volatile

    def lookup_here(self, name):
        if False:
            while True:
                i = 10
        entry = self.base_type_scope.lookup_here(name)
        if entry is not None:
            entry = copy.copy(entry)
            entry.type = PyrexTypes.c_const_or_volatile_type(entry.type, self.is_const, self.is_volatile)
            return entry

class TemplateScope(Scope):

    def __init__(self, name, outer_scope):
        if False:
            return 10
        Scope.__init__(self, name, outer_scope, None)
        self.directives = outer_scope.directives