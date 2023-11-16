from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
invisible = ['__cinit__', '__dealloc__', '__richcmp__', '__nonzero__', '__bool__']
richcmp_special_methods = ['__eq__', '__ne__', '__lt__', '__gt__', '__le__', '__ge__']

class Signature(object):
    format_map = {'O': PyrexTypes.py_object_type, 'v': PyrexTypes.c_void_type, 'p': PyrexTypes.c_void_ptr_type, 'P': PyrexTypes.c_void_ptr_ptr_type, 'i': PyrexTypes.c_int_type, 'b': PyrexTypes.c_bint_type, 'I': PyrexTypes.c_int_ptr_type, 'l': PyrexTypes.c_long_type, 'f': PyrexTypes.c_float_type, 'd': PyrexTypes.c_double_type, 'h': PyrexTypes.c_py_hash_t_type, 'z': PyrexTypes.c_py_ssize_t_type, 'Z': PyrexTypes.c_py_ssize_t_ptr_type, 's': PyrexTypes.c_char_ptr_type, 'S': PyrexTypes.c_char_ptr_ptr_type, 'r': PyrexTypes.c_returncode_type, 'B': PyrexTypes.c_py_buffer_ptr_type, '?': PyrexTypes.py_object_type}
    type_to_format_map = dict(((type_, format_) for (format_, type_) in format_map.items()))
    error_value_map = {'O': 'NULL', 'T': 'NULL', 'i': '-1', 'b': '-1', 'l': '-1', 'r': '-1', 'h': '-1', 'z': '-1'}
    use_fastcall = False

    def __init__(self, arg_format, ret_format, nogil=False):
        if False:
            while True:
                i = 10
        self.has_dummy_arg = False
        self.has_generic_args = False
        self.optional_object_arg_count = 0
        if arg_format[:1] == '-':
            self.has_dummy_arg = True
            arg_format = arg_format[1:]
        if arg_format[-1:] == '*':
            self.has_generic_args = True
            arg_format = arg_format[:-1]
        if arg_format[-1:] == '?':
            self.optional_object_arg_count += 1
        self.fixed_arg_format = arg_format
        self.ret_format = ret_format
        self.error_value = self.error_value_map.get(ret_format, None)
        self.exception_check = ret_format != 'r' and self.error_value is not None
        self.is_staticmethod = False
        self.nogil = nogil

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<Signature[%s(%s%s)]>' % (self.ret_format, ', '.join(self.fixed_arg_format), '*' if self.has_generic_args else '')

    def min_num_fixed_args(self):
        if False:
            i = 10
            return i + 15
        return self.max_num_fixed_args() - self.optional_object_arg_count

    def max_num_fixed_args(self):
        if False:
            return 10
        return len(self.fixed_arg_format)

    def is_self_arg(self, i):
        if False:
            return 10
        return self.fixed_arg_format[i] == 'T'

    def returns_self_type(self):
        if False:
            i = 10
            return i + 15
        return self.ret_format == 'T'

    def fixed_arg_type(self, i):
        if False:
            for i in range(10):
                print('nop')
        return self.format_map[self.fixed_arg_format[i]]

    def return_type(self):
        if False:
            return 10
        return self.format_map[self.ret_format]

    def format_from_type(self, arg_type):
        if False:
            return 10
        if arg_type.is_pyobject:
            arg_type = PyrexTypes.py_object_type
        return self.type_to_format_map[arg_type]

    def exception_value(self):
        if False:
            for i in range(10):
                print('nop')
        return self.error_value_map.get(self.ret_format)

    def function_type(self, self_arg_override=None):
        if False:
            for i in range(10):
                print('nop')
        args = []
        for i in range(self.max_num_fixed_args()):
            if self_arg_override is not None and self.is_self_arg(i):
                assert isinstance(self_arg_override, PyrexTypes.CFuncTypeArg)
                args.append(self_arg_override)
            else:
                arg_type = self.fixed_arg_type(i)
                args.append(PyrexTypes.CFuncTypeArg('', arg_type, None))
        if self_arg_override is not None and self.returns_self_type():
            ret_type = self_arg_override.type
        else:
            ret_type = self.return_type()
        exc_value = self.exception_value()
        return PyrexTypes.CFuncType(ret_type, args, exception_value=exc_value, exception_check=self.exception_check, nogil=self.nogil)

    def method_flags(self):
        if False:
            return 10
        if self.ret_format == 'O':
            full_args = self.fixed_arg_format
            if self.has_dummy_arg:
                full_args = 'O' + full_args
            if full_args in ['O', 'T']:
                if not self.has_generic_args:
                    return [method_noargs]
                elif self.use_fastcall:
                    return [method_fastcall, method_keywords]
                else:
                    return [method_varargs, method_keywords]
            elif full_args in ['OO', 'TO'] and (not self.has_generic_args):
                return [method_onearg]
            if self.is_staticmethod:
                if self.use_fastcall:
                    return [method_fastcall, method_keywords]
                else:
                    return [method_varargs, method_keywords]
        return None

    def method_function_type(self):
        if False:
            for i in range(10):
                print('nop')
        mflags = self.method_flags()
        kw = 'WithKeywords' if method_keywords in mflags else ''
        for m in mflags:
            if m == method_noargs or m == method_onearg:
                return 'PyCFunction'
            if m == method_varargs:
                return 'PyCFunction' + kw
            if m == method_fastcall:
                return '__Pyx_PyCFunction_FastCall' + kw
        return None

    def with_fastcall(self):
        if False:
            for i in range(10):
                print('nop')
        sig = copy.copy(self)
        sig.use_fastcall = True
        return sig

    @property
    def fastvar(self):
        if False:
            return 10
        if self.use_fastcall:
            return 'FASTCALL'
        else:
            return 'VARARGS'

class SlotDescriptor(object):

    def __init__(self, slot_name, dynamic=False, inherited=False, py3=True, py2=True, ifdef=None, is_binop=False, used_ifdef=None):
        if False:
            for i in range(10):
                print('nop')
        self.slot_name = slot_name
        self.is_initialised_dynamically = dynamic
        self.is_inherited = inherited
        self.ifdef = ifdef
        self.used_ifdef = used_ifdef
        self.py3 = py3
        self.py2 = py2
        self.is_binop = is_binop

    def slot_code(self, scope):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def spec_value(self, scope):
        if False:
            while True:
                i = 10
        return self.slot_code(scope)

    def preprocessor_guard_code(self):
        if False:
            print('Hello World!')
        ifdef = self.ifdef
        py2 = self.py2
        py3 = self.py3
        guard = None
        if ifdef:
            guard = '#if %s' % ifdef
        elif not py3 or py3 == '<RESERVED>':
            guard = '#if PY_MAJOR_VERSION < 3'
        elif not py2:
            guard = '#if PY_MAJOR_VERSION >= 3'
        return guard

    def generate_spec(self, scope, code):
        if False:
            i = 10
            return i + 15
        if self.is_initialised_dynamically:
            return
        value = self.spec_value(scope)
        if value == '0':
            return
        preprocessor_guard = self.preprocessor_guard_code()
        if not preprocessor_guard:
            if self.py3 and self.slot_name.startswith('bf_'):
                preprocessor_guard = '#if defined(Py_%s)' % self.slot_name
        if preprocessor_guard:
            code.putln(preprocessor_guard)
        code.putln('{Py_%s, (void *)%s},' % (self.slot_name, value))
        if preprocessor_guard:
            code.putln('#endif')

    def generate(self, scope, code):
        if False:
            while True:
                i = 10
        preprocessor_guard = self.preprocessor_guard_code()
        if preprocessor_guard:
            code.putln(preprocessor_guard)
        end_pypy_guard = False
        if self.is_initialised_dynamically:
            value = '0'
        else:
            value = self.slot_code(scope)
            if value == '0' and self.is_inherited:
                inherited_value = value
                current_scope = scope
                while inherited_value == '0' and current_scope.parent_type and current_scope.parent_type.base_type and current_scope.parent_type.base_type.scope:
                    current_scope = current_scope.parent_type.base_type.scope
                    inherited_value = self.slot_code(current_scope)
                if inherited_value != '0':
                    is_buffer_slot = int(self.slot_name in ('bf_getbuffer', 'bf_releasebuffer'))
                    code.putln('#if CYTHON_COMPILING_IN_PYPY || %d' % is_buffer_slot)
                    code.putln('%s, /*%s*/' % (inherited_value, self.slot_name))
                    code.putln('#else')
                    end_pypy_guard = True
        if self.used_ifdef:
            code.putln('#if %s' % self.used_ifdef)
        code.putln('%s, /*%s*/' % (value, self.slot_name))
        if self.used_ifdef:
            code.putln('#else')
            code.putln('NULL, /*%s*/' % self.slot_name)
            code.putln('#endif')
        if end_pypy_guard:
            code.putln('#endif')
        if self.py3 == '<RESERVED>':
            code.putln('#else')
            code.putln('0, /*reserved*/')
        if preprocessor_guard:
            code.putln('#endif')

    def generate_dynamic_init_code(self, scope, code):
        if False:
            while True:
                i = 10
        if self.is_initialised_dynamically:
            self.generate_set_slot_code(self.slot_code(scope), scope, code)

    def generate_set_slot_code(self, value, scope, code):
        if False:
            for i in range(10):
                print('nop')
        if value == '0':
            return
        if scope.parent_type.typeptr_cname:
            target = '%s->%s' % (scope.parent_type.typeptr_cname, self.slot_name)
        else:
            assert scope.parent_type.typeobj_cname
            target = '%s.%s' % (scope.parent_type.typeobj_cname, self.slot_name)
        code.putln('%s = %s;' % (target, value))

class FixedSlot(SlotDescriptor):

    def __init__(self, slot_name, value, py3=True, py2=True, ifdef=None):
        if False:
            print('Hello World!')
        SlotDescriptor.__init__(self, slot_name, py3=py3, py2=py2, ifdef=ifdef)
        self.value = value

    def slot_code(self, scope):
        if False:
            for i in range(10):
                print('nop')
        return self.value

class EmptySlot(FixedSlot):

    def __init__(self, slot_name, py3=True, py2=True, ifdef=None):
        if False:
            print('Hello World!')
        FixedSlot.__init__(self, slot_name, '0', py3=py3, py2=py2, ifdef=ifdef)

class MethodSlot(SlotDescriptor):

    def __init__(self, signature, slot_name, method_name, method_name_to_slot, fallback=None, py3=True, py2=True, ifdef=None, inherited=True):
        if False:
            for i in range(10):
                print('nop')
        SlotDescriptor.__init__(self, slot_name, py3=py3, py2=py2, ifdef=ifdef, inherited=inherited)
        self.signature = signature
        self.slot_name = slot_name
        self.method_name = method_name
        self.alternatives = []
        method_name_to_slot[method_name] = self
        if fallback:
            self.alternatives.append(fallback)
        for alt in (self.py2, self.py3):
            if isinstance(alt, (tuple, list)):
                (slot_name, method_name) = alt
                self.alternatives.append(method_name)
                method_name_to_slot[method_name] = self

    def slot_code(self, scope):
        if False:
            return 10
        entry = scope.lookup_here(self.method_name)
        if entry and entry.is_special and entry.func_cname:
            return entry.func_cname
        for method_name in self.alternatives:
            entry = scope.lookup_here(method_name)
            if entry and entry.is_special and entry.func_cname:
                return entry.func_cname
        return '0'

class InternalMethodSlot(SlotDescriptor):

    def __init__(self, slot_name, **kargs):
        if False:
            while True:
                i = 10
        SlotDescriptor.__init__(self, slot_name, **kargs)

    def slot_code(self, scope):
        if False:
            i = 10
            return i + 15
        return scope.mangle_internal(self.slot_name)

class GCDependentSlot(InternalMethodSlot):

    def __init__(self, slot_name, **kargs):
        if False:
            while True:
                i = 10
        InternalMethodSlot.__init__(self, slot_name, **kargs)

    def slot_code(self, scope):
        if False:
            i = 10
            return i + 15
        if not scope.needs_gc():
            return '0'
        if not scope.has_cyclic_pyobject_attrs:
            parent_type_scope = scope.parent_type.base_type.scope
            if scope.parent_scope is parent_type_scope.parent_scope:
                entry = scope.parent_scope.lookup_here(scope.parent_type.base_type.name)
                if entry.visibility != 'extern':
                    return self.slot_code(parent_type_scope)
        return InternalMethodSlot.slot_code(self, scope)

class GCClearReferencesSlot(GCDependentSlot):

    def slot_code(self, scope):
        if False:
            return 10
        if scope.needs_tp_clear():
            return GCDependentSlot.slot_code(self, scope)
        return '0'

class ConstructorSlot(InternalMethodSlot):

    def __init__(self, slot_name, method=None, **kargs):
        if False:
            print('Hello World!')
        InternalMethodSlot.__init__(self, slot_name, **kargs)
        self.method = method

    def _needs_own(self, scope):
        if False:
            for i in range(10):
                print('nop')
        if scope.parent_type.base_type and (not scope.has_pyobject_attrs) and (not scope.has_memoryview_attrs) and (not scope.has_cpp_constructable_attrs) and (not (self.slot_name == 'tp_new' and scope.parent_type.vtabslot_cname)):
            entry = scope.lookup_here(self.method) if self.method else None
            if not (entry and entry.is_special):
                return False
        return True

    def _parent_slot_function(self, scope):
        if False:
            for i in range(10):
                print('nop')
        parent_type_scope = scope.parent_type.base_type.scope
        if scope.parent_scope is parent_type_scope.parent_scope:
            entry = scope.parent_scope.lookup_here(scope.parent_type.base_type.name)
            if entry.visibility != 'extern':
                return self.slot_code(parent_type_scope)
        return None

    def slot_code(self, scope):
        if False:
            return 10
        if not self._needs_own(scope):
            slot_code = self._parent_slot_function(scope)
            return slot_code or '0'
        return InternalMethodSlot.slot_code(self, scope)

    def spec_value(self, scope):
        if False:
            print('Hello World!')
        slot_function = self.slot_code(scope)
        if self.slot_name == 'tp_dealloc' and slot_function != scope.mangle_internal('tp_dealloc'):
            return '0'
        return slot_function

    def generate_dynamic_init_code(self, scope, code):
        if False:
            print('Hello World!')
        if self.slot_code(scope) != '0':
            return
        base_type = scope.parent_type.base_type
        if base_type.typeptr_cname:
            src = '%s->%s' % (base_type.typeptr_cname, self.slot_name)
        elif base_type.is_extension_type and base_type.typeobj_cname:
            src = '%s.%s' % (base_type.typeobj_cname, self.slot_name)
        else:
            return
        self.generate_set_slot_code(src, scope, code)

class SyntheticSlot(InternalMethodSlot):

    def __init__(self, slot_name, user_methods, default_value, **kargs):
        if False:
            return 10
        InternalMethodSlot.__init__(self, slot_name, **kargs)
        self.user_methods = user_methods
        self.default_value = default_value

    def slot_code(self, scope):
        if False:
            for i in range(10):
                print('nop')
        if scope.defines_any_special(self.user_methods):
            return InternalMethodSlot.slot_code(self, scope)
        else:
            return self.default_value

    def spec_value(self, scope):
        if False:
            while True:
                i = 10
        return self.slot_code(scope)

class BinopSlot(SyntheticSlot):

    def __init__(self, signature, slot_name, left_method, method_name_to_slot, **kargs):
        if False:
            while True:
                i = 10
        assert left_method.startswith('__')
        right_method = '__r' + left_method[2:]
        SyntheticSlot.__init__(self, slot_name, [left_method, right_method], '0', is_binop=True, **kargs)
        self.left_slot = MethodSlot(signature, '', left_method, method_name_to_slot, **kargs)
        self.right_slot = MethodSlot(signature, '', right_method, method_name_to_slot, **kargs)

class RichcmpSlot(MethodSlot):

    def slot_code(self, scope):
        if False:
            while True:
                i = 10
        entry = scope.lookup_here(self.method_name)
        if entry and entry.is_special and entry.func_cname:
            return entry.func_cname
        elif scope.defines_any_special(richcmp_special_methods):
            return scope.mangle_internal(self.slot_name)
        else:
            return '0'

class TypeFlagsSlot(SlotDescriptor):

    def slot_code(self, scope):
        if False:
            for i in range(10):
                print('nop')
        value = 'Py_TPFLAGS_DEFAULT'
        if scope.directives['type_version_tag']:
            value += '|Py_TPFLAGS_HAVE_VERSION_TAG'
        else:
            value = '(%s&~Py_TPFLAGS_HAVE_VERSION_TAG)' % value
        value += '|Py_TPFLAGS_CHECKTYPES|Py_TPFLAGS_HAVE_NEWBUFFER'
        if not scope.parent_type.is_final_type:
            value += '|Py_TPFLAGS_BASETYPE'
        if scope.needs_gc():
            value += '|Py_TPFLAGS_HAVE_GC'
        if scope.may_have_finalize():
            value += '|Py_TPFLAGS_HAVE_FINALIZE'
        if scope.parent_type.has_sequence_flag:
            value += '|Py_TPFLAGS_SEQUENCE'
        return value

    def generate_spec(self, scope, code):
        if False:
            while True:
                i = 10
        return

class DocStringSlot(SlotDescriptor):

    def slot_code(self, scope):
        if False:
            for i in range(10):
                print('nop')
        doc = scope.doc
        if doc is None:
            return '0'
        if doc.is_unicode:
            doc = doc.as_utf8_string()
        return 'PyDoc_STR(%s)' % doc.as_c_string_literal()

class SuiteSlot(SlotDescriptor):

    def __init__(self, sub_slots, slot_type, slot_name, substructures, ifdef=None):
        if False:
            return 10
        SlotDescriptor.__init__(self, slot_name, ifdef=ifdef)
        self.sub_slots = sub_slots
        self.slot_type = slot_type
        substructures.append(self)

    def is_empty(self, scope):
        if False:
            for i in range(10):
                print('nop')
        for slot in self.sub_slots:
            if slot.slot_code(scope) != '0':
                return False
        return True

    def substructure_cname(self, scope):
        if False:
            for i in range(10):
                print('nop')
        return '%s%s_%s' % (Naming.pyrex_prefix, self.slot_name, scope.class_name)

    def slot_code(self, scope):
        if False:
            print('Hello World!')
        if not self.is_empty(scope):
            return '&%s' % self.substructure_cname(scope)
        return '0'

    def generate_substructure(self, scope, code):
        if False:
            while True:
                i = 10
        if not self.is_empty(scope):
            code.putln('')
            if self.ifdef:
                code.putln('#if %s' % self.ifdef)
            code.putln('static %s %s = {' % (self.slot_type, self.substructure_cname(scope)))
            for slot in self.sub_slots:
                slot.generate(scope, code)
            code.putln('};')
            if self.ifdef:
                code.putln('#endif')

    def generate_spec(self, scope, code):
        if False:
            print('Hello World!')
        for slot in self.sub_slots:
            slot.generate_spec(scope, code)

class MethodTableSlot(SlotDescriptor):

    def slot_code(self, scope):
        if False:
            print('Hello World!')
        if scope.pyfunc_entries:
            return scope.method_table_cname
        else:
            return '0'

class MemberTableSlot(SlotDescriptor):

    def slot_code(self, scope):
        if False:
            for i in range(10):
                print('nop')
        return '0'

    def get_member_specs(self, scope):
        if False:
            for i in range(10):
                print('nop')
        return [get_slot_by_name('tp_dictoffset', scope.directives).members_slot_value(scope)]

    def is_empty(self, scope):
        if False:
            for i in range(10):
                print('nop')
        for member_entry in self.get_member_specs(scope):
            if member_entry:
                return False
        return True

    def substructure_cname(self, scope):
        if False:
            for i in range(10):
                print('nop')
        return '%s%s_%s' % (Naming.pyrex_prefix, self.slot_name, scope.class_name)

    def generate_substructure_spec(self, scope, code):
        if False:
            for i in range(10):
                print('nop')
        if self.is_empty(scope):
            return
        from .Code import UtilityCode
        code.globalstate.use_utility_code(UtilityCode.load_cached('IncludeStructmemberH', 'ModuleSetupCode.c'))
        code.putln('static struct PyMemberDef %s[] = {' % self.substructure_cname(scope))
        for member_entry in self.get_member_specs(scope):
            if member_entry:
                code.putln(member_entry)
        code.putln('{NULL, 0, 0, 0, NULL}')
        code.putln('};')

    def spec_value(self, scope):
        if False:
            print('Hello World!')
        if self.is_empty(scope):
            return '0'
        return self.substructure_cname(scope)

class GetSetSlot(SlotDescriptor):

    def slot_code(self, scope):
        if False:
            print('Hello World!')
        if scope.property_entries:
            return scope.getset_table_cname
        else:
            return '0'

class BaseClassSlot(SlotDescriptor):

    def __init__(self, name):
        if False:
            while True:
                i = 10
        SlotDescriptor.__init__(self, name, dynamic=True)

    def generate_dynamic_init_code(self, scope, code):
        if False:
            while True:
                i = 10
        base_type = scope.parent_type.base_type
        if base_type:
            code.putln('%s->%s = %s;' % (scope.parent_type.typeptr_cname, self.slot_name, base_type.typeptr_cname))

class DictOffsetSlot(SlotDescriptor):

    def slot_code(self, scope):
        if False:
            return 10
        dict_entry = scope.lookup_here('__dict__') if not scope.is_closure_class_scope else None
        if dict_entry and dict_entry.is_variable:
            from . import Builtin
            if dict_entry.type is not Builtin.dict_type:
                error(dict_entry.pos, "__dict__ slot must be of type 'dict'")
                return '0'
            type = scope.parent_type
            if type.typedef_flag:
                objstruct = type.objstruct_cname
            else:
                objstruct = 'struct %s' % type.objstruct_cname
            return 'offsetof(%s, %s)' % (objstruct, dict_entry.cname)
        else:
            return '0'

    def members_slot_value(self, scope):
        if False:
            i = 10
            return i + 15
        dict_offset = self.slot_code(scope)
        if dict_offset == '0':
            return None
        return '{"__dictoffset__", T_PYSSIZET, %s, READONLY, NULL},' % dict_offset

def get_property_accessor_signature(name):
    if False:
        return 10
    return property_accessor_signatures.get(name)

def get_base_slot_function(scope, slot):
    if False:
        while True:
            i = 10
    base_type = scope.parent_type.base_type
    if base_type and scope.parent_scope is base_type.scope.parent_scope:
        parent_slot = slot.slot_code(base_type.scope)
        if parent_slot != '0':
            entry = scope.parent_scope.lookup_here(scope.parent_type.base_type.name)
            if entry.visibility != 'extern':
                return parent_slot
    return None

def get_slot_function(scope, slot):
    if False:
        return 10
    slot_code = slot.slot_code(scope)
    if slot_code != '0':
        entry = scope.parent_scope.lookup_here(scope.parent_type.name)
        if entry.visibility != 'extern':
            return slot_code
    return None

def get_slot_by_name(slot_name, compiler_directives):
    if False:
        i = 10
        return i + 15
    for slot in get_slot_table(compiler_directives).slot_table:
        if slot.slot_name == slot_name:
            return slot
    assert False, 'Slot not found: %s' % slot_name

def get_slot_code_by_name(scope, slot_name):
    if False:
        print('Hello World!')
    slot = get_slot_by_name(slot_name, scope.directives)
    return slot.slot_code(scope)

def is_reverse_number_slot(name):
    if False:
        return 10
    "\n    Tries to identify __radd__ and friends (so the METH_COEXIST flag can be applied).\n\n    There's no great consequence if it inadvertently identifies a few other methods\n    so just use a simple rule rather than an exact list.\n    "
    if name.startswith('__r') and name.endswith('__'):
        forward_name = name.replace('r', '', 1)
        for meth in get_slot_table(None).PyNumberMethods:
            if hasattr(meth, 'right_slot'):
                return True
    return False
pyfunction_signature = Signature('-*', 'O')
pymethod_signature = Signature('T*', 'O')
pyfunction_noargs = Signature('-', 'O')
pyfunction_onearg = Signature('-O', 'O')
unaryfunc = Signature('T', 'O')
binaryfunc = Signature('OO', 'O')
ibinaryfunc = Signature('TO', 'O')
powternaryfunc = Signature('OO?', 'O')
ipowternaryfunc = Signature('TO?', 'O')
callfunc = Signature('T*', 'O')
inquiry = Signature('T', 'i')
lenfunc = Signature('T', 'z')
intargfunc = Signature('Ti', 'O')
ssizeargfunc = Signature('Tz', 'O')
intintargfunc = Signature('Tii', 'O')
ssizessizeargfunc = Signature('Tzz', 'O')
intobjargproc = Signature('TiO', 'r')
ssizeobjargproc = Signature('TzO', 'r')
intintobjargproc = Signature('TiiO', 'r')
ssizessizeobjargproc = Signature('TzzO', 'r')
intintargproc = Signature('Tii', 'r')
ssizessizeargproc = Signature('Tzz', 'r')
objargfunc = Signature('TO', 'O')
objobjargproc = Signature('TOO', 'r')
readbufferproc = Signature('TzP', 'z')
writebufferproc = Signature('TzP', 'z')
segcountproc = Signature('TZ', 'z')
charbufferproc = Signature('TzS', 'z')
objargproc = Signature('TO', 'r')
destructor = Signature('T', 'v')
getattrofunc = Signature('TO', 'O')
setattrofunc = Signature('TOO', 'r')
delattrofunc = Signature('TO', 'r')
cmpfunc = Signature('TO', 'i')
reprfunc = Signature('T', 'O')
hashfunc = Signature('T', 'h')
richcmpfunc = Signature('TOi', 'O')
getiterfunc = Signature('T', 'O')
iternextfunc = Signature('T', 'O')
descrgetfunc = Signature('TOO', 'O')
descrsetfunc = Signature('TOO', 'r')
descrdelfunc = Signature('TO', 'r')
initproc = Signature('T*', 'r')
getbufferproc = Signature('TBi', 'r')
releasebufferproc = Signature('TB', 'v')
property_accessor_signatures = {'__get__': Signature('T', 'O'), '__set__': Signature('TO', 'r'), '__del__': Signature('T', 'r')}
PyNumberMethods_Py2only_GUARD = 'PY_MAJOR_VERSION < 3 || (CYTHON_COMPILING_IN_PYPY && PY_VERSION_HEX < 0x03050000)'

class SlotTable(object):

    def __init__(self, old_binops):
        if False:
            return 10
        method_name_to_slot = {}
        self._get_slot_by_method_name = method_name_to_slot.get
        self.substructures = []
        bf = binaryfunc if old_binops else ibinaryfunc
        ptf = powternaryfunc if old_binops else ipowternaryfunc
        self.PyNumberMethods = (BinopSlot(bf, 'nb_add', '__add__', method_name_to_slot), BinopSlot(bf, 'nb_subtract', '__sub__', method_name_to_slot), BinopSlot(bf, 'nb_multiply', '__mul__', method_name_to_slot), BinopSlot(bf, 'nb_divide', '__div__', method_name_to_slot, ifdef=PyNumberMethods_Py2only_GUARD), BinopSlot(bf, 'nb_remainder', '__mod__', method_name_to_slot), BinopSlot(bf, 'nb_divmod', '__divmod__', method_name_to_slot), BinopSlot(ptf, 'nb_power', '__pow__', method_name_to_slot), MethodSlot(unaryfunc, 'nb_negative', '__neg__', method_name_to_slot), MethodSlot(unaryfunc, 'nb_positive', '__pos__', method_name_to_slot), MethodSlot(unaryfunc, 'nb_absolute', '__abs__', method_name_to_slot), MethodSlot(inquiry, 'nb_bool', '__bool__', method_name_to_slot, py2=('nb_nonzero', '__nonzero__')), MethodSlot(unaryfunc, 'nb_invert', '__invert__', method_name_to_slot), BinopSlot(bf, 'nb_lshift', '__lshift__', method_name_to_slot), BinopSlot(bf, 'nb_rshift', '__rshift__', method_name_to_slot), BinopSlot(bf, 'nb_and', '__and__', method_name_to_slot), BinopSlot(bf, 'nb_xor', '__xor__', method_name_to_slot), BinopSlot(bf, 'nb_or', '__or__', method_name_to_slot), EmptySlot('nb_coerce', ifdef=PyNumberMethods_Py2only_GUARD), MethodSlot(unaryfunc, 'nb_int', '__int__', method_name_to_slot, fallback='__long__'), MethodSlot(unaryfunc, 'nb_long', '__long__', method_name_to_slot, fallback='__int__', py3='<RESERVED>'), MethodSlot(unaryfunc, 'nb_float', '__float__', method_name_to_slot), MethodSlot(unaryfunc, 'nb_oct', '__oct__', method_name_to_slot, ifdef=PyNumberMethods_Py2only_GUARD), MethodSlot(unaryfunc, 'nb_hex', '__hex__', method_name_to_slot, ifdef=PyNumberMethods_Py2only_GUARD), MethodSlot(ibinaryfunc, 'nb_inplace_add', '__iadd__', method_name_to_slot), MethodSlot(ibinaryfunc, 'nb_inplace_subtract', '__isub__', method_name_to_slot), MethodSlot(ibinaryfunc, 'nb_inplace_multiply', '__imul__', method_name_to_slot), MethodSlot(ibinaryfunc, 'nb_inplace_divide', '__idiv__', method_name_to_slot, ifdef=PyNumberMethods_Py2only_GUARD), MethodSlot(ibinaryfunc, 'nb_inplace_remainder', '__imod__', method_name_to_slot), MethodSlot(ptf, 'nb_inplace_power', '__ipow__', method_name_to_slot), MethodSlot(ibinaryfunc, 'nb_inplace_lshift', '__ilshift__', method_name_to_slot), MethodSlot(ibinaryfunc, 'nb_inplace_rshift', '__irshift__', method_name_to_slot), MethodSlot(ibinaryfunc, 'nb_inplace_and', '__iand__', method_name_to_slot), MethodSlot(ibinaryfunc, 'nb_inplace_xor', '__ixor__', method_name_to_slot), MethodSlot(ibinaryfunc, 'nb_inplace_or', '__ior__', method_name_to_slot), BinopSlot(bf, 'nb_floor_divide', '__floordiv__', method_name_to_slot), BinopSlot(bf, 'nb_true_divide', '__truediv__', method_name_to_slot), MethodSlot(ibinaryfunc, 'nb_inplace_floor_divide', '__ifloordiv__', method_name_to_slot), MethodSlot(ibinaryfunc, 'nb_inplace_true_divide', '__itruediv__', method_name_to_slot), MethodSlot(unaryfunc, 'nb_index', '__index__', method_name_to_slot), BinopSlot(bf, 'nb_matrix_multiply', '__matmul__', method_name_to_slot, ifdef='PY_VERSION_HEX >= 0x03050000'), MethodSlot(ibinaryfunc, 'nb_inplace_matrix_multiply', '__imatmul__', method_name_to_slot, ifdef='PY_VERSION_HEX >= 0x03050000'))
        self.PySequenceMethods = (MethodSlot(lenfunc, 'sq_length', '__len__', method_name_to_slot), EmptySlot('sq_concat'), EmptySlot('sq_repeat'), SyntheticSlot('sq_item', ['__getitem__'], '0'), MethodSlot(ssizessizeargfunc, 'sq_slice', '__getslice__', method_name_to_slot), EmptySlot('sq_ass_item'), SyntheticSlot('sq_ass_slice', ['__setslice__', '__delslice__'], '0'), MethodSlot(cmpfunc, 'sq_contains', '__contains__', method_name_to_slot), EmptySlot('sq_inplace_concat'), EmptySlot('sq_inplace_repeat'))
        self.PyMappingMethods = (MethodSlot(lenfunc, 'mp_length', '__len__', method_name_to_slot), MethodSlot(objargfunc, 'mp_subscript', '__getitem__', method_name_to_slot), SyntheticSlot('mp_ass_subscript', ['__setitem__', '__delitem__'], '0'))
        self.PyBufferProcs = (MethodSlot(readbufferproc, 'bf_getreadbuffer', '__getreadbuffer__', method_name_to_slot, py3=False), MethodSlot(writebufferproc, 'bf_getwritebuffer', '__getwritebuffer__', method_name_to_slot, py3=False), MethodSlot(segcountproc, 'bf_getsegcount', '__getsegcount__', method_name_to_slot, py3=False), MethodSlot(charbufferproc, 'bf_getcharbuffer', '__getcharbuffer__', method_name_to_slot, py3=False), MethodSlot(getbufferproc, 'bf_getbuffer', '__getbuffer__', method_name_to_slot), MethodSlot(releasebufferproc, 'bf_releasebuffer', '__releasebuffer__', method_name_to_slot))
        self.PyAsyncMethods = (MethodSlot(unaryfunc, 'am_await', '__await__', method_name_to_slot), MethodSlot(unaryfunc, 'am_aiter', '__aiter__', method_name_to_slot), MethodSlot(unaryfunc, 'am_anext', '__anext__', method_name_to_slot), EmptySlot('am_send', ifdef='PY_VERSION_HEX >= 0x030A00A3'))
        self.slot_table = (ConstructorSlot('tp_dealloc', '__dealloc__'), EmptySlot('tp_print', ifdef='PY_VERSION_HEX < 0x030800b4'), EmptySlot('tp_vectorcall_offset', ifdef='PY_VERSION_HEX >= 0x030800b4'), EmptySlot('tp_getattr'), EmptySlot('tp_setattr'), MethodSlot(cmpfunc, 'tp_compare', '__cmp__', method_name_to_slot, ifdef='PY_MAJOR_VERSION < 3'), SuiteSlot(self.PyAsyncMethods, '__Pyx_PyAsyncMethodsStruct', 'tp_as_async', self.substructures, ifdef='PY_MAJOR_VERSION >= 3'), MethodSlot(reprfunc, 'tp_repr', '__repr__', method_name_to_slot), SuiteSlot(self.PyNumberMethods, 'PyNumberMethods', 'tp_as_number', self.substructures), SuiteSlot(self.PySequenceMethods, 'PySequenceMethods', 'tp_as_sequence', self.substructures), SuiteSlot(self.PyMappingMethods, 'PyMappingMethods', 'tp_as_mapping', self.substructures), MethodSlot(hashfunc, 'tp_hash', '__hash__', method_name_to_slot, inherited=False), MethodSlot(callfunc, 'tp_call', '__call__', method_name_to_slot), MethodSlot(reprfunc, 'tp_str', '__str__', method_name_to_slot), SyntheticSlot('tp_getattro', ['__getattr__', '__getattribute__'], '0'), SyntheticSlot('tp_setattro', ['__setattr__', '__delattr__'], '0'), SuiteSlot(self.PyBufferProcs, 'PyBufferProcs', 'tp_as_buffer', self.substructures), TypeFlagsSlot('tp_flags'), DocStringSlot('tp_doc'), GCDependentSlot('tp_traverse'), GCClearReferencesSlot('tp_clear'), RichcmpSlot(richcmpfunc, 'tp_richcompare', '__richcmp__', method_name_to_slot, inherited=False), EmptySlot('tp_weaklistoffset'), MethodSlot(getiterfunc, 'tp_iter', '__iter__', method_name_to_slot), MethodSlot(iternextfunc, 'tp_iternext', '__next__', method_name_to_slot), MethodTableSlot('tp_methods'), MemberTableSlot('tp_members'), GetSetSlot('tp_getset'), BaseClassSlot('tp_base'), EmptySlot('tp_dict'), SyntheticSlot('tp_descr_get', ['__get__'], '0'), SyntheticSlot('tp_descr_set', ['__set__', '__delete__'], '0'), DictOffsetSlot('tp_dictoffset', ifdef='!CYTHON_USE_TYPE_SPECS'), MethodSlot(initproc, 'tp_init', '__init__', method_name_to_slot), EmptySlot('tp_alloc'), ConstructorSlot('tp_new', '__cinit__'), EmptySlot('tp_free'), EmptySlot('tp_is_gc'), EmptySlot('tp_bases'), EmptySlot('tp_mro'), EmptySlot('tp_cache'), EmptySlot('tp_subclasses'), EmptySlot('tp_weaklist'), EmptySlot('tp_del'), EmptySlot('tp_version_tag'), SyntheticSlot('tp_finalize', ['__del__'], '0', ifdef='PY_VERSION_HEX >= 0x030400a1', used_ifdef='CYTHON_USE_TP_FINALIZE'), EmptySlot('tp_vectorcall', ifdef='PY_VERSION_HEX >= 0x030800b1 && (!CYTHON_COMPILING_IN_PYPY || PYPY_VERSION_NUM >= 0x07030800)'), EmptySlot('tp_print', ifdef='__PYX_NEED_TP_PRINT_SLOT == 1'), EmptySlot('tp_watched', ifdef='PY_VERSION_HEX >= 0x030C0000'), EmptySlot('tp_pypy_flags', ifdef='CYTHON_COMPILING_IN_PYPY && PY_VERSION_HEX >= 0x03090000 && PY_VERSION_HEX < 0x030a0000'))
        MethodSlot(initproc, '', '__cinit__', method_name_to_slot)
        MethodSlot(destructor, '', '__dealloc__', method_name_to_slot)
        MethodSlot(destructor, '', '__del__', method_name_to_slot)
        MethodSlot(objobjargproc, '', '__setitem__', method_name_to_slot)
        MethodSlot(objargproc, '', '__delitem__', method_name_to_slot)
        MethodSlot(ssizessizeobjargproc, '', '__setslice__', method_name_to_slot)
        MethodSlot(ssizessizeargproc, '', '__delslice__', method_name_to_slot)
        MethodSlot(getattrofunc, '', '__getattr__', method_name_to_slot)
        MethodSlot(getattrofunc, '', '__getattribute__', method_name_to_slot)
        MethodSlot(setattrofunc, '', '__setattr__', method_name_to_slot)
        MethodSlot(delattrofunc, '', '__delattr__', method_name_to_slot)
        MethodSlot(descrgetfunc, '', '__get__', method_name_to_slot)
        MethodSlot(descrsetfunc, '', '__set__', method_name_to_slot)
        MethodSlot(descrdelfunc, '', '__delete__', method_name_to_slot)

    def get_special_method_signature(self, name):
        if False:
            return 10
        slot = self._get_slot_by_method_name(name)
        if slot:
            return slot.signature
        elif name in richcmp_special_methods:
            return ibinaryfunc
        else:
            return None

    def get_slot_by_method_name(self, method_name):
        if False:
            print('Hello World!')
        return self._get_slot_by_method_name(method_name)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self.slot_table)
_slot_table_dict = {}

def get_slot_table(compiler_directives):
    if False:
        return 10
    if not compiler_directives:
        from .Options import get_directive_defaults
        compiler_directives = get_directive_defaults()
    old_binops = compiler_directives['c_api_binop_methods']
    key = (old_binops,)
    if key not in _slot_table_dict:
        _slot_table_dict[key] = SlotTable(old_binops=old_binops)
    return _slot_table_dict[key]
special_method_names = set(get_slot_table(compiler_directives=None))
method_noargs = 'METH_NOARGS'
method_onearg = 'METH_O'
method_varargs = 'METH_VARARGS'
method_fastcall = '__Pyx_METH_FASTCALL'
method_keywords = 'METH_KEYWORDS'
method_coexist = 'METH_COEXIST'