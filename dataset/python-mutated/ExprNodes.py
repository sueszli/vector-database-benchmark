from __future__ import absolute_import
import cython
cython.declare(error=object, warning=object, warn_once=object, InternalError=object, CompileError=object, UtilityCode=object, TempitaUtilityCode=object, StringEncoding=object, operator=object, local_errors=object, report_error=object, Naming=object, Nodes=object, PyrexTypes=object, py_object_type=object, list_type=object, tuple_type=object, set_type=object, dict_type=object, unicode_type=object, str_type=object, bytes_type=object, type_type=object, Builtin=object, Symtab=object, Utils=object, find_coercion_error=object, debug_disposal_code=object, debug_temp_alloc=object, debug_coercion=object, bytearray_type=object, slice_type=object, memoryview_type=object, builtin_sequence_types=object, _py_int_types=object, IS_PYTHON3=cython.bint)
import re
import sys
import copy
import os.path
import operator
from .Errors import error, warning, InternalError, CompileError, report_error, local_errors, CannotSpecialize, performance_hint
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, unspecified_type
from . import TypeSlots
from .Builtin import list_type, tuple_type, set_type, dict_type, type_type, unicode_type, str_type, bytes_type, bytearray_type, basestring_type, slice_type, long_type, sequence_types as builtin_sequence_types, memoryview_type
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type, is_pythran_expr, pythran_func_type, pythran_binop_type, pythran_unaryop_type, has_np_pythran, pythran_indexing_code, pythran_indexing_type, is_pythran_supported_node_or_none, pythran_type, pythran_is_numpy_func_supported, pythran_get_func_include_file, pythran_functor
from .PyrexTypes import PythranExpr
try:
    from __builtin__ import basestring
except ImportError:
    basestring = str
    any_string_type = (bytes, str)
else:
    any_string_type = (bytes, unicode)
if sys.version_info[0] >= 3:
    IS_PYTHON3 = True
    _py_int_types = int
else:
    IS_PYTHON3 = False
    _py_int_types = (int, long)

class NotConstant(object):
    _obj = None

    def __new__(cls):
        if False:
            print('Hello World!')
        if NotConstant._obj is None:
            NotConstant._obj = super(NotConstant, cls).__new__(cls)
        return NotConstant._obj

    def __repr__(self):
        if False:
            return 10
        return '<NOT CONSTANT>'
not_a_constant = NotConstant()
constant_value_not_set = object()
coercion_error_dict = {(unicode_type, str_type): "Cannot convert Unicode string to 'str' implicitly. This is not portable and requires explicit encoding.", (unicode_type, bytes_type): "Cannot convert Unicode string to 'bytes' implicitly, encoding required.", (unicode_type, PyrexTypes.c_char_ptr_type): 'Unicode objects only support coercion to Py_UNICODE*.', (unicode_type, PyrexTypes.c_const_char_ptr_type): 'Unicode objects only support coercion to Py_UNICODE*.', (unicode_type, PyrexTypes.c_uchar_ptr_type): 'Unicode objects only support coercion to Py_UNICODE*.', (unicode_type, PyrexTypes.c_const_uchar_ptr_type): 'Unicode objects only support coercion to Py_UNICODE*.', (bytes_type, unicode_type): "Cannot convert 'bytes' object to unicode implicitly, decoding required", (bytes_type, str_type): "Cannot convert 'bytes' object to str implicitly. This is not portable to Py3.", (bytes_type, basestring_type): "Cannot convert 'bytes' object to basestring implicitly. This is not portable to Py3.", (bytes_type, PyrexTypes.c_py_unicode_ptr_type): "Cannot convert 'bytes' object to Py_UNICODE*, use 'unicode'.", (bytes_type, PyrexTypes.c_const_py_unicode_ptr_type): "Cannot convert 'bytes' object to Py_UNICODE*, use 'unicode'.", (basestring_type, bytes_type): "Cannot convert 'basestring' object to bytes implicitly. This is not portable.", (str_type, unicode_type): "str objects do not support coercion to unicode, use a unicode string literal instead (u'')", (str_type, bytes_type): "Cannot convert 'str' to 'bytes' implicitly. This is not portable.", (str_type, PyrexTypes.c_char_ptr_type): "'str' objects do not support coercion to C types (use 'bytes'?).", (str_type, PyrexTypes.c_const_char_ptr_type): "'str' objects do not support coercion to C types (use 'bytes'?).", (str_type, PyrexTypes.c_uchar_ptr_type): "'str' objects do not support coercion to C types (use 'bytes'?).", (str_type, PyrexTypes.c_const_uchar_ptr_type): "'str' objects do not support coercion to C types (use 'bytes'?).", (str_type, PyrexTypes.c_py_unicode_ptr_type): "'str' objects do not support coercion to C types (use 'unicode'?).", (str_type, PyrexTypes.c_const_py_unicode_ptr_type): "'str' objects do not support coercion to C types (use 'unicode'?).", (PyrexTypes.c_char_ptr_type, unicode_type): "Cannot convert 'char*' to unicode implicitly, decoding required", (PyrexTypes.c_const_char_ptr_type, unicode_type): "Cannot convert 'char*' to unicode implicitly, decoding required", (PyrexTypes.c_uchar_ptr_type, unicode_type): "Cannot convert 'char*' to unicode implicitly, decoding required", (PyrexTypes.c_const_uchar_ptr_type, unicode_type): "Cannot convert 'char*' to unicode implicitly, decoding required"}

def find_coercion_error(type_tuple, default, env):
    if False:
        for i in range(10):
            print('nop')
    err = coercion_error_dict.get(type_tuple)
    if err is None:
        return default
    elif env.directives['c_string_encoding'] and any((t in type_tuple for t in (PyrexTypes.c_char_ptr_type, PyrexTypes.c_uchar_ptr_type, PyrexTypes.c_const_char_ptr_type, PyrexTypes.c_const_uchar_ptr_type))):
        if type_tuple[1].is_pyobject:
            return default
        elif env.directives['c_string_encoding'] in ('ascii', 'default'):
            return default
        else:
            return "'%s' objects do not support coercion to C types with non-ascii or non-default c_string_encoding" % type_tuple[0].name
    else:
        return err

def default_str_type(env):
    if False:
        while True:
            i = 10
    return {'bytes': bytes_type, 'bytearray': bytearray_type, 'str': str_type, 'unicode': unicode_type}.get(env.directives['c_string_type'])

def check_negative_indices(*nodes):
    if False:
        return 10
    '\n    Raise a warning on nodes that are known to have negative numeric values.\n    Used to find (potential) bugs inside of "wraparound=False" sections.\n    '
    for node in nodes:
        if node is None or (not isinstance(node.constant_result, _py_int_types) and (not isinstance(node.constant_result, float))):
            continue
        if node.constant_result < 0:
            warning(node.pos, "the result of using negative indices inside of code sections marked as 'wraparound=False' is undefined", level=1)

def infer_sequence_item_type(env, seq_node, index_node=None, seq_type=None):
    if False:
        return 10
    if not seq_node.is_sequence_constructor:
        if seq_type is None:
            seq_type = seq_node.infer_type(env)
        if seq_type is tuple_type:
            if seq_node.cf_state and len(seq_node.cf_state) == 1:
                try:
                    seq_node = seq_node.cf_state[0].rhs
                except AttributeError:
                    pass
    if seq_node is not None and seq_node.is_sequence_constructor:
        if index_node is not None and index_node.has_constant_result():
            try:
                item = seq_node.args[index_node.constant_result]
            except (ValueError, TypeError, IndexError):
                pass
            else:
                return item.infer_type(env)
        item_types = {item.infer_type(env) for item in seq_node.args}
        if len(item_types) == 1:
            return item_types.pop()
    return None

def make_dedup_key(outer_type, item_nodes):
    if False:
        i = 10
        return i + 15
    '\n    Recursively generate a deduplication key from a sequence of values.\n    Includes Cython node types to work around the fact that (1, 2.0) == (1.0, 2), for example.\n\n    @param outer_type: The type of the outer container.\n    @param item_nodes: A sequence of constant nodes that will be traversed recursively.\n    @return: A tuple that can be used as a dict key for deduplication.\n    '
    item_keys = [(py_object_type, None, type(None)) if node is None else make_dedup_key(node.type, [node.mult_factor if node.is_literal else None] + node.args) if node.is_sequence_constructor else make_dedup_key(node.type, (node.start, node.stop, node.step)) if node.is_slice else (node.type, node.constant_result, type(node.constant_result) if node.type is py_object_type else None) if node.has_constant_result() else (node.type, node.value, node.unicode_value, 'IdentifierStringNode') if isinstance(node, IdentifierStringNode) else None for node in item_nodes]
    if None in item_keys:
        return None
    return (outer_type, tuple(item_keys))

def get_exception_handler(exception_value):
    if False:
        print('Hello World!')
    if exception_value is None:
        return ('__Pyx_CppExn2PyErr();', False)
    elif exception_value.type == PyrexTypes.c_char_type and exception_value.value == '*':
        return ('__Pyx_CppExn2PyErr();', True)
    elif exception_value.type.is_pyobject:
        return ('try { throw; } catch(const std::exception& exn) {PyErr_SetString(%s, exn.what());} catch(...) { PyErr_SetNone(%s); }' % (exception_value.entry.cname, exception_value.entry.cname), False)
    else:
        return ('%s(); if (!PyErr_Occurred())PyErr_SetString(PyExc_RuntimeError, "Error converting c++ exception.");' % exception_value.entry.cname, False)

def maybe_check_py_error(code, check_py_exception, pos, nogil):
    if False:
        for i in range(10):
            print('nop')
    if check_py_exception:
        if nogil:
            code.globalstate.use_utility_code(UtilityCode.load_cached('ErrOccurredWithGIL', 'Exceptions.c'))
            code.putln(code.error_goto_if('__Pyx_ErrOccurredWithGIL()', pos))
        else:
            code.putln(code.error_goto_if('PyErr_Occurred()', pos))

def translate_cpp_exception(code, pos, inside, py_result, exception_value, nogil):
    if False:
        return 10
    (raise_py_exception, check_py_exception) = get_exception_handler(exception_value)
    code.putln('try {')
    code.putln('%s' % inside)
    if py_result:
        code.putln(code.error_goto_if_null(py_result, pos))
    maybe_check_py_error(code, check_py_exception, pos, nogil)
    code.putln('} catch(...) {')
    if nogil:
        code.put_ensure_gil(declare_gilstate=True)
    code.putln(raise_py_exception)
    if nogil:
        code.put_release_ensured_gil()
    code.putln(code.error_goto(pos))
    code.putln('}')

def needs_cpp_exception_conversion(node):
    if False:
        print('Hello World!')
    assert node.exception_check == '+'
    if node.exception_value is None:
        return True
    if node.exception_value.is_name:
        return False
    if isinstance(node.exception_value, CharNode) and node.exception_value.value == '*':
        return True
    return False

def translate_double_cpp_exception(code, pos, lhs_type, lhs_code, rhs_code, lhs_exc_val, assign_exc_val, nogil):
    if False:
        for i in range(10):
            print('nop')
    (handle_lhs_exc, lhc_check_py_exc) = get_exception_handler(lhs_exc_val)
    (handle_assignment_exc, assignment_check_py_exc) = get_exception_handler(assign_exc_val)
    code.putln('try {')
    code.putln(lhs_type.declaration_code('__pyx_local_lvalue = %s;' % lhs_code))
    maybe_check_py_error(code, lhc_check_py_exc, pos, nogil)
    code.putln('try {')
    code.putln('__pyx_local_lvalue = %s;' % rhs_code)
    maybe_check_py_error(code, assignment_check_py_exc, pos, nogil)
    code.putln('} catch(...) {')
    if nogil:
        code.put_ensure_gil(declare_gilstate=True)
    code.putln(handle_assignment_exc)
    if nogil:
        code.put_release_ensured_gil()
    code.putln(code.error_goto(pos))
    code.putln('}')
    code.putln('} catch(...) {')
    if nogil:
        code.put_ensure_gil(declare_gilstate=True)
    code.putln(handle_lhs_exc)
    if nogil:
        code.put_release_ensured_gil()
    code.putln(code.error_goto(pos))
    code.putln('}')

class ExprNode(Node):
    result_ctype = None
    type = None
    annotation = None
    temp_code = None
    old_temp = None
    use_managed_ref = True
    result_is_used = True
    is_numpy_attribute = False
    generator_arg_tag = None
    is_sequence_constructor = False
    is_dict_literal = False
    is_set_literal = False
    is_string_literal = False
    is_attribute = False
    is_subscript = False
    is_slice = False
    is_buffer_access = False
    is_memview_index = False
    is_memview_slice = False
    is_memview_broadcast = False
    is_memview_copy_assignment = False
    is_temp = False
    has_temp_moved = False
    is_target = False
    is_starred = False
    constant_result = constant_value_not_set
    child_attrs = property(fget=operator.attrgetter('subexprs'))

    def analyse_annotations(self, env):
        if False:
            while True:
                i = 10
        pass

    def not_implemented(self, method_name):
        if False:
            for i in range(10):
                print('nop')
        print_call_chain(method_name, 'not implemented')
        raise InternalError('%s.%s not implemented' % (self.__class__.__name__, method_name))

    def is_lvalue(self):
        if False:
            i = 10
            return i + 15
        return 0

    def is_addressable(self):
        if False:
            for i in range(10):
                print('nop')
        return self.is_lvalue() and (not self.type.is_memoryviewslice)

    def is_ephemeral(self):
        if False:
            print('Hello World!')
        return self.type.is_pyobject and self.is_temp

    def subexpr_nodes(self):
        if False:
            while True:
                i = 10
        nodes = []
        for name in self.subexprs:
            item = getattr(self, name)
            if item is not None:
                if type(item) is list:
                    nodes.extend(item)
                else:
                    nodes.append(item)
        return nodes

    def result(self):
        if False:
            return 10
        if self.is_temp:
            return self.temp_code
        else:
            return self.calculate_result_code()

    def _make_move_result_rhs(self, result, optional=False):
        if False:
            i = 10
            return i + 15
        if optional and (not (self.is_temp and self.type.is_cpp_class and (not self.type.is_reference))):
            return result
        self.has_temp_moved = True
        return '{}({})'.format('__PYX_STD_MOVE_IF_SUPPORTED' if optional else 'std::move', result)

    def move_result_rhs(self):
        if False:
            i = 10
            return i + 15
        return self._make_move_result_rhs(self.result(), optional=True)

    def move_result_rhs_as(self, type):
        if False:
            return 10
        result = self.result_as(type)
        if not (type.is_reference or type.needs_refcounting):
            requires_move = type.is_rvalue_reference and self.is_temp
            result = self._make_move_result_rhs(result, optional=not requires_move)
        return result

    def pythran_result(self, type_=None):
        if False:
            while True:
                i = 10
        if is_pythran_supported_node_or_none(self):
            return to_pythran(self)
        assert type_ is not None
        return to_pythran(self, type_)

    def is_c_result_required(self):
        if False:
            print('Hello World!')
        '\n        Subtypes may return False here if result temp allocation can be skipped.\n        '
        return True

    def result_as(self, type=None):
        if False:
            print('Hello World!')
        if self.is_temp and self.type.is_pyobject and (type != py_object_type):
            return typecast(type, py_object_type, self.result())
        return typecast(type, self.ctype(), self.result())

    def py_result(self):
        if False:
            for i in range(10):
                print('nop')
        return self.result_as(py_object_type)

    def ctype(self):
        if False:
            return 10
        return self.result_ctype or self.type

    def get_constant_c_result_code(self):
        if False:
            print('Hello World!')
        return None

    def calculate_constant_result(self):
        if False:
            print('Hello World!')
        pass

    def has_constant_result(self):
        if False:
            i = 10
            return i + 15
        return self.constant_result is not constant_value_not_set and self.constant_result is not not_a_constant

    def compile_time_value(self, denv):
        if False:
            print('Hello World!')
        error(self.pos, 'Invalid compile-time expression')

    def compile_time_value_error(self, e):
        if False:
            i = 10
            return i + 15
        error(self.pos, 'Error in compile-time expression: %s: %s' % (e.__class__.__name__, e))

    def analyse_target_declaration(self, env):
        if False:
            return 10
        error(self.pos, 'Cannot assign to or delete this')

    def analyse_assignment_expression_target_declaration(self, env):
        if False:
            for i in range(10):
                print('nop')
        error(self.pos, 'Cannot use anything except a name in an assignment expression')

    def analyse_const_expression(self, env):
        if False:
            for i in range(10):
                print('nop')
        node = self.analyse_types(env)
        node.check_const()
        return node

    def analyse_expressions(self, env):
        if False:
            while True:
                i = 10
        return self.analyse_types(env)

    def analyse_target_expression(self, env, rhs):
        if False:
            for i in range(10):
                print('nop')
        return self.analyse_target_types(env)

    def analyse_boolean_expression(self, env):
        if False:
            for i in range(10):
                print('nop')
        node = self.analyse_types(env)
        bool = node.coerce_to_boolean(env)
        return bool

    def analyse_temp_boolean_expression(self, env):
        if False:
            return 10
        node = self.analyse_types(env)
        return node.coerce_to_boolean(env).coerce_to_simple(env)

    def type_dependencies(self, env):
        if False:
            for i in range(10):
                print('nop')
        if getattr(self, 'type', None) is not None:
            return ()
        return sum([node.type_dependencies(env) for node in self.subexpr_nodes()], ())

    def infer_type(self, env):
        if False:
            i = 10
            return i + 15
        type = getattr(self, 'type', None)
        if type is not None:
            return type
        entry = getattr(self, 'entry', None)
        if entry is not None:
            return entry.type
        self.not_implemented('infer_type')

    def nonlocally_immutable(self):
        if False:
            while True:
                i = 10
        return self.is_literal or self.is_temp or self.type.is_array or self.type.is_cfunction

    def inferable_item_node(self, index=0):
        if False:
            return 10
        '\n        Return a node that represents the (type) result of an indexing operation,\n        e.g. for tuple unpacking or iteration.\n        '
        return IndexNode(self.pos, base=self, index=IntNode(self.pos, value=str(index), constant_result=index, type=PyrexTypes.c_py_ssize_t_type))

    def analyse_as_module(self, env):
        if False:
            i = 10
            return i + 15
        return None

    def analyse_as_type(self, env):
        if False:
            print('Hello World!')
        return None

    def analyse_as_specialized_type(self, env):
        if False:
            return 10
        type = self.analyse_as_type(env)
        if type and type.is_fused and env.fused_to_specific:
            try:
                return type.specialize(env.fused_to_specific)
            except KeyError:
                pass
        if type and type.is_fused:
            error(self.pos, 'Type is not specific')
        return type

    def analyse_as_extension_type(self, env):
        if False:
            return 10
        return None

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        self.not_implemented('analyse_types')

    def analyse_target_types(self, env):
        if False:
            print('Hello World!')
        return self.analyse_types(env)

    def nogil_check(self, env):
        if False:
            print('Hello World!')
        if self.type and self.type.is_pyobject:
            self.gil_error()

    def gil_assignment_check(self, env):
        if False:
            print('Hello World!')
        if env.nogil and self.type.is_pyobject:
            error(self.pos, 'Assignment of Python object not allowed without gil')

    def check_const(self):
        if False:
            for i in range(10):
                print('nop')
        self.not_const()
        return False

    def not_const(self):
        if False:
            for i in range(10):
                print('nop')
        error(self.pos, 'Not allowed in a constant expression')

    def check_const_addr(self):
        if False:
            return 10
        self.addr_not_const()
        return False

    def addr_not_const(self):
        if False:
            return 10
        error(self.pos, 'Address is not constant')

    def result_in_temp(self):
        if False:
            for i in range(10):
                print('nop')
        return self.is_temp

    def target_code(self):
        if False:
            print('Hello World!')
        return self.calculate_result_code()

    def calculate_result_code(self):
        if False:
            i = 10
            return i + 15
        self.not_implemented('calculate_result_code')

    def allocate_temp_result(self, code):
        if False:
            while True:
                i = 10
        if self.temp_code:
            raise RuntimeError('Temp allocated multiple times in %r: %r' % (self.__class__.__name__, self.pos))
        type = self.type
        if not type.is_void:
            if type.is_pyobject:
                type = PyrexTypes.py_object_type
            elif not (self.result_is_used or type.is_memoryviewslice or self.is_c_result_required()):
                self.temp_code = None
                return
            self.temp_code = code.funcstate.allocate_temp(type, manage_ref=self.use_managed_ref)
        else:
            self.temp_code = None

    def release_temp_result(self, code):
        if False:
            while True:
                i = 10
        if not self.temp_code:
            if not self.result_is_used:
                return
            pos = (os.path.basename(self.pos[0].get_description()),) + self.pos[1:] if self.pos else '(?)'
            if self.old_temp:
                raise RuntimeError('temp %s released multiple times in %s at %r' % (self.old_temp, self.__class__.__name__, pos))
            else:
                raise RuntimeError('no temp, but release requested in %s at %r' % (self.__class__.__name__, pos))
        code.funcstate.release_temp(self.temp_code)
        self.old_temp = self.temp_code
        self.temp_code = None

    def make_owned_reference(self, code):
        if False:
            i = 10
            return i + 15
        '\n        Make sure we own a reference to result.\n        If the result is in a temp, it is already a new reference.\n        '
        if not self.result_in_temp():
            code.put_incref(self.result(), self.ctype())

    def make_owned_memoryviewslice(self, code):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make sure we own the reference to this memoryview slice.\n        '
        if not self.result_in_temp():
            code.put_incref_memoryviewslice(self.result(), self.type, have_gil=not self.in_nogil_context)

    def generate_evaluation_code(self, code):
        if False:
            i = 10
            return i + 15
        self.generate_subexpr_evaluation_code(code)
        code.mark_pos(self.pos)
        if self.is_temp:
            self.allocate_temp_result(code)
        self.generate_result_code(code)
        if self.is_temp and (not (self.type.is_string or self.type.is_pyunicode_ptr)):
            self.generate_subexpr_disposal_code(code)
            self.free_subexpr_temps(code)

    def generate_subexpr_evaluation_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        for node in self.subexpr_nodes():
            node.generate_evaluation_code(code)

    def generate_result_code(self, code):
        if False:
            return 10
        self.not_implemented('generate_result_code')

    def generate_disposal_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        if self.has_temp_moved:
            code.globalstate.use_utility_code(UtilityCode.load_cached('MoveIfSupported', 'CppSupport.cpp'))
        if self.is_temp:
            if self.type.is_string or self.type.is_pyunicode_ptr:
                self.generate_subexpr_disposal_code(code)
                self.free_subexpr_temps(code)
            if self.result():
                code.put_decref_clear(self.result(), self.ctype(), have_gil=not self.in_nogil_context)
        else:
            self.generate_subexpr_disposal_code(code)

    def generate_subexpr_disposal_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        for node in self.subexpr_nodes():
            node.generate_disposal_code(code)

    def generate_post_assignment_code(self, code):
        if False:
            return 10
        if self.is_temp:
            if self.type.is_string or self.type.is_pyunicode_ptr:
                self.generate_subexpr_disposal_code(code)
                self.free_subexpr_temps(code)
            elif self.type.is_pyobject:
                code.putln('%s = 0;' % self.result())
            elif self.type.is_memoryviewslice:
                code.putln('%s.memview = NULL;' % self.result())
                code.putln('%s.data = NULL;' % self.result())
            if self.has_temp_moved:
                code.globalstate.use_utility_code(UtilityCode.load_cached('MoveIfSupported', 'CppSupport.cpp'))
        else:
            self.generate_subexpr_disposal_code(code)

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False, exception_check=None, exception_value=None):
        if False:
            while True:
                i = 10
        pass

    def generate_deletion_code(self, code, ignore_nonexisting=False):
        if False:
            i = 10
            return i + 15
        pass

    def free_temps(self, code):
        if False:
            print('Hello World!')
        if self.is_temp:
            if not self.type.is_void:
                self.release_temp_result(code)
        else:
            self.free_subexpr_temps(code)

    def free_subexpr_temps(self, code):
        if False:
            return 10
        for sub in self.subexpr_nodes():
            sub.free_temps(code)

    def generate_function_definitions(self, env, code):
        if False:
            i = 10
            return i + 15
        pass

    def generate_decref_set(self, code, rhs):
        if False:
            for i in range(10):
                print('nop')
        code.put_decref_set(self.result(), self.ctype(), rhs)

    def generate_xdecref_set(self, code, rhs):
        if False:
            for i in range(10):
                print('nop')
        code.put_xdecref_set(self.result(), self.ctype(), rhs)

    def generate_gotref(self, code, handle_null=False, maybe_null_extra_check=True):
        if False:
            while True:
                i = 10
        if not (handle_null and self.cf_is_null):
            if handle_null and self.cf_maybe_null and maybe_null_extra_check:
                self.generate_xgotref(code)
            else:
                code.put_gotref(self.result(), self.ctype())

    def generate_xgotref(self, code):
        if False:
            return 10
        code.put_xgotref(self.result(), self.ctype())

    def generate_giveref(self, code):
        if False:
            i = 10
            return i + 15
        code.put_giveref(self.result(), self.ctype())

    def generate_xgiveref(self, code):
        if False:
            print('Hello World!')
        code.put_xgiveref(self.result(), self.ctype())

    def annotate(self, code):
        if False:
            return 10
        for node in self.subexpr_nodes():
            node.annotate(code)

    def coerce_to(self, dst_type, env):
        if False:
            print('Hello World!')
        src = self
        src_type = self.type
        if self.check_for_coercion_error(dst_type, env):
            return self
        used_as_reference = dst_type.is_reference
        if used_as_reference and (not src_type.is_reference):
            dst_type = dst_type.ref_base_type
        if src_type.is_cv_qualified:
            src_type = src_type.cv_base_type
        if src_type.is_fused or dst_type.is_fused:
            if src_type.is_cfunction and (not dst_type.is_fused) and dst_type.is_ptr and dst_type.base_type.is_cfunction:
                dst_type = dst_type.base_type
                for signature in src_type.get_all_specialized_function_types():
                    if signature.same_as(dst_type):
                        src.type = signature
                        src.entry = src.type.entry
                        src.entry.used = True
                        return self
            if src_type.is_fused:
                error(self.pos, 'Type is not specialized')
            elif src_type.is_null_ptr and dst_type.is_ptr:
                return self
            else:
                error(self.pos, 'Cannot coerce to a type that is not specialized')
            self.type = error_type
            return self
        if self.coercion_type is not None:
            node = NameNode(self.pos, name='', type=self.coercion_type)
            node.coerce_to(dst_type, env)
        if dst_type.is_memoryviewslice:
            from . import MemoryView
            if not src.type.is_memoryviewslice:
                if src.type.is_pyobject:
                    src = CoerceToMemViewSliceNode(src, dst_type, env)
                elif src.type.is_array:
                    src = CythonArrayNode.from_carray(src, env).coerce_to(dst_type, env)
                elif not src_type.is_error:
                    error(self.pos, "Cannot convert '%s' to memoryviewslice" % (src_type,))
            else:
                if src.type.writable_needed:
                    dst_type.writable_needed = True
                if not src.type.conforms_to(dst_type, broadcast=self.is_memview_broadcast, copying=self.is_memview_copy_assignment):
                    if src.type.dtype.same_as(dst_type.dtype):
                        msg = "Memoryview '%s' not conformable to memoryview '%s'."
                        tup = (src.type, dst_type)
                    else:
                        msg = 'Different base types for memoryviews (%s, %s)'
                        tup = (src.type.dtype, dst_type.dtype)
                    error(self.pos, msg % tup)
        elif dst_type.is_pyobject:
            if src.is_none:
                pass
            elif src.constant_result is None:
                src = NoneNode(src.pos).coerce_to(dst_type, env)
            else:
                if not src.type.is_pyobject:
                    if dst_type is bytes_type and src.type.is_int:
                        src = CoerceIntToBytesNode(src, env)
                    else:
                        src = CoerceToPyTypeNode(src, env, type=dst_type)
                if not src.type.subtype_of(dst_type):
                    src = PyTypeTestNode(src, dst_type, env)
        elif is_pythran_expr(dst_type) and is_pythran_supported_type(src.type):
            return src
        elif is_pythran_expr(src.type):
            if is_pythran_supported_type(dst_type):
                return src
            src = CoerceToPyTypeNode(src, env, type=dst_type)
        elif src.type.is_pyobject:
            if used_as_reference and dst_type.is_cpp_class:
                warning(self.pos, 'Cannot pass Python object as C++ data structure reference (%s &), will pass by copy.' % dst_type)
            src = CoerceFromPyTypeNode(dst_type, src, env)
        elif dst_type.is_complex and src_type != dst_type and dst_type.assignable_from(src_type):
            src = CoerceToComplexNode(src, dst_type, env)
        elif src_type is PyrexTypes.soft_complex_type and src_type != dst_type and (not dst_type.assignable_from(src_type)):
            src = coerce_from_soft_complex(src, dst_type, env)
        elif not (src.type == dst_type or str(src.type) == str(dst_type) or dst_type.assignable_from(src_type)):
            self.fail_assignment(dst_type)
        return src

    def fail_assignment(self, dst_type):
        if False:
            for i in range(10):
                print('nop')
        extra_diagnostics = dst_type.assignment_failure_extra_info(self.type)
        if extra_diagnostics:
            extra_diagnostics = '. ' + extra_diagnostics
        error(self.pos, "Cannot assign type '%s' to '%s'%s" % (self.type, dst_type, extra_diagnostics))

    def check_for_coercion_error(self, dst_type, env, fail=False, default=None):
        if False:
            while True:
                i = 10
        if fail and (not default):
            default = "Cannot assign type '%(FROM)s' to '%(TO)s'"
        message = find_coercion_error((self.type, dst_type), default, env)
        if message is not None:
            error(self.pos, message % {'FROM': self.type, 'TO': dst_type})
            return True
        if fail:
            self.fail_assignment(dst_type)
            return True
        return False

    def coerce_to_pyobject(self, env):
        if False:
            return 10
        return self.coerce_to(PyrexTypes.py_object_type, env)

    def coerce_to_boolean(self, env):
        if False:
            while True:
                i = 10
        if self.has_constant_result():
            bool_value = bool(self.constant_result)
            return BoolNode(self.pos, value=bool_value, constant_result=bool_value)
        type = self.type
        if type.is_enum or type.is_error:
            return self
        elif type is PyrexTypes.c_bint_type:
            return self
        elif type.is_pyobject or type.is_int or type.is_ptr or type.is_float:
            return CoerceToBooleanNode(self, env)
        elif type.is_cpp_class and type.scope and type.scope.lookup('operator bool'):
            return SimpleCallNode(self.pos, function=AttributeNode(self.pos, obj=self, attribute=StringEncoding.EncodedString('operator bool')), args=[]).analyse_types(env)
        elif type.is_ctuple:
            bool_value = len(type.components) == 0
            return BoolNode(self.pos, value=bool_value, constant_result=bool_value)
        else:
            error(self.pos, "Type '%s' not acceptable as a boolean" % type)
            return self

    def coerce_to_integer(self, env):
        if False:
            print('Hello World!')
        if self.type.is_int:
            return self
        else:
            return self.coerce_to(PyrexTypes.c_long_type, env)

    def coerce_to_temp(self, env):
        if False:
            return 10
        if self.result_in_temp():
            return self
        else:
            return CoerceToTempNode(self, env)

    def coerce_to_simple(self, env):
        if False:
            while True:
                i = 10
        if self.is_simple():
            return self
        else:
            return self.coerce_to_temp(env)

    def is_simple(self):
        if False:
            i = 10
            return i + 15
        return self.result_in_temp()

    def may_be_none(self):
        if False:
            for i in range(10):
                print('nop')
        if self.type and (not (self.type.is_pyobject or self.type.is_memoryviewslice)):
            return False
        if self.has_constant_result():
            return self.constant_result is not None
        return True

    def as_cython_attribute(self):
        if False:
            while True:
                i = 10
        return None

    def as_none_safe_node(self, message, error='PyExc_TypeError', format_args=()):
        if False:
            return 10
        if self.may_be_none():
            return NoneCheckNode(self, error, message, format_args)
        else:
            return self

    @classmethod
    def from_node(cls, node, **kwargs):
        if False:
            while True:
                i = 10
        'Instantiate this node class from another node, properly\n        copying over all attributes that one would forget otherwise.\n        '
        attributes = 'cf_state cf_maybe_null cf_is_null constant_result'.split()
        for attr_name in attributes:
            if attr_name in kwargs:
                continue
            try:
                value = getattr(node, attr_name)
            except AttributeError:
                pass
            else:
                kwargs[attr_name] = value
        return cls(node.pos, **kwargs)

    def get_known_standard_library_import(self):
        if False:
            print('Hello World!')
        '\n        Gets the module.path that this node was imported from.\n\n        Many nodes do not have one, or it is ambiguous, in which case\n        this function returns a false value.\n        '
        return None

class AtomicExprNode(ExprNode):
    subexprs = []

    def generate_subexpr_evaluation_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        pass

    def generate_subexpr_disposal_code(self, code):
        if False:
            print('Hello World!')
        pass

class PyConstNode(AtomicExprNode):
    is_literal = 1
    type = py_object_type
    nogil_check = None

    def is_simple(self):
        if False:
            return 10
        return 1

    def may_be_none(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def analyse_types(self, env):
        if False:
            return 10
        return self

    def calculate_result_code(self):
        if False:
            for i in range(10):
                print('nop')
        return self.value

    def generate_result_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        pass

class NoneNode(PyConstNode):
    is_none = 1
    value = 'Py_None'
    constant_result = None

    def compile_time_value(self, denv):
        if False:
            for i in range(10):
                print('nop')
        return None

    def may_be_none(self):
        if False:
            while True:
                i = 10
        return True

    def coerce_to(self, dst_type, env):
        if False:
            for i in range(10):
                print('nop')
        if not (dst_type.is_pyobject or dst_type.is_memoryviewslice or dst_type.is_error):
            error(self.pos, 'Cannot assign None to %s' % dst_type)
        return super(NoneNode, self).coerce_to(dst_type, env)

class EllipsisNode(PyConstNode):
    value = 'Py_Ellipsis'
    constant_result = Ellipsis

    def compile_time_value(self, denv):
        if False:
            print('Hello World!')
        return Ellipsis

class ConstNode(AtomicExprNode):
    is_literal = 1
    nogil_check = None

    def is_simple(self):
        if False:
            print('Hello World!')
        return 1

    def nonlocally_immutable(self):
        if False:
            for i in range(10):
                print('nop')
        return 1

    def may_be_none(self):
        if False:
            return 10
        return False

    def analyse_types(self, env):
        if False:
            print('Hello World!')
        return self

    def check_const(self):
        if False:
            print('Hello World!')
        return True

    def get_constant_c_result_code(self):
        if False:
            print('Hello World!')
        return self.calculate_result_code()

    def calculate_result_code(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.value)

    def generate_result_code(self, code):
        if False:
            i = 10
            return i + 15
        pass

class BoolNode(ConstNode):
    type = PyrexTypes.c_bint_type

    def calculate_constant_result(self):
        if False:
            return 10
        self.constant_result = self.value

    def compile_time_value(self, denv):
        if False:
            while True:
                i = 10
        return self.value

    def calculate_result_code(self):
        if False:
            print('Hello World!')
        if self.type.is_pyobject:
            return 'Py_True' if self.value else 'Py_False'
        else:
            return str(int(self.value))

    def coerce_to(self, dst_type, env):
        if False:
            for i in range(10):
                print('nop')
        if dst_type == self.type:
            return self
        if dst_type is py_object_type and self.type is Builtin.bool_type:
            return self
        if dst_type.is_pyobject and self.type.is_int:
            return BoolNode(self.pos, value=self.value, constant_result=self.constant_result, type=Builtin.bool_type)
        if dst_type.is_int and self.type.is_pyobject:
            return BoolNode(self.pos, value=self.value, constant_result=self.constant_result, type=PyrexTypes.c_bint_type)
        return ConstNode.coerce_to(self, dst_type, env)

class NullNode(ConstNode):
    type = PyrexTypes.c_null_ptr_type
    value = 'NULL'
    constant_result = 0

    def get_constant_c_result_code(self):
        if False:
            while True:
                i = 10
        return self.value

class CharNode(ConstNode):
    type = PyrexTypes.c_char_type

    def calculate_constant_result(self):
        if False:
            i = 10
            return i + 15
        self.constant_result = ord(self.value)

    def compile_time_value(self, denv):
        if False:
            while True:
                i = 10
        return ord(self.value)

    def calculate_result_code(self):
        if False:
            i = 10
            return i + 15
        return "'%s'" % StringEncoding.escape_char(self.value)

class IntNode(ConstNode):
    unsigned = ''
    longness = ''
    is_c_literal = None

    @property
    def hex_value(self):
        if False:
            i = 10
            return i + 15
        return Utils.strip_py2_long_suffix(hex(Utils.str_to_number(self.value)))

    @property
    def base_10_value(self):
        if False:
            return 10
        return str(Utils.str_to_number(self.value))

    def __init__(self, pos, **kwds):
        if False:
            for i in range(10):
                print('nop')
        ExprNode.__init__(self, pos, **kwds)
        if 'type' not in kwds:
            self.type = self.find_suitable_type_for_value()

    def find_suitable_type_for_value(self):
        if False:
            i = 10
            return i + 15
        if self.constant_result is constant_value_not_set:
            try:
                self.calculate_constant_result()
            except ValueError:
                pass
        if self.is_c_literal or not self.has_constant_result() or self.unsigned or (self.longness == 'LL'):
            rank = self.longness == 'LL' and 2 or 1
            suitable_type = PyrexTypes.modifiers_and_name_to_type[not self.unsigned, rank, 'int']
            if self.type:
                suitable_type = PyrexTypes.widest_numeric_type(suitable_type, self.type)
        elif -2 ** 31 <= self.constant_result < 2 ** 31:
            if self.type and self.type.is_int:
                suitable_type = self.type
            else:
                suitable_type = PyrexTypes.c_long_type
        else:
            suitable_type = PyrexTypes.py_object_type
        return suitable_type

    def coerce_to(self, dst_type, env):
        if False:
            for i in range(10):
                print('nop')
        if self.type is dst_type:
            return self
        elif dst_type.is_float:
            if self.has_constant_result():
                return FloatNode(self.pos, value='%d.0' % int(self.constant_result), type=dst_type, constant_result=float(self.constant_result))
            else:
                return FloatNode(self.pos, value=self.value, type=dst_type, constant_result=not_a_constant)
        if dst_type.is_numeric and (not dst_type.is_complex):
            node = IntNode(self.pos, value=self.value, constant_result=self.constant_result, type=dst_type, is_c_literal=True, unsigned=self.unsigned, longness=self.longness)
            return node
        elif dst_type.is_pyobject:
            node = IntNode(self.pos, value=self.value, constant_result=self.constant_result, type=PyrexTypes.py_object_type, is_c_literal=False, unsigned=self.unsigned, longness=self.longness)
        else:
            node = IntNode(self.pos, value=self.value, constant_result=self.constant_result, unsigned=self.unsigned, longness=self.longness)
        return ConstNode.coerce_to(node, dst_type, env)

    def coerce_to_boolean(self, env):
        if False:
            print('Hello World!')
        return IntNode(self.pos, value=self.value, constant_result=self.constant_result, type=PyrexTypes.c_bint_type, unsigned=self.unsigned, longness=self.longness)

    def generate_evaluation_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        if self.type.is_pyobject:
            value = Utils.str_to_number(self.value)
            formatter = hex if value > 10 ** 13 else str
            plain_integer_string = formatter(value)
            plain_integer_string = Utils.strip_py2_long_suffix(plain_integer_string)
            self.result_code = code.get_py_int(plain_integer_string, self.longness)
        else:
            self.result_code = self.get_constant_c_result_code()

    def get_constant_c_result_code(self):
        if False:
            print('Hello World!')
        (unsigned, longness) = (self.unsigned, self.longness)
        literal = self.value_as_c_integer_string()
        if not (unsigned or longness) and self.type.is_int and (literal[0] == '-') and (literal[1] != '0'):
            if self.type.rank >= PyrexTypes.c_longlong_type.rank:
                longness = 'LL'
            elif self.type.rank >= PyrexTypes.c_long_type.rank:
                longness = 'L'
        return literal + unsigned + longness

    def value_as_c_integer_string(self):
        if False:
            i = 10
            return i + 15
        value = self.value
        if len(value) <= 2:
            return value
        neg_sign = ''
        if value[0] == '-':
            neg_sign = '-'
            value = value[1:]
        if value[0] == '0':
            literal_type = value[1]
            if neg_sign and literal_type in 'oOxX0123456789' and value[2:].isdigit():
                value = str(Utils.str_to_number(value))
            elif literal_type in 'oO':
                value = '0' + value[2:]
            elif literal_type in 'bB':
                value = str(int(value[2:], 2))
        elif value.isdigit() and (not self.unsigned) and (not self.longness):
            if not neg_sign:
                value = '0x%X' % int(value)
        return neg_sign + value

    def calculate_result_code(self):
        if False:
            print('Hello World!')
        return self.result_code

    def calculate_constant_result(self):
        if False:
            return 10
        self.constant_result = Utils.str_to_number(self.value)

    def compile_time_value(self, denv):
        if False:
            while True:
                i = 10
        return Utils.str_to_number(self.value)

class FloatNode(ConstNode):
    type = PyrexTypes.c_double_type

    def calculate_constant_result(self):
        if False:
            i = 10
            return i + 15
        self.constant_result = float(self.value)

    def compile_time_value(self, denv):
        if False:
            while True:
                i = 10
        float_value = float(self.value)
        str_float_value = ('%.330f' % float_value).strip('0')
        str_value = Utils.normalise_float_repr(self.value)
        if str_value not in (str_float_value, repr(float_value).lstrip('0')):
            warning(self.pos, 'Using this floating point value with DEF may lose precision, using %r' % float_value)
        return float_value

    def coerce_to(self, dst_type, env):
        if False:
            i = 10
            return i + 15
        if dst_type.is_pyobject and self.type.is_float:
            return FloatNode(self.pos, value=self.value, constant_result=self.constant_result, type=Builtin.float_type)
        if dst_type.is_float and self.type.is_pyobject:
            return FloatNode(self.pos, value=self.value, constant_result=self.constant_result, type=dst_type)
        return ConstNode.coerce_to(self, dst_type, env)

    def calculate_result_code(self):
        if False:
            i = 10
            return i + 15
        return self.result_code

    def get_constant_c_result_code(self):
        if False:
            while True:
                i = 10
        strval = self.value
        assert isinstance(strval, basestring)
        cmpval = repr(float(strval))
        if cmpval == 'nan':
            return '(Py_HUGE_VAL * 0)'
        elif cmpval == 'inf':
            return 'Py_HUGE_VAL'
        elif cmpval == '-inf':
            return '(-Py_HUGE_VAL)'
        else:
            return strval

    def generate_evaluation_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        c_value = self.get_constant_c_result_code()
        if self.type.is_pyobject:
            self.result_code = code.get_py_float(self.value, c_value)
        else:
            self.result_code = c_value

def _analyse_name_as_type(name, pos, env):
    if False:
        return 10
    ctype = PyrexTypes.parse_basic_type(name)
    if ctype is not None and env.in_c_type_context:
        return ctype
    global_scope = env.global_scope()
    global_entry = global_scope.lookup(name)
    if global_entry and global_entry.is_type:
        type = global_entry.type
        if not env.in_c_type_context and type is Builtin.int_type and (global_scope.context.language_level == 2):
            type = py_object_type
        if type and (type.is_pyobject or env.in_c_type_context):
            return type
        ctype = ctype or type
    from .TreeFragment import TreeFragment
    with local_errors(ignore=True):
        pos = (pos[0], pos[1], pos[2] - 7)
        try:
            declaration = TreeFragment(u'sizeof(%s)' % name, name=pos[0].filename, initial_pos=pos)
        except CompileError:
            pass
        else:
            sizeof_node = declaration.root.stats[0].expr
            if isinstance(sizeof_node, SizeofTypeNode):
                sizeof_node = sizeof_node.analyse_types(env)
                if isinstance(sizeof_node, SizeofTypeNode):
                    type = sizeof_node.arg_type
                    if type and (type.is_pyobject or env.in_c_type_context):
                        return type
                    ctype = ctype or type
    return ctype

class BytesNode(ConstNode):
    is_string_literal = True
    type = bytes_type

    def calculate_constant_result(self):
        if False:
            i = 10
            return i + 15
        self.constant_result = self.value

    def as_sliced_node(self, start, stop, step=None):
        if False:
            for i in range(10):
                print('nop')
        value = StringEncoding.bytes_literal(self.value[start:stop:step], self.value.encoding)
        return BytesNode(self.pos, value=value, constant_result=value)

    def compile_time_value(self, denv):
        if False:
            return 10
        return self.value.byteencode()

    def analyse_as_type(self, env):
        if False:
            return 10
        return _analyse_name_as_type(self.value.decode('ISO8859-1'), self.pos, env)

    def can_coerce_to_char_literal(self):
        if False:
            return 10
        return len(self.value) == 1

    def coerce_to_boolean(self, env):
        if False:
            for i in range(10):
                print('nop')
        bool_value = bool(self.value)
        return BoolNode(self.pos, value=bool_value, constant_result=bool_value)

    def coerce_to(self, dst_type, env):
        if False:
            print('Hello World!')
        if self.type == dst_type:
            return self
        if dst_type.is_int:
            if not self.can_coerce_to_char_literal():
                error(self.pos, 'Only single-character string literals can be coerced into ints.')
                return self
            if dst_type.is_unicode_char:
                error(self.pos, 'Bytes literals cannot coerce to Py_UNICODE/Py_UCS4, use a unicode literal instead.')
                return self
            return CharNode(self.pos, value=self.value, constant_result=ord(self.value))
        node = BytesNode(self.pos, value=self.value, constant_result=self.constant_result)
        if dst_type.is_pyobject:
            if dst_type in (py_object_type, Builtin.bytes_type):
                node.type = Builtin.bytes_type
            else:
                self.check_for_coercion_error(dst_type, env, fail=True)
            return node
        elif dst_type in (PyrexTypes.c_char_ptr_type, PyrexTypes.c_const_char_ptr_type):
            node.type = dst_type
            return node
        elif dst_type in (PyrexTypes.c_uchar_ptr_type, PyrexTypes.c_const_uchar_ptr_type, PyrexTypes.c_void_ptr_type):
            node.type = PyrexTypes.c_const_char_ptr_type if dst_type == PyrexTypes.c_const_uchar_ptr_type else PyrexTypes.c_char_ptr_type
            return CastNode(node, dst_type)
        elif dst_type.assignable_from(PyrexTypes.c_char_ptr_type):
            if not dst_type.is_cpp_class or dst_type.is_const:
                node.type = dst_type
                return node
        return ConstNode.coerce_to(node, dst_type, env)

    def generate_evaluation_code(self, code):
        if False:
            i = 10
            return i + 15
        if self.type.is_pyobject:
            result = code.get_py_string_const(self.value)
        elif self.type.is_const:
            result = code.get_string_const(self.value)
        else:
            literal = self.value.as_c_string_literal()
            result = typecast(self.type, PyrexTypes.c_void_ptr_type, literal)
        self.result_code = result

    def get_constant_c_result_code(self):
        if False:
            for i in range(10):
                print('nop')
        return None

    def calculate_result_code(self):
        if False:
            print('Hello World!')
        return self.result_code

class UnicodeNode(ConstNode):
    is_string_literal = True
    bytes_value = None
    type = unicode_type

    def calculate_constant_result(self):
        if False:
            i = 10
            return i + 15
        self.constant_result = self.value

    def analyse_as_type(self, env):
        if False:
            return 10
        return _analyse_name_as_type(self.value, self.pos, env)

    def as_sliced_node(self, start, stop, step=None):
        if False:
            i = 10
            return i + 15
        if StringEncoding.string_contains_surrogates(self.value[:stop]):
            return None
        value = StringEncoding.EncodedString(self.value[start:stop:step])
        value.encoding = self.value.encoding
        if self.bytes_value is not None:
            bytes_value = StringEncoding.bytes_literal(self.bytes_value[start:stop:step], self.bytes_value.encoding)
        else:
            bytes_value = None
        return UnicodeNode(self.pos, value=value, bytes_value=bytes_value, constant_result=value)

    def coerce_to(self, dst_type, env):
        if False:
            i = 10
            return i + 15
        if dst_type is self.type:
            pass
        elif dst_type.is_unicode_char:
            if not self.can_coerce_to_char_literal():
                error(self.pos, 'Only single-character Unicode string literals or surrogate pairs can be coerced into Py_UCS4/Py_UNICODE.')
                return self
            int_value = ord(self.value)
            return IntNode(self.pos, type=dst_type, value=str(int_value), constant_result=int_value)
        elif not dst_type.is_pyobject:
            if dst_type.is_string and self.bytes_value is not None:
                return BytesNode(self.pos, value=self.bytes_value).coerce_to(dst_type, env)
            if dst_type.is_pyunicode_ptr:
                return UnicodeNode(self.pos, value=self.value, type=dst_type)
            error(self.pos, 'Unicode literals do not support coercion to C types other than Py_UNICODE/Py_UCS4 (for characters) or Py_UNICODE* (for strings).')
        elif dst_type not in (py_object_type, Builtin.basestring_type):
            self.check_for_coercion_error(dst_type, env, fail=True)
        return self

    def can_coerce_to_char_literal(self):
        if False:
            print('Hello World!')
        return len(self.value) == 1

    def coerce_to_boolean(self, env):
        if False:
            return 10
        bool_value = bool(self.value)
        return BoolNode(self.pos, value=bool_value, constant_result=bool_value)

    def contains_surrogates(self):
        if False:
            while True:
                i = 10
        return StringEncoding.string_contains_surrogates(self.value)

    def generate_evaluation_code(self, code):
        if False:
            print('Hello World!')
        if self.type.is_pyobject:
            if StringEncoding.string_contains_lone_surrogates(self.value):
                self.result_code = code.get_py_const(py_object_type, 'ustring')
                data_cname = code.get_string_const(StringEncoding.BytesLiteral(self.value.encode('unicode_escape')))
                const_code = code.get_cached_constants_writer(self.result_code)
                if const_code is None:
                    return
                const_code.mark_pos(self.pos)
                const_code.putln('%s = PyUnicode_DecodeUnicodeEscape(%s, sizeof(%s) - 1, NULL); %s' % (self.result_code, data_cname, data_cname, const_code.error_goto_if_null(self.result_code, self.pos)))
                const_code.put_error_if_neg(self.pos, '__Pyx_PyUnicode_READY(%s)' % self.result_code)
            else:
                self.result_code = code.get_py_string_const(self.value)
        else:
            self.result_code = code.get_pyunicode_ptr_const(self.value)

    def calculate_result_code(self):
        if False:
            i = 10
            return i + 15
        return self.result_code

    def compile_time_value(self, env):
        if False:
            i = 10
            return i + 15
        return self.value

class StringNode(PyConstNode):
    type = str_type
    is_string_literal = True
    is_identifier = None
    unicode_value = None

    def calculate_constant_result(self):
        if False:
            print('Hello World!')
        if self.unicode_value is not None:
            self.constant_result = self.unicode_value

    def analyse_as_type(self, env):
        if False:
            print('Hello World!')
        return _analyse_name_as_type(self.unicode_value or self.value.decode('ISO8859-1'), self.pos, env)

    def as_sliced_node(self, start, stop, step=None):
        if False:
            return 10
        value = type(self.value)(self.value[start:stop:step])
        value.encoding = self.value.encoding
        if self.unicode_value is not None:
            if StringEncoding.string_contains_surrogates(self.unicode_value[:stop]):
                return None
            unicode_value = StringEncoding.EncodedString(self.unicode_value[start:stop:step])
        else:
            unicode_value = None
        return StringNode(self.pos, value=value, unicode_value=unicode_value, constant_result=value, is_identifier=self.is_identifier)

    def coerce_to(self, dst_type, env):
        if False:
            while True:
                i = 10
        if dst_type is not py_object_type and (not str_type.subtype_of(dst_type)):
            if not dst_type.is_pyobject:
                return BytesNode(self.pos, value=self.value).coerce_to(dst_type, env)
            if dst_type is not Builtin.basestring_type:
                self.check_for_coercion_error(dst_type, env, fail=True)
        return self

    def can_coerce_to_char_literal(self):
        if False:
            return 10
        return not self.is_identifier and len(self.value) == 1

    def generate_evaluation_code(self, code):
        if False:
            i = 10
            return i + 15
        self.result_code = code.get_py_string_const(self.value, identifier=self.is_identifier, is_str=True, unicode_value=self.unicode_value)

    def get_constant_c_result_code(self):
        if False:
            for i in range(10):
                print('nop')
        return None

    def calculate_result_code(self):
        if False:
            return 10
        return self.result_code

    def compile_time_value(self, env):
        if False:
            return 10
        if self.value.is_unicode:
            return self.value
        if not IS_PYTHON3:
            return self.value.byteencode()
        if self.unicode_value is not None:
            return self.unicode_value
        return self.value.decode('iso8859-1')

class IdentifierStringNode(StringNode):
    is_identifier = True

class ImagNode(AtomicExprNode):
    type = PyrexTypes.c_double_complex_type

    def calculate_constant_result(self):
        if False:
            return 10
        self.constant_result = complex(0.0, float(self.value))

    def compile_time_value(self, denv):
        if False:
            print('Hello World!')
        return complex(0.0, float(self.value))

    def analyse_types(self, env):
        if False:
            return 10
        self.type.create_declaration_utility_code(env)
        return self

    def may_be_none(self):
        if False:
            i = 10
            return i + 15
        return False

    def coerce_to(self, dst_type, env):
        if False:
            while True:
                i = 10
        if self.type is dst_type:
            return self
        node = ImagNode(self.pos, value=self.value)
        if dst_type.is_pyobject:
            node.is_temp = 1
            node.type = Builtin.complex_type
        return AtomicExprNode.coerce_to(node, dst_type, env)
    gil_message = 'Constructing complex number'

    def calculate_result_code(self):
        if False:
            return 10
        if self.type.is_pyobject:
            return self.result()
        else:
            return '%s(0, %r)' % (self.type.from_parts, float(self.value))

    def generate_result_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        if self.type.is_pyobject:
            code.putln('%s = PyComplex_FromDoubles(0.0, %r); %s' % (self.result(), float(self.value), code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)

class NewExprNode(AtomicExprNode):
    type = None

    def infer_type(self, env):
        if False:
            for i in range(10):
                print('nop')
        type = self.cppclass.analyse_as_type(env)
        if type is None or not type.is_cpp_class:
            error(self.pos, 'new operator can only be applied to a C++ class')
            self.type = error_type
            return
        self.cpp_check(env)
        constructor = type.get_constructor(self.pos)
        self.class_type = type
        self.entry = constructor
        self.type = constructor.type
        return self.type

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        if self.type is None:
            self.infer_type(env)
        return self

    def may_be_none(self):
        if False:
            while True:
                i = 10
        return False

    def generate_result_code(self, code):
        if False:
            return 10
        pass

    def calculate_result_code(self):
        if False:
            for i in range(10):
                print('nop')
        return 'new ' + self.class_type.empty_declaration_code()

class NameNode(AtomicExprNode):
    is_name = True
    is_cython_module = False
    cython_attribute = None
    lhs_of_first_assignment = False
    is_used_as_rvalue = 0
    entry = None
    type_entry = None
    cf_maybe_null = True
    cf_is_null = False
    allow_null = False
    nogil = False
    inferred_type = None

    def as_cython_attribute(self):
        if False:
            i = 10
            return i + 15
        return self.cython_attribute

    def type_dependencies(self, env):
        if False:
            return 10
        if self.entry is None:
            self.entry = env.lookup(self.name)
        if self.entry is not None and self.entry.type.is_unspecified:
            return (self,)
        else:
            return ()

    def infer_type(self, env):
        if False:
            while True:
                i = 10
        if self.entry is None:
            self.entry = env.lookup(self.name)
        if self.entry is None or self.entry.type is unspecified_type:
            if self.inferred_type is not None:
                return self.inferred_type
            return py_object_type
        elif (self.entry.type.is_extension_type or self.entry.type.is_builtin_type) and self.name == self.entry.type.name:
            return type_type
        elif self.entry.type.is_cfunction:
            if self.entry.scope.is_builtin_scope:
                return py_object_type
            else:
                return PyrexTypes.CPtrType(self.entry.type)
        else:
            if self.entry.type.is_pyobject and self.inferred_type:
                if not (self.inferred_type.is_int and self.entry.might_overflow):
                    return self.inferred_type
            return self.entry.type

    def compile_time_value(self, denv):
        if False:
            while True:
                i = 10
        try:
            return denv.lookup(self.name)
        except KeyError:
            error(self.pos, "Compile-time name '%s' not defined" % self.name)

    def get_constant_c_result_code(self):
        if False:
            return 10
        if not self.entry or self.entry.type.is_pyobject:
            return None
        return self.entry.cname

    def coerce_to(self, dst_type, env):
        if False:
            while True:
                i = 10
        if dst_type is py_object_type:
            entry = self.entry
            if entry and entry.is_cfunction:
                var_entry = entry.as_variable
                if var_entry:
                    if var_entry.is_builtin and var_entry.is_const:
                        var_entry = env.declare_builtin(var_entry.name, self.pos)
                    node = NameNode(self.pos, name=self.name)
                    node.entry = var_entry
                    node.analyse_rvalue_entry(env)
                    return node
        return super(NameNode, self).coerce_to(dst_type, env)

    def declare_from_annotation(self, env, as_target=False):
        if False:
            print('Hello World!')
        "Implements PEP 526 annotation typing in a fairly relaxed way.\n\n        Annotations are ignored for global variables.\n        All other annotations are stored on the entry in the symbol table.\n        String literals are allowed and not evaluated.\n        The ambiguous Python types 'int' and 'long' are not evaluated - the 'cython.int' form must be used instead.\n        "
        name = self.name
        annotation = self.annotation
        entry = self.entry or env.lookup_here(name)
        if not entry:
            if env.is_module_scope:
                return
            modifiers = ()
            if annotation.expr.is_string_literal or not env.directives['annotation_typing']:
                atype = None
            elif env.is_py_class_scope:
                atype = py_object_type
            else:
                (modifiers, atype) = annotation.analyse_type_annotation(env)
            if atype is None:
                atype = unspecified_type if as_target and env.directives['infer_types'] != False else py_object_type
            elif atype.is_fused and env.fused_to_specific:
                try:
                    atype = atype.specialize(env.fused_to_specific)
                except CannotSpecialize:
                    error(self.pos, "'%s' cannot be specialized since its type is not a fused argument to this function" % self.name)
                    atype = error_type
            visibility = 'private'
            if env.is_c_dataclass_scope:
                is_frozen = env.is_c_dataclass_scope == 'frozen'
                if atype.is_pyobject or atype.can_coerce_to_pyobject(env):
                    visibility = 'readonly' if is_frozen else 'public'
            if as_target and env.is_c_class_scope and (not (atype.is_pyobject or atype.is_error)):
                atype = py_object_type
                warning(annotation.pos, 'Annotation ignored since class-level attributes must be Python objects. Were you trying to set up an instance attribute?', 2)
            entry = self.entry = env.declare_var(name, atype, self.pos, is_cdef=not as_target, visibility=visibility, pytyping_modifiers=modifiers)
        if annotation and (not entry.annotation):
            entry.annotation = annotation

    def analyse_as_module(self, env):
        if False:
            i = 10
            return i + 15
        entry = self.entry
        if not entry:
            entry = env.lookup(self.name)
        if entry and entry.as_module:
            return entry.as_module
        if entry and entry.known_standard_library_import:
            scope = Builtin.get_known_standard_library_module_scope(entry.known_standard_library_import)
            if scope and scope.is_module_scope:
                return scope
        return None

    def analyse_as_type(self, env):
        if False:
            while True:
                i = 10
        type = None
        if self.cython_attribute:
            type = PyrexTypes.parse_basic_type(self.cython_attribute)
        elif env.in_c_type_context:
            type = PyrexTypes.parse_basic_type(self.name)
        if type:
            return type
        entry = self.entry
        if not entry:
            entry = env.lookup(self.name)
        if entry and (not entry.is_type) and entry.known_standard_library_import:
            entry = Builtin.get_known_standard_library_entry(entry.known_standard_library_import)
        if entry and entry.is_type:
            type = entry.type
            if not env.in_c_type_context and type is Builtin.long_type:
                warning(self.pos, "Found Python 2.x type 'long' in a Python annotation. Did you mean to use 'cython.long'?")
                type = py_object_type
            elif type.is_pyobject and type.equivalent_type:
                type = type.equivalent_type
            elif type is Builtin.int_type and env.global_scope().context.language_level == 2:
                type = py_object_type
            return type
        if self.name == 'object':
            return py_object_type
        if not env.in_c_type_context and PyrexTypes.parse_basic_type(self.name):
            warning(self.pos, "Found C type '%s' in a Python annotation. Did you mean to use 'cython.%s'?" % (self.name, self.name))
        return None

    def analyse_as_extension_type(self, env):
        if False:
            while True:
                i = 10
        entry = self.entry
        if not entry:
            entry = env.lookup(self.name)
        if entry and entry.is_type:
            if entry.type.is_extension_type or entry.type.is_builtin_type:
                return entry.type
        return None

    def analyse_target_declaration(self, env):
        if False:
            i = 10
            return i + 15
        return self._analyse_target_declaration(env, is_assignment_expression=False)

    def analyse_assignment_expression_target_declaration(self, env):
        if False:
            i = 10
            return i + 15
        return self._analyse_target_declaration(env, is_assignment_expression=True)

    def _analyse_target_declaration(self, env, is_assignment_expression):
        if False:
            while True:
                i = 10
        self.is_target = True
        if not self.entry:
            if is_assignment_expression:
                self.entry = env.lookup_assignment_expression_target(self.name)
            else:
                self.entry = env.lookup_here(self.name)
        if self.entry:
            self.entry.known_standard_library_import = ''
        if not self.entry and self.annotation is not None:
            is_dataclass = env.is_c_dataclass_scope
            self.declare_from_annotation(env, as_target=not is_dataclass)
        elif self.entry and self.entry.is_inherited and self.annotation and env.is_c_dataclass_scope:
            error(self.pos, 'Cannot redeclare inherited fields in Cython dataclasses')
        if not self.entry:
            if env.directives['warn.undeclared']:
                warning(self.pos, "implicit declaration of '%s'" % self.name, 1)
            if env.directives['infer_types'] != False:
                type = unspecified_type
            else:
                type = py_object_type
            if is_assignment_expression:
                self.entry = env.declare_assignment_expression_target(self.name, type, self.pos)
            else:
                self.entry = env.declare_var(self.name, type, self.pos)
        if self.entry.is_declared_generic:
            self.result_ctype = py_object_type
        if self.entry.as_module:
            self.entry.is_variable = 1

    def analyse_types(self, env):
        if False:
            print('Hello World!')
        self.initialized_check = env.directives['initializedcheck']
        entry = self.entry
        if entry is None:
            entry = env.lookup(self.name)
            if not entry:
                entry = env.declare_builtin(self.name, self.pos)
                if entry and entry.is_builtin and entry.is_const:
                    self.is_literal = True
            if not entry:
                self.type = PyrexTypes.error_type
                return self
            self.entry = entry
        entry.used = 1
        if entry.type.is_buffer:
            from . import Buffer
            Buffer.used_buffer_aux_vars(entry)
        self.analyse_rvalue_entry(env)
        return self

    def analyse_target_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        self.analyse_entry(env, is_target=True)
        entry = self.entry
        if entry.is_cfunction and entry.as_variable:
            if (entry.is_overridable or entry.type.is_overridable) or (not self.is_lvalue() and entry.fused_cfunction):
                entry = self.entry = entry.as_variable
                self.type = entry.type
        if self.type.is_const:
            error(self.pos, "Assignment to const '%s'" % self.name)
        if not self.is_lvalue():
            error(self.pos, "Assignment to non-lvalue '%s'" % self.name)
            self.type = PyrexTypes.error_type
        entry.used = 1
        if entry.type.is_buffer:
            from . import Buffer
            Buffer.used_buffer_aux_vars(entry)
        return self

    def analyse_rvalue_entry(self, env):
        if False:
            return 10
        self.analyse_entry(env)
        entry = self.entry
        if entry.is_declared_generic:
            self.result_ctype = py_object_type
        if entry.is_pyglobal or entry.is_builtin:
            if entry.is_builtin and entry.is_const:
                self.is_temp = 0
            else:
                self.is_temp = 1
            self.is_used_as_rvalue = 1
        elif entry.type.is_memoryviewslice:
            self.is_temp = False
            self.is_used_as_rvalue = True
            self.use_managed_ref = True
        return self

    def nogil_check(self, env):
        if False:
            i = 10
            return i + 15
        self.nogil = True
        if self.is_used_as_rvalue:
            entry = self.entry
            if entry.is_builtin:
                if not entry.is_const:
                    self.gil_error()
            elif entry.is_pyglobal:
                self.gil_error()
    gil_message = 'Accessing Python global or builtin'

    def analyse_entry(self, env, is_target=False):
        if False:
            print('Hello World!')
        self.check_identifier_kind()
        entry = self.entry
        type = entry.type
        if not is_target and type.is_pyobject and self.inferred_type and self.inferred_type.is_builtin_type:
            type = self.inferred_type
        self.type = type

    def check_identifier_kind(self):
        if False:
            for i in range(10):
                print('nop')
        entry = self.entry
        if entry.is_type and entry.type.is_extension_type:
            self.type_entry = entry
        if entry.is_type and (entry.type.is_enum or entry.type.is_cpp_enum) and entry.create_wrapper:
            py_entry = Symtab.Entry(self.name, None, py_object_type)
            py_entry.is_pyglobal = True
            py_entry.scope = self.entry.scope
            self.entry = py_entry
        elif not (entry.is_const or entry.is_variable or entry.is_builtin or entry.is_cfunction or entry.is_cpp_class):
            if self.entry.as_variable:
                self.entry = self.entry.as_variable
            elif not self.is_cython_module:
                error(self.pos, "'%s' is not a constant, variable or function identifier" % self.name)

    def is_cimported_module_without_shadow(self, env):
        if False:
            while True:
                i = 10
        if self.is_cython_module or self.cython_attribute:
            return False
        entry = self.entry or env.lookup(self.name)
        return entry.as_module and (not entry.is_variable)

    def is_simple(self):
        if False:
            while True:
                i = 10
        return 1

    def may_be_none(self):
        if False:
            i = 10
            return i + 15
        if self.cf_state and self.type and (self.type.is_pyobject or self.type.is_memoryviewslice):
            if getattr(self, '_none_checking', False):
                return False
            self._none_checking = True
            may_be_none = False
            for assignment in self.cf_state:
                if assignment.rhs.may_be_none():
                    may_be_none = True
                    break
            del self._none_checking
            return may_be_none
        return super(NameNode, self).may_be_none()

    def nonlocally_immutable(self):
        if False:
            return 10
        if ExprNode.nonlocally_immutable(self):
            return True
        entry = self.entry
        if not entry or entry.in_closure:
            return False
        return entry.is_local or entry.is_arg or entry.is_builtin or entry.is_readonly

    def calculate_target_results(self, env):
        if False:
            print('Hello World!')
        pass

    def check_const(self):
        if False:
            print('Hello World!')
        entry = self.entry
        if entry is not None and (not (entry.is_const or entry.is_cfunction or entry.is_builtin or entry.type.is_const)):
            self.not_const()
            return False
        return True

    def check_const_addr(self):
        if False:
            print('Hello World!')
        entry = self.entry
        if not (entry.is_cglobal or entry.is_cfunction or entry.is_builtin):
            self.addr_not_const()
            return False
        return True

    def is_lvalue(self):
        if False:
            print('Hello World!')
        return self.entry.is_variable and (not self.entry.is_readonly) or (self.entry.is_cfunction and self.entry.is_overridable)

    def is_addressable(self):
        if False:
            print('Hello World!')
        return self.entry.is_variable and (not self.type.is_memoryviewslice)

    def is_ephemeral(self):
        if False:
            return 10
        return 0

    def calculate_result_code(self):
        if False:
            print('Hello World!')
        entry = self.entry
        if not entry:
            return '<error>'
        if self.entry.is_cpp_optional and (not self.is_target):
            return '(*%s)' % entry.cname
        return entry.cname

    def generate_result_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        entry = self.entry
        if entry is None:
            return
        if entry.utility_code:
            code.globalstate.use_utility_code(entry.utility_code)
        if entry.is_builtin and entry.is_const:
            return
        elif entry.is_pyclass_attr:
            assert entry.type.is_pyobject, 'Python global or builtin not a Python object'
            interned_cname = code.intern_identifier(self.entry.name)
            if entry.is_builtin:
                namespace = Naming.builtins_cname
            else:
                namespace = entry.scope.namespace_cname
            if not self.cf_is_null:
                code.putln('%s = PyObject_GetItem(%s, %s);' % (self.result(), namespace, interned_cname))
                code.putln('if (unlikely(!%s)) {' % self.result())
                code.putln('PyErr_Clear();')
            code.globalstate.use_utility_code(UtilityCode.load_cached('GetModuleGlobalName', 'ObjectHandling.c'))
            code.putln('__Pyx_GetModuleGlobalName(%s, %s);' % (self.result(), interned_cname))
            if not self.cf_is_null:
                code.putln('}')
            code.putln(code.error_goto_if_null(self.result(), self.pos))
            self.generate_gotref(code)
        elif entry.is_builtin and (not entry.scope.is_module_scope):
            assert entry.type.is_pyobject, 'Python global or builtin not a Python object'
            interned_cname = code.intern_identifier(self.entry.name)
            code.globalstate.use_utility_code(UtilityCode.load_cached('GetBuiltinName', 'ObjectHandling.c'))
            code.putln('%s = __Pyx_GetBuiltinName(%s); %s' % (self.result(), interned_cname, code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
        elif entry.is_pyglobal or (entry.is_builtin and entry.scope.is_module_scope):
            assert entry.type.is_pyobject, 'Python global or builtin not a Python object'
            interned_cname = code.intern_identifier(self.entry.name)
            if entry.scope.is_module_scope:
                code.globalstate.use_utility_code(UtilityCode.load_cached('GetModuleGlobalName', 'ObjectHandling.c'))
                code.putln('__Pyx_GetModuleGlobalName(%s, %s); %s' % (self.result(), interned_cname, code.error_goto_if_null(self.result(), self.pos)))
            else:
                code.globalstate.use_utility_code(UtilityCode.load_cached('GetNameInClass', 'ObjectHandling.c'))
                code.putln('__Pyx_GetNameInClass(%s, %s, %s); %s' % (self.result(), entry.scope.namespace_cname, interned_cname, code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
        elif entry.is_local or entry.in_closure or entry.from_closure or entry.type.is_memoryviewslice:
            raise_unbound = (self.cf_maybe_null or self.cf_is_null) and (not self.allow_null)
            memslice_check = entry.type.is_memoryviewslice and self.initialized_check
            optional_cpp_check = entry.is_cpp_optional and self.initialized_check
            if optional_cpp_check:
                unbound_check_code = entry.type.cpp_optional_check_for_null_code(entry.cname)
            else:
                unbound_check_code = entry.type.check_for_null_code(entry.cname)
            if unbound_check_code and raise_unbound and (entry.type.is_pyobject or memslice_check or optional_cpp_check):
                code.put_error_if_unbound(self.pos, entry, self.in_nogil_context, unbound_check_code=unbound_check_code)
        elif entry.is_cglobal and entry.is_cpp_optional and self.initialized_check:
            unbound_check_code = entry.type.cpp_optional_check_for_null_code(entry.cname)
            code.put_error_if_unbound(self.pos, entry, unbound_check_code=unbound_check_code)

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False, exception_check=None, exception_value=None):
        if False:
            for i in range(10):
                print('nop')
        entry = self.entry
        if entry is None:
            return
        if self.entry.type.is_ptr and isinstance(rhs, ListNode) and (not self.lhs_of_first_assignment) and (not rhs.in_module_scope):
            error(self.pos, 'Literal list must be assigned to pointer at time of declaration')
        if entry.is_pyglobal:
            assert entry.type.is_pyobject, 'Python global or builtin not a Python object'
            interned_cname = code.intern_identifier(self.entry.name)
            namespace = self.entry.scope.namespace_cname
            if entry.is_member:
                setter = '__Pyx_SetItemOnTypeDict'
            elif entry.scope.is_module_scope:
                setter = 'PyDict_SetItem'
                namespace = Naming.moddict_cname
            elif entry.is_pyclass_attr:
                n = 'SetNewInClass' if self.name == '__new__' else 'SetNameInClass'
                code.globalstate.use_utility_code(UtilityCode.load_cached(n, 'ObjectHandling.c'))
                setter = '__Pyx_' + n
            else:
                assert False, repr(entry)
            code.put_error_if_neg(self.pos, '%s(%s, %s, %s)' % (setter, namespace, interned_cname, rhs.py_result()))
            if debug_disposal_code:
                print('NameNode.generate_assignment_code:')
                print('...generating disposal code for %s' % rhs)
            rhs.generate_disposal_code(code)
            rhs.free_temps(code)
            if entry.is_member:
                code.putln('PyType_Modified(%s);' % entry.scope.parent_type.typeptr_cname)
        else:
            if self.type.is_memoryviewslice:
                self.generate_acquire_memoryviewslice(rhs, code)
            elif self.type.is_buffer:
                self.generate_acquire_buffer(rhs, code)
            assigned = False
            if self.type.is_pyobject:
                if self.use_managed_ref:
                    rhs.make_owned_reference(code)
                    is_external_ref = entry.is_cglobal or self.entry.in_closure or self.entry.from_closure
                    if is_external_ref:
                        self.generate_gotref(code, handle_null=True)
                    assigned = True
                    if entry.is_cglobal:
                        self.generate_decref_set(code, rhs.result_as(self.ctype()))
                    elif not self.cf_is_null:
                        if self.cf_maybe_null:
                            self.generate_xdecref_set(code, rhs.result_as(self.ctype()))
                        else:
                            self.generate_decref_set(code, rhs.result_as(self.ctype()))
                    else:
                        assigned = False
                    if is_external_ref:
                        rhs.generate_giveref(code)
            if not self.type.is_memoryviewslice:
                if not assigned:
                    if overloaded_assignment:
                        result = rhs.move_result_rhs()
                        if exception_check == '+':
                            translate_cpp_exception(code, self.pos, '%s = %s;' % (self.result(), result), self.result() if self.type.is_pyobject else None, exception_value, self.in_nogil_context)
                        else:
                            code.putln('%s = %s;' % (self.result(), result))
                    else:
                        result = rhs.move_result_rhs_as(self.ctype())
                        if is_pythran_expr(self.type):
                            code.putln('new (&%s) decltype(%s){%s};' % (self.result(), self.result(), result))
                        elif result != self.result():
                            code.putln('%s = %s;' % (self.result(), result))
                if debug_disposal_code:
                    print('NameNode.generate_assignment_code:')
                    print('...generating post-assignment code for %s' % rhs)
                rhs.generate_post_assignment_code(code)
            elif rhs.result_in_temp():
                rhs.generate_post_assignment_code(code)
            rhs.free_temps(code)

    def generate_acquire_memoryviewslice(self, rhs, code):
        if False:
            i = 10
            return i + 15
        '\n        Slices, coercions from objects, return values etc are new references.\n        We have a borrowed reference in case of dst = src\n        '
        from . import MemoryView
        MemoryView.put_acquire_memoryviewslice(lhs_cname=self.result(), lhs_type=self.type, lhs_pos=self.pos, rhs=rhs, code=code, have_gil=not self.in_nogil_context, first_assignment=self.cf_is_null)

    def generate_acquire_buffer(self, rhs, code):
        if False:
            print('Hello World!')
        pretty_rhs = isinstance(rhs, NameNode) or rhs.is_temp
        if pretty_rhs:
            rhstmp = rhs.result_as(self.ctype())
        else:
            rhstmp = code.funcstate.allocate_temp(self.entry.type, manage_ref=False)
            code.putln('%s = %s;' % (rhstmp, rhs.result_as(self.ctype())))
        from . import Buffer
        Buffer.put_assign_to_buffer(self.result(), rhstmp, self.entry, is_initialized=not self.lhs_of_first_assignment, pos=self.pos, code=code)
        if not pretty_rhs:
            code.putln('%s = 0;' % rhstmp)
            code.funcstate.release_temp(rhstmp)

    def generate_deletion_code(self, code, ignore_nonexisting=False):
        if False:
            while True:
                i = 10
        if self.entry is None:
            return
        elif self.entry.is_pyclass_attr:
            namespace = self.entry.scope.namespace_cname
            interned_cname = code.intern_identifier(self.entry.name)
            if ignore_nonexisting:
                key_error_code = 'PyErr_Clear(); else'
            else:
                key_error_code = '{ PyErr_Clear(); PyErr_Format(PyExc_NameError, "name \'%%s\' is not defined", "%s"); }' % self.entry.name
            code.putln('if (unlikely(PyObject_DelItem(%s, %s) < 0)) { if (likely(PyErr_ExceptionMatches(PyExc_KeyError))) %s %s }' % (namespace, interned_cname, key_error_code, code.error_goto(self.pos)))
        elif self.entry.is_pyglobal:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectSetAttrStr', 'ObjectHandling.c'))
            interned_cname = code.intern_identifier(self.entry.name)
            del_code = '__Pyx_PyObject_DelAttrStr(%s, %s)' % (Naming.module_cname, interned_cname)
            if ignore_nonexisting:
                code.putln('if (unlikely(%s < 0)) { if (likely(PyErr_ExceptionMatches(PyExc_AttributeError))) PyErr_Clear(); else %s }' % (del_code, code.error_goto(self.pos)))
            else:
                code.put_error_if_neg(self.pos, del_code)
        elif self.entry.type.is_pyobject or self.entry.type.is_memoryviewslice:
            if not self.cf_is_null:
                if self.cf_maybe_null and (not ignore_nonexisting):
                    code.put_error_if_unbound(self.pos, self.entry)
                if self.entry.in_closure:
                    self.generate_gotref(code, handle_null=True, maybe_null_extra_check=ignore_nonexisting)
                if ignore_nonexisting and self.cf_maybe_null:
                    code.put_xdecref_clear(self.result(), self.ctype(), have_gil=not self.nogil)
                else:
                    code.put_decref_clear(self.result(), self.ctype(), have_gil=not self.nogil)
        else:
            error(self.pos, 'Deletion of C names not supported')

    def annotate(self, code):
        if False:
            while True:
                i = 10
        if getattr(self, 'is_called', False):
            pos = (self.pos[0], self.pos[1], self.pos[2] - len(self.name) - 1)
            if self.type.is_pyobject:
                (style, text) = ('py_call', 'python function (%s)')
            else:
                (style, text) = ('c_call', 'c function (%s)')
            code.annotate(pos, AnnotationItem(style, text % self.type, size=len(self.name)))

    def get_known_standard_library_import(self):
        if False:
            print('Hello World!')
        if self.entry:
            return self.entry.known_standard_library_import
        return None

class BackquoteNode(ExprNode):
    type = py_object_type
    subexprs = ['arg']

    def analyse_types(self, env):
        if False:
            print('Hello World!')
        self.arg = self.arg.analyse_types(env)
        self.arg = self.arg.coerce_to_pyobject(env)
        self.is_temp = 1
        return self
    gil_message = 'Backquote expression'

    def calculate_constant_result(self):
        if False:
            return 10
        self.constant_result = repr(self.arg.constant_result)

    def generate_result_code(self, code):
        if False:
            return 10
        code.putln('%s = PyObject_Repr(%s); %s' % (self.result(), self.arg.py_result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

class ImportNode(ExprNode):
    type = py_object_type
    module_names = None
    get_top_level_module = False
    is_temp = True
    subexprs = ['module_name', 'name_list', 'module_names']

    def analyse_types(self, env):
        if False:
            print('Hello World!')
        if self.level is None:
            if env.global_scope().parent_module and (env.directives['py2_import'] or Future.absolute_import not in env.global_scope().context.future_directives):
                self.level = -1
            else:
                self.level = 0
        module_name = self.module_name.analyse_types(env)
        self.module_name = module_name.coerce_to_pyobject(env)
        assert self.module_name.is_string_literal
        if self.name_list:
            name_list = self.name_list.analyse_types(env)
            self.name_list = name_list.coerce_to_pyobject(env)
        elif '.' in self.module_name.value:
            self.module_names = TupleNode(self.module_name.pos, args=[IdentifierStringNode(self.module_name.pos, value=part, constant_result=part) for part in map(StringEncoding.EncodedString, self.module_name.value.split('.'))]).analyse_types(env)
        return self
    gil_message = 'Python import'

    def generate_result_code(self, code):
        if False:
            return 10
        assert self.module_name.is_string_literal
        module_name = self.module_name.value
        if self.level <= 0 and (not self.name_list) and (not self.get_top_level_module):
            if self.module_names:
                assert self.module_names.is_literal
            if self.level == 0:
                utility_code = UtilityCode.load_cached('ImportDottedModule', 'ImportExport.c')
                helper_func = '__Pyx_ImportDottedModule'
            else:
                utility_code = UtilityCode.load_cached('ImportDottedModuleRelFirst', 'ImportExport.c')
                helper_func = '__Pyx_ImportDottedModuleRelFirst'
            code.globalstate.use_utility_code(utility_code)
            import_code = '%s(%s, %s)' % (helper_func, self.module_name.py_result(), self.module_names.py_result() if self.module_names else 'NULL')
        else:
            code.globalstate.use_utility_code(UtilityCode.load_cached('Import', 'ImportExport.c'))
            import_code = '__Pyx_Import(%s, %s, %d)' % (self.module_name.py_result(), self.name_list.py_result() if self.name_list else '0', self.level)
        if self.level <= 0 and module_name in utility_code_for_imports:
            (helper_func, code_name, code_file) = utility_code_for_imports[module_name]
            code.globalstate.use_utility_code(UtilityCode.load_cached(code_name, code_file))
            import_code = '%s(%s)' % (helper_func, import_code)
        code.putln('%s = %s; %s' % (self.result(), import_code, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

    def get_known_standard_library_import(self):
        if False:
            while True:
                i = 10
        return self.module_name.value

class ScopedExprNode(ExprNode):
    subexprs = []
    expr_scope = None
    has_local_scope = True

    def init_scope(self, outer_scope, expr_scope=None):
        if False:
            for i in range(10):
                print('nop')
        if expr_scope is not None:
            self.expr_scope = expr_scope
        elif self.has_local_scope:
            self.expr_scope = Symtab.ComprehensionScope(outer_scope)
        elif not self.expr_scope:
            self.expr_scope = None

    def analyse_declarations(self, env):
        if False:
            return 10
        self.init_scope(env)

    def analyse_scoped_declarations(self, env):
        if False:
            while True:
                i = 10
        pass

    def analyse_types(self, env):
        if False:
            return 10
        return self

    def analyse_scoped_expressions(self, env):
        if False:
            while True:
                i = 10
        return self

    def generate_evaluation_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        generate_inner_evaluation_code = super(ScopedExprNode, self).generate_evaluation_code
        if not self.has_local_scope or not self.expr_scope.var_entries:
            generate_inner_evaluation_code(code)
            return
        code.putln('{ /* enter inner scope */')
        py_entries = []
        for (_, entry) in sorted((item for item in self.expr_scope.entries.items() if item[0])):
            if not entry.in_closure:
                if entry.type.is_pyobject and entry.used:
                    py_entries.append(entry)
        if not py_entries:
            generate_inner_evaluation_code(code)
            code.putln('} /* exit inner scope */')
            return
        old_loop_labels = code.new_loop_labels()
        old_error_label = code.new_error_label()
        generate_inner_evaluation_code(code)
        self._generate_vars_cleanup(code, py_entries)
        exit_scope = code.new_label('exit_scope')
        code.put_goto(exit_scope)
        for (label, old_label) in [(code.error_label, old_error_label)] + list(zip(code.get_loop_labels(), old_loop_labels)):
            if code.label_used(label):
                code.put_label(label)
                self._generate_vars_cleanup(code, py_entries)
                code.put_goto(old_label)
        code.put_label(exit_scope)
        code.putln('} /* exit inner scope */')
        code.set_loop_labels(old_loop_labels)
        code.error_label = old_error_label

    def _generate_vars_cleanup(self, code, py_entries):
        if False:
            for i in range(10):
                print('nop')
        for entry in py_entries:
            if entry.is_cglobal:
                code.put_var_gotref(entry)
                code.put_var_decref_set(entry, 'Py_None')
            else:
                code.put_var_xdecref_clear(entry)

class IteratorNode(ScopedExprNode):
    type = py_object_type
    iter_func_ptr = None
    counter_cname = None
    reversed = False
    is_async = False
    has_local_scope = False
    subexprs = ['sequence']

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        if self.expr_scope:
            env = self.expr_scope
        self.sequence = self.sequence.analyse_types(env)
        if (self.sequence.type.is_array or self.sequence.type.is_ptr) and (not self.sequence.type.is_string):
            self.type = self.sequence.type
        elif self.sequence.type.is_cpp_class:
            return CppIteratorNode(self.pos, sequence=self.sequence).analyse_types(env)
        elif self.is_reversed_cpp_iteration():
            sequence = self.sequence.arg_tuple.args[0].arg
            return CppIteratorNode(self.pos, sequence=sequence, reversed=True).analyse_types(env)
        else:
            self.sequence = self.sequence.coerce_to_pyobject(env)
            if self.sequence.type in (list_type, tuple_type):
                self.sequence = self.sequence.as_none_safe_node("'NoneType' object is not iterable")
        self.is_temp = 1
        return self
    gil_message = 'Iterating over Python object'
    _func_iternext_type = PyrexTypes.CPtrType(PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('it', PyrexTypes.py_object_type, None)]))

    def is_reversed_cpp_iteration(self):
        if False:
            while True:
                i = 10
        "\n        Returns True if the 'reversed' function is applied to a C++ iterable.\n\n        This supports C++ classes with reverse_iterator implemented.\n        "
        if not (isinstance(self.sequence, SimpleCallNode) and self.sequence.arg_tuple and (len(self.sequence.arg_tuple.args) == 1)):
            return False
        func = self.sequence.function
        if func.is_name and func.name == 'reversed':
            if not func.entry.is_builtin:
                return False
            arg = self.sequence.arg_tuple.args[0]
            if isinstance(arg, CoercionNode) and arg.arg.is_name:
                arg = arg.arg.entry
                return arg.type.is_cpp_class
        return False

    def type_dependencies(self, env):
        if False:
            i = 10
            return i + 15
        return self.sequence.type_dependencies(self.expr_scope or env)

    def infer_type(self, env):
        if False:
            while True:
                i = 10
        sequence_type = self.sequence.infer_type(env)
        if sequence_type.is_array or sequence_type.is_ptr:
            return sequence_type
        elif sequence_type.is_cpp_class:
            begin = sequence_type.scope.lookup('begin')
            if begin is not None:
                return begin.type.return_type
        elif sequence_type.is_pyobject:
            return sequence_type
        return py_object_type

    def generate_result_code(self, code):
        if False:
            i = 10
            return i + 15
        sequence_type = self.sequence.type
        if sequence_type.is_cpp_class:
            assert False, 'Should have been changed to CppIteratorNode'
        if sequence_type.is_array or sequence_type.is_ptr:
            raise InternalError('for in carray slice not transformed')
        is_builtin_sequence = sequence_type in (list_type, tuple_type)
        if not is_builtin_sequence:
            assert not self.reversed, 'internal error: reversed() only implemented for list/tuple objects'
        self.may_be_a_sequence = not sequence_type.is_builtin_type
        if self.may_be_a_sequence:
            code.putln('if (likely(PyList_CheckExact(%s)) || PyTuple_CheckExact(%s)) {' % (self.sequence.py_result(), self.sequence.py_result()))
        if is_builtin_sequence or self.may_be_a_sequence:
            code.putln('%s = %s; __Pyx_INCREF(%s);' % (self.result(), self.sequence.py_result(), self.result()))
            self.counter_cname = code.funcstate.allocate_temp(PyrexTypes.c_py_ssize_t_type, manage_ref=False)
            if self.reversed:
                if sequence_type is list_type:
                    len_func = '__Pyx_PyList_GET_SIZE'
                else:
                    len_func = '__Pyx_PyTuple_GET_SIZE'
                code.putln('%s = %s(%s);' % (self.counter_cname, len_func, self.result()))
                code.putln('#if !CYTHON_ASSUME_SAFE_MACROS')
                code.putln(code.error_goto_if_neg(self.counter_cname, self.pos))
                code.putln('#endif')
                code.putln('--%s;' % self.counter_cname)
            else:
                code.putln('%s = 0;' % self.counter_cname)
        if not is_builtin_sequence:
            self.iter_func_ptr = code.funcstate.allocate_temp(self._func_iternext_type, manage_ref=False)
            if self.may_be_a_sequence:
                code.putln('%s = NULL;' % self.iter_func_ptr)
                code.putln('} else {')
                code.put('%s = -1; ' % self.counter_cname)
            code.putln('%s = PyObject_GetIter(%s); %s' % (self.result(), self.sequence.py_result(), code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
            code.putln('%s = __Pyx_PyObject_GetIterNextFunc(%s); %s' % (self.iter_func_ptr, self.py_result(), code.error_goto_if_null(self.iter_func_ptr, self.pos)))
        if self.may_be_a_sequence:
            code.putln('}')

    def generate_next_sequence_item(self, test_name, result_name, code):
        if False:
            i = 10
            return i + 15
        assert self.counter_cname, 'internal error: counter_cname temp not prepared'
        assert test_name in ('List', 'Tuple')
        final_size = '__Pyx_Py%s_GET_SIZE(%s)' % (test_name, self.py_result())
        size_is_safe = False
        if self.sequence.is_sequence_constructor:
            item_count = len(self.sequence.args)
            if self.sequence.mult_factor is None:
                final_size = item_count
                size_is_safe = True
            elif isinstance(self.sequence.mult_factor.constant_result, _py_int_types):
                final_size = item_count * self.sequence.mult_factor.constant_result
                size_is_safe = True
        if size_is_safe:
            code.putln('if (%s >= %s) break;' % (self.counter_cname, final_size))
        else:
            code.putln('{')
            code.putln('Py_ssize_t %s = %s;' % (Naming.quick_temp_cname, final_size))
            code.putln('#if !CYTHON_ASSUME_SAFE_MACROS')
            code.putln(code.error_goto_if_neg(Naming.quick_temp_cname, self.pos))
            code.putln('#endif')
            code.putln('if (%s >= %s) break;' % (self.counter_cname, Naming.quick_temp_cname))
            code.putln('}')
        if self.reversed:
            inc_dec = '--'
        else:
            inc_dec = '++'
        code.putln('#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS')
        code.putln('%s = Py%s_GET_ITEM(%s, %s); __Pyx_INCREF(%s); %s%s; %s' % (result_name, test_name, self.py_result(), self.counter_cname, result_name, self.counter_cname, inc_dec, code.error_goto_if_neg('0', self.pos)))
        code.putln('#else')
        code.putln('%s = __Pyx_PySequence_ITEM(%s, %s); %s%s; %s' % (result_name, self.py_result(), self.counter_cname, self.counter_cname, inc_dec, code.error_goto_if_null(result_name, self.pos)))
        code.put_gotref(result_name, py_object_type)
        code.putln('#endif')

    def generate_iter_next_result_code(self, result_name, code):
        if False:
            while True:
                i = 10
        sequence_type = self.sequence.type
        if self.reversed:
            code.putln('if (%s < 0) break;' % self.counter_cname)
        if sequence_type is list_type:
            self.generate_next_sequence_item('List', result_name, code)
            return
        elif sequence_type is tuple_type:
            self.generate_next_sequence_item('Tuple', result_name, code)
            return
        if self.may_be_a_sequence:
            code.putln('if (likely(!%s)) {' % self.iter_func_ptr)
            code.putln('if (likely(PyList_CheckExact(%s))) {' % self.py_result())
            self.generate_next_sequence_item('List', result_name, code)
            code.putln('} else {')
            self.generate_next_sequence_item('Tuple', result_name, code)
            code.putln('}')
            code.put('} else ')
        code.putln('{')
        code.putln('%s = %s(%s);' % (result_name, self.iter_func_ptr, self.py_result()))
        code.putln('if (unlikely(!%s)) {' % result_name)
        code.putln('PyObject* exc_type = PyErr_Occurred();')
        code.putln('if (exc_type) {')
        code.putln('if (likely(__Pyx_PyErr_GivenExceptionMatches(exc_type, PyExc_StopIteration))) PyErr_Clear();')
        code.putln('else %s' % code.error_goto(self.pos))
        code.putln('}')
        code.putln('break;')
        code.putln('}')
        code.put_gotref(result_name, py_object_type)
        code.putln('}')

    def free_temps(self, code):
        if False:
            while True:
                i = 10
        if self.counter_cname:
            code.funcstate.release_temp(self.counter_cname)
        if self.iter_func_ptr:
            code.funcstate.release_temp(self.iter_func_ptr)
            self.iter_func_ptr = None
        ExprNode.free_temps(self, code)

class CppIteratorNode(ExprNode):
    cpp_sequence_cname = None
    cpp_attribute_op = '.'
    extra_dereference = ''
    is_temp = True
    reversed = False
    subexprs = ['sequence']

    def get_iterator_func_names(self):
        if False:
            while True:
                i = 10
        return ('begin', 'end') if not self.reversed else ('rbegin', 'rend')

    def analyse_types(self, env):
        if False:
            return 10
        sequence_type = self.sequence.type
        if sequence_type.is_ptr:
            sequence_type = sequence_type.base_type
        (begin_name, end_name) = self.get_iterator_func_names()
        begin = sequence_type.scope.lookup(begin_name)
        end = sequence_type.scope.lookup(end_name)
        if begin is None or not begin.type.is_cfunction or begin.type.args:
            error(self.pos, 'missing %s() on %s' % (begin_name, self.sequence.type))
            self.type = error_type
            return self
        if end is None or not end.type.is_cfunction or end.type.args:
            error(self.pos, 'missing %s() on %s' % (end_name, self.sequence.type))
            self.type = error_type
            return self
        iter_type = begin.type.return_type
        if iter_type.is_cpp_class:
            if env.directives['cpp_locals']:
                self.extra_dereference = '*'
            if env.lookup_operator_for_types(self.pos, '!=', [iter_type, end.type.return_type]) is None:
                error(self.pos, 'missing operator!= on result of %s() on %s' % (begin_name, self.sequence.type))
                self.type = error_type
                return self
            if env.lookup_operator_for_types(self.pos, '++', [iter_type]) is None:
                error(self.pos, 'missing operator++ on result of %s() on %s' % (begin_name, self.sequence.type))
                self.type = error_type
                return self
            if env.lookup_operator_for_types(self.pos, '*', [iter_type]) is None:
                error(self.pos, 'missing operator* on result of %s() on %s' % (begin_name, self.sequence.type))
                self.type = error_type
                return self
            self.type = iter_type
        elif iter_type.is_ptr:
            if not iter_type == end.type.return_type:
                error(self.pos, 'incompatible types for %s() and %s()' % (begin_name, end_name))
            self.type = iter_type
        else:
            error(self.pos, 'result type of %s() on %s must be a C++ class or pointer' % (begin_name, self.sequence.type))
            self.type = error_type
        return self

    def generate_result_code(self, code):
        if False:
            print('Hello World!')
        sequence_type = self.sequence.type
        (begin_name, _) = self.get_iterator_func_names()
        if self.sequence.is_simple():
            code.putln('%s = %s%s%s();' % (self.result(), self.sequence.result(), self.cpp_attribute_op, begin_name))
        else:
            temp_type = sequence_type
            if temp_type.is_reference:
                temp_type = PyrexTypes.CPtrType(sequence_type.ref_base_type)
            if temp_type.is_ptr or code.globalstate.directives['cpp_locals']:
                self.cpp_attribute_op = '->'
            self.cpp_sequence_cname = code.funcstate.allocate_temp(temp_type, manage_ref=False)
            code.putln('%s = %s%s;' % (self.cpp_sequence_cname, '&' if temp_type.is_ptr else '', self.sequence.move_result_rhs()))
            code.putln('%s = %s%s%s();' % (self.result(), self.cpp_sequence_cname, self.cpp_attribute_op, begin_name))

    def generate_iter_next_result_code(self, result_name, code):
        if False:
            for i in range(10):
                print('nop')
        (_, end_name) = self.get_iterator_func_names()
        code.putln('if (!(%s%s != %s%s%s())) break;' % (self.extra_dereference, self.result(), self.cpp_sequence_cname or self.sequence.result(), self.cpp_attribute_op, end_name))
        code.putln('%s = *%s%s;' % (result_name, self.extra_dereference, self.result()))
        code.putln('++%s%s;' % (self.extra_dereference, self.result()))

    def generate_subexpr_disposal_code(self, code):
        if False:
            print('Hello World!')
        if not self.cpp_sequence_cname:
            return
        ExprNode.generate_subexpr_disposal_code(self, code)

    def free_subexpr_temps(self, code):
        if False:
            for i in range(10):
                print('nop')
        if not self.cpp_sequence_cname:
            return
        ExprNode.free_subexpr_temps(self, code)

    def generate_disposal_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        if not self.cpp_sequence_cname:
            ExprNode.generate_subexpr_disposal_code(self, code)
            ExprNode.free_subexpr_temps(self, code)
        ExprNode.generate_disposal_code(self, code)

    def free_temps(self, code):
        if False:
            i = 10
            return i + 15
        if self.cpp_sequence_cname:
            code.funcstate.release_temp(self.cpp_sequence_cname)
        ExprNode.free_temps(self, code)

class NextNode(AtomicExprNode):

    def __init__(self, iterator):
        if False:
            while True:
                i = 10
        AtomicExprNode.__init__(self, iterator.pos)
        self.iterator = iterator

    def nogil_check(self, env):
        if False:
            i = 10
            return i + 15
        pass

    def type_dependencies(self, env):
        if False:
            print('Hello World!')
        return self.iterator.type_dependencies(env)

    def infer_type(self, env, iterator_type=None):
        if False:
            print('Hello World!')
        if iterator_type is None:
            iterator_type = self.iterator.infer_type(env)
        if iterator_type.is_ptr or iterator_type.is_array:
            return iterator_type.base_type
        elif iterator_type.is_cpp_class:
            item_type = env.lookup_operator_for_types(self.pos, '*', [iterator_type]).type.return_type
            item_type = PyrexTypes.remove_cv_ref(item_type, remove_fakeref=True)
            return item_type
        else:
            fake_index_node = IndexNode(self.pos, base=self.iterator.sequence, index=IntNode(self.pos, value='PY_SSIZE_T_MAX', type=PyrexTypes.c_py_ssize_t_type))
            return fake_index_node.infer_type(env)

    def analyse_types(self, env):
        if False:
            print('Hello World!')
        self.type = self.infer_type(env, self.iterator.type)
        self.is_temp = 1
        return self

    def generate_result_code(self, code):
        if False:
            return 10
        self.iterator.generate_iter_next_result_code(self.result(), code)

class AsyncIteratorNode(ScopedExprNode):
    subexprs = ['sequence']
    is_async = True
    type = py_object_type
    is_temp = 1
    has_local_scope = False

    def infer_type(self, env):
        if False:
            return 10
        return py_object_type

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        if self.expr_scope:
            env = self.expr_scope
        self.sequence = self.sequence.analyse_types(env)
        if not self.sequence.type.is_pyobject:
            error(self.pos, 'async for loops not allowed on C/C++ types')
            self.sequence = self.sequence.coerce_to_pyobject(env)
        return self

    def generate_result_code(self, code):
        if False:
            print('Hello World!')
        code.globalstate.use_utility_code(UtilityCode.load_cached('AsyncIter', 'Coroutine.c'))
        code.putln('%s = __Pyx_Coroutine_GetAsyncIter(%s); %s' % (self.result(), self.sequence.py_result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

class AsyncNextNode(AtomicExprNode):
    type = py_object_type
    is_temp = 1

    def __init__(self, iterator):
        if False:
            return 10
        AtomicExprNode.__init__(self, iterator.pos)
        self.iterator = iterator

    def infer_type(self, env):
        if False:
            print('Hello World!')
        return py_object_type

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        return self

    def generate_result_code(self, code):
        if False:
            print('Hello World!')
        code.globalstate.use_utility_code(UtilityCode.load_cached('AsyncIter', 'Coroutine.c'))
        code.putln('%s = __Pyx_Coroutine_AsyncIterNext(%s); %s' % (self.result(), self.iterator.py_result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

class WithExitCallNode(ExprNode):
    subexprs = ['args', 'await_expr']
    test_if_run = True
    await_expr = None

    def analyse_types(self, env):
        if False:
            print('Hello World!')
        self.args = self.args.analyse_types(env)
        if self.await_expr:
            self.await_expr = self.await_expr.analyse_types(env)
        self.type = PyrexTypes.c_bint_type
        self.is_temp = True
        return self

    def generate_evaluation_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        if self.test_if_run:
            code.putln('if (%s) {' % self.with_stat.exit_var)
        self.args.generate_evaluation_code(code)
        result_var = code.funcstate.allocate_temp(py_object_type, manage_ref=False)
        code.mark_pos(self.pos)
        code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectCall', 'ObjectHandling.c'))
        code.putln('%s = __Pyx_PyObject_Call(%s, %s, NULL);' % (result_var, self.with_stat.exit_var, self.args.result()))
        code.put_decref_clear(self.with_stat.exit_var, type=py_object_type)
        self.args.generate_disposal_code(code)
        self.args.free_temps(code)
        code.putln(code.error_goto_if_null(result_var, self.pos))
        code.put_gotref(result_var, py_object_type)
        if self.await_expr:
            self.await_expr.generate_evaluation_code(code, source_cname=result_var, decref_source=True)
            code.putln('%s = %s;' % (result_var, self.await_expr.py_result()))
            self.await_expr.generate_post_assignment_code(code)
            self.await_expr.free_temps(code)
        if self.result_is_used:
            self.allocate_temp_result(code)
            code.putln('%s = __Pyx_PyObject_IsTrue(%s);' % (self.result(), result_var))
        code.put_decref_clear(result_var, type=py_object_type)
        if self.result_is_used:
            code.put_error_if_neg(self.pos, self.result())
        code.funcstate.release_temp(result_var)
        if self.test_if_run:
            code.putln('}')

class ExcValueNode(AtomicExprNode):
    type = py_object_type

    def __init__(self, pos):
        if False:
            while True:
                i = 10
        ExprNode.__init__(self, pos)

    def set_var(self, var):
        if False:
            print('Hello World!')
        self.var = var

    def calculate_result_code(self):
        if False:
            while True:
                i = 10
        return self.var

    def generate_result_code(self, code):
        if False:
            i = 10
            return i + 15
        pass

    def analyse_types(self, env):
        if False:
            print('Hello World!')
        return self

class TempNode(ExprNode):
    subexprs = []

    def __init__(self, pos, type, env=None):
        if False:
            while True:
                i = 10
        ExprNode.__init__(self, pos)
        self.type = type
        if type.is_pyobject:
            self.result_ctype = py_object_type
        self.is_temp = 1

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        return self

    def analyse_target_declaration(self, env):
        if False:
            i = 10
            return i + 15
        self.is_target = True

    def generate_result_code(self, code):
        if False:
            i = 10
            return i + 15
        pass

    def allocate(self, code):
        if False:
            print('Hello World!')
        self.temp_cname = code.funcstate.allocate_temp(self.type, manage_ref=True)

    def release(self, code):
        if False:
            print('Hello World!')
        code.funcstate.release_temp(self.temp_cname)
        self.temp_cname = None

    def result(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.temp_cname
        except:
            assert False, 'Remember to call allocate/release on TempNode'
            raise

    def allocate_temp_result(self, code):
        if False:
            while True:
                i = 10
        pass

    def release_temp_result(self, code):
        if False:
            print('Hello World!')
        pass

class PyTempNode(TempNode):

    def __init__(self, pos, env):
        if False:
            while True:
                i = 10
        TempNode.__init__(self, pos, PyrexTypes.py_object_type, env)

class RawCNameExprNode(ExprNode):
    subexprs = []

    def __init__(self, pos, type=None, cname=None):
        if False:
            while True:
                i = 10
        ExprNode.__init__(self, pos, type=type)
        if cname is not None:
            self.cname = cname

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        return self

    def set_cname(self, cname):
        if False:
            while True:
                i = 10
        self.cname = cname

    def result(self):
        if False:
            while True:
                i = 10
        return self.cname

    def generate_result_code(self, code):
        if False:
            i = 10
            return i + 15
        pass

class JoinedStrNode(ExprNode):
    type = unicode_type
    is_temp = True
    gil_message = 'String concatenation'
    subexprs = ['values']

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        self.values = [v.analyse_types(env).coerce_to_pyobject(env) for v in self.values]
        return self

    def may_be_none(self):
        if False:
            i = 10
            return i + 15
        return False

    def generate_evaluation_code(self, code):
        if False:
            i = 10
            return i + 15
        code.mark_pos(self.pos)
        num_items = len(self.values)
        list_var = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
        ulength_var = code.funcstate.allocate_temp(PyrexTypes.c_py_ssize_t_type, manage_ref=False)
        max_char_var = code.funcstate.allocate_temp(PyrexTypes.c_py_ucs4_type, manage_ref=False)
        code.putln('%s = PyTuple_New(%s); %s' % (list_var, num_items, code.error_goto_if_null(list_var, self.pos)))
        code.put_gotref(list_var, py_object_type)
        code.putln('%s = 0;' % ulength_var)
        code.putln('%s = 127;' % max_char_var)
        for (i, node) in enumerate(self.values):
            node.generate_evaluation_code(code)
            node.make_owned_reference(code)
            ulength = '__Pyx_PyUnicode_GET_LENGTH(%s)' % node.py_result()
            max_char_value = '__Pyx_PyUnicode_MAX_CHAR_VALUE(%s)' % node.py_result()
            is_ascii = False
            if isinstance(node, UnicodeNode):
                try:
                    node.value.encode('iso8859-1')
                    max_char_value = '255'
                    node.value.encode('us-ascii')
                    is_ascii = True
                except UnicodeEncodeError:
                    if max_char_value != '255':
                        max_char = max(map(ord, node.value))
                        if max_char < 55296:
                            max_char_value = '65535'
                            ulength = str(len(node.value))
                        elif max_char >= 65536:
                            max_char_value = '1114111'
                            ulength = str(len(node.value))
                        else:
                            pass
                else:
                    ulength = str(len(node.value))
            elif isinstance(node, FormattedValueNode) and node.value.type.is_numeric:
                is_ascii = True
            if not is_ascii:
                code.putln('%s = (%s > %s) ? %s : %s;' % (max_char_var, max_char_value, max_char_var, max_char_value, max_char_var))
            code.putln('%s += %s;' % (ulength_var, ulength))
            node.generate_giveref(code)
            code.putln('PyTuple_SET_ITEM(%s, %s, %s);' % (list_var, i, node.py_result()))
            node.generate_post_assignment_code(code)
            node.free_temps(code)
        code.mark_pos(self.pos)
        self.allocate_temp_result(code)
        code.globalstate.use_utility_code(UtilityCode.load_cached('JoinPyUnicode', 'StringTools.c'))
        code.putln('%s = __Pyx_PyUnicode_Join(%s, %d, %s, %s); %s' % (self.result(), list_var, num_items, ulength_var, max_char_var, code.error_goto_if_null(self.py_result(), self.pos)))
        self.generate_gotref(code)
        code.put_decref_clear(list_var, py_object_type)
        code.funcstate.release_temp(list_var)
        code.funcstate.release_temp(ulength_var)
        code.funcstate.release_temp(max_char_var)

class FormattedValueNode(ExprNode):
    subexprs = ['value', 'format_spec']
    type = unicode_type
    is_temp = True
    c_format_spec = None
    gil_message = 'String formatting'
    find_conversion_func = {'s': 'PyObject_Unicode', 'r': 'PyObject_Repr', 'a': 'PyObject_ASCII', 'd': '__Pyx_PyNumber_IntOrLong'}.get

    def may_be_none(self):
        if False:
            print('Hello World!')
        return False

    def analyse_types(self, env):
        if False:
            print('Hello World!')
        self.value = self.value.analyse_types(env)
        if not self.format_spec or self.format_spec.is_string_literal:
            c_format_spec = self.format_spec.value if self.format_spec else self.value.type.default_format_spec
            if self.value.type.can_coerce_to_pystring(env, format_spec=c_format_spec):
                self.c_format_spec = c_format_spec
        if self.format_spec:
            self.format_spec = self.format_spec.analyse_types(env).coerce_to_pyobject(env)
        if self.c_format_spec is None:
            self.value = self.value.coerce_to_pyobject(env)
            if not self.format_spec and (not self.conversion_char or self.conversion_char == 's'):
                if self.value.type is unicode_type and (not self.value.may_be_none()):
                    return self.value
        return self

    def generate_result_code(self, code):
        if False:
            i = 10
            return i + 15
        if self.c_format_spec is not None and (not self.value.type.is_pyobject):
            convert_func_call = self.value.type.convert_to_pystring(self.value.result(), code, self.c_format_spec)
            code.putln('%s = %s; %s' % (self.result(), convert_func_call, code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
            return
        value_result = self.value.py_result()
        value_is_unicode = self.value.type is unicode_type and (not self.value.may_be_none())
        if self.format_spec:
            format_func = '__Pyx_PyObject_Format'
            format_spec = self.format_spec.py_result()
        else:
            format_func = '__Pyx_PyObject_FormatSimple'
            format_spec = Naming.empty_unicode
        conversion_char = self.conversion_char
        if conversion_char == 's' and value_is_unicode:
            conversion_char = None
        if conversion_char:
            fn = self.find_conversion_func(conversion_char)
            assert fn is not None, "invalid conversion character found: '%s'" % conversion_char
            value_result = '%s(%s)' % (fn, value_result)
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectFormatAndDecref', 'StringTools.c'))
            format_func += 'AndDecref'
        elif self.format_spec:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectFormat', 'StringTools.c'))
        else:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectFormatSimple', 'StringTools.c'))
        code.putln('%s = %s(%s, %s); %s' % (self.result(), format_func, value_result, format_spec, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

class ParallelThreadsAvailableNode(AtomicExprNode):
    """
    Note: this is disabled and not a valid directive at this moment

    Implements cython.parallel.threadsavailable(). If we are called from the
    sequential part of the application, we need to call omp_get_max_threads(),
    and in the parallel part we can just call omp_get_num_threads()
    """
    type = PyrexTypes.c_int_type

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        self.is_temp = True
        return self

    def generate_result_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        code.putln('#ifdef _OPENMP')
        code.putln('if (omp_in_parallel()) %s = omp_get_max_threads();' % self.temp_code)
        code.putln('else %s = omp_get_num_threads();' % self.temp_code)
        code.putln('#else')
        code.putln('%s = 1;' % self.temp_code)
        code.putln('#endif')

    def result(self):
        if False:
            while True:
                i = 10
        return self.temp_code

class ParallelThreadIdNode(AtomicExprNode):
    """
    Implements cython.parallel.threadid()
    """
    type = PyrexTypes.c_int_type

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        self.is_temp = True
        return self

    def generate_result_code(self, code):
        if False:
            return 10
        code.putln('#ifdef _OPENMP')
        code.putln('%s = omp_get_thread_num();' % self.temp_code)
        code.putln('#else')
        code.putln('%s = 0;' % self.temp_code)
        code.putln('#endif')

    def result(self):
        if False:
            i = 10
            return i + 15
        return self.temp_code

class _IndexingBaseNode(ExprNode):

    def is_ephemeral(self):
        if False:
            i = 10
            return i + 15
        return self.base.is_ephemeral() or self.base.type in (basestring_type, str_type, bytes_type, bytearray_type, unicode_type)

    def check_const_addr(self):
        if False:
            while True:
                i = 10
        return self.base.check_const_addr() and self.index.check_const()

    def is_lvalue(self):
        if False:
            print('Hello World!')
        if self.type.is_reference:
            if self.type.ref_base_type.is_array:
                return False
        elif self.type.is_ptr:
            return True
        return True

class IndexNode(_IndexingBaseNode):
    subexprs = ['base', 'index']
    type_indices = None
    is_subscript = True
    is_fused_index = False

    def calculate_constant_result(self):
        if False:
            return 10
        self.constant_result = self.base.constant_result[self.index.constant_result]

    def compile_time_value(self, denv):
        if False:
            for i in range(10):
                print('nop')
        base = self.base.compile_time_value(denv)
        index = self.index.compile_time_value(denv)
        try:
            return base[index]
        except Exception as e:
            self.compile_time_value_error(e)

    def is_simple(self):
        if False:
            i = 10
            return i + 15
        base = self.base
        return base.is_simple() and self.index.is_simple() and base.type and (base.type.is_ptr or base.type.is_array)

    def may_be_none(self):
        if False:
            return 10
        base_type = self.base.type
        if base_type:
            if base_type.is_string:
                return False
            if isinstance(self.index, SliceNode):
                if base_type in (bytes_type, bytearray_type, str_type, unicode_type, basestring_type, list_type, tuple_type):
                    return False
        return ExprNode.may_be_none(self)

    def analyse_target_declaration(self, env):
        if False:
            while True:
                i = 10
        pass

    def analyse_as_type(self, env):
        if False:
            while True:
                i = 10
        base_type = self.base.analyse_as_type(env)
        if base_type:
            if base_type.is_cpp_class or base_type.python_type_constructor_name:
                if self.index.is_sequence_constructor:
                    template_values = self.index.args
                else:
                    template_values = [self.index]
                type_node = Nodes.TemplatedTypeNode(pos=self.pos, positional_args=template_values, keyword_args=None)
                return type_node.analyse(env, base_type=base_type)
            elif self.index.is_slice or self.index.is_sequence_constructor:
                from . import MemoryView
                env.use_utility_code(MemoryView.view_utility_code)
                axes = [self.index] if self.index.is_slice else list(self.index.args)
                return PyrexTypes.MemoryViewSliceType(base_type, MemoryView.get_axes_specs(env, axes))
            elif not base_type.is_pyobject:
                index = self.index.compile_time_value(env)
                if index is not None:
                    try:
                        index = int(index)
                    except (ValueError, TypeError):
                        pass
                    else:
                        return PyrexTypes.CArrayType(base_type, index)
                error(self.pos, 'Array size must be a compile time constant')
        return None

    def analyse_pytyping_modifiers(self, env):
        if False:
            return 10
        modifiers = []
        modifier_node = self
        while modifier_node.is_subscript:
            modifier_type = modifier_node.base.analyse_as_type(env)
            if modifier_type and modifier_type.python_type_constructor_name and modifier_type.modifier_name:
                modifiers.append(modifier_type.modifier_name)
            modifier_node = modifier_node.index
        return modifiers

    def type_dependencies(self, env):
        if False:
            i = 10
            return i + 15
        return self.base.type_dependencies(env) + self.index.type_dependencies(env)

    def infer_type(self, env):
        if False:
            while True:
                i = 10
        base_type = self.base.infer_type(env)
        if self.index.is_slice:
            if base_type.is_string:
                return bytes_type
            elif base_type.is_pyunicode_ptr:
                return unicode_type
            elif base_type in (unicode_type, bytes_type, str_type, bytearray_type, list_type, tuple_type):
                return base_type
            elif base_type.is_memoryviewslice:
                return base_type
            else:
                return py_object_type
        index_type = self.index.infer_type(env)
        if index_type and index_type.is_int or isinstance(self.index, IntNode):
            if base_type is unicode_type:
                return PyrexTypes.c_py_ucs4_type
            elif base_type is str_type:
                return base_type
            elif base_type is bytearray_type:
                return PyrexTypes.c_uchar_type
            elif isinstance(self.base, BytesNode):
                return py_object_type
            elif base_type in (tuple_type, list_type):
                item_type = infer_sequence_item_type(env, self.base, self.index, seq_type=base_type)
                if item_type is not None:
                    return item_type
            elif base_type.is_ptr or base_type.is_array:
                return base_type.base_type
            elif base_type.is_ctuple and isinstance(self.index, IntNode):
                if self.index.has_constant_result():
                    index = self.index.constant_result
                    if index < 0:
                        index += base_type.size
                    if 0 <= index < base_type.size:
                        return base_type.components[index]
            elif base_type.is_memoryviewslice:
                if base_type.ndim == 0:
                    pass
                if base_type.ndim == 1:
                    return base_type.dtype
                else:
                    return PyrexTypes.MemoryViewSliceType(base_type.dtype, base_type.axes[1:])
        if self.index.is_sequence_constructor and base_type.is_memoryviewslice:
            inferred_type = base_type
            for a in self.index.args:
                if not inferred_type.is_memoryviewslice:
                    break
                inferred_type = IndexNode(self.pos, base=ExprNode(self.base.pos, type=inferred_type), index=a).infer_type(env)
            else:
                return inferred_type
        if base_type.is_cpp_class:

            class FakeOperand:

                def __init__(self, **kwds):
                    if False:
                        return 10
                    self.__dict__.update(kwds)
            operands = [FakeOperand(pos=self.pos, type=base_type), FakeOperand(pos=self.pos, type=index_type)]
            index_func = env.lookup_operator('[]', operands)
            if index_func is not None:
                return index_func.type.return_type
        if is_pythran_expr(base_type) and is_pythran_expr(index_type):
            index_with_type = (self.index, index_type)
            return PythranExpr(pythran_indexing_type(base_type, [index_with_type]))
        if base_type in (unicode_type, str_type):
            return base_type
        else:
            return py_object_type

    def analyse_types(self, env):
        if False:
            return 10
        return self.analyse_base_and_index_types(env, getting=True)

    def analyse_target_types(self, env):
        if False:
            i = 10
            return i + 15
        node = self.analyse_base_and_index_types(env, setting=True)
        if node.type.is_const:
            error(self.pos, 'Assignment to const dereference')
        if node is self and (not node.is_lvalue()):
            error(self.pos, "Assignment to non-lvalue of type '%s'" % node.type)
        return node

    def analyse_base_and_index_types(self, env, getting=False, setting=False, analyse_base=True):
        if False:
            return 10
        if analyse_base:
            self.base = self.base.analyse_types(env)
        if self.base.type.is_error:
            self.type = PyrexTypes.error_type
            return self
        is_slice = self.index.is_slice
        if not env.directives['wraparound']:
            if is_slice:
                check_negative_indices(self.index.start, self.index.stop)
            else:
                check_negative_indices(self.index)
        if not is_slice and isinstance(self.index, IntNode) and Utils.long_literal(self.index.value):
            self.index = self.index.coerce_to_pyobject(env)
        is_memslice = self.base.type.is_memoryviewslice
        if not is_memslice and (isinstance(self.base, BytesNode) or is_slice):
            if self.base.type.is_string or not (self.base.type.is_ptr or self.base.type.is_array):
                self.base = self.base.coerce_to_pyobject(env)
        replacement_node = self.analyse_as_buffer_operation(env, getting)
        if replacement_node is not None:
            return replacement_node
        self.nogil = env.nogil
        base_type = self.base.type
        if not base_type.is_cfunction:
            self.index = self.index.analyse_types(env)
            self.original_index_type = self.index.type
            if self.original_index_type.is_reference:
                self.original_index_type = self.original_index_type.ref_base_type
            if base_type.is_unicode_char:
                if setting:
                    warning(self.pos, 'cannot assign to Unicode string index', level=1)
                elif self.index.constant_result in (0, -1):
                    return self.base
                self.base = self.base.coerce_to_pyobject(env)
                base_type = self.base.type
        if base_type.is_pyobject:
            return self.analyse_as_pyobject(env, is_slice, getting, setting)
        elif base_type.is_ptr or base_type.is_array:
            return self.analyse_as_c_array(env, is_slice)
        elif base_type.is_cpp_class:
            return self.analyse_as_cpp(env, setting)
        elif base_type.is_cfunction:
            return self.analyse_as_c_function(env)
        elif base_type.is_ctuple:
            return self.analyse_as_c_tuple(env, getting, setting)
        else:
            error(self.pos, "Attempting to index non-array type '%s'" % base_type)
            self.type = PyrexTypes.error_type
            return self

    def analyse_as_pyobject(self, env, is_slice, getting, setting):
        if False:
            i = 10
            return i + 15
        base_type = self.base.type
        if self.index.type.is_unicode_char and base_type is not dict_type:
            warning(self.pos, 'Item lookup of unicode character codes now always converts to a Unicode string. Use an explicit C integer cast to get back the previous integer lookup behaviour.', level=1)
            self.index = self.index.coerce_to_pyobject(env)
            self.is_temp = 1
        elif self.index.type.is_int and base_type is not dict_type:
            if getting and (not env.directives['boundscheck']) and (base_type in (list_type, tuple_type, bytearray_type)) and (not self.index.type.signed or not env.directives['wraparound'] or (isinstance(self.index, IntNode) and self.index.has_constant_result() and (self.index.constant_result >= 0))):
                self.is_temp = 0
            else:
                self.is_temp = 1
            self.index = self.index.coerce_to(PyrexTypes.c_py_ssize_t_type, env).coerce_to_simple(env)
            self.original_index_type.create_to_py_utility_code(env)
        else:
            self.index = self.index.coerce_to_pyobject(env)
            self.is_temp = 1
        if self.index.type.is_int and base_type is unicode_type:
            self.type = PyrexTypes.c_py_ucs4_type
        elif self.index.type.is_int and base_type is bytearray_type:
            if setting:
                self.type = PyrexTypes.c_uchar_type
            else:
                self.type = PyrexTypes.c_int_type
        elif is_slice and base_type in (bytes_type, bytearray_type, str_type, unicode_type, list_type, tuple_type):
            self.type = base_type
        else:
            item_type = None
            if base_type in (list_type, tuple_type) and self.index.type.is_int:
                item_type = infer_sequence_item_type(env, self.base, self.index, seq_type=base_type)
            if base_type in (list_type, tuple_type, dict_type):
                self.base = self.base.as_none_safe_node("'NoneType' object is not subscriptable")
            if item_type is None or not item_type.is_pyobject:
                self.type = py_object_type
            else:
                self.type = item_type
        self.wrap_in_nonecheck_node(env, getting)
        return self

    def analyse_as_c_array(self, env, is_slice):
        if False:
            while True:
                i = 10
        base_type = self.base.type
        self.type = base_type.base_type
        if self.type.is_cpp_class:
            self.type = PyrexTypes.CReferenceType(self.type)
        if is_slice:
            self.type = base_type
        elif self.index.type.is_pyobject:
            self.index = self.index.coerce_to(PyrexTypes.c_py_ssize_t_type, env)
        elif not self.index.type.is_int:
            error(self.pos, "Invalid index type '%s'" % self.index.type)
        return self

    def analyse_as_cpp(self, env, setting):
        if False:
            print('Hello World!')
        base_type = self.base.type
        function = env.lookup_operator('[]', [self.base, self.index])
        if function is None:
            error(self.pos, "Indexing '%s' not supported for index type '%s'" % (base_type, self.index.type))
            self.type = PyrexTypes.error_type
            self.result_code = '<error>'
            return self
        func_type = function.type
        if func_type.is_ptr:
            func_type = func_type.base_type
        self.exception_check = func_type.exception_check
        self.exception_value = func_type.exception_value
        if self.exception_check:
            if not setting:
                self.is_temp = True
            if needs_cpp_exception_conversion(self):
                env.use_utility_code(UtilityCode.load_cached('CppExceptionConversion', 'CppSupport.cpp'))
        self.index = self.index.coerce_to(func_type.args[0].type, env)
        self.type = func_type.return_type
        if setting and (not func_type.return_type.is_reference):
            error(self.pos, "Can't set non-reference result '%s'" % self.type)
        return self

    def analyse_as_c_function(self, env):
        if False:
            print('Hello World!')
        base_type = self.base.type
        if base_type.is_fused:
            self.parse_indexed_fused_cdef(env)
        else:
            self.type_indices = self.parse_index_as_types(env)
            self.index = None
            if base_type.templates is None:
                error(self.pos, 'Can only parameterize template functions.')
                self.type = error_type
            elif self.type_indices is None:
                self.type = error_type
            elif len(base_type.templates) != len(self.type_indices):
                error(self.pos, 'Wrong number of template arguments: expected %s, got %s' % (len(base_type.templates), len(self.type_indices)))
                self.type = error_type
            else:
                self.type = base_type.specialize(dict(zip(base_type.templates, self.type_indices)))
        return self

    def analyse_as_c_tuple(self, env, getting, setting):
        if False:
            return 10
        base_type = self.base.type
        if isinstance(self.index, IntNode) and self.index.has_constant_result():
            index = self.index.constant_result
            if -base_type.size <= index < base_type.size:
                if index < 0:
                    index += base_type.size
                self.type = base_type.components[index]
            else:
                error(self.pos, "Index %s out of bounds for '%s'" % (index, base_type))
                self.type = PyrexTypes.error_type
            return self
        else:
            self.base = self.base.coerce_to_pyobject(env)
            return self.analyse_base_and_index_types(env, getting=getting, setting=setting, analyse_base=False)

    def analyse_as_buffer_operation(self, env, getting):
        if False:
            i = 10
            return i + 15
        '\n        Analyse buffer indexing and memoryview indexing/slicing\n        '
        if isinstance(self.index, TupleNode):
            indices = self.index.args
        else:
            indices = [self.index]
        base = self.base
        base_type = base.type
        replacement_node = None
        if base_type.is_memoryviewslice:
            from . import MemoryView
            if base.is_memview_slice:
                merged_indices = base.merged_indices(indices)
                if merged_indices is not None:
                    base = base.base
                    base_type = base.type
                    indices = merged_indices
            (have_slices, indices, newaxes) = MemoryView.unellipsify(indices, base_type.ndim)
            if have_slices:
                replacement_node = MemoryViewSliceNode(self.pos, indices=indices, base=base)
            else:
                replacement_node = MemoryViewIndexNode(self.pos, indices=indices, base=base)
        elif base_type.is_buffer or base_type.is_pythran_expr:
            if base_type.is_pythran_expr or len(indices) == base_type.ndim:
                is_buffer_access = True
                indices = [index.analyse_types(env) for index in indices]
                if base_type.is_pythran_expr:
                    do_replacement = all((index.type.is_int or index.is_slice or index.type.is_pythran_expr for index in indices))
                    if do_replacement:
                        for (i, index) in enumerate(indices):
                            if index.is_slice:
                                index = SliceIntNode(index.pos, start=index.start, stop=index.stop, step=index.step)
                                index = index.analyse_types(env)
                                indices[i] = index
                else:
                    do_replacement = all((index.type.is_int for index in indices))
                if do_replacement:
                    replacement_node = BufferIndexNode(self.pos, indices=indices, base=base)
                    assert not isinstance(self.index, CloneNode)
        if replacement_node is not None:
            replacement_node = replacement_node.analyse_types(env, getting)
        return replacement_node

    def wrap_in_nonecheck_node(self, env, getting):
        if False:
            print('Hello World!')
        if not env.directives['nonecheck'] or not self.base.may_be_none():
            return
        self.base = self.base.as_none_safe_node("'NoneType' object is not subscriptable")

    def parse_index_as_types(self, env, required=True):
        if False:
            while True:
                i = 10
        if isinstance(self.index, TupleNode):
            indices = self.index.args
        else:
            indices = [self.index]
        type_indices = []
        for index in indices:
            type_indices.append(index.analyse_as_type(env))
            if type_indices[-1] is None:
                if required:
                    error(index.pos, 'not parsable as a type')
                return None
        return type_indices

    def parse_indexed_fused_cdef(self, env):
        if False:
            return 10
        '\n        Interpret fused_cdef_func[specific_type1, ...]\n\n        Note that if this method is called, we are an indexed cdef function\n        with fused argument types, and this IndexNode will be replaced by the\n        NameNode with specific entry just after analysis of expressions by\n        AnalyseExpressionsTransform.\n        '
        self.type = PyrexTypes.error_type
        self.is_fused_index = True
        base_type = self.base.type
        positions = []
        if self.index.is_name or self.index.is_attribute:
            positions.append(self.index.pos)
        elif isinstance(self.index, TupleNode):
            for arg in self.index.args:
                positions.append(arg.pos)
        specific_types = self.parse_index_as_types(env, required=False)
        if specific_types is None:
            self.index = self.index.analyse_types(env)
            if not self.base.entry.as_variable:
                error(self.pos, 'Can only index fused functions with types')
            else:
                self.base.entry = self.entry = self.base.entry.as_variable
                self.base.type = self.type = self.entry.type
                self.base.is_temp = True
                self.is_temp = True
                self.entry.used = True
            self.is_fused_index = False
            return
        for (i, type) in enumerate(specific_types):
            specific_types[i] = type.specialize_fused(env)
        fused_types = base_type.get_fused_types()
        if len(specific_types) > len(fused_types):
            return error(self.pos, 'Too many types specified')
        elif len(specific_types) < len(fused_types):
            t = fused_types[len(specific_types)]
            return error(self.pos, 'Not enough types specified to specialize the function, %s is still fused' % t)
        for (pos, specific_type, fused_type) in zip(positions, specific_types, fused_types):
            if not any([specific_type.same_as(t) for t in fused_type.types]):
                return error(pos, 'Type not in fused type')
            if specific_type is None or specific_type.is_error:
                return
        fused_to_specific = dict(zip(fused_types, specific_types))
        type = base_type.specialize(fused_to_specific)
        if type.is_fused:
            error(self.pos, 'Index operation makes function only partially specific')
        else:
            for signature in self.base.type.get_all_specialized_function_types():
                if type.same_as(signature):
                    self.type = signature
                    if self.base.is_attribute:
                        self.entry = signature.entry
                        self.is_attribute = True
                        self.obj = self.base.obj
                    self.type.entry.used = True
                    self.base.type = signature
                    self.base.entry = signature.entry
                    break
            else:
                raise InternalError("Couldn't find the right signature")
    gil_message = 'Indexing Python object'

    def calculate_result_code(self):
        if False:
            for i in range(10):
                print('nop')
        if self.base.type in (list_type, tuple_type, bytearray_type):
            if self.base.type is list_type:
                index_code = 'PyList_GET_ITEM(%s, %s)'
            elif self.base.type is tuple_type:
                index_code = 'PyTuple_GET_ITEM(%s, %s)'
            elif self.base.type is bytearray_type:
                index_code = '((unsigned char)(PyByteArray_AS_STRING(%s)[%s]))'
            else:
                assert False, 'unexpected base type in indexing: %s' % self.base.type
        elif self.base.type.is_cfunction:
            return '%s<%s>' % (self.base.result(), ','.join([param.empty_declaration_code() for param in self.type_indices]))
        elif self.base.type.is_ctuple:
            index = self.index.constant_result
            if index < 0:
                index += self.base.type.size
            return '%s.f%s' % (self.base.result(), index)
        else:
            if (self.type.is_ptr or self.type.is_array) and self.type == self.base.type:
                error(self.pos, 'Invalid use of pointer slice')
                return
            index_code = '(%s[%s])'
        return index_code % (self.base.result(), self.index.result())

    def extra_index_params(self, code):
        if False:
            i = 10
            return i + 15
        if self.index.type.is_int:
            is_list = self.base.type is list_type
            wraparound = bool(code.globalstate.directives['wraparound']) and self.original_index_type.signed and (not (isinstance(self.index.constant_result, _py_int_types) and self.index.constant_result >= 0))
            boundscheck = bool(code.globalstate.directives['boundscheck'])
            return ', %s, %d, %s, %d, %d, %d' % (self.original_index_type.empty_declaration_code(), self.original_index_type.signed and 1 or 0, self.original_index_type.to_py_function, is_list, wraparound, boundscheck)
        else:
            return ''

    def generate_result_code(self, code):
        if False:
            i = 10
            return i + 15
        if not self.is_temp:
            return
        utility_code = None
        error_value = None
        if self.type.is_pyobject:
            error_value = 'NULL'
            if self.index.type.is_int:
                if self.base.type is list_type:
                    function = '__Pyx_GetItemInt_List'
                elif self.base.type is tuple_type:
                    function = '__Pyx_GetItemInt_Tuple'
                else:
                    function = '__Pyx_GetItemInt'
                utility_code = TempitaUtilityCode.load_cached('GetItemInt', 'ObjectHandling.c')
            elif self.base.type is dict_type:
                function = '__Pyx_PyDict_GetItem'
                utility_code = UtilityCode.load_cached('DictGetItem', 'ObjectHandling.c')
            elif self.base.type is py_object_type and self.index.type in (str_type, unicode_type):
                function = '__Pyx_PyObject_Dict_GetItem'
                utility_code = UtilityCode.load_cached('DictGetItem', 'ObjectHandling.c')
            else:
                function = '__Pyx_PyObject_GetItem'
                code.globalstate.use_utility_code(TempitaUtilityCode.load_cached('GetItemInt', 'ObjectHandling.c'))
                utility_code = UtilityCode.load_cached('ObjectGetItem', 'ObjectHandling.c')
        elif self.type.is_unicode_char and self.base.type is unicode_type:
            assert self.index.type.is_int
            function = '__Pyx_GetItemInt_Unicode'
            error_value = '(Py_UCS4)-1'
            utility_code = UtilityCode.load_cached('GetItemIntUnicode', 'StringTools.c')
        elif self.base.type is bytearray_type:
            assert self.index.type.is_int
            assert self.type.is_int
            function = '__Pyx_GetItemInt_ByteArray'
            error_value = '-1'
            utility_code = UtilityCode.load_cached('GetItemIntByteArray', 'StringTools.c')
        elif not (self.base.type.is_cpp_class and self.exception_check):
            assert False, 'unexpected type %s and base type %s for indexing (%s)' % (self.type, self.base.type, self.pos)
        if utility_code is not None:
            code.globalstate.use_utility_code(utility_code)
        if self.index.type.is_int:
            index_code = self.index.result()
        else:
            index_code = self.index.py_result()
        if self.base.type.is_cpp_class and self.exception_check:
            translate_cpp_exception(code, self.pos, '%s = %s[%s];' % (self.result(), self.base.result(), self.index.result()), self.result() if self.type.is_pyobject else None, self.exception_value, self.in_nogil_context)
        else:
            error_check = '!%s' if error_value == 'NULL' else '%%s == %s' % error_value
            code.putln('%s = %s(%s, %s%s); %s' % (self.result(), function, self.base.py_result(), index_code, self.extra_index_params(code), code.error_goto_if(error_check % self.result(), self.pos)))
        if self.type.is_pyobject:
            self.generate_gotref(code)

    def generate_setitem_code(self, value_code, code):
        if False:
            i = 10
            return i + 15
        if self.index.type.is_int:
            if self.base.type is bytearray_type:
                code.globalstate.use_utility_code(UtilityCode.load_cached('SetItemIntByteArray', 'StringTools.c'))
                function = '__Pyx_SetItemInt_ByteArray'
            else:
                code.globalstate.use_utility_code(UtilityCode.load_cached('SetItemInt', 'ObjectHandling.c'))
                function = '__Pyx_SetItemInt'
            index_code = self.index.result()
        else:
            index_code = self.index.py_result()
            if self.base.type is dict_type:
                function = 'PyDict_SetItem'
            else:
                function = 'PyObject_SetItem'
        code.putln(code.error_goto_if_neg('%s(%s, %s, %s%s)' % (function, self.base.py_result(), index_code, value_code, self.extra_index_params(code)), self.pos))

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False, exception_check=None, exception_value=None):
        if False:
            i = 10
            return i + 15
        self.generate_subexpr_evaluation_code(code)
        if self.type.is_pyobject:
            self.generate_setitem_code(rhs.py_result(), code)
        elif self.base.type is bytearray_type:
            value_code = self._check_byte_value(code, rhs)
            self.generate_setitem_code(value_code, code)
        elif self.base.type.is_cpp_class and self.exception_check and (self.exception_check == '+'):
            if overloaded_assignment and exception_check and (self.exception_value != exception_value):
                translate_double_cpp_exception(code, self.pos, self.type, self.result(), rhs.result(), self.exception_value, exception_value, self.in_nogil_context)
            else:
                translate_cpp_exception(code, self.pos, '%s = %s;' % (self.result(), rhs.result()), self.result() if self.type.is_pyobject else None, self.exception_value, self.in_nogil_context)
        else:
            code.putln('%s = %s;' % (self.result(), rhs.result()))
        self.generate_subexpr_disposal_code(code)
        self.free_subexpr_temps(code)
        rhs.generate_disposal_code(code)
        rhs.free_temps(code)

    def _check_byte_value(self, code, rhs):
        if False:
            return 10
        assert rhs.type.is_int, repr(rhs.type)
        value_code = rhs.result()
        if rhs.has_constant_result():
            if 0 <= rhs.constant_result < 256:
                return value_code
            needs_cast = True
            warning(rhs.pos, 'value outside of range(0, 256) when assigning to byte: %s' % rhs.constant_result, level=1)
        else:
            needs_cast = rhs.type != PyrexTypes.c_uchar_type
        if not self.nogil:
            conditions = []
            if rhs.is_literal or rhs.type.signed:
                conditions.append('%s < 0' % value_code)
            if rhs.is_literal or not (rhs.is_temp and rhs.type in (PyrexTypes.c_uchar_type, PyrexTypes.c_char_type, PyrexTypes.c_schar_type)):
                conditions.append('%s > 255' % value_code)
            if conditions:
                code.putln('if (unlikely(%s)) {' % ' || '.join(conditions))
                code.putln('PyErr_SetString(PyExc_ValueError, "byte must be in range(0, 256)"); %s' % code.error_goto(self.pos))
                code.putln('}')
        if needs_cast:
            value_code = '((unsigned char)%s)' % value_code
        return value_code

    def generate_deletion_code(self, code, ignore_nonexisting=False):
        if False:
            i = 10
            return i + 15
        self.generate_subexpr_evaluation_code(code)
        if self.index.type.is_int:
            function = '__Pyx_DelItemInt'
            index_code = self.index.result()
            code.globalstate.use_utility_code(UtilityCode.load_cached('DelItemInt', 'ObjectHandling.c'))
        else:
            index_code = self.index.py_result()
            if self.base.type is dict_type:
                function = 'PyDict_DelItem'
            else:
                function = 'PyObject_DelItem'
        code.putln(code.error_goto_if_neg('%s(%s, %s%s)' % (function, self.base.py_result(), index_code, self.extra_index_params(code)), self.pos))
        self.generate_subexpr_disposal_code(code)
        self.free_subexpr_temps(code)

class BufferIndexNode(_IndexingBaseNode):
    """
    Indexing of buffers and memoryviews. This node is created during type
    analysis from IndexNode and replaces it.

    Attributes:
        base - base node being indexed
        indices - list of indexing expressions
    """
    subexprs = ['base', 'indices']
    is_buffer_access = True
    writable_needed = False
    index_temps = ()

    def analyse_target_types(self, env):
        if False:
            while True:
                i = 10
        self.analyse_types(env, getting=False)

    def analyse_types(self, env, getting=True):
        if False:
            i = 10
            return i + 15
        '\n        Analyse types for buffer indexing only. Overridden by memoryview\n        indexing and slicing subclasses\n        '
        if not self.base.is_name and (not is_pythran_expr(self.base.type)):
            error(self.pos, 'Can only index buffer variables')
            self.type = error_type
            return self
        if not getting:
            if not self.base.entry.type.writable:
                error(self.pos, 'Writing to readonly buffer')
            else:
                self.writable_needed = True
                if self.base.type.is_buffer:
                    self.base.entry.buffer_aux.writable_needed = True
        self.none_error_message = "'NoneType' object is not subscriptable"
        self.analyse_buffer_index(env, getting)
        self.wrap_in_nonecheck_node(env)
        return self

    def analyse_buffer_index(self, env, getting):
        if False:
            return 10
        if is_pythran_expr(self.base.type):
            index_with_type_list = [(idx, idx.type) for idx in self.indices]
            self.type = PythranExpr(pythran_indexing_type(self.base.type, index_with_type_list))
        else:
            self.base = self.base.coerce_to_simple(env)
            self.type = self.base.type.dtype
        self.buffer_type = self.base.type
        if getting and (self.type.is_pyobject or self.type.is_pythran_expr):
            self.is_temp = True

    def analyse_assignment(self, rhs):
        if False:
            return 10
        '\n        Called by IndexNode when this node is assigned to,\n        with the rhs of the assignment\n        '

    def wrap_in_nonecheck_node(self, env):
        if False:
            print('Hello World!')
        if not env.directives['nonecheck'] or not self.base.may_be_none():
            return
        self.base = self.base.as_none_safe_node(self.none_error_message)

    def nogil_check(self, env):
        if False:
            print('Hello World!')
        if self.is_buffer_access or self.is_memview_index:
            if self.type.is_pyobject:
                error(self.pos, 'Cannot access buffer with object dtype without gil')
                self.type = error_type

    def calculate_result_code(self):
        if False:
            for i in range(10):
                print('nop')
        return '(*%s)' % self.buffer_ptr_code

    def buffer_entry(self):
        if False:
            while True:
                i = 10
        base = self.base
        if self.base.is_nonecheck:
            base = base.arg
        return base.type.get_entry(base)

    def get_index_in_temp(self, code, ivar):
        if False:
            return 10
        ret = code.funcstate.allocate_temp(PyrexTypes.widest_numeric_type(ivar.type, PyrexTypes.c_ssize_t_type if ivar.type.signed else PyrexTypes.c_size_t_type), manage_ref=False)
        code.putln('%s = %s;' % (ret, ivar.result()))
        return ret

    def buffer_lookup_code(self, code):
        if False:
            return 10
        '\n        ndarray[1, 2, 3] and memslice[1, 2, 3]\n        '
        if self.in_nogil_context:
            if self.is_buffer_access or self.is_memview_index:
                if code.globalstate.directives['boundscheck']:
                    warning(self.pos, 'Use boundscheck(False) for faster access', level=1)
        self.index_temps = index_temps = [self.get_index_in_temp(code, ivar) for ivar in self.indices]
        from . import Buffer
        buffer_entry = self.buffer_entry()
        if buffer_entry.type.is_buffer:
            negative_indices = buffer_entry.type.negative_indices
        else:
            negative_indices = Buffer.buffer_defaults['negative_indices']
        return (buffer_entry, Buffer.put_buffer_lookup_code(entry=buffer_entry, index_signeds=[ivar.type.signed for ivar in self.indices], index_cnames=index_temps, directives=code.globalstate.directives, pos=self.pos, code=code, negative_indices=negative_indices, in_nogil_context=self.in_nogil_context))

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False):
        if False:
            for i in range(10):
                print('nop')
        self.generate_subexpr_evaluation_code(code)
        self.generate_buffer_setitem_code(rhs, code)
        self.generate_subexpr_disposal_code(code)
        self.free_subexpr_temps(code)
        rhs.generate_disposal_code(code)
        rhs.free_temps(code)

    def generate_buffer_setitem_code(self, rhs, code, op=''):
        if False:
            return 10
        base_type = self.base.type
        if is_pythran_expr(base_type) and is_pythran_supported_type(rhs.type):
            obj = code.funcstate.allocate_temp(PythranExpr(pythran_type(self.base.type)), manage_ref=False)
            code.putln('__Pyx_call_destructor(%s);' % obj)
            code.putln('new (&%s) decltype(%s){%s};' % (obj, obj, self.base.pythran_result()))
            code.putln('%s%s %s= %s;' % (obj, pythran_indexing_code(self.indices), op, rhs.pythran_result()))
            code.funcstate.release_temp(obj)
            return
        (buffer_entry, ptrexpr) = self.buffer_lookup_code(code)
        if self.buffer_type.dtype.is_pyobject:
            ptr = code.funcstate.allocate_temp(buffer_entry.buf_ptr_type, manage_ref=False)
            rhs_code = rhs.result()
            code.putln('%s = %s;' % (ptr, ptrexpr))
            code.put_xgotref('*%s' % ptr, self.buffer_type.dtype)
            code.putln('__Pyx_INCREF(%s); __Pyx_XDECREF(*%s);' % (rhs_code, ptr))
            code.putln('*%s %s= %s;' % (ptr, op, rhs_code))
            code.put_xgiveref('*%s' % ptr, self.buffer_type.dtype)
            code.funcstate.release_temp(ptr)
        else:
            code.putln('*%s %s= %s;' % (ptrexpr, op, rhs.result()))

    def generate_result_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        if is_pythran_expr(self.base.type):
            res = self.result()
            code.putln('__Pyx_call_destructor(%s);' % res)
            code.putln('new (&%s) decltype(%s){%s%s};' % (res, res, self.base.pythran_result(), pythran_indexing_code(self.indices)))
            return
        (buffer_entry, self.buffer_ptr_code) = self.buffer_lookup_code(code)
        if self.type.is_pyobject:
            res = self.result()
            code.putln('%s = (PyObject *) *%s;' % (res, self.buffer_ptr_code))
            code.putln('if (unlikely(%s == NULL)) %s = Py_None;' % (res, res))
            code.putln('__Pyx_INCREF((PyObject*)%s);' % res)

    def free_subexpr_temps(self, code):
        if False:
            for i in range(10):
                print('nop')
        for temp in self.index_temps:
            code.funcstate.release_temp(temp)
        self.index_temps = ()
        super(BufferIndexNode, self).free_subexpr_temps(code)

class MemoryViewIndexNode(BufferIndexNode):
    is_memview_index = True
    is_buffer_access = False

    def analyse_types(self, env, getting=True):
        if False:
            return 10
        from . import MemoryView
        self.is_pythran_mode = has_np_pythran(env)
        indices = self.indices
        (have_slices, indices, newaxes) = MemoryView.unellipsify(indices, self.base.type.ndim)
        if not getting:
            self.writable_needed = True
            if self.base.is_name or self.base.is_attribute:
                self.base.entry.type.writable_needed = True
        self.memslice_index = not newaxes and len(indices) == self.base.type.ndim
        axes = []
        index_type = PyrexTypes.c_py_ssize_t_type
        new_indices = []
        if len(indices) - len(newaxes) > self.base.type.ndim:
            self.type = error_type
            error(indices[self.base.type.ndim].pos, 'Too many indices specified for type %s' % self.base.type)
            return self
        axis_idx = 0
        for (i, index) in enumerate(indices[:]):
            index = index.analyse_types(env)
            if index.is_none:
                self.is_memview_slice = True
                new_indices.append(index)
                axes.append(('direct', 'strided'))
                continue
            (access, packing) = self.base.type.axes[axis_idx]
            axis_idx += 1
            if index.is_slice:
                self.is_memview_slice = True
                if index.step.is_none:
                    axes.append((access, packing))
                else:
                    axes.append((access, 'strided'))
                for attr in ('start', 'stop', 'step'):
                    value = getattr(index, attr)
                    if not value.is_none:
                        value = value.coerce_to(index_type, env)
                        setattr(index, attr, value)
                        new_indices.append(value)
            elif index.type.is_int or index.type.is_pyobject:
                if index.type.is_pyobject:
                    performance_hint(index.pos, 'Index should be typed for more efficient access', env)
                self.is_memview_index = True
                index = index.coerce_to(index_type, env)
                indices[i] = index
                new_indices.append(index)
            else:
                self.type = error_type
                error(index.pos, 'Invalid index for memoryview specified, type %s' % index.type)
                return self
        self.is_memview_index = self.is_memview_index and (not self.is_memview_slice)
        self.indices = new_indices
        self.original_indices = indices
        self.nogil = env.nogil
        self.analyse_operation(env, getting, axes)
        self.wrap_in_nonecheck_node(env)
        return self

    def analyse_operation(self, env, getting, axes):
        if False:
            while True:
                i = 10
        self.none_error_message = 'Cannot index None memoryview slice'
        self.analyse_buffer_index(env, getting)

    def analyse_broadcast_operation(self, rhs):
        if False:
            while True:
                i = 10
        '\n        Support broadcasting for slice assignment.\n        E.g.\n            m_2d[...] = m_1d  # or,\n            m_1d[...] = m_2d  # if the leading dimension has extent 1\n        '
        if self.type.is_memoryviewslice:
            lhs = self
            if lhs.is_memview_broadcast or rhs.is_memview_broadcast:
                lhs.is_memview_broadcast = True
                rhs.is_memview_broadcast = True

    def analyse_as_memview_scalar_assignment(self, rhs):
        if False:
            i = 10
            return i + 15
        lhs = self.analyse_assignment(rhs)
        if lhs:
            rhs.is_memview_copy_assignment = lhs.is_memview_copy_assignment
            return lhs
        return self

class MemoryViewSliceNode(MemoryViewIndexNode):
    is_memview_slice = True
    is_ellipsis_noop = False
    is_memview_scalar_assignment = False
    is_memview_index = False
    is_memview_broadcast = False

    def analyse_ellipsis_noop(self, env, getting):
        if False:
            while True:
                i = 10
        'Slicing operations needing no evaluation, i.e. m[...] or m[:, :]'
        self.is_ellipsis_noop = all((index.is_slice and index.start.is_none and index.stop.is_none and index.step.is_none for index in self.indices))
        if self.is_ellipsis_noop:
            self.type = self.base.type

    def analyse_operation(self, env, getting, axes):
        if False:
            print('Hello World!')
        from . import MemoryView
        if not getting:
            self.is_memview_broadcast = True
            self.none_error_message = 'Cannot assign to None memoryview slice'
        else:
            self.none_error_message = 'Cannot slice None memoryview slice'
        self.analyse_ellipsis_noop(env, getting)
        if self.is_ellipsis_noop:
            return
        self.index = None
        self.is_temp = True
        self.use_managed_ref = True
        if not MemoryView.validate_axes(self.pos, axes):
            self.type = error_type
            return
        self.type = PyrexTypes.MemoryViewSliceType(self.base.type.dtype, axes)
        if not (self.base.is_simple() or self.base.result_in_temp()):
            self.base = self.base.coerce_to_temp(env)

    def analyse_assignment(self, rhs):
        if False:
            i = 10
            return i + 15
        if not rhs.type.is_memoryviewslice and (self.type.dtype.assignable_from(rhs.type) or rhs.type.is_pyobject):
            return MemoryCopyScalar(self.pos, self)
        else:
            return MemoryCopySlice(self.pos, self)

    def merged_indices(self, indices):
        if False:
            while True:
                i = 10
        'Return a new list of indices/slices with \'indices\' merged into the current ones\n        according to slicing rules.\n        Is used to implement "view[i][j]" => "view[i, j]".\n        Return None if the indices cannot (easily) be merged at compile time.\n        '
        if not indices:
            return None
        new_indices = self.original_indices[:]
        indices = indices[:]
        for (i, s) in enumerate(self.original_indices):
            if s.is_slice:
                if s.start.is_none and s.stop.is_none and s.step.is_none:
                    new_indices[i] = indices[0]
                    indices.pop(0)
                    if not indices:
                        return new_indices
                else:
                    return None
            elif not s.type.is_int:
                return None
        if indices:
            if len(new_indices) + len(indices) > self.base.type.ndim:
                return None
            new_indices += indices
        return new_indices

    def is_simple(self):
        if False:
            i = 10
            return i + 15
        if self.is_ellipsis_noop:
            return self.base.is_simple() or self.base.result_in_temp()
        return self.result_in_temp()

    def calculate_result_code(self):
        if False:
            print('Hello World!')
        'This is called in case this is a no-op slicing node'
        return self.base.result()

    def generate_result_code(self, code):
        if False:
            print('Hello World!')
        if self.is_ellipsis_noop:
            return
        buffer_entry = self.buffer_entry()
        have_gil = not self.in_nogil_context
        have_slices = False
        it = iter(self.indices)
        for index in self.original_indices:
            if index.is_slice:
                have_slices = True
                if not index.start.is_none:
                    index.start = next(it)
                if not index.stop.is_none:
                    index.stop = next(it)
                if not index.step.is_none:
                    index.step = next(it)
            else:
                next(it)
        assert not list(it)
        buffer_entry.generate_buffer_slice_code(code, self.original_indices, self.result(), self.type, have_gil=have_gil, have_slices=have_slices, directives=code.globalstate.directives)

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False):
        if False:
            return 10
        if self.is_ellipsis_noop:
            self.generate_subexpr_evaluation_code(code)
        else:
            self.generate_evaluation_code(code)
        if self.is_memview_scalar_assignment:
            self.generate_memoryviewslice_assign_scalar_code(rhs, code)
        else:
            self.generate_memoryviewslice_setslice_code(rhs, code)
        if self.is_ellipsis_noop:
            self.generate_subexpr_disposal_code(code)
        else:
            self.generate_disposal_code(code)
        rhs.generate_disposal_code(code)
        rhs.free_temps(code)

class MemoryCopyNode(ExprNode):
    """
    Wraps a memoryview slice for slice assignment.

        dst: destination mememoryview slice
    """
    subexprs = ['dst']

    def __init__(self, pos, dst):
        if False:
            while True:
                i = 10
        super(MemoryCopyNode, self).__init__(pos)
        self.dst = dst
        self.type = dst.type

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False):
        if False:
            return 10
        self.dst.generate_evaluation_code(code)
        self._generate_assignment_code(rhs, code)
        self.dst.generate_disposal_code(code)
        self.dst.free_temps(code)
        rhs.generate_disposal_code(code)
        rhs.free_temps(code)

class MemoryCopySlice(MemoryCopyNode):
    """
    Copy the contents of slice src to slice dst. Does not support indirect
    slices.

        memslice1[...] = memslice2
        memslice1[:] = memslice2
    """
    is_memview_copy_assignment = True
    copy_slice_cname = '__pyx_memoryview_copy_contents'

    def _generate_assignment_code(self, src, code):
        if False:
            for i in range(10):
                print('nop')
        dst = self.dst
        src.type.assert_direct_dims(src.pos)
        dst.type.assert_direct_dims(dst.pos)
        code.putln(code.error_goto_if_neg('%s(%s, %s, %d, %d, %d)' % (self.copy_slice_cname, src.result(), dst.result(), src.type.ndim, dst.type.ndim, dst.type.dtype.is_pyobject), dst.pos))

class MemoryCopyScalar(MemoryCopyNode):
    """
    Assign a scalar to a slice. dst must be simple, scalar will be assigned
    to a correct type and not just something assignable.

        memslice1[...] = 0.0
        memslice1[:] = 0.0
    """

    def __init__(self, pos, dst):
        if False:
            i = 10
            return i + 15
        super(MemoryCopyScalar, self).__init__(pos, dst)
        self.type = dst.type.dtype

    def _generate_assignment_code(self, scalar, code):
        if False:
            return 10
        from . import MemoryView
        self.dst.type.assert_direct_dims(self.dst.pos)
        dtype = self.dst.type.dtype
        type_decl = dtype.declaration_code('')
        slice_decl = self.dst.type.declaration_code('')
        code.begin_block()
        code.putln('%s __pyx_temp_scalar = %s;' % (type_decl, scalar.result()))
        if self.dst.result_in_temp() or self.dst.is_simple():
            dst_temp = self.dst.result()
        else:
            code.putln('%s __pyx_temp_slice = %s;' % (slice_decl, self.dst.result()))
            dst_temp = '__pyx_temp_slice'
        force_strided = False
        indices = self.dst.original_indices
        for idx in indices:
            if isinstance(idx, SliceNode) and (not (idx.start.is_none and idx.stop.is_none and idx.step.is_none)):
                force_strided = True
        slice_iter_obj = MemoryView.slice_iter(self.dst.type, dst_temp, self.dst.type.ndim, code, force_strided=force_strided)
        p = slice_iter_obj.start_loops()
        if dtype.is_pyobject:
            code.putln('Py_DECREF(*(PyObject **) %s);' % p)
        code.putln('*((%s *) %s) = __pyx_temp_scalar;' % (type_decl, p))
        if dtype.is_pyobject:
            code.putln('Py_INCREF(__pyx_temp_scalar);')
        slice_iter_obj.end_loops()
        code.end_block()

class SliceIndexNode(ExprNode):
    subexprs = ['base', 'start', 'stop', 'slice']
    nogil = False
    slice = None

    def infer_type(self, env):
        if False:
            i = 10
            return i + 15
        base_type = self.base.infer_type(env)
        if base_type.is_string or base_type.is_cpp_class:
            return bytes_type
        elif base_type.is_pyunicode_ptr:
            return unicode_type
        elif base_type in (bytes_type, bytearray_type, str_type, unicode_type, basestring_type, list_type, tuple_type):
            return base_type
        elif base_type.is_ptr or base_type.is_array:
            return PyrexTypes.c_array_type(base_type.base_type, None)
        return py_object_type

    def inferable_item_node(self, index=0):
        if False:
            while True:
                i = 10
        if index is not not_a_constant and self.start:
            if self.start.has_constant_result():
                index += self.start.constant_result
            else:
                index = not_a_constant
        return self.base.inferable_item_node(index)

    def may_be_none(self):
        if False:
            print('Hello World!')
        base_type = self.base.type
        if base_type:
            if base_type.is_string:
                return False
            if base_type in (bytes_type, str_type, unicode_type, basestring_type, list_type, tuple_type):
                return False
        return ExprNode.may_be_none(self)

    def calculate_constant_result(self):
        if False:
            i = 10
            return i + 15
        if self.start is None:
            start = None
        else:
            start = self.start.constant_result
        if self.stop is None:
            stop = None
        else:
            stop = self.stop.constant_result
        self.constant_result = self.base.constant_result[start:stop]

    def compile_time_value(self, denv):
        if False:
            i = 10
            return i + 15
        base = self.base.compile_time_value(denv)
        if self.start is None:
            start = 0
        else:
            start = self.start.compile_time_value(denv)
        if self.stop is None:
            stop = None
        else:
            stop = self.stop.compile_time_value(denv)
        try:
            return base[start:stop]
        except Exception as e:
            self.compile_time_value_error(e)

    def analyse_target_declaration(self, env):
        if False:
            print('Hello World!')
        pass

    def analyse_target_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        node = self.analyse_types(env, getting=False)
        if node.type.is_pyobject:
            node.type = py_object_type
        return node

    def analyse_types(self, env, getting=True):
        if False:
            while True:
                i = 10
        self.base = self.base.analyse_types(env)
        if self.base.type.is_buffer or self.base.type.is_pythran_expr or self.base.type.is_memoryviewslice:
            none_node = NoneNode(self.pos)
            index = SliceNode(self.pos, start=self.start or none_node, stop=self.stop or none_node, step=none_node)
            index_node = IndexNode(self.pos, index=index, base=self.base)
            return index_node.analyse_base_and_index_types(env, getting=getting, setting=not getting, analyse_base=False)
        if self.start:
            self.start = self.start.analyse_types(env)
        if self.stop:
            self.stop = self.stop.analyse_types(env)
        if not env.directives['wraparound']:
            check_negative_indices(self.start, self.stop)
        base_type = self.base.type
        if base_type.is_array and (not getting):
            if not self.start and (not self.stop):
                self.type = base_type
            else:
                self.type = PyrexTypes.CPtrType(base_type.base_type)
        elif base_type.is_string or base_type.is_cpp_string:
            self.type = default_str_type(env)
        elif base_type.is_pyunicode_ptr:
            self.type = unicode_type
        elif base_type.is_ptr:
            self.type = base_type
        elif base_type.is_array:
            self.type = PyrexTypes.CPtrType(base_type.base_type)
        else:
            self.base = self.base.coerce_to_pyobject(env)
            self.type = py_object_type
        if base_type.is_builtin_type:
            self.type = base_type
            self.base = self.base.as_none_safe_node("'NoneType' object is not subscriptable")
        if self.type is py_object_type:
            if (not self.start or self.start.is_literal) and (not self.stop or self.stop.is_literal):
                none_node = NoneNode(self.pos)
                self.slice = SliceNode(self.pos, start=copy.deepcopy(self.start or none_node), stop=copy.deepcopy(self.stop or none_node), step=none_node).analyse_types(env)
        else:
            c_int = PyrexTypes.c_py_ssize_t_type

            def allow_none(node, default_value, env):
                if False:
                    print('Hello World!')
                from .UtilNodes import EvalWithTempExprNode, ResultRefNode
                node_ref = ResultRefNode(node)
                new_expr = CondExprNode(node.pos, true_val=IntNode(node.pos, type=c_int, value=default_value, constant_result=int(default_value) if default_value.isdigit() else not_a_constant), false_val=node_ref.coerce_to(c_int, env), test=PrimaryCmpNode(node.pos, operand1=node_ref, operator='is', operand2=NoneNode(node.pos)).analyse_types(env)).analyse_result_type(env)
                return EvalWithTempExprNode(node_ref, new_expr)
            if self.start:
                if self.start.type.is_pyobject:
                    self.start = allow_none(self.start, '0', env)
                self.start = self.start.coerce_to(c_int, env)
            if self.stop:
                if self.stop.type.is_pyobject:
                    self.stop = allow_none(self.stop, 'PY_SSIZE_T_MAX', env)
                self.stop = self.stop.coerce_to(c_int, env)
        self.is_temp = 1
        return self

    def analyse_as_type(self, env):
        if False:
            i = 10
            return i + 15
        base_type = self.base.analyse_as_type(env)
        if base_type:
            if not self.start and (not self.stop):
                from . import MemoryView
                env.use_utility_code(MemoryView.view_utility_code)
                none_node = NoneNode(self.pos)
                slice_node = SliceNode(self.pos, start=none_node, stop=none_node, step=none_node)
                return PyrexTypes.MemoryViewSliceType(base_type, MemoryView.get_axes_specs(env, [slice_node]))
        return None

    def nogil_check(self, env):
        if False:
            for i in range(10):
                print('nop')
        self.nogil = env.nogil
        return super(SliceIndexNode, self).nogil_check(env)
    gil_message = 'Slicing Python object'
    get_slice_utility_code = TempitaUtilityCode.load('SliceObject', 'ObjectHandling.c', context={'access': 'Get'})
    set_slice_utility_code = TempitaUtilityCode.load('SliceObject', 'ObjectHandling.c', context={'access': 'Set'})

    def coerce_to(self, dst_type, env):
        if False:
            while True:
                i = 10
        if (self.base.type.is_string or self.base.type.is_cpp_string) and dst_type in (bytes_type, bytearray_type, str_type, unicode_type):
            if dst_type not in (bytes_type, bytearray_type) and (not env.directives['c_string_encoding']):
                error(self.pos, "default encoding required for conversion from '%s' to '%s'" % (self.base.type, dst_type))
            self.type = dst_type
        if dst_type.is_array and self.base.type.is_array:
            if not self.start and (not self.stop):
                return self.base.coerce_to(dst_type, env)
        return super(SliceIndexNode, self).coerce_to(dst_type, env)

    def generate_result_code(self, code):
        if False:
            i = 10
            return i + 15
        if not self.type.is_pyobject:
            error(self.pos, "Slicing is not currently supported for '%s'." % self.type)
            return
        base_result = self.base.result()
        result = self.result()
        start_code = self.start_code()
        stop_code = self.stop_code()
        if self.base.type.is_string:
            base_result = self.base.result()
            if self.base.type not in (PyrexTypes.c_char_ptr_type, PyrexTypes.c_const_char_ptr_type):
                base_result = '((const char*)%s)' % base_result
            if self.type is bytearray_type:
                type_name = 'ByteArray'
            else:
                type_name = self.type.name.title()
            if self.stop is None:
                code.putln('%s = __Pyx_Py%s_FromString(%s + %s); %s' % (result, type_name, base_result, start_code, code.error_goto_if_null(result, self.pos)))
            else:
                code.putln('%s = __Pyx_Py%s_FromStringAndSize(%s + %s, %s - %s); %s' % (result, type_name, base_result, start_code, stop_code, start_code, code.error_goto_if_null(result, self.pos)))
        elif self.base.type.is_pyunicode_ptr:
            base_result = self.base.result()
            if self.base.type != PyrexTypes.c_py_unicode_ptr_type:
                base_result = '((const Py_UNICODE*)%s)' % base_result
            if self.stop is None:
                code.putln('%s = __Pyx_PyUnicode_FromUnicode(%s + %s); %s' % (result, base_result, start_code, code.error_goto_if_null(result, self.pos)))
            else:
                code.putln('%s = __Pyx_PyUnicode_FromUnicodeAndLength(%s + %s, %s - %s); %s' % (result, base_result, start_code, stop_code, start_code, code.error_goto_if_null(result, self.pos)))
        elif self.base.type is unicode_type:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyUnicode_Substring', 'StringTools.c'))
            code.putln('%s = __Pyx_PyUnicode_Substring(%s, %s, %s); %s' % (result, base_result, start_code, stop_code, code.error_goto_if_null(result, self.pos)))
        elif self.type is py_object_type:
            code.globalstate.use_utility_code(self.get_slice_utility_code)
            (has_c_start, has_c_stop, c_start, c_stop, py_start, py_stop, py_slice) = self.get_slice_config()
            code.putln('%s = __Pyx_PyObject_GetSlice(%s, %s, %s, %s, %s, %s, %d, %d, %d); %s' % (result, self.base.py_result(), c_start, c_stop, py_start, py_stop, py_slice, has_c_start, has_c_stop, bool(code.globalstate.directives['wraparound']), code.error_goto_if_null(result, self.pos)))
        else:
            if self.base.type is list_type:
                code.globalstate.use_utility_code(TempitaUtilityCode.load_cached('SliceTupleAndList', 'ObjectHandling.c'))
                cfunc = '__Pyx_PyList_GetSlice'
            elif self.base.type is tuple_type:
                code.globalstate.use_utility_code(TempitaUtilityCode.load_cached('SliceTupleAndList', 'ObjectHandling.c'))
                cfunc = '__Pyx_PyTuple_GetSlice'
            else:
                cfunc = 'PySequence_GetSlice'
            code.putln('%s = %s(%s, %s, %s); %s' % (result, cfunc, self.base.py_result(), start_code, stop_code, code.error_goto_if_null(result, self.pos)))
        self.generate_gotref(code)

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False, exception_check=None, exception_value=None):
        if False:
            while True:
                i = 10
        self.generate_subexpr_evaluation_code(code)
        if self.type.is_pyobject:
            code.globalstate.use_utility_code(self.set_slice_utility_code)
            (has_c_start, has_c_stop, c_start, c_stop, py_start, py_stop, py_slice) = self.get_slice_config()
            code.put_error_if_neg(self.pos, '__Pyx_PyObject_SetSlice(%s, %s, %s, %s, %s, %s, %s, %d, %d, %d)' % (self.base.py_result(), rhs.py_result(), c_start, c_stop, py_start, py_stop, py_slice, has_c_start, has_c_stop, bool(code.globalstate.directives['wraparound'])))
        else:
            start_offset = self.start_code() if self.start else '0'
            if rhs.type.is_array:
                array_length = rhs.type.size
                self.generate_slice_guard_code(code, array_length)
            else:
                array_length = '%s - %s' % (self.stop_code(), start_offset)
            code.globalstate.use_utility_code(UtilityCode.load_cached('IncludeStringH', 'StringTools.c'))
            code.putln('memcpy(&(%s[%s]), %s, sizeof(%s[0]) * (%s));' % (self.base.result(), start_offset, rhs.result(), self.base.result(), array_length))
        self.generate_subexpr_disposal_code(code)
        self.free_subexpr_temps(code)
        rhs.generate_disposal_code(code)
        rhs.free_temps(code)

    def generate_deletion_code(self, code, ignore_nonexisting=False):
        if False:
            return 10
        if not self.base.type.is_pyobject:
            error(self.pos, "Deleting slices is only supported for Python types, not '%s'." % self.type)
            return
        self.generate_subexpr_evaluation_code(code)
        code.globalstate.use_utility_code(self.set_slice_utility_code)
        (has_c_start, has_c_stop, c_start, c_stop, py_start, py_stop, py_slice) = self.get_slice_config()
        code.put_error_if_neg(self.pos, '__Pyx_PyObject_DelSlice(%s, %s, %s, %s, %s, %s, %d, %d, %d)' % (self.base.py_result(), c_start, c_stop, py_start, py_stop, py_slice, has_c_start, has_c_stop, bool(code.globalstate.directives['wraparound'])))
        self.generate_subexpr_disposal_code(code)
        self.free_subexpr_temps(code)

    def get_slice_config(self):
        if False:
            return 10
        (has_c_start, c_start, py_start) = (False, '0', 'NULL')
        if self.start:
            has_c_start = not self.start.type.is_pyobject
            if has_c_start:
                c_start = self.start.result()
            else:
                py_start = '&%s' % self.start.py_result()
        (has_c_stop, c_stop, py_stop) = (False, '0', 'NULL')
        if self.stop:
            has_c_stop = not self.stop.type.is_pyobject
            if has_c_stop:
                c_stop = self.stop.result()
            else:
                py_stop = '&%s' % self.stop.py_result()
        py_slice = self.slice and '&%s' % self.slice.py_result() or 'NULL'
        return (has_c_start, has_c_stop, c_start, c_stop, py_start, py_stop, py_slice)

    def generate_slice_guard_code(self, code, target_size):
        if False:
            return 10
        if not self.base.type.is_array:
            return
        slice_size = self.base.type.size
        try:
            total_length = slice_size = int(slice_size)
        except ValueError:
            total_length = None
        start = stop = None
        if self.stop:
            stop = self.stop.result()
            try:
                stop = int(stop)
                if stop < 0:
                    if total_length is None:
                        slice_size = '%s + %d' % (slice_size, stop)
                    else:
                        slice_size += stop
                else:
                    slice_size = stop
                stop = None
            except ValueError:
                pass
        if self.start:
            start = self.start.result()
            try:
                start = int(start)
                if start < 0:
                    if total_length is None:
                        start = '%s + %d' % (self.base.type.size, start)
                    else:
                        start += total_length
                if isinstance(slice_size, _py_int_types):
                    slice_size -= start
                else:
                    slice_size = '%s - (%s)' % (slice_size, start)
                start = None
            except ValueError:
                pass
        runtime_check = None
        compile_time_check = False
        try:
            int_target_size = int(target_size)
        except ValueError:
            int_target_size = None
        else:
            compile_time_check = isinstance(slice_size, _py_int_types)
        if compile_time_check and slice_size < 0:
            if int_target_size > 0:
                error(self.pos, 'Assignment to empty slice.')
        elif compile_time_check and start is None and (stop is None):
            if int_target_size != slice_size:
                error(self.pos, 'Assignment to slice of wrong length, expected %s, got %s' % (slice_size, target_size))
        elif start is not None:
            if stop is None:
                stop = slice_size
            runtime_check = '(%s)-(%s)' % (stop, start)
        elif stop is not None:
            runtime_check = stop
        else:
            runtime_check = slice_size
        if runtime_check:
            code.putln('if (unlikely((%s) != (%s))) {' % (runtime_check, target_size))
            if self.nogil:
                code.put_ensure_gil()
            code.putln('PyErr_Format(PyExc_ValueError, "Assignment to slice of wrong length, expected %%" CYTHON_FORMAT_SSIZE_T "d, got %%" CYTHON_FORMAT_SSIZE_T "d", (Py_ssize_t)(%s), (Py_ssize_t)(%s));' % (target_size, runtime_check))
            if self.nogil:
                code.put_release_ensured_gil()
            code.putln(code.error_goto(self.pos))
            code.putln('}')

    def start_code(self):
        if False:
            for i in range(10):
                print('nop')
        if self.start:
            return self.start.result()
        else:
            return '0'

    def stop_code(self):
        if False:
            return 10
        if self.stop:
            return self.stop.result()
        elif self.base.type.is_array:
            return self.base.type.size
        else:
            return 'PY_SSIZE_T_MAX'

    def calculate_result_code(self):
        if False:
            i = 10
            return i + 15
        return '<unused>'

class SliceNode(ExprNode):
    subexprs = ['start', 'stop', 'step']
    is_slice = True
    type = slice_type
    is_temp = 1

    def calculate_constant_result(self):
        if False:
            i = 10
            return i + 15
        self.constant_result = slice(self.start.constant_result, self.stop.constant_result, self.step.constant_result)

    def compile_time_value(self, denv):
        if False:
            return 10
        start = self.start.compile_time_value(denv)
        stop = self.stop.compile_time_value(denv)
        step = self.step.compile_time_value(denv)
        try:
            return slice(start, stop, step)
        except Exception as e:
            self.compile_time_value_error(e)

    def may_be_none(self):
        if False:
            while True:
                i = 10
        return False

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        start = self.start.analyse_types(env)
        stop = self.stop.analyse_types(env)
        step = self.step.analyse_types(env)
        self.start = start.coerce_to_pyobject(env)
        self.stop = stop.coerce_to_pyobject(env)
        self.step = step.coerce_to_pyobject(env)
        if self.start.is_literal and self.stop.is_literal and self.step.is_literal:
            self.is_literal = True
            self.is_temp = False
        return self
    gil_message = 'Constructing Python slice object'

    def calculate_result_code(self):
        if False:
            for i in range(10):
                print('nop')
        return self.result_code

    def generate_result_code(self, code):
        if False:
            i = 10
            return i + 15
        if self.is_literal:
            dedup_key = make_dedup_key(self.type, (self,))
            self.result_code = code.get_py_const(py_object_type, 'slice', cleanup_level=2, dedup_key=dedup_key)
            code = code.get_cached_constants_writer(self.result_code)
            if code is None:
                return
            code.mark_pos(self.pos)
        code.putln('%s = PySlice_New(%s, %s, %s); %s' % (self.result(), self.start.py_result(), self.stop.py_result(), self.step.py_result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)
        if self.is_literal:
            self.generate_giveref(code)

class SliceIntNode(SliceNode):
    is_temp = 0

    def calculate_constant_result(self):
        if False:
            print('Hello World!')
        self.constant_result = slice(self.start.constant_result, self.stop.constant_result, self.step.constant_result)

    def compile_time_value(self, denv):
        if False:
            while True:
                i = 10
        start = self.start.compile_time_value(denv)
        stop = self.stop.compile_time_value(denv)
        step = self.step.compile_time_value(denv)
        try:
            return slice(start, stop, step)
        except Exception as e:
            self.compile_time_value_error(e)

    def may_be_none(self):
        if False:
            print('Hello World!')
        return False

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        self.start = self.start.analyse_types(env)
        self.stop = self.stop.analyse_types(env)
        self.step = self.step.analyse_types(env)
        if not self.start.is_none:
            self.start = self.start.coerce_to_integer(env)
        if not self.stop.is_none:
            self.stop = self.stop.coerce_to_integer(env)
        if not self.step.is_none:
            self.step = self.step.coerce_to_integer(env)
        if self.start.is_literal and self.stop.is_literal and self.step.is_literal:
            self.is_literal = True
            self.is_temp = False
        return self

    def calculate_result_code(self):
        if False:
            return 10
        pass

    def generate_result_code(self, code):
        if False:
            i = 10
            return i + 15
        for a in (self.start, self.stop, self.step):
            if isinstance(a, CloneNode):
                a.arg.result()

class CallNode(ExprNode):
    may_return_none = None

    def infer_type(self, env):
        if False:
            while True:
                i = 10
        function = self.function
        func_type = function.infer_type(env)
        if isinstance(function, NewExprNode):
            return PyrexTypes.CPtrType(function.class_type)
        if func_type is py_object_type:
            entry = getattr(function, 'entry', None)
            if entry is not None:
                func_type = entry.type or func_type
        if func_type.is_ptr:
            func_type = func_type.base_type
        if func_type.is_cfunction:
            if getattr(self.function, 'entry', None) and hasattr(self, 'args'):
                alternatives = self.function.entry.all_alternatives()
                arg_types = [arg.infer_type(env) for arg in self.args]
                func_entry = PyrexTypes.best_match(arg_types, alternatives)
                if func_entry:
                    func_type = func_entry.type
                    if func_type.is_ptr:
                        func_type = func_type.base_type
                    return func_type.return_type
            return func_type.return_type
        elif func_type is type_type:
            if function.is_name and function.entry and function.entry.type:
                result_type = function.entry.type
                if result_type.is_extension_type:
                    return result_type
                elif result_type.is_builtin_type:
                    if function.entry.name == 'float':
                        return PyrexTypes.c_double_type
                    elif function.entry.name in Builtin.types_that_construct_their_instance:
                        return result_type
        func_type = self.function.analyse_as_type(env)
        if func_type and (func_type.is_struct_or_union or func_type.is_cpp_class):
            return func_type
        return py_object_type

    def type_dependencies(self, env):
        if False:
            return 10
        return self.function.type_dependencies(env)

    def is_simple(self):
        if False:
            while True:
                i = 10
        return False

    def may_be_none(self):
        if False:
            for i in range(10):
                print('nop')
        if self.may_return_none is not None:
            return self.may_return_none
        func_type = self.function.type
        if func_type is type_type and self.function.is_name:
            entry = self.function.entry
            if entry.type.is_extension_type:
                return False
            if entry.type.is_builtin_type and entry.name in Builtin.types_that_construct_their_instance:
                return False
        return ExprNode.may_be_none(self)

    def set_py_result_type(self, function, func_type=None):
        if False:
            for i in range(10):
                print('nop')
        if func_type is None:
            func_type = function.type
        if func_type is Builtin.type_type and (function.is_name and function.entry and function.entry.is_builtin and (function.entry.name in Builtin.types_that_construct_their_instance)):
            if function.entry.name == 'float':
                self.type = PyrexTypes.c_double_type
                self.result_ctype = PyrexTypes.c_double_type
            else:
                self.type = Builtin.builtin_types[function.entry.name]
                self.result_ctype = py_object_type
            self.may_return_none = False
        elif function.is_name and function.type_entry:
            self.type = function.type_entry.type
            self.result_ctype = py_object_type
            self.may_return_none = False
        else:
            self.type = py_object_type

    def analyse_as_type_constructor(self, env):
        if False:
            while True:
                i = 10
        type = self.function.analyse_as_type(env)
        if type and type.is_struct_or_union:
            (args, kwds) = self.explicit_args_kwds()
            items = []
            for (arg, member) in zip(args, type.scope.var_entries):
                items.append(DictItemNode(pos=arg.pos, key=StringNode(pos=arg.pos, value=member.name), value=arg))
            if kwds:
                items += kwds.key_value_pairs
            self.key_value_pairs = items
            self.__class__ = DictNode
            self.analyse_types(env)
            self.coerce_to(type, env)
            return True
        elif type and type.is_cpp_class:
            self.args = [arg.analyse_types(env) for arg in self.args]
            constructor = type.scope.lookup('<init>')
            if not constructor:
                error(self.function.pos, "no constructor found for C++  type '%s'" % self.function.name)
                self.type = error_type
                return self
            self.function = RawCNameExprNode(self.function.pos, constructor.type)
            self.function.entry = constructor
            self.function.set_cname(type.empty_declaration_code())
            self.analyse_c_function_call(env)
            self.type = type
            return True

    def is_lvalue(self):
        if False:
            while True:
                i = 10
        return self.type.is_reference

    def nogil_check(self, env):
        if False:
            return 10
        func_type = self.function_type()
        if func_type.is_pyobject:
            self.gil_error()
        elif not func_type.is_error and (not getattr(func_type, 'nogil', False)):
            self.gil_error()
    gil_message = 'Calling gil-requiring function'

class SimpleCallNode(CallNode):
    subexprs = ['self', 'coerced_self', 'function', 'args', 'arg_tuple']
    self = None
    coerced_self = None
    arg_tuple = None
    wrapper_call = False
    has_optional_args = False
    nogil = False
    analysed = False
    overflowcheck = False

    def compile_time_value(self, denv):
        if False:
            print('Hello World!')
        function = self.function.compile_time_value(denv)
        args = [arg.compile_time_value(denv) for arg in self.args]
        try:
            return function(*args)
        except Exception as e:
            self.compile_time_value_error(e)

    @classmethod
    def for_cproperty(cls, pos, obj, entry):
        if False:
            while True:
                i = 10
        property_scope = entry.scope
        getter_entry = property_scope.lookup_here(entry.name)
        assert getter_entry, 'Getter not found in scope %s: %s' % (property_scope, property_scope.entries)
        function = NameNode(pos, name=entry.name, entry=getter_entry, type=getter_entry.type)
        node = cls(pos, function=function, args=[obj])
        return node

    def analyse_as_type(self, env):
        if False:
            return 10
        attr = self.function.as_cython_attribute()
        if attr == 'pointer':
            if len(self.args) != 1:
                error(self.args.pos, 'only one type allowed.')
            else:
                type = self.args[0].analyse_as_type(env)
                if not type:
                    error(self.args[0].pos, 'Unknown type')
                else:
                    return PyrexTypes.CPtrType(type)
        elif attr == 'typeof':
            if len(self.args) != 1:
                error(self.args.pos, 'only one type allowed.')
            operand = self.args[0].analyse_types(env)
            return operand.type

    def explicit_args_kwds(self):
        if False:
            print('Hello World!')
        return (self.args, None)

    def analyse_types(self, env):
        if False:
            return 10
        if self.analysed:
            return self
        self.analysed = True
        if self.analyse_as_type_constructor(env):
            return self
        self.function.is_called = 1
        self.function = self.function.analyse_types(env)
        function = self.function
        if function.is_attribute and function.entry and function.entry.is_cmethod:
            self.self = function.obj
            function.obj = CloneNode(self.self)
        func_type = self.function_type()
        self.is_numpy_call_with_exprs = False
        if has_np_pythran(env) and function.is_numpy_attribute and pythran_is_numpy_func_supported(function):
            has_pythran_args = True
            self.arg_tuple = TupleNode(self.pos, args=self.args)
            self.arg_tuple = self.arg_tuple.analyse_types(env)
            for arg in self.arg_tuple.args:
                has_pythran_args &= is_pythran_supported_node_or_none(arg)
            self.is_numpy_call_with_exprs = bool(has_pythran_args)
        if self.is_numpy_call_with_exprs:
            env.add_include_file(pythran_get_func_include_file(function))
            return NumPyMethodCallNode.from_node(self, function_cname=pythran_functor(function), arg_tuple=self.arg_tuple, type=PythranExpr(pythran_func_type(function, self.arg_tuple.args)))
        elif func_type.is_pyobject:
            self.arg_tuple = TupleNode(self.pos, args=self.args)
            self.arg_tuple = self.arg_tuple.analyse_types(env).coerce_to_pyobject(env)
            self.args = None
            self.set_py_result_type(function, func_type)
            self.is_temp = 1
        else:
            self.args = [arg.analyse_types(env) for arg in self.args]
            self.analyse_c_function_call(env)
            if func_type.exception_check == '+':
                self.is_temp = True
        return self

    def function_type(self):
        if False:
            print('Hello World!')
        func_type = self.function.type
        if func_type.is_ptr:
            func_type = func_type.base_type
        return func_type

    def analyse_c_function_call(self, env):
        if False:
            for i in range(10):
                print('nop')
        func_type = self.function.type
        if func_type is error_type:
            self.type = error_type
            return
        if func_type.is_cfunction and func_type.is_static_method:
            if self.self and self.self.type.is_extension_type:
                error(self.pos, 'Cannot call a static method on an instance variable.')
            args = self.args
        elif self.self:
            args = [self.self] + self.args
        else:
            args = self.args
        if func_type.is_cpp_class:
            overloaded_entry = self.function.type.scope.lookup('operator()')
            if overloaded_entry is None:
                self.type = PyrexTypes.error_type
                self.result_code = '<error>'
                return
        elif hasattr(self.function, 'entry'):
            overloaded_entry = self.function.entry
        elif self.function.is_subscript and self.function.is_fused_index:
            overloaded_entry = self.function.type.entry
        else:
            overloaded_entry = None
        if overloaded_entry:
            if self.function.type.is_fused:
                functypes = self.function.type.get_all_specialized_function_types()
                alternatives = [f.entry for f in functypes]
            else:
                alternatives = overloaded_entry.all_alternatives()
            entry = PyrexTypes.best_match([arg.type for arg in args], alternatives, self.pos, env, args)
            if not entry:
                self.type = PyrexTypes.error_type
                self.result_code = '<error>'
                return
            entry.used = True
            if not func_type.is_cpp_class:
                self.function.entry = entry
            self.function.type = entry.type
            func_type = self.function_type()
        else:
            entry = None
            func_type = self.function_type()
            if not func_type.is_cfunction:
                error(self.pos, "Calling non-function type '%s'" % func_type)
                self.type = PyrexTypes.error_type
                self.result_code = '<error>'
                return
        max_nargs = len(func_type.args)
        expected_nargs = max_nargs - func_type.optional_arg_count
        actual_nargs = len(args)
        if func_type.optional_arg_count and expected_nargs != actual_nargs:
            self.has_optional_args = 1
            self.is_temp = 1
        if entry and entry.is_cmethod and func_type.args and (not func_type.is_static_method):
            formal_arg = func_type.args[0]
            arg = args[0]
            if formal_arg.not_none:
                if self.self:
                    self.self = self.self.as_none_safe_node("'NoneType' object has no attribute '%{0}s'".format('.30' if len(entry.name) <= 30 else ''), error='PyExc_AttributeError', format_args=[entry.name])
                else:
                    arg = arg.as_none_safe_node("descriptor '%s' requires a '%s' object but received a 'NoneType'", format_args=[entry.name, formal_arg.type.name])
            if self.self:
                if formal_arg.accept_builtin_subtypes:
                    arg = CMethodSelfCloneNode(self.self)
                else:
                    arg = CloneNode(self.self)
                arg = self.coerced_self = arg.coerce_to(formal_arg.type, env)
            elif formal_arg.type.is_builtin_type:
                arg = arg.coerce_to(formal_arg.type, env)
                if arg.type.is_builtin_type and isinstance(arg, PyTypeTestNode):
                    arg.exact_builtin_type = False
            args[0] = arg
        some_args_in_temps = False
        for i in range(min(max_nargs, actual_nargs)):
            formal_arg = func_type.args[i]
            formal_type = formal_arg.type
            arg = args[i].coerce_to(formal_type, env)
            if formal_arg.not_none:
                arg = arg.as_none_safe_node("cannot pass None into a C function argument that is declared 'not None'")
            if arg.is_temp:
                if i > 0:
                    some_args_in_temps = True
            elif arg.type.is_pyobject and (not env.nogil):
                if i == 0 and self.self is not None:
                    pass
                elif arg.nonlocally_immutable():
                    pass
                else:
                    if i > 0:
                        some_args_in_temps = True
                    arg = arg.coerce_to_temp(env)
            args[i] = arg
        for i in range(max_nargs, actual_nargs):
            arg = args[i]
            if arg.type.is_pyobject:
                if arg.type is str_type:
                    arg_ctype = PyrexTypes.c_char_ptr_type
                else:
                    arg_ctype = arg.type.default_coerced_ctype()
                if arg_ctype is None:
                    error(self.args[i - 1].pos, 'Python object cannot be passed as a varargs parameter')
                else:
                    args[i] = arg = arg.coerce_to(arg_ctype, env)
            if arg.is_temp and i > 0:
                some_args_in_temps = True
        if some_args_in_temps:
            for i in range(actual_nargs - 1):
                if i == 0 and self.self is not None:
                    continue
                arg = args[i]
                if arg.nonlocally_immutable():
                    pass
                elif arg.type.is_cpp_class:
                    pass
                elif env.nogil and arg.type.is_pyobject:
                    pass
                elif i > 0 or (i == 1 and self.self is not None):
                    warning(arg.pos, 'Argument evaluation order in C function call is undefined and may not be as expected', 0)
                    break
        self.args[:] = args
        if isinstance(self.function, NewExprNode):
            self.type = PyrexTypes.CPtrType(self.function.class_type)
        else:
            self.type = func_type.return_type
        if self.function.is_name or self.function.is_attribute:
            func_entry = self.function.entry
            if func_entry and (func_entry.utility_code or func_entry.utility_code_definition):
                self.is_temp = 1
        if self.type.is_pyobject:
            self.result_ctype = py_object_type
            self.is_temp = 1
        elif func_type.exception_value is not None or func_type.exception_check:
            self.is_temp = 1
        elif self.type.is_memoryviewslice:
            self.is_temp = 1
        if self.is_temp and self.type.is_reference:
            self.type = PyrexTypes.CFakeReferenceType(self.type.ref_base_type)
        if func_type.exception_check == '+':
            if needs_cpp_exception_conversion(func_type):
                env.use_utility_code(UtilityCode.load_cached('CppExceptionConversion', 'CppSupport.cpp'))
        self.overflowcheck = env.directives['overflowcheck']

    def calculate_result_code(self):
        if False:
            i = 10
            return i + 15
        return self.c_call_code()

    def c_call_code(self):
        if False:
            print('Hello World!')
        func_type = self.function_type()
        if self.type is PyrexTypes.error_type or not func_type.is_cfunction:
            return '<error>'
        formal_args = func_type.args
        arg_list_code = []
        args = list(zip(formal_args, self.args))
        max_nargs = len(func_type.args)
        expected_nargs = max_nargs - func_type.optional_arg_count
        actual_nargs = len(self.args)
        for (formal_arg, actual_arg) in args[:expected_nargs]:
            arg_code = actual_arg.move_result_rhs_as(formal_arg.type)
            arg_list_code.append(arg_code)
        if func_type.is_overridable:
            arg_list_code.append(str(int(self.wrapper_call or self.function.entry.is_unbound_cmethod)))
        if func_type.optional_arg_count:
            if expected_nargs == actual_nargs:
                optional_args = 'NULL'
            else:
                optional_args = '&%s' % self.opt_arg_struct
            arg_list_code.append(optional_args)
        for actual_arg in self.args[len(formal_args):]:
            arg_list_code.append(actual_arg.move_result_rhs())
        result = '%s(%s)' % (self.function.result(), ', '.join(arg_list_code))
        return result

    def is_c_result_required(self):
        if False:
            i = 10
            return i + 15
        func_type = self.function_type()
        if not func_type.exception_value or func_type.exception_check == '+':
            return False
        return True

    def generate_evaluation_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        function = self.function
        if function.is_name or function.is_attribute:
            code.globalstate.use_entry_utility_code(function.entry)
        abs_function_cnames = ('abs', 'labs', '__Pyx_abs_longlong')
        is_signed_int = self.type.is_int and self.type.signed
        if self.overflowcheck and is_signed_int and (function.result() in abs_function_cnames):
            code.globalstate.use_utility_code(UtilityCode.load_cached('Common', 'Overflow.c'))
            code.putln('if (unlikely(%s == __PYX_MIN(%s))) {                PyErr_SetString(PyExc_OverflowError,                                "Trying to take the absolute value of the most negative integer is not defined."); %s; }' % (self.args[0].result(), self.args[0].type.empty_declaration_code(), code.error_goto(self.pos)))
        if not function.type.is_pyobject or len(self.arg_tuple.args) > 1 or (self.arg_tuple.args and self.arg_tuple.is_literal):
            super(SimpleCallNode, self).generate_evaluation_code(code)
            return
        arg = self.arg_tuple.args[0] if self.arg_tuple.args else None
        subexprs = (self.self, self.coerced_self, function, arg)
        for subexpr in subexprs:
            if subexpr is not None:
                subexpr.generate_evaluation_code(code)
        code.mark_pos(self.pos)
        assert self.is_temp
        self.allocate_temp_result(code)
        if arg is None:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectCallNoArg', 'ObjectHandling.c'))
            code.putln('%s = __Pyx_PyObject_CallNoArg(%s); %s' % (self.result(), function.py_result(), code.error_goto_if_null(self.result(), self.pos)))
        else:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectCallOneArg', 'ObjectHandling.c'))
            code.putln('%s = __Pyx_PyObject_CallOneArg(%s, %s); %s' % (self.result(), function.py_result(), arg.py_result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)
        for subexpr in subexprs:
            if subexpr is not None:
                subexpr.generate_disposal_code(code)
                subexpr.free_temps(code)

    def generate_result_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        func_type = self.function_type()
        if func_type.is_pyobject:
            arg_code = self.arg_tuple.py_result()
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectCall', 'ObjectHandling.c'))
            code.putln('%s = __Pyx_PyObject_Call(%s, %s, NULL); %s' % (self.result(), self.function.py_result(), arg_code, code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
        elif func_type.is_cfunction:
            nogil = not code.funcstate.gil_owned
            if self.has_optional_args:
                actual_nargs = len(self.args)
                expected_nargs = len(func_type.args) - func_type.optional_arg_count
                self.opt_arg_struct = code.funcstate.allocate_temp(func_type.op_arg_struct.base_type, manage_ref=True)
                code.putln('%s.%s = %s;' % (self.opt_arg_struct, Naming.pyrex_prefix + 'n', len(self.args) - expected_nargs))
                args = list(zip(func_type.args, self.args))
                for (formal_arg, actual_arg) in args[expected_nargs:actual_nargs]:
                    code.putln('%s.%s = %s;' % (self.opt_arg_struct, func_type.opt_arg_cname(formal_arg.name), actual_arg.result_as(formal_arg.type)))
            exc_checks = []
            if self.type.is_pyobject and self.is_temp:
                exc_checks.append('!%s' % self.result())
            elif self.type.is_memoryviewslice:
                assert self.is_temp
                exc_checks.append(self.type.error_condition(self.result()))
            elif func_type.exception_check != '+':
                exc_val = func_type.exception_value
                exc_check = func_type.exception_check
                if exc_val is not None:
                    exc_checks.append('%s == %s' % (self.result(), func_type.return_type.cast_code(exc_val)))
                if exc_check:
                    if nogil:
                        if not exc_checks:
                            PyrexTypes.write_noexcept_performance_hint(self.pos, code.funcstate.scope, function_name=None, void_return=self.type.is_void)
                        code.globalstate.use_utility_code(UtilityCode.load_cached('ErrOccurredWithGIL', 'Exceptions.c'))
                        exc_checks.append('__Pyx_ErrOccurredWithGIL()')
                    else:
                        exc_checks.append('PyErr_Occurred()')
            if self.is_temp or exc_checks:
                rhs = self.c_call_code()
                if self.result():
                    lhs = '%s = ' % self.result()
                    if self.is_temp and self.type.is_pyobject:
                        rhs = typecast(py_object_type, self.type, rhs)
                else:
                    lhs = ''
                if func_type.exception_check == '+':
                    translate_cpp_exception(code, self.pos, '%s%s;' % (lhs, rhs), self.result() if self.type.is_pyobject else None, func_type.exception_value, nogil)
                else:
                    if exc_checks:
                        goto_error = code.error_goto_if(' && '.join(exc_checks), self.pos)
                    else:
                        goto_error = ''
                    code.putln('%s%s; %s' % (lhs, rhs, goto_error))
                if self.type.is_pyobject and self.result():
                    self.generate_gotref(code)
            if self.has_optional_args:
                code.funcstate.release_temp(self.opt_arg_struct)

class NumPyMethodCallNode(ExprNode):
    subexprs = ['arg_tuple']
    is_temp = True
    may_return_none = True

    def generate_evaluation_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        code.mark_pos(self.pos)
        self.allocate_temp_result(code)
        assert self.arg_tuple.mult_factor is None
        args = self.arg_tuple.args
        for arg in args:
            arg.generate_evaluation_code(code)
        code.putln('// function evaluation code for numpy function')
        code.putln('__Pyx_call_destructor(%s);' % self.result())
        code.putln('new (&%s) decltype(%s){%s{}(%s)};' % (self.result(), self.result(), self.function_cname, ', '.join((a.pythran_result() for a in args))))

class PyMethodCallNode(SimpleCallNode):
    subexprs = ['function', 'arg_tuple']
    is_temp = True

    def generate_evaluation_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        code.mark_pos(self.pos)
        self.allocate_temp_result(code)
        self.function.generate_evaluation_code(code)
        assert self.arg_tuple.mult_factor is None
        args = self.arg_tuple.args
        for arg in args:
            arg.generate_evaluation_code(code)
        reuse_function_temp = self.function.is_temp
        if reuse_function_temp:
            function = self.function.result()
        else:
            function = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
            self.function.make_owned_reference(code)
            code.put('%s = %s; ' % (function, self.function.py_result()))
            self.function.generate_disposal_code(code)
            self.function.free_temps(code)
        self_arg = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
        code.putln('%s = NULL;' % self_arg)
        arg_offset_cname = code.funcstate.allocate_temp(PyrexTypes.c_int_type, manage_ref=False)
        code.putln('%s = 0;' % arg_offset_cname)

        def attribute_is_likely_method(attr):
            if False:
                for i in range(10):
                    print('nop')
            obj = attr.obj
            if obj.is_name and obj.entry.is_pyglobal:
                return False
            return True
        if self.function.is_attribute:
            likely_method = 'likely' if attribute_is_likely_method(self.function) else 'unlikely'
        elif self.function.is_name and self.function.cf_state:
            for assignment in self.function.cf_state:
                value = assignment.rhs
                if value and value.is_attribute and value.obj.type and value.obj.type.is_pyobject:
                    if attribute_is_likely_method(value):
                        likely_method = 'likely'
                        break
            else:
                likely_method = 'unlikely'
        else:
            likely_method = 'unlikely'
        code.putln('#if CYTHON_UNPACK_METHODS')
        code.putln('if (%s(PyMethod_Check(%s))) {' % (likely_method, function))
        code.putln('%s = PyMethod_GET_SELF(%s);' % (self_arg, function))
        code.putln('if (likely(%s)) {' % self_arg)
        code.putln('PyObject* function = PyMethod_GET_FUNCTION(%s);' % function)
        code.put_incref(self_arg, py_object_type)
        code.put_incref('function', py_object_type)
        code.put_decref_set(function, py_object_type, 'function')
        code.putln('%s = 1;' % arg_offset_cname)
        code.putln('}')
        code.putln('}')
        code.putln('#endif')
        code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectFastCall', 'ObjectHandling.c'))
        code.putln('{')
        code.putln('PyObject *__pyx_callargs[%d] = {%s, %s};' % (len(args) + 1 if args else 2, self_arg, ', '.join((arg.py_result() for arg in args)) if args else 'NULL'))
        code.putln('%s = __Pyx_PyObject_FastCall(%s, __pyx_callargs+1-%s, %d+%s);' % (self.result(), function, arg_offset_cname, len(args), arg_offset_cname))
        code.put_xdecref_clear(self_arg, py_object_type)
        code.funcstate.release_temp(self_arg)
        code.funcstate.release_temp(arg_offset_cname)
        for arg in args:
            arg.generate_disposal_code(code)
            arg.free_temps(code)
        code.putln(code.error_goto_if_null(self.result(), self.pos))
        self.generate_gotref(code)
        if reuse_function_temp:
            self.function.generate_disposal_code(code)
            self.function.free_temps(code)
        else:
            code.put_decref_clear(function, py_object_type)
            code.funcstate.release_temp(function)
        code.putln('}')

class InlinedDefNodeCallNode(CallNode):
    subexprs = ['args', 'function_name']
    is_temp = 1
    type = py_object_type
    function = None
    function_name = None

    def can_be_inlined(self):
        if False:
            while True:
                i = 10
        func_type = self.function.def_node
        if func_type.star_arg or func_type.starstar_arg:
            return False
        if len(func_type.args) != len(self.args):
            return False
        if func_type.num_kwonly_args:
            return False
        return True

    def analyse_types(self, env):
        if False:
            i = 10
            return i + 15
        self.function_name = self.function_name.analyse_types(env)
        self.args = [arg.analyse_types(env) for arg in self.args]
        func_type = self.function.def_node
        actual_nargs = len(self.args)
        some_args_in_temps = False
        for i in range(actual_nargs):
            formal_type = func_type.args[i].type
            arg = self.args[i].coerce_to(formal_type, env)
            if arg.is_temp:
                if i > 0:
                    some_args_in_temps = True
            elif arg.type.is_pyobject and (not env.nogil):
                if arg.nonlocally_immutable():
                    pass
                else:
                    if i > 0:
                        some_args_in_temps = True
                    arg = arg.coerce_to_temp(env)
            self.args[i] = arg
        if some_args_in_temps:
            for i in range(actual_nargs - 1):
                arg = self.args[i]
                if arg.nonlocally_immutable():
                    pass
                elif arg.type.is_cpp_class:
                    pass
                elif env.nogil and arg.type.is_pyobject:
                    pass
                elif i > 0:
                    warning(arg.pos, 'Argument evaluation order in C function call is undefined and may not be as expected', 0)
                    break
        return self

    def generate_result_code(self, code):
        if False:
            return 10
        arg_code = [self.function_name.py_result()]
        func_type = self.function.def_node
        for (arg, proto_arg) in zip(self.args, func_type.args):
            if arg.type.is_pyobject:
                arg_code.append(arg.result_as(proto_arg.type))
            else:
                arg_code.append(arg.result())
        arg_code = ', '.join(arg_code)
        code.putln('%s = %s(%s); %s' % (self.result(), self.function.def_node.entry.pyfunc_cname, arg_code, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

class PythonCapiFunctionNode(ExprNode):
    subexprs = []

    def __init__(self, pos, py_name, cname, func_type, utility_code=None):
        if False:
            i = 10
            return i + 15
        ExprNode.__init__(self, pos, name=py_name, cname=cname, type=func_type, utility_code=utility_code)

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        return self

    def generate_result_code(self, code):
        if False:
            i = 10
            return i + 15
        if self.utility_code:
            code.globalstate.use_utility_code(self.utility_code)

    def calculate_result_code(self):
        if False:
            for i in range(10):
                print('nop')
        return self.cname

class PythonCapiCallNode(SimpleCallNode):
    may_return_none = False

    def __init__(self, pos, function_name, func_type, utility_code=None, py_name=None, **kwargs):
        if False:
            print('Hello World!')
        self.type = func_type.return_type
        self.result_ctype = self.type
        self.function = PythonCapiFunctionNode(pos, py_name, function_name, func_type, utility_code=utility_code)
        SimpleCallNode.__init__(self, pos, **kwargs)

class CachedBuiltinMethodCallNode(CallNode):
    subexprs = ['obj', 'args']
    is_temp = True

    def __init__(self, call_node, obj, method_name, args):
        if False:
            print('Hello World!')
        super(CachedBuiltinMethodCallNode, self).__init__(call_node.pos, obj=obj, method_name=method_name, args=args, may_return_none=call_node.may_return_none, type=call_node.type)

    def may_be_none(self):
        if False:
            return 10
        if self.may_return_none is not None:
            return self.may_return_none
        return ExprNode.may_be_none(self)

    def generate_result_code(self, code):
        if False:
            print('Hello World!')
        type_cname = self.obj.type.typeptr_cname
        obj_cname = self.obj.py_result()
        args = [arg.py_result() for arg in self.args]
        call_code = code.globalstate.cached_unbound_method_call_code(obj_cname, type_cname, self.method_name, args)
        code.putln('%s = %s; %s' % (self.result(), call_code, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

class GeneralCallNode(CallNode):
    type = py_object_type
    subexprs = ['function', 'positional_args', 'keyword_args']
    nogil_check = Node.gil_error

    def compile_time_value(self, denv):
        if False:
            print('Hello World!')
        function = self.function.compile_time_value(denv)
        positional_args = self.positional_args.compile_time_value(denv)
        keyword_args = self.keyword_args.compile_time_value(denv)
        try:
            return function(*positional_args, **keyword_args)
        except Exception as e:
            self.compile_time_value_error(e)

    def explicit_args_kwds(self):
        if False:
            for i in range(10):
                print('nop')
        if self.keyword_args and (not self.keyword_args.is_dict_literal) or not self.positional_args.is_sequence_constructor:
            raise CompileError(self.pos, 'Compile-time keyword arguments must be explicit.')
        return (self.positional_args.args, self.keyword_args)

    def analyse_types(self, env):
        if False:
            return 10
        if self.analyse_as_type_constructor(env):
            return self
        self.function = self.function.analyse_types(env)
        if not self.function.type.is_pyobject:
            if self.function.type.is_error:
                self.type = error_type
                return self
            if hasattr(self.function, 'entry'):
                node = self.map_to_simple_call_node()
                if node is not None and node is not self:
                    return node.analyse_types(env)
                elif self.function.entry.as_variable:
                    self.function = self.function.coerce_to_pyobject(env)
                elif node is self:
                    error(self.pos, 'Non-trivial keyword arguments and starred arguments not allowed in cdef functions.')
                else:
                    pass
            else:
                self.function = self.function.coerce_to_pyobject(env)
        if self.keyword_args:
            self.keyword_args = self.keyword_args.analyse_types(env)
        self.positional_args = self.positional_args.analyse_types(env)
        self.positional_args = self.positional_args.coerce_to_pyobject(env)
        self.set_py_result_type(self.function)
        self.is_temp = 1
        return self

    def map_to_simple_call_node(self):
        if False:
            i = 10
            return i + 15
        '\n        Tries to map keyword arguments to declared positional arguments.\n        Returns self to try a Python call, None to report an error\n        or a SimpleCallNode if the mapping succeeds.\n        '
        if not isinstance(self.positional_args, TupleNode):
            return self
        if not self.keyword_args.is_dict_literal:
            return self
        function = self.function
        entry = getattr(function, 'entry', None)
        if not entry:
            return self
        function_type = entry.type
        if function_type.is_ptr:
            function_type = function_type.base_type
        if not function_type.is_cfunction:
            return self
        pos_args = self.positional_args.args
        kwargs = self.keyword_args
        declared_args = function_type.args
        if entry.is_cmethod:
            declared_args = declared_args[1:]
        if len(pos_args) > len(declared_args):
            error(self.pos, 'function call got too many positional arguments, expected %d, got %s' % (len(declared_args), len(pos_args)))
            return None
        matched_args = {arg.name for arg in declared_args[:len(pos_args)] if arg.name}
        unmatched_args = declared_args[len(pos_args):]
        matched_kwargs_count = 0
        args = list(pos_args)
        seen = set(matched_args)
        has_errors = False
        for arg in kwargs.key_value_pairs:
            name = arg.key.value
            if name in seen:
                error(arg.pos, "argument '%s' passed twice" % name)
                has_errors = True
            seen.add(name)
        for (decl_arg, arg) in zip(unmatched_args, kwargs.key_value_pairs):
            name = arg.key.value
            if decl_arg.name == name:
                matched_args.add(name)
                matched_kwargs_count += 1
                args.append(arg.value)
            else:
                break
        from .UtilNodes import EvalWithTempExprNode, LetRefNode
        temps = []
        if len(kwargs.key_value_pairs) > matched_kwargs_count:
            unmatched_args = declared_args[len(args):]
            keywords = dict([(arg.key.value, (i + len(pos_args), arg)) for (i, arg) in enumerate(kwargs.key_value_pairs)])
            first_missing_keyword = None
            for decl_arg in unmatched_args:
                name = decl_arg.name
                if name not in keywords:
                    if not first_missing_keyword:
                        first_missing_keyword = name
                    continue
                elif first_missing_keyword:
                    if entry.as_variable:
                        return self
                    error(self.pos, "C function call is missing argument '%s'" % first_missing_keyword)
                    return None
                (pos, arg) = keywords[name]
                matched_args.add(name)
                matched_kwargs_count += 1
                if arg.value.is_simple():
                    args.append(arg.value)
                else:
                    temp = LetRefNode(arg.value)
                    assert temp.is_simple()
                    args.append(temp)
                    temps.append((pos, temp))
            if temps:
                final_args = []
                new_temps = []
                first_temp_arg = temps[0][-1]
                for arg_value in args:
                    if arg_value is first_temp_arg:
                        break
                    if arg_value.is_simple():
                        final_args.append(arg_value)
                    else:
                        temp = LetRefNode(arg_value)
                        new_temps.append(temp)
                        final_args.append(temp)
                if new_temps:
                    args = final_args
                temps = new_temps + [arg for (i, arg) in sorted(temps)]
        for arg in kwargs.key_value_pairs:
            name = arg.key.value
            if name not in matched_args:
                has_errors = True
                error(arg.pos, "C function got unexpected keyword argument '%s'" % name)
        if has_errors:
            return None
        node = SimpleCallNode(self.pos, function=function, args=args)
        for temp in temps[::-1]:
            node = EvalWithTempExprNode(temp, node)
        return node

    def generate_result_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        if self.type.is_error:
            return
        if self.keyword_args:
            kwargs = self.keyword_args.py_result()
        else:
            kwargs = 'NULL'
        code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectCall', 'ObjectHandling.c'))
        code.putln('%s = __Pyx_PyObject_Call(%s, %s, %s); %s' % (self.result(), self.function.py_result(), self.positional_args.py_result(), kwargs, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

class AsTupleNode(ExprNode):
    subexprs = ['arg']
    is_temp = 1

    def calculate_constant_result(self):
        if False:
            return 10
        self.constant_result = tuple(self.arg.constant_result)

    def compile_time_value(self, denv):
        if False:
            print('Hello World!')
        arg = self.arg.compile_time_value(denv)
        try:
            return tuple(arg)
        except Exception as e:
            self.compile_time_value_error(e)

    def analyse_types(self, env):
        if False:
            return 10
        self.arg = self.arg.analyse_types(env).coerce_to_pyobject(env)
        if self.arg.type is tuple_type:
            return self.arg.as_none_safe_node("'NoneType' object is not iterable")
        self.type = tuple_type
        return self

    def may_be_none(self):
        if False:
            while True:
                i = 10
        return False
    nogil_check = Node.gil_error
    gil_message = 'Constructing Python tuple'

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        cfunc = '__Pyx_PySequence_Tuple' if self.arg.type in (py_object_type, tuple_type) else 'PySequence_Tuple'
        code.putln('%s = %s(%s); %s' % (self.result(), cfunc, self.arg.py_result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

class MergedDictNode(ExprNode):
    subexprs = ['keyword_args']
    is_temp = 1
    type = dict_type
    reject_duplicates = True

    def calculate_constant_result(self):
        if False:
            while True:
                i = 10
        result = {}
        reject_duplicates = self.reject_duplicates
        for item in self.keyword_args:
            if item.is_dict_literal:
                items = ((key.constant_result, value.constant_result) for (key, value) in item.key_value_pairs)
            else:
                items = item.constant_result.iteritems()
            for (key, value) in items:
                if reject_duplicates and key in result:
                    raise ValueError('duplicate keyword argument found: %s' % key)
                result[key] = value
        self.constant_result = result

    def compile_time_value(self, denv):
        if False:
            i = 10
            return i + 15
        result = {}
        reject_duplicates = self.reject_duplicates
        for item in self.keyword_args:
            if item.is_dict_literal:
                items = [(key.compile_time_value(denv), value.compile_time_value(denv)) for (key, value) in item.key_value_pairs]
            else:
                items = item.compile_time_value(denv).iteritems()
            try:
                for (key, value) in items:
                    if reject_duplicates and key in result:
                        raise ValueError('duplicate keyword argument found: %s' % key)
                    result[key] = value
            except Exception as e:
                self.compile_time_value_error(e)
        return result

    def type_dependencies(self, env):
        if False:
            for i in range(10):
                print('nop')
        return ()

    def infer_type(self, env):
        if False:
            i = 10
            return i + 15
        return dict_type

    def analyse_types(self, env):
        if False:
            return 10
        self.keyword_args = [arg.analyse_types(env).coerce_to_pyobject(env).as_none_safe_node('argument after ** must be a mapping, not NoneType') for arg in self.keyword_args]
        return self

    def may_be_none(self):
        if False:
            for i in range(10):
                print('nop')
        return False
    gil_message = 'Constructing Python dict'

    def generate_evaluation_code(self, code):
        if False:
            return 10
        code.mark_pos(self.pos)
        self.allocate_temp_result(code)
        args = iter(self.keyword_args)
        item = next(args)
        item.generate_evaluation_code(code)
        if item.type is not dict_type:
            code.putln('if (likely(PyDict_CheckExact(%s))) {' % item.py_result())
        if item.is_dict_literal:
            item.make_owned_reference(code)
            code.putln('%s = %s;' % (self.result(), item.py_result()))
            item.generate_post_assignment_code(code)
        else:
            code.putln('%s = PyDict_Copy(%s); %s' % (self.result(), item.py_result(), code.error_goto_if_null(self.result(), item.pos)))
            self.generate_gotref(code)
            item.generate_disposal_code(code)
        if item.type is not dict_type:
            code.putln('} else {')
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectCallOneArg', 'ObjectHandling.c'))
            code.putln('%s = __Pyx_PyObject_CallOneArg((PyObject*)&PyDict_Type, %s); %s' % (self.result(), item.py_result(), code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
            item.generate_disposal_code(code)
            code.putln('}')
        item.free_temps(code)
        helpers = set()
        for item in args:
            if item.is_dict_literal:
                for arg in item.key_value_pairs:
                    arg.generate_evaluation_code(code)
                    if self.reject_duplicates:
                        code.putln('if (unlikely(PyDict_Contains(%s, %s))) {' % (self.result(), arg.key.py_result()))
                        helpers.add('RaiseDoubleKeywords')
                        code.putln('__Pyx_RaiseDoubleKeywordsError("function", %s); %s' % (arg.key.py_result(), code.error_goto(self.pos)))
                        code.putln('}')
                    code.put_error_if_neg(arg.key.pos, 'PyDict_SetItem(%s, %s, %s)' % (self.result(), arg.key.py_result(), arg.value.py_result()))
                    arg.generate_disposal_code(code)
                    arg.free_temps(code)
            else:
                item.generate_evaluation_code(code)
                if self.reject_duplicates:
                    helpers.add('MergeKeywords')
                    code.put_error_if_neg(item.pos, '__Pyx_MergeKeywords(%s, %s)' % (self.result(), item.py_result()))
                else:
                    helpers.add('RaiseMappingExpected')
                    code.putln('if (unlikely(PyDict_Update(%s, %s) < 0)) {' % (self.result(), item.py_result()))
                    code.putln('if (PyErr_ExceptionMatches(PyExc_AttributeError)) __Pyx_RaiseMappingExpectedError(%s);' % item.py_result())
                    code.putln(code.error_goto(item.pos))
                    code.putln('}')
                item.generate_disposal_code(code)
                item.free_temps(code)
        for helper in sorted(helpers):
            code.globalstate.use_utility_code(UtilityCode.load_cached(helper, 'FunctionArguments.c'))

    def annotate(self, code):
        if False:
            return 10
        for item in self.keyword_args:
            item.annotate(code)

class AttributeNode(ExprNode):
    is_attribute = 1
    subexprs = ['obj']
    entry = None
    is_called = 0
    needs_none_check = True
    is_memslice_transpose = False
    is_special_lookup = False
    is_py_attr = 0

    def as_cython_attribute(self):
        if False:
            return 10
        if isinstance(self.obj, NameNode) and self.obj.is_cython_module and (not self.attribute == u'parallel'):
            return self.attribute
        cy = self.obj.as_cython_attribute()
        if cy:
            return '%s.%s' % (cy, self.attribute)
        return None

    def coerce_to(self, dst_type, env):
        if False:
            i = 10
            return i + 15
        if dst_type is py_object_type:
            entry = self.entry
            if entry and entry.is_cfunction and entry.as_variable:
                self.is_temp = 1
                self.entry = entry.as_variable
                self.analyse_as_python_attribute(env)
                return self
            elif entry and entry.is_cfunction and (self.obj.type is not Builtin.type_type):
                from .UtilNodes import EvalWithTempExprNode, ResultRefNode
                obj_node = ResultRefNode(self.obj, type=self.obj.type)
                obj_node.result_ctype = self.obj.result_ctype
                self.obj = obj_node
                unbound_node = ExprNode.coerce_to(self, dst_type, env)
                utility_code = UtilityCode.load_cached('PyMethodNew2Arg', 'ObjectHandling.c')
                func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('func', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('self', PyrexTypes.py_object_type, None)])
                binding_call = PythonCapiCallNode(self.pos, function_name='__Pyx_PyMethod_New2Arg', func_type=func_type, args=[unbound_node, obj_node], utility_code=utility_code)
                complete_call = EvalWithTempExprNode(obj_node, binding_call)
                return complete_call.analyse_types(env)
        return ExprNode.coerce_to(self, dst_type, env)

    def calculate_constant_result(self):
        if False:
            return 10
        attr = self.attribute
        if attr.startswith('__') and attr.endswith('__'):
            return
        self.constant_result = getattr(self.obj.constant_result, attr)

    def compile_time_value(self, denv):
        if False:
            return 10
        attr = self.attribute
        if attr.startswith('__') and attr.endswith('__'):
            error(self.pos, "Invalid attribute name '%s' in compile-time expression" % attr)
            return None
        obj = self.obj.compile_time_value(denv)
        try:
            return getattr(obj, attr)
        except Exception as e:
            self.compile_time_value_error(e)

    def type_dependencies(self, env):
        if False:
            i = 10
            return i + 15
        return self.obj.type_dependencies(env)

    def infer_type(self, env):
        if False:
            i = 10
            return i + 15
        node = self.analyse_as_cimported_attribute_node(env, target=False)
        if node is not None:
            if node.entry.type and node.entry.type.is_cfunction:
                return PyrexTypes.CPtrType(node.entry.type)
            else:
                return node.entry.type
        node = self.analyse_as_type_attribute(env)
        if node is not None:
            return node.entry.type
        obj_type = self.obj.infer_type(env)
        self.analyse_attribute(env, obj_type=obj_type)
        if obj_type.is_builtin_type and self.type.is_cfunction:
            return py_object_type
        elif self.entry and self.entry.is_cmethod:
            return py_object_type
        return self.type

    def analyse_target_declaration(self, env):
        if False:
            print('Hello World!')
        self.is_target = True

    def analyse_target_types(self, env):
        if False:
            i = 10
            return i + 15
        node = self.analyse_types(env, target=1)
        if node.type.is_const:
            error(self.pos, "Assignment to const attribute '%s'" % self.attribute)
        if not node.is_lvalue():
            error(self.pos, "Assignment to non-lvalue of type '%s'" % self.type)
        return node

    def analyse_types(self, env, target=0):
        if False:
            while True:
                i = 10
        if not self.type:
            self.type = PyrexTypes.error_type
        self.initialized_check = env.directives['initializedcheck']
        node = self.analyse_as_cimported_attribute_node(env, target)
        if node is None and (not target):
            node = self.analyse_as_type_attribute(env)
        if node is None:
            node = self.analyse_as_ordinary_attribute_node(env, target)
            assert node is not None
        if (node.is_attribute or node.is_name) and node.entry:
            node.entry.used = True
        if node.is_attribute:
            node.wrap_obj_in_nonecheck(env)
        return node

    def analyse_as_cimported_attribute_node(self, env, target):
        if False:
            return 10
        module_scope = self.obj.analyse_as_module(env)
        if module_scope:
            entry = module_scope.lookup_here(self.attribute)
            if entry and (not entry.known_standard_library_import) and (entry.is_cglobal or entry.is_cfunction or entry.is_type or entry.is_const):
                return self.as_name_node(env, entry, target)
            if self.is_cimported_module_without_shadow(env):
                error(self.pos, "cimported module has no attribute '%s'" % self.attribute)
                return self
        return None

    def analyse_as_type_attribute(self, env):
        if False:
            i = 10
            return i + 15
        if self.obj.is_string_literal:
            return
        type = self.obj.analyse_as_type(env)
        if type:
            if type.is_extension_type or type.is_builtin_type or type.is_cpp_class:
                entry = type.scope.lookup_here(self.attribute)
                if entry and (entry.is_cmethod or (type.is_cpp_class and entry.type.is_cfunction)):
                    if type.is_builtin_type:
                        if not self.is_called:
                            return None
                        ubcm_entry = entry
                    else:
                        ubcm_entry = self._create_unbound_cmethod_entry(type, entry, env)
                        ubcm_entry.overloaded_alternatives = [self._create_unbound_cmethod_entry(type, overloaded_alternative, env) for overloaded_alternative in entry.overloaded_alternatives]
                    return self.as_name_node(env, ubcm_entry, target=False)
            elif type.is_enum or type.is_cpp_enum:
                if self.attribute in type.values:
                    for entry in type.entry.enum_values:
                        if entry.name == self.attribute:
                            return self.as_name_node(env, entry, target=False)
                    else:
                        error(self.pos, '%s not a known value of %s' % (self.attribute, type))
                else:
                    error(self.pos, '%s not a known value of %s' % (self.attribute, type))
        return None

    def _create_unbound_cmethod_entry(self, type, entry, env):
        if False:
            return 10
        if entry.func_cname and entry.type.op_arg_struct is None:
            cname = entry.func_cname
            if entry.type.is_static_method or (env.parent_scope and env.parent_scope.is_cpp_class_scope):
                ctype = entry.type
            elif type.is_cpp_class:
                error(self.pos, '%s not a static member of %s' % (entry.name, type))
                ctype = PyrexTypes.error_type
            else:
                ctype = copy.copy(entry.type)
                ctype.args = ctype.args[:]
                ctype.args[0] = PyrexTypes.CFuncTypeArg('self', type, 'self', None)
        else:
            cname = '%s->%s' % (type.vtabptr_cname, entry.cname)
            ctype = entry.type
        ubcm_entry = Symtab.Entry(entry.name, cname, ctype)
        ubcm_entry.is_cfunction = 1
        ubcm_entry.func_cname = entry.func_cname
        ubcm_entry.is_unbound_cmethod = 1
        ubcm_entry.scope = entry.scope
        return ubcm_entry

    def analyse_as_type(self, env):
        if False:
            print('Hello World!')
        module_scope = self.obj.analyse_as_module(env)
        if module_scope:
            return module_scope.lookup_type(self.attribute)
        if not self.obj.is_string_literal:
            base_type = self.obj.analyse_as_type(env)
            if base_type and getattr(base_type, 'scope', None) is not None:
                return base_type.scope.lookup_type(self.attribute)
        return None

    def analyse_as_extension_type(self, env):
        if False:
            return 10
        module_scope = self.obj.analyse_as_module(env)
        if module_scope:
            entry = module_scope.lookup_here(self.attribute)
            if entry and entry.is_type:
                if entry.type.is_extension_type or entry.type.is_builtin_type:
                    return entry.type
        return None

    def analyse_as_module(self, env):
        if False:
            for i in range(10):
                print('nop')
        module_scope = self.obj.analyse_as_module(env)
        if module_scope:
            entry = module_scope.lookup_here(self.attribute)
            if entry and entry.as_module:
                return entry.as_module
        return None

    def as_name_node(self, env, entry, target):
        if False:
            while True:
                i = 10
        node = NameNode.from_node(self, name=self.attribute, entry=entry)
        if target:
            node = node.analyse_target_types(env)
        else:
            node = node.analyse_rvalue_entry(env)
        node.entry.used = 1
        return node

    def analyse_as_ordinary_attribute_node(self, env, target):
        if False:
            print('Hello World!')
        self.obj = self.obj.analyse_types(env)
        self.analyse_attribute(env)
        if self.entry and self.entry.is_cmethod and (not self.is_called):
            pass
        if self.is_py_attr:
            if not target:
                self.is_temp = 1
                self.result_ctype = py_object_type
        elif target and self.obj.type.is_builtin_type:
            error(self.pos, 'Assignment to an immutable object field')
        elif self.entry and self.entry.is_cproperty:
            if not target:
                return SimpleCallNode.for_cproperty(self.pos, self.obj, self.entry).analyse_types(env)
            error(self.pos, 'Assignment to a read-only property')
        return self

    def analyse_attribute(self, env, obj_type=None):
        if False:
            while True:
                i = 10
        immutable_obj = obj_type is not None
        self.is_py_attr = 0
        self.member = self.attribute
        if obj_type is None:
            if self.obj.type.is_string or self.obj.type.is_pyunicode_ptr:
                self.obj = self.obj.coerce_to_pyobject(env)
            obj_type = self.obj.type
        elif obj_type.is_string or obj_type.is_pyunicode_ptr:
            obj_type = py_object_type
        if obj_type.is_ptr or obj_type.is_array:
            obj_type = obj_type.base_type
            self.op = '->'
        elif obj_type.is_extension_type or obj_type.is_builtin_type:
            self.op = '->'
        elif obj_type.is_reference and obj_type.is_fake_reference:
            self.op = '->'
        else:
            self.op = '.'
        if obj_type.has_attributes:
            if obj_type.attributes_known():
                entry = obj_type.scope.lookup_here(self.attribute)
                if obj_type.is_memoryviewslice and (not entry):
                    if self.attribute == 'T':
                        self.is_memslice_transpose = True
                        self.is_temp = True
                        self.use_managed_ref = True
                        self.type = self.obj.type.transpose(self.pos)
                        return
                    else:
                        obj_type.declare_attribute(self.attribute, env, self.pos)
                        entry = obj_type.scope.lookup_here(self.attribute)
                if entry and entry.is_member:
                    entry = None
            else:
                error(self.pos, "Cannot select attribute of incomplete type '%s'" % obj_type)
                self.type = PyrexTypes.error_type
                return
            self.entry = entry
            if entry:
                if obj_type.is_extension_type and entry.name == '__weakref__':
                    error(self.pos, 'Illegal use of special attribute __weakref__')
                if entry.is_cproperty:
                    self.type = entry.type
                    return
                elif entry.is_variable and (not entry.fused_cfunction) or entry.is_cmethod:
                    self.type = entry.type
                    self.member = entry.cname
                    return
                else:
                    pass
        self.analyse_as_python_attribute(env, obj_type, immutable_obj)

    def analyse_as_python_attribute(self, env, obj_type=None, immutable_obj=False):
        if False:
            i = 10
            return i + 15
        if obj_type is None:
            obj_type = self.obj.type
        self.attribute = env.mangle_class_private_name(self.attribute)
        self.member = self.attribute
        self.type = py_object_type
        self.is_py_attr = 1
        if not obj_type.is_pyobject and (not obj_type.is_error):
            if obj_type.is_string or obj_type.is_cpp_string or obj_type.is_buffer or obj_type.is_memoryviewslice or obj_type.is_numeric or (obj_type.is_ctuple and obj_type.can_coerce_to_pyobject(env)) or (obj_type.is_struct and obj_type.can_coerce_to_pyobject(env)):
                if not immutable_obj:
                    self.obj = self.obj.coerce_to_pyobject(env)
            elif obj_type.is_cfunction and (self.obj.is_name or self.obj.is_attribute) and self.obj.entry.as_variable and self.obj.entry.as_variable.type.is_pyobject:
                if not immutable_obj:
                    self.obj = self.obj.coerce_to_pyobject(env)
            else:
                error(self.pos, "Object of type '%s' has no attribute '%s'" % (obj_type, self.attribute))

    def wrap_obj_in_nonecheck(self, env):
        if False:
            print('Hello World!')
        if not env.directives['nonecheck']:
            return
        msg = None
        format_args = ()
        if self.obj.type.is_extension_type and self.needs_none_check and (not self.is_py_attr):
            msg = "'NoneType' object has no attribute '%{0}s'".format('.30' if len(self.attribute) <= 30 else '')
            format_args = (self.attribute,)
        elif self.obj.type.is_memoryviewslice:
            if self.is_memslice_transpose:
                msg = 'Cannot transpose None memoryview slice'
            else:
                entry = self.obj.type.scope.lookup_here(self.attribute)
                if entry:
                    msg = "Cannot access '%s' attribute of None memoryview slice"
                    format_args = (entry.name,)
        if msg:
            self.obj = self.obj.as_none_safe_node(msg, 'PyExc_AttributeError', format_args=format_args)

    def nogil_check(self, env):
        if False:
            for i in range(10):
                print('nop')
        if self.is_py_attr:
            self.gil_error()
    gil_message = 'Accessing Python attribute'

    def is_cimported_module_without_shadow(self, env):
        if False:
            i = 10
            return i + 15
        return self.obj.is_cimported_module_without_shadow(env)

    def is_simple(self):
        if False:
            while True:
                i = 10
        if self.obj:
            return self.result_in_temp() or self.obj.is_simple()
        else:
            return NameNode.is_simple(self)

    def is_lvalue(self):
        if False:
            print('Hello World!')
        if self.obj:
            return True
        else:
            return NameNode.is_lvalue(self)

    def is_ephemeral(self):
        if False:
            print('Hello World!')
        if self.obj:
            return self.obj.is_ephemeral()
        else:
            return NameNode.is_ephemeral(self)

    def calculate_result_code(self):
        if False:
            i = 10
            return i + 15
        result = self.calculate_access_code()
        if self.entry and self.entry.is_cpp_optional and (not self.is_target):
            result = '(*%s)' % result
        return result

    def calculate_access_code(self):
        if False:
            while True:
                i = 10
        obj = self.obj
        obj_code = obj.result_as(obj.type)
        if self.entry and self.entry.is_cmethod:
            if obj.type.is_extension_type and (not self.entry.is_builtin_cmethod):
                if self.entry.final_func_cname:
                    return self.entry.final_func_cname
                if self.type.from_fused:
                    self.member = self.entry.cname
                return '((struct %s *)%s%s%s)->%s' % (obj.type.vtabstruct_cname, obj_code, self.op, obj.type.vtabslot_cname, self.member)
            elif self.result_is_used:
                return self.member
            return
        elif obj.type.is_complex:
            return '__Pyx_C%s(%s)' % (self.member.upper(), obj_code)
        else:
            if obj.type.is_builtin_type and self.entry and self.entry.is_variable:
                obj_code = obj.type.cast_code(obj.result(), to_object_struct=True)
            return '%s%s%s' % (obj_code, self.op, self.member)

    def generate_result_code(self, code):
        if False:
            return 10
        if self.is_py_attr:
            if self.is_special_lookup:
                code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectLookupSpecial', 'ObjectHandling.c'))
                lookup_func_name = '__Pyx_PyObject_LookupSpecial'
            else:
                code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectGetAttrStr', 'ObjectHandling.c'))
                lookup_func_name = '__Pyx_PyObject_GetAttrStr'
            code.putln('%s = %s(%s, %s); %s' % (self.result(), lookup_func_name, self.obj.py_result(), code.intern_identifier(self.attribute), code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
        elif self.type.is_memoryviewslice:
            if self.is_memslice_transpose:
                for (access, packing) in self.type.axes:
                    if access == 'ptr':
                        error(self.pos, 'Transposing not supported for slices with indirect dimensions')
                        return
                code.putln('%s = %s;' % (self.result(), self.obj.result()))
                code.put_incref_memoryviewslice(self.result(), self.type, have_gil=True)
                T = '__pyx_memslice_transpose(&%s)' % self.result()
                code.putln(code.error_goto_if_neg(T, self.pos))
            elif self.initialized_check:
                code.putln('if (unlikely(!%s.memview)) {PyErr_SetString(PyExc_AttributeError,"Memoryview is not initialized");%s}' % (self.result(), code.error_goto(self.pos)))
        elif self.entry.is_cpp_optional and self.initialized_check:
            if self.is_target:
                undereferenced_result = self.result()
            else:
                assert not self.is_temp
                undereferenced_result = self.calculate_access_code()
            unbound_check_code = self.type.cpp_optional_check_for_null_code(undereferenced_result)
            code.put_error_if_unbound(self.pos, self.entry, unbound_check_code=unbound_check_code)
        elif self.obj.type and self.obj.type.is_extension_type:
            pass
        elif self.entry and self.entry.is_cmethod:
            code.globalstate.use_entry_utility_code(self.entry)

    def generate_disposal_code(self, code):
        if False:
            i = 10
            return i + 15
        if self.is_temp and self.type.is_memoryviewslice and self.is_memslice_transpose:
            code.put_xdecref_clear(self.result(), self.type, have_gil=True)
        else:
            ExprNode.generate_disposal_code(self, code)

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False, exception_check=None, exception_value=None):
        if False:
            for i in range(10):
                print('nop')
        self.obj.generate_evaluation_code(code)
        if self.is_py_attr:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectSetAttrStr', 'ObjectHandling.c'))
            code.put_error_if_neg(self.pos, '__Pyx_PyObject_SetAttrStr(%s, %s, %s)' % (self.obj.py_result(), code.intern_identifier(self.attribute), rhs.py_result()))
            rhs.generate_disposal_code(code)
            rhs.free_temps(code)
        elif self.obj.type.is_complex:
            code.putln('__Pyx_SET_C%s%s(%s, %s);' % (self.member.upper(), self.obj.type.implementation_suffix, self.obj.result_as(self.obj.type), rhs.result_as(self.ctype())))
            rhs.generate_disposal_code(code)
            rhs.free_temps(code)
        else:
            select_code = self.result()
            if self.type.is_pyobject and self.use_managed_ref:
                rhs.make_owned_reference(code)
                rhs.generate_giveref(code)
                code.put_gotref(select_code, self.type)
                code.put_decref(select_code, self.ctype())
            elif self.type.is_memoryviewslice:
                from . import MemoryView
                MemoryView.put_assign_to_memviewslice(select_code, rhs, rhs.result(), self.type, code)
            if not self.type.is_memoryviewslice:
                code.putln('%s = %s;' % (select_code, rhs.move_result_rhs_as(self.ctype())))
            rhs.generate_post_assignment_code(code)
            rhs.free_temps(code)
        self.obj.generate_disposal_code(code)
        self.obj.free_temps(code)

    def generate_deletion_code(self, code, ignore_nonexisting=False):
        if False:
            while True:
                i = 10
        self.obj.generate_evaluation_code(code)
        if self.is_py_attr or (self.entry.scope.is_property_scope and u'__del__' in self.entry.scope.entries):
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectSetAttrStr', 'ObjectHandling.c'))
            code.put_error_if_neg(self.pos, '__Pyx_PyObject_DelAttrStr(%s, %s)' % (self.obj.py_result(), code.intern_identifier(self.attribute)))
        else:
            error(self.pos, 'Cannot delete C attribute of extension type')
        self.obj.generate_disposal_code(code)
        self.obj.free_temps(code)

    def annotate(self, code):
        if False:
            i = 10
            return i + 15
        if self.is_py_attr:
            (style, text) = ('py_attr', 'python attribute (%s)')
        else:
            (style, text) = ('c_attr', 'c attribute (%s)')
        code.annotate(self.pos, AnnotationItem(style, text % self.type, size=len(self.attribute)))

    def get_known_standard_library_import(self):
        if False:
            i = 10
            return i + 15
        module_name = self.obj.get_known_standard_library_import()
        if module_name:
            return StringEncoding.EncodedString('%s.%s' % (module_name, self.attribute))
        return None

class StarredUnpackingNode(ExprNode):
    subexprs = ['target']
    is_starred = 1
    type = py_object_type
    is_temp = 1
    starred_expr_allowed_here = False

    def __init__(self, pos, target):
        if False:
            print('Hello World!')
        ExprNode.__init__(self, pos, target=target)

    def analyse_declarations(self, env):
        if False:
            return 10
        if not self.starred_expr_allowed_here:
            error(self.pos, 'starred expression is not allowed here')
        self.target.analyse_declarations(env)

    def infer_type(self, env):
        if False:
            for i in range(10):
                print('nop')
        return self.target.infer_type(env)

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        if not self.starred_expr_allowed_here:
            error(self.pos, 'starred expression is not allowed here')
        self.target = self.target.analyse_types(env)
        self.type = self.target.type
        return self

    def analyse_target_declaration(self, env):
        if False:
            while True:
                i = 10
        self.target.analyse_target_declaration(env)

    def analyse_target_types(self, env):
        if False:
            while True:
                i = 10
        self.target = self.target.analyse_target_types(env)
        self.type = self.target.type
        return self

    def calculate_result_code(self):
        if False:
            return 10
        return ''

    def generate_result_code(self, code):
        if False:
            return 10
        pass

class SequenceNode(ExprNode):
    subexprs = ['args', 'mult_factor']
    is_sequence_constructor = 1
    unpacked_items = None
    mult_factor = None
    slow = False

    def compile_time_value_list(self, denv):
        if False:
            return 10
        return [arg.compile_time_value(denv) for arg in self.args]

    def replace_starred_target_node(self):
        if False:
            print('Hello World!')
        self.starred_assignment = False
        args = []
        for arg in self.args:
            if arg.is_starred:
                if self.starred_assignment:
                    error(arg.pos, 'more than 1 starred expression in assignment')
                self.starred_assignment = True
                arg = arg.target
                arg.is_starred = True
            args.append(arg)
        self.args = args

    def analyse_target_declaration(self, env):
        if False:
            return 10
        self.replace_starred_target_node()
        for arg in self.args:
            arg.analyse_target_declaration(env)

    def analyse_types(self, env, skip_children=False):
        if False:
            i = 10
            return i + 15
        for (i, arg) in enumerate(self.args):
            if not skip_children:
                arg = arg.analyse_types(env)
            self.args[i] = arg.coerce_to_pyobject(env)
        if self.mult_factor:
            mult_factor = self.mult_factor.analyse_types(env)
            if not mult_factor.type.is_int:
                mult_factor = mult_factor.coerce_to_pyobject(env)
            self.mult_factor = mult_factor.coerce_to_simple(env)
        self.is_temp = 1
        return self

    def coerce_to_ctuple(self, dst_type, env):
        if False:
            print('Hello World!')
        if self.type == dst_type:
            return self
        assert not self.mult_factor
        if len(self.args) != dst_type.size:
            error(self.pos, 'trying to coerce sequence to ctuple of wrong length, expected %d, got %d' % (dst_type.size, len(self.args)))
        coerced_args = [arg.coerce_to(type, env) for (arg, type) in zip(self.args, dst_type.components)]
        return TupleNode(self.pos, args=coerced_args, type=dst_type, is_temp=True)

    def _create_merge_node_if_necessary(self, env):
        if False:
            return 10
        self._flatten_starred_args()
        if not any((arg.is_starred for arg in self.args)):
            return self
        args = []
        values = []
        for arg in self.args:
            if arg.is_starred:
                if values:
                    args.append(TupleNode(values[0].pos, args=values).analyse_types(env, skip_children=True))
                    values = []
                args.append(arg.target)
            else:
                values.append(arg)
        if values:
            args.append(TupleNode(values[0].pos, args=values).analyse_types(env, skip_children=True))
        node = MergedSequenceNode(self.pos, args, self.type)
        if self.mult_factor:
            node = binop_node(self.pos, '*', node, self.mult_factor.coerce_to_pyobject(env), inplace=True, type=self.type, is_temp=True)
        return node

    def _flatten_starred_args(self):
        if False:
            while True:
                i = 10
        args = []
        for arg in self.args:
            if arg.is_starred and arg.target.is_sequence_constructor and (not arg.target.mult_factor):
                args.extend(arg.target.args)
            else:
                args.append(arg)
        self.args[:] = args

    def may_be_none(self):
        if False:
            i = 10
            return i + 15
        return False

    def analyse_target_types(self, env):
        if False:
            while True:
                i = 10
        if self.mult_factor:
            error(self.pos, "can't assign to multiplied sequence")
        self.unpacked_items = []
        self.coerced_unpacked_items = []
        self.any_coerced_items = False
        for (i, arg) in enumerate(self.args):
            arg = self.args[i] = arg.analyse_target_types(env)
            if arg.is_starred:
                if not arg.type.assignable_from(list_type):
                    error(arg.pos, 'starred target must have Python object (list) type')
                if arg.type is py_object_type:
                    arg.type = list_type
            unpacked_item = PyTempNode(self.pos, env)
            coerced_unpacked_item = unpacked_item.coerce_to(arg.type, env)
            if unpacked_item is not coerced_unpacked_item:
                self.any_coerced_items = True
            self.unpacked_items.append(unpacked_item)
            self.coerced_unpacked_items.append(coerced_unpacked_item)
        self.type = py_object_type
        return self

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        self.generate_operation_code(code)

    def generate_sequence_packing_code(self, code, target=None, plain=False):
        if False:
            i = 10
            return i + 15
        if target is None:
            target = self.result()
        size_factor = c_mult = ''
        mult_factor = None
        if self.mult_factor and (not plain):
            mult_factor = self.mult_factor
            if mult_factor.type.is_int:
                c_mult = mult_factor.result()
                if isinstance(mult_factor.constant_result, _py_int_types) and mult_factor.constant_result > 0:
                    size_factor = ' * %s' % mult_factor.constant_result
                elif mult_factor.type.signed:
                    size_factor = ' * ((%s<0) ? 0:%s)' % (c_mult, c_mult)
                else:
                    size_factor = ' * (%s)' % (c_mult,)
        if self.type is tuple_type and (self.is_literal or self.slow) and (not c_mult):
            code.putln('%s = PyTuple_Pack(%d, %s); %s' % (target, len(self.args), ', '.join((arg.py_result() for arg in self.args)), code.error_goto_if_null(target, self.pos)))
            code.put_gotref(target, py_object_type)
        elif self.type.is_ctuple:
            for (i, arg) in enumerate(self.args):
                code.putln('%s.f%s = %s;' % (target, i, arg.result()))
        else:
            if self.type is list_type:
                (create_func, set_item_func) = ('PyList_New', '__Pyx_PyList_SET_ITEM')
            elif self.type is tuple_type:
                (create_func, set_item_func) = ('PyTuple_New', '__Pyx_PyTuple_SET_ITEM')
            else:
                raise InternalError('sequence packing for unexpected type %s' % self.type)
            arg_count = len(self.args)
            code.putln('%s = %s(%s%s); %s' % (target, create_func, arg_count, size_factor, code.error_goto_if_null(target, self.pos)))
            code.put_gotref(target, py_object_type)
            if c_mult:
                counter = Naming.quick_temp_cname
                code.putln('{ Py_ssize_t %s;' % counter)
                if arg_count == 1:
                    offset = counter
                else:
                    offset = '%s * %s' % (counter, arg_count)
                code.putln('for (%s=0; %s < %s; %s++) {' % (counter, counter, c_mult, counter))
            else:
                offset = ''
            for i in range(arg_count):
                arg = self.args[i]
                if c_mult or not arg.result_in_temp():
                    code.put_incref(arg.result(), arg.ctype())
                arg.generate_giveref(code)
                code.putln('if (%s(%s, %s, %s)) %s;' % (set_item_func, target, (offset and i) and '%s + %s' % (offset, i) or (offset or i), arg.py_result(), code.error_goto(self.pos)))
            if c_mult:
                code.putln('}')
                code.putln('}')
        if mult_factor is not None and mult_factor.type.is_pyobject:
            code.putln('{ PyObject* %s = PyNumber_InPlaceMultiply(%s, %s); %s' % (Naming.quick_temp_cname, target, mult_factor.py_result(), code.error_goto_if_null(Naming.quick_temp_cname, self.pos)))
            code.put_gotref(Naming.quick_temp_cname, py_object_type)
            code.put_decref(target, py_object_type)
            code.putln('%s = %s;' % (target, Naming.quick_temp_cname))
            code.putln('}')

    def generate_subexpr_disposal_code(self, code):
        if False:
            return 10
        if self.mult_factor and self.mult_factor.type.is_int:
            super(SequenceNode, self).generate_subexpr_disposal_code(code)
        elif self.type is tuple_type and (self.is_literal or self.slow):
            super(SequenceNode, self).generate_subexpr_disposal_code(code)
        else:
            for arg in self.args:
                arg.generate_post_assignment_code(code)
            if self.mult_factor:
                self.mult_factor.generate_disposal_code(code)

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False, exception_check=None, exception_value=None):
        if False:
            for i in range(10):
                print('nop')
        if self.starred_assignment:
            self.generate_starred_assignment_code(rhs, code)
        else:
            self.generate_parallel_assignment_code(rhs, code)
        for item in self.unpacked_items:
            item.release(code)
        rhs.free_temps(code)
    _func_iternext_type = PyrexTypes.CPtrType(PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('it', PyrexTypes.py_object_type, None)]))

    def generate_parallel_assignment_code(self, rhs, code):
        if False:
            i = 10
            return i + 15
        for item in self.unpacked_items:
            item.allocate(code)
        special_unpack = rhs.type is py_object_type or rhs.type in (tuple_type, list_type) or (not rhs.type.is_builtin_type)
        long_enough_for_a_loop = len(self.unpacked_items) > 3
        if special_unpack:
            self.generate_special_parallel_unpacking_code(code, rhs, use_loop=long_enough_for_a_loop)
        else:
            code.putln('{')
            self.generate_generic_parallel_unpacking_code(code, rhs, self.unpacked_items, use_loop=long_enough_for_a_loop)
            code.putln('}')
        for value_node in self.coerced_unpacked_items:
            value_node.generate_evaluation_code(code)
        for i in range(len(self.args)):
            self.args[i].generate_assignment_code(self.coerced_unpacked_items[i], code)

    def generate_special_parallel_unpacking_code(self, code, rhs, use_loop):
        if False:
            i = 10
            return i + 15
        sequence_type_test = '1'
        none_check = 'likely(%s != Py_None)' % rhs.py_result()
        if rhs.type is list_type:
            sequence_types = ['List']
            if rhs.may_be_none():
                sequence_type_test = none_check
        elif rhs.type is tuple_type:
            sequence_types = ['Tuple']
            if rhs.may_be_none():
                sequence_type_test = none_check
        else:
            sequence_types = ['Tuple', 'List']
            tuple_check = 'likely(PyTuple_CheckExact(%s))' % rhs.py_result()
            list_check = 'PyList_CheckExact(%s)' % rhs.py_result()
            sequence_type_test = '(%s) || (%s)' % (tuple_check, list_check)
        code.putln('if (%s) {' % sequence_type_test)
        code.putln('PyObject* sequence = %s;' % rhs.py_result())
        code.putln('Py_ssize_t size = __Pyx_PySequence_SIZE(sequence);')
        code.putln('if (unlikely(size != %d)) {' % len(self.args))
        code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseTooManyValuesToUnpack', 'ObjectHandling.c'))
        code.putln('if (size > %d) __Pyx_RaiseTooManyValuesError(%d);' % (len(self.args), len(self.args)))
        code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseNeedMoreValuesToUnpack', 'ObjectHandling.c'))
        code.putln('else if (size >= 0) __Pyx_RaiseNeedMoreValuesError(size);')
        code.putln(code.error_goto(self.pos))
        code.putln('}')
        code.putln('#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS')
        if len(sequence_types) == 2:
            code.putln('if (likely(Py%s_CheckExact(sequence))) {' % sequence_types[0])
        for (i, item) in enumerate(self.unpacked_items):
            code.putln('%s = Py%s_GET_ITEM(sequence, %d); ' % (item.result(), sequence_types[0], i))
        if len(sequence_types) == 2:
            code.putln('} else {')
            for (i, item) in enumerate(self.unpacked_items):
                code.putln('%s = Py%s_GET_ITEM(sequence, %d); ' % (item.result(), sequence_types[1], i))
            code.putln('}')
        for item in self.unpacked_items:
            code.put_incref(item.result(), item.ctype())
        code.putln('#else')
        if not use_loop:
            for (i, item) in enumerate(self.unpacked_items):
                code.putln('%s = PySequence_ITEM(sequence, %d); %s' % (item.result(), i, code.error_goto_if_null(item.result(), self.pos)))
                code.put_gotref(item.result(), item.type)
        else:
            code.putln('{')
            code.putln('Py_ssize_t i;')
            code.putln('PyObject** temps[%s] = {%s};' % (len(self.unpacked_items), ','.join(['&%s' % item.result() for item in self.unpacked_items])))
            code.putln('for (i=0; i < %s; i++) {' % len(self.unpacked_items))
            code.putln('PyObject* item = PySequence_ITEM(sequence, i); %s' % code.error_goto_if_null('item', self.pos))
            code.put_gotref('item', py_object_type)
            code.putln('*(temps[i]) = item;')
            code.putln('}')
            code.putln('}')
        code.putln('#endif')
        rhs.generate_disposal_code(code)
        if sequence_type_test == '1':
            code.putln('}')
        elif sequence_type_test == none_check:
            code.putln('} else {')
            code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseNoneIterError', 'ObjectHandling.c'))
            code.putln('__Pyx_RaiseNoneNotIterableError(); %s' % code.error_goto(self.pos))
            code.putln('}')
        else:
            code.putln('} else {')
            self.generate_generic_parallel_unpacking_code(code, rhs, self.unpacked_items, use_loop=use_loop)
            code.putln('}')

    def generate_generic_parallel_unpacking_code(self, code, rhs, unpacked_items, use_loop, terminate=True):
        if False:
            i = 10
            return i + 15
        code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseNeedMoreValuesToUnpack', 'ObjectHandling.c'))
        code.globalstate.use_utility_code(UtilityCode.load_cached('IterFinish', 'ObjectHandling.c'))
        code.putln('Py_ssize_t index = -1;')
        if use_loop:
            code.putln('PyObject** temps[%s] = {%s};' % (len(self.unpacked_items), ','.join(['&%s' % item.result() for item in unpacked_items])))
        iterator_temp = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
        code.putln('%s = PyObject_GetIter(%s); %s' % (iterator_temp, rhs.py_result(), code.error_goto_if_null(iterator_temp, self.pos)))
        code.put_gotref(iterator_temp, py_object_type)
        rhs.generate_disposal_code(code)
        iternext_func = code.funcstate.allocate_temp(self._func_iternext_type, manage_ref=False)
        code.putln('%s = __Pyx_PyObject_GetIterNextFunc(%s);' % (iternext_func, iterator_temp))
        unpacking_error_label = code.new_label('unpacking_failed')
        unpack_code = '%s(%s)' % (iternext_func, iterator_temp)
        if use_loop:
            code.putln('for (index=0; index < %s; index++) {' % len(unpacked_items))
            code.put('PyObject* item = %s; if (unlikely(!item)) ' % unpack_code)
            code.put_goto(unpacking_error_label)
            code.put_gotref('item', py_object_type)
            code.putln('*(temps[index]) = item;')
            code.putln('}')
        else:
            for (i, item) in enumerate(unpacked_items):
                code.put('index = %d; %s = %s; if (unlikely(!%s)) ' % (i, item.result(), unpack_code, item.result()))
                code.put_goto(unpacking_error_label)
                item.generate_gotref(code)
        if terminate:
            code.globalstate.use_utility_code(UtilityCode.load_cached('UnpackItemEndCheck', 'ObjectHandling.c'))
            code.put_error_if_neg(self.pos, '__Pyx_IternextUnpackEndCheck(%s, %d)' % (unpack_code, len(unpacked_items)))
            code.putln('%s = NULL;' % iternext_func)
            code.put_decref_clear(iterator_temp, py_object_type)
        unpacking_done_label = code.new_label('unpacking_done')
        code.put_goto(unpacking_done_label)
        code.put_label(unpacking_error_label)
        code.put_decref_clear(iterator_temp, py_object_type)
        code.putln('%s = NULL;' % iternext_func)
        code.putln('if (__Pyx_IterFinish() == 0) __Pyx_RaiseNeedMoreValuesError(index);')
        code.putln(code.error_goto(self.pos))
        code.put_label(unpacking_done_label)
        code.funcstate.release_temp(iternext_func)
        if terminate:
            code.funcstate.release_temp(iterator_temp)
            iterator_temp = None
        return iterator_temp

    def generate_starred_assignment_code(self, rhs, code):
        if False:
            i = 10
            return i + 15
        for (i, arg) in enumerate(self.args):
            if arg.is_starred:
                starred_target = self.unpacked_items[i]
                unpacked_fixed_items_left = self.unpacked_items[:i]
                unpacked_fixed_items_right = self.unpacked_items[i + 1:]
                break
        else:
            assert False
        iterator_temp = None
        if unpacked_fixed_items_left:
            for item in unpacked_fixed_items_left:
                item.allocate(code)
            code.putln('{')
            iterator_temp = self.generate_generic_parallel_unpacking_code(code, rhs, unpacked_fixed_items_left, use_loop=True, terminate=False)
            for (i, item) in enumerate(unpacked_fixed_items_left):
                value_node = self.coerced_unpacked_items[i]
                value_node.generate_evaluation_code(code)
            code.putln('}')
        starred_target.allocate(code)
        target_list = starred_target.result()
        code.putln('%s = %s(%s); %s' % (target_list, '__Pyx_PySequence_ListKeepNew' if not iterator_temp and rhs.is_temp and (rhs.type in (py_object_type, list_type)) else 'PySequence_List', iterator_temp or rhs.py_result(), code.error_goto_if_null(target_list, self.pos)))
        starred_target.generate_gotref(code)
        if iterator_temp:
            code.put_decref_clear(iterator_temp, py_object_type)
            code.funcstate.release_temp(iterator_temp)
        else:
            rhs.generate_disposal_code(code)
        if unpacked_fixed_items_right:
            code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseNeedMoreValuesToUnpack', 'ObjectHandling.c'))
            length_temp = code.funcstate.allocate_temp(PyrexTypes.c_py_ssize_t_type, manage_ref=False)
            code.putln('%s = PyList_GET_SIZE(%s);' % (length_temp, target_list))
            code.putln('if (unlikely(%s < %d)) {' % (length_temp, len(unpacked_fixed_items_right)))
            code.putln('__Pyx_RaiseNeedMoreValuesError(%d+%s); %s' % (len(unpacked_fixed_items_left), length_temp, code.error_goto(self.pos)))
            code.putln('}')
            for item in unpacked_fixed_items_right[::-1]:
                item.allocate(code)
            for (i, (item, coerced_arg)) in enumerate(zip(unpacked_fixed_items_right[::-1], self.coerced_unpacked_items[::-1])):
                code.putln('#if CYTHON_COMPILING_IN_CPYTHON')
                code.putln('%s = PyList_GET_ITEM(%s, %s-%d); ' % (item.py_result(), target_list, length_temp, i + 1))
                code.putln('((PyVarObject*)%s)->ob_size--;' % target_list)
                code.putln('#else')
                code.putln('%s = PySequence_ITEM(%s, %s-%d); ' % (item.py_result(), target_list, length_temp, i + 1))
                code.putln('#endif')
                item.generate_gotref(code)
                coerced_arg.generate_evaluation_code(code)
            code.putln('#if !CYTHON_COMPILING_IN_CPYTHON')
            sublist_temp = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
            code.putln('%s = PySequence_GetSlice(%s, 0, %s-%d); %s' % (sublist_temp, target_list, length_temp, len(unpacked_fixed_items_right), code.error_goto_if_null(sublist_temp, self.pos)))
            code.put_gotref(sublist_temp, py_object_type)
            code.funcstate.release_temp(length_temp)
            code.put_decref(target_list, py_object_type)
            code.putln('%s = %s; %s = NULL;' % (target_list, sublist_temp, sublist_temp))
            code.putln('#else')
            code.putln('CYTHON_UNUSED_VAR(%s);' % sublist_temp)
            code.funcstate.release_temp(sublist_temp)
            code.putln('#endif')
        for (i, arg) in enumerate(self.args):
            arg.generate_assignment_code(self.coerced_unpacked_items[i], code)

    def annotate(self, code):
        if False:
            while True:
                i = 10
        for arg in self.args:
            arg.annotate(code)
        if self.unpacked_items:
            for arg in self.unpacked_items:
                arg.annotate(code)
            for arg in self.coerced_unpacked_items:
                arg.annotate(code)

class TupleNode(SequenceNode):
    type = tuple_type
    is_partly_literal = False
    gil_message = 'Constructing Python tuple'

    def infer_type(self, env):
        if False:
            for i in range(10):
                print('nop')
        if self.mult_factor or not self.args:
            return tuple_type
        arg_types = [arg.infer_type(env) for arg in self.args]
        if any((type.is_pyobject or type.is_memoryviewslice or type.is_unspecified or type.is_fused for type in arg_types)):
            return tuple_type
        return env.declare_tuple_type(self.pos, arg_types).type

    def analyse_types(self, env, skip_children=False):
        if False:
            print('Hello World!')
        if self.is_literal:
            self.is_literal = False
        if self.is_partly_literal:
            self.is_partly_literal = False
        if len(self.args) == 0:
            self.is_temp = False
            self.is_literal = True
            return self
        if not skip_children:
            for (i, arg) in enumerate(self.args):
                if arg.is_starred:
                    arg.starred_expr_allowed_here = True
                self.args[i] = arg.analyse_types(env)
        if not self.mult_factor and (not any((arg.is_starred or arg.type.is_pyobject or arg.type.is_memoryviewslice or arg.type.is_fused for arg in self.args))):
            self.type = env.declare_tuple_type(self.pos, (arg.type for arg in self.args)).type
            self.is_temp = 1
            return self
        node = SequenceNode.analyse_types(self, env, skip_children=True)
        node = node._create_merge_node_if_necessary(env)
        if not node.is_sequence_constructor:
            return node
        if not all((child.is_literal for child in node.args)):
            return node
        if not node.mult_factor or (node.mult_factor.is_literal and isinstance(node.mult_factor.constant_result, _py_int_types)):
            node.is_temp = False
            node.is_literal = True
        else:
            if not node.mult_factor.type.is_pyobject and (not node.mult_factor.type.is_int):
                node.mult_factor = node.mult_factor.coerce_to_pyobject(env)
            node.is_temp = True
            node.is_partly_literal = True
        return node

    def analyse_as_type(self, env):
        if False:
            for i in range(10):
                print('nop')
        if not self.args:
            return None
        item_types = [arg.analyse_as_type(env) for arg in self.args]
        if any((t is None for t in item_types)):
            return None
        entry = env.declare_tuple_type(self.pos, item_types)
        return entry.type

    def coerce_to(self, dst_type, env):
        if False:
            i = 10
            return i + 15
        if self.type.is_ctuple:
            if dst_type.is_ctuple and self.type.size == dst_type.size:
                return self.coerce_to_ctuple(dst_type, env)
            elif dst_type is tuple_type or dst_type is py_object_type:
                coerced_args = [arg.coerce_to_pyobject(env) for arg in self.args]
                return TupleNode(self.pos, args=coerced_args, type=tuple_type, mult_factor=self.mult_factor, is_temp=1).analyse_types(env, skip_children=True)
            else:
                return self.coerce_to_pyobject(env).coerce_to(dst_type, env)
        elif dst_type.is_ctuple and (not self.mult_factor):
            return self.coerce_to_ctuple(dst_type, env)
        else:
            return SequenceNode.coerce_to(self, dst_type, env)

    def as_list(self):
        if False:
            i = 10
            return i + 15
        t = ListNode(self.pos, args=self.args, mult_factor=self.mult_factor)
        if isinstance(self.constant_result, tuple):
            t.constant_result = list(self.constant_result)
        return t

    def is_simple(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def nonlocally_immutable(self):
        if False:
            i = 10
            return i + 15
        return True

    def calculate_result_code(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self.args) > 0:
            return self.result_code
        else:
            return Naming.empty_tuple

    def calculate_constant_result(self):
        if False:
            print('Hello World!')
        self.constant_result = tuple([arg.constant_result for arg in self.args])

    def compile_time_value(self, denv):
        if False:
            return 10
        values = self.compile_time_value_list(denv)
        try:
            return tuple(values)
        except Exception as e:
            self.compile_time_value_error(e)

    def generate_operation_code(self, code):
        if False:
            return 10
        if len(self.args) == 0:
            return
        if self.is_literal or self.is_partly_literal:
            dedup_key = make_dedup_key(self.type, [self.mult_factor if self.is_literal else None] + self.args)
            tuple_target = code.get_py_const(py_object_type, 'tuple', cleanup_level=2, dedup_key=dedup_key)
            const_code = code.get_cached_constants_writer(tuple_target)
            if const_code is not None:
                const_code.mark_pos(self.pos)
                self.generate_sequence_packing_code(const_code, tuple_target, plain=not self.is_literal)
                const_code.put_giveref(tuple_target, py_object_type)
            if self.is_literal:
                self.result_code = tuple_target
            elif self.mult_factor.type.is_int:
                code.globalstate.use_utility_code(UtilityCode.load_cached('PySequenceMultiply', 'ObjectHandling.c'))
                code.putln('%s = __Pyx_PySequence_Multiply(%s, %s); %s' % (self.result(), tuple_target, self.mult_factor.result(), code.error_goto_if_null(self.result(), self.pos)))
                self.generate_gotref(code)
            else:
                code.putln('%s = PyNumber_Multiply(%s, %s); %s' % (self.result(), tuple_target, self.mult_factor.py_result(), code.error_goto_if_null(self.result(), self.pos)))
                self.generate_gotref(code)
        else:
            self.type.entry.used = True
            self.generate_sequence_packing_code(code)

class ListNode(SequenceNode):
    obj_conversion_errors = []
    type = list_type
    in_module_scope = False
    gil_message = 'Constructing Python list'

    def type_dependencies(self, env):
        if False:
            return 10
        return ()

    def infer_type(self, env):
        if False:
            i = 10
            return i + 15
        return list_type

    def analyse_expressions(self, env):
        if False:
            print('Hello World!')
        for arg in self.args:
            if arg.is_starred:
                arg.starred_expr_allowed_here = True
        node = SequenceNode.analyse_expressions(self, env)
        return node.coerce_to_pyobject(env)

    def analyse_types(self, env):
        if False:
            print('Hello World!')
        with local_errors(ignore=True) as errors:
            self.original_args = list(self.args)
            node = SequenceNode.analyse_types(self, env)
        node.obj_conversion_errors = errors
        if env.is_module_scope:
            self.in_module_scope = True
        node = node._create_merge_node_if_necessary(env)
        return node

    def coerce_to(self, dst_type, env):
        if False:
            print('Hello World!')
        if dst_type.is_pyobject:
            for err in self.obj_conversion_errors:
                report_error(err)
            self.obj_conversion_errors = []
            if not self.type.subtype_of(dst_type):
                error(self.pos, "Cannot coerce list to type '%s'" % dst_type)
        elif (dst_type.is_array or dst_type.is_ptr) and dst_type.base_type is not PyrexTypes.c_void_type:
            array_length = len(self.args)
            if self.mult_factor:
                if isinstance(self.mult_factor.constant_result, _py_int_types):
                    if self.mult_factor.constant_result <= 0:
                        error(self.pos, "Cannot coerce non-positively multiplied list to '%s'" % dst_type)
                    else:
                        array_length *= self.mult_factor.constant_result
                else:
                    error(self.pos, "Cannot coerce dynamically multiplied list to '%s'" % dst_type)
            base_type = dst_type.base_type
            self.type = PyrexTypes.CArrayType(base_type, array_length)
            for i in range(len(self.original_args)):
                arg = self.args[i]
                if isinstance(arg, CoerceToPyTypeNode):
                    arg = arg.arg
                self.args[i] = arg.coerce_to(base_type, env)
        elif dst_type.is_cpp_class:
            return TypecastNode(self.pos, operand=self, type=PyrexTypes.py_object_type).coerce_to(dst_type, env)
        elif self.mult_factor:
            error(self.pos, "Cannot coerce multiplied list to '%s'" % dst_type)
        elif dst_type.is_struct:
            if len(self.args) > len(dst_type.scope.var_entries):
                error(self.pos, "Too many members for '%s'" % dst_type)
            else:
                if len(self.args) < len(dst_type.scope.var_entries):
                    warning(self.pos, "Too few members for '%s'" % dst_type, 1)
                for (i, (arg, member)) in enumerate(zip(self.original_args, dst_type.scope.var_entries)):
                    if isinstance(arg, CoerceToPyTypeNode):
                        arg = arg.arg
                    self.args[i] = arg.coerce_to(member.type, env)
            self.type = dst_type
        elif dst_type.is_ctuple:
            return self.coerce_to_ctuple(dst_type, env)
        else:
            self.type = error_type
            error(self.pos, "Cannot coerce list to type '%s'" % dst_type)
        return self

    def as_list(self):
        if False:
            return 10
        return self

    def as_tuple(self):
        if False:
            while True:
                i = 10
        t = TupleNode(self.pos, args=self.args, mult_factor=self.mult_factor)
        if isinstance(self.constant_result, list):
            t.constant_result = tuple(self.constant_result)
        return t

    def allocate_temp_result(self, code):
        if False:
            i = 10
            return i + 15
        if self.type.is_array:
            if self.in_module_scope:
                self.temp_code = code.funcstate.allocate_temp(self.type, manage_ref=False, static=True, reusable=False)
            else:
                self.temp_code = code.funcstate.allocate_temp(self.type, manage_ref=False, reusable=False)
        else:
            SequenceNode.allocate_temp_result(self, code)

    def calculate_constant_result(self):
        if False:
            for i in range(10):
                print('nop')
        if self.mult_factor:
            raise ValueError()
        self.constant_result = [arg.constant_result for arg in self.args]

    def compile_time_value(self, denv):
        if False:
            print('Hello World!')
        l = self.compile_time_value_list(denv)
        if self.mult_factor:
            l *= self.mult_factor.compile_time_value(denv)
        return l

    def generate_operation_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        if self.type.is_pyobject:
            for err in self.obj_conversion_errors:
                report_error(err)
            self.generate_sequence_packing_code(code)
        elif self.type.is_array:
            if self.mult_factor:
                code.putln('{')
                code.putln('Py_ssize_t %s;' % Naming.quick_temp_cname)
                code.putln('for ({i} = 0; {i} < {count}; {i}++) {{'.format(i=Naming.quick_temp_cname, count=self.mult_factor.result()))
                offset = '+ (%d * %s)' % (len(self.args), Naming.quick_temp_cname)
            else:
                offset = ''
            for (i, arg) in enumerate(self.args):
                if arg.type.is_array:
                    code.globalstate.use_utility_code(UtilityCode.load_cached('IncludeStringH', 'StringTools.c'))
                    code.putln('memcpy(&(%s[%s%s]), %s, sizeof(%s[0]));' % (self.result(), i, offset, arg.result(), self.result()))
                else:
                    code.putln('%s[%s%s] = %s;' % (self.result(), i, offset, arg.result()))
            if self.mult_factor:
                code.putln('}')
                code.putln('}')
        elif self.type.is_struct:
            for (arg, member) in zip(self.args, self.type.scope.var_entries):
                code.putln('%s.%s = %s;' % (self.result(), member.cname, arg.result()))
        else:
            raise InternalError('List type never specified')

class ComprehensionNode(ScopedExprNode):
    child_attrs = ['loop']
    is_temp = True
    constant_result = not_a_constant

    def infer_type(self, env):
        if False:
            while True:
                i = 10
        return self.type

    def analyse_declarations(self, env):
        if False:
            while True:
                i = 10
        self.append.target = self
        self.init_scope(env)
        if isinstance(self.loop, Nodes._ForInStatNode):
            assert isinstance(self.loop.iterator, ScopedExprNode), self.loop.iterator
            self.loop.iterator.init_scope(None, env)
        else:
            assert isinstance(self.loop, Nodes.ForFromStatNode), self.loop

    def analyse_scoped_declarations(self, env):
        if False:
            return 10
        self.loop.analyse_declarations(env)

    def analyse_types(self, env):
        if False:
            print('Hello World!')
        if not self.has_local_scope:
            self.loop = self.loop.analyse_expressions(env)
        return self

    def analyse_scoped_expressions(self, env):
        if False:
            print('Hello World!')
        if self.has_local_scope:
            self.loop = self.loop.analyse_expressions(env)
        return self

    def may_be_none(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def generate_result_code(self, code):
        if False:
            return 10
        self.generate_operation_code(code)

    def generate_operation_code(self, code):
        if False:
            i = 10
            return i + 15
        if self.type is Builtin.list_type:
            create_code = 'PyList_New(0)'
        elif self.type is Builtin.set_type:
            create_code = 'PySet_New(NULL)'
        elif self.type is Builtin.dict_type:
            create_code = 'PyDict_New()'
        else:
            raise InternalError('illegal type for comprehension: %s' % self.type)
        code.putln('%s = %s; %s' % (self.result(), create_code, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)
        self.loop.generate_execution_code(code)

    def annotate(self, code):
        if False:
            return 10
        self.loop.annotate(code)

class ComprehensionAppendNode(Node):
    child_attrs = ['expr']
    target = None
    type = PyrexTypes.c_int_type

    def analyse_expressions(self, env):
        if False:
            for i in range(10):
                print('nop')
        self.expr = self.expr.analyse_expressions(env)
        if not self.expr.type.is_pyobject:
            self.expr = self.expr.coerce_to_pyobject(env)
        return self

    def generate_execution_code(self, code):
        if False:
            while True:
                i = 10
        if self.target.type is list_type:
            code.globalstate.use_utility_code(UtilityCode.load_cached('ListCompAppend', 'Optimize.c'))
            function = '__Pyx_ListComp_Append'
        elif self.target.type is set_type:
            function = 'PySet_Add'
        else:
            raise InternalError('Invalid type for comprehension node: %s' % self.target.type)
        self.expr.generate_evaluation_code(code)
        code.putln(code.error_goto_if('%s(%s, (PyObject*)%s)' % (function, self.target.result(), self.expr.result()), self.pos))
        self.expr.generate_disposal_code(code)
        self.expr.free_temps(code)

    def generate_function_definitions(self, env, code):
        if False:
            return 10
        self.expr.generate_function_definitions(env, code)

    def annotate(self, code):
        if False:
            while True:
                i = 10
        self.expr.annotate(code)

class DictComprehensionAppendNode(ComprehensionAppendNode):
    child_attrs = ['key_expr', 'value_expr']

    def analyse_expressions(self, env):
        if False:
            while True:
                i = 10
        self.key_expr = self.key_expr.analyse_expressions(env)
        if not self.key_expr.type.is_pyobject:
            self.key_expr = self.key_expr.coerce_to_pyobject(env)
        self.value_expr = self.value_expr.analyse_expressions(env)
        if not self.value_expr.type.is_pyobject:
            self.value_expr = self.value_expr.coerce_to_pyobject(env)
        return self

    def generate_execution_code(self, code):
        if False:
            print('Hello World!')
        self.key_expr.generate_evaluation_code(code)
        self.value_expr.generate_evaluation_code(code)
        code.putln(code.error_goto_if('PyDict_SetItem(%s, (PyObject*)%s, (PyObject*)%s)' % (self.target.result(), self.key_expr.result(), self.value_expr.result()), self.pos))
        self.key_expr.generate_disposal_code(code)
        self.key_expr.free_temps(code)
        self.value_expr.generate_disposal_code(code)
        self.value_expr.free_temps(code)

    def generate_function_definitions(self, env, code):
        if False:
            for i in range(10):
                print('nop')
        self.key_expr.generate_function_definitions(env, code)
        self.value_expr.generate_function_definitions(env, code)

    def annotate(self, code):
        if False:
            i = 10
            return i + 15
        self.key_expr.annotate(code)
        self.value_expr.annotate(code)

class InlinedGeneratorExpressionNode(ExprNode):
    subexprs = ['gen']
    orig_func = None
    target = None
    is_temp = True
    type = py_object_type

    def __init__(self, pos, gen, comprehension_type=None, **kwargs):
        if False:
            while True:
                i = 10
        gbody = gen.def_node.gbody
        gbody.is_inlined = True
        if comprehension_type is not None:
            assert comprehension_type in (list_type, set_type, dict_type), comprehension_type
            gbody.inlined_comprehension_type = comprehension_type
            kwargs.update(target=RawCNameExprNode(pos, comprehension_type, Naming.retval_cname), type=comprehension_type)
        super(InlinedGeneratorExpressionNode, self).__init__(pos, gen=gen, **kwargs)

    def may_be_none(self):
        if False:
            return 10
        return self.orig_func not in ('any', 'all', 'sorted')

    def infer_type(self, env):
        if False:
            i = 10
            return i + 15
        return self.type

    def analyse_types(self, env):
        if False:
            print('Hello World!')
        self.gen = self.gen.analyse_expressions(env)
        return self

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        code.putln('%s = __Pyx_Generator_Next(%s); %s' % (self.result(), self.gen.result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

class MergedSequenceNode(ExprNode):
    """
    Merge a sequence of iterables into a set/list/tuple.

    The target collection is determined by self.type, which must be set externally.

    args    [ExprNode]
    """
    subexprs = ['args']
    is_temp = True
    gil_message = 'Constructing Python collection'

    def __init__(self, pos, args, type):
        if False:
            for i in range(10):
                print('nop')
        if type in (list_type, tuple_type) and args and args[0].is_sequence_constructor:
            if args[0].type is not list_type:
                args[0] = ListNode(args[0].pos, args=args[0].args, is_temp=True, mult_factor=args[0].mult_factor)
        ExprNode.__init__(self, pos, args=args, type=type)

    def calculate_constant_result(self):
        if False:
            return 10
        result = []
        for item in self.args:
            if item.is_sequence_constructor and item.mult_factor:
                if item.mult_factor.constant_result <= 0:
                    continue
            if item.is_set_literal or item.is_sequence_constructor:
                items = (arg.constant_result for arg in item.args)
            else:
                items = item.constant_result
            result.extend(items)
        if self.type is set_type:
            result = set(result)
        elif self.type is tuple_type:
            result = tuple(result)
        else:
            assert self.type is list_type
        self.constant_result = result

    def compile_time_value(self, denv):
        if False:
            i = 10
            return i + 15
        result = []
        for item in self.args:
            if item.is_sequence_constructor and item.mult_factor:
                if item.mult_factor.compile_time_value(denv) <= 0:
                    continue
            if item.is_set_literal or item.is_sequence_constructor:
                items = (arg.compile_time_value(denv) for arg in item.args)
            else:
                items = item.compile_time_value(denv)
            result.extend(items)
        if self.type is set_type:
            try:
                result = set(result)
            except Exception as e:
                self.compile_time_value_error(e)
        elif self.type is tuple_type:
            result = tuple(result)
        else:
            assert self.type is list_type
        return result

    def type_dependencies(self, env):
        if False:
            print('Hello World!')
        return ()

    def infer_type(self, env):
        if False:
            print('Hello World!')
        return self.type

    def analyse_types(self, env):
        if False:
            i = 10
            return i + 15
        args = [arg.analyse_types(env).coerce_to_pyobject(env).as_none_safe_node('argument after * must be an iterable, not NoneType') for arg in self.args]
        if len(args) == 1 and args[0].type is self.type:
            return args[0]
        assert self.type in (set_type, list_type, tuple_type)
        self.args = args
        return self

    def may_be_none(self):
        if False:
            while True:
                i = 10
        return False

    def generate_evaluation_code(self, code):
        if False:
            return 10
        code.mark_pos(self.pos)
        self.allocate_temp_result(code)
        is_set = self.type is set_type
        args = iter(self.args)
        item = next(args)
        item.generate_evaluation_code(code)
        if is_set and item.is_set_literal or (not is_set and item.is_sequence_constructor and (item.type is list_type)):
            code.putln('%s = %s;' % (self.result(), item.py_result()))
            item.generate_post_assignment_code(code)
        else:
            code.putln('%s = %s(%s); %s' % (self.result(), 'PySet_New' if is_set else '__Pyx_PySequence_ListKeepNew' if item.is_temp and item.type in (py_object_type, list_type) else 'PySequence_List', item.py_result(), code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
            item.generate_disposal_code(code)
        item.free_temps(code)
        helpers = set()
        if is_set:
            add_func = 'PySet_Add'
            extend_func = '__Pyx_PySet_Update'
        else:
            add_func = '__Pyx_ListComp_Append'
            extend_func = '__Pyx_PyList_Extend'
        for item in args:
            if is_set and (item.is_set_literal or item.is_sequence_constructor) or (item.is_sequence_constructor and (not item.mult_factor)):
                if not is_set and item.args:
                    helpers.add(('ListCompAppend', 'Optimize.c'))
                for arg in item.args:
                    arg.generate_evaluation_code(code)
                    code.put_error_if_neg(arg.pos, '%s(%s, %s)' % (add_func, self.result(), arg.py_result()))
                    arg.generate_disposal_code(code)
                    arg.free_temps(code)
                continue
            if is_set:
                helpers.add(('PySet_Update', 'Builtins.c'))
            else:
                helpers.add(('ListExtend', 'Optimize.c'))
            item.generate_evaluation_code(code)
            code.put_error_if_neg(item.pos, '%s(%s, %s)' % (extend_func, self.result(), item.py_result()))
            item.generate_disposal_code(code)
            item.free_temps(code)
        if self.type is tuple_type:
            code.putln('{')
            code.putln('PyObject *%s = PyList_AsTuple(%s);' % (Naming.quick_temp_cname, self.result()))
            code.put_decref(self.result(), py_object_type)
            code.putln('%s = %s; %s' % (self.result(), Naming.quick_temp_cname, code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
            code.putln('}')
        for helper in sorted(helpers):
            code.globalstate.use_utility_code(UtilityCode.load_cached(*helper))

    def annotate(self, code):
        if False:
            while True:
                i = 10
        for item in self.args:
            item.annotate(code)

class SetNode(ExprNode):
    """
    Set constructor.
    """
    subexprs = ['args']
    type = set_type
    is_set_literal = True
    gil_message = 'Constructing Python set'

    def analyse_types(self, env):
        if False:
            i = 10
            return i + 15
        for i in range(len(self.args)):
            arg = self.args[i]
            arg = arg.analyse_types(env)
            self.args[i] = arg.coerce_to_pyobject(env)
        self.type = set_type
        self.is_temp = 1
        return self

    def may_be_none(self):
        if False:
            i = 10
            return i + 15
        return False

    def calculate_constant_result(self):
        if False:
            for i in range(10):
                print('nop')
        self.constant_result = {arg.constant_result for arg in self.args}

    def compile_time_value(self, denv):
        if False:
            i = 10
            return i + 15
        values = [arg.compile_time_value(denv) for arg in self.args]
        try:
            return set(values)
        except Exception as e:
            self.compile_time_value_error(e)

    def generate_evaluation_code(self, code):
        if False:
            return 10
        for arg in self.args:
            arg.generate_evaluation_code(code)
        self.allocate_temp_result(code)
        code.putln('%s = PySet_New(0); %s' % (self.result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)
        for arg in self.args:
            code.put_error_if_neg(self.pos, 'PySet_Add(%s, %s)' % (self.result(), arg.py_result()))
            arg.generate_disposal_code(code)
            arg.free_temps(code)

class DictNode(ExprNode):
    subexprs = ['key_value_pairs']
    is_temp = 1
    exclude_null_values = False
    type = dict_type
    is_dict_literal = True
    reject_duplicates = False
    obj_conversion_errors = []

    @classmethod
    def from_pairs(cls, pos, pairs):
        if False:
            print('Hello World!')
        return cls(pos, key_value_pairs=[DictItemNode(pos, key=k, value=v) for (k, v) in pairs])

    def calculate_constant_result(self):
        if False:
            for i in range(10):
                print('nop')
        self.constant_result = dict([item.constant_result for item in self.key_value_pairs])

    def compile_time_value(self, denv):
        if False:
            return 10
        pairs = [(item.key.compile_time_value(denv), item.value.compile_time_value(denv)) for item in self.key_value_pairs]
        try:
            return dict(pairs)
        except Exception as e:
            self.compile_time_value_error(e)

    def type_dependencies(self, env):
        if False:
            while True:
                i = 10
        return ()

    def infer_type(self, env):
        if False:
            for i in range(10):
                print('nop')
        return dict_type

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        with local_errors(ignore=True) as errors:
            self.key_value_pairs = [item.analyse_types(env) for item in self.key_value_pairs]
        self.obj_conversion_errors = errors
        return self

    def may_be_none(self):
        if False:
            return 10
        return False

    def coerce_to(self, dst_type, env):
        if False:
            print('Hello World!')
        if dst_type.is_pyobject:
            self.release_errors()
            if self.type.is_struct_or_union:
                if not dict_type.subtype_of(dst_type):
                    error(self.pos, "Cannot interpret struct as non-dict type '%s'" % dst_type)
                return DictNode(self.pos, key_value_pairs=[DictItemNode(item.pos, key=item.key.coerce_to_pyobject(env), value=item.value.coerce_to_pyobject(env)) for item in self.key_value_pairs])
            if not self.type.subtype_of(dst_type):
                error(self.pos, "Cannot interpret dict as type '%s'" % dst_type)
        elif dst_type.is_struct_or_union:
            self.type = dst_type
            if not dst_type.is_struct and len(self.key_value_pairs) != 1:
                error(self.pos, "Exactly one field must be specified to convert to union '%s'" % dst_type)
            elif dst_type.is_struct and len(self.key_value_pairs) < len(dst_type.scope.var_entries):
                warning(self.pos, "Not all members given for struct '%s'" % dst_type, 1)
            for item in self.key_value_pairs:
                if isinstance(item.key, CoerceToPyTypeNode):
                    item.key = item.key.arg
                if not item.key.is_string_literal:
                    error(item.key.pos, 'Invalid struct field identifier')
                    item.key = StringNode(item.key.pos, value='<error>')
                else:
                    key = str(item.key.value)
                    member = dst_type.scope.lookup_here(key)
                    if not member:
                        error(item.key.pos, "struct '%s' has no field '%s'" % (dst_type, key))
                    else:
                        value = item.value
                        if isinstance(value, CoerceToPyTypeNode):
                            value = value.arg
                        item.value = value.coerce_to(member.type, env)
        else:
            return super(DictNode, self).coerce_to(dst_type, env)
        return self

    def release_errors(self):
        if False:
            for i in range(10):
                print('nop')
        for err in self.obj_conversion_errors:
            report_error(err)
        self.obj_conversion_errors = []
    gil_message = 'Constructing Python dict'

    def generate_evaluation_code(self, code):
        if False:
            i = 10
            return i + 15
        code.mark_pos(self.pos)
        self.allocate_temp_result(code)
        is_dict = self.type.is_pyobject
        if is_dict:
            self.release_errors()
            code.putln('%s = __Pyx_PyDict_NewPresized(%d); %s' % (self.result(), len(self.key_value_pairs), code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
        keys_seen = set()
        key_type = None
        needs_error_helper = False
        for item in self.key_value_pairs:
            item.generate_evaluation_code(code)
            if is_dict:
                if self.exclude_null_values:
                    code.putln('if (%s) {' % item.value.py_result())
                key = item.key
                if self.reject_duplicates:
                    if keys_seen is not None:
                        if not key.is_string_literal:
                            keys_seen = None
                        elif key.value in keys_seen:
                            keys_seen = None
                        elif key_type is not type(key.value):
                            if key_type is None:
                                key_type = type(key.value)
                                keys_seen.add(key.value)
                            else:
                                keys_seen = None
                        else:
                            keys_seen.add(key.value)
                    if keys_seen is None:
                        code.putln('if (unlikely(PyDict_Contains(%s, %s))) {' % (self.result(), key.py_result()))
                        needs_error_helper = True
                        code.putln('__Pyx_RaiseDoubleKeywordsError("function", %s); %s' % (key.py_result(), code.error_goto(item.pos)))
                        code.putln('} else {')
                code.put_error_if_neg(self.pos, 'PyDict_SetItem(%s, %s, %s)' % (self.result(), item.key.py_result(), item.value.py_result()))
                if self.reject_duplicates and keys_seen is None:
                    code.putln('}')
                if self.exclude_null_values:
                    code.putln('}')
            elif item.value.type.is_array:
                code.putln('memcpy(%s.%s, %s, sizeof(%s));' % (self.result(), item.key.value, item.value.result(), item.value.result()))
            else:
                code.putln('%s.%s = %s;' % (self.result(), item.key.value, item.value.result()))
            item.generate_disposal_code(code)
            item.free_temps(code)
        if needs_error_helper:
            code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseDoubleKeywords', 'FunctionArguments.c'))

    def annotate(self, code):
        if False:
            while True:
                i = 10
        for item in self.key_value_pairs:
            item.annotate(code)

    def as_python_dict(self):
        if False:
            return 10
        return dict([(key.value, value) for (key, value) in self.key_value_pairs])

class DictItemNode(ExprNode):
    subexprs = ['key', 'value']
    nogil_check = None

    def calculate_constant_result(self):
        if False:
            print('Hello World!')
        self.constant_result = (self.key.constant_result, self.value.constant_result)

    def analyse_types(self, env):
        if False:
            return 10
        self.key = self.key.analyse_types(env)
        self.value = self.value.analyse_types(env)
        self.key = self.key.coerce_to_pyobject(env)
        self.value = self.value.coerce_to_pyobject(env)
        return self

    def generate_evaluation_code(self, code):
        if False:
            while True:
                i = 10
        self.key.generate_evaluation_code(code)
        self.value.generate_evaluation_code(code)

    def generate_disposal_code(self, code):
        if False:
            i = 10
            return i + 15
        self.key.generate_disposal_code(code)
        self.value.generate_disposal_code(code)

    def free_temps(self, code):
        if False:
            i = 10
            return i + 15
        self.key.free_temps(code)
        self.value.free_temps(code)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter([self.key, self.value])

class SortedDictKeysNode(ExprNode):
    subexprs = ['arg']
    is_temp = True

    def __init__(self, arg):
        if False:
            print('Hello World!')
        ExprNode.__init__(self, arg.pos, arg=arg)
        self.type = Builtin.list_type

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        arg = self.arg.analyse_types(env)
        if arg.type is Builtin.dict_type:
            arg = arg.as_none_safe_node("'NoneType' object is not iterable")
        self.arg = arg
        return self

    def may_be_none(self):
        if False:
            print('Hello World!')
        return False

    def generate_result_code(self, code):
        if False:
            print('Hello World!')
        dict_result = self.arg.py_result()
        if self.arg.type is Builtin.dict_type:
            code.putln('%s = PyDict_Keys(%s); %s' % (self.result(), dict_result, code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
        else:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectCallMethod0', 'ObjectHandling.c'))
            keys_cname = code.intern_identifier(StringEncoding.EncodedString('keys'))
            code.putln('%s = __Pyx_PyObject_CallMethod0(%s, %s); %s' % (self.result(), dict_result, keys_cname, code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
            code.putln('if (unlikely(!PyList_Check(%s))) {' % self.result())
            self.generate_decref_set(code, 'PySequence_List(%s)' % self.result())
            code.putln(code.error_goto_if_null(self.result(), self.pos))
            self.generate_gotref(code)
            code.putln('}')
        code.put_error_if_neg(self.pos, 'PyList_Sort(%s)' % self.py_result())

class ModuleNameMixin(object):

    def get_py_mod_name(self, code):
        if False:
            print('Hello World!')
        return code.get_py_string_const(self.module_name, identifier=True)

    def get_py_qualified_name(self, code):
        if False:
            print('Hello World!')
        return code.get_py_string_const(self.qualname, identifier=True)

class ClassNode(ExprNode, ModuleNameMixin):
    subexprs = ['doc']
    type = py_object_type
    is_temp = True

    def analyse_annotations(self, env):
        if False:
            print('Hello World!')
        pass

    def infer_type(self, env):
        if False:
            while True:
                i = 10
        return py_object_type

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        if self.doc:
            self.doc = self.doc.analyse_types(env)
            self.doc = self.doc.coerce_to_pyobject(env)
        env.use_utility_code(UtilityCode.load_cached('CreateClass', 'ObjectHandling.c'))
        return self

    def may_be_none(self):
        if False:
            while True:
                i = 10
        return True
    gil_message = 'Constructing Python class'

    def generate_result_code(self, code):
        if False:
            return 10
        class_def_node = self.class_def_node
        cname = code.intern_identifier(self.name)
        if self.doc:
            code.put_error_if_neg(self.pos, 'PyDict_SetItem(%s, %s, %s)' % (class_def_node.dict.py_result(), code.intern_identifier(StringEncoding.EncodedString('__doc__')), self.doc.py_result()))
        py_mod_name = self.get_py_mod_name(code)
        qualname = self.get_py_qualified_name(code)
        code.putln('%s = __Pyx_CreateClass(%s, %s, %s, %s, %s); %s' % (self.result(), class_def_node.bases.py_result(), class_def_node.dict.py_result(), cname, qualname, py_mod_name, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

class Py3ClassNode(ExprNode):
    subexprs = []
    type = py_object_type
    force_type = False
    is_temp = True

    def infer_type(self, env):
        if False:
            i = 10
            return i + 15
        return py_object_type

    def analyse_types(self, env):
        if False:
            return 10
        return self

    def may_be_none(self):
        if False:
            return 10
        return True
    gil_message = 'Constructing Python class'

    def analyse_annotations(self, env):
        if False:
            print('Hello World!')
        from .AutoDocTransforms import AnnotationWriter
        position = self.class_def_node.pos
        dict_items = [DictItemNode(entry.pos, key=IdentifierStringNode(entry.pos, value=entry.name), value=entry.annotation.string) for entry in env.entries.values() if entry.annotation]
        if dict_items:
            annotations_dict = DictNode(position, key_value_pairs=dict_items)
            lhs = NameNode(position, name=StringEncoding.EncodedString(u'__annotations__'))
            lhs.entry = env.lookup_here(lhs.name) or env.declare_var(lhs.name, dict_type, position)
            node = SingleAssignmentNode(position, lhs=lhs, rhs=annotations_dict)
            node.analyse_declarations(env)
            self.class_def_node.body.stats.insert(0, node)

    def generate_result_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        code.globalstate.use_utility_code(UtilityCode.load_cached('Py3ClassCreate', 'ObjectHandling.c'))
        cname = code.intern_identifier(self.name)
        class_def_node = self.class_def_node
        mkw = class_def_node.mkw.py_result() if class_def_node.mkw else 'NULL'
        if class_def_node.metaclass:
            metaclass = class_def_node.metaclass.py_result()
        elif self.force_type:
            metaclass = '((PyObject*)&PyType_Type)'
        else:
            metaclass = '((PyObject*)&__Pyx_DefaultClassType)'
        code.putln('%s = __Pyx_Py3ClassCreate(%s, %s, %s, %s, %s, %d, %d); %s' % (self.result(), metaclass, cname, class_def_node.bases.py_result(), class_def_node.dict.py_result(), mkw, self.calculate_metaclass, self.allow_py2_metaclass, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

class PyClassMetaclassNode(ExprNode):
    subexprs = []

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        self.type = py_object_type
        self.is_temp = True
        return self

    def may_be_none(self):
        if False:
            print('Hello World!')
        return True

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        bases = self.class_def_node.bases
        mkw = self.class_def_node.mkw
        if mkw:
            code.globalstate.use_utility_code(UtilityCode.load_cached('Py3MetaclassGet', 'ObjectHandling.c'))
            call = '__Pyx_Py3MetaclassGet(%s, %s)' % (bases.result(), mkw.result())
        else:
            code.globalstate.use_utility_code(UtilityCode.load_cached('CalculateMetaclass', 'ObjectHandling.c'))
            call = '__Pyx_CalculateMetaclass(NULL, %s)' % bases.result()
        code.putln('%s = %s; %s' % (self.result(), call, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

class PyClassNamespaceNode(ExprNode, ModuleNameMixin):
    subexprs = ['doc']

    def analyse_types(self, env):
        if False:
            i = 10
            return i + 15
        if self.doc:
            self.doc = self.doc.analyse_types(env).coerce_to_pyobject(env)
        self.type = py_object_type
        self.is_temp = 1
        return self

    def may_be_none(self):
        if False:
            i = 10
            return i + 15
        return True

    def generate_result_code(self, code):
        if False:
            print('Hello World!')
        cname = code.intern_identifier(self.name)
        py_mod_name = self.get_py_mod_name(code)
        qualname = self.get_py_qualified_name(code)
        class_def_node = self.class_def_node
        null = '(PyObject *) NULL'
        doc_code = self.doc.result() if self.doc else null
        mkw = class_def_node.mkw.py_result() if class_def_node.mkw else null
        metaclass = class_def_node.metaclass.py_result() if class_def_node.metaclass else null
        code.putln('%s = __Pyx_Py3MetaclassPrepare(%s, %s, %s, %s, %s, %s, %s); %s' % (self.result(), metaclass, class_def_node.bases.result(), cname, qualname, mkw, py_mod_name, doc_code, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

class ClassCellInjectorNode(ExprNode):
    is_temp = True
    type = py_object_type
    subexprs = []
    is_active = False

    def analyse_expressions(self, env):
        if False:
            print('Hello World!')
        return self

    def generate_result_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        assert self.is_active
        code.putln('%s = PyList_New(0); %s' % (self.result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

    def generate_injection_code(self, code, classobj_cname):
        if False:
            i = 10
            return i + 15
        assert self.is_active
        code.globalstate.use_utility_code(UtilityCode.load_cached('CyFunctionClassCell', 'CythonFunction.c'))
        code.put_error_if_neg(self.pos, '__Pyx_CyFunction_InitClassCell(%s, %s)' % (self.result(), classobj_cname))

class ClassCellNode(ExprNode):
    subexprs = []
    is_temp = True
    is_generator = False
    type = py_object_type

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        return self

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        if not self.is_generator:
            code.putln('%s = __Pyx_CyFunction_GetClassObj(%s);' % (self.result(), Naming.self_cname))
        else:
            code.putln('%s =  %s->classobj;' % (self.result(), Naming.generator_cname))
        code.putln('if (!%s) { PyErr_SetString(PyExc_SystemError, "super(): empty __class__ cell"); %s }' % (self.result(), code.error_goto(self.pos)))
        code.put_incref(self.result(), py_object_type)

class PyCFunctionNode(ExprNode, ModuleNameMixin):
    subexprs = ['code_object', 'defaults_tuple', 'defaults_kwdict', 'annotations_dict']
    code_object = None
    binding = False
    def_node = None
    defaults = None
    defaults_struct = None
    defaults_pyobjects = 0
    defaults_tuple = None
    defaults_kwdict = None
    annotations_dict = None
    type = py_object_type
    is_temp = 1
    specialized_cpdefs = None
    is_specialization = False

    @classmethod
    def from_defnode(cls, node, binding):
        if False:
            for i in range(10):
                print('nop')
        return cls(node.pos, def_node=node, pymethdef_cname=node.entry.pymethdef_cname, binding=binding or node.specialized_cpdefs, specialized_cpdefs=node.specialized_cpdefs, code_object=CodeObjectNode(node))

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        if self.binding:
            self.analyse_default_args(env)
        return self

    def analyse_default_args(self, env):
        if False:
            i = 10
            return i + 15
        "\n        Handle non-literal function's default arguments.\n        "
        nonliteral_objects = []
        nonliteral_other = []
        default_args = []
        default_kwargs = []
        annotations = []
        must_use_constants = env.is_c_class_scope or (self.def_node.is_wrapper and env.is_module_scope)
        for arg in self.def_node.args:
            if arg.default:
                if not must_use_constants:
                    if arg.default.is_literal:
                        arg.default = DefaultLiteralArgNode(arg.pos, arg.default)
                        if arg.default.type:
                            arg.default = arg.default.coerce_to(arg.type, env)
                    else:
                        arg.is_dynamic = True
                        if arg.type.is_pyobject:
                            nonliteral_objects.append(arg)
                        else:
                            nonliteral_other.append(arg)
                if arg.default.type and arg.default.type.can_coerce_to_pyobject(env):
                    if arg.kw_only:
                        default_kwargs.append(arg)
                    else:
                        default_args.append(arg)
            if arg.annotation:
                arg.annotation = arg.annotation.analyse_types(env)
                annotations.append((arg.pos, arg.name, arg.annotation.string))
        for arg in (self.def_node.star_arg, self.def_node.starstar_arg):
            if arg and arg.annotation:
                arg.annotation = arg.annotation.analyse_types(env)
                annotations.append((arg.pos, arg.name, arg.annotation.string))
        annotation = self.def_node.return_type_annotation
        if annotation:
            self.def_node.return_type_annotation = annotation.analyse_types(env)
            annotations.append((annotation.pos, StringEncoding.EncodedString('return'), annotation.string))
        if nonliteral_objects or nonliteral_other:
            module_scope = env.global_scope()
            cname = module_scope.next_id(Naming.defaults_struct_prefix)
            scope = Symtab.StructOrUnionScope(cname)
            self.defaults = []
            for arg in nonliteral_objects:
                type_ = arg.type
                if type_.is_buffer:
                    type_ = type_.base
                entry = scope.declare_var(arg.name, type_, None, Naming.arg_prefix + arg.name, allow_pyobject=True)
                self.defaults.append((arg, entry))
            for arg in nonliteral_other:
                entry = scope.declare_var(arg.name, arg.type, None, Naming.arg_prefix + arg.name, allow_pyobject=False, allow_memoryview=True)
                self.defaults.append((arg, entry))
            entry = module_scope.declare_struct_or_union(None, 'struct', scope, 1, None, cname=cname)
            self.defaults_struct = scope
            self.defaults_pyobjects = len(nonliteral_objects)
            for (arg, entry) in self.defaults:
                arg.default_value = '%s->%s' % (Naming.dynamic_args_cname, entry.cname)
            self.def_node.defaults_struct = self.defaults_struct.name
        if default_args or default_kwargs:
            if self.defaults_struct is None:
                if default_args:
                    defaults_tuple = TupleNode(self.pos, args=[arg.default for arg in default_args])
                    self.defaults_tuple = defaults_tuple.analyse_types(env).coerce_to_pyobject(env)
                if default_kwargs:
                    defaults_kwdict = DictNode(self.pos, key_value_pairs=[DictItemNode(arg.pos, key=IdentifierStringNode(arg.pos, value=arg.name), value=arg.default) for arg in default_kwargs])
                    self.defaults_kwdict = defaults_kwdict.analyse_types(env)
            elif not self.specialized_cpdefs:
                if default_args:
                    defaults_tuple = DefaultsTupleNode(self.pos, default_args, self.defaults_struct)
                else:
                    defaults_tuple = NoneNode(self.pos)
                if default_kwargs:
                    defaults_kwdict = DefaultsKwDictNode(self.pos, default_kwargs, self.defaults_struct)
                else:
                    defaults_kwdict = NoneNode(self.pos)
                defaults_getter = Nodes.DefNode(self.pos, args=[], star_arg=None, starstar_arg=None, body=Nodes.ReturnStatNode(self.pos, return_type=py_object_type, value=TupleNode(self.pos, args=[defaults_tuple, defaults_kwdict])), decorators=None, name=StringEncoding.EncodedString('__defaults__'))
                module_scope = env.global_scope()
                defaults_getter.analyse_declarations(module_scope)
                defaults_getter = defaults_getter.analyse_expressions(module_scope)
                defaults_getter.body = defaults_getter.body.analyse_expressions(defaults_getter.local_scope)
                defaults_getter.py_wrapper_required = False
                defaults_getter.pymethdef_required = False
                self.def_node.defaults_getter = defaults_getter
        if annotations:
            annotations_dict = DictNode(self.pos, key_value_pairs=[DictItemNode(pos, key=IdentifierStringNode(pos, value=name), value=value) for (pos, name, value) in annotations])
            self.annotations_dict = annotations_dict.analyse_types(env)

    def may_be_none(self):
        if False:
            return 10
        return False
    gil_message = 'Constructing Python function'

    def closure_result_code(self):
        if False:
            print('Hello World!')
        return 'NULL'

    def generate_result_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        if self.binding:
            self.generate_cyfunction_code(code)
        else:
            self.generate_pycfunction_code(code)

    def generate_pycfunction_code(self, code):
        if False:
            return 10
        py_mod_name = self.get_py_mod_name(code)
        code.putln('%s = PyCFunction_NewEx(&%s, %s, %s); %s' % (self.result(), self.pymethdef_cname, self.closure_result_code(), py_mod_name, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

    def generate_cyfunction_code(self, code):
        if False:
            while True:
                i = 10
        if self.specialized_cpdefs:
            def_node = self.specialized_cpdefs[0]
        else:
            def_node = self.def_node
        if self.specialized_cpdefs or self.is_specialization:
            code.globalstate.use_utility_code(UtilityCode.load_cached('FusedFunction', 'CythonFunction.c'))
            constructor = '__pyx_FusedFunction_New'
        else:
            code.globalstate.use_utility_code(UtilityCode.load_cached('CythonFunction', 'CythonFunction.c'))
            constructor = '__Pyx_CyFunction_New'
        if self.code_object:
            code_object_result = self.code_object.py_result()
        else:
            code_object_result = 'NULL'
        flags = []
        if def_node.is_staticmethod:
            flags.append('__Pyx_CYFUNCTION_STATICMETHOD')
        elif def_node.is_classmethod:
            flags.append('__Pyx_CYFUNCTION_CLASSMETHOD')
        if def_node.local_scope.parent_scope.is_c_class_scope and (not def_node.entry.is_anonymous):
            flags.append('__Pyx_CYFUNCTION_CCLASS')
        if def_node.is_coroutine:
            flags.append('__Pyx_CYFUNCTION_COROUTINE')
        if flags:
            flags = ' | '.join(flags)
        else:
            flags = '0'
        code.putln('%s = %s(&%s, %s, %s, %s, %s, %s, %s); %s' % (self.result(), constructor, self.pymethdef_cname, flags, self.get_py_qualified_name(code), self.closure_result_code(), self.get_py_mod_name(code), Naming.moddict_cname, code_object_result, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)
        if def_node.requires_classobj:
            assert code.pyclass_stack, 'pyclass_stack is empty'
            class_node = code.pyclass_stack[-1]
            code.put_incref(self.py_result(), py_object_type)
            code.putln('PyList_Append(%s, %s);' % (class_node.class_cell.result(), self.result()))
            self.generate_giveref(code)
        if self.defaults:
            code.putln('if (!__Pyx_CyFunction_InitDefaults(%s, sizeof(%s), %d)) %s' % (self.result(), self.defaults_struct.name, self.defaults_pyobjects, code.error_goto(self.pos)))
            defaults = '__Pyx_CyFunction_Defaults(%s, %s)' % (self.defaults_struct.name, self.result())
            for (arg, entry) in self.defaults:
                arg.generate_assignment_code(code, target='%s->%s' % (defaults, entry.cname))
        if self.defaults_tuple:
            code.putln('__Pyx_CyFunction_SetDefaultsTuple(%s, %s);' % (self.result(), self.defaults_tuple.py_result()))
        if not self.specialized_cpdefs:
            if self.defaults_kwdict:
                code.putln('__Pyx_CyFunction_SetDefaultsKwDict(%s, %s);' % (self.result(), self.defaults_kwdict.py_result()))
            if def_node.defaults_getter:
                code.putln('__Pyx_CyFunction_SetDefaultsGetter(%s, %s);' % (self.result(), def_node.defaults_getter.entry.pyfunc_cname))
            if self.annotations_dict:
                code.putln('__Pyx_CyFunction_SetAnnotationsDict(%s, %s);' % (self.result(), self.annotations_dict.py_result()))

class InnerFunctionNode(PyCFunctionNode):
    binding = True
    needs_closure_code = True

    def closure_result_code(self):
        if False:
            print('Hello World!')
        if self.needs_closure_code:
            return '((PyObject*)%s)' % Naming.cur_scope_cname
        return 'NULL'

class CodeObjectNode(ExprNode):
    subexprs = ['varnames']
    is_temp = False
    result_code = None

    def __init__(self, def_node):
        if False:
            while True:
                i = 10
        ExprNode.__init__(self, def_node.pos, def_node=def_node)
        args = list(def_node.args)
        local_vars = [arg for arg in def_node.local_scope.var_entries if arg.name]
        self.varnames = TupleNode(def_node.pos, args=[IdentifierStringNode(arg.pos, value=arg.name) for arg in args + local_vars], is_temp=0, is_literal=1)

    def may_be_none(self):
        if False:
            i = 10
            return i + 15
        return False

    def calculate_result_code(self, code=None):
        if False:
            while True:
                i = 10
        if self.result_code is None:
            self.result_code = code.get_py_const(py_object_type, 'codeobj', cleanup_level=2)
        return self.result_code

    def generate_result_code(self, code):
        if False:
            i = 10
            return i + 15
        if self.result_code is None:
            self.result_code = code.get_py_const(py_object_type, 'codeobj', cleanup_level=2)
        code = code.get_cached_constants_writer(self.result_code)
        if code is None:
            return
        code.mark_pos(self.pos)
        func = self.def_node
        func_name = code.get_py_string_const(func.name, identifier=True, is_str=False, unicode_value=func.name)
        file_path = StringEncoding.bytes_literal(func.pos[0].get_filenametable_entry().encode('utf8'), 'utf8')
        file_path_const = code.get_py_string_const(file_path, identifier=False, is_str=True)
        flags = ['CO_OPTIMIZED', 'CO_NEWLOCALS']
        if self.def_node.star_arg:
            flags.append('CO_VARARGS')
        if self.def_node.starstar_arg:
            flags.append('CO_VARKEYWORDS')
        if self.def_node.is_asyncgen:
            flags.append('CO_ASYNC_GENERATOR')
        elif self.def_node.is_coroutine:
            flags.append('CO_COROUTINE')
        elif self.def_node.is_generator:
            flags.append('CO_GENERATOR')
        code.putln('%s = (PyObject*)__Pyx_PyCode_New(%d, %d, %d, %d, 0, %s, %s, %s, %s, %s, %s, %s, %s, %s, %d, %s); %s' % (self.result_code, len(func.args) - func.num_kwonly_args, func.num_posonly_args, func.num_kwonly_args, len(self.varnames.args), '|'.join(flags) or '0', Naming.empty_bytes, Naming.empty_tuple, Naming.empty_tuple, self.varnames.result(), Naming.empty_tuple, Naming.empty_tuple, file_path_const, func_name, self.pos[1], Naming.empty_bytes, code.error_goto_if_null(self.result_code, self.pos)))

class DefaultLiteralArgNode(ExprNode):
    subexprs = []
    is_literal = True
    is_temp = False

    def __init__(self, pos, arg):
        if False:
            print('Hello World!')
        super(DefaultLiteralArgNode, self).__init__(pos)
        self.arg = arg
        self.constant_result = arg.constant_result
        self.type = self.arg.type
        self.evaluated = False

    def analyse_types(self, env):
        if False:
            i = 10
            return i + 15
        return self

    def generate_result_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        pass

    def generate_evaluation_code(self, code):
        if False:
            i = 10
            return i + 15
        if not self.evaluated:
            self.arg.generate_evaluation_code(code)
            self.evaluated = True

    def result(self):
        if False:
            while True:
                i = 10
        return self.type.cast_code(self.arg.result())

class DefaultNonLiteralArgNode(ExprNode):
    subexprs = []

    def __init__(self, pos, arg, defaults_struct):
        if False:
            while True:
                i = 10
        super(DefaultNonLiteralArgNode, self).__init__(pos)
        self.arg = arg
        self.defaults_struct = defaults_struct

    def analyse_types(self, env):
        if False:
            i = 10
            return i + 15
        self.type = self.arg.type
        self.is_temp = False
        return self

    def generate_result_code(self, code):
        if False:
            return 10
        pass

    def result(self):
        if False:
            for i in range(10):
                print('nop')
        return '__Pyx_CyFunction_Defaults(%s, %s)->%s' % (self.defaults_struct.name, Naming.self_cname, self.defaults_struct.lookup(self.arg.name).cname)

class DefaultsTupleNode(TupleNode):

    def __init__(self, pos, defaults, defaults_struct):
        if False:
            return 10
        args = []
        for arg in defaults:
            if not arg.default.is_literal:
                arg = DefaultNonLiteralArgNode(pos, arg, defaults_struct)
            else:
                arg = arg.default
            args.append(arg)
        super(DefaultsTupleNode, self).__init__(pos, args=args)

    def analyse_types(self, env, skip_children=False):
        if False:
            print('Hello World!')
        return super(DefaultsTupleNode, self).analyse_types(env, skip_children).coerce_to_pyobject(env)

class DefaultsKwDictNode(DictNode):

    def __init__(self, pos, defaults, defaults_struct):
        if False:
            while True:
                i = 10
        items = []
        for arg in defaults:
            name = IdentifierStringNode(arg.pos, value=arg.name)
            if not arg.default.is_literal:
                arg = DefaultNonLiteralArgNode(pos, arg, defaults_struct)
            else:
                arg = arg.default
            items.append(DictItemNode(arg.pos, key=name, value=arg))
        super(DefaultsKwDictNode, self).__init__(pos, key_value_pairs=items)

class LambdaNode(InnerFunctionNode):
    child_attrs = ['def_node']
    name = StringEncoding.EncodedString('<lambda>')

    def analyse_declarations(self, env):
        if False:
            print('Hello World!')
        if hasattr(self, 'lambda_name'):
            return
        self.lambda_name = self.def_node.lambda_name = env.next_id('lambda')
        self.def_node.no_assignment_synthesis = True
        self.def_node.pymethdef_required = True
        self.def_node.is_cyfunction = True
        self.def_node.analyse_declarations(env)
        self.pymethdef_cname = self.def_node.entry.pymethdef_cname
        env.add_lambda_def(self.def_node)

    def analyse_types(self, env):
        if False:
            i = 10
            return i + 15
        self.def_node = self.def_node.analyse_expressions(env)
        return super(LambdaNode, self).analyse_types(env)

    def generate_result_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        self.def_node.generate_execution_code(code)
        super(LambdaNode, self).generate_result_code(code)

class GeneratorExpressionNode(LambdaNode):
    name = StringEncoding.EncodedString('genexpr')
    binding = False
    child_attrs = LambdaNode.child_attrs + ['call_parameters']
    subexprs = LambdaNode.subexprs + ['call_parameters']

    def __init__(self, pos, *args, **kwds):
        if False:
            i = 10
            return i + 15
        super(GeneratorExpressionNode, self).__init__(pos, *args, **kwds)
        self.call_parameters = []

    def analyse_declarations(self, env):
        if False:
            return 10
        if hasattr(self, 'genexpr_name'):
            return
        self.genexpr_name = env.next_id('genexpr')
        super(GeneratorExpressionNode, self).analyse_declarations(env)
        self.def_node.pymethdef_required = False
        self.def_node.py_wrapper_required = False
        self.def_node.is_cyfunction = False
        self.def_node.entry.signature = TypeSlots.pyfunction_noargs
        if isinstance(self.loop, Nodes._ForInStatNode):
            assert isinstance(self.loop.iterator, ScopedExprNode)
            self.loop.iterator.init_scope(None, env)
        else:
            assert isinstance(self.loop, Nodes.ForFromStatNode)

    def generate_result_code(self, code):
        if False:
            print('Hello World!')
        args_to_call = [self.closure_result_code()] + [cp.result() for cp in self.call_parameters]
        args_to_call = ', '.join(args_to_call)
        code.putln('%s = %s(%s); %s' % (self.result(), self.def_node.entry.pyfunc_cname, args_to_call, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

class YieldExprNode(ExprNode):
    subexprs = ['arg']
    type = py_object_type
    label_num = 0
    is_yield_from = False
    is_await = False
    in_async_gen = False
    expr_keyword = 'yield'

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        if not self.label_num or (self.is_yield_from and self.in_async_gen):
            error(self.pos, "'%s' not supported here" % self.expr_keyword)
        self.is_temp = 1
        if self.arg is not None:
            self.arg = self.arg.analyse_types(env)
            if not self.arg.type.is_pyobject:
                self.coerce_yield_argument(env)
        return self

    def coerce_yield_argument(self, env):
        if False:
            print('Hello World!')
        self.arg = self.arg.coerce_to_pyobject(env)

    def generate_evaluation_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        if self.arg:
            self.arg.generate_evaluation_code(code)
            self.arg.make_owned_reference(code)
            code.putln('%s = %s;' % (Naming.retval_cname, self.arg.result_as(py_object_type)))
            self.arg.generate_post_assignment_code(code)
            self.arg.free_temps(code)
        else:
            code.put_init_to_py_none(Naming.retval_cname, py_object_type)
        self.generate_yield_code(code)

    def generate_yield_code(self, code):
        if False:
            i = 10
            return i + 15
        "\n        Generate the code to return the argument in 'Naming.retval_cname'\n        and to continue at the yield label.\n        "
        (label_num, label_name) = code.new_yield_label(self.expr_keyword.replace(' ', '_'))
        code.use_label(label_name)
        saved = []
        code.funcstate.closure_temps.reset()
        for (cname, type, manage_ref) in code.funcstate.temps_in_use():
            save_cname = code.funcstate.closure_temps.allocate_temp(type)
            saved.append((cname, save_cname, type))
            if type.is_cpp_class:
                code.globalstate.use_utility_code(UtilityCode.load_cached('MoveIfSupported', 'CppSupport.cpp'))
                cname = '__PYX_STD_MOVE_IF_SUPPORTED(%s)' % cname
            else:
                code.put_xgiveref(cname, type)
            code.putln('%s->%s = %s;' % (Naming.cur_scope_cname, save_cname, cname))
        code.put_xgiveref(Naming.retval_cname, py_object_type)
        profile = code.globalstate.directives['profile']
        linetrace = code.globalstate.directives['linetrace']
        if profile or linetrace:
            code.put_trace_return(Naming.retval_cname, nogil=not code.funcstate.gil_owned)
        code.put_finish_refcount_context()
        if code.funcstate.current_except is not None:
            code.putln('__Pyx_Coroutine_SwapException(%s);' % Naming.generator_cname)
        else:
            code.putln('__Pyx_Coroutine_ResetAndClearException(%s);' % Naming.generator_cname)
        code.putln('/* return from %sgenerator, %sing value */' % ('async ' if self.in_async_gen else '', 'await' if self.is_await else 'yield'))
        code.putln('%s->resume_label = %d;' % (Naming.generator_cname, label_num))
        if self.in_async_gen and (not self.is_await):
            code.putln('return __Pyx__PyAsyncGenValueWrapperNew(%s);' % Naming.retval_cname)
        else:
            code.putln('return %s;' % Naming.retval_cname)
        code.put_label(label_name)
        for (cname, save_cname, type) in saved:
            save_cname = '%s->%s' % (Naming.cur_scope_cname, save_cname)
            if type.is_cpp_class:
                save_cname = '__PYX_STD_MOVE_IF_SUPPORTED(%s)' % save_cname
            code.putln('%s = %s;' % (cname, save_cname))
            if type.is_pyobject:
                code.putln('%s = 0;' % save_cname)
                code.put_xgotref(cname, type)
            elif type.is_memoryviewslice:
                code.putln('%s.memview = NULL; %s.data = NULL;' % (save_cname, save_cname))
        self.generate_sent_value_handling_code(code, Naming.sent_value_cname)
        if self.result_is_used:
            self.allocate_temp_result(code)
            code.put('%s = %s; ' % (self.result(), Naming.sent_value_cname))
            code.put_incref(self.result(), py_object_type)

    def generate_sent_value_handling_code(self, code, value_cname):
        if False:
            while True:
                i = 10
        code.putln(code.error_goto_if_null(value_cname, self.pos))

class _YieldDelegationExprNode(YieldExprNode):

    def yield_from_func(self, code):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def generate_evaluation_code(self, code, source_cname=None, decref_source=False):
        if False:
            print('Hello World!')
        if source_cname is None:
            self.arg.generate_evaluation_code(code)
        code.putln('%s = %s(%s, %s);' % (Naming.retval_cname, self.yield_from_func(code), Naming.generator_cname, self.arg.py_result() if source_cname is None else source_cname))
        if source_cname is None:
            self.arg.generate_disposal_code(code)
            self.arg.free_temps(code)
        elif decref_source:
            code.put_decref_clear(source_cname, py_object_type)
        code.put_xgotref(Naming.retval_cname, py_object_type)
        code.putln('if (likely(%s)) {' % Naming.retval_cname)
        self.generate_yield_code(code)
        code.putln('} else {')
        if self.result_is_used:
            self.fetch_iteration_result(code)
        else:
            self.handle_iteration_exception(code)
        code.putln('}')

    def fetch_iteration_result(self, code):
        if False:
            while True:
                i = 10
        code.putln('%s = NULL;' % self.result())
        code.put_error_if_neg(self.pos, '__Pyx_PyGen_FetchStopIterationValue(&%s)' % self.result())
        self.generate_gotref(code)

    def handle_iteration_exception(self, code):
        if False:
            i = 10
            return i + 15
        code.putln('PyObject* exc_type = __Pyx_PyErr_CurrentExceptionType();')
        code.putln('if (exc_type) {')
        code.putln('if (likely(exc_type == PyExc_StopIteration || (exc_type != PyExc_GeneratorExit && __Pyx_PyErr_GivenExceptionMatches(exc_type, PyExc_StopIteration)))) PyErr_Clear();')
        code.putln('else %s' % code.error_goto(self.pos))
        code.putln('}')

class YieldFromExprNode(_YieldDelegationExprNode):
    is_yield_from = True
    expr_keyword = 'yield from'

    def coerce_yield_argument(self, env):
        if False:
            i = 10
            return i + 15
        if not self.arg.type.is_string:
            error(self.pos, 'yielding from non-Python object not supported')
        self.arg = self.arg.coerce_to_pyobject(env)

    def yield_from_func(self, code):
        if False:
            return 10
        code.globalstate.use_utility_code(UtilityCode.load_cached('GeneratorYieldFrom', 'Coroutine.c'))
        return '__Pyx_Generator_Yield_From'

class AwaitExprNode(_YieldDelegationExprNode):
    is_await = True
    expr_keyword = 'await'

    def coerce_yield_argument(self, env):
        if False:
            i = 10
            return i + 15
        if self.arg is not None:
            self.arg = self.arg.coerce_to_pyobject(env)

    def yield_from_func(self, code):
        if False:
            return 10
        code.globalstate.use_utility_code(UtilityCode.load_cached('CoroutineYieldFrom', 'Coroutine.c'))
        return '__Pyx_Coroutine_Yield_From'

class AwaitIterNextExprNode(AwaitExprNode):

    def _generate_break(self, code):
        if False:
            for i in range(10):
                print('nop')
        code.globalstate.use_utility_code(UtilityCode.load_cached('StopAsyncIteration', 'Coroutine.c'))
        code.putln('PyObject* exc_type = __Pyx_PyErr_CurrentExceptionType();')
        code.putln('if (unlikely(exc_type && (exc_type == __Pyx_PyExc_StopAsyncIteration || ( exc_type != PyExc_StopIteration && exc_type != PyExc_GeneratorExit && __Pyx_PyErr_GivenExceptionMatches(exc_type, __Pyx_PyExc_StopAsyncIteration))))) {')
        code.putln('PyErr_Clear();')
        code.putln('break;')
        code.putln('}')

    def fetch_iteration_result(self, code):
        if False:
            while True:
                i = 10
        assert code.break_label, "AwaitIterNextExprNode outside of 'async for' loop"
        self._generate_break(code)
        super(AwaitIterNextExprNode, self).fetch_iteration_result(code)

    def generate_sent_value_handling_code(self, code, value_cname):
        if False:
            for i in range(10):
                print('nop')
        assert code.break_label, "AwaitIterNextExprNode outside of 'async for' loop"
        code.putln('if (unlikely(!%s)) {' % value_cname)
        self._generate_break(code)
        code.putln(code.error_goto(self.pos))
        code.putln('}')

class GlobalsExprNode(AtomicExprNode):
    type = dict_type
    is_temp = 1

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        env.use_utility_code(Builtin.globals_utility_code)
        return self
    gil_message = 'Constructing globals dict'

    def may_be_none(self):
        if False:
            return 10
        return False

    def generate_result_code(self, code):
        if False:
            return 10
        code.putln('%s = __Pyx_Globals(); %s' % (self.result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

class LocalsDictItemNode(DictItemNode):

    def analyse_types(self, env):
        if False:
            i = 10
            return i + 15
        self.key = self.key.analyse_types(env)
        self.value = self.value.analyse_types(env)
        self.key = self.key.coerce_to_pyobject(env)
        if self.value.type.can_coerce_to_pyobject(env):
            self.value = self.value.coerce_to_pyobject(env)
        else:
            self.value = None
        return self

class FuncLocalsExprNode(DictNode):

    def __init__(self, pos, env):
        if False:
            return 10
        local_vars = sorted([entry.name for entry in env.entries.values() if entry.name])
        items = [LocalsDictItemNode(pos, key=IdentifierStringNode(pos, value=var), value=NameNode(pos, name=var, allow_null=True)) for var in local_vars]
        DictNode.__init__(self, pos, key_value_pairs=items, exclude_null_values=True)

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        node = super(FuncLocalsExprNode, self).analyse_types(env)
        node.key_value_pairs = [i for i in node.key_value_pairs if i.value is not None]
        return node

class PyClassLocalsExprNode(AtomicExprNode):

    def __init__(self, pos, pyclass_dict):
        if False:
            for i in range(10):
                print('nop')
        AtomicExprNode.__init__(self, pos)
        self.pyclass_dict = pyclass_dict

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        self.type = self.pyclass_dict.type
        self.is_temp = False
        return self

    def may_be_none(self):
        if False:
            i = 10
            return i + 15
        return False

    def result(self):
        if False:
            print('Hello World!')
        return self.pyclass_dict.result()

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        pass

def LocalsExprNode(pos, scope_node, env):
    if False:
        print('Hello World!')
    if env.is_module_scope:
        return GlobalsExprNode(pos)
    if env.is_py_class_scope:
        return PyClassLocalsExprNode(pos, scope_node.dict)
    return FuncLocalsExprNode(pos, env)
compile_time_unary_operators = {'not': operator.not_, '~': operator.inv, '-': operator.neg, '+': operator.pos}

class UnopNode(ExprNode):
    subexprs = ['operand']
    infix = True
    is_inc_dec_op = False

    def calculate_constant_result(self):
        if False:
            while True:
                i = 10
        func = compile_time_unary_operators[self.operator]
        self.constant_result = func(self.operand.constant_result)

    def compile_time_value(self, denv):
        if False:
            for i in range(10):
                print('nop')
        func = compile_time_unary_operators.get(self.operator)
        if not func:
            error(self.pos, "Unary '%s' not supported in compile-time expression" % self.operator)
        operand = self.operand.compile_time_value(denv)
        try:
            return func(operand)
        except Exception as e:
            self.compile_time_value_error(e)

    def infer_type(self, env):
        if False:
            print('Hello World!')
        operand_type = self.operand.infer_type(env)
        if operand_type.is_cpp_class or operand_type.is_ptr:
            cpp_type = operand_type.find_cpp_operation_type(self.operator)
            if cpp_type is not None:
                return cpp_type
        return self.infer_unop_type(env, operand_type)

    def infer_unop_type(self, env, operand_type):
        if False:
            return 10
        if operand_type.is_pyobject:
            return py_object_type
        else:
            return operand_type

    def may_be_none(self):
        if False:
            return 10
        if self.operand.type and self.operand.type.is_builtin_type:
            if self.operand.type is not type_type:
                return False
        return ExprNode.may_be_none(self)

    def analyse_types(self, env):
        if False:
            return 10
        self.operand = self.operand.analyse_types(env)
        if self.is_pythran_operation(env):
            self.type = PythranExpr(pythran_unaryop_type(self.operator, self.operand.type))
            self.is_temp = 1
        elif self.is_py_operation():
            self.coerce_operand_to_pyobject(env)
            self.type = py_object_type
            self.is_temp = 1
        elif self.is_cpp_operation():
            self.analyse_cpp_operation(env)
        else:
            self.analyse_c_operation(env)
        return self

    def check_const(self):
        if False:
            print('Hello World!')
        return self.operand.check_const()

    def is_py_operation(self):
        if False:
            for i in range(10):
                print('nop')
        return self.operand.type.is_pyobject or self.operand.type.is_ctuple

    def is_pythran_operation(self, env):
        if False:
            return 10
        np_pythran = has_np_pythran(env)
        op_type = self.operand.type
        return np_pythran and (op_type.is_buffer or op_type.is_pythran_expr)

    def nogil_check(self, env):
        if False:
            while True:
                i = 10
        if self.is_py_operation():
            self.gil_error()

    def is_cpp_operation(self):
        if False:
            return 10
        type = self.operand.type
        return type.is_cpp_class

    def coerce_operand_to_pyobject(self, env):
        if False:
            while True:
                i = 10
        self.operand = self.operand.coerce_to_pyobject(env)

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        if self.type.is_pythran_expr:
            code.putln('// Pythran unaryop')
            code.putln('__Pyx_call_destructor(%s);' % self.result())
            code.putln('new (&%s) decltype(%s){%s%s};' % (self.result(), self.result(), self.operator, self.operand.pythran_result()))
        elif self.operand.type.is_pyobject:
            self.generate_py_operation_code(code)
        elif self.is_temp:
            if self.is_cpp_operation() and self.exception_check == '+':
                translate_cpp_exception(code, self.pos, '%s = %s %s;' % (self.result(), self.operator, self.operand.result()), self.result() if self.type.is_pyobject else None, self.exception_value, self.in_nogil_context)
            else:
                code.putln('%s = %s %s;' % (self.result(), self.operator, self.operand.result()))

    def generate_py_operation_code(self, code):
        if False:
            i = 10
            return i + 15
        function = self.py_operation_function(code)
        code.putln('%s = %s(%s); %s' % (self.result(), function, self.operand.py_result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

    def type_error(self):
        if False:
            i = 10
            return i + 15
        if not self.operand.type.is_error:
            error(self.pos, "Invalid operand type for '%s' (%s)" % (self.operator, self.operand.type))
        self.type = PyrexTypes.error_type

    def analyse_cpp_operation(self, env, overload_check=True):
        if False:
            return 10
        operand_types = [self.operand.type]
        if self.is_inc_dec_op and (not self.is_prefix):
            operand_types.append(PyrexTypes.c_int_type)
        entry = env.lookup_operator_for_types(self.pos, self.operator, operand_types)
        if overload_check and (not entry):
            self.type_error()
            return
        if entry:
            self.exception_check = entry.type.exception_check
            self.exception_value = entry.type.exception_value
            if self.exception_check == '+':
                self.is_temp = True
                if needs_cpp_exception_conversion(self):
                    env.use_utility_code(UtilityCode.load_cached('CppExceptionConversion', 'CppSupport.cpp'))
        else:
            self.exception_check = ''
            self.exception_value = ''
        if self.is_inc_dec_op and (not self.is_prefix):
            cpp_type = self.operand.type.find_cpp_operation_type(self.operator, operand_type=PyrexTypes.c_int_type)
        else:
            cpp_type = self.operand.type.find_cpp_operation_type(self.operator)
        if overload_check and cpp_type is None:
            error(self.pos, "'%s' operator not defined for %s" % (self.operator, type))
            self.type_error()
            return
        self.type = cpp_type

class NotNode(UnopNode):
    operator = '!'
    type = PyrexTypes.c_bint_type

    def calculate_constant_result(self):
        if False:
            for i in range(10):
                print('nop')
        self.constant_result = not self.operand.constant_result

    def compile_time_value(self, denv):
        if False:
            return 10
        operand = self.operand.compile_time_value(denv)
        try:
            return not operand
        except Exception as e:
            self.compile_time_value_error(e)

    def infer_unop_type(self, env, operand_type):
        if False:
            return 10
        return PyrexTypes.c_bint_type

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        self.operand = self.operand.analyse_types(env)
        operand_type = self.operand.type
        if operand_type.is_cpp_class:
            self.analyse_cpp_operation(env)
        else:
            self.operand = self.operand.coerce_to_boolean(env)
        return self

    def calculate_result_code(self):
        if False:
            for i in range(10):
                print('nop')
        return '(!%s)' % self.operand.result()

class UnaryPlusNode(UnopNode):
    operator = '+'

    def analyse_c_operation(self, env):
        if False:
            print('Hello World!')
        self.type = PyrexTypes.widest_numeric_type(self.operand.type, PyrexTypes.c_int_type)

    def py_operation_function(self, code):
        if False:
            i = 10
            return i + 15
        return 'PyNumber_Positive'

    def calculate_result_code(self):
        if False:
            return 10
        if self.is_cpp_operation():
            return '(+%s)' % self.operand.result()
        else:
            return self.operand.result()

class UnaryMinusNode(UnopNode):
    operator = '-'

    def analyse_c_operation(self, env):
        if False:
            for i in range(10):
                print('nop')
        if self.operand.type.is_numeric:
            self.type = PyrexTypes.widest_numeric_type(self.operand.type, PyrexTypes.c_int_type)
        elif self.operand.type.is_enum:
            self.type = PyrexTypes.c_int_type
        else:
            self.type_error()
        if self.type.is_complex:
            self.infix = False

    def py_operation_function(self, code):
        if False:
            return 10
        return 'PyNumber_Negative'

    def calculate_result_code(self):
        if False:
            while True:
                i = 10
        if self.infix:
            return '(-%s)' % self.operand.result()
        else:
            return '%s(%s)' % (self.operand.type.unary_op('-'), self.operand.result())

    def get_constant_c_result_code(self):
        if False:
            print('Hello World!')
        value = self.operand.get_constant_c_result_code()
        if value:
            return '(-%s)' % value

class TildeNode(UnopNode):

    def analyse_c_operation(self, env):
        if False:
            for i in range(10):
                print('nop')
        if self.operand.type.is_int:
            self.type = PyrexTypes.widest_numeric_type(self.operand.type, PyrexTypes.c_int_type)
        elif self.operand.type.is_enum:
            self.type = PyrexTypes.c_int_type
        else:
            self.type_error()

    def py_operation_function(self, code):
        if False:
            return 10
        return 'PyNumber_Invert'

    def calculate_result_code(self):
        if False:
            i = 10
            return i + 15
        return '(~%s)' % self.operand.result()

class CUnopNode(UnopNode):

    def is_py_operation(self):
        if False:
            i = 10
            return i + 15
        return False

class DereferenceNode(CUnopNode):
    operator = '*'

    def infer_unop_type(self, env, operand_type):
        if False:
            while True:
                i = 10
        if operand_type.is_ptr:
            return operand_type.base_type
        else:
            return PyrexTypes.error_type

    def analyse_c_operation(self, env):
        if False:
            return 10
        if self.operand.type.is_ptr:
            if env.is_cpp:
                self.type = PyrexTypes.CReferenceType(self.operand.type.base_type)
            else:
                self.type = self.operand.type.base_type
        else:
            self.type_error()

    def calculate_result_code(self):
        if False:
            return 10
        return '(*%s)' % self.operand.result()

class DecrementIncrementNode(CUnopNode):
    is_inc_dec_op = True

    def type_error(self):
        if False:
            return 10
        if not self.operand.type.is_error:
            if self.is_prefix:
                error(self.pos, "No match for 'operator%s' (operand type is '%s')" % (self.operator, self.operand.type))
            else:
                error(self.pos, "No 'operator%s(int)' declared for postfix '%s' (operand type is '%s')" % (self.operator, self.operator, self.operand.type))
        self.type = PyrexTypes.error_type

    def analyse_c_operation(self, env):
        if False:
            for i in range(10):
                print('nop')
        if self.operand.type.is_numeric:
            self.type = PyrexTypes.widest_numeric_type(self.operand.type, PyrexTypes.c_int_type)
        elif self.operand.type.is_ptr:
            self.type = self.operand.type
        else:
            self.type_error()

    def calculate_result_code(self):
        if False:
            while True:
                i = 10
        if self.is_prefix:
            return '(%s%s)' % (self.operator, self.operand.result())
        else:
            return '(%s%s)' % (self.operand.result(), self.operator)

def inc_dec_constructor(is_prefix, operator):
    if False:
        for i in range(10):
            print('nop')
    return lambda pos, **kwds: DecrementIncrementNode(pos, is_prefix=is_prefix, operator=operator, **kwds)

class AmpersandNode(CUnopNode):
    operator = '&'

    def infer_unop_type(self, env, operand_type):
        if False:
            print('Hello World!')
        return PyrexTypes.c_ptr_type(operand_type)

    def analyse_types(self, env):
        if False:
            return 10
        self.operand = self.operand.analyse_types(env)
        argtype = self.operand.type
        if argtype.is_cpp_class:
            self.analyse_cpp_operation(env, overload_check=False)
        if not (argtype.is_cfunction or argtype.is_reference or self.operand.is_addressable()):
            if argtype.is_memoryviewslice:
                self.error('Cannot take address of memoryview slice')
            else:
                self.error('Taking address of non-lvalue (type %s)' % argtype)
            return self
        if argtype.is_pyobject:
            self.error('Cannot take address of Python %s' % ("variable '%s'" % self.operand.name if self.operand.is_name else "object attribute '%s'" % self.operand.attribute if self.operand.is_attribute else 'object'))
            return self
        if not argtype.is_cpp_class or not self.type:
            self.type = PyrexTypes.c_ptr_type(argtype)
        return self

    def check_const(self):
        if False:
            return 10
        return self.operand.check_const_addr()

    def error(self, mess):
        if False:
            for i in range(10):
                print('nop')
        error(self.pos, mess)
        self.type = PyrexTypes.error_type
        self.result_code = '<error>'

    def calculate_result_code(self):
        if False:
            for i in range(10):
                print('nop')
        return '(&%s)' % self.operand.result()

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        if self.operand.type.is_cpp_class and self.exception_check == '+':
            translate_cpp_exception(code, self.pos, '%s = %s %s;' % (self.result(), self.operator, self.operand.result()), self.result() if self.type.is_pyobject else None, self.exception_value, self.in_nogil_context)
unop_node_classes = {'+': UnaryPlusNode, '-': UnaryMinusNode, '~': TildeNode}

def unop_node(pos, operator, operand):
    if False:
        print('Hello World!')
    if isinstance(operand, IntNode) and operator == '-':
        return IntNode(pos=operand.pos, value=str(-Utils.str_to_number(operand.value)), longness=operand.longness, unsigned=operand.unsigned)
    elif isinstance(operand, UnopNode) and operand.operator == operator in '+-':
        warning(pos, 'Python has no increment/decrement operator: %s%sx == %s(%sx) == x' % ((operator,) * 4), 5)
    return unop_node_classes[operator](pos, operator=operator, operand=operand)

class TypecastNode(ExprNode):
    subexprs = ['operand']
    base_type = declarator = type = None

    def type_dependencies(self, env):
        if False:
            print('Hello World!')
        return ()

    def infer_type(self, env):
        if False:
            while True:
                i = 10
        if self.type is None:
            base_type = self.base_type.analyse(env)
            (_, self.type) = self.declarator.analyse(base_type, env)
        return self.type

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        if self.type is None:
            base_type = self.base_type.analyse(env)
            (_, self.type) = self.declarator.analyse(base_type, env)
        if self.operand.has_constant_result():
            self.calculate_constant_result()
        if self.type.is_cfunction:
            error(self.pos, 'Cannot cast to a function type')
            self.type = PyrexTypes.error_type
        self.operand = self.operand.analyse_types(env)
        if self.type is PyrexTypes.c_bint_type:
            return self.operand.coerce_to_boolean(env)
        to_py = self.type.is_pyobject
        from_py = self.operand.type.is_pyobject
        if from_py and (not to_py) and self.operand.is_ephemeral():
            if not self.type.is_numeric and (not self.type.is_cpp_class):
                error(self.pos, 'Casting temporary Python object to non-numeric non-Python type')
        if to_py and (not from_py):
            if self.type is bytes_type and self.operand.type.is_int:
                return CoerceIntToBytesNode(self.operand, env)
            elif self.operand.type.can_coerce_to_pyobject(env):
                self.result_ctype = py_object_type
                self.operand = self.operand.coerce_to(self.type, env)
            else:
                if self.operand.type.is_ptr:
                    if not (self.operand.type.base_type.is_void or self.operand.type.base_type.is_struct):
                        error(self.pos, 'Python objects cannot be cast from pointers of primitive types')
                else:
                    warning(self.pos, 'No conversion from %s to %s, python object pointer used.' % (self.operand.type, self.type))
                self.operand = self.operand.coerce_to_simple(env)
        elif from_py and (not to_py):
            if self.type.create_from_py_utility_code(env):
                self.operand = self.operand.coerce_to(self.type, env)
            elif self.type.is_ptr:
                if not (self.type.base_type.is_void or self.type.base_type.is_struct):
                    error(self.pos, 'Python objects cannot be cast to pointers of primitive types')
            else:
                warning(self.pos, 'No conversion from %s to %s, python object pointer used.' % (self.type, self.operand.type))
        elif from_py and to_py:
            if self.typecheck:
                self.operand = PyTypeTestNode(self.operand, self.type, env, notnone=True)
            elif isinstance(self.operand, SliceIndexNode):
                self.operand = self.operand.coerce_to(self.type, env)
        elif self.type.is_complex and self.operand.type.is_complex:
            self.operand = self.operand.coerce_to_simple(env)
        elif self.operand.type.is_fused:
            self.operand = self.operand.coerce_to(self.type, env)
        if self.type.is_ptr and self.type.base_type.is_cfunction and self.type.base_type.nogil:
            op_type = self.operand.type
            if op_type.is_ptr:
                op_type = op_type.base_type
            if op_type.is_cfunction and (not op_type.nogil):
                warning(self.pos, 'Casting a GIL-requiring function into a nogil function circumvents GIL validation', 1)
        return self

    def is_simple(self):
        if False:
            return 10
        return self.operand.is_simple()

    def is_ephemeral(self):
        if False:
            for i in range(10):
                print('nop')
        return self.operand.is_ephemeral()

    def nonlocally_immutable(self):
        if False:
            return 10
        return self.is_temp or self.operand.nonlocally_immutable()

    def nogil_check(self, env):
        if False:
            while True:
                i = 10
        if self.type and self.type.is_pyobject and self.is_temp:
            self.gil_error()

    def check_const(self):
        if False:
            i = 10
            return i + 15
        return self.operand.check_const()

    def calculate_constant_result(self):
        if False:
            for i in range(10):
                print('nop')
        self.constant_result = self.calculate_result_code(self.operand.constant_result)

    def calculate_result_code(self, operand_result=None):
        if False:
            i = 10
            return i + 15
        if operand_result is None:
            operand_result = self.operand.result()
        if self.type.is_complex:
            operand_result = self.operand.result()
            if self.operand.type.is_complex:
                real_part = self.type.real_type.cast_code(self.operand.type.real_code(operand_result))
                imag_part = self.type.real_type.cast_code(self.operand.type.imag_code(operand_result))
            else:
                real_part = self.type.real_type.cast_code(operand_result)
                imag_part = '0'
            return '%s(%s, %s)' % (self.type.from_parts, real_part, imag_part)
        else:
            return self.type.cast_code(operand_result)

    def get_constant_c_result_code(self):
        if False:
            return 10
        operand_result = self.operand.get_constant_c_result_code()
        if operand_result:
            return self.type.cast_code(operand_result)

    def result_as(self, type):
        if False:
            return 10
        if self.type.is_pyobject and (not self.is_temp):
            return self.operand.result_as(type)
        else:
            return ExprNode.result_as(self, type)

    def generate_result_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        if self.is_temp:
            code.putln('%s = (PyObject *)%s;' % (self.result(), self.operand.result()))
            code.put_incref(self.result(), self.ctype())
ERR_START = 'Start may not be given'
ERR_NOT_STOP = 'Stop must be provided to indicate shape'
ERR_STEPS = 'Strides may only be given to indicate contiguity. Consider slicing it after conversion'
ERR_NOT_POINTER = 'Can only create cython.array from pointer or array'
ERR_BASE_TYPE = 'Pointer base type does not match cython.array base type'

class CythonArrayNode(ExprNode):
    """
    Used when a pointer of base_type is cast to a memoryviewslice with that
    base type. i.e.

        <int[:M:1, :N]> p

    creates a fortran-contiguous cython.array.

    We leave the type set to object so coercions to object are more efficient
    and less work. Acquiring a memoryviewslice from this will be just as
    efficient. ExprNode.coerce_to() will do the additional typecheck on
    self.compile_time_type

    This also handles <int[:, :]> my_c_array


    operand             ExprNode                 the thing we're casting
    base_type_node      MemoryViewSliceTypeNode  the cast expression node
    """
    subexprs = ['operand', 'shapes']
    shapes = None
    is_temp = True
    mode = 'c'
    array_dtype = None
    shape_type = PyrexTypes.c_py_ssize_t_type

    def analyse_types(self, env):
        if False:
            print('Hello World!')
        from . import MemoryView
        self.operand = self.operand.analyse_types(env)
        if self.array_dtype:
            array_dtype = self.array_dtype
        else:
            array_dtype = self.base_type_node.base_type_node.analyse(env)
        axes = self.base_type_node.axes
        self.type = error_type
        self.shapes = []
        ndim = len(axes)
        base_type = self.operand.type
        if not self.operand.type.is_ptr and (not self.operand.type.is_array):
            error(self.operand.pos, ERR_NOT_POINTER)
            return self
        array_dimension_sizes = []
        if base_type.is_array:
            while base_type.is_array:
                array_dimension_sizes.append(base_type.size)
                base_type = base_type.base_type
        elif base_type.is_ptr:
            base_type = base_type.base_type
        else:
            error(self.pos, 'unexpected base type %s found' % base_type)
            return self
        if not (base_type.same_as(array_dtype) or base_type.is_void):
            error(self.operand.pos, ERR_BASE_TYPE)
            return self
        elif self.operand.type.is_array and len(array_dimension_sizes) != ndim:
            error(self.operand.pos, 'Expected %d dimensions, array has %d dimensions' % (ndim, len(array_dimension_sizes)))
            return self
        for (axis_no, axis) in enumerate(axes):
            if not axis.start.is_none:
                error(axis.start.pos, ERR_START)
                return self
            if axis.stop.is_none:
                if array_dimension_sizes:
                    dimsize = array_dimension_sizes[axis_no]
                    axis.stop = IntNode(self.pos, value=str(dimsize), constant_result=dimsize, type=PyrexTypes.c_int_type)
                else:
                    error(axis.pos, ERR_NOT_STOP)
                    return self
            axis.stop = axis.stop.analyse_types(env)
            shape = axis.stop.coerce_to(self.shape_type, env)
            if not shape.is_literal:
                shape.coerce_to_temp(env)
            self.shapes.append(shape)
            first_or_last = axis_no in (0, ndim - 1)
            if not axis.step.is_none and first_or_last:
                axis.step = axis.step.analyse_types(env)
                if not axis.step.type.is_int and axis.step.is_literal and (not axis.step.type.is_error):
                    error(axis.step.pos, 'Expected an integer literal')
                    return self
                if axis.step.compile_time_value(env) != 1:
                    error(axis.step.pos, ERR_STEPS)
                    return self
                if axis_no == 0:
                    self.mode = 'fortran'
            elif not axis.step.is_none and (not first_or_last):
                error(axis.step.pos, ERR_STEPS)
                return self
        if not self.operand.is_name:
            self.operand = self.operand.coerce_to_temp(env)
        axes = [('direct', 'follow')] * len(axes)
        if self.mode == 'fortran':
            axes[0] = ('direct', 'contig')
        else:
            axes[-1] = ('direct', 'contig')
        self.coercion_type = PyrexTypes.MemoryViewSliceType(array_dtype, axes)
        self.coercion_type.validate_memslice_dtype(self.pos)
        self.type = self.get_cython_array_type(env)
        MemoryView.use_cython_array_utility_code(env)
        env.use_utility_code(MemoryView.typeinfo_to_format_code)
        return self

    def allocate_temp_result(self, code):
        if False:
            print('Hello World!')
        if self.temp_code:
            raise RuntimeError('temp allocated multiple times')
        self.temp_code = code.funcstate.allocate_temp(self.type, True)

    def infer_type(self, env):
        if False:
            for i in range(10):
                print('nop')
        return self.get_cython_array_type(env)

    def get_cython_array_type(self, env):
        if False:
            while True:
                i = 10
        cython_scope = env.global_scope().context.cython_scope
        cython_scope.load_cythonscope()
        return cython_scope.viewscope.lookup('array').type

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        from . import Buffer
        shapes = [self.shape_type.cast_code(shape.result()) for shape in self.shapes]
        dtype = self.coercion_type.dtype
        shapes_temp = code.funcstate.allocate_temp(py_object_type, True)
        format_temp = code.funcstate.allocate_temp(py_object_type, True)
        itemsize = 'sizeof(%s)' % dtype.empty_declaration_code()
        type_info = Buffer.get_type_information_cname(code, dtype)
        if self.operand.type.is_ptr:
            code.putln('if (!%s) {' % self.operand.result())
            code.putln('PyErr_SetString(PyExc_ValueError,"Cannot create cython.array from NULL pointer");')
            code.putln(code.error_goto(self.operand.pos))
            code.putln('}')
        code.putln('%s = __pyx_format_from_typeinfo(&%s); %s' % (format_temp, type_info, code.error_goto_if_null(format_temp, self.pos)))
        code.put_gotref(format_temp, py_object_type)
        buildvalue_fmt = ' __PYX_BUILD_PY_SSIZE_T ' * len(shapes)
        code.putln('%s = Py_BuildValue("(" %s ")", %s); %s' % (shapes_temp, buildvalue_fmt, ', '.join(shapes), code.error_goto_if_null(shapes_temp, self.pos)))
        code.put_gotref(shapes_temp, py_object_type)
        code.putln('%s = __pyx_array_new(%s, %s, PyBytes_AS_STRING(%s), "%s", (char *) %s); %s' % (self.result(), shapes_temp, itemsize, format_temp, self.mode, self.operand.result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

        def dispose(temp):
            if False:
                return 10
            code.put_decref_clear(temp, py_object_type)
            code.funcstate.release_temp(temp)
        dispose(shapes_temp)
        dispose(format_temp)

    @classmethod
    def from_carray(cls, src_node, env):
        if False:
            i = 10
            return i + 15
        '\n        Given a C array type, return a CythonArrayNode\n        '
        pos = src_node.pos
        base_type = src_node.type
        none_node = NoneNode(pos)
        axes = []
        while base_type.is_array:
            axes.append(SliceNode(pos, start=none_node, stop=none_node, step=none_node))
            base_type = base_type.base_type
        axes[-1].step = IntNode(pos, value='1', is_c_literal=True)
        memslicenode = Nodes.MemoryViewSliceTypeNode(pos, axes=axes, base_type_node=base_type)
        result = CythonArrayNode(pos, base_type_node=memslicenode, operand=src_node, array_dtype=base_type)
        result = result.analyse_types(env)
        return result

class SizeofNode(ExprNode):
    type = PyrexTypes.c_size_t_type

    def check_const(self):
        if False:
            while True:
                i = 10
        return True

    def generate_result_code(self, code):
        if False:
            print('Hello World!')
        pass

class SizeofTypeNode(SizeofNode):
    subexprs = []
    arg_type = None

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        if 0 and self.base_type.module_path:
            path = self.base_type.module_path
            obj = env.lookup(path[0])
            if obj.as_module is None:
                operand = NameNode(pos=self.pos, name=path[0])
                for attr in path[1:]:
                    operand = AttributeNode(pos=self.pos, obj=operand, attribute=attr)
                operand = AttributeNode(pos=self.pos, obj=operand, attribute=self.base_type.name)
                node = SizeofVarNode(self.pos, operand=operand).analyse_types(env)
                return node
        if self.arg_type is None:
            base_type = self.base_type.analyse(env)
            (_, arg_type) = self.declarator.analyse(base_type, env)
            self.arg_type = arg_type
        self.check_type()
        return self

    def check_type(self):
        if False:
            i = 10
            return i + 15
        arg_type = self.arg_type
        if not arg_type:
            return
        if arg_type.is_pyobject and (not arg_type.is_extension_type):
            error(self.pos, 'Cannot take sizeof Python object')
        elif arg_type.is_void:
            error(self.pos, 'Cannot take sizeof void')
        elif not arg_type.is_complete():
            error(self.pos, "Cannot take sizeof incomplete type '%s'" % arg_type)

    def calculate_result_code(self):
        if False:
            print('Hello World!')
        if self.arg_type.is_extension_type:
            arg_code = self.arg_type.declaration_code('', deref=1)
        else:
            arg_code = self.arg_type.empty_declaration_code()
        return '(sizeof(%s))' % arg_code

class SizeofVarNode(SizeofNode):
    subexprs = ['operand']

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        operand_as_type = self.operand.analyse_as_type(env)
        if operand_as_type:
            self.arg_type = operand_as_type
            if self.arg_type.is_fused:
                try:
                    self.arg_type = self.arg_type.specialize(env.fused_to_specific)
                except CannotSpecialize:
                    error(self.operand.pos, 'Type cannot be specialized since it is not a fused argument to this function')
            self.__class__ = SizeofTypeNode
            self.check_type()
        else:
            self.operand = self.operand.analyse_types(env)
        return self

    def calculate_result_code(self):
        if False:
            i = 10
            return i + 15
        return '(sizeof(%s))' % self.operand.result()

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        pass

class TypeidNode(ExprNode):
    subexprs = ['operand']
    arg_type = None
    is_variable = None
    is_temp = 1

    def get_type_info_type(self, env):
        if False:
            i = 10
            return i + 15
        env_module = env
        while not env_module.is_module_scope:
            env_module = env_module.outer_scope
        typeinfo_module = env_module.find_module('libcpp.typeinfo', self.pos)
        typeinfo_entry = typeinfo_module.lookup('type_info')
        return PyrexTypes.CFakeReferenceType(PyrexTypes.c_const_type(typeinfo_entry.type))
    cpp_message = 'typeid operator'

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        if not self.type:
            self.type = PyrexTypes.error_type
        self.cpp_check(env)
        type_info = self.get_type_info_type(env)
        if not type_info:
            self.error("The 'libcpp.typeinfo' module must be cimported to use the typeid() operator")
            return self
        if self.operand is None:
            return self
        self.type = type_info
        as_type = self.operand.analyse_as_specialized_type(env)
        if as_type:
            self.arg_type = as_type
            self.is_type = True
            self.operand = None
        else:
            self.arg_type = self.operand.analyse_types(env)
            self.is_type = False
            self.operand = None
            if self.arg_type.type.is_pyobject:
                self.error('Cannot use typeid on a Python object')
                return self
            elif self.arg_type.type.is_void:
                self.error('Cannot use typeid on void')
                return self
            elif not self.arg_type.type.is_complete():
                self.error("Cannot use typeid on incomplete type '%s'" % self.arg_type.type)
                return self
        env.use_utility_code(UtilityCode.load_cached('CppExceptionConversion', 'CppSupport.cpp'))
        return self

    def error(self, mess):
        if False:
            i = 10
            return i + 15
        error(self.pos, mess)
        self.type = PyrexTypes.error_type
        self.result_code = '<error>'

    def check_const(self):
        if False:
            return 10
        return True

    def calculate_result_code(self):
        if False:
            while True:
                i = 10
        return self.temp_code

    def generate_result_code(self, code):
        if False:
            i = 10
            return i + 15
        if self.is_type:
            arg_code = self.arg_type.empty_declaration_code()
        else:
            arg_code = self.arg_type.result()
        translate_cpp_exception(code, self.pos, '%s = typeid(%s);' % (self.temp_code, arg_code), None, None, self.in_nogil_context)

class TypeofNode(ExprNode):
    literal = None
    type = py_object_type
    subexprs = ['literal']

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        self.operand = self.operand.analyse_types(env)
        value = StringEncoding.EncodedString(str(self.operand.type))
        literal = StringNode(self.pos, value=value)
        literal = literal.analyse_types(env)
        self.literal = literal.coerce_to_pyobject(env)
        return self

    def analyse_as_type(self, env):
        if False:
            print('Hello World!')
        self.operand = self.operand.analyse_types(env)
        return self.operand.type

    def may_be_none(self):
        if False:
            while True:
                i = 10
        return False

    def generate_evaluation_code(self, code):
        if False:
            print('Hello World!')
        self.literal.generate_evaluation_code(code)

    def calculate_result_code(self):
        if False:
            print('Hello World!')
        return self.literal.calculate_result_code()
try:
    matmul_operator = operator.matmul
except AttributeError:

    def matmul_operator(a, b):
        if False:
            return 10
        try:
            func = a.__matmul__
        except AttributeError:
            func = b.__rmatmul__
        return func(a, b)
compile_time_binary_operators = {'<': operator.lt, '<=': operator.le, '==': operator.eq, '!=': operator.ne, '>=': operator.ge, '>': operator.gt, 'is': operator.is_, 'is_not': operator.is_not, '+': operator.add, '&': operator.and_, '/': operator.truediv, '//': operator.floordiv, '<<': operator.lshift, '%': operator.mod, '*': operator.mul, '|': operator.or_, '**': operator.pow, '>>': operator.rshift, '-': operator.sub, '^': operator.xor, '@': matmul_operator, 'in': lambda x, seq: x in seq, 'not_in': lambda x, seq: x not in seq}

def get_compile_time_binop(node):
    if False:
        print('Hello World!')
    func = compile_time_binary_operators.get(node.operator)
    if not func:
        error(node.pos, "Binary '%s' not supported in compile-time expression" % node.operator)
    return func

class BinopNode(ExprNode):
    subexprs = ['operand1', 'operand2']
    inplace = False

    def calculate_constant_result(self):
        if False:
            i = 10
            return i + 15
        func = compile_time_binary_operators[self.operator]
        self.constant_result = func(self.operand1.constant_result, self.operand2.constant_result)

    def compile_time_value(self, denv):
        if False:
            return 10
        func = get_compile_time_binop(self)
        operand1 = self.operand1.compile_time_value(denv)
        operand2 = self.operand2.compile_time_value(denv)
        try:
            return func(operand1, operand2)
        except Exception as e:
            self.compile_time_value_error(e)

    def infer_type(self, env):
        if False:
            for i in range(10):
                print('nop')
        return self.result_type(self.operand1.infer_type(env), self.operand2.infer_type(env), env)

    def analyse_types(self, env):
        if False:
            i = 10
            return i + 15
        self.operand1 = self.operand1.analyse_types(env)
        self.operand2 = self.operand2.analyse_types(env)
        self.analyse_operation(env)
        return self

    def analyse_operation(self, env):
        if False:
            while True:
                i = 10
        if self.is_pythran_operation(env):
            self.type = self.result_type(self.operand1.type, self.operand2.type, env)
            assert self.type.is_pythran_expr
            self.is_temp = 1
        elif self.is_py_operation():
            self.coerce_operands_to_pyobjects(env)
            self.type = self.result_type(self.operand1.type, self.operand2.type, env)
            assert self.type.is_pyobject
            self.is_temp = 1
        elif self.is_cpp_operation():
            self.analyse_cpp_operation(env)
        else:
            self.analyse_c_operation(env)

    def is_py_operation(self):
        if False:
            print('Hello World!')
        return self.is_py_operation_types(self.operand1.type, self.operand2.type)

    def is_py_operation_types(self, type1, type2):
        if False:
            return 10
        return type1.is_pyobject or type2.is_pyobject or type1.is_ctuple or type2.is_ctuple

    def is_pythran_operation(self, env):
        if False:
            return 10
        return self.is_pythran_operation_types(self.operand1.type, self.operand2.type, env)

    def is_pythran_operation_types(self, type1, type2, env):
        if False:
            i = 10
            return i + 15
        return has_np_pythran(env) and (is_pythran_supported_operation_type(type1) and is_pythran_supported_operation_type(type2)) and (is_pythran_expr(type1) or is_pythran_expr(type2))

    def is_cpp_operation(self):
        if False:
            print('Hello World!')
        return self.operand1.type.is_cpp_class or self.operand2.type.is_cpp_class

    def analyse_cpp_operation(self, env):
        if False:
            for i in range(10):
                print('nop')
        entry = env.lookup_operator(self.operator, [self.operand1, self.operand2])
        if not entry:
            self.type_error()
            return
        func_type = entry.type
        self.exception_check = func_type.exception_check
        self.exception_value = func_type.exception_value
        if self.exception_check == '+':
            self.is_temp = 1
            if needs_cpp_exception_conversion(self):
                env.use_utility_code(UtilityCode.load_cached('CppExceptionConversion', 'CppSupport.cpp'))
        if func_type.is_ptr:
            func_type = func_type.base_type
        if len(func_type.args) == 1:
            self.operand2 = self.operand2.coerce_to(func_type.args[0].type, env)
        else:
            self.operand1 = self.operand1.coerce_to(func_type.args[0].type, env)
            self.operand2 = self.operand2.coerce_to(func_type.args[1].type, env)
        self.type = func_type.return_type

    def result_type(self, type1, type2, env):
        if False:
            print('Hello World!')
        if self.is_pythran_operation_types(type1, type2, env):
            return PythranExpr(pythran_binop_type(self.operator, type1, type2))
        if self.is_py_operation_types(type1, type2):
            if type2.is_string:
                type2 = Builtin.bytes_type
            elif type2.is_pyunicode_ptr:
                type2 = Builtin.unicode_type
            if type1.is_string:
                type1 = Builtin.bytes_type
            elif type1.is_pyunicode_ptr:
                type1 = Builtin.unicode_type
            if type1.is_builtin_type or type2.is_builtin_type:
                if type1 is type2 and self.operator in '**%+|&^':
                    return type1
                result_type = self.infer_builtin_types_operation(type1, type2)
                if result_type is not None:
                    return result_type
            return py_object_type
        elif type1.is_error or type2.is_error:
            return PyrexTypes.error_type
        else:
            return self.compute_c_result_type(type1, type2)

    def infer_builtin_types_operation(self, type1, type2):
        if False:
            print('Hello World!')
        return None

    def nogil_check(self, env):
        if False:
            i = 10
            return i + 15
        if self.is_py_operation():
            self.gil_error()

    def coerce_operands_to_pyobjects(self, env):
        if False:
            i = 10
            return i + 15
        self.operand1 = self.operand1.coerce_to_pyobject(env)
        self.operand2 = self.operand2.coerce_to_pyobject(env)

    def check_const(self):
        if False:
            print('Hello World!')
        return self.operand1.check_const() and self.operand2.check_const()

    def is_ephemeral(self):
        if False:
            print('Hello World!')
        return super(BinopNode, self).is_ephemeral() or self.operand1.is_ephemeral() or self.operand2.is_ephemeral()

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        type1 = self.operand1.type
        type2 = self.operand2.type
        if self.type.is_pythran_expr:
            code.putln('// Pythran binop')
            code.putln('__Pyx_call_destructor(%s);' % self.result())
            if self.operator == '**':
                code.putln('new (&%s) decltype(%s){pythonic::numpy::functor::power{}(%s, %s)};' % (self.result(), self.result(), self.operand1.pythran_result(), self.operand2.pythran_result()))
            else:
                code.putln('new (&%s) decltype(%s){%s %s %s};' % (self.result(), self.result(), self.operand1.pythran_result(), self.operator, self.operand2.pythran_result()))
        elif type1.is_pyobject or type2.is_pyobject:
            function = self.py_operation_function(code)
            extra_args = ', Py_None' if self.operator == '**' else ''
            op1_result = self.operand1.py_result() if type1.is_pyobject else self.operand1.result()
            op2_result = self.operand2.py_result() if type2.is_pyobject else self.operand2.result()
            code.putln('%s = %s(%s, %s%s); %s' % (self.result(), function, op1_result, op2_result, extra_args, code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
        elif self.is_temp:
            if self.is_cpp_operation() and self.exception_check == '+':
                translate_cpp_exception(code, self.pos, '%s = %s;' % (self.result(), self.calculate_result_code()), self.result() if self.type.is_pyobject else None, self.exception_value, self.in_nogil_context)
            else:
                code.putln('%s = %s;' % (self.result(), self.calculate_result_code()))

    def type_error(self):
        if False:
            i = 10
            return i + 15
        if not (self.operand1.type.is_error or self.operand2.type.is_error):
            error(self.pos, "Invalid operand types for '%s' (%s; %s)" % (self.operator, self.operand1.type, self.operand2.type))
        self.type = PyrexTypes.error_type

class CBinopNode(BinopNode):

    def analyse_types(self, env):
        if False:
            i = 10
            return i + 15
        node = BinopNode.analyse_types(self, env)
        if node.is_py_operation():
            node.type = PyrexTypes.error_type
        return node

    def py_operation_function(self, code):
        if False:
            while True:
                i = 10
        return ''

    def calculate_result_code(self):
        if False:
            i = 10
            return i + 15
        return '(%s %s %s)' % (self.operand1.result(), self.operator, self.operand2.result())

    def compute_c_result_type(self, type1, type2):
        if False:
            return 10
        cpp_type = None
        if type1.is_cpp_class or type1.is_ptr:
            cpp_type = type1.find_cpp_operation_type(self.operator, type2)
        if cpp_type is None and (type2.is_cpp_class or type2.is_ptr):
            cpp_type = type2.find_cpp_operation_type(self.operator, type1)
        return cpp_type

def c_binop_constructor(operator):
    if False:
        print('Hello World!')

    def make_binop_node(pos, **operands):
        if False:
            while True:
                i = 10
        return CBinopNode(pos, operator=operator, **operands)
    return make_binop_node

class NumBinopNode(BinopNode):
    infix = True
    overflow_check = False
    overflow_bit_node = None

    def analyse_c_operation(self, env):
        if False:
            for i in range(10):
                print('nop')
        type1 = self.operand1.type
        type2 = self.operand2.type
        self.type = self.compute_c_result_type(type1, type2)
        if not self.type:
            self.type_error()
            return
        if self.type.is_complex:
            self.infix = False
        if self.type.is_int and env.directives['overflowcheck'] and (self.operator in self.overflow_op_names):
            if self.operator in ('+', '*') and self.operand1.has_constant_result() and (not self.operand2.has_constant_result()):
                (self.operand1, self.operand2) = (self.operand2, self.operand1)
            self.overflow_check = True
            self.overflow_fold = env.directives['overflowcheck.fold']
            self.func = self.type.overflow_check_binop(self.overflow_op_names[self.operator], env, const_rhs=self.operand2.has_constant_result())
            self.is_temp = True
        if not self.infix or (type1.is_numeric and type2.is_numeric):
            self.operand1 = self.operand1.coerce_to(self.type, env)
            self.operand2 = self.operand2.coerce_to(self.type, env)

    def compute_c_result_type(self, type1, type2):
        if False:
            print('Hello World!')
        if self.c_types_okay(type1, type2):
            widest_type = PyrexTypes.widest_numeric_type(type1, type2)
            if widest_type is PyrexTypes.c_bint_type:
                if self.operator not in '|^&':
                    widest_type = PyrexTypes.c_int_type
            else:
                widest_type = PyrexTypes.widest_numeric_type(widest_type, PyrexTypes.c_int_type)
            return widest_type
        else:
            return None

    def may_be_none(self):
        if False:
            i = 10
            return i + 15
        if self.type and self.type.is_builtin_type:
            return False
        type1 = self.operand1.type
        type2 = self.operand2.type
        if type1 and type1.is_builtin_type and type2 and type2.is_builtin_type:
            return False
        return super(NumBinopNode, self).may_be_none()

    def get_constant_c_result_code(self):
        if False:
            return 10
        value1 = self.operand1.get_constant_c_result_code()
        value2 = self.operand2.get_constant_c_result_code()
        if value1 and value2:
            return '(%s %s %s)' % (value1, self.operator, value2)
        else:
            return None

    def c_types_okay(self, type1, type2):
        if False:
            for i in range(10):
                print('nop')
        return (type1.is_numeric or type1.is_enum) and (type2.is_numeric or type2.is_enum)

    def generate_evaluation_code(self, code):
        if False:
            return 10
        if self.overflow_check:
            self.overflow_bit_node = self
            self.overflow_bit = code.funcstate.allocate_temp(PyrexTypes.c_int_type, manage_ref=False)
            code.putln('%s = 0;' % self.overflow_bit)
        super(NumBinopNode, self).generate_evaluation_code(code)
        if self.overflow_check:
            code.putln('if (unlikely(%s)) {' % self.overflow_bit)
            code.putln('PyErr_SetString(PyExc_OverflowError, "value too large");')
            code.putln(code.error_goto(self.pos))
            code.putln('}')
            code.funcstate.release_temp(self.overflow_bit)

    def calculate_result_code(self):
        if False:
            return 10
        if self.overflow_bit_node is not None:
            return '%s(%s, %s, &%s)' % (self.func, self.operand1.result(), self.operand2.result(), self.overflow_bit_node.overflow_bit)
        elif self.type.is_cpp_class or self.infix:
            if is_pythran_expr(self.type):
                (result1, result2) = (self.operand1.pythran_result(), self.operand2.pythran_result())
            else:
                (result1, result2) = (self.operand1.result(), self.operand2.result())
            return '(%s %s %s)' % (result1, self.operator, result2)
        else:
            func = self.type.binary_op(self.operator)
            if func is None:
                error(self.pos, 'binary operator %s not supported for %s' % (self.operator, self.type))
            return '%s(%s, %s)' % (func, self.operand1.result(), self.operand2.result())

    def is_py_operation_types(self, type1, type2):
        if False:
            while True:
                i = 10
        return type1.is_unicode_char or type2.is_unicode_char or BinopNode.is_py_operation_types(self, type1, type2)

    def py_operation_function(self, code):
        if False:
            i = 10
            return i + 15
        function_name = self.py_functions[self.operator]
        if self.inplace:
            function_name = function_name.replace('PyNumber_', 'PyNumber_InPlace')
        return function_name
    py_functions = {'|': 'PyNumber_Or', '^': 'PyNumber_Xor', '&': 'PyNumber_And', '<<': 'PyNumber_Lshift', '>>': 'PyNumber_Rshift', '+': 'PyNumber_Add', '-': 'PyNumber_Subtract', '*': 'PyNumber_Multiply', '@': '__Pyx_PyNumber_MatrixMultiply', '/': '__Pyx_PyNumber_Divide', '//': 'PyNumber_FloorDivide', '%': 'PyNumber_Remainder', '**': 'PyNumber_Power'}
    overflow_op_names = {'+': 'add', '-': 'sub', '*': 'mul', '<<': 'lshift'}

class IntBinopNode(NumBinopNode):

    def c_types_okay(self, type1, type2):
        if False:
            for i in range(10):
                print('nop')
        return (type1.is_int or type1.is_enum) and (type2.is_int or type2.is_enum)

class AddNode(NumBinopNode):

    def is_py_operation_types(self, type1, type2):
        if False:
            return 10
        if type1.is_string and type2.is_string or (type1.is_pyunicode_ptr and type2.is_pyunicode_ptr):
            return 1
        else:
            return NumBinopNode.is_py_operation_types(self, type1, type2)

    def infer_builtin_types_operation(self, type1, type2):
        if False:
            i = 10
            return i + 15
        string_types = (bytes_type, bytearray_type, str_type, basestring_type, unicode_type)
        if type1 in string_types and type2 in string_types:
            return string_types[max(string_types.index(type1), string_types.index(type2))]
        return None

    def compute_c_result_type(self, type1, type2):
        if False:
            for i in range(10):
                print('nop')
        if (type1.is_ptr or type1.is_array) and (type2.is_int or type2.is_enum):
            return type1
        elif (type2.is_ptr or type2.is_array) and (type1.is_int or type1.is_enum):
            return type2
        else:
            return NumBinopNode.compute_c_result_type(self, type1, type2)

    def py_operation_function(self, code):
        if False:
            for i in range(10):
                print('nop')
        (type1, type2) = (self.operand1.type, self.operand2.type)
        func = None
        if type1 is unicode_type or type2 is unicode_type:
            if type1 in (unicode_type, str_type) and type2 in (unicode_type, str_type):
                is_unicode_concat = True
            elif isinstance(self.operand1, FormattedValueNode) or isinstance(self.operand2, FormattedValueNode):
                is_unicode_concat = True
            else:
                is_unicode_concat = False
            if is_unicode_concat:
                if self.inplace or self.operand1.is_temp:
                    code.globalstate.use_utility_code(UtilityCode.load_cached('UnicodeConcatInPlace', 'ObjectHandling.c'))
                func = '__Pyx_PyUnicode_Concat'
        elif type1 is str_type and type2 is str_type:
            code.globalstate.use_utility_code(UtilityCode.load_cached('StrConcatInPlace', 'ObjectHandling.c'))
            func = '__Pyx_PyStr_Concat'
        if func:
            if self.inplace or self.operand1.is_temp:
                func += 'InPlace'
            if self.operand1.may_be_none() or self.operand2.may_be_none():
                func += 'Safe'
            return func
        return super(AddNode, self).py_operation_function(code)

class SubNode(NumBinopNode):

    def compute_c_result_type(self, type1, type2):
        if False:
            for i in range(10):
                print('nop')
        if (type1.is_ptr or type1.is_array) and (type2.is_int or type2.is_enum):
            return type1
        elif (type1.is_ptr or type1.is_array) and (type2.is_ptr or type2.is_array):
            return PyrexTypes.c_ptrdiff_t_type
        else:
            return NumBinopNode.compute_c_result_type(self, type1, type2)

class MulNode(NumBinopNode):
    is_sequence_mul = False

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        self.operand1 = self.operand1.analyse_types(env)
        self.operand2 = self.operand2.analyse_types(env)
        self.is_sequence_mul = self.calculate_is_sequence_mul()
        if self.is_sequence_mul:
            operand1 = self.operand1
            operand2 = self.operand2
            if operand1.is_sequence_constructor and operand1.mult_factor is None:
                return self.analyse_sequence_mul(env, operand1, operand2)
            elif operand2.is_sequence_constructor and operand2.mult_factor is None:
                return self.analyse_sequence_mul(env, operand2, operand1)
        self.analyse_operation(env)
        return self

    @staticmethod
    def is_builtin_seqmul_type(type):
        if False:
            while True:
                i = 10
        return type.is_builtin_type and type in builtin_sequence_types and (type is not memoryview_type)

    def calculate_is_sequence_mul(self):
        if False:
            while True:
                i = 10
        type1 = self.operand1.type
        type2 = self.operand2.type
        if type1 is long_type or type1.is_int:
            (type1, type2) = (type2, type1)
        if type2 is long_type or type2.is_int:
            if type1.is_string or type1.is_ctuple:
                return True
            if self.is_builtin_seqmul_type(type1):
                return True
        return False

    def analyse_sequence_mul(self, env, seq, mult):
        if False:
            while True:
                i = 10
        assert seq.mult_factor is None
        seq = seq.coerce_to_pyobject(env)
        seq.mult_factor = mult
        return seq.analyse_types(env)

    def coerce_operands_to_pyobjects(self, env):
        if False:
            while True:
                i = 10
        if self.is_sequence_mul:
            if self.operand1.type.is_ctuple:
                self.operand1 = self.operand1.coerce_to_pyobject(env)
            elif self.operand2.type.is_ctuple:
                self.operand2 = self.operand2.coerce_to_pyobject(env)
            return
        super(MulNode, self).coerce_operands_to_pyobjects(env)

    def is_py_operation_types(self, type1, type2):
        if False:
            for i in range(10):
                print('nop')
        return self.is_sequence_mul or super(MulNode, self).is_py_operation_types(type1, type2)

    def py_operation_function(self, code):
        if False:
            for i in range(10):
                print('nop')
        if self.is_sequence_mul:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PySequenceMultiply', 'ObjectHandling.c'))
            return '__Pyx_PySequence_Multiply' if self.operand1.type.is_pyobject else '__Pyx_PySequence_Multiply_Left'
        return super(MulNode, self).py_operation_function(code)

    def infer_builtin_types_operation(self, type1, type2):
        if False:
            while True:
                i = 10
        if type1.is_builtin_type and type2.is_builtin_type:
            if self.is_builtin_seqmul_type(type1):
                return type1
            if self.is_builtin_seqmul_type(type2):
                return type2
        if type1.is_int:
            return type2
        if type2.is_int:
            return type1
        return None

class MatMultNode(NumBinopNode):

    def is_py_operation_types(self, type1, type2):
        if False:
            for i in range(10):
                print('nop')
        return True

    def generate_evaluation_code(self, code):
        if False:
            print('Hello World!')
        code.globalstate.use_utility_code(UtilityCode.load_cached('MatrixMultiply', 'ObjectHandling.c'))
        super(MatMultNode, self).generate_evaluation_code(code)

class DivNode(NumBinopNode):
    cdivision = None
    truedivision = None
    ctruedivision = False
    cdivision_warnings = False
    zerodivision_check = None

    def find_compile_time_binary_operator(self, op1, op2):
        if False:
            for i in range(10):
                print('nop')
        func = compile_time_binary_operators[self.operator]
        if self.operator == '/' and self.truedivision is None:
            if isinstance(op1, _py_int_types) and isinstance(op2, _py_int_types):
                func = compile_time_binary_operators['//']
        return func

    def calculate_constant_result(self):
        if False:
            print('Hello World!')
        op1 = self.operand1.constant_result
        op2 = self.operand2.constant_result
        func = self.find_compile_time_binary_operator(op1, op2)
        self.constant_result = func(self.operand1.constant_result, self.operand2.constant_result)

    def compile_time_value(self, denv):
        if False:
            print('Hello World!')
        operand1 = self.operand1.compile_time_value(denv)
        operand2 = self.operand2.compile_time_value(denv)
        try:
            func = self.find_compile_time_binary_operator(operand1, operand2)
            return func(operand1, operand2)
        except Exception as e:
            self.compile_time_value_error(e)

    def _check_truedivision(self, env):
        if False:
            i = 10
            return i + 15
        if self.cdivision or env.directives['cdivision']:
            self.ctruedivision = False
        else:
            self.ctruedivision = self.truedivision

    def infer_type(self, env):
        if False:
            i = 10
            return i + 15
        self._check_truedivision(env)
        return self.result_type(self.operand1.infer_type(env), self.operand2.infer_type(env), env)

    def analyse_operation(self, env):
        if False:
            i = 10
            return i + 15
        self._check_truedivision(env)
        NumBinopNode.analyse_operation(self, env)
        if self.is_cpp_operation():
            self.cdivision = True
        if not self.type.is_pyobject:
            self.zerodivision_check = self.cdivision is None and (not env.directives['cdivision']) and (not self.operand2.has_constant_result() or self.operand2.constant_result == 0)
            if self.zerodivision_check or env.directives['cdivision_warnings']:
                self.operand1 = self.operand1.coerce_to_simple(env)
                self.operand2 = self.operand2.coerce_to_simple(env)

    def compute_c_result_type(self, type1, type2):
        if False:
            i = 10
            return i + 15
        if self.operator == '/' and self.ctruedivision and (not type1.is_cpp_class) and (not type2.is_cpp_class):
            if not type1.is_float and (not type2.is_float):
                widest_type = PyrexTypes.widest_numeric_type(type1, PyrexTypes.c_double_type)
                widest_type = PyrexTypes.widest_numeric_type(type2, widest_type)
                return widest_type
        return NumBinopNode.compute_c_result_type(self, type1, type2)

    def zero_division_message(self):
        if False:
            return 10
        if self.type.is_int:
            return 'integer division or modulo by zero'
        else:
            return 'float division'

    def generate_evaluation_code(self, code):
        if False:
            return 10
        if not self.type.is_pyobject and (not self.type.is_complex):
            if self.cdivision is None:
                self.cdivision = code.globalstate.directives['cdivision'] or self.type.is_float or ((self.type.is_numeric or self.type.is_enum) and (not self.type.signed))
            if not self.cdivision:
                code.globalstate.use_utility_code(UtilityCode.load_cached('DivInt', 'CMath.c').specialize(self.type))
        NumBinopNode.generate_evaluation_code(self, code)
        self.generate_div_warning_code(code)

    def generate_div_warning_code(self, code):
        if False:
            print('Hello World!')
        in_nogil = self.in_nogil_context
        if not self.type.is_pyobject:
            if self.zerodivision_check:
                if not self.infix:
                    zero_test = '%s(%s)' % (self.type.unary_op('zero'), self.operand2.result())
                else:
                    zero_test = '%s == 0' % self.operand2.result()
                code.putln('if (unlikely(%s)) {' % zero_test)
                if in_nogil:
                    code.put_ensure_gil()
                code.putln('PyErr_SetString(PyExc_ZeroDivisionError, "%s");' % self.zero_division_message())
                if in_nogil:
                    code.put_release_ensured_gil()
                code.putln(code.error_goto(self.pos))
                code.putln('}')
                if self.type.is_int and self.type.signed and (self.operator != '%'):
                    code.globalstate.use_utility_code(UtilityCode.load_cached('UnaryNegOverflows', 'Overflow.c'))
                    if self.operand2.type.signed == 2:
                        minus1_check = 'unlikely(%s == -1)' % self.operand2.result()
                    else:
                        type_of_op2 = self.operand2.type.empty_declaration_code()
                        minus1_check = '(!(((%s)-1) > 0)) && unlikely(%s == (%s)-1)' % (type_of_op2, self.operand2.result(), type_of_op2)
                    code.putln('else if (sizeof(%s) == sizeof(long) && %s  && unlikely(__Pyx_UNARY_NEG_WOULD_OVERFLOW(%s))) {' % (self.type.empty_declaration_code(), minus1_check, self.operand1.result()))
                    if in_nogil:
                        code.put_ensure_gil()
                    code.putln('PyErr_SetString(PyExc_OverflowError, "value too large to perform division");')
                    if in_nogil:
                        code.put_release_ensured_gil()
                    code.putln(code.error_goto(self.pos))
                    code.putln('}')
            if code.globalstate.directives['cdivision_warnings'] and self.operator != '/':
                code.globalstate.use_utility_code(UtilityCode.load_cached('CDivisionWarning', 'CMath.c'))
                code.putln('if (unlikely((%s < 0) ^ (%s < 0))) {' % (self.operand1.result(), self.operand2.result()))
                warning_code = '__Pyx_cdivision_warning(%(FILENAME)s, %(LINENO)s)' % {'FILENAME': Naming.filename_cname, 'LINENO': Naming.lineno_cname}
                if in_nogil:
                    result_code = 'result'
                    code.putln('int %s;' % result_code)
                    code.put_ensure_gil()
                    code.putln(code.set_error_info(self.pos, used=True))
                    code.putln('%s = %s;' % (result_code, warning_code))
                    code.put_release_ensured_gil()
                else:
                    result_code = warning_code
                    code.putln(code.set_error_info(self.pos, used=True))
                code.put('if (unlikely(%s)) ' % result_code)
                code.put_goto(code.error_label)
                code.putln('}')

    def calculate_result_code(self):
        if False:
            print('Hello World!')
        if self.type.is_complex or self.is_cpp_operation():
            return NumBinopNode.calculate_result_code(self)
        elif self.type.is_float and self.operator == '//':
            return 'floor(%s / %s)' % (self.operand1.result(), self.operand2.result())
        elif self.truedivision or self.cdivision:
            op1 = self.operand1.result()
            op2 = self.operand2.result()
            if self.truedivision:
                if self.type != self.operand1.type:
                    op1 = self.type.cast_code(op1)
                if self.type != self.operand2.type:
                    op2 = self.type.cast_code(op2)
            return '(%s / %s)' % (op1, op2)
        else:
            return '__Pyx_div_%s(%s, %s)' % (self.type.specialization_name(), self.operand1.result(), self.operand2.result())
_find_formatting_types = re.compile(b'%(?:%|(?:\\([^)]+\\))?[-+#,0-9 ]*([a-z]))').findall
_safe_bytes_formats = frozenset({b'd', b'i', b'o', b'u', b'x', b'X', b'e', b'E', b'f', b'F', b'g', b'G', b'c', b'b', b'a'})

class ModNode(DivNode):

    def is_py_operation_types(self, type1, type2):
        if False:
            i = 10
            return i + 15
        return type1.is_string or type2.is_string or NumBinopNode.is_py_operation_types(self, type1, type2)

    def infer_builtin_types_operation(self, type1, type2):
        if False:
            print('Hello World!')
        if type1 is unicode_type:
            if type2.is_builtin_type or not self.operand1.may_be_none():
                return type1
        elif type1 in (bytes_type, str_type, basestring_type):
            if type2 is unicode_type:
                return type2
            elif type2.is_numeric:
                return type1
            elif self.operand1.is_string_literal:
                if type1 is str_type or type1 is bytes_type:
                    if set(_find_formatting_types(self.operand1.value)) <= _safe_bytes_formats:
                        return type1
                return basestring_type
            elif type1 is bytes_type and (not type2.is_builtin_type):
                return None
            else:
                return basestring_type
        return None

    def zero_division_message(self):
        if False:
            i = 10
            return i + 15
        if self.type.is_int:
            return 'integer division or modulo by zero'
        else:
            return 'float divmod()'

    def analyse_operation(self, env):
        if False:
            for i in range(10):
                print('nop')
        DivNode.analyse_operation(self, env)
        if not self.type.is_pyobject:
            if self.cdivision is None:
                self.cdivision = env.directives['cdivision'] or not self.type.signed
            if not self.cdivision and (not self.type.is_int) and (not self.type.is_float):
                error(self.pos, "mod operator not supported for type '%s'" % self.type)

    def generate_evaluation_code(self, code):
        if False:
            print('Hello World!')
        if not self.type.is_pyobject and (not self.cdivision):
            if self.type.is_int:
                code.globalstate.use_utility_code(UtilityCode.load_cached('ModInt', 'CMath.c').specialize(self.type))
            else:
                code.globalstate.use_utility_code(UtilityCode.load_cached('ModFloat', 'CMath.c').specialize(self.type, math_h_modifier=self.type.math_h_modifier))
        NumBinopNode.generate_evaluation_code(self, code)
        self.generate_div_warning_code(code)

    def calculate_result_code(self):
        if False:
            i = 10
            return i + 15
        if self.cdivision:
            if self.type.is_float:
                return 'fmod%s(%s, %s)' % (self.type.math_h_modifier, self.operand1.result(), self.operand2.result())
            else:
                return '(%s %% %s)' % (self.operand1.result(), self.operand2.result())
        else:
            return '__Pyx_mod_%s(%s, %s)' % (self.type.specialization_name(), self.operand1.result(), self.operand2.result())

    def py_operation_function(self, code):
        if False:
            while True:
                i = 10
        (type1, type2) = (self.operand1.type, self.operand2.type)
        if type1 is unicode_type:
            if self.operand1.may_be_none() or (type2.is_extension_type and type2.subtype_of(type1) or (type2 is py_object_type and (not isinstance(self.operand2, CoerceToPyTypeNode)))):
                return '__Pyx_PyUnicode_FormatSafe'
            else:
                return 'PyUnicode_Format'
        elif type1 is str_type:
            if self.operand1.may_be_none() or (type2.is_extension_type and type2.subtype_of(type1) or (type2 is py_object_type and (not isinstance(self.operand2, CoerceToPyTypeNode)))):
                return '__Pyx_PyString_FormatSafe'
            else:
                return '__Pyx_PyString_Format'
        return super(ModNode, self).py_operation_function(code)

class PowNode(NumBinopNode):
    is_cpow = None
    type_was_inferred = False

    def _check_cpow(self, env):
        if False:
            for i in range(10):
                print('nop')
        if self.is_cpow is not None:
            return
        self.is_cpow = env.directives['cpow']

    def infer_type(self, env):
        if False:
            i = 10
            return i + 15
        self._check_cpow(env)
        return super(PowNode, self).infer_type(env)

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        self._check_cpow(env)
        return super(PowNode, self).analyse_types(env)

    def analyse_c_operation(self, env):
        if False:
            return 10
        NumBinopNode.analyse_c_operation(self, env)
        if self.type.is_complex:
            if self.type.real_type.is_float:
                self.operand1 = self.operand1.coerce_to(self.type, env)
                self.operand2 = self.operand2.coerce_to(self.type, env)
                self.pow_func = self.type.binary_op('**')
            else:
                error(self.pos, 'complex int powers not supported')
                self.pow_func = '<error>'
        elif self.type.is_float:
            self.pow_func = 'pow' + self.type.math_h_modifier
        elif self.type.is_int:
            self.pow_func = '__Pyx_pow_%s' % self.type.empty_declaration_code().replace(' ', '_')
            env.use_utility_code(UtilityCode.load_cached('IntPow', 'CMath.c').specialize(func_name=self.pow_func, type=self.type.empty_declaration_code(), signed=self.type.signed and 1 or 0))
        elif not self.type.is_error:
            error(self.pos, 'got unexpected types for C power operator: %s, %s' % (self.operand1.type, self.operand2.type))

    def compute_c_result_type(self, type1, type2):
        if False:
            while True:
                i = 10
        from numbers import Real
        c_result_type = None
        op1_is_definitely_positive = self.operand1.has_constant_result() and self.operand1.constant_result >= 0 or (type1.is_int and type1.signed == 0)
        type2_is_int = type2.is_int or (self.operand2.has_constant_result() and isinstance(self.operand2.constant_result, Real) and (int(self.operand2.constant_result) == self.operand2.constant_result))
        needs_widening = False
        if self.is_cpow:
            c_result_type = super(PowNode, self).compute_c_result_type(type1, type2)
            if not self.operand2.has_constant_result():
                needs_widening = isinstance(self.operand2.constant_result, _py_int_types) and self.operand2.constant_result < 0
        elif op1_is_definitely_positive or type2_is_int:
            c_result_type = super(PowNode, self).compute_c_result_type(type1, type2)
            if not self.operand2.has_constant_result():
                needs_widening = type2.is_int and type2.signed
                if needs_widening:
                    self.type_was_inferred = True
            else:
                needs_widening = isinstance(self.operand2.constant_result, _py_int_types) and self.operand2.constant_result < 0
        elif self.c_types_okay(type1, type2):
            c_result_type = PyrexTypes.soft_complex_type
            self.type_was_inferred = True
        if needs_widening:
            c_result_type = PyrexTypes.widest_numeric_type(c_result_type, PyrexTypes.c_double_type)
        return c_result_type

    def calculate_result_code(self):
        if False:
            print('Hello World!')

        def typecast(operand):
            if False:
                return 10
            if self.type == operand.type:
                return operand.result()
            else:
                return self.type.cast_code(operand.result())
        return '%s(%s, %s)' % (self.pow_func, typecast(self.operand1), typecast(self.operand2))

    def py_operation_function(self, code):
        if False:
            return 10
        if self.type.is_pyobject and self.operand1.constant_result == 2 and isinstance(self.operand1.constant_result, _py_int_types) and (self.operand2.type is py_object_type):
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyNumberPow2', 'Optimize.c'))
            if self.inplace:
                return '__Pyx_PyNumber_InPlacePowerOf2'
            else:
                return '__Pyx_PyNumber_PowerOf2'
        return super(PowNode, self).py_operation_function(code)

    def coerce_to(self, dst_type, env):
        if False:
            for i in range(10):
                print('nop')
        if dst_type == self.type:
            return self
        if self.is_cpow is None and self.type_was_inferred and (dst_type.is_float or dst_type.is_int):
            if self.type is PyrexTypes.soft_complex_type:

                def check_types(operand, recurse=True):
                    if False:
                        i = 10
                        return i + 15
                    if operand.type.is_float or operand.type.is_int:
                        return (True, operand)
                    if recurse and isinstance(operand, CoerceToComplexNode):
                        return (check_types(operand.arg, recurse=False), operand.arg)
                    return (False, None)
                msg_detail = 'a non-complex C numeric type'
            elif dst_type.is_int:

                def check_types(operand):
                    if False:
                        while True:
                            i = 10
                    if operand.type.is_int:
                        return (True, operand)
                    else:
                        return (False, None)
                msg_detail = 'an integer C numeric type'
            else:

                def check_types(operand):
                    if False:
                        while True:
                            i = 10
                    return (False, None)
            (check_op1, op1) = check_types(self.operand1)
            (check_op2, op2) = check_types(self.operand2)
            if check_op1 and check_op2:
                warning(self.pos, "Treating '**' as if 'cython.cpow(True)' since it is directly assigned to a %s. This is likely to be fragile and we recommend setting 'cython.cpow' explicitly." % msg_detail)
                self.is_cpow = True
                self.operand1 = op1
                self.operand2 = op2
                result = self.analyse_types(env)
                if result.type != dst_type:
                    result = result.coerce_to(dst_type, env)
                return result
        return super(PowNode, self).coerce_to(dst_type, env)

class BoolBinopNode(ExprNode):
    """
    Short-circuiting boolean operation.

    Note that this node provides the same code generation method as
    BoolBinopResultNode to simplify expression nesting.

    operator  string                              "and"/"or"
    operand1  BoolBinopNode/BoolBinopResultNode   left operand
    operand2  BoolBinopNode/BoolBinopResultNode   right operand
    """
    subexprs = ['operand1', 'operand2']
    is_temp = True
    operator = None
    operand1 = None
    operand2 = None

    def infer_type(self, env):
        if False:
            i = 10
            return i + 15
        type1 = self.operand1.infer_type(env)
        type2 = self.operand2.infer_type(env)
        return PyrexTypes.independent_spanning_type(type1, type2)

    def may_be_none(self):
        if False:
            i = 10
            return i + 15
        if self.operator == 'or':
            return self.operand2.may_be_none()
        else:
            return self.operand1.may_be_none() or self.operand2.may_be_none()

    def calculate_constant_result(self):
        if False:
            return 10
        operand1 = self.operand1.constant_result
        operand2 = self.operand2.constant_result
        if self.operator == 'and':
            self.constant_result = operand1 and operand2
        else:
            self.constant_result = operand1 or operand2

    def compile_time_value(self, denv):
        if False:
            i = 10
            return i + 15
        operand1 = self.operand1.compile_time_value(denv)
        operand2 = self.operand2.compile_time_value(denv)
        if self.operator == 'and':
            return operand1 and operand2
        else:
            return operand1 or operand2

    def is_ephemeral(self):
        if False:
            while True:
                i = 10
        return self.operand1.is_ephemeral() or self.operand2.is_ephemeral()

    def analyse_types(self, env):
        if False:
            i = 10
            return i + 15
        operand1 = self.operand1.analyse_types(env)
        operand2 = self.operand2.analyse_types(env)
        self.type = PyrexTypes.independent_spanning_type(operand1.type, operand2.type)
        self.operand1 = self._wrap_operand(operand1, env)
        self.operand2 = self._wrap_operand(operand2, env)
        return self

    def _wrap_operand(self, operand, env):
        if False:
            print('Hello World!')
        if not isinstance(operand, (BoolBinopNode, BoolBinopResultNode)):
            operand = BoolBinopResultNode(operand, self.type, env)
        return operand

    def wrap_operands(self, env):
        if False:
            for i in range(10):
                print('nop')
        '\n        Must get called by transforms that want to create a correct BoolBinopNode\n        after the type analysis phase.\n        '
        self.operand1 = self._wrap_operand(self.operand1, env)
        self.operand2 = self._wrap_operand(self.operand2, env)

    def coerce_to_boolean(self, env):
        if False:
            while True:
                i = 10
        return self.coerce_to(PyrexTypes.c_bint_type, env)

    def coerce_to(self, dst_type, env):
        if False:
            i = 10
            return i + 15
        operand1 = self.operand1.coerce_to(dst_type, env)
        operand2 = self.operand2.coerce_to(dst_type, env)
        return BoolBinopNode.from_node(self, type=dst_type, operator=self.operator, operand1=operand1, operand2=operand2)

    def generate_bool_evaluation_code(self, code, final_result_temp, final_result_type, and_label, or_label, end_label, fall_through):
        if False:
            i = 10
            return i + 15
        code.mark_pos(self.pos)
        outer_labels = (and_label, or_label)
        if self.operator == 'and':
            my_label = and_label = code.new_label('next_and')
        else:
            my_label = or_label = code.new_label('next_or')
        self.operand1.generate_bool_evaluation_code(code, final_result_temp, final_result_type, and_label, or_label, end_label, my_label)
        (and_label, or_label) = outer_labels
        code.put_label(my_label)
        self.operand2.generate_bool_evaluation_code(code, final_result_temp, final_result_type, and_label, or_label, end_label, fall_through)

    def generate_evaluation_code(self, code):
        if False:
            while True:
                i = 10
        self.allocate_temp_result(code)
        result_type = PyrexTypes.py_object_type if self.type.is_pyobject else self.type
        or_label = and_label = None
        end_label = code.new_label('bool_binop_done')
        self.generate_bool_evaluation_code(code, self.result(), result_type, and_label, or_label, end_label, end_label)
        code.put_label(end_label)
    gil_message = 'Truth-testing Python object'

    def check_const(self):
        if False:
            return 10
        return self.operand1.check_const() and self.operand2.check_const()

    def generate_subexpr_disposal_code(self, code):
        if False:
            print('Hello World!')
        pass

    def free_subexpr_temps(self, code):
        if False:
            for i in range(10):
                print('nop')
        pass

    def generate_operand1_test(self, code):
        if False:
            return 10
        if self.type.is_pyobject:
            test_result = code.funcstate.allocate_temp(PyrexTypes.c_bint_type, manage_ref=False)
            code.putln('%s = __Pyx_PyObject_IsTrue(%s); %s' % (test_result, self.operand1.py_result(), code.error_goto_if_neg(test_result, self.pos)))
        else:
            test_result = self.operand1.result()
        return (test_result, self.type.is_pyobject)

class BoolBinopResultNode(ExprNode):
    """
    Intermediate result of a short-circuiting and/or expression.
    Tests the result for 'truthiness' and takes care of coercing the final result
    of the overall expression to the target type.

    Note that this node provides the same code generation method as
    BoolBinopNode to simplify expression nesting.

    arg     ExprNode    the argument to test
    value   ExprNode    the coerced result value node
    """
    subexprs = ['arg', 'value']
    is_temp = True
    arg = None
    value = None

    def __init__(self, arg, result_type, env):
        if False:
            for i in range(10):
                print('nop')
        arg = arg.coerce_to_simple(env)
        arg = ProxyNode(arg)
        super(BoolBinopResultNode, self).__init__(arg.pos, arg=arg, type=result_type, value=CloneNode(arg).coerce_to(result_type, env))

    def coerce_to_boolean(self, env):
        if False:
            i = 10
            return i + 15
        return self.coerce_to(PyrexTypes.c_bint_type, env)

    def coerce_to(self, dst_type, env):
        if False:
            while True:
                i = 10
        arg = self.arg.arg
        if dst_type is PyrexTypes.c_bint_type:
            arg = arg.coerce_to_boolean(env)
        return BoolBinopResultNode(arg, dst_type, env)

    def nogil_check(self, env):
        if False:
            while True:
                i = 10
        pass

    def generate_operand_test(self, code):
        if False:
            for i in range(10):
                print('nop')
        if self.arg.type.is_pyobject:
            test_result = code.funcstate.allocate_temp(PyrexTypes.c_bint_type, manage_ref=False)
            code.putln('%s = __Pyx_PyObject_IsTrue(%s); %s' % (test_result, self.arg.py_result(), code.error_goto_if_neg(test_result, self.pos)))
        else:
            test_result = self.arg.result()
        return (test_result, self.arg.type.is_pyobject)

    def generate_bool_evaluation_code(self, code, final_result_temp, final_result_type, and_label, or_label, end_label, fall_through):
        if False:
            i = 10
            return i + 15
        code.mark_pos(self.pos)
        self.arg.generate_evaluation_code(code)
        if and_label or or_label:
            (test_result, uses_temp) = self.generate_operand_test(code)
            if uses_temp and (and_label and or_label):
                self.arg.generate_disposal_code(code)
            sense = '!' if or_label else ''
            code.putln('if (%s%s) {' % (sense, test_result))
            if uses_temp:
                code.funcstate.release_temp(test_result)
            if not uses_temp or not (and_label and or_label):
                self.arg.generate_disposal_code(code)
            if or_label and or_label != fall_through:
                code.put_goto(or_label)
            if and_label:
                if or_label:
                    code.putln('} else {')
                    if not uses_temp:
                        self.arg.generate_disposal_code(code)
                if and_label != fall_through:
                    code.put_goto(and_label)
        if not and_label or not or_label:
            if and_label or or_label:
                code.putln('} else {')
            self.value.generate_evaluation_code(code)
            self.value.make_owned_reference(code)
            code.putln('%s = %s;' % (final_result_temp, self.value.result_as(final_result_type)))
            self.value.generate_post_assignment_code(code)
            self.arg.generate_disposal_code(code)
            self.value.free_temps(code)
            if end_label != fall_through:
                code.put_goto(end_label)
        if and_label or or_label:
            code.putln('}')
        self.arg.free_temps(code)

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        return self

class CondExprNode(ExprNode):
    true_val = None
    false_val = None
    is_temp = True
    subexprs = ['test', 'true_val', 'false_val']

    def type_dependencies(self, env):
        if False:
            i = 10
            return i + 15
        return self.true_val.type_dependencies(env) + self.false_val.type_dependencies(env)

    def infer_type(self, env):
        if False:
            i = 10
            return i + 15
        return PyrexTypes.independent_spanning_type(self.true_val.infer_type(env), self.false_val.infer_type(env))

    def calculate_constant_result(self):
        if False:
            for i in range(10):
                print('nop')
        if self.test.constant_result:
            self.constant_result = self.true_val.constant_result
        else:
            self.constant_result = self.false_val.constant_result

    def is_ephemeral(self):
        if False:
            for i in range(10):
                print('nop')
        return self.true_val.is_ephemeral() or self.false_val.is_ephemeral()

    def analyse_types(self, env):
        if False:
            return 10
        self.test = self.test.analyse_temp_boolean_expression(env)
        self.true_val = self.true_val.analyse_types(env)
        self.false_val = self.false_val.analyse_types(env)
        return self.analyse_result_type(env)

    def analyse_result_type(self, env):
        if False:
            while True:
                i = 10
        true_val_type = self.true_val.type
        false_val_type = self.false_val.type
        self.type = PyrexTypes.independent_spanning_type(true_val_type, false_val_type)
        if self.type.is_reference:
            self.type = PyrexTypes.CFakeReferenceType(self.type.ref_base_type)
        if self.type.is_pyobject:
            self.result_ctype = py_object_type
        elif self.true_val.is_ephemeral() or self.false_val.is_ephemeral():
            error(self.pos, 'Unsafe C derivative of temporary Python reference used in conditional expression')
        if true_val_type.is_pyobject or false_val_type.is_pyobject or self.type.is_pyobject:
            if true_val_type != self.type:
                self.true_val = self.true_val.coerce_to(self.type, env)
            if false_val_type != self.type:
                self.false_val = self.false_val.coerce_to(self.type, env)
        if self.type.is_error:
            self.type_error()
        return self

    def coerce_to_integer(self, env):
        if False:
            while True:
                i = 10
        if not self.true_val.type.is_int:
            self.true_val = self.true_val.coerce_to_integer(env)
        if not self.false_val.type.is_int:
            self.false_val = self.false_val.coerce_to_integer(env)
        self.result_ctype = None
        out = self.analyse_result_type(env)
        if not out.type.is_int:
            if out is self:
                out = super(CondExprNode, out).coerce_to_integer(env)
            else:
                out = out.coerce_to_integer(env)
        return out

    def coerce_to(self, dst_type, env):
        if False:
            i = 10
            return i + 15
        if self.true_val.type != dst_type:
            self.true_val = self.true_val.coerce_to(dst_type, env)
        if self.false_val.type != dst_type:
            self.false_val = self.false_val.coerce_to(dst_type, env)
        self.result_ctype = None
        out = self.analyse_result_type(env)
        if out.type != dst_type:
            if out is self:
                out = super(CondExprNode, out).coerce_to(dst_type, env)
            else:
                out = out.coerce_to(dst_type, env)
        return out

    def type_error(self):
        if False:
            i = 10
            return i + 15
        if not (self.true_val.type.is_error or self.false_val.type.is_error):
            error(self.pos, 'Incompatible types in conditional expression (%s; %s)' % (self.true_val.type, self.false_val.type))
        self.type = PyrexTypes.error_type

    def check_const(self):
        if False:
            print('Hello World!')
        return self.test.check_const() and self.true_val.check_const() and self.false_val.check_const()

    def generate_evaluation_code(self, code):
        if False:
            print('Hello World!')
        code.mark_pos(self.pos)
        self.allocate_temp_result(code)
        self.test.generate_evaluation_code(code)
        code.putln('if (%s) {' % self.test.result())
        self.eval_and_get(code, self.true_val)
        code.putln('} else {')
        self.eval_and_get(code, self.false_val)
        code.putln('}')
        self.test.generate_disposal_code(code)
        self.test.free_temps(code)

    def eval_and_get(self, code, expr):
        if False:
            i = 10
            return i + 15
        expr.generate_evaluation_code(code)
        if self.type.is_memoryviewslice:
            expr.make_owned_memoryviewslice(code)
        else:
            expr.make_owned_reference(code)
        code.putln('%s = %s;' % (self.result(), expr.result_as(self.ctype())))
        expr.generate_post_assignment_code(code)
        expr.free_temps(code)

    def generate_subexpr_disposal_code(self, code):
        if False:
            print('Hello World!')
        pass

    def free_subexpr_temps(self, code):
        if False:
            i = 10
            return i + 15
        pass
richcmp_constants = {'<': 'Py_LT', '<=': 'Py_LE', '==': 'Py_EQ', '!=': 'Py_NE', '<>': 'Py_NE', '>': 'Py_GT', '>=': 'Py_GE', 'in': 'Py_EQ', 'not_in': 'Py_NE'}

class CmpNode(object):
    special_bool_cmp_function = None
    special_bool_cmp_utility_code = None
    special_bool_extra_args = []

    def infer_type(self, env):
        if False:
            i = 10
            return i + 15
        return py_object_type

    def calculate_cascaded_constant_result(self, operand1_result):
        if False:
            while True:
                i = 10
        func = compile_time_binary_operators[self.operator]
        operand2_result = self.operand2.constant_result
        if isinstance(operand1_result, any_string_type) and isinstance(operand2_result, any_string_type) and (type(operand1_result) != type(operand2_result)):
            return
        if self.operator in ('in', 'not_in'):
            if isinstance(self.operand2, (ListNode, TupleNode, SetNode)):
                if not self.operand2.args:
                    self.constant_result = self.operator == 'not_in'
                    return
                elif isinstance(self.operand2, ListNode) and (not self.cascade):
                    self.operand2 = self.operand2.as_tuple()
            elif isinstance(self.operand2, DictNode):
                if not self.operand2.key_value_pairs:
                    self.constant_result = self.operator == 'not_in'
                    return
        self.constant_result = func(operand1_result, operand2_result)

    def cascaded_compile_time_value(self, operand1, denv):
        if False:
            print('Hello World!')
        func = get_compile_time_binop(self)
        operand2 = self.operand2.compile_time_value(denv)
        try:
            result = func(operand1, operand2)
        except Exception as e:
            self.compile_time_value_error(e)
            result = None
        if result:
            cascade = self.cascade
            if cascade:
                result = result and cascade.cascaded_compile_time_value(operand2, denv)
        return result

    def is_cpp_comparison(self):
        if False:
            print('Hello World!')
        return self.operand1.type.is_cpp_class or self.operand2.type.is_cpp_class

    def find_common_int_type(self, env, op, operand1, operand2):
        if False:
            return 10
        type1 = operand1.type
        type2 = operand2.type
        type1_can_be_int = False
        type2_can_be_int = False
        if operand1.is_string_literal and operand1.can_coerce_to_char_literal():
            type1_can_be_int = True
        if operand2.is_string_literal and operand2.can_coerce_to_char_literal():
            type2_can_be_int = True
        if type1.is_int:
            if type2_can_be_int:
                return type1
        elif type2.is_int:
            if type1_can_be_int:
                return type2
        elif type1_can_be_int:
            if type2_can_be_int:
                if Builtin.unicode_type in (type1, type2):
                    return PyrexTypes.c_py_ucs4_type
                else:
                    return PyrexTypes.c_uchar_type
        return None

    def find_common_type(self, env, op, operand1, common_type=None):
        if False:
            for i in range(10):
                print('nop')
        operand2 = self.operand2
        type1 = operand1.type
        type2 = operand2.type
        new_common_type = None
        if type1 == str_type and (type2.is_string or type2 in (bytes_type, unicode_type)) or (type2 == str_type and (type1.is_string or type1 in (bytes_type, unicode_type))):
            error(self.pos, 'Comparisons between bytes/unicode and str are not portable to Python 3')
            new_common_type = error_type
        elif type1.is_complex or type2.is_complex:
            if op not in ('==', '!=') and (type1.is_complex or type1.is_numeric) and (type2.is_complex or type2.is_numeric):
                error(self.pos, 'complex types are unordered')
                new_common_type = error_type
            elif type1.is_pyobject:
                new_common_type = Builtin.complex_type if type1.subtype_of(Builtin.complex_type) else py_object_type
            elif type2.is_pyobject:
                new_common_type = Builtin.complex_type if type2.subtype_of(Builtin.complex_type) else py_object_type
            else:
                new_common_type = PyrexTypes.widest_numeric_type(type1, type2)
        elif type1.is_numeric and type2.is_numeric:
            new_common_type = PyrexTypes.widest_numeric_type(type1, type2)
        elif common_type is None or not common_type.is_pyobject:
            new_common_type = self.find_common_int_type(env, op, operand1, operand2)
        if new_common_type is None:
            if type1.is_ctuple or type2.is_ctuple:
                new_common_type = py_object_type
            elif type1 == type2:
                new_common_type = type1
            elif type1.is_pyobject or type2.is_pyobject:
                if type2.is_numeric or type2.is_string:
                    if operand2.check_for_coercion_error(type1, env):
                        new_common_type = error_type
                    else:
                        new_common_type = py_object_type
                elif type1.is_numeric or type1.is_string:
                    if operand1.check_for_coercion_error(type2, env):
                        new_common_type = error_type
                    else:
                        new_common_type = py_object_type
                elif py_object_type.assignable_from(type1) and py_object_type.assignable_from(type2):
                    new_common_type = py_object_type
                else:
                    self.invalid_types_error(operand1, op, operand2)
                    new_common_type = error_type
            elif type1.assignable_from(type2):
                new_common_type = type1
            elif type2.assignable_from(type1):
                new_common_type = type2
            else:
                self.invalid_types_error(operand1, op, operand2)
                new_common_type = error_type
        if new_common_type.is_string and (isinstance(operand1, BytesNode) or isinstance(operand2, BytesNode)):
            new_common_type = bytes_type
        if common_type is None or new_common_type.is_error:
            common_type = new_common_type
        else:
            common_type = PyrexTypes.spanning_type(common_type, new_common_type)
        if self.cascade:
            common_type = self.cascade.find_common_type(env, self.operator, operand2, common_type)
        return common_type

    def invalid_types_error(self, operand1, op, operand2):
        if False:
            print('Hello World!')
        error(self.pos, "Invalid types for '%s' (%s, %s)" % (op, operand1.type, operand2.type))

    def is_python_comparison(self):
        if False:
            while True:
                i = 10
        return not self.is_ptr_contains() and (not self.is_c_string_contains()) and (self.has_python_operands() or (self.cascade and self.cascade.is_python_comparison()) or self.operator in ('in', 'not_in'))

    def coerce_operands_to(self, dst_type, env):
        if False:
            print('Hello World!')
        operand2 = self.operand2
        if operand2.type != dst_type:
            self.operand2 = operand2.coerce_to(dst_type, env)
        if self.cascade:
            self.cascade.coerce_operands_to(dst_type, env)

    def is_python_result(self):
        if False:
            while True:
                i = 10
        return self.has_python_operands() and self.special_bool_cmp_function is None and (self.operator not in ('is', 'is_not', 'in', 'not_in')) and (not self.is_c_string_contains()) and (not self.is_ptr_contains()) or (self.cascade and self.cascade.is_python_result())

    def is_c_string_contains(self):
        if False:
            return 10
        return self.operator in ('in', 'not_in') and (self.operand1.type.is_int and (self.operand2.type.is_string or self.operand2.type is bytes_type) or (self.operand1.type.is_unicode_char and self.operand2.type is unicode_type))

    def is_ptr_contains(self):
        if False:
            i = 10
            return i + 15
        if self.operator in ('in', 'not_in'):
            container_type = self.operand2.type
            return (container_type.is_ptr or container_type.is_array) and (not container_type.is_string)

    def find_special_bool_compare_function(self, env, operand1, result_is_bool=False):
        if False:
            i = 10
            return i + 15
        if self.operator in ('==', '!='):
            (type1, type2) = (operand1.type, self.operand2.type)
            if result_is_bool or (type1.is_builtin_type and type2.is_builtin_type):
                if type1 is Builtin.unicode_type or type2 is Builtin.unicode_type:
                    self.special_bool_cmp_utility_code = UtilityCode.load_cached('UnicodeEquals', 'StringTools.c')
                    self.special_bool_cmp_function = '__Pyx_PyUnicode_Equals'
                    return True
                elif type1 is Builtin.bytes_type or type2 is Builtin.bytes_type:
                    self.special_bool_cmp_utility_code = UtilityCode.load_cached('BytesEquals', 'StringTools.c')
                    self.special_bool_cmp_function = '__Pyx_PyBytes_Equals'
                    return True
                elif type1 is Builtin.basestring_type or type2 is Builtin.basestring_type:
                    self.special_bool_cmp_utility_code = UtilityCode.load_cached('UnicodeEquals', 'StringTools.c')
                    self.special_bool_cmp_function = '__Pyx_PyUnicode_Equals'
                    return True
                elif type1 is Builtin.str_type or type2 is Builtin.str_type:
                    self.special_bool_cmp_utility_code = UtilityCode.load_cached('StrEquals', 'StringTools.c')
                    self.special_bool_cmp_function = '__Pyx_PyString_Equals'
                    return True
                elif result_is_bool:
                    from .Optimize import optimise_numeric_binop
                    result = optimise_numeric_binop('Eq' if self.operator == '==' else 'Ne', self, PyrexTypes.c_bint_type, operand1, self.operand2)
                    if result:
                        (self.special_bool_cmp_function, self.special_bool_cmp_utility_code, self.special_bool_extra_args, _) = result
                        return True
        elif self.operator in ('in', 'not_in'):
            if self.operand2.type is Builtin.dict_type:
                self.operand2 = self.operand2.as_none_safe_node("'NoneType' object is not iterable")
                self.special_bool_cmp_utility_code = UtilityCode.load_cached('PyDictContains', 'ObjectHandling.c')
                self.special_bool_cmp_function = '__Pyx_PyDict_ContainsTF'
                return True
            elif self.operand2.type is Builtin.set_type:
                self.operand2 = self.operand2.as_none_safe_node("'NoneType' object is not iterable")
                self.special_bool_cmp_utility_code = UtilityCode.load_cached('PySetContains', 'ObjectHandling.c')
                self.special_bool_cmp_function = '__Pyx_PySet_ContainsTF'
                return True
            elif self.operand2.type is Builtin.unicode_type:
                self.operand2 = self.operand2.as_none_safe_node("'NoneType' object is not iterable")
                self.special_bool_cmp_utility_code = UtilityCode.load_cached('PyUnicodeContains', 'StringTools.c')
                self.special_bool_cmp_function = '__Pyx_PyUnicode_ContainsTF'
                return True
            else:
                if not self.operand2.type.is_pyobject:
                    self.operand2 = self.operand2.coerce_to_pyobject(env)
                self.special_bool_cmp_utility_code = UtilityCode.load_cached('PySequenceContains', 'ObjectHandling.c')
                self.special_bool_cmp_function = '__Pyx_PySequence_ContainsTF'
                return True
        return False

    def generate_operation_code(self, code, result_code, operand1, op, operand2):
        if False:
            for i in range(10):
                print('nop')
        if self.type.is_pyobject:
            error_clause = code.error_goto_if_null
            got_ref = '__Pyx_XGOTREF(%s); ' % result_code
            if self.special_bool_cmp_function:
                code.globalstate.use_utility_code(UtilityCode.load_cached('PyBoolOrNullFromLong', 'ObjectHandling.c'))
                coerce_result = '__Pyx_PyBoolOrNull_FromLong'
            else:
                coerce_result = '__Pyx_PyBool_FromLong'
        else:
            error_clause = code.error_goto_if_neg
            got_ref = ''
            coerce_result = ''
        if self.special_bool_cmp_function:
            if operand1.type.is_pyobject:
                result1 = operand1.py_result()
            else:
                result1 = operand1.result()
            if operand2.type.is_pyobject:
                result2 = operand2.py_result()
            else:
                result2 = operand2.result()
            special_bool_extra_args_result = ', '.join([extra_arg.result() for extra_arg in self.special_bool_extra_args])
            if self.special_bool_cmp_utility_code:
                code.globalstate.use_utility_code(self.special_bool_cmp_utility_code)
            code.putln('%s = %s(%s(%s, %s, %s)); %s%s' % (result_code, coerce_result, self.special_bool_cmp_function, result1, result2, special_bool_extra_args_result if self.special_bool_extra_args else richcmp_constants[op], got_ref, error_clause(result_code, self.pos)))
        elif operand1.type.is_pyobject and op not in ('is', 'is_not'):
            assert op not in ('in', 'not_in'), op
            assert self.type.is_pyobject or self.type is PyrexTypes.c_bint_type
            code.putln('%s = PyObject_RichCompare%s(%s, %s, %s); %s%s' % (result_code, '' if self.type.is_pyobject else 'Bool', operand1.py_result(), operand2.py_result(), richcmp_constants[op], got_ref, error_clause(result_code, self.pos)))
        elif operand1.type.is_complex:
            code.putln('%s = %s(%s%s(%s, %s));' % (result_code, coerce_result, op == '!=' and '!' or '', operand1.type.unary_op('eq'), operand1.result(), operand2.result()))
        else:
            type1 = operand1.type
            type2 = operand2.type
            if (type1.is_extension_type or type2.is_extension_type) and (not type1.same_as(type2)):
                common_type = py_object_type
            elif type1.is_numeric:
                common_type = PyrexTypes.widest_numeric_type(type1, type2)
            else:
                common_type = type1
            code1 = operand1.result_as(common_type)
            code2 = operand2.result_as(common_type)
            statement = '%s = %s(%s %s %s);' % (result_code, coerce_result, code1, self.c_operator(op), code2)
            if self.is_cpp_comparison() and self.exception_check == '+':
                translate_cpp_exception(code, self.pos, statement, result_code if self.type.is_pyobject else None, self.exception_value, self.in_nogil_context)
            else:
                code.putln(statement)

    def c_operator(self, op):
        if False:
            i = 10
            return i + 15
        if op == 'is':
            return '=='
        elif op == 'is_not':
            return '!='
        else:
            return op

class PrimaryCmpNode(ExprNode, CmpNode):
    child_attrs = ['operand1', 'operand2', 'coerced_operand2', 'cascade', 'special_bool_extra_args']
    cascade = None
    coerced_operand2 = None
    is_memslice_nonecheck = False

    def infer_type(self, env):
        if False:
            print('Hello World!')
        type1 = self.operand1.infer_type(env)
        type2 = self.operand2.infer_type(env)
        if is_pythran_expr(type1) or is_pythran_expr(type2):
            if is_pythran_supported_type(type1) and is_pythran_supported_type(type2):
                return PythranExpr(pythran_binop_type(self.operator, type1, type2))
        return py_object_type

    def type_dependencies(self, env):
        if False:
            return 10
        return ()

    def calculate_constant_result(self):
        if False:
            print('Hello World!')
        assert not self.cascade
        self.calculate_cascaded_constant_result(self.operand1.constant_result)

    def compile_time_value(self, denv):
        if False:
            print('Hello World!')
        operand1 = self.operand1.compile_time_value(denv)
        return self.cascaded_compile_time_value(operand1, denv)

    def unify_cascade_type(self):
        if False:
            i = 10
            return i + 15
        cdr = self.cascade
        while cdr:
            cdr.type = self.type
            cdr = cdr.cascade

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        self.operand1 = self.operand1.analyse_types(env)
        self.operand2 = self.operand2.analyse_types(env)
        if self.is_cpp_comparison():
            self.analyse_cpp_comparison(env)
            if self.cascade:
                error(self.pos, 'Cascading comparison not yet supported for cpp types.')
            return self
        type1 = self.operand1.type
        type2 = self.operand2.type
        if is_pythran_expr(type1) or is_pythran_expr(type2):
            if is_pythran_supported_type(type1) and is_pythran_supported_type(type2):
                self.type = PythranExpr(pythran_binop_type(self.operator, type1, type2))
                self.is_pycmp = False
                return self
        if self.analyse_memoryviewslice_comparison(env):
            return self
        if self.cascade:
            self.cascade = self.cascade.analyse_types(env)
        if self.operator in ('in', 'not_in'):
            if self.is_c_string_contains():
                self.is_pycmp = False
                common_type = None
                if self.cascade:
                    error(self.pos, "Cascading comparison not yet supported for 'int_val in string'.")
                    return self
                if self.operand2.type is unicode_type:
                    env.use_utility_code(UtilityCode.load_cached('PyUCS4InUnicode', 'StringTools.c'))
                else:
                    if self.operand1.type is PyrexTypes.c_uchar_type:
                        self.operand1 = self.operand1.coerce_to(PyrexTypes.c_char_type, env)
                    if self.operand2.type is not bytes_type:
                        self.operand2 = self.operand2.coerce_to(bytes_type, env)
                    env.use_utility_code(UtilityCode.load_cached('BytesContains', 'StringTools.c'))
                self.operand2 = self.operand2.as_none_safe_node("argument of type 'NoneType' is not iterable")
            elif self.is_ptr_contains():
                if self.cascade:
                    error(self.pos, "Cascading comparison not supported for 'val in sliced pointer'.")
                self.type = PyrexTypes.c_bint_type
                return self
            elif self.find_special_bool_compare_function(env, self.operand1):
                if not self.operand1.type.is_pyobject:
                    self.operand1 = self.operand1.coerce_to_pyobject(env)
                common_type = None
                self.is_pycmp = False
            else:
                common_type = py_object_type
                self.is_pycmp = True
        elif self.find_special_bool_compare_function(env, self.operand1):
            if not self.operand1.type.is_pyobject:
                self.operand1 = self.operand1.coerce_to_pyobject(env)
            common_type = None
            self.is_pycmp = False
        else:
            common_type = self.find_common_type(env, self.operator, self.operand1)
            self.is_pycmp = common_type.is_pyobject
        if common_type is not None and (not common_type.is_error):
            if self.operand1.type != common_type:
                self.operand1 = self.operand1.coerce_to(common_type, env)
            self.coerce_operands_to(common_type, env)
        if self.cascade:
            self.operand2 = self.operand2.coerce_to_simple(env)
            self.cascade.coerce_cascaded_operands_to_temp(env)
            operand2 = self.cascade.optimise_comparison(self.operand2, env)
            if operand2 is not self.operand2:
                self.coerced_operand2 = operand2
        if self.is_python_result():
            self.type = PyrexTypes.py_object_type
        else:
            self.type = PyrexTypes.c_bint_type
        self.unify_cascade_type()
        if self.is_pycmp or self.cascade or self.special_bool_cmp_function:
            self.is_temp = 1
        return self

    def analyse_cpp_comparison(self, env):
        if False:
            print('Hello World!')
        type1 = self.operand1.type
        type2 = self.operand2.type
        self.is_pycmp = False
        entry = env.lookup_operator(self.operator, [self.operand1, self.operand2])
        if entry is None:
            error(self.pos, "Invalid types for '%s' (%s, %s)" % (self.operator, type1, type2))
            self.type = PyrexTypes.error_type
            self.result_code = '<error>'
            return
        func_type = entry.type
        if func_type.is_ptr:
            func_type = func_type.base_type
        self.exception_check = func_type.exception_check
        self.exception_value = func_type.exception_value
        if self.exception_check == '+':
            self.is_temp = True
            if needs_cpp_exception_conversion(self):
                env.use_utility_code(UtilityCode.load_cached('CppExceptionConversion', 'CppSupport.cpp'))
        if len(func_type.args) == 1:
            self.operand2 = self.operand2.coerce_to(func_type.args[0].type, env)
        else:
            self.operand1 = self.operand1.coerce_to(func_type.args[0].type, env)
            self.operand2 = self.operand2.coerce_to(func_type.args[1].type, env)
        self.type = func_type.return_type

    def analyse_memoryviewslice_comparison(self, env):
        if False:
            return 10
        have_none = self.operand1.is_none or self.operand2.is_none
        have_slice = self.operand1.type.is_memoryviewslice or self.operand2.type.is_memoryviewslice
        ops = ('==', '!=', 'is', 'is_not')
        if have_slice and have_none and (self.operator in ops):
            self.is_pycmp = False
            self.type = PyrexTypes.c_bint_type
            self.is_memslice_nonecheck = True
            return True
        return False

    def coerce_to_boolean(self, env):
        if False:
            i = 10
            return i + 15
        if self.is_pycmp:
            if self.find_special_bool_compare_function(env, self.operand1, result_is_bool=True):
                self.is_pycmp = False
                self.type = PyrexTypes.c_bint_type
                self.is_temp = 1
                if self.cascade:
                    operand2 = self.cascade.optimise_comparison(self.operand2, env, result_is_bool=True)
                    if operand2 is not self.operand2:
                        self.coerced_operand2 = operand2
                self.unify_cascade_type()
                return self
        return ExprNode.coerce_to_boolean(self, env)

    def has_python_operands(self):
        if False:
            return 10
        return self.operand1.type.is_pyobject or self.operand2.type.is_pyobject

    def check_const(self):
        if False:
            i = 10
            return i + 15
        if self.cascade:
            self.not_const()
            return False
        else:
            return self.operand1.check_const() and self.operand2.check_const()

    def calculate_result_code(self):
        if False:
            return 10
        (operand1, operand2) = (self.operand1, self.operand2)
        if operand1.type.is_complex:
            if self.operator == '!=':
                negation = '!'
            else:
                negation = ''
            return '(%s%s(%s, %s))' % (negation, operand1.type.binary_op('=='), operand1.result(), operand2.result())
        elif self.is_c_string_contains():
            if operand2.type is unicode_type:
                method = '__Pyx_UnicodeContainsUCS4'
            else:
                method = '__Pyx_BytesContains'
            if self.operator == 'not_in':
                negation = '!'
            else:
                negation = ''
            return '(%s%s(%s, %s))' % (negation, method, operand2.result(), operand1.result())
        else:
            if is_pythran_expr(self.type):
                (result1, result2) = (operand1.pythran_result(), operand2.pythran_result())
            else:
                (result1, result2) = (operand1.result(), operand2.result())
                if self.is_memslice_nonecheck:
                    if operand1.type.is_memoryviewslice:
                        result1 = '((PyObject *) %s.memview)' % result1
                    else:
                        result2 = '((PyObject *) %s.memview)' % result2
            return '(%s %s %s)' % (result1, self.c_operator(self.operator), result2)

    def generate_evaluation_code(self, code):
        if False:
            while True:
                i = 10
        self.operand1.generate_evaluation_code(code)
        self.operand2.generate_evaluation_code(code)
        for extra_arg in self.special_bool_extra_args:
            extra_arg.generate_evaluation_code(code)
        if self.is_temp:
            self.allocate_temp_result(code)
            self.generate_operation_code(code, self.result(), self.operand1, self.operator, self.operand2)
            if self.cascade:
                self.cascade.generate_evaluation_code(code, self.result(), self.coerced_operand2 or self.operand2, needs_evaluation=self.coerced_operand2 is not None)
            self.operand1.generate_disposal_code(code)
            self.operand1.free_temps(code)
            self.operand2.generate_disposal_code(code)
            self.operand2.free_temps(code)

    def generate_subexpr_disposal_code(self, code):
        if False:
            return 10
        self.operand1.generate_disposal_code(code)
        self.operand2.generate_disposal_code(code)

    def free_subexpr_temps(self, code):
        if False:
            i = 10
            return i + 15
        self.operand1.free_temps(code)
        self.operand2.free_temps(code)

    def annotate(self, code):
        if False:
            for i in range(10):
                print('nop')
        self.operand1.annotate(code)
        self.operand2.annotate(code)
        if self.cascade:
            self.cascade.annotate(code)

class CascadedCmpNode(Node, CmpNode):
    child_attrs = ['operand2', 'coerced_operand2', 'cascade', 'special_bool_extra_args']
    cascade = None
    coerced_operand2 = None
    constant_result = constant_value_not_set

    def infer_type(self, env):
        if False:
            i = 10
            return i + 15
        return py_object_type

    def type_dependencies(self, env):
        if False:
            while True:
                i = 10
        return ()

    def has_constant_result(self):
        if False:
            return 10
        return self.constant_result is not constant_value_not_set and self.constant_result is not not_a_constant

    def analyse_types(self, env):
        if False:
            i = 10
            return i + 15
        self.operand2 = self.operand2.analyse_types(env)
        if self.cascade:
            self.cascade = self.cascade.analyse_types(env)
        return self

    def has_python_operands(self):
        if False:
            i = 10
            return i + 15
        return self.operand2.type.is_pyobject

    def is_cpp_comparison(self):
        if False:
            i = 10
            return i + 15
        return False

    def optimise_comparison(self, operand1, env, result_is_bool=False):
        if False:
            i = 10
            return i + 15
        if self.find_special_bool_compare_function(env, operand1, result_is_bool):
            self.is_pycmp = False
            self.type = PyrexTypes.c_bint_type
            if not operand1.type.is_pyobject:
                operand1 = operand1.coerce_to_pyobject(env)
        if self.cascade:
            operand2 = self.cascade.optimise_comparison(self.operand2, env, result_is_bool)
            if operand2 is not self.operand2:
                self.coerced_operand2 = operand2
        return operand1

    def coerce_operands_to_pyobjects(self, env):
        if False:
            return 10
        self.operand2 = self.operand2.coerce_to_pyobject(env)
        if self.operand2.type is dict_type and self.operator in ('in', 'not_in'):
            self.operand2 = self.operand2.as_none_safe_node("'NoneType' object is not iterable")
        if self.cascade:
            self.cascade.coerce_operands_to_pyobjects(env)

    def coerce_cascaded_operands_to_temp(self, env):
        if False:
            print('Hello World!')
        if self.cascade:
            self.operand2 = self.operand2.coerce_to_simple(env)
            self.cascade.coerce_cascaded_operands_to_temp(env)

    def generate_evaluation_code(self, code, result, operand1, needs_evaluation=False):
        if False:
            for i in range(10):
                print('nop')
        if self.type.is_pyobject:
            code.putln('if (__Pyx_PyObject_IsTrue(%s)) {' % result)
            code.put_decref(result, self.type)
        else:
            code.putln('if (%s) {' % result)
        if needs_evaluation:
            operand1.generate_evaluation_code(code)
        self.operand2.generate_evaluation_code(code)
        for extra_arg in self.special_bool_extra_args:
            extra_arg.generate_evaluation_code(code)
        self.generate_operation_code(code, result, operand1, self.operator, self.operand2)
        if self.cascade:
            self.cascade.generate_evaluation_code(code, result, self.coerced_operand2 or self.operand2, needs_evaluation=self.coerced_operand2 is not None)
        if needs_evaluation:
            operand1.generate_disposal_code(code)
            operand1.free_temps(code)
        self.operand2.generate_disposal_code(code)
        self.operand2.free_temps(code)
        code.putln('}')

    def annotate(self, code):
        if False:
            i = 10
            return i + 15
        self.operand2.annotate(code)
        if self.cascade:
            self.cascade.annotate(code)
binop_node_classes = {'or': BoolBinopNode, 'and': BoolBinopNode, '|': IntBinopNode, '^': IntBinopNode, '&': IntBinopNode, '<<': IntBinopNode, '>>': IntBinopNode, '+': AddNode, '-': SubNode, '*': MulNode, '@': MatMultNode, '/': DivNode, '//': DivNode, '%': ModNode, '**': PowNode}

def binop_node(pos, operator, operand1, operand2, inplace=False, **kwargs):
    if False:
        print('Hello World!')
    return binop_node_classes[operator](pos, operator=operator, operand1=operand1, operand2=operand2, inplace=inplace, **kwargs)

class CoercionNode(ExprNode):
    subexprs = ['arg']
    constant_result = not_a_constant

    def __init__(self, arg):
        if False:
            print('Hello World!')
        super(CoercionNode, self).__init__(arg.pos)
        self.arg = arg
        if debug_coercion:
            print('%s Coercing %s' % (self, self.arg))

    def calculate_constant_result(self):
        if False:
            while True:
                i = 10
        pass

    def annotate(self, code):
        if False:
            return 10
        self.arg.annotate(code)
        if self.arg.type != self.type:
            (file, line, col) = self.pos
            code.annotate((file, line, col - 1), AnnotationItem(style='coerce', tag='coerce', text='[%s] to [%s]' % (self.arg.type, self.type)))

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        return self

class CoerceToMemViewSliceNode(CoercionNode):
    """
    Coerce an object to a memoryview slice. This holds a new reference in
    a managed temp.
    """

    def __init__(self, arg, dst_type, env):
        if False:
            print('Hello World!')
        assert dst_type.is_memoryviewslice
        assert not arg.type.is_memoryviewslice
        CoercionNode.__init__(self, arg)
        self.type = dst_type
        self.is_temp = 1
        self.use_managed_ref = True
        self.arg = arg
        self.type.create_from_py_utility_code(env)

    def generate_result_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        code.putln(self.type.from_py_call_code(self.arg.py_result(), self.result(), self.pos, code))

class CastNode(CoercionNode):

    def __init__(self, arg, new_type):
        if False:
            while True:
                i = 10
        CoercionNode.__init__(self, arg)
        self.type = new_type

    def may_be_none(self):
        if False:
            print('Hello World!')
        return self.arg.may_be_none()

    def calculate_result_code(self):
        if False:
            print('Hello World!')
        return self.arg.result_as(self.type)

    def generate_result_code(self, code):
        if False:
            i = 10
            return i + 15
        self.arg.generate_result_code(code)

class PyTypeTestNode(CoercionNode):
    exact_builtin_type = True

    def __init__(self, arg, dst_type, env, notnone=False):
        if False:
            return 10
        assert dst_type.is_extension_type or dst_type.is_builtin_type, 'PyTypeTest for %s against non extension type %s' % (arg.type, dst_type)
        CoercionNode.__init__(self, arg)
        self.type = dst_type
        self.result_ctype = arg.ctype()
        self.notnone = notnone
    nogil_check = Node.gil_error
    gil_message = 'Python type test'

    def analyse_types(self, env):
        if False:
            print('Hello World!')
        return self

    def may_be_none(self):
        if False:
            print('Hello World!')
        if self.notnone:
            return False
        return self.arg.may_be_none()

    def is_simple(self):
        if False:
            i = 10
            return i + 15
        return self.arg.is_simple()

    def result_in_temp(self):
        if False:
            i = 10
            return i + 15
        return self.arg.result_in_temp()

    def is_ephemeral(self):
        if False:
            for i in range(10):
                print('nop')
        return self.arg.is_ephemeral()

    def nonlocally_immutable(self):
        if False:
            print('Hello World!')
        return self.arg.nonlocally_immutable()

    def reanalyse(self):
        if False:
            while True:
                i = 10
        if self.type != self.arg.type or not self.arg.is_temp:
            return self
        if not self.type.typeobj_is_available():
            return self
        if self.arg.may_be_none() and self.notnone:
            return self.arg.as_none_safe_node('Cannot convert NoneType to %.200s' % self.type.name)
        return self.arg

    def calculate_constant_result(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def calculate_result_code(self):
        if False:
            while True:
                i = 10
        return self.arg.result()

    def generate_result_code(self, code):
        if False:
            print('Hello World!')
        if self.type.typeobj_is_available():
            if self.type.is_builtin_type:
                type_test = self.type.type_test_code(self.arg.py_result(), self.notnone, exact=self.exact_builtin_type)
                code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseUnexpectedTypeError', 'ObjectHandling.c'))
            else:
                type_test = self.type.type_test_code(self.arg.py_result(), self.notnone)
                code.globalstate.use_utility_code(UtilityCode.load_cached('ExtTypeTest', 'ObjectHandling.c'))
            code.putln('if (!(%s)) %s' % (type_test, code.error_goto(self.pos)))
        else:
            error(self.pos, 'Cannot test type of extern C class without type object name specification')

    def generate_post_assignment_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        self.arg.generate_post_assignment_code(code)

    def allocate_temp_result(self, code):
        if False:
            while True:
                i = 10
        pass

    def release_temp_result(self, code):
        if False:
            print('Hello World!')
        pass

    def free_temps(self, code):
        if False:
            return 10
        self.arg.free_temps(code)

    def free_subexpr_temps(self, code):
        if False:
            i = 10
            return i + 15
        self.arg.free_subexpr_temps(code)

class NoneCheckNode(CoercionNode):
    is_nonecheck = True

    def __init__(self, arg, exception_type_cname, exception_message, exception_format_args=()):
        if False:
            for i in range(10):
                print('nop')
        CoercionNode.__init__(self, arg)
        self.type = arg.type
        self.result_ctype = arg.ctype()
        self.exception_type_cname = exception_type_cname
        self.exception_message = exception_message
        self.exception_format_args = tuple(exception_format_args or ())
    nogil_check = None

    def analyse_types(self, env):
        if False:
            return 10
        return self

    def may_be_none(self):
        if False:
            print('Hello World!')
        return False

    def is_simple(self):
        if False:
            while True:
                i = 10
        return self.arg.is_simple()

    def result_in_temp(self):
        if False:
            return 10
        return self.arg.result_in_temp()

    def nonlocally_immutable(self):
        if False:
            i = 10
            return i + 15
        return self.arg.nonlocally_immutable()

    def calculate_result_code(self):
        if False:
            print('Hello World!')
        return self.arg.result()

    def condition(self):
        if False:
            i = 10
            return i + 15
        if self.type.is_pyobject:
            return self.arg.py_result()
        elif self.type.is_memoryviewslice:
            return '((PyObject *) %s.memview)' % self.arg.result()
        else:
            raise Exception('unsupported type')

    @classmethod
    def generate(cls, arg, code, exception_message, exception_type_cname='PyExc_TypeError', exception_format_args=(), in_nogil_context=False):
        if False:
            print('Hello World!')
        node = cls(arg, exception_type_cname, exception_message, exception_format_args)
        node.in_nogil_context = in_nogil_context
        node.put_nonecheck(code)

    @classmethod
    def generate_if_needed(cls, arg, code, exception_message, exception_type_cname='PyExc_TypeError', exception_format_args=(), in_nogil_context=False):
        if False:
            i = 10
            return i + 15
        if arg.may_be_none():
            cls.generate(arg, code, exception_message, exception_type_cname, exception_format_args, in_nogil_context)

    def put_nonecheck(self, code):
        if False:
            for i in range(10):
                print('nop')
        code.putln('if (unlikely(%s == Py_None)) {' % self.condition())
        if self.in_nogil_context:
            code.put_ensure_gil()
        escape = StringEncoding.escape_byte_string
        if self.exception_format_args:
            code.putln('PyErr_Format(%s, "%s", %s);' % (self.exception_type_cname, StringEncoding.escape_byte_string(self.exception_message.encode('UTF-8')), ', '.join(['"%s"' % escape(str(arg).encode('UTF-8')) for arg in self.exception_format_args])))
        else:
            code.putln('PyErr_SetString(%s, "%s");' % (self.exception_type_cname, escape(self.exception_message.encode('UTF-8'))))
        if self.in_nogil_context:
            code.put_release_ensured_gil()
        code.putln(code.error_goto(self.pos))
        code.putln('}')

    def generate_result_code(self, code):
        if False:
            i = 10
            return i + 15
        self.put_nonecheck(code)

    def generate_post_assignment_code(self, code):
        if False:
            return 10
        self.arg.generate_post_assignment_code(code)

    def free_temps(self, code):
        if False:
            return 10
        self.arg.free_temps(code)

class CoerceToPyTypeNode(CoercionNode):
    type = py_object_type
    target_type = py_object_type
    is_temp = 1

    def __init__(self, arg, env, type=py_object_type):
        if False:
            for i in range(10):
                print('nop')
        if not arg.type.create_to_py_utility_code(env):
            error(arg.pos, "Cannot convert '%s' to Python object" % arg.type)
        elif arg.type.is_complex:
            arg = arg.coerce_to_simple(env)
        CoercionNode.__init__(self, arg)
        if type is py_object_type:
            if arg.type.is_string or arg.type.is_cpp_string:
                self.type = default_str_type(env)
            elif arg.type.is_pyunicode_ptr or arg.type.is_unicode_char:
                self.type = unicode_type
            elif arg.type.is_complex:
                self.type = Builtin.complex_type
            self.target_type = self.type
        elif arg.type.is_string or arg.type.is_cpp_string:
            if type not in (bytes_type, bytearray_type) and (not env.directives['c_string_encoding']):
                error(arg.pos, "default encoding required for conversion from '%s' to '%s'" % (arg.type, type))
            self.type = self.target_type = type
        else:
            self.target_type = type
    gil_message = 'Converting to Python object'

    def may_be_none(self):
        if False:
            while True:
                i = 10
        return False

    def coerce_to_boolean(self, env):
        if False:
            i = 10
            return i + 15
        arg_type = self.arg.type
        if arg_type == PyrexTypes.c_bint_type or (arg_type.is_pyobject and arg_type.name == 'bool'):
            return self.arg.coerce_to_temp(env)
        else:
            return CoerceToBooleanNode(self, env)

    def coerce_to_integer(self, env):
        if False:
            print('Hello World!')
        if self.arg.type.is_int:
            return self.arg
        else:
            return self.arg.coerce_to(PyrexTypes.c_long_type, env)

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        return self

    def generate_result_code(self, code):
        if False:
            print('Hello World!')
        code.putln('%s; %s' % (self.arg.type.to_py_call_code(self.arg.result(), self.result(), self.target_type), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

class CoerceIntToBytesNode(CoerceToPyTypeNode):
    is_temp = 1

    def __init__(self, arg, env):
        if False:
            return 10
        arg = arg.coerce_to_simple(env)
        CoercionNode.__init__(self, arg)
        self.type = Builtin.bytes_type

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        arg = self.arg
        arg_result = arg.result()
        if arg.type not in (PyrexTypes.c_char_type, PyrexTypes.c_uchar_type, PyrexTypes.c_schar_type):
            if arg.type.signed:
                code.putln('if ((%s < 0) || (%s > 255)) {' % (arg_result, arg_result))
            else:
                code.putln('if (%s > 255) {' % arg_result)
            code.putln('PyErr_SetString(PyExc_OverflowError, "value too large to pack into a byte"); %s' % code.error_goto(self.pos))
            code.putln('}')
        temp = None
        if arg.type is not PyrexTypes.c_char_type:
            temp = code.funcstate.allocate_temp(PyrexTypes.c_char_type, manage_ref=False)
            code.putln('%s = (char)%s;' % (temp, arg_result))
            arg_result = temp
        code.putln('%s = PyBytes_FromStringAndSize(&%s, 1); %s' % (self.result(), arg_result, code.error_goto_if_null(self.result(), self.pos)))
        if temp is not None:
            code.funcstate.release_temp(temp)
        self.generate_gotref(code)

class CoerceFromPyTypeNode(CoercionNode):
    special_none_cvalue = None

    def __init__(self, result_type, arg, env):
        if False:
            i = 10
            return i + 15
        CoercionNode.__init__(self, arg)
        self.type = result_type
        self.is_temp = 1
        if not result_type.create_from_py_utility_code(env):
            error(arg.pos, "Cannot convert Python object to '%s'" % result_type)
        if self.type.is_string or self.type.is_pyunicode_ptr:
            if self.arg.is_name and self.arg.entry and self.arg.entry.is_pyglobal:
                warning(arg.pos, "Obtaining '%s' from externally modifiable global Python value" % result_type, level=1)

    def analyse_types(self, env):
        if False:
            while True:
                i = 10
        return self

    def is_ephemeral(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.type.is_ptr and (not self.type.is_array)) and self.arg.is_ephemeral()

    def generate_result_code(self, code):
        if False:
            i = 10
            return i + 15
        from_py_function = None
        if self.type.is_string and self.arg.type is bytes_type:
            if self.type.from_py_function.startswith('__Pyx_PyObject_As'):
                from_py_function = '__Pyx_PyBytes' + self.type.from_py_function[len('__Pyx_PyObject'):]
                NoneCheckNode.generate_if_needed(self.arg, code, 'expected bytes, NoneType found')
        code.putln(self.type.from_py_call_code(self.arg.py_result(), self.result(), self.pos, code, from_py_function=from_py_function, special_none_cvalue=self.special_none_cvalue))
        if self.type.is_pyobject:
            self.generate_gotref(code)

    def nogil_check(self, env):
        if False:
            while True:
                i = 10
        error(self.pos, 'Coercion from Python not allowed without the GIL')

class CoerceToBooleanNode(CoercionNode):
    type = PyrexTypes.c_bint_type
    _special_builtins = {Builtin.list_type: 'PyList_GET_SIZE', Builtin.tuple_type: 'PyTuple_GET_SIZE', Builtin.set_type: 'PySet_GET_SIZE', Builtin.frozenset_type: 'PySet_GET_SIZE', Builtin.bytes_type: 'PyBytes_GET_SIZE', Builtin.bytearray_type: 'PyByteArray_GET_SIZE', Builtin.unicode_type: '__Pyx_PyUnicode_IS_TRUE'}

    def __init__(self, arg, env):
        if False:
            print('Hello World!')
        CoercionNode.__init__(self, arg)
        if arg.type.is_pyobject:
            self.is_temp = 1

    def nogil_check(self, env):
        if False:
            while True:
                i = 10
        if self.arg.type.is_pyobject and self._special_builtins.get(self.arg.type) is None:
            self.gil_error()
    gil_message = 'Truth-testing Python object'

    def check_const(self):
        if False:
            i = 10
            return i + 15
        if self.is_temp:
            self.not_const()
            return False
        return self.arg.check_const()

    def calculate_result_code(self):
        if False:
            for i in range(10):
                print('nop')
        return '(%s != 0)' % self.arg.result()

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        if not self.is_temp:
            return
        test_func = self._special_builtins.get(self.arg.type)
        if test_func is not None:
            checks = ['(%s != Py_None)' % self.arg.py_result()] if self.arg.may_be_none() else []
            checks.append('(%s(%s) != 0)' % (test_func, self.arg.py_result()))
            code.putln('%s = %s;' % (self.result(), '&&'.join(checks)))
        else:
            code.putln('%s = __Pyx_PyObject_IsTrue(%s); %s' % (self.result(), self.arg.py_result(), code.error_goto_if_neg(self.result(), self.pos)))

    def analyse_types(self, env):
        if False:
            return 10
        return self

class CoerceToComplexNode(CoercionNode):

    def __init__(self, arg, dst_type, env):
        if False:
            print('Hello World!')
        if arg.type.is_complex:
            arg = arg.coerce_to_simple(env)
        self.type = dst_type
        CoercionNode.__init__(self, arg)
        dst_type.create_declaration_utility_code(env)

    def calculate_result_code(self):
        if False:
            while True:
                i = 10
        if self.arg.type.is_complex:
            real_part = self.arg.type.real_code(self.arg.result())
            imag_part = self.arg.type.imag_code(self.arg.result())
        else:
            real_part = self.arg.result()
            imag_part = '0'
        return '%s(%s, %s)' % (self.type.from_parts, real_part, imag_part)

    def generate_result_code(self, code):
        if False:
            return 10
        pass

    def analyse_types(self, env):
        if False:
            return 10
        return self

def coerce_from_soft_complex(arg, dst_type, env):
    if False:
        return 10
    from .UtilNodes import HasGilNode
    cfunc_type = PyrexTypes.CFuncType(PyrexTypes.c_double_type, [PyrexTypes.CFuncTypeArg('value', PyrexTypes.soft_complex_type, None), PyrexTypes.CFuncTypeArg('have_gil', PyrexTypes.c_bint_type, None)], exception_value='-1', exception_check=True, nogil=True)
    call = PythonCapiCallNode(arg.pos, '__Pyx_SoftComplexToDouble', cfunc_type, utility_code=UtilityCode.load_cached('SoftComplexToDouble', 'Complex.c'), args=[arg, HasGilNode(arg.pos)])
    call = call.analyse_types(env)
    if call.type != dst_type:
        call = call.coerce_to(dst_type, env)
    return call

class CoerceToTempNode(CoercionNode):

    def __init__(self, arg, env):
        if False:
            i = 10
            return i + 15
        CoercionNode.__init__(self, arg)
        self.type = self.arg.type.as_argument_type()
        self.constant_result = self.arg.constant_result
        self.is_temp = 1
        if self.type.is_pyobject:
            self.result_ctype = py_object_type
    gil_message = 'Creating temporary Python reference'

    def analyse_types(self, env):
        if False:
            i = 10
            return i + 15
        return self

    def may_be_none(self):
        if False:
            i = 10
            return i + 15
        return self.arg.may_be_none()

    def coerce_to_boolean(self, env):
        if False:
            while True:
                i = 10
        self.arg = self.arg.coerce_to_boolean(env)
        if self.arg.is_simple():
            return self.arg
        self.type = self.arg.type
        self.result_ctype = self.type
        return self

    def generate_result_code(self, code):
        if False:
            print('Hello World!')
        code.putln('%s = %s;' % (self.result(), self.arg.result_as(self.ctype())))
        if self.use_managed_ref:
            if not self.type.is_memoryviewslice:
                code.put_incref(self.result(), self.ctype())
            else:
                code.put_incref_memoryviewslice(self.result(), self.type, have_gil=not self.in_nogil_context)

class ProxyNode(CoercionNode):
    """
    A node that should not be replaced by transforms or other means,
    and hence can be useful to wrap the argument to a clone node

    MyNode    -> ProxyNode -> ArgNode
    CloneNode -^
    """
    nogil_check = None

    def __init__(self, arg):
        if False:
            while True:
                i = 10
        super(ProxyNode, self).__init__(arg)
        self.constant_result = arg.constant_result
        self.update_type_and_entry()

    def analyse_types(self, env):
        if False:
            print('Hello World!')
        self.arg = self.arg.analyse_expressions(env)
        self.update_type_and_entry()
        return self

    def infer_type(self, env):
        if False:
            while True:
                i = 10
        return self.arg.infer_type(env)

    def update_type_and_entry(self):
        if False:
            return 10
        type = getattr(self.arg, 'type', None)
        if type:
            self.type = type
            self.result_ctype = self.arg.result_ctype
        arg_entry = getattr(self.arg, 'entry', None)
        if arg_entry:
            self.entry = arg_entry

    def generate_result_code(self, code):
        if False:
            return 10
        self.arg.generate_result_code(code)

    def result(self):
        if False:
            while True:
                i = 10
        return self.arg.result()

    def is_simple(self):
        if False:
            print('Hello World!')
        return self.arg.is_simple()

    def may_be_none(self):
        if False:
            i = 10
            return i + 15
        return self.arg.may_be_none()

    def generate_evaluation_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        self.arg.generate_evaluation_code(code)

    def generate_disposal_code(self, code):
        if False:
            return 10
        self.arg.generate_disposal_code(code)

    def free_temps(self, code):
        if False:
            print('Hello World!')
        self.arg.free_temps(code)

class CloneNode(CoercionNode):
    subexprs = []
    nogil_check = None

    def __init__(self, arg):
        if False:
            while True:
                i = 10
        CoercionNode.__init__(self, arg)
        self.constant_result = arg.constant_result
        type = getattr(arg, 'type', None)
        if type:
            self.type = type
            self.result_ctype = arg.result_ctype
        arg_entry = getattr(arg, 'entry', None)
        if arg_entry:
            self.entry = arg_entry

    def result(self):
        if False:
            print('Hello World!')
        return self.arg.result()

    def may_be_none(self):
        if False:
            for i in range(10):
                print('nop')
        return self.arg.may_be_none()

    def type_dependencies(self, env):
        if False:
            i = 10
            return i + 15
        return self.arg.type_dependencies(env)

    def infer_type(self, env):
        if False:
            for i in range(10):
                print('nop')
        return self.arg.infer_type(env)

    def analyse_types(self, env):
        if False:
            for i in range(10):
                print('nop')
        self.type = self.arg.type
        self.result_ctype = self.arg.result_ctype
        self.is_temp = 1
        arg_entry = getattr(self.arg, 'entry', None)
        if arg_entry:
            self.entry = arg_entry
        return self

    def coerce_to(self, dest_type, env):
        if False:
            while True:
                i = 10
        if self.arg.is_literal:
            return self.arg.coerce_to(dest_type, env)
        return super(CloneNode, self).coerce_to(dest_type, env)

    def is_simple(self):
        if False:
            print('Hello World!')
        return True

    def generate_evaluation_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        pass

    def generate_result_code(self, code):
        if False:
            print('Hello World!')
        pass

    def generate_disposal_code(self, code):
        if False:
            print('Hello World!')
        pass

    def generate_post_assignment_code(self, code):
        if False:
            i = 10
            return i + 15
        if self.is_temp:
            code.put_incref(self.result(), self.ctype())

    def free_temps(self, code):
        if False:
            i = 10
            return i + 15
        pass

class CppOptionalTempCoercion(CoercionNode):
    """
    Used only in CoerceCppTemps - handles cases the temp is actually a OptionalCppClassType (and thus needs dereferencing when on the rhs)
    """
    is_temp = False

    @property
    def type(self):
        if False:
            i = 10
            return i + 15
        return self.arg.type

    def calculate_result_code(self):
        if False:
            for i in range(10):
                print('nop')
        return '(*%s)' % self.arg.result()

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        pass

    def _make_move_result_rhs(self, result, optional=False):
        if False:
            i = 10
            return i + 15
        return super(CppOptionalTempCoercion, self)._make_move_result_rhs(result, optional=False)

class CMethodSelfCloneNode(CloneNode):

    def coerce_to(self, dst_type, env):
        if False:
            i = 10
            return i + 15
        if dst_type.is_builtin_type and self.type.subtype_of(dst_type):
            return self
        return CloneNode.coerce_to(self, dst_type, env)

class ModuleRefNode(ExprNode):
    type = py_object_type
    is_temp = False
    subexprs = []

    def analyse_types(self, env):
        if False:
            return 10
        return self

    def may_be_none(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def calculate_result_code(self):
        if False:
            for i in range(10):
                print('nop')
        return Naming.module_cname

    def generate_result_code(self, code):
        if False:
            print('Hello World!')
        pass

class DocstringRefNode(ExprNode):
    subexprs = ['body']
    type = py_object_type
    is_temp = True

    def __init__(self, pos, body):
        if False:
            for i in range(10):
                print('nop')
        ExprNode.__init__(self, pos)
        assert body.type.is_pyobject
        self.body = body

    def analyse_types(self, env):
        if False:
            print('Hello World!')
        return self

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        code.putln('%s = __Pyx_GetAttr(%s, %s); %s' % (self.result(), self.body.result(), code.intern_identifier(StringEncoding.EncodedString('__doc__')), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

class AnnotationNode(ExprNode):
    subexprs = []
    untyped = False

    def __init__(self, pos, expr, string=None):
        if False:
            for i in range(10):
                print('nop')
        'string is expected to already be a StringNode or None'
        ExprNode.__init__(self, pos)
        if string is None:
            from .AutoDocTransforms import AnnotationWriter
            string = StringEncoding.EncodedString(AnnotationWriter(description='annotation').write(expr))
            string = StringNode(pos, unicode_value=string, value=string.as_utf8_string())
        self.string = string
        self.expr = expr

    def analyse_types(self, env):
        if False:
            i = 10
            return i + 15
        return self

    def analyse_as_type(self, env):
        if False:
            for i in range(10):
                print('nop')
        return self.analyse_type_annotation(env)[1]

    def _warn_on_unknown_annotation(self, env, annotation):
        if False:
            i = 10
            return i + 15
        'Method checks for cases when user should be warned that annotation contains unknown types.'
        if isinstance(annotation, SliceIndexNode):
            annotation = annotation.base
        if annotation.is_name:
            if not env.lookup(annotation.name):
                warning(annotation.pos, "Unknown type declaration '%s' in annotation, ignoring" % self.string.value, level=1)
        elif annotation.is_attribute and annotation.obj.is_name:
            if not env.lookup(annotation.obj.name):
                warning(annotation.pos, "Unknown type declaration '%s' in annotation, ignoring" % self.string.value, level=1)
            elif annotation.obj.is_cython_module:
                module_scope = annotation.obj.analyse_as_module(env)
                if module_scope and (not module_scope.lookup_type(annotation.attribute)):
                    error(annotation.pos, "Unknown type declaration '%s' in annotation" % self.string.value)
            else:
                module_scope = annotation.obj.analyse_as_module(env)
                if module_scope and module_scope.pxd_file_loaded:
                    warning(annotation.pos, "Unknown type declaration '%s' in annotation, ignoring" % self.string.value, level=1)
        else:
            warning(annotation.pos, 'Unknown type declaration in annotation, ignoring')

    def analyse_type_annotation(self, env, assigned_value=None):
        if False:
            for i in range(10):
                print('nop')
        if self.untyped:
            return ([], None)
        annotation = self.expr
        explicit_pytype = explicit_ctype = False
        if annotation.is_dict_literal:
            warning(annotation.pos, "Dicts should no longer be used as type annotations. Use 'cython.int' etc. directly.", level=1)
            for (name, value) in annotation.key_value_pairs:
                if not name.is_string_literal:
                    continue
                if name.value in ('type', b'type'):
                    explicit_pytype = True
                    if not explicit_ctype:
                        annotation = value
                elif name.value in ('ctype', b'ctype'):
                    explicit_ctype = True
                    annotation = value
            if explicit_pytype and explicit_ctype:
                warning(annotation.pos, 'Duplicate type declarations found in signature annotation', level=1)
        elif isinstance(annotation, TupleNode):
            warning(annotation.pos, "Tuples cannot be declared as simple tuples of types. Use 'tuple[type1, type2, ...]'.", level=1)
            return ([], None)
        with env.new_c_type_context(in_c_type_context=explicit_ctype):
            arg_type = annotation.analyse_as_type(env)
            if arg_type is None:
                self._warn_on_unknown_annotation(env, annotation)
                return ([], arg_type)
            if annotation.is_string_literal:
                warning(annotation.pos, "Strings should no longer be used for type declarations. Use 'cython.int' etc. directly.", level=1)
            if explicit_pytype and (not explicit_ctype) and (not (arg_type.is_pyobject or arg_type.equivalent_type)):
                warning(annotation.pos, 'Python type declaration in signature annotation does not refer to a Python type')
            if arg_type.is_complex:
                arg_type.create_declaration_utility_code(env)
            modifiers = annotation.analyse_pytyping_modifiers(env) if annotation.is_subscript else []
        return (modifiers, arg_type)

class AssignmentExpressionNode(ExprNode):
    """
    Also known as a named expression or the walrus operator

    Arguments
    lhs - NameNode - not stored directly as an attribute of the node
    rhs - ExprNode

    Attributes
    rhs        - ExprNode
    assignment - SingleAssignmentNode
    """
    subexprs = ['rhs']
    child_attrs = ['rhs', 'assignment']
    is_temp = False
    assignment = None
    clone_node = None

    def __init__(self, pos, lhs, rhs, **kwds):
        if False:
            for i in range(10):
                print('nop')
        super(AssignmentExpressionNode, self).__init__(pos, **kwds)
        self.rhs = ProxyNode(rhs)
        assign_expr_rhs = CloneNode(self.rhs)
        self.assignment = SingleAssignmentNode(pos, lhs=lhs, rhs=assign_expr_rhs, is_assignment_expression=True)

    @property
    def type(self):
        if False:
            for i in range(10):
                print('nop')
        return self.rhs.type

    @property
    def target_name(self):
        if False:
            return 10
        return self.assignment.lhs.name

    def infer_type(self, env):
        if False:
            while True:
                i = 10
        return self.rhs.infer_type(env)

    def analyse_declarations(self, env):
        if False:
            return 10
        self.assignment.analyse_declarations(env)

    def analyse_types(self, env):
        if False:
            i = 10
            return i + 15
        self.rhs = self.rhs.analyse_types(env)
        if not self.rhs.arg.is_temp:
            if not self.rhs.arg.is_literal:
                self.rhs.arg = self.rhs.arg.coerce_to_temp(env)
            else:
                self.assignment.rhs = copy.copy(self.rhs)
        self.assignment = self.assignment.analyse_types(env)
        return self

    def coerce_to(self, dst_type, env):
        if False:
            return 10
        if dst_type == self.assignment.rhs.type:
            old_rhs_arg = self.rhs.arg
            if isinstance(old_rhs_arg, CoerceToTempNode):
                old_rhs_arg = old_rhs_arg.arg
            rhs_arg = old_rhs_arg.coerce_to(dst_type, env)
            if rhs_arg is not old_rhs_arg:
                self.rhs.arg = rhs_arg
                self.rhs.update_type_and_entry()
                if isinstance(self.assignment.rhs, CoercionNode) and (not isinstance(self.assignment.rhs, CloneNode)):
                    self.assignment.rhs = self.assignment.rhs.arg
                    self.assignment.rhs.type = self.assignment.rhs.arg.type
                return self
        return super(AssignmentExpressionNode, self).coerce_to(dst_type, env)

    def calculate_result_code(self):
        if False:
            for i in range(10):
                print('nop')
        return self.rhs.result()

    def generate_result_code(self, code):
        if False:
            while True:
                i = 10
        self.assignment.generate_execution_code(code)