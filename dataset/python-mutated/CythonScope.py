from __future__ import absolute_import
from .Symtab import ModuleScope
from .PyrexTypes import *
from .UtilityCode import CythonUtilityCode
from .Errors import error
from .Scanning import StringSourceDescriptor
from . import MemoryView
from .StringEncoding import EncodedString

class CythonScope(ModuleScope):
    is_cython_builtin = 1
    _cythonscope_initialized = False

    def __init__(self, context):
        if False:
            while True:
                i = 10
        ModuleScope.__init__(self, u'cython', None, None)
        self.pxd_file_loaded = True
        self.populate_cython_scope()
        self.context = context
        for fused_type in (cy_integral_type, cy_floating_type, cy_numeric_type):
            entry = self.declare_typedef(fused_type.name, fused_type, None, cname='<error>')
            entry.in_cinclude = True

    def is_cpp(self):
        if False:
            i = 10
            return i + 15
        return self.context.cpp

    def lookup_type(self, name):
        if False:
            while True:
                i = 10
        type = parse_basic_type(name)
        if type:
            return type
        return super(CythonScope, self).lookup_type(name)

    def lookup(self, name):
        if False:
            while True:
                i = 10
        entry = super(CythonScope, self).lookup(name)
        if entry is None and (not self._cythonscope_initialized):
            self.load_cythonscope()
            entry = super(CythonScope, self).lookup(name)
        return entry

    def find_module(self, module_name, pos):
        if False:
            return 10
        error('cython.%s is not available' % module_name, pos)

    def find_submodule(self, module_name, as_package=False):
        if False:
            for i in range(10):
                print('nop')
        entry = self.entries.get(module_name, None)
        if not entry:
            self.load_cythonscope()
            entry = self.entries.get(module_name, None)
        if entry and entry.as_module:
            return entry.as_module
        else:
            raise error((StringSourceDescriptor(u'cython', u''), 0, 0), 'cython.%s is not available' % module_name)

    def lookup_qualified_name(self, qname):
        if False:
            i = 10
            return i + 15
        name_path = qname.split(u'.')
        scope = self
        while len(name_path) > 1:
            scope = scope.lookup_here(name_path[0])
            if scope:
                scope = scope.as_module
            del name_path[0]
            if scope is None:
                return None
        else:
            return scope.lookup_here(name_path[0])

    def populate_cython_scope(self):
        if False:
            return 10
        type_object = self.declare_typedef('PyTypeObject', base_type=c_void_type, pos=None, cname='PyTypeObject')
        type_object.is_void = True
        type_object_type = type_object.type
        self.declare_cfunction('PyObject_TypeCheck', CFuncType(c_bint_type, [CFuncTypeArg('o', py_object_type, None), CFuncTypeArg('t', c_ptr_type(type_object_type), None)]), pos=None, defining=1, cname='PyObject_TypeCheck')

    def load_cythonscope(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates some entries for testing purposes and entries for\n        cython.array() and for cython.view.*.\n        '
        if self._cythonscope_initialized:
            return
        self._cythonscope_initialized = True
        cython_testscope_utility_code.declare_in_scope(self, cython_scope=self)
        cython_test_extclass_utility_code.declare_in_scope(self, cython_scope=self)
        self.viewscope = viewscope = ModuleScope(u'view', self, None)
        self.declare_module('view', viewscope, None).as_module = viewscope
        viewscope.is_cython_builtin = True
        viewscope.pxd_file_loaded = True
        cythonview_testscope_utility_code.declare_in_scope(viewscope, cython_scope=self)
        view_utility_scope = MemoryView.view_utility_code.declare_in_scope(self.viewscope, cython_scope=self, allowlist=MemoryView.view_utility_allowlist)
        ext_types = [entry.type for entry in view_utility_scope.entries.values() if entry.type.is_extension_type]
        for ext_type in ext_types:
            ext_type.is_cython_builtin_type = 1
        dc_str = EncodedString(u'dataclasses')
        dataclassesscope = ModuleScope(dc_str, self, context=None)
        self.declare_module(dc_str, dataclassesscope, pos=None).as_module = dataclassesscope
        dataclassesscope.is_cython_builtin = True
        dataclassesscope.pxd_file_loaded = True

def create_cython_scope(context):
    if False:
        return 10
    return CythonScope(context)

def load_testscope_utility(cy_util_name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return CythonUtilityCode.load(cy_util_name, 'TestCythonScope.pyx', **kwargs)
undecorated_methods_protos = UtilityCode(proto=u'\n    /* These methods are undecorated and have therefore no prototype */\n    static PyObject *__pyx_TestClass_cdef_method(\n            struct __pyx_TestClass_obj *self, int value);\n    static PyObject *__pyx_TestClass_cpdef_method(\n            struct __pyx_TestClass_obj *self, int value, int skip_dispatch);\n    static PyObject *__pyx_TestClass_def_method(\n            PyObject *self, PyObject *value);\n')
cython_testscope_utility_code = load_testscope_utility('TestScope')
test_cython_utility_dep = load_testscope_utility('TestDep')
cython_test_extclass_utility_code = load_testscope_utility('TestClass', name='TestClass', requires=[undecorated_methods_protos, test_cython_utility_dep])
cythonview_testscope_utility_code = load_testscope_utility('View.TestScope')