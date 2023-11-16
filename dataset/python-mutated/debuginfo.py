"""
Implements helpers to build LLVM debuginfo.
"""
import abc
import os.path
from contextlib import contextmanager
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel.models import ComplexModel, UniTupleModel
from numba.core import config

@contextmanager
def suspend_emission(builder):
    if False:
        i = 10
        return i + 15
    'Suspends the emission of debug_metadata for the duration of the context\n    managed block.'
    ref = builder.debug_metadata
    builder.debug_metadata = None
    try:
        yield
    finally:
        builder.debug_metadata = ref

class AbstractDIBuilder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def mark_variable(self, builder, allocavalue, name, lltype, size, line, datamodel=None, argidx=None):
        if False:
            while True:
                i = 10
        'Emit debug info for the variable.\n        '
        pass

    @abc.abstractmethod
    def mark_location(self, builder, line):
        if False:
            i = 10
            return i + 15
        'Emit source location information to the given IRBuilder.\n        '
        pass

    @abc.abstractmethod
    def mark_subprogram(self, function, qualname, argnames, argtypes, line):
        if False:
            for i in range(10):
                print('nop')
        'Emit source location information for the given function.\n        '
        pass

    @abc.abstractmethod
    def initialize(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the debug info. An opportunity for the debuginfo to\n        prepare any necessary data structures.\n        '

    @abc.abstractmethod
    def finalize(self):
        if False:
            print('Hello World!')
        'Finalize the debuginfo by emitting all necessary metadata.\n        '
        pass

class DummyDIBuilder(AbstractDIBuilder):

    def __init__(self, module, filepath, cgctx, directives_only):
        if False:
            while True:
                i = 10
        pass

    def mark_variable(self, builder, allocavalue, name, lltype, size, line, datamodel=None, argidx=None):
        if False:
            print('Hello World!')
        pass

    def mark_location(self, builder, line):
        if False:
            for i in range(10):
                print('nop')
        pass

    def mark_subprogram(self, function, qualname, argnames, argtypes, line):
        if False:
            while True:
                i = 10
        pass

    def initialize(self):
        if False:
            print('Hello World!')
        pass

    def finalize(self):
        if False:
            i = 10
            return i + 15
        pass
_BYTE_SIZE = 8

class DIBuilder(AbstractDIBuilder):
    DWARF_VERSION = 4
    DEBUG_INFO_VERSION = 3
    DBG_CU_NAME = 'llvm.dbg.cu'
    _DEBUG = False

    def __init__(self, module, filepath, cgctx, directives_only):
        if False:
            i = 10
            return i + 15
        self.module = module
        self.filepath = os.path.abspath(filepath)
        self.difile = self._di_file()
        self.subprograms = []
        self.cgctx = cgctx
        if directives_only:
            self.emission_kind = 'DebugDirectivesOnly'
        else:
            self.emission_kind = 'FullDebug'
        self.initialize()

    def initialize(self):
        if False:
            i = 10
            return i + 15
        self.dicompileunit = self._di_compile_unit()

    def _var_type(self, lltype, size, datamodel=None):
        if False:
            while True:
                i = 10
        if self._DEBUG:
            print('-->', lltype, size, datamodel, getattr(datamodel, 'fe_type', 'NO FE TYPE'))
        m = self.module
        bitsize = _BYTE_SIZE * size
        int_type = (ir.IntType,)
        real_type = (ir.FloatType, ir.DoubleType)
        if isinstance(lltype, int_type + real_type):
            if datamodel is None:
                name = str(lltype)
                if isinstance(lltype, int_type):
                    ditok = 'DW_ATE_unsigned'
                else:
                    ditok = 'DW_ATE_float'
            else:
                name = str(datamodel.fe_type)
                if isinstance(datamodel.fe_type, types.Integer):
                    if datamodel.fe_type.signed:
                        ditok = 'DW_ATE_signed'
                    else:
                        ditok = 'DW_ATE_unsigned'
                else:
                    ditok = 'DW_ATE_float'
            mdtype = m.add_debug_info('DIBasicType', {'name': name, 'size': bitsize, 'encoding': ir.DIToken(ditok)})
        elif isinstance(datamodel, ComplexModel):
            meta = []
            offset = 0
            for (ix, name) in enumerate(('real', 'imag')):
                component = lltype.elements[ix]
                component_size = self.cgctx.get_abi_sizeof(component)
                component_basetype = m.add_debug_info('DIBasicType', {'name': str(component), 'size': _BYTE_SIZE * component_size, 'encoding': ir.DIToken('DW_ATE_float')})
                derived_type = m.add_debug_info('DIDerivedType', {'tag': ir.DIToken('DW_TAG_member'), 'name': name, 'baseType': component_basetype, 'size': _BYTE_SIZE * component_size, 'offset': offset})
                meta.append(derived_type)
                offset += _BYTE_SIZE * component_size
            mdtype = m.add_debug_info('DICompositeType', {'tag': ir.DIToken('DW_TAG_structure_type'), 'name': f'{datamodel.fe_type} ({str(lltype)})', 'identifier': str(lltype), 'elements': m.add_metadata(meta), 'size': offset}, is_distinct=True)
        elif isinstance(datamodel, UniTupleModel):
            element = lltype.element
            el_size = self.cgctx.get_abi_sizeof(element)
            basetype = self._var_type(element, el_size)
            name = f'{datamodel.fe_type} ({str(lltype)})'
            count = size // el_size
            mdrange = m.add_debug_info('DISubrange', {'count': count})
            mdtype = m.add_debug_info('DICompositeType', {'tag': ir.DIToken('DW_TAG_array_type'), 'baseType': basetype, 'name': name, 'size': bitsize, 'identifier': str(lltype), 'elements': m.add_metadata([mdrange])})
        elif isinstance(lltype, ir.PointerType):
            model = getattr(datamodel, '_pointee_model', None)
            basetype = self._var_type(lltype.pointee, self.cgctx.get_abi_sizeof(lltype.pointee), model)
            mdtype = m.add_debug_info('DIDerivedType', {'tag': ir.DIToken('DW_TAG_pointer_type'), 'baseType': basetype, 'size': _BYTE_SIZE * self.cgctx.get_abi_sizeof(lltype)})
        elif isinstance(lltype, ir.LiteralStructType):
            meta = []
            offset = 0
            if datamodel is None or not datamodel.inner_models():
                name = f'Anonymous struct ({str(lltype)})'
                for (field_id, element) in enumerate(lltype.elements):
                    size = self.cgctx.get_abi_sizeof(element)
                    basetype = self._var_type(element, size)
                    derived_type = m.add_debug_info('DIDerivedType', {'tag': ir.DIToken('DW_TAG_member'), 'name': f'<field {field_id}>', 'baseType': basetype, 'size': _BYTE_SIZE * size, 'offset': offset})
                    meta.append(derived_type)
                    offset += _BYTE_SIZE * size
            else:
                name = f'{datamodel.fe_type} ({str(lltype)})'
                for (element, field, model) in zip(lltype.elements, datamodel._fields, datamodel.inner_models()):
                    size = self.cgctx.get_abi_sizeof(element)
                    basetype = self._var_type(element, size, datamodel=model)
                    derived_type = m.add_debug_info('DIDerivedType', {'tag': ir.DIToken('DW_TAG_member'), 'name': field, 'baseType': basetype, 'size': _BYTE_SIZE * size, 'offset': offset})
                    meta.append(derived_type)
                    offset += _BYTE_SIZE * size
            mdtype = m.add_debug_info('DICompositeType', {'tag': ir.DIToken('DW_TAG_structure_type'), 'name': name, 'identifier': str(lltype), 'elements': m.add_metadata(meta), 'size': offset}, is_distinct=True)
        elif isinstance(lltype, ir.ArrayType):
            element = lltype.element
            el_size = self.cgctx.get_abi_sizeof(element)
            basetype = self._var_type(element, el_size)
            count = size // el_size
            mdrange = m.add_debug_info('DISubrange', {'count': count})
            mdtype = m.add_debug_info('DICompositeType', {'tag': ir.DIToken('DW_TAG_array_type'), 'baseType': basetype, 'name': str(lltype), 'size': bitsize, 'identifier': str(lltype), 'elements': m.add_metadata([mdrange])})
        else:
            count = size
            mdrange = m.add_debug_info('DISubrange', {'count': count})
            mdbase = m.add_debug_info('DIBasicType', {'name': 'byte', 'size': _BYTE_SIZE, 'encoding': ir.DIToken('DW_ATE_unsigned_char')})
            mdtype = m.add_debug_info('DICompositeType', {'tag': ir.DIToken('DW_TAG_array_type'), 'baseType': mdbase, 'name': str(lltype), 'size': bitsize, 'identifier': str(lltype), 'elements': m.add_metadata([mdrange])})
        return mdtype

    def mark_variable(self, builder, allocavalue, name, lltype, size, line, datamodel=None, argidx=None):
        if False:
            return 10
        arg_index = 0 if argidx is None else argidx
        m = self.module
        fnty = ir.FunctionType(ir.VoidType(), [ir.MetaDataType()] * 3)
        decl = cgutils.get_or_insert_function(m, fnty, 'llvm.dbg.declare')
        mdtype = self._var_type(lltype, size, datamodel=datamodel)
        name = name.replace('.', '$')
        mdlocalvar = m.add_debug_info('DILocalVariable', {'name': name, 'arg': arg_index, 'scope': self.subprograms[-1], 'file': self.difile, 'line': line, 'type': mdtype})
        mdexpr = m.add_debug_info('DIExpression', {})
        return builder.call(decl, [allocavalue, mdlocalvar, mdexpr])

    def mark_location(self, builder, line):
        if False:
            print('Hello World!')
        builder.debug_metadata = self._add_location(line)

    def mark_subprogram(self, function, qualname, argnames, argtypes, line):
        if False:
            print('Hello World!')
        name = qualname
        argmap = dict(zip(argnames, argtypes))
        di_subp = self._add_subprogram(name=name, linkagename=function.name, line=line, function=function, argmap=argmap)
        function.set_metadata('dbg', di_subp)

    def finalize(self):
        if False:
            return 10
        dbgcu = cgutils.get_or_insert_named_metadata(self.module, self.DBG_CU_NAME)
        dbgcu.add(self.dicompileunit)
        self._set_module_flags()

    def _set_module_flags(self):
        if False:
            print('Hello World!')
        'Set the module flags metadata\n        '
        module = self.module
        mflags = cgutils.get_or_insert_named_metadata(module, 'llvm.module.flags')
        require_warning_behavior = self._const_int(2)
        if self.DWARF_VERSION is not None:
            dwarf_version = module.add_metadata([require_warning_behavior, 'Dwarf Version', self._const_int(self.DWARF_VERSION)])
            if dwarf_version not in mflags.operands:
                mflags.add(dwarf_version)
        debuginfo_version = module.add_metadata([require_warning_behavior, 'Debug Info Version', self._const_int(self.DEBUG_INFO_VERSION)])
        if debuginfo_version not in mflags.operands:
            mflags.add(debuginfo_version)

    def _add_subprogram(self, name, linkagename, line, function, argmap):
        if False:
            return 10
        'Emit subprogram metadata\n        '
        subp = self._di_subprogram(name, linkagename, line, function, argmap)
        self.subprograms.append(subp)
        return subp

    def _add_location(self, line):
        if False:
            return 10
        'Emit location metatdaa\n        '
        loc = self._di_location(line)
        return loc

    @classmethod
    def _const_int(cls, num, bits=32):
        if False:
            while True:
                i = 10
        'Util to create constant int in metadata\n        '
        return ir.IntType(bits)(num)

    @classmethod
    def _const_bool(cls, boolean):
        if False:
            i = 10
            return i + 15
        'Util to create constant boolean in metadata\n        '
        return ir.IntType(1)(boolean)

    def _di_file(self):
        if False:
            while True:
                i = 10
        return self.module.add_debug_info('DIFile', {'directory': os.path.dirname(self.filepath), 'filename': os.path.basename(self.filepath)})

    def _di_compile_unit(self):
        if False:
            print('Hello World!')
        return self.module.add_debug_info('DICompileUnit', {'language': ir.DIToken('DW_LANG_C_plus_plus'), 'file': self.difile, 'producer': 'clang (Numba)', 'runtimeVersion': 0, 'isOptimized': config.OPT != 0, 'emissionKind': ir.DIToken(self.emission_kind)}, is_distinct=True)

    def _di_subroutine_type(self, line, function, argmap):
        if False:
            i = 10
            return i + 15
        llfunc = function
        md = []
        for (idx, llarg) in enumerate(llfunc.args):
            if not llarg.name.startswith('arg.'):
                name = llarg.name.replace('.', '$')
                lltype = llarg.type
                size = self.cgctx.get_abi_sizeof(lltype)
                mdtype = self._var_type(lltype, size, datamodel=None)
                md.append(mdtype)
        for (idx, (name, nbtype)) in enumerate(argmap.items()):
            name = name.replace('.', '$')
            datamodel = self.cgctx.data_model_manager[nbtype]
            lltype = self.cgctx.get_value_type(nbtype)
            size = self.cgctx.get_abi_sizeof(lltype)
            mdtype = self._var_type(lltype, size, datamodel=datamodel)
            md.append(mdtype)
        return self.module.add_debug_info('DISubroutineType', {'types': self.module.add_metadata(md)})

    def _di_subprogram(self, name, linkagename, line, function, argmap):
        if False:
            while True:
                i = 10
        return self.module.add_debug_info('DISubprogram', {'name': name, 'linkageName': linkagename, 'scope': self.difile, 'file': self.difile, 'line': line, 'type': self._di_subroutine_type(line, function, argmap), 'isLocal': False, 'isDefinition': True, 'scopeLine': line, 'isOptimized': config.OPT != 0, 'unit': self.dicompileunit}, is_distinct=True)

    def _di_location(self, line):
        if False:
            i = 10
            return i + 15
        return self.module.add_debug_info('DILocation', {'line': line, 'column': 1, 'scope': self.subprograms[-1]})