""" CType classes for PyObject *, PyObject **, and struct Nuitka_CellObject *

"""
from nuitka.__past__ import iterItems, xrange
from nuitka.code_generation.ErrorCodes import getErrorExitBoolCode, getReleaseCode
from nuitka.code_generation.templates.CodeTemplatesVariables import template_del_local_intolerant, template_del_local_known, template_del_local_tolerant, template_del_shared_intolerant, template_del_shared_known, template_del_shared_tolerant, template_release_object_clear, template_release_object_unclear, template_write_local_clear_ref0, template_write_local_clear_ref1, template_write_local_empty_ref0, template_write_local_empty_ref1, template_write_local_inplace, template_write_local_unclear_ref0, template_write_local_unclear_ref1, template_write_shared_clear_ref0, template_write_shared_clear_ref1, template_write_shared_inplace, template_write_shared_unclear_ref0, template_write_shared_unclear_ref1
from nuitka.Constants import getConstantValueGuide, isMutable
from .CTypeBases import CTypeBase
make_list_constant_direct_threshold = 4
make_list_constant_hinted_threshold = 13

class CPythonPyObjectPtrBase(CTypeBase):

    @classmethod
    def emitVariableAssignCode(cls, value_name, needs_release, tmp_name, ref_count, inplace, emit, context):
        if False:
            while True:
                i = 10
        if inplace:
            template = template_write_local_inplace
        elif ref_count:
            if needs_release is False:
                template = template_write_local_empty_ref0
            elif needs_release is True:
                template = template_write_local_clear_ref0
            else:
                template = template_write_local_unclear_ref0
        elif needs_release is False:
            template = template_write_local_empty_ref1
        elif needs_release is True:
            template = template_write_local_clear_ref1
        else:
            template = template_write_local_unclear_ref1
        emit(template % {'identifier': value_name, 'tmp_name': tmp_name})

    @classmethod
    def emitAssignmentCodeToNuitkaIntOrLong(cls, to_name, value_name, needs_check, emit, context):
        if False:
            i = 10
            return i + 15
        to_type = to_name.getCType()
        to_type.emitVariantAssignmentCode(int_name=to_name, value_name=value_name, int_value=None, emit=emit, context=context)

    @classmethod
    def getTruthCheckCode(cls, value_name):
        if False:
            i = 10
            return i + 15
        return 'CHECK_IF_TRUE(%s) == 1' % value_name

    @classmethod
    def emitTruthCheckCode(cls, to_name, value_name, emit):
        if False:
            for i in range(10):
                print('nop')
        assert to_name.c_type == 'int', to_name
        emit('%s = CHECK_IF_TRUE(%s);' % (to_name, value_name))

    @classmethod
    def getReleaseCode(cls, value_name, needs_check, emit):
        if False:
            print('Hello World!')
        if needs_check:
            template = template_release_object_unclear
        else:
            template = template_release_object_clear
        emit(template % {'identifier': value_name})

    @classmethod
    def emitAssignInplaceNegatedValueCode(cls, to_name, needs_check, emit, context):
        if False:
            print('Hello World!')
        update_code = '%(to_name)s = (%(truth_check)s) ? Py_False : Py_True' % {'truth_check': cls.getTruthCheckCode(to_name), 'to_name': to_name}
        if context.needsCleanup(to_name):
            assert cls is CTypePyObjectPtr
            emit('{\n    %(tmp_decl)s = %(to_name)s;\n    %(update_code)s;\n    Py_INCREF(%(to_name)s);\n    Py_DECREF(old);\n}\n' % {'tmp_decl': cls.getVariableArgDeclarationCode('old'), 'update_code': update_code, 'to_name': to_name})
        else:
            emit('%s;' % update_code)

    @classmethod
    def emitAssignmentCodeToNuitkaBool(cls, to_name, value_name, needs_check, emit, context):
        if False:
            while True:
                i = 10
        truth_name = context.allocateTempName('truth_name', 'int')
        emit('%s = CHECK_IF_TRUE(%s);' % (truth_name, value_name))
        getErrorExitBoolCode(condition='%s == -1' % truth_name, needs_check=needs_check, emit=emit, context=context)
        emit('%s = %s == 0 ? NUITKA_BOOL_FALSE : NUITKA_BOOL_TRUE;' % (to_name, truth_name))

    @classmethod
    def emitAssignmentCodeFromConstant(cls, to_name, constant, may_escape, emit, context):
        if False:
            return 10
        if type(constant) is dict:
            if not may_escape:
                code = context.getConstantCode(constant)
                ref_count = 0
            elif constant:
                for (key, value) in iterItems(constant):
                    assert not isMutable(key)
                    if isMutable(value):
                        needs_deep = True
                        break
                else:
                    needs_deep = False
                if needs_deep:
                    code = 'DEEP_COPY_DICT(tstate, %s)' % context.getConstantCode(constant, deep_check=False)
                    ref_count = 1
                else:
                    code = 'DICT_COPY(%s)' % context.getConstantCode(constant, deep_check=False)
                    ref_count = 1
            else:
                code = 'MAKE_DICT_EMPTY()'
                ref_count = 1
        elif type(constant) is set:
            if not may_escape:
                code = context.getConstantCode(constant)
                ref_count = 0
            elif constant:
                code = 'PySet_New(%s)' % context.getConstantCode(constant)
                ref_count = 1
            else:
                code = 'PySet_New(NULL)'
                ref_count = 1
        elif type(constant) is list:
            if not may_escape:
                code = context.getConstantCode(constant)
                ref_count = 0
            elif constant:
                for value in constant:
                    if isMutable(value):
                        needs_deep = True
                        break
                else:
                    needs_deep = False
                if needs_deep:
                    code = 'DEEP_COPY_LIST_GUIDED(tstate, %s, "%s")' % (context.getConstantCode(constant, deep_check=False), getConstantValueGuide(constant, elements_only=True))
                    ref_count = 1
                else:
                    constant_size = len(constant)
                    if constant_size > 1 and all((constant[i] is constant[0] for i in xrange(1, len(constant)))):
                        code = 'MAKE_LIST_REPEATED(%s, %s)' % (constant_size, context.getConstantCode(constant[0], deep_check=False))
                    elif constant_size < make_list_constant_direct_threshold:
                        code = 'MAKE_LIST%d(%s)' % (constant_size, ','.join((context.getConstantCode(constant[i], deep_check=False) for i in xrange(constant_size))))
                    elif constant_size < make_list_constant_hinted_threshold:
                        code = 'MAKE_LIST%d(%s)' % (constant_size, context.getConstantCode(constant, deep_check=False))
                    else:
                        code = 'LIST_COPY(%s)' % context.getConstantCode(constant, deep_check=False)
                    ref_count = 1
            else:
                code = 'MAKE_LIST_EMPTY(0)'
                ref_count = 1
        elif type(constant) is tuple:
            needs_deep = False
            if may_escape:
                for value in constant:
                    if isMutable(value):
                        needs_deep = True
                        break
            if needs_deep:
                code = 'DEEP_COPY_TUPLE_GUIDED(tstate, %s, "%s")' % (context.getConstantCode(constant, deep_check=False), getConstantValueGuide(constant, elements_only=True))
                ref_count = 1
            else:
                code = context.getConstantCode(constant)
                ref_count = 0
        elif type(constant) is bytearray:
            if may_escape:
                code = 'BYTEARRAY_COPY(tstate, %s)' % context.getConstantCode(constant)
                ref_count = 1
            else:
                code = context.getConstantCode(constant)
                ref_count = 0
        else:
            code = context.getConstantCode(constant=constant)
            ref_count = 0
        if to_name.c_type == 'PyObject *':
            value_name = to_name
        else:
            value_name = context.allocateTempName('constant_value')
        emit('%s = %s;' % (value_name, code))
        if to_name is not value_name:
            cls.emitAssignConversionCode(to_name=to_name, value_name=value_name, needs_check=False, emit=emit, context=context)
            if ref_count:
                getReleaseCode(value_name, emit, context)
        elif ref_count:
            context.addCleanupTempName(value_name)

class CTypePyObjectPtr(CPythonPyObjectPtrBase):
    c_type = 'PyObject *'
    helper_code = 'OBJECT'

    @classmethod
    def getInitValue(cls, init_from):
        if False:
            print('Hello World!')
        if init_from is None:
            return 'NULL'
        else:
            return init_from

    @classmethod
    def getInitTestConditionCode(cls, value_name, inverted):
        if False:
            i = 10
            return i + 15
        return '%s %s NULL' % (value_name, '==' if inverted else '!=')

    @classmethod
    def emitReinitCode(cls, value_name, emit):
        if False:
            return 10
        emit('%s = NULL;' % value_name)

    @classmethod
    def getVariableArgDeclarationCode(cls, variable_code_name):
        if False:
            for i in range(10):
                print('nop')
        return 'PyObject *%s' % variable_code_name

    @classmethod
    def getVariableArgReferencePassingCode(cls, variable_code_name):
        if False:
            for i in range(10):
                print('nop')
        return '&%s' % variable_code_name

    @classmethod
    def getCellObjectAssignmentCode(cls, target_cell_code, variable_code_name, emit):
        if False:
            i = 10
            return i + 15
        emit('%s = Nuitka_Cell_New0(%s);' % (target_cell_code, variable_code_name))

    @classmethod
    def getDeleteObjectCode(cls, to_name, value_name, needs_check, tolerant, emit, context):
        if False:
            for i in range(10):
                print('nop')
        if not needs_check:
            emit(template_del_local_known % {'identifier': value_name})
        elif tolerant:
            emit(template_del_local_tolerant % {'identifier': value_name})
        else:
            emit(template_del_local_intolerant % {'identifier': value_name, 'result': to_name})

    @classmethod
    def emitAssignmentCodeFromBoolCondition(cls, to_name, condition, emit):
        if False:
            print('Hello World!')
        emit('%(to_name)s = (%(condition)s) ? Py_True : Py_False;' % {'to_name': to_name, 'condition': condition})

    @classmethod
    def emitValueAccessCode(cls, value_name, emit, context):
        if False:
            return 10
        return value_name

    @classmethod
    def emitValueAssertionCode(cls, value_name, emit):
        if False:
            return 10
        emit('CHECK_OBJECT(%s);' % value_name)

    @classmethod
    def emitAssignConversionCode(cls, to_name, value_name, needs_check, emit, context):
        if False:
            while True:
                i = 10
        if value_name.c_type == cls.c_type:
            emit('%s = %s;' % (to_name, value_name))
            context.transferCleanupTempName(value_name, to_name)
        elif value_name.c_type in ('nuitka_bool', 'bool'):
            cls.emitAssignmentCodeFromBoolCondition(condition=value_name.getCType().getTruthCheckCode(value_name), to_name=to_name, emit=emit)
        elif value_name.c_type == 'nuitka_ilong':
            emit('ENFORCE_ILONG_OBJECT_VALUE(&%s);' % value_name)
            emit('%s = %s.ilong_object;' % (to_name, value_name))
            context.transferCleanupTempName(value_name, to_name)
        else:
            assert False, to_name.c_type

    @classmethod
    def getExceptionCheckCondition(cls, value_name):
        if False:
            while True:
                i = 10
        return '%s == NULL' % value_name

    @classmethod
    def hasErrorIndicator(cls):
        if False:
            while True:
                i = 10
        return True

    @classmethod
    def getReleaseCode(cls, value_name, needs_check, emit):
        if False:
            print('Hello World!')
        if needs_check:
            template = template_release_object_unclear
        else:
            template = template_release_object_clear
        emit(template % {'identifier': value_name})

    @classmethod
    def getTakeReferenceCode(cls, value_name, emit):
        if False:
            while True:
                i = 10
        'Take reference code for given object.'
        emit('Py_INCREF(%s);' % value_name)

class CTypePyObjectPtrPtr(CPythonPyObjectPtrBase):
    c_type = 'PyObject **'

    @classmethod
    def getInitTestConditionCode(cls, value_name, inverted):
        if False:
            print('Hello World!')
        return '*%s %s NULL' % (value_name, '==' if inverted else '!=')

    @classmethod
    def getVariableArgDeclarationCode(cls, variable_code_name):
        if False:
            return 10
        return 'PyObject **%s' % variable_code_name

    @classmethod
    def getVariableArgReferencePassingCode(cls, variable_code_name):
        if False:
            i = 10
            return i + 15
        return variable_code_name

    @classmethod
    def emitValueAccessCode(cls, value_name, emit, context):
        if False:
            return 10
        from ..VariableDeclarations import VariableDeclaration
        return VariableDeclaration('PyObject *', '*%s' % value_name, None, None)

    @classmethod
    def emitAssignmentCodeFromBoolCondition(cls, to_name, condition, emit):
        if False:
            for i in range(10):
                print('nop')
        emit('*%(to_name)s = (%(condition)s) ? Py_True : Py_False;' % {'to_name': to_name, 'condition': condition})

class CTypeCellObject(CTypeBase):
    c_type = 'struct Nuitka_CellObject *'

    @classmethod
    def getInitValue(cls, init_from):
        if False:
            return 10
        if init_from is not None:
            return 'Nuitka_Cell_New1(%s)' % init_from
        else:
            return 'Nuitka_Cell_Empty()'

    @classmethod
    def getInitTestConditionCode(cls, value_name, inverted):
        if False:
            return 10
        return '%s->ob_ref %s NULL' % (value_name, '==' if inverted else '!=')

    @classmethod
    def getCellObjectAssignmentCode(cls, target_cell_code, variable_code_name, emit):
        if False:
            return 10
        emit('%s = %s;' % (target_cell_code, variable_code_name))
        emit('Py_INCREF(%s);' % target_cell_code)

    @classmethod
    def emitVariableAssignCode(cls, value_name, needs_release, tmp_name, ref_count, inplace, emit, context):
        if False:
            i = 10
            return i + 15
        if inplace:
            template = template_write_shared_inplace
        elif ref_count:
            if needs_release is False:
                template = template_write_shared_clear_ref0
            else:
                template = template_write_shared_unclear_ref0
        elif needs_release is False:
            template = template_write_shared_clear_ref1
        else:
            template = template_write_shared_unclear_ref1
        emit(template % {'identifier': value_name, 'tmp_name': tmp_name})

    @classmethod
    def emitValueAccessCode(cls, value_name, emit, context):
        if False:
            return 10
        from ..VariableDeclarations import VariableDeclaration
        return VariableDeclaration('PyObject *', 'Nuitka_Cell_GET(%s)' % value_name, None, None)

    @classmethod
    def getVariableArgDeclarationCode(cls, variable_code_name):
        if False:
            for i in range(10):
                print('nop')
        return 'struct Nuitka_CellObject *%s' % variable_code_name

    @classmethod
    def getVariableArgReferencePassingCode(cls, variable_code_name):
        if False:
            print('Hello World!')
        return variable_code_name

    @classmethod
    def emitAssignmentCodeFromBoolCondition(cls, to_name, condition, emit):
        if False:
            return 10
        emit('%(to_name)s->ob_ref = (%(condition)s) ? Py_True : Py_False;' % {'to_name': to_name, 'condition': condition})

    @classmethod
    def getDeleteObjectCode(cls, to_name, value_name, needs_check, tolerant, emit, context):
        if False:
            for i in range(10):
                print('nop')
        if not needs_check:
            emit(template_del_shared_known % {'identifier': value_name})
        elif tolerant:
            emit(template_del_shared_tolerant % {'identifier': value_name})
        else:
            emit(template_del_shared_intolerant % {'identifier': value_name, 'result': to_name})

    @classmethod
    def getReleaseCode(cls, value_name, needs_check, emit):
        if False:
            for i in range(10):
                print('nop')
        if needs_check:
            template = template_release_object_unclear
        else:
            template = template_release_object_clear
        emit(template % {'identifier': value_name})

    @classmethod
    def emitReinitCode(cls, value_name, emit):
        if False:
            print('Hello World!')
        emit('%s = NULL;' % value_name)

    @classmethod
    def emitValueAssertionCode(cls, value_name, emit):
        if False:
            i = 10
            return i + 15
        emit('CHECK_OBJECT(%s->ob_ref);' % value_name)

    @classmethod
    def emitReleaseAssertionCode(cls, value_name, emit):
        if False:
            for i in range(10):
                print('nop')
        emit('CHECK_OBJECT(%s);' % value_name)