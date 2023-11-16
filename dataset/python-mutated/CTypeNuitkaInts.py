""" CType classes for nuitka_ilong, a struct to represent long values.

"""
from nuitka.code_generation.templates.CodeTemplatesVariables import template_release_object_clear, template_release_object_unclear
from .CTypeBases import CTypeBase

class CTypeNuitkaIntOrLongStruct(CTypeBase):
    c_type = 'nuitka_ilong'
    helper_code = 'NILONG'

    @classmethod
    def emitVariableAssignCode(cls, value_name, needs_release, tmp_name, ref_count, inplace, emit, context):
        if False:
            i = 10
            return i + 15
        assert not inplace
        if tmp_name.c_type == 'nuitka_ilong':
            emit('%s = %s;' % (value_name, tmp_name))
            if ref_count:
                emit('/* REFCOUNT ? */')
        elif tmp_name.c_type == 'PyObject *':
            emit('%s.validity = NUITKA_ILONG_OBJECT_VALID;' % value_name)
            emit('%s.ilong_object = %s;' % (value_name, tmp_name))
            if ref_count:
                emit('/* REFCOUNT ? */')
        else:
            assert False, repr(tmp_name)

    @classmethod
    def emitVariantAssignmentCode(cls, int_name, value_name, int_value, emit, context):
        if False:
            for i in range(10):
                print('nop')
        if value_name is None:
            assert int_value is not None
            assert False
        elif int_value is None:
            emit('%s.validity = NUITKA_ILONG_OBJECT_VALID;' % int_name)
            emit('%s.ilong_object = %s;' % (int_name, value_name))
        else:
            emit('%s.validity = NUITKA_ILONG_BOTH_VALID;' % int_name)
            emit('%s.ilong_object = %s;' % (int_name, value_name))
            emit('%s.ilong_value = %s;' % (int_name, int_value))

    @classmethod
    def getTruthCheckCode(cls, value_name):
        if False:
            i = 10
            return i + 15
        return '%s != 0' % value_name

    @classmethod
    def emitValueAccessCode(cls, value_name, emit, context):
        if False:
            print('Hello World!')
        return value_name

    @classmethod
    def emitValueAssertionCode(cls, value_name, emit):
        if False:
            for i in range(10):
                print('nop')
        emit('assert(%s.validity != NUITKA_ILONG_UNASSIGNED);' % value_name)

    @classmethod
    def emitAssignConversionCode(cls, to_name, value_name, needs_check, emit, context):
        if False:
            for i in range(10):
                print('nop')
        if value_name.c_type == cls.c_type:
            emit('%s = %s;' % (to_name, value_name))
        else:
            value_name.getCType().emitAssignmentCodeToNuitkaIntOrLong(to_name=to_name, value_name=value_name, needs_check=needs_check, emit=emit, context=context)

    @classmethod
    def getInitValue(cls, init_from):
        if False:
            while True:
                i = 10
        if init_from is None:
            return '{NUITKA_ILONG_UNASSIGNED, NULL, 0}'
        else:
            assert False, init_from
            return init_from

    @classmethod
    def getInitTestConditionCode(cls, value_name, inverted):
        if False:
            print('Hello World!')
        return '%s.validity %s NUITKA_ILONG_UNASSIGNED' % (value_name, '==' if inverted else '!=')

    @classmethod
    def getReleaseCode(cls, value_name, needs_check, emit):
        if False:
            i = 10
            return i + 15
        emit('if ((%s.validity & NUITKA_ILONG_OBJECT_VALID) == NUITKA_ILONG_OBJECT_VALID) {' % value_name)
        if needs_check:
            template = template_release_object_unclear
        else:
            template = template_release_object_clear
        emit(template % {'identifier': '%s.ilong_object' % value_name})
        emit('}')

    @classmethod
    def getDeleteObjectCode(cls, to_name, value_name, needs_check, tolerant, emit, context):
        if False:
            for i in range(10):
                print('nop')
        assert False, 'TODO'
        if not needs_check:
            emit('%s = NUITKA_BOOL_UNASSIGNED;' % value_name)
        elif tolerant:
            emit('%s = NUITKA_BOOL_UNASSIGNED;' % value_name)
        else:
            emit('%s = %s == NUITKA_BOOL_UNASSIGNED;' % (to_name, value_name))
            emit('%s = NUITKA_BOOL_UNASSIGNED;' % value_name)

    @classmethod
    def emitAssignmentCodeFromBoolCondition(cls, to_name, condition, emit):
        if False:
            print('Hello World!')
        assert False, 'TODO'
        emit('%(to_name)s = (%(condition)s) ? NUITKA_BOOL_TRUE : NUITKA_BOOL_FALSE;' % {'to_name': to_name, 'condition': condition})