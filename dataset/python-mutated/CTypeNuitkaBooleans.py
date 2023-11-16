""" CType classes for nuitka_bool, an enum to represent True, False, unassigned.

"""
from nuitka.code_generation.ErrorCodes import getReleaseCode
from .CTypeBases import CTypeBase, CTypeNotReferenceCountedMixin

class CTypeNuitkaBoolEnum(CTypeNotReferenceCountedMixin, CTypeBase):
    c_type = 'nuitka_bool'
    helper_code = 'NBOOL'

    @classmethod
    def emitVariableAssignCode(cls, value_name, needs_release, tmp_name, ref_count, inplace, emit, context):
        if False:
            i = 10
            return i + 15
        assert not inplace
        if tmp_name.c_type == 'nuitka_bool':
            emit('%s = %s;' % (value_name, tmp_name))
        else:
            if tmp_name.c_type == 'PyObject *':
                test_code = '%s == Py_True' % tmp_name
            else:
                assert False, tmp_name
            cls.emitAssignmentCodeFromBoolCondition(to_name=value_name, condition=test_code, emit=emit)
            if ref_count:
                getReleaseCode(tmp_name, emit, context)

    @classmethod
    def emitAssignmentCodeToNuitkaIntOrLong(cls, to_name, value_name, needs_check, emit, context):
        if False:
            i = 10
            return i + 15
        assert False, to_name

    @classmethod
    def getTruthCheckCode(cls, value_name):
        if False:
            return 10
        return '%s == NUITKA_BOOL_TRUE' % value_name

    @classmethod
    def emitValueAccessCode(cls, value_name, emit, context):
        if False:
            for i in range(10):
                print('nop')
        return value_name

    @classmethod
    def emitValueAssertionCode(cls, value_name, emit):
        if False:
            while True:
                i = 10
        emit('assert(%s != NUITKA_BOOL_UNASSIGNED);' % value_name)

    @classmethod
    def emitAssignConversionCode(cls, to_name, value_name, needs_check, emit, context):
        if False:
            while True:
                i = 10
        if value_name.c_type == cls.c_type:
            emit('%s = %s;' % (to_name, value_name))
        else:
            value_name.getCType().emitAssignmentCodeToNuitkaBool(to_name=to_name, value_name=value_name, needs_check=needs_check, emit=emit, context=context)
            getReleaseCode(value_name, emit, context)

    @classmethod
    def emitAssignmentCodeFromConstant(cls, to_name, constant, may_escape, emit, context):
        if False:
            while True:
                i = 10
        emit('%s = %s;' % (to_name, 'NUITKA_BOOL_TRUE' if constant else 'NUITKA_BOOL_FALSE'))

    @classmethod
    def getInitValue(cls, init_from):
        if False:
            i = 10
            return i + 15
        if init_from is None:
            return 'NUITKA_BOOL_UNASSIGNED'
        else:
            assert False, init_from
            return init_from

    @classmethod
    def getInitTestConditionCode(cls, value_name, inverted):
        if False:
            while True:
                i = 10
        return '%s %s NUITKA_BOOL_UNASSIGNED' % (value_name, '==' if inverted else '!=')

    @classmethod
    def emitReinitCode(cls, value_name, emit):
        if False:
            i = 10
            return i + 15
        emit('%s = NUITKA_BOOL_UNASSIGNED;' % value_name)

    @classmethod
    def getDeleteObjectCode(cls, to_name, value_name, needs_check, tolerant, emit, context):
        if False:
            while True:
                i = 10
        if not needs_check:
            emit('%s = NUITKA_BOOL_UNASSIGNED;' % value_name)
        elif tolerant:
            emit('%s = NUITKA_BOOL_UNASSIGNED;' % value_name)
        else:
            emit('%s = %s != NUITKA_BOOL_UNASSIGNED;' % (to_name, value_name))
            emit('%s = NUITKA_BOOL_UNASSIGNED;' % value_name)

    @classmethod
    def emitAssignmentCodeFromBoolCondition(cls, to_name, condition, emit):
        if False:
            i = 10
            return i + 15
        emit('%(to_name)s = (%(condition)s) ? NUITKA_BOOL_TRUE : NUITKA_BOOL_FALSE;' % {'to_name': to_name, 'condition': condition})

    @classmethod
    def emitAssignInplaceNegatedValueCode(cls, to_name, needs_check, emit, context):
        if False:
            while True:
                i = 10
        cls.emitValueAssertionCode(to_name, emit=emit)
        emit('assert(%s != NUITKA_BOOL_EXCEPTION);' % to_name)
        cls.emitAssignmentCodeFromBoolCondition(to_name=to_name, condition='%s == NUITKA_BOOL_FALSE' % to_name, emit=emit)

    @classmethod
    def getExceptionCheckCondition(cls, value_name):
        if False:
            for i in range(10):
                print('nop')
        return '%s == NUITKA_BOOL_EXCEPTION' % value_name

    @classmethod
    def hasErrorIndicator(cls):
        if False:
            for i in range(10):
                print('nop')
        return True