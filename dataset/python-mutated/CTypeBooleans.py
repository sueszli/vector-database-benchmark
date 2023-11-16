""" CType classes for C bool, this cannot represent unassigned, nor indicate exception.

"""
from .CTypeBases import CTypeBase, CTypeNotReferenceCountedMixin

class CTypeBool(CTypeNotReferenceCountedMixin, CTypeBase):
    c_type = 'bool'
    helper_code = 'CBOOL'

    @classmethod
    def emitValueAccessCode(cls, value_name, emit, context):
        if False:
            return 10
        assert False

    @classmethod
    def emitValueAssertionCode(cls, value_name, emit):
        if False:
            for i in range(10):
                print('nop')
        pass

    @classmethod
    def emitAssignConversionCode(cls, to_name, value_name, needs_check, emit, context):
        if False:
            i = 10
            return i + 15
        if value_name.c_type == cls.c_type:
            emit('%s = %s;' % (to_name, value_name))
        else:
            emit('%s = %s;' % (to_name, value_name.getCType().getTruthCheckCode(value_name=value_name)))

    @classmethod
    def emitAssignmentCodeFromConstant(cls, to_name, constant, may_escape, emit, context):
        if False:
            while True:
                i = 10
        emit('%s = %s;' % (to_name, 'true' if constant else 'false'))

    @classmethod
    def getInitValue(cls, init_from):
        if False:
            print('Hello World!')
        return '<not_possible>'

    @classmethod
    def getInitTestConditionCode(cls, value_name, inverted):
        if False:
            print('Hello World!')
        return '<not_possible>'

    @classmethod
    def getDeleteObjectCode(cls, to_name, value_name, needs_check, tolerant, emit, context):
        if False:
            return 10
        assert False

    @classmethod
    def emitAssignmentCodeToNuitkaBool(cls, to_name, value_name, needs_check, emit, context):
        if False:
            print('Hello World!')
        emit('%s = %s ? NUITKA_BOOL_TRUE : NUITKA_BOOL_FALSE;' % (to_name, value_name))

    @classmethod
    def emitAssignmentCodeFromBoolCondition(cls, to_name, condition, emit):
        if False:
            return 10
        emit('%s = (%s) ? true : false;' % (to_name, condition))

    @classmethod
    def emitAssignInplaceNegatedValueCode(cls, to_name, needs_check, emit, context):
        if False:
            return 10
        emit('%s = !%s;' % (to_name, to_name))

    @classmethod
    def getExceptionCheckCondition(cls, value_name):
        if False:
            while True:
                i = 10
        assert False

    @classmethod
    def hasErrorIndicator(cls):
        if False:
            return 10
        return False

    @classmethod
    def getTruthCheckCode(cls, value_name):
        if False:
            while True:
                i = 10
        return '%s != false' % value_name