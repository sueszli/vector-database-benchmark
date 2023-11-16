""" CType classes for C void, this cannot represent unassigned, nor indicate exception.

"""
from nuitka.code_generation.ErrorCodes import getReleaseCode
from .CTypeBases import CTypeBase, CTypeNotReferenceCountedMixin

class CTypeVoid(CTypeNotReferenceCountedMixin, CTypeBase):
    c_type = 'bool'
    helper_code = 'CVOID'

    @classmethod
    def emitValueAccessCode(cls, value_name, emit, context):
        if False:
            i = 10
            return i + 15
        assert False

    @classmethod
    def emitValueAssertionCode(cls, value_name, emit):
        if False:
            print('Hello World!')
        pass

    @classmethod
    def emitAssignConversionCode(cls, to_name, value_name, needs_check, emit, context):
        if False:
            return 10
        getReleaseCode(value_name, emit, context)

    @classmethod
    def emitAssignInplaceNegatedValueCode(cls, to_name, needs_check, emit, context):
        if False:
            for i in range(10):
                print('nop')
        pass

    @classmethod
    def emitAssignmentCodeFromConstant(cls, to_name, constant, may_escape, emit, context):
        if False:
            i = 10
            return i + 15
        assert False

    @classmethod
    def getInitValue(cls, init_from):
        if False:
            return 10
        return '<not_possible>'

    @classmethod
    def getInitTestConditionCode(cls, value_name, inverted):
        if False:
            print('Hello World!')
        return '<not_possible>'

    @classmethod
    def getDeleteObjectCode(cls, to_name, value_name, needs_check, tolerant, emit, context):
        if False:
            while True:
                i = 10
        assert False

    @classmethod
    def emitAssignmentCodeFromBoolCondition(cls, to_name, condition, emit):
        if False:
            for i in range(10):
                print('nop')
        assert False

    @classmethod
    def getExceptionCheckCondition(cls, value_name):
        if False:
            i = 10
            return i + 15
        assert False

    @classmethod
    def hasErrorIndicator(cls):
        if False:
            print('Hello World!')
        return False

    @classmethod
    def getTruthCheckCode(cls, value_name):
        if False:
            while True:
                i = 10
        assert False