""" CType enum class for void, a special value to represent discarding stuff.

Cannot be read from obviously. Also drops references immediately when trying
to assign to it, but allows to check for exception.
"""
from nuitka import Options
from nuitka.code_generation.ErrorCodes import getReleaseCode
from .CTypeBases import CTypeBase, CTypeNotReferenceCountedMixin

class CTypeNuitkaVoidEnum(CTypeNotReferenceCountedMixin, CTypeBase):
    c_type = 'nuitka_void'
    helper_code = 'NVOID'

    @classmethod
    def emitValueAccessCode(cls, value_name, emit, context):
        if False:
            print('Hello World!')
        assert False

    @classmethod
    def emitValueAssertionCode(cls, value_name, emit):
        if False:
            return 10
        emit('assert(%s == NUITKA_VOID_OK);' % value_name)

    @classmethod
    def emitReinitCode(cls, value_name, emit):
        if False:
            print('Hello World!')
        emit('%s = NUITKA_VOID_OK;' % value_name)

    @classmethod
    def emitAssignConversionCode(cls, to_name, value_name, needs_check, emit, context):
        if False:
            while True:
                i = 10
        getReleaseCode(value_name, emit, context)
        if Options.is_debug:
            emit('%s = NUITKA_VOID_OK;' % to_name)

    @classmethod
    def emitAssignmentCodeFromConstant(cls, to_name, constant, may_escape, emit, context):
        if False:
            print('Hello World!')
        assert constant is None
        if Options.is_debug:
            emit('%s = NUITKA_VOID_OK;' % to_name)

    @classmethod
    def getInitValue(cls, init_from):
        if False:
            i = 10
            return i + 15
        assert False

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
        if Options.is_debug:
            emit('%s = NUITKA_VOID_OK;' % to_name)

    @classmethod
    def emitAssignInplaceNegatedValueCode(cls, to_name, needs_check, emit, context):
        if False:
            i = 10
            return i + 15
        if Options.is_debug:
            emit('%s = NUITKA_VOID_OK;' % to_name)

    @classmethod
    def getExceptionCheckCondition(cls, value_name):
        if False:
            i = 10
            return i + 15
        return '%s == NUITKA_VOID_EXCEPTION' % value_name

    @classmethod
    def hasErrorIndicator(cls):
        if False:
            while True:
                i = 10
        return True