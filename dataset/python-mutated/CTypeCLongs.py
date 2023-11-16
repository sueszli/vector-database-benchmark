""" CType classes for C "long", and C "digit" (used in conjunction with PyLongObject *)

"""
from .CTypeBases import CTypeBase

class CTypeCLongMixin(CTypeBase):

    @classmethod
    def emitAssignmentCodeFromConstant(cls, to_name, constant, may_escape, emit, context):
        if False:
            return 10
        emit('%s = %s;' % (to_name, constant))

class CTypeCLong(CTypeCLongMixin, CTypeBase):
    c_type = 'long'
    helper_code = 'CLONG'

class CTypeCLongDigit(CTypeCLongMixin, CTypeBase):
    c_type = 'nuitka_digit'
    helper_code = 'DIGIT'