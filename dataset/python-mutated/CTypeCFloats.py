""" CType classes for C "float" (double), (used in conjunction with PyFloatObject *)

"""
from math import copysign, isinf, isnan
from .CTypeBases import CTypeBase

class CTypeCFloat(CTypeBase):
    c_type = 'double'
    helper_code = 'CFLOAT'

    @classmethod
    def emitAssignmentCodeFromConstant(cls, to_name, constant, may_escape, emit, context):
        if False:
            while True:
                i = 10
        if constant == 0.0:
            if copysign(1, constant) == 1:
                c_constant = '0.0'
            else:
                c_constant = '-0.0'
        elif isnan(constant):
            if copysign(1, constant) == 1:
                c_constant = 'NAN'
            else:
                c_constant = '-NAN'
        elif isinf(constant):
            if copysign(1, constant) == 1:
                c_constant = 'HUGE_VAL'
            else:
                c_constant = '-HUGE_VAL'
        else:
            c_constant = constant
        emit('%s = %s;' % (to_name, c_constant))