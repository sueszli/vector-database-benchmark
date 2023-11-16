""" Codes for operations.

There are unary and binary operations. Many of them have specializations and
of course types could play into it. Then there is also the added difficulty of
in-place assignments, which have other operation variants.
"""
from nuitka.nodes.shapes.StandardShapes import tshape_unknown
from .BinaryOperationHelperDefinitions import getCodeNameForBinaryOperation, getNonSpecializedBinaryOperations, getSpecializedBinaryOperations
from .c_types.CTypeBooleans import CTypeBool
from .c_types.CTypeNuitkaBooleans import CTypeNuitkaBoolEnum
from .c_types.CTypeNuitkaVoids import CTypeNuitkaVoidEnum
from .c_types.CTypePyObjectPointers import CTypePyObjectPtr
from .c_types.CTypeVoids import CTypeVoid
from .CodeHelpers import generateChildExpressionsCode, generateExpressionCode, withObjectCodeTemporaryAssignment
from .CodeHelperSelection import selectCodeHelper
from .ErrorCodes import getErrorExitBoolCode, getErrorExitCode, getReleaseCodes, getTakeReferenceCode
from .ExpressionCTypeSelectionHelpers import decideExpressionCTypes

def generateOperationBinaryCode(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    _getBinaryOperationCode(to_name=to_name, operator=expression.getOperator(), inplace=expression.isInplaceSuspect(), needs_check=expression.mayRaiseExceptionOperation(), left=expression.subnode_left, right=expression.subnode_right, source_ref=expression.source_ref, emit=emit, context=context)

def generateOperationNotCode(to_name, expression, emit, context):
    if False:
        i = 10
        return i + 15
    (arg_name,) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    res_name = context.getIntResName()
    emit('%s = CHECK_IF_TRUE(%s);' % (res_name, arg_name))
    getErrorExitBoolCode(condition='%s == -1' % res_name, release_name=arg_name, needs_check=expression.subnode_operand.mayRaiseExceptionBool(BaseException), emit=emit, context=context)
    to_name.getCType().emitAssignmentCodeFromBoolCondition(to_name=to_name, condition='%s == 0' % res_name, emit=emit)

def generateOperationUnaryCode(to_name, expression, emit, context):
    if False:
        return 10
    (arg_name,) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    _getUnaryOperationCode(to_name=to_name, expression=expression, operator=expression.getOperator(), arg_name=arg_name, needs_check=expression.mayRaiseException(BaseException), emit=emit, context=context)

def _getBinaryOperationCode(to_name, operator, inplace, left, right, needs_check, source_ref, emit, context):
    if False:
        return 10
    (_unknown_types, needs_argument_swap, left_shape, right_shape, left_c_type, right_c_type) = decideExpressionCTypes(left=left, right=right, may_swap_arguments='never' if inplace else 'number' if operator in ('Add', 'Mult', 'BitOr', 'BitAnd', 'BitXor') else 'never')
    prefix = '%s_OPERATION_%s' % ('INPLACE' if operator[0] == 'I' else 'BINARY', getCodeNameForBinaryOperation(operator))
    specialized_helpers_set = getSpecializedBinaryOperations(operator)
    non_specialized_helpers_set = getNonSpecializedBinaryOperations(operator)
    report_missing = True
    helper_type = target_type = None if operator[0] == 'I' else to_name.getCType()
    if helper_type is not None:
        if needs_check and helper_type is not None:
            if helper_type is CTypeNuitkaVoidEnum:
                helper_type = CTypeNuitkaBoolEnum
                report_missing = False
        else:
            if helper_type is CTypeVoid:
                helper_type = CTypeNuitkaBoolEnum
            report_missing = False
    (helper_type, helper_function) = selectCodeHelper(prefix=prefix, specialized_helpers_set=specialized_helpers_set, non_specialized_helpers_set=non_specialized_helpers_set, result_type=helper_type, left_shape=left_shape, right_shape=right_shape, left_c_type=left_c_type, right_c_type=right_c_type, argument_swap=needs_argument_swap, report_missing=report_missing, source_ref=source_ref)
    if helper_function is None and target_type is CTypeBool:
        (helper_type, helper_function) = selectCodeHelper(prefix=prefix, specialized_helpers_set=specialized_helpers_set, non_specialized_helpers_set=non_specialized_helpers_set, result_type=CTypeNuitkaBoolEnum, left_shape=left_shape, right_shape=right_shape, left_c_type=left_c_type, right_c_type=right_c_type, argument_swap=needs_argument_swap, report_missing=True, source_ref=source_ref)
    if helper_function is None:
        left_c_type = CTypePyObjectPtr
        right_c_type = CTypePyObjectPtr
        (helper_type, helper_function) = selectCodeHelper(prefix=prefix, specialized_helpers_set=specialized_helpers_set, non_specialized_helpers_set=non_specialized_helpers_set, result_type=CTypePyObjectPtr if helper_type is not None else None, left_shape=tshape_unknown, right_shape=tshape_unknown, left_c_type=left_c_type, right_c_type=right_c_type, argument_swap=False, report_missing=True, source_ref=source_ref)
        assert helper_function is not None, (left, right)
    left_name = context.allocateTempName('%s_expr_left' % operator.lower(), type_name=left_c_type.c_type)
    right_name = context.allocateTempName('%s_expr_right' % operator.lower(), type_name=right_c_type.c_type)
    generateExpressionCode(to_name=left_name, expression=left, emit=emit, context=context)
    generateExpressionCode(to_name=right_name, expression=right, emit=emit, context=context)
    if inplace or 'INPLACE' in helper_function:
        assert not needs_argument_swap
        res_name = context.getBoolResName()
        if left.isExpressionVariableRef() and left.getVariable().isModuleVariable():
            emit('%s = %s;' % (context.getInplaceLeftName(), left_name))
        if not left.isExpressionVariableRef() and (not left.isExpressionTempVariableRef()):
            if not context.needsCleanup(left_name):
                getTakeReferenceCode(left_name, emit)
        emit('%s = %s(&%s, %s);' % (res_name, helper_function, left_name, right_name))
        getErrorExitBoolCode(condition='%s == false' % res_name, release_names=(left_name, right_name), needs_check=needs_check, emit=emit, context=context)
        emit('%s = %s;' % (to_name, left_name))
        if not left.isExpressionVariableRef() and (not left.isExpressionTempVariableRef()):
            context.addCleanupTempName(to_name)
    else:
        if needs_argument_swap:
            arg1_name = right_name
            arg2_name = left_name
        else:
            arg1_name = left_name
            arg2_name = right_name
        if helper_type is not target_type:
            value_name = context.allocateTempName(to_name.code_name + '_' + helper_type.helper_code.lower(), type_name=helper_type.c_type, unique=to_name.code_name == 'tmp_unused')
        else:
            value_name = to_name
        emit('%s = %s(%s, %s);' % (value_name, helper_function, arg1_name, arg2_name))
        if value_name.getCType().hasErrorIndicator():
            getErrorExitCode(check_name=value_name, release_names=(left_name, right_name), needs_check=needs_check, emit=emit, context=context)
        else:
            assert not needs_check, value_name.getCType()
            getReleaseCodes(release_names=(left_name, right_name), emit=emit, context=context)
        if helper_type is CTypePyObjectPtr:
            context.addCleanupTempName(value_name)
        if value_name is not to_name:
            target_type.emitAssignConversionCode(to_name=to_name, value_name=value_name, needs_check=False, emit=emit, context=context)
unary_operator_codes = {'UAdd': ('PyNumber_Positive', 1), 'USub': ('PyNumber_Negative', 1), 'Invert': ('PyNumber_Invert', 1), 'Repr': ('PyObject_Repr', 1), 'Not': ('UNARY_NOT', 0)}

def _getUnaryOperationCode(to_name, expression, operator, arg_name, needs_check, emit, context):
    if False:
        return 10
    (impl_helper, ref_count) = unary_operator_codes[operator]
    helper = 'UNARY_OPERATION'
    prefix_args = (impl_helper,)
    with withObjectCodeTemporaryAssignment(to_name, 'op_%s_res' % operator.lower(), expression, emit, context) as value_name:
        emit('%s = %s(%s);' % (value_name, helper, ', '.join((str(arg_name) for arg_name in prefix_args + (arg_name,)))))
        getErrorExitCode(check_name=value_name, release_name=arg_name, needs_check=needs_check, emit=emit, context=context)
        if ref_count:
            context.addCleanupTempName(value_name)