""" Error codes

These are the helper functions that will emit the error exit codes. They
can abstractly check conditions or values directly. The release of statement
temporaries from context is automatic.

Also formatting errors is done here, avoiding PyErr_Format as much as
possible.

And releasing of values, as this is what the error case commonly does.

"""
from nuitka.PythonVersions import python_version
from .ExceptionCodes import getExceptionIdentifier
from .Indentation import indented
from .LineNumberCodes import getErrorLineNumberUpdateCode
from .templates.CodeTemplatesExceptions import template_error_catch_exception, template_error_catch_quick_exception, template_error_format_name_error_exception, template_error_format_string_exception

def getErrorExitReleaseCode(context):
    if False:
        print('Hello World!')
    temp_release = '\n'.join(('Py_DECREF(%s);' % tmp_name for tmp_name in context.getCleanupTempNames()))
    keeper_variables = context.getExceptionKeeperVariables()
    if keeper_variables[0] is not None:
        temp_release += '\nPy_DECREF(%s);' % keeper_variables[0]
        temp_release += '\nPy_XDECREF(%s);' % keeper_variables[1]
        temp_release += '\nPy_XDECREF(%s);' % keeper_variables[2]
    return temp_release

def getFrameVariableTypeDescriptionCode(context):
    if False:
        print('Hello World!')
    type_description = context.getFrameVariableTypeDescription()
    if type_description:
        return '%s = "%s";' % (context.getFrameTypeDescriptionDeclaration(), type_description)
    else:
        return ''

def getErrorExitBoolCode(condition, emit, context, release_names=(), release_name=None, needs_check=True, quick_exception=None):
    if False:
        for i in range(10):
            print('nop')
    assert not condition.endswith(';')
    if release_names:
        getReleaseCodes(release_names, emit, context)
        assert not release_name
    if release_name is not None:
        assert type(release_name) is not tuple
        getReleaseCode(release_name, emit, context)
        assert not release_names
    if not needs_check:
        getAssertionCode('!(%s)' % condition, emit)
        return
    (exception_type, exception_value, exception_tb, _exception_lineno) = context.variable_storage.getExceptionVariableDescriptions()
    if quick_exception:
        emit(indented(template_error_catch_quick_exception % {'condition': condition, 'exception_type': exception_type, 'exception_value': exception_value, 'exception_tb': exception_tb, 'exception_exit': context.getExceptionEscape(), 'quick_exception': getExceptionIdentifier(quick_exception), 'release_temps': indented(getErrorExitReleaseCode(context)), 'var_description_code': indented(getFrameVariableTypeDescriptionCode(context)), 'line_number_code': indented(getErrorLineNumberUpdateCode(context))}, 0))
    else:
        emit(indented(template_error_catch_exception % {'condition': condition, 'exception_type': exception_type, 'exception_value': exception_value, 'exception_tb': exception_tb, 'exception_exit': context.getExceptionEscape(), 'release_temps': indented(getErrorExitReleaseCode(context)), 'var_description_code': indented(getFrameVariableTypeDescriptionCode(context)), 'line_number_code': indented(getErrorLineNumberUpdateCode(context))}, 0))

def getErrorExitCode(check_name, emit, context, release_names=(), release_name=None, quick_exception=None, needs_check=True):
    if False:
        for i in range(10):
            print('nop')
    getErrorExitBoolCode(condition=check_name.getCType().getExceptionCheckCondition(check_name), release_names=release_names, release_name=release_name, needs_check=needs_check, quick_exception=quick_exception, emit=emit, context=context)

def _getExceptionChainingCode(context):
    if False:
        print('Hello World!')
    (exception_type, exception_value, exception_tb, _exception_lineno) = context.variable_storage.getExceptionVariableDescriptions()
    keeper_vars = context.getExceptionKeeperVariables()
    if keeper_vars[0] is not None:
        return ('ADD_EXCEPTION_CONTEXT(tstate, &%s, &%s);' % (keeper_vars[0], keeper_vars[1]),)
    else:
        return ('NORMALIZE_EXCEPTION(tstate, &%s, &%s, &%s);' % (exception_type, exception_value, exception_tb), 'CHAIN_EXCEPTION(tstate, %s);' % exception_value)

def getTakeReferenceCode(value_name, emit):
    if False:
        return 10
    value_name.getCType().getTakeReferenceCode(value_name=value_name, emit=emit)

def getReleaseCode(release_name, emit, context):
    if False:
        print('Hello World!')
    if context.needsCleanup(release_name):
        release_name.getCType().getReleaseCode(value_name=release_name, needs_check=False, emit=emit)
        context.removeCleanupTempName(release_name)

def getReleaseCodes(release_names, emit, context):
    if False:
        i = 10
        return i + 15
    for release_name in release_names:
        getReleaseCode(release_name=release_name, emit=emit, context=context)

def getMustNotGetHereCode(reason, emit):
    if False:
        return 10
    emit('NUITKA_CANNOT_GET_HERE("%s");\nreturn NULL;' % reason)

def getAssertionCode(check, emit):
    if False:
        i = 10
        return i + 15
    emit('assert(%s);' % check)

def getLocalVariableReferenceErrorCode(variable, condition, emit, context):
    if False:
        while True:
            i = 10
    variable_name = variable.getName()
    (exception_type, exception_value, exception_tb, _exception_lineno) = context.variable_storage.getExceptionVariableDescriptions()
    if variable.getOwner() is not context.getOwner():
        helper_code = 'FORMAT_UNBOUND_CLOSURE_ERROR'
    else:
        helper_code = 'FORMAT_UNBOUND_LOCAL_ERROR'
    set_exception = ['%s(&%s, &%s, %s);' % (helper_code, exception_type, exception_value, context.getConstantCode(variable_name)), '%s = NULL;' % exception_tb]
    if python_version >= 768:
        set_exception.extend(_getExceptionChainingCode(context))
    emit(template_error_format_string_exception % {'condition': condition, 'exception_exit': context.getExceptionEscape(), 'set_exception': indented(set_exception), 'release_temps': indented(getErrorExitReleaseCode(context)), 'var_description_code': indented(getFrameVariableTypeDescriptionCode(context)), 'line_number_code': indented(getErrorLineNumberUpdateCode(context))})

def getNameReferenceErrorCode(variable_name, condition, emit, context):
    if False:
        print('Hello World!')
    helper_code = 'FORMAT_NAME_ERROR'
    if python_version < 832:
        owner = context.getOwner()
        if not owner.isCompiledPythonModule() and (not owner.isExpressionClassBodyBase()):
            helper_code = 'FORMAT_GLOBAL_NAME_ERROR'
    (exception_type, exception_value, _exception_tb, _exception_lineno) = context.variable_storage.getExceptionVariableDescriptions()
    set_exception = '%s(&%s, &%s, %s);' % (helper_code, exception_type, exception_value, context.getConstantCode(variable_name))
    if python_version >= 768:
        set_exception = [set_exception]
        set_exception.extend(_getExceptionChainingCode(context))
    emit(template_error_format_name_error_exception % {'condition': condition, 'exception_exit': context.getExceptionEscape(), 'set_exception': indented(set_exception), 'release_temps': indented(getErrorExitReleaseCode(context)), 'var_description_code': indented(getFrameVariableTypeDescriptionCode(context)), 'line_number_code': indented(getErrorLineNumberUpdateCode(context))})