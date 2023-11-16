""" Code to generate and interact with compiled asyncgen objects.

"""
from .CodeHelpers import generateStatementSequenceCode
from .Emission import SourceCodeCollector
from .FunctionCodes import finalizeFunctionLocalVariables, getClosureCopyCode, getFunctionCreationArgs, getFunctionQualnameObj, setupFunctionLocalVariables
from .Indentation import indented
from .ModuleCodes import getModuleAccessCode
from .templates.CodeTemplatesAsyncgens import template_asyncgen_exception_exit, template_asyncgen_noexception_exit, template_asyncgen_object_body, template_asyncgen_object_maker_template, template_asyncgen_return_exit, template_make_asyncgen
from .YieldCodes import getYieldReturnDispatchCode

def _getAsyncgenMakerIdentifier(function_identifier):
    if False:
        for i in range(10):
            print('nop')
    return 'MAKE_ASYNCGEN_' + function_identifier

def getAsyncgenObjectDeclCode(function_identifier, closure_variables):
    if False:
        return 10
    asyncgen_creation_args = getFunctionCreationArgs(defaults_name=None, kw_defaults_name=None, annotations_name=None, closure_variables=closure_variables, tstate=False)
    return template_asyncgen_object_maker_template % {'asyncgen_maker_identifier': _getAsyncgenMakerIdentifier(function_identifier), 'asyncgen_creation_args': ', '.join(asyncgen_creation_args)}

def getAsyncgenObjectCode(context, function_identifier, closure_variables, user_variables, outline_variables, temp_variables, needs_exception_exit, needs_generator_return):
    if False:
        while True:
            i = 10
    setupFunctionLocalVariables(context=context, parameters=None, closure_variables=closure_variables, user_variables=user_variables + outline_variables, temp_variables=temp_variables)
    function_codes = SourceCodeCollector()
    asyncgen_object_body = context.getOwner()
    generateStatementSequenceCode(statement_sequence=asyncgen_object_body.subnode_body, allow_none=True, emit=function_codes, context=context)
    function_cleanup = finalizeFunctionLocalVariables(context)
    if needs_exception_exit:
        (exception_type, exception_value, exception_tb, _exception_lineno) = context.variable_storage.getExceptionVariableDescriptions()
        generator_exit = template_asyncgen_exception_exit % {'function_cleanup': indented(function_cleanup), 'exception_type': exception_type, 'exception_value': exception_value, 'exception_tb': exception_tb}
    else:
        generator_exit = template_asyncgen_noexception_exit % {'function_cleanup': indented(function_cleanup)}
    if needs_generator_return:
        generator_exit += template_asyncgen_return_exit % {}
    function_locals = context.variable_storage.makeCFunctionLevelDeclarations()
    local_type_decl = context.variable_storage.makeCStructLevelDeclarations()
    function_locals += context.variable_storage.makeCStructInits()
    if local_type_decl:
        heap_declaration = 'struct %(function_identifier)s_locals *asyncgen_heap = (struct %(function_identifier)s_locals *)asyncgen->m_heap_storage;' % {'function_identifier': function_identifier}
    else:
        heap_declaration = ''
    asyncgen_creation_args = getFunctionCreationArgs(defaults_name=None, kw_defaults_name=None, annotations_name=None, closure_variables=closure_variables, tstate=False)
    return template_asyncgen_object_body % {'function_identifier': function_identifier, 'function_body': indented(function_codes.codes), 'heap_declaration': indented(heap_declaration), 'has_heap_declaration': 1 if heap_declaration != '' else 0, 'function_local_types': indented(local_type_decl), 'function_var_inits': indented(function_locals), 'function_dispatch': indented(getYieldReturnDispatchCode(context)), 'asyncgen_maker_identifier': _getAsyncgenMakerIdentifier(function_identifier), 'asyncgen_creation_args': ', '.join(asyncgen_creation_args), 'asyncgen_exit': generator_exit, 'asyncgen_module': getModuleAccessCode(context), 'asyncgen_name_obj': context.getConstantCode(constant=asyncgen_object_body.getFunctionName()), 'asyncgen_qualname_obj': getFunctionQualnameObj(asyncgen_object_body, context), 'code_identifier': context.getCodeObjectHandle(code_object=asyncgen_object_body.getCodeObject()), 'closure_name': 'closure' if closure_variables else 'NULL', 'closure_count': len(closure_variables)}

def generateMakeAsyncgenObjectCode(to_name, expression, emit, context):
    if False:
        i = 10
        return i + 15
    asyncgen_object_body = expression.subnode_asyncgen_ref.getFunctionBody()
    closure_variables = expression.getClosureVariableVersions()
    (closure_name, closure_copy) = getClosureCopyCode(closure_variables=closure_variables, context=context)
    args = []
    if closure_name:
        args.append(closure_name)
    emit(template_make_asyncgen % {'to_name': to_name, 'asyncgen_maker_identifier': _getAsyncgenMakerIdentifier(asyncgen_object_body.getCodeName()), 'args': ', '.join((str(arg) for arg in args)), 'closure_copy': indented(closure_copy, 0, True)})
    context.addCleanupTempName(to_name)