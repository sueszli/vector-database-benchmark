""" Code to generate and interact with compiled module objects.

"""
from nuitka import Options
from nuitka.__past__ import iterItems
from nuitka.code_generation import Emission
from nuitka.Version import getNuitkaVersion, getNuitkaVersionYear
from .CodeHelpers import decideConversionCheckNeeded, generateStatementSequenceCode, withObjectCodeTemporaryAssignment
from .CodeObjectCodes import getCodeObjectsDeclCode, getCodeObjectsInitCode
from .Indentation import indented
from .templates.CodeTemplatesModules import template_global_copyright, template_module_body_template, template_module_exception_exit, template_module_external_entry_point, template_module_no_exception_exit
from .VariableCodes import getVariableReferenceCode

def getModuleAccessCode(context):
    if False:
        return 10
    return 'module_%s' % context.getModuleCodeName()

def getModuleCode(module, function_decl_codes, function_body_codes, module_const_blob_name, context):
    if False:
        return 10
    from .FunctionCodes import finalizeFunctionLocalVariables, setupFunctionLocalVariables
    setupFunctionLocalVariables(context=context, parameters=None, closure_variables=(), user_variables=module.getOutlineLocalVariables(), temp_variables=module.getTempVariables())
    module_codes = Emission.SourceCodeCollector()
    module = context.getOwner()
    module_body = module.subnode_body
    generateStatementSequenceCode(statement_sequence=module_body, emit=module_codes, allow_none=True, context=context)
    for (_identifier, code) in sorted(iterItems(context.getHelperCodes())):
        function_body_codes.append(code)
    for (_identifier, code) in sorted(iterItems(context.getDeclarations())):
        function_decl_codes.append(code)
    function_body_codes = '\n\n'.join(function_body_codes)
    function_decl_codes = '\n\n'.join(function_decl_codes)
    _cleanup = finalizeFunctionLocalVariables(context)
    module_identifier = module.getCodeName()
    if module_body is not None and module_body.mayRaiseException(BaseException):
        module_exit = template_module_exception_exit % {'module_identifier': module_identifier, 'is_top': 1 if module.isTopModule() else 0}
    else:
        module_exit = template_module_no_exception_exit
    local_var_inits = context.variable_storage.makeCFunctionLevelDeclarations()
    function_table_entries_decl = []
    for func_impl_identifier in context.getFunctionCreationInfos():
        function_table_entries_decl.append('%s,' % func_impl_identifier)
    module_name = module.getFullName()
    is_package = module.isCompiledPythonPackage()
    is_top = module.isTopModule()
    module_identifier = module.getCodeName()
    template = template_global_copyright + template_module_body_template
    if is_top == 1 and Options.shallMakeModule():
        template += template_module_external_entry_point
    module_code_objects_decl = getCodeObjectsDeclCode(context)
    module_code_objects_init = getCodeObjectsInitCode(context)
    is_dunder_main = module.isMainModule()
    dunder_main_package = context.getConstantCode(module.getRuntimePackageValue() if is_dunder_main else '')
    return template % {'module_name': module_name, 'version': getNuitkaVersion(), 'year': getNuitkaVersionYear(), 'is_top': 1 if module.isTopModule() else 0, 'is_dunder_main': 1 if is_dunder_main else 0, 'dunder_main_package': dunder_main_package, 'is_package': 1 if is_package else 0, 'module_identifier': module_identifier, 'module_functions_decl': function_decl_codes, 'module_functions_code': function_body_codes, 'module_function_table_entries': indented(function_table_entries_decl), 'temps_decl': indented(local_var_inits), 'module_code': indented(module_codes.codes), 'module_exit': module_exit, 'module_code_objects_decl': indented(module_code_objects_decl, 0), 'module_code_objects_init': indented(module_code_objects_init, 1), 'constants_count': context.getConstantsCount(), 'module_const_blob_name': module_const_blob_name}

def generateModuleAttributeFileCode(to_name, expression, emit, context):
    if False:
        print('Hello World!')
    with withObjectCodeTemporaryAssignment(to_name, 'module_fileattr_value', expression, emit, context) as result_name:
        emit('%s = module_filename_obj;' % result_name)

def generateModuleAttributeCode(to_name, expression, emit, context):
    if False:
        for i in range(10):
            print('nop')
    getVariableReferenceCode(to_name=to_name, variable=expression.getVariable(), variable_trace=None, needs_check=False, conversion_check=decideConversionCheckNeeded(to_name, expression), emit=emit, context=context)

def generateNuitkaLoaderCreationCode(to_name, expression, emit, context):
    if False:
        i = 10
        return i + 15
    with withObjectCodeTemporaryAssignment(to_name, 'nuitka_loader_value', expression, emit, context) as result_name:
        emit('%s = Nuitka_Loader_New(loader_entry);' % result_name)