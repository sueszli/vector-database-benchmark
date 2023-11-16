""" Import related codes.

That is import as expression, and star import.
"""
import os
from nuitka.HardImportRegistry import isHardModule, isHardModuleDynamic
from nuitka.nodes.LocalsScopes import GlobalsDictHandle
from nuitka.PythonVersions import python_version
from nuitka.utils.Jinja2 import renderTemplateFromString
from nuitka.utils.ModuleNames import ModuleName
from .CodeHelpers import generateChildExpressionsCode, generateExpressionCode, withObjectCodeTemporaryAssignment
from .ErrorCodes import getErrorExitBoolCode, getErrorExitCode
from .LineNumberCodes import emitLineNumberUpdateCode
from .ModuleCodes import getModuleAccessCode

def generateBuiltinImportCode(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    (module_name, globals_name, locals_name, import_list_name, level_name) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    with withObjectCodeTemporaryAssignment(to_name, 'imported_value', expression, emit, context) as value_name:
        _getBuiltinImportCode(expression=expression, to_name=value_name, module_name=module_name, globals_name=globals_name, locals_name=locals_name, import_list_name=import_list_name, level_name=level_name, needs_check=expression.mayRaiseException(BaseException), emit=emit, context=context)

def _getCountedArgumentsHelperCallCode(helper_prefix, to_name, args, min_args, needs_check, emit, context):
    if False:
        for i in range(10):
            print('nop')
    orig_args = args
    args = list(args)
    while args[-1] is None:
        del args[-1]
    if None in args:
        emit('%s = %s_KW(tstate, %s);' % (to_name, helper_prefix, ', '.join(('NULL' if arg is None else str(arg) for arg in orig_args))))
    else:
        assert len(args) >= min_args
        emit('%s = %s%d(tstate, %s);' % (to_name, helper_prefix, len(args), ', '.join((str(arg) for arg in args))))
    getErrorExitCode(check_name=to_name, release_names=args, needs_check=needs_check, emit=emit, context=context)
    context.addCleanupTempName(to_name)

def _getBuiltinImportCode(expression, to_name, module_name, globals_name, locals_name, import_list_name, level_name, needs_check, emit, context):
    if False:
        print('Hello World!')
    emitLineNumberUpdateCode(expression, emit, context)
    _getCountedArgumentsHelperCallCode(helper_prefix='IMPORT_MODULE', to_name=to_name, args=(module_name, globals_name, locals_name, import_list_name, level_name), min_args=1, needs_check=needs_check, emit=emit, context=context)

def generateImportModuleFixedCode(to_name, expression, emit, context):
    if False:
        for i in range(10):
            print('nop')
    needs_check = expression.mayRaiseException(BaseException)
    if needs_check:
        emitLineNumberUpdateCode(expression, emit, context)
    with withObjectCodeTemporaryAssignment(to_name, 'imported_value', expression, emit, context) as value_name:
        emit('%s = IMPORT_MODULE_FIXED(tstate, %s, %s);' % (value_name, context.getConstantCode(expression.getModuleName().asString()), context.getConstantCode(expression.getValueName().asString())))
        getErrorExitCode(check_name=value_name, needs_check=needs_check, emit=emit, context=context)
        context.addCleanupTempName(value_name)

def getImportModuleHardCodeName(module_name):
    if False:
        i = 10
        return i + 15
    'Encoding hard module name for code name.'
    module_name = ModuleName(module_name)
    return 'IMPORT_HARD_%s' % module_name.asPath().replace(os.path.sep, '__').upper()

def generateImportModuleHardCode(to_name, expression, emit, context):
    if False:
        for i in range(10):
            print('nop')
    imported_module_name = expression.getModuleName()
    module_value_name = expression.getValueName()
    needs_check = expression.mayRaiseException(BaseException)
    if needs_check:
        emitLineNumberUpdateCode(expression, emit, context)
    with withObjectCodeTemporaryAssignment(to_name, 'imported_value', expression, emit, context) as value_name:
        (import_gives_ref, module_getter_code) = getImportHardModuleGetterCode(module_name=imported_module_name, context=context)
        if imported_module_name == module_value_name:
            emit('%s = %s;' % (value_name, module_getter_code))
        else:
            (import_gives_ref, module_getter_code2) = getImportHardModuleGetterCode(module_name=module_value_name, context=context)
            emit(renderTemplateFromString('\n{\n    PyObject *hard_module = {{module_getter_code1}};\n{% if import_gives_ref %}\n    Py_DECREF(hard_module);\n{% endif %}\n}\n{{value_name}} = {{module_getter_code2}};\n', value_name=value_name, module_getter_code1=module_getter_code, module_getter_code2=module_getter_code2, import_gives_ref=import_gives_ref))
        getErrorExitCode(check_name=value_name, needs_check=needs_check, emit=emit, context=context)
        if import_gives_ref:
            context.addCleanupTempName(value_name)

def generateConstantSysVersionInfoCode(to_name, expression, emit, context):
    if False:
        return 10
    with withObjectCodeTemporaryAssignment(to_name, 'imported_value', expression, emit, context) as value_name:
        emit('%s = Nuitka_SysGetObject("%s");' % (value_name, 'version_info'))
    getErrorExitCode(check_name=value_name, needs_check=False, emit=emit, context=context)

def getImportHardModuleGetterCode(module_name, context):
    if False:
        while True:
            i = 10
    if isHardModuleDynamic(module_name):
        module_name_code = context.getConstantCode(module_name.asString())
        module_getter_code = 'IMPORT_MODULE_FIXED(tstate, %s, %s)' % (module_name_code, module_name_code)
        gives_ref = True
    else:
        module_getter_code = '%s()' % getImportModuleHardCodeName(module_name)
        gives_ref = False
    return (gives_ref, module_getter_code)

def getImportModuleNameHardCode(to_name, module_name, import_name, needs_check, emit, context):
    if False:
        while True:
            i = 10
    if module_name == 'sys':
        emit('%s = Nuitka_SysGetObject("%s");' % (to_name, import_name))
        needs_release = False
    elif isHardModule(module_name):
        if needs_check:
            emitLineNumberUpdateCode(expression=None, emit=emit, context=context)
        (import_gives_ref, module_getter_code) = getImportHardModuleGetterCode(module_name=module_name, context=context)
        emit(renderTemplateFromString('\n{\n    PyObject *hard_module = {{module_getter_code}};\n{% if needs_check %}\n    if (likely(hard_module != NULL)) {\n        {{to_name}} = LOOKUP_ATTRIBUTE(tstate, hard_module, {{import_name}});\n\n{% if import_gives_ref %}\n        Py_DECREF(hard_module);\n{% endif %}\n\n    } else {\n        {{to_name}} = NULL;\n    }\n{% else %}\n    {{to_name}} = LOOKUP_ATTRIBUTE(tstate, hard_module, {{import_name}});\n{% if import_gives_ref %}\n    Py_DECREF(hard_module);\n{% endif %}\n{% endif %}\n}\n', to_name=to_name, module_name=str(module_name), module_getter_code=module_getter_code, import_name=context.getConstantCode(import_name), needs_check=needs_check, import_gives_ref=import_gives_ref))
        needs_release = True
    else:
        assert False, module_name
    getErrorExitCode(check_name=to_name, needs_check=needs_check, emit=emit, context=context)
    if needs_release:
        context.addCleanupTempName(to_name)

def generateImportModuleNameHardCode(to_name, expression, emit, context):
    if False:
        print('Hello World!')
    with withObjectCodeTemporaryAssignment(to_name, 'imported_value', expression, emit, context) as value_name:
        context.setCurrentSourceCodeReference(expression.getCompatibleSourceReference())
        getImportModuleNameHardCode(to_name=value_name, module_name=expression.getModuleName(), import_name=expression.getImportName(), needs_check=expression.mayRaiseException(BaseException), emit=emit, context=context)

def generateImportlibImportCallCode(to_name, expression, emit, context):
    if False:
        i = 10
        return i + 15
    needs_check = expression.mayRaiseException(BaseException)
    with withObjectCodeTemporaryAssignment(to_name, 'imported_module', expression, emit, context) as value_name:
        (import_name, package_name) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
        emitLineNumberUpdateCode(expression, emit, context)
        emit(renderTemplateFromString('\n{\n    PyObject *hard_module = {{import_hard_importlib}}();\n    PyObject *import_module_func = LOOKUP_ATTRIBUTE(tstate, hard_module, {{context.getConstantCode("import_module")}});\n{% if package_name == None %}\n    {{to_name}} = CALL_FUNCTION_WITH_SINGLE_ARG(tstate, import_module_func, {{import_name}});\n{% else %}\n    PyObject *args[2] = { {{import_name}}, {{package_name}} };\n    {{to_name}} = CALL_FUNCTION_WITH_ARGS2(tstate, import_module_func, args);\n{% endif %}\n    Py_DECREF(import_module_func);\n}\n', context=context, to_name=value_name, import_name=import_name, package_name=package_name, import_hard_importlib=getImportModuleHardCodeName('importlib')))
        getErrorExitCode(check_name=value_name, release_names=(import_name, package_name), needs_check=needs_check, emit=emit, context=context)

def generateImportStarCode(statement, emit, context):
    if False:
        while True:
            i = 10
    module_name = context.allocateTempName('star_imported')
    generateExpressionCode(to_name=module_name, expression=statement.subnode_module, emit=emit, context=context)
    with context.withCurrentSourceCodeReference(statement.getSourceReference()):
        res_name = context.getBoolResName()
        target_scope = statement.getTargetDictScope()
        if type(target_scope) is GlobalsDictHandle:
            emit('%s = IMPORT_MODULE_STAR(tstate, %s, true, %s);' % (res_name, getModuleAccessCode(context=context), module_name))
        else:
            locals_declaration = context.addLocalsDictName(target_scope.getCodeName())
            emit('%(res_name)s = IMPORT_MODULE_STAR(tstate, %(locals_dict)s, false, %(module_name)s);' % {'res_name': res_name, 'locals_dict': locals_declaration, 'module_name': module_name})
        getErrorExitBoolCode(condition='%s == false' % res_name, release_name=module_name, emit=emit, context=context)

def generateImportNameCode(to_name, expression, emit, context):
    if False:
        i = 10
        return i + 15
    from_arg_name = context.allocateTempName('import_name_from')
    generateExpressionCode(to_name=from_arg_name, expression=expression.subnode_module, emit=emit, context=context)
    with withObjectCodeTemporaryAssignment(to_name, 'imported_value', expression, emit, context) as value_name:
        if python_version >= 848:
            emit('if (PyModule_Check(%(from_arg_name)s)) {\n    %(to_name)s = IMPORT_NAME_OR_MODULE(\n        tstate,\n        %(from_arg_name)s,\n        (PyObject *)moduledict_%(module_identifier)s,\n        %(import_name)s,\n        %(import_level)s\n    );\n} else {\n    %(to_name)s = IMPORT_NAME_FROM_MODULE(tstate, %(from_arg_name)s, %(import_name)s);\n}\n' % {'to_name': value_name, 'from_arg_name': from_arg_name, 'import_name': context.getConstantCode(constant=expression.getImportName()), 'import_level': context.getConstantCode(constant=expression.getImportLevel()), 'module_identifier': context.getModuleCodeName()})
        else:
            emit('%s = IMPORT_NAME_FROM_MODULE(tstate, %s, %s);' % (value_name, from_arg_name, context.getConstantCode(constant=expression.getImportName())))
        getErrorExitCode(check_name=value_name, release_name=from_arg_name, needs_check=expression.mayRaiseException(BaseException), emit=emit, context=context)
        context.addCleanupTempName(value_name)