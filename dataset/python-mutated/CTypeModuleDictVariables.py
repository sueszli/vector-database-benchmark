""" Pseudo CType for module variables, object values contained in a dictionary.

    These are to integrate module variables with what is normally local
    stuff. Going from an to "PyObject *" is mostly its trick, then put
    into the dict.

"""
from nuitka.code_generation.templates.CodeTemplatesVariables import template_del_global_known, template_del_global_unclear, template_read_mvar_unclear
from .CTypeBases import CTypeBase

class CTypeModuleDictVariable(CTypeBase):

    @classmethod
    def emitVariableAssignCode(cls, value_name, needs_release, tmp_name, ref_count, inplace, emit, context):
        if False:
            i = 10
            return i + 15
        if inplace:
            orig_name = context.getInplaceLeftName()
            emit('if (%(orig_name)s != %(tmp_name)s) {\n    UPDATE_STRING_DICT_INPLACE(moduledict_%(module_identifier)s, (Nuitka_StringObject *)%(variable_name_str)s, %(tmp_name)s);\n}' % {'orig_name': orig_name, 'tmp_name': tmp_name, 'module_identifier': context.getModuleCodeName(), 'variable_name_str': context.getConstantCode(constant=value_name.code_name)})
        else:
            emit('UPDATE_STRING_DICT%s(moduledict_%s, (Nuitka_StringObject *)%s, %s);' % (ref_count, context.getModuleCodeName(), context.getConstantCode(constant=value_name.code_name), tmp_name))

    @classmethod
    def emitValueAccessCode(cls, value_name, emit, context):
        if False:
            print('Hello World!')
        tmp_name = context.allocateTempName('mvar_value')
        emit(template_read_mvar_unclear % {'module_identifier': context.getModuleCodeName(), 'tmp_name': tmp_name, 'var_name': context.getConstantCode(constant=value_name.code_name)})
        return tmp_name

    @classmethod
    def getDeleteObjectCode(cls, to_name, value_name, needs_check, tolerant, emit, context):
        if False:
            for i in range(10):
                print('nop')
        if not needs_check or tolerant:
            emit(template_del_global_known % {'module_identifier': context.getModuleCodeName(), 'var_name': context.getConstantCode(constant=value_name.code_name)})
        else:
            emit(template_del_global_unclear % {'module_identifier': context.getModuleCodeName(), 'result': to_name, 'var_name': context.getConstantCode(constant=value_name.code_name)})