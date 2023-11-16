""" Low level constant code generation.

This deals with constants, there creation, there access, and some checks about
them. Even mutable constants should not change during the course of the
program.

There are shared constants, which are created for multiple modules to use, you
can think of them as globals. And there are module local constants, which are
for a single module only.

"""
import os
import sys
from nuitka import Options
from nuitka.PythonVersions import python_version
from nuitka.Serialization import ConstantAccessor
from nuitka.utils.Distributions import getDistributionTopLevelPackageNames
from nuitka.Version import getNuitkaVersionTuple
from .CodeHelpers import withObjectCodeTemporaryAssignment
from .ErrorCodes import getAssertionCode
from .GlobalConstants import getConstantDefaultPopulation
from .Namify import namifyConstant
from .templates.CodeTemplatesConstants import template_constants_reading
from .templates.CodeTemplatesModules import template_header_guard

def generateConstantReferenceCode(to_name, expression, emit, context):
    if False:
        print('Hello World!')
    'Assign the constant behind the expression to to_name.'
    to_name.getCType().emitAssignmentCodeFromConstant(to_name=to_name, constant=expression.getCompileTimeConstant(), may_escape=True, emit=emit, context=context)

def generateConstantGenericAliasCode(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    origin_name = context.allocateTempName('generic_alias_origin')
    args_name = context.allocateTempName('generic_alias_args')
    origin_name.getCType().emitAssignmentCodeFromConstant(to_name=origin_name, constant=expression.getCompileTimeConstant().__origin__, may_escape=True, emit=emit, context=context)
    args_name.getCType().emitAssignmentCodeFromConstant(to_name=args_name, constant=expression.getCompileTimeConstant().__args__, may_escape=True, emit=emit, context=context)
    with withObjectCodeTemporaryAssignment(to_name, 'builtin_value', expression, emit, context) as value_name:
        emit('%s = Py_GenericAlias(%s, %s);' % (value_name, origin_name, args_name))
        getAssertionCode(check='%s != NULL' % value_name, emit=emit)
        context.addCleanupTempName(value_name)

def getConstantsDefinitionCode():
    if False:
        print('Hello World!')
    'Create the code code "__constants.c" and "__constants.h" files.\n\n    This needs to create code to make all global constants (used in more\n    than one module) and create them.\n\n    '
    constant_accessor = ConstantAccessor(data_filename='__constants.const', top_level_name='global_constants')
    lines = []
    for constant_value in getConstantDefaultPopulation():
        identifier = constant_accessor.getConstantCode(constant_value)
        assert '[' in identifier, (identifier, constant_value)
        lines.append('// %s' % repr(constant_value))
        lines.append('#define const_%s %s' % (namifyConstant(constant_value), identifier))
    sys_executable = None
    if not Options.shallMakeModule():
        if Options.isStandaloneMode():
            sys_executable = constant_accessor.getConstantCode(os.path.basename(sys.executable))
        else:
            sys_executable = constant_accessor.getConstantCode(sys.executable)
    sys_prefix = None
    sys_base_prefix = None
    sys_exec_prefix = None
    sys_base_exec_prefix = None
    if not Options.shallMakeModule() and (not Options.isStandaloneMode()):
        sys_prefix = constant_accessor.getConstantCode(sys.prefix)
        sys_exec_prefix = constant_accessor.getConstantCode(sys.exec_prefix)
        if python_version >= 768:
            sys_base_prefix = constant_accessor.getConstantCode(sys.base_prefix)
            sys_base_exec_prefix = constant_accessor.getConstantCode(sys.base_exec_prefix)
    metadata_values_code = constant_accessor.getConstantCode(metadata_values)
    lines.insert(0, 'extern PyObject *global_constants[%d];' % constant_accessor.getConstantsCount())
    header = template_header_guard % {'header_guard_name': '__NUITKA_GLOBAL_CONSTANTS_H__', 'header_body': '\n'.join(lines)}
    (major, minor, micro, is_final, _rc_number) = getNuitkaVersionTuple()
    body = template_constants_reading % {'global_constants_count': constant_accessor.getConstantsCount(), 'sys_executable': sys_executable, 'sys_prefix': sys_prefix, 'sys_base_prefix': sys_base_prefix, 'sys_exec_prefix': sys_exec_prefix, 'sys_base_exec_prefix': sys_base_exec_prefix, 'nuitka_version_major': major, 'nuitka_version_minor': minor, 'nuitka_version_micro': micro, 'nuitka_version_level': 'release' if is_final else 'candidate', 'metadata_values': metadata_values_code}
    return (header, body)
metadata_values = {}

def addDistributionMetadataValue(name, distribution):
    if False:
        i = 10
        return i + 15
    metadata = str(distribution.read_text('METADATA') or distribution.read_text('PKG-INFO') or '')
    entry_points = str(distribution.read_text('entry_points.txt') or '')
    package_name = getDistributionTopLevelPackageNames(distribution)[0]
    metadata_values[name] = (package_name, metadata, entry_points)

def getDistributionMetadataValues():
    if False:
        print('Hello World!')
    return sorted(tuple(metadata_values.items()))