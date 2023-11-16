""" This tool is generating node variants from Jinja templates.

"""
import sys
import nuitka.Options
nuitka.Options.is_full_compat = False
from collections import namedtuple
import nuitka.code_generation.BinaryOperationHelperDefinitions
import nuitka.code_generation.CodeGeneration
import nuitka.code_generation.ComparisonCodes
import nuitka.code_generation.Namify
import nuitka.nodes.PackageMetadataNodes
import nuitka.nodes.PackageResourceNodes
import nuitka.nodes.SideEffectNodes
import nuitka.specs.BuiltinBytesOperationSpecs
import nuitka.specs.BuiltinDictOperationSpecs
import nuitka.specs.BuiltinListOperationSpecs
import nuitka.specs.BuiltinStrOperationSpecs
import nuitka.specs.BuiltinTypeOperationSpecs
import nuitka.specs.HardImportSpecs
import nuitka.tree.Building
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.nodes.ImportNodes import hard_modules_non_stdlib
from nuitka.nodes.NodeMetaClasses import NodeCheckMetaClass
from nuitka.nodes.shapes.BuiltinTypeShapes import tshape_bool, tshape_bytes, tshape_dict, tshape_int, tshape_list, tshape_none, tshape_str, tshape_tuple
from nuitka.utils.Jinja2 import getTemplate
from .Common import formatArgs, getMethodVariations, getSpecs, python2_dict_methods, python2_list_methods, python2_str_methods, python2_type_methods, python3_bytes_methods, python3_dict_methods, python3_list_methods, python3_str_methods, python3_type_methods, withFileOpenedAndAutoFormatted, writeLine
attribute_information = {}
attribute_shape_operations = {}
attribute_shape_operations_result_types = {}
attribute_shape_operations_mixin_classes = {}
attribute_shape_versions = {}
attribute_shape_variations = {}
attribute_shape_node_arg_mapping = {}
attribute_shape_args = {}
attribute_shape_arg_tests = {}
attribute_shape_static = {}
node_factory_translations = {}

def _getMixinForShape(shape):
    if False:
        while True:
            i = 10
    if shape is tshape_str:
        return 'nuitka.nodes.ExpressionShapeMixins.ExpressionStrShapeExactMixin'
    elif shape is tshape_list:
        return 'nuitka.nodes.ExpressionShapeMixins.ExpressionListShapeExactMixin'
    elif shape is tshape_tuple:
        return 'nuitka.nodes.ExpressionShapeMixins.ExpressionTupleShapeExactMixin'
    elif shape is tshape_int:
        return 'nuitka.nodes.ExpressionShapeMixins.ExpressionIntShapeExactMixin'
    elif shape is tshape_bool:
        return 'nuitka.nodes.ExpressionShapeMixins.ExpressionBoolShapeExactMixin'
    elif shape is tshape_none:
        return 'nuitka.nodes.ExpressionShapeMixins.ExpressionNoneShapeExactMixin'
    elif shape is tshape_dict:
        return 'nuitka.nodes.ExpressionShapeMixins.ExpressionDictShapeExactMixin'
    elif shape is tshape_bytes:
        return 'nuitka.nodes.ExpressionShapeMixins.ExpressionBytesShapeExactMixin'
    else:
        assert False, shape

def processTypeShapeAttribute(shape_name, spec_module, python2_methods, python3_methods, staticmethod_names=()):
    if False:
        for i in range(10):
            print('nop')
    for method_name in python2_methods:
        attribute_information.setdefault(method_name, set()).add(shape_name)
        key = (method_name, shape_name)
        if method_name not in python3_methods:
            attribute_shape_versions[key] = 'str is bytes'
        (present, arg_names, arg_tests, arg_name_mapping, arg_counts, result_shape) = getMethodVariations(spec_module=spec_module, shape_name=shape_name, method_name=method_name)
        attribute_shape_operations[key] = present
        attribute_shape_operations_result_types[key] = result_shape
        if result_shape is not None:
            attribute_shape_operations_mixin_classes[key] = [_getMixinForShape(result_shape)]
        if present:
            attribute_shape_args[key] = tuple(arg_names)
            attribute_shape_arg_tests[key] = arg_tests
            attribute_shape_static[key] = method_name in staticmethod_names
            if len(arg_counts) > 1:
                attribute_shape_variations[key] = arg_counts
            attribute_shape_node_arg_mapping[key] = arg_name_mapping
    for method_name in python3_methods:
        attribute_information.setdefault(method_name, set()).add(shape_name)
        key = (method_name, shape_name)
        if method_name not in python2_methods:
            attribute_shape_versions[key] = 'str is not bytes'
        (present, arg_names, arg_tests, arg_name_mapping, arg_counts, result_shape) = getMethodVariations(spec_module=spec_module, shape_name=shape_name, method_name=method_name)
        attribute_shape_operations[key] = present
        attribute_shape_operations_result_types[key] = result_shape
        if result_shape is not None:
            attribute_shape_operations_mixin_classes[key] = [_getMixinForShape(result_shape)]
        if present:
            attribute_shape_args[key] = tuple(arg_names)
            attribute_shape_arg_tests[key] = arg_tests
            attribute_shape_static[key] = method_name in staticmethod_names
            if len(arg_counts) > 1:
                attribute_shape_variations[key] = arg_counts
            attribute_shape_node_arg_mapping[key] = arg_name_mapping
processTypeShapeAttribute('tshape_dict', nuitka.specs.BuiltinDictOperationSpecs, python2_dict_methods, python3_dict_methods, ('fromkeys',))
processTypeShapeAttribute('tshape_str', nuitka.specs.BuiltinStrOperationSpecs, python2_str_methods, python3_str_methods)
processTypeShapeAttribute('tshape_bytes', nuitka.specs.BuiltinBytesOperationSpecs, (), python3_bytes_methods)
processTypeShapeAttribute('tshape_list', nuitka.specs.BuiltinListOperationSpecs, python2_list_methods, python3_list_methods)
processTypeShapeAttribute('tshape_type', nuitka.specs.BuiltinTypeOperationSpecs, python2_type_methods, python3_type_methods)
attribute_shape_empty = {}
attribute_shape_empty['update', 'tshape_dict'] = 'lambda source_ref: wrapExpressionWithNodeSideEffects(\n    new_node=makeConstantRefNode(\n        constant=None,\n        source_ref=source_ref\n    ),\n    old_node=dict_arg\n)\n'

def emitGenerationWarning(emit, doc_string, template_name):
    if False:
        while True:
            i = 10
    attribute_code_names = set(attribute_information.keys())
    attribute_code_names = set((attribute_name.replace('_', '') for attribute_name in attribute_information))
    attribute_arg_names = set(sum(attribute_shape_args.values(), ()))
    emit('\n# We are not avoiding these in generated code at all\n# pylint: disable=I0021,too-many-lines\n# pylint: disable=I0021,line-too-long\n# pylint: disable=I0021,too-many-instance-attributes\n# pylint: disable=I0021,too-many-return-statements\n')
    emit('\n"""%s\n\nWARNING, this code is GENERATED. Modify the template %s instead!\n\nspell-checker: ignore %s\nspell-checker: ignore %s\n"""\n\n' % (doc_string, template_name, ' '.join(sorted(attribute_code_names)), ' '.join(sorted(attribute_arg_names))))

def formatCallArgs(operation_node_arg_mapping, args, starting=True):
    if False:
        while True:
            i = 10

    def mapName(arg):
        if False:
            for i in range(10):
                print('nop')
        if not operation_node_arg_mapping:
            return arg
        else:
            return operation_node_arg_mapping.get(arg, arg)

    def mapValue(arg):
        if False:
            for i in range(10):
                print('nop')
        if arg == 'pairs':
            return 'makeKeyValuePairExpressionsFromKwArgs(pairs)'
        else:
            return arg
    if args is None:
        result = ''
    else:
        result = ','.join(('%s=%s' % (mapName(arg), mapValue(arg)) for arg in args))
    if not starting and result:
        result = ',' + result
    return result

def _getPython3OperationName(attribute_name):
    if False:
        i = 10
        return i + 15
    if attribute_name == 'items':
        return 'iteritems'
    elif attribute_name == 'keys':
        return 'iterkeys'
    elif attribute_name == 'values':
        return 'itervalues'
    else:
        return None

def makeAttributeNodes():
    if False:
        for i in range(10):
            print('nop')
    filename_python = 'nuitka/nodes/AttributeNodesGenerated.py'
    template = getTemplate(package_name=__package__, template_subdir='templates_python', template_name='AttributeNodeFixed.py.j2')
    with withFileOpenedAndAutoFormatted(filename_python, ignore_errors=True) as output_python:

        def emit(*args):
            if False:
                while True:
                    i = 10
            writeLine(output_python, *args)
        emitGenerationWarning(emit, 'Specialized attribute nodes', template.name)
        emit('from .AttributeLookupNodes import ExpressionAttributeLookupFixedBase')
        emit('from nuitka.specs.BuiltinParameterSpecs import extractBuiltinArgs')
        emit('from nuitka.nodes.ConstantRefNodes import makeConstantRefNode')
        emit('from nuitka.nodes.NodeMakingHelpers import wrapExpressionWithNodeSideEffects')
        emit('from nuitka.nodes.KeyValuePairNodes import makeKeyValuePairExpressionsFromKwArgs')
        emit('from nuitka.nodes.AttributeNodes import makeExpressionAttributeLookup')
        emit('from .NodeBases import SideEffectsFromChildrenMixin')
        emit('attribute_classes = {}')
        emit('attribute_typed_classes = set()')
        for (attribute_name, shape_names) in sorted(attribute_information.items()):
            code = template.render(attribute_name=attribute_name, python3_operation_name=_getPython3OperationName(attribute_name), shape_names=shape_names, attribute_shape_versions=attribute_shape_versions, attribute_shape_operations=attribute_shape_operations, attribute_shape_variations=attribute_shape_variations, attribute_shape_node_arg_mapping=attribute_shape_node_arg_mapping, attribute_shape_args=attribute_shape_args, attribute_shape_arg_tests=attribute_shape_arg_tests, attribute_shape_empty=attribute_shape_empty, attribute_shape_static=attribute_shape_static, formatArgs=formatArgs, formatCallArgs=formatCallArgs, translateNodeClassName=translateNodeClassName, reversed=reversed, str=str, name=template.name)
            emit(code)

def makeBuiltinOperationNodes():
    if False:
        i = 10
        return i + 15
    filename_python = 'nuitka/nodes/BuiltinOperationNodeBasesGenerated.py'
    template = getTemplate(package_name=__package__, template_subdir='templates_python', template_name='BuiltinOperationNodeBases.py.j2')
    with withFileOpenedAndAutoFormatted(filename_python, ignore_errors=True) as output_python:

        def emit(*args):
            if False:
                for i in range(10):
                    print('nop')
            writeLine(output_python, *args)
        emitGenerationWarning(emit, 'Specialized attribute nodes', template.name)
        for (attribute_name, shape_names) in sorted(attribute_information.items()):
            attribute_name_class = attribute_name.replace('_', '').title()
            code = template.render(attribute_name=attribute_name, attribute_name_class=attribute_name_class, python3_operation_name=_getPython3OperationName(attribute_name), shape_names=shape_names, attribute_shape_versions=attribute_shape_versions, attribute_shape_operations=attribute_shape_operations, attribute_shape_variations=attribute_shape_variations, attribute_shape_node_arg_mapping=attribute_shape_node_arg_mapping, attribute_shape_args=attribute_shape_args, attribute_shape_arg_tests=attribute_shape_arg_tests, attribute_shape_empty=attribute_shape_empty, attribute_shape_static=attribute_shape_static, attribute_shape_operations_mixin_classes=attribute_shape_operations_mixin_classes, formatArgs=formatArgs, formatCallArgs=formatCallArgs, addChildrenMixin=addChildrenMixin, reversed=reversed, str=str, name=template.name)
            emit(code)

def adaptModuleName(value):
    if False:
        return 10
    if value == 'importlib_metadata':
        return 'importlib_metadata_backport'
    if value == 'importlib_resources':
        return 'importlib_resources_backport'
    return value

def makeTitleCased(value):
    if False:
        for i in range(10):
            print('nop')
    return ''.join((s.title() for s in value.split('_'))).replace('.', '')

def makeCodeCased(value):
    if False:
        i = 10
        return i + 15
    return value.replace('.', '_')

def getCallModuleName(module_name, function_name):
    if False:
        return 10
    if module_name in ('pkg_resources', 'importlib.metadata', 'importlib_metadata'):
        if function_name in ('resource_stream', 'resource_string'):
            return 'PackageResourceNodes'
        return 'PackageMetadataNodes'
    if module_name in ('pkgutil', 'importlib.resources', 'importlib_resources'):
        return 'PackageResourceNodes'
    if module_name in ('os', 'sys', 'os.path'):
        return 'OsSysNodes'
    if module_name == 'ctypes':
        return 'CtypesNodes'
    if module_name == 'builtins':
        if function_name == 'open':
            return 'BuiltinOpenNodes'
    assert False, (module_name, function_name)

def translateNodeClassName(node_class_name):
    if False:
        print('Hello World!')
    return node_factory_translations.get(node_class_name, node_class_name)

def makeMixinName(is_expression, is_statement, named_children, named_children_types, named_children_checkers, auto_compute_handling, node_attributes):
    if False:
        i = 10
        return i + 15

    def _addType(name):
        if False:
            return 10
        if name in named_children_types:
            if named_children_types[name] == 'optional' and named_children_checkers.get(name) == 'convertNoneConstantToNone':
                return ''
            return '_' + named_children_types[name]
        else:
            return ''

    def _addChecker(name):
        if False:
            i = 10
            return i + 15
        if name in named_children_checkers:
            if named_children_checkers[name] == 'convertNoneConstantToNone':
                return '_auto_none'
            if named_children_checkers[name] == 'convertEmptyStrConstantToNone':
                return '_auto_none_empty_str'
            if named_children_checkers[name] == 'checkStatementsSequenceOrNone':
                return '_statements_or_none'
            if named_children_checkers[name] == 'checkStatementsSequence':
                return '_statements'
            else:
                assert False, named_children_checkers[name]
        else:
            return ''
    mixin_name = ''.join((makeTitleCased(named_child + _addType(named_child) + _addChecker(named_child)) for named_child in named_children))
    mixin_name += '_'.join(sorted(auto_compute_handling)).title().replace('_', '').replace(':', '')
    mixin_name += '_'.join(sorted(node_attributes)).title().replace('_', '')
    if len(named_children) == 0:
        mixin_name = 'NoChildHaving' + mixin_name + 'Mixin'
    elif len(named_children) == 1:
        mixin_name = 'ChildHaving' + mixin_name + 'Mixin'
    else:
        mixin_name = 'ChildrenHaving' + mixin_name + 'Mixin'
    if is_statement:
        mixin_name = 'Statement' + mixin_name
    elif is_expression:
        pass
    else:
        mixin_name = 'Module' + mixin_name
    return mixin_name
children_mixins = []
children_mixins_intentions = {}
children_mixing_setters_needed = {}

def addChildrenMixin(is_expression, is_statement, intended_for, named_children, named_children_types, named_children_checkers, auto_compute_handling=(), node_attributes=()):
    if False:
        while True:
            i = 10
    assert type(is_statement) is bool
    children_mixins.append((is_expression, is_statement, named_children, named_children_types, named_children_checkers, auto_compute_handling, node_attributes))
    mixin_name = makeMixinName(is_expression, is_statement, named_children, named_children_types, named_children_checkers, auto_compute_handling, node_attributes)
    if mixin_name not in children_mixins_intentions:
        children_mixins_intentions[mixin_name] = []
    if intended_for not in children_mixins_intentions[mixin_name]:
        children_mixins_intentions[mixin_name].append(intended_for)
    for named_child in named_children_types:
        assert named_child in named_children, named_child
    for (named_child, named_child_checker) in named_children_checkers.items():
        if named_child_checker == 'convertNoneConstantToNone':
            assert named_children_types[named_child] == 'optional'
    return mixin_name

def _parseNamedChildrenSpec(named_children):
    if False:
        i = 10
        return i + 15
    new_named_children = []
    setters_needed = set()
    named_children_types = {}
    named_children_checkers = {}
    for named_child_spec in named_children:
        if '|' in named_child_spec:
            (named_child, named_child_properties) = named_child_spec.split('|', 1)
            for named_child_property in named_child_properties.split('+'):
                if named_child_property == 'setter':
                    setters_needed.add(named_child)
                elif named_child_property == 'tuple':
                    named_children_types[named_child] = 'tuple'
                elif named_child_property == 'auto_none':
                    named_children_types[named_child] = 'optional'
                    named_children_checkers[named_child] = 'convertNoneConstantToNone'
                elif named_child_property == 'auto_none_empty_str':
                    named_children_types[named_child] = 'optional'
                    named_children_checkers[named_child] = 'convertEmptyStrConstantToNone'
                elif named_child_property == 'statements_or_none':
                    named_children_types[named_child] = 'optional'
                    named_children_checkers[named_child] = 'checkStatementsSequenceOrNone'
                elif named_child_property == 'statements':
                    named_children_checkers[named_child] = 'checkStatementsSequence'
                elif named_child_property == 'optional':
                    named_children_types[named_child] = 'optional'
                else:
                    assert False, named_child_property
        else:
            named_child = named_child_spec
        new_named_children.append(named_child)
    return (new_named_children, named_children_types, named_children_checkers, setters_needed)

def _addFromNode(node_class):
    if False:
        print('Hello World!')
    named_children = getattr(node_class, 'named_children', ())
    if hasattr(node_class, 'auto_compute_handling'):
        auto_compute_handling = frozenset(getattr(node_class, 'auto_compute_handling').split(','))
    else:
        auto_compute_handling = ()
    node_attributes = getattr(node_class, 'node_attributes', ())
    if not named_children and (not auto_compute_handling) and (not node_attributes):
        return
    (new_named_children, named_children_types, named_children_checkers, setters_needed) = _parseNamedChildrenSpec(named_children)
    mixin_name = makeMixinName(node_class.kind.startswith('EXPRESSION'), node_class.kind.startswith('STATEMENT'), tuple(new_named_children), named_children_types, named_children_checkers, auto_compute_handling, node_attributes)
    if mixin_name not in children_mixing_setters_needed:
        children_mixing_setters_needed[mixin_name] = set()
    children_mixing_setters_needed[mixin_name].update(setters_needed)
    for base in node_class.__mro__:
        if base.__name__ == mixin_name:
            break
    else:
        print('Not done', node_class.__name__, named_children, mixin_name)
    addChildrenMixin(node_class.kind.startswith('EXPRESSION'), node_class.kind.startswith('STATEMENT'), node_class.__name__, tuple(new_named_children), named_children_types, named_children_checkers, auto_compute_handling, node_attributes)

def addFromNodes():
    if False:
        return 10
    for node_class in NodeCheckMetaClass.kinds.values():
        if hasattr(sys.modules[node_class.__module__], 'make' + node_class.__name__):
            node_factory_translations[node_class.__name__] = 'make' + node_class.__name__
        _addFromNode(node_class)
    node_factory_translations['ExpressionImportlibMetadataMetadataCall'] = 'makeExpressionImportlibMetadataMetadataCall'
    node_factory_translations['ExpressionImportlibMetadataBackportMetadataCall'] = 'makeExpressionImportlibMetadataBackportMetadataCall'
    node_factory_translations['ExpressionBuiltinsOpenCall'] = 'makeExpressionBuiltinsOpenCall'
addFromNodes()

def makeChildrenHavingMixinNodes():
    if False:
        return 10
    filename_python = 'nuitka/nodes/ChildrenHavingMixins.py'
    filename_python2 = 'nuitka/nodes/ExpressionBasesGenerated.py'
    filename_python3 = 'nuitka/nodes/StatementBasesGenerated.py'
    template = getTemplate(package_name=__package__, template_subdir='templates_python', template_name='ChildrenHavingMixin.py.j2')
    mixins_done = set()
    with withFileOpenedAndAutoFormatted(filename_python, ignore_errors=True) as output_python, withFileOpenedAndAutoFormatted(filename_python2, ignore_errors=True) as output_python2, withFileOpenedAndAutoFormatted(filename_python3, ignore_errors=True) as output_python3:

        def emit1(*args):
            if False:
                print('Hello World!')
            writeLine(output_python, *args)

        def emit2(*args):
            if False:
                print('Hello World!')
            writeLine(output_python2, *args)

        def emit3(*args):
            if False:
                for i in range(10):
                    print('nop')
            writeLine(output_python3, *args)

        def emit(*args):
            if False:
                for i in range(10):
                    print('nop')
            emit1(*args)
            emit2(*args)
            emit3(*args)
        emitGenerationWarning(emit1, 'Children having mixins', template.name)
        emitGenerationWarning(emit2, 'Children having expression bases', template.name)
        emitGenerationWarning(emit3, 'Children having statement bases', template.name)
        emit('# Loop unrolling over child names, pylint: disable=too-many-branches')
        emit1('\nfrom nuitka.nodes.Checkers import (\n    checkStatementsSequenceOrNone,\n    convertNoneConstantToNone,\n    convertEmptyStrConstantToNone\n)\n')
        emit3('\nfrom nuitka.nodes.Checkers import (\n    checkStatementsSequenceOrNone,     checkStatementsSequence,\n    convertNoneConstantToNone\n)\n')
        for (is_expression, is_statement, named_children, named_children_types, named_children_checkers, auto_compute_handling, node_attributes) in sorted(children_mixins, key=lambda x: (x[0], x[1], x[2], x[3].items(), x[4].items())):
            mixin_name = makeMixinName(is_expression, is_statement, named_children, named_children_types, named_children_checkers, auto_compute_handling, node_attributes)
            if mixin_name in mixins_done:
                continue
            intended_for = [value for value in children_mixins_intentions[mixin_name] if not value.endswith('Base') or value.rstrip('Base') not in children_mixins_intentions[mixin_name]]
            intended_for.sort()
            auto_compute_handling_set = set(auto_compute_handling)

            def pop(name):
                if False:
                    return 10
                result = name in auto_compute_handling_set
                auto_compute_handling_set.discard(name)
                return result
            is_compute_final = pop('final')
            is_compute_final_children = pop('final_children')
            is_compute_no_raise = pop('no_raise')
            is_compute_raise = pop('raise')
            is_compute_raise_operation = pop('raise_operation')
            assert is_compute_no_raise + is_compute_raise + is_compute_raise_operation < 2
            if is_compute_raise:
                raise_mode = 'raise'
            elif is_compute_no_raise:
                raise_mode = 'no_raise'
            elif is_compute_raise_operation:
                raise_mode = 'raise_operation'
            else:
                raise_mode = None
            is_compute_statement = pop('operation')
            has_post_node_init = pop('post_init')
            awaited_constant_attributes = OrderedSet((value.split(':', 1)[1] for value in auto_compute_handling_set if value.startswith('wait_constant:')))
            auto_compute_handling_set -= {'wait_constant:%s' % value for value in awaited_constant_attributes}
            assert not auto_compute_handling_set, auto_compute_handling_set
            code = template.render(name=template.name, is_expression=is_expression, is_statement=is_statement, mixin_name=mixin_name, named_children=named_children, named_children_types=named_children_types, named_children_checkers=named_children_checkers, children_mixing_setters_needed=sorted(tuple(children_mixing_setters_needed.get(mixin_name, ()))), intended_for=intended_for, is_compute_final=is_compute_final, is_compute_final_children=is_compute_final_children, raise_mode=raise_mode, is_compute_statement=is_compute_statement, awaited_constant_attributes=awaited_constant_attributes, has_post_node_init=has_post_node_init, node_attributes=node_attributes, len=len)
            if is_statement:
                emit3(code)
            elif auto_compute_handling or node_attributes:
                emit2(code)
            else:
                emit1(code)
            mixins_done.add(mixin_name)
SpecVersion = namedtuple('SpecVersion', ('spec_name', 'python_criterion', 'spec', 'suffix'))

def getSpecVersions(spec_module):
    if False:
        i = 10
        return i + 15
    result = {}
    for (spec_name, spec) in getSpecs(spec_module):
        for (version, str_version) in ((880, '37'), (896, '38'), (912, '39'), (928, '310'), (944, '311')):
            if 'since_%s' % str_version in spec_name:
                python_criterion = '>= 0x%x' % version
                suffix = 'Since%s' % str_version
                break
            if 'before_%s' % str_version in spec_name:
                python_criterion = '< 0x%x' % version
                suffix = 'Before%s' % str_version
                break
        else:
            python_criterion = None
            suffix = ''
        assert '.entry_points' not in spec_name or python_criterion is not None
        if spec.name not in result:
            result[spec.name] = []
        result[spec.name].append(SpecVersion(spec_name, python_criterion, spec, suffix))
        result[spec.name].sort(key=lambda spec_version: spec_version.python_criterion or '', reverse=True)
    return tuple(sorted(result.values()))

def makeHardImportNodes():
    if False:
        for i in range(10):
            print('nop')
    filename_python = 'nuitka/nodes/HardImportNodesGenerated.py'
    template_ref_node = getTemplate(package_name=__package__, template_subdir='templates_python', template_name='HardImportReferenceNode.py.j2')
    template_call_node = getTemplate(package_name=__package__, template_subdir='templates_python', template_name='HardImportCallNode.py.j2')
    with withFileOpenedAndAutoFormatted(filename_python, ignore_errors=True) as output_python:

        def emit(*args):
            if False:
                for i in range(10):
                    print('nop')
            writeLine(output_python, *args)
        emitGenerationWarning(emit, 'Hard import nodes', template_ref_node.name)
        emit('\nhard_import_node_classes = {}\n\n')
        for spec_descriptions in getSpecVersions(nuitka.specs.HardImportSpecs):
            spec = spec_descriptions[0][2]
            named_children_checkers = {}
            (module_name, function_name) = spec.name.rsplit('.', 1)
            module_name_title = makeTitleCased(adaptModuleName(module_name))
            function_name_title = makeTitleCased(function_name)
            node_class_name = 'Expression%s%s' % (module_name_title, function_name_title)
            code = template_ref_node.render(name=template_ref_node.name, parameter_names_count=len(spec.getParameterNames()), function_name=function_name, function_name_title=function_name_title, function_name_code=makeCodeCased(function_name), module_name=module_name, module_name_code=makeCodeCased(adaptModuleName(module_name)), module_name_title=module_name_title, call_node_module_name=getCallModuleName(module_name, function_name), translateNodeClassName=translateNodeClassName, is_stdlib=module_name not in hard_modules_non_stdlib, specs=spec_descriptions)
            emit(code)
            for spec_desc in spec_descriptions:
                spec = spec_desc.spec
                parameter_names = spec.getParameterNames()
                named_children_types = {}
                if spec.name == 'pkg_resources.require':
                    named_children_types['requirements'] = 'tuple'
                if spec.getDefaultCount():
                    for optional_name in spec.getArgumentNames()[-spec.getDefaultCount():]:
                        assert optional_name not in named_children_types
                        named_children_types[optional_name] = 'optional'
                if spec.getStarDictArgumentName():
                    named_children_types[spec.getStarDictArgumentName()] = 'tuple'
                if parameter_names:
                    mixin_name = addChildrenMixin(True, False, node_class_name, parameter_names, named_children_types, named_children_checkers)
                else:
                    mixin_name = None
                extra_mixins = []
                result_shape = spec.getTypeShape()
                if result_shape is not None:
                    extra_mixins.append(_getMixinForShape(result_shape))
                code = template_call_node.render(name=template_call_node.name, mixin_name=mixin_name, suffix=spec_desc.suffix, python_criterion=spec_desc.python_criterion, extra_mixins=extra_mixins, parameter_names_count=len(spec.getParameterNames()), named_children=parameter_names, named_children_types=named_children_types, argument_names=spec.getArgumentNames(), star_list_argument_name=spec.getStarListArgumentName(), star_dict_argument_name=spec.getStarDictArgumentName(), function_name=function_name, function_name_title=function_name_title, function_name_code=makeCodeCased(function_name), module_name=module_name, is_stdlib_module=module_name in ('builtins', 'os', 'os.path', 'pkgutil', 'ctypes', 'importlib.metadata', 'importlib.resources'), module_name_code=makeCodeCased(adaptModuleName(module_name)), module_name_title=module_name_title, call_node_module_name=getCallModuleName(module_name, function_name), spec_name=spec_desc.spec_name)
                emit(code)

def main():
    if False:
        i = 10
        return i + 15
    makeHardImportNodes()
    makeAttributeNodes()
    makeBuiltinOperationNodes()
    makeChildrenHavingMixinNodes()