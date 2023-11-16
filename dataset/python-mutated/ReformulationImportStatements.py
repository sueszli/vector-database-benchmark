""" Reformulation of import statements.

Consult the Developer Manual for information. TODO: Add ability to sync
source code comments with Developer Manual sections.

"""
from nuitka.importing.ImportResolving import resolveModuleName
from nuitka.nodes.ConstantRefNodes import makeConstantRefNode
from nuitka.nodes.FutureSpecs import FutureSpec
from nuitka.nodes.GlobalsLocalsNodes import ExpressionBuiltinGlobals
from nuitka.nodes.ImportNodes import ExpressionBuiltinImport, ExpressionImportName, StatementImportStar, makeExpressionImportModuleFixed
from nuitka.nodes.NodeMakingHelpers import mergeStatements
from nuitka.nodes.StatementNodes import StatementsSequence
from nuitka.nodes.VariableAssignNodes import makeStatementAssignmentVariable
from nuitka.nodes.VariableNameNodes import StatementAssignmentVariableName
from nuitka.nodes.VariableRefNodes import ExpressionTempVariableRef
from nuitka.nodes.VariableReleaseNodes import makeStatementReleaseVariable
from nuitka.PythonVersions import python_version
from nuitka.utils.ModuleNames import ModuleName
from .ReformulationTryFinallyStatements import makeTryFinallyStatement
from .SyntaxErrors import raiseSyntaxError
from .TreeHelpers import makeStatementsSequenceOrStatement, mangleName
_future_import_nodes = []

def checkFutureImportsOnlyAtStart(body):
    if False:
        for i in range(10):
            print('nop')
    for node in body:
        if node in _future_import_nodes:
            _future_import_nodes.remove(node)
        elif _future_import_nodes:
            raiseSyntaxError('from __future__ imports must occur at the beginning of the file', _future_import_nodes[0].source_ref.atColumnNumber(_future_import_nodes[0].col_offset))

def _handleFutureImport(provider, node, source_ref):
    if False:
        while True:
            i = 10
    if not provider.isCompiledPythonModule():
        raiseSyntaxError('from __future__ imports must occur at the beginning of the file', source_ref.atColumnNumber(node.col_offset))
    for import_desc in node.names:
        (object_name, _local_name) = (import_desc.name, import_desc.asname)
        _enableFutureFeature(node=node, object_name=object_name, source_ref=source_ref)
    node.source_ref = source_ref
    _future_import_nodes.append(node)
_future_specs = []

def pushFutureSpec():
    if False:
        for i in range(10):
            print('nop')
    _future_specs.append(FutureSpec())

def getFutureSpec():
    if False:
        for i in range(10):
            print('nop')
    return _future_specs[-1]

def popFutureSpec():
    if False:
        while True:
            i = 10
    del _future_specs[-1]

def _enableFutureFeature(node, object_name, source_ref):
    if False:
        return 10
    future_spec = _future_specs[-1]
    if object_name == 'unicode_literals':
        future_spec.enableUnicodeLiterals()
    elif object_name == 'absolute_import':
        future_spec.enableAbsoluteImport()
    elif object_name == 'division':
        future_spec.enableFutureDivision()
    elif object_name == 'print_function':
        future_spec.enableFuturePrint()
    elif object_name == 'barry_as_FLUFL' and python_version >= 768:
        future_spec.enableBarry()
    elif object_name == 'generator_stop':
        future_spec.enableGeneratorStop()
    elif object_name == 'braces':
        raiseSyntaxError('not a chance', source_ref.atColumnNumber(node.col_offset))
    elif object_name in ('nested_scopes', 'generators', 'with_statement'):
        pass
    elif object_name == 'annotations' and python_version >= 880:
        future_spec.enableFutureAnnotations()
    else:
        raiseSyntaxError('future feature %s is not defined' % object_name, source_ref.atColumnNumber(node.col_offset))

def _resolveImportModuleName(module_name):
    if False:
        i = 10
        return i + 15
    if module_name:
        module_name = resolveModuleName(ModuleName(module_name)).asString()
    return module_name

def buildImportFromNode(provider, node, source_ref):
    if False:
        i = 10
        return i + 15
    module_name = node.module if node.module is not None else ''
    module_name = _resolveImportModuleName(module_name)
    level = node.level
    if level == -1:
        level = None
    elif level == 0 and (not _future_specs[-1].isAbsoluteImport()):
        level = None
    if level is not None:
        level_obj = makeConstantRefNode(level, source_ref, True)
    else:
        level_obj = None
    if module_name == '__future__':
        _handleFutureImport(provider, node, source_ref)
    target_names = []
    import_names = []
    for import_desc in node.names:
        (object_name, local_name) = (import_desc.name, import_desc.asname)
        if object_name == '*':
            target_names.append(None)
            assert local_name is None
        else:
            target_names.append(local_name if local_name is not None else object_name)
        import_names.append(object_name)
    if None in target_names:
        assert target_names == [None]
        if not provider.isCompiledPythonModule() and python_version >= 768:
            raiseSyntaxError('import * only allowed at module level', source_ref.atColumnNumber(node.col_offset))
        if provider.isCompiledPythonModule():
            import_globals = ExpressionBuiltinGlobals(source_ref)
            import_locals = ExpressionBuiltinGlobals(source_ref)
        else:
            import_globals = ExpressionBuiltinGlobals(source_ref)
            import_locals = makeConstantRefNode({}, source_ref, True)
        return StatementImportStar(target_scope=provider.getLocalsScope(), module=ExpressionBuiltinImport(name=makeConstantRefNode(module_name, source_ref, True), globals_arg=import_globals, locals_arg=import_locals, fromlist=makeConstantRefNode(('*',), source_ref, True), level=level_obj, source_ref=source_ref), source_ref=source_ref)
    else:
        if module_name == '__future__':
            imported_from_module = makeExpressionImportModuleFixed(module_name='__future__', value_name='__future__', source_ref=source_ref)
        else:
            imported_from_module = ExpressionBuiltinImport(name=makeConstantRefNode(module_name, source_ref, True), globals_arg=ExpressionBuiltinGlobals(source_ref), locals_arg=makeConstantRefNode(None, source_ref, True), fromlist=makeConstantRefNode(tuple(import_names), source_ref, True), level=level_obj, source_ref=source_ref)
        multi_names = len(target_names) > 1
        statements = []
        if multi_names:
            tmp_import_from = provider.allocateTempVariable(temp_scope=provider.allocateTempScope('import_from'), name='module', temp_type='object')
            statements.append(makeStatementAssignmentVariable(variable=tmp_import_from, source=imported_from_module, source_ref=source_ref))
            imported_from_module = ExpressionTempVariableRef(variable=tmp_import_from, source_ref=source_ref)
        import_statements = []
        first = True
        for (target_name, import_name) in zip(target_names, import_names):
            if not first:
                imported_from_module = imported_from_module.makeClone()
            first = False
            import_statements.append(StatementAssignmentVariableName(provider=provider, variable_name=mangleName(target_name, provider), source=ExpressionImportName(module=imported_from_module, import_name=import_name, level=0, source_ref=source_ref), source_ref=source_ref))
        if multi_names:
            statements.append(makeTryFinallyStatement(provider=provider, tried=import_statements, final=(makeStatementReleaseVariable(variable=tmp_import_from, source_ref=source_ref),), source_ref=source_ref))
        else:
            statements.extend(import_statements)
        return StatementsSequence(statements=mergeStatements(statements), source_ref=source_ref)

def buildImportModulesNode(provider, node, source_ref):
    if False:
        for i in range(10):
            print('nop')
    import_names = [(import_desc.name, import_desc.asname) for import_desc in node.names]
    import_nodes = []
    for import_desc in import_names:
        (module_name, local_name) = import_desc
        module_top_name = module_name.split('.')[0]
        level = makeConstantRefNode(0, source_ref, True) if _future_specs[-1].isAbsoluteImport() else None
        module_name = _resolveImportModuleName(module_name)
        import_node = ExpressionBuiltinImport(name=makeConstantRefNode(module_name, source_ref, True), globals_arg=ExpressionBuiltinGlobals(source_ref), locals_arg=makeConstantRefNode(None, source_ref, True), fromlist=makeConstantRefNode(None, source_ref, True), level=level, source_ref=source_ref)
        if local_name:
            for import_name in module_name.split('.')[1:]:
                import_node = ExpressionImportName(module=import_node, import_name=import_name, level=0, source_ref=source_ref)
        import_nodes.append(StatementAssignmentVariableName(provider=provider, variable_name=mangleName(local_name if local_name is not None else module_top_name, provider), source=import_node, source_ref=source_ref))
    return makeStatementsSequenceOrStatement(statements=import_nodes, source_ref=source_ref)