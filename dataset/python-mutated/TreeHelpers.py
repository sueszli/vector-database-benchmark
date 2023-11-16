""" Helper functions for parsing the AST nodes and building the Nuitka node tree.

"""
import __future__
import ast
from nuitka import Constants, Options
from nuitka.Errors import CodeTooComplexCode
from nuitka.nodes.CallNodes import makeExpressionCall
from nuitka.nodes.CodeObjectSpecs import CodeObjectSpec
from nuitka.nodes.ConstantRefNodes import makeConstantRefNode
from nuitka.nodes.ContainerMakingNodes import makeExpressionMakeTupleOrConstant
from nuitka.nodes.DictionaryNodes import makeExpressionMakeDict
from nuitka.nodes.ExceptionNodes import StatementReraiseException
from nuitka.nodes.FrameNodes import StatementsFrameAsyncgen, StatementsFrameClass, StatementsFrameCoroutine, StatementsFrameFunction, StatementsFrameGenerator, StatementsFrameModule
from nuitka.nodes.KeyValuePairNodes import makeKeyValuePairExpressionsFromKwArgs
from nuitka.nodes.NodeBases import NodeBase
from nuitka.nodes.NodeMakingHelpers import mergeStatements
from nuitka.nodes.StatementNodes import StatementsSequence
from nuitka.PythonVersions import python_version
from nuitka.Tracing import optimization_logger, printLine

def dump(node):
    if False:
        return 10
    printLine(ast.dump(node))

def getKind(node):
    if False:
        while True:
            i = 10
    return node.__class__.__name__.rsplit('.', 1)[-1]

def extractDocFromBody(node):
    if False:
        while True:
            i = 10
    body = node.body
    doc = None
    if body and getKind(body[0]) == 'Expr':
        if getKind(body[0].value) == 'Str':
            doc = body[0].value.s
            body = body[1:]
        elif getKind(body[0].value) == 'Constant':
            if type(body[0].value.value) is str:
                doc = body[0].value.value
            body = body[1:]
        if Options.hasPythonFlagNoDocStrings():
            doc = None
    return (body, doc)

def parseSourceCodeToAst(source_code, module_name, filename, line_offset):
    if False:
        for i in range(10):
            print('nop')
    if not source_code.endswith('\n'):
        source_code = source_code + '\n'
    try:
        body = ast.parse(source_code, filename)
    except RuntimeError as e:
        if 'maximum recursion depth' in e.args[0]:
            raise CodeTooComplexCode(module_name, filename)
        raise
    assert getKind(body) == 'Module'
    if line_offset > 0:
        ast.increment_lineno(body, line_offset)
    return body

def detectFunctionBodyKind(nodes, start_value=None):
    if False:
        print('Hello World!')
    indications = set()
    if start_value is not None:
        indications.add(start_value)
    flags = set()

    def _checkCoroutine(field):
        if False:
            while True:
                i = 10
        'Check only for co-routine nature of the field and only update that.'
        old = set(indications)
        indications.clear()
        _check(field)
        if 'Coroutine' in indications:
            old.add('Coroutine')
        indications.clear()
        indications.update(old)

    def _check(node):
        if False:
            return 10
        node_class = node.__class__
        if node_class is ast.Yield:
            indications.add('Generator')
        elif python_version >= 768 and node_class is ast.YieldFrom:
            indications.add('Generator')
        elif python_version >= 848 and node_class in (ast.Await, ast.AsyncWith):
            indications.add('Coroutine')
        if node_class is ast.ClassDef:
            for (name, field) in ast.iter_fields(node):
                if name in ('name', 'body'):
                    pass
                elif name in ('bases', 'decorator_list', 'keywords'):
                    for child in field:
                        _check(child)
                elif name == 'starargs':
                    if field is not None:
                        _check(field)
                elif name == 'kwargs':
                    if field is not None:
                        _check(field)
                else:
                    assert False, (name, field, ast.dump(node))
        elif node_class in (ast.FunctionDef, ast.Lambda) or (python_version >= 848 and node_class is ast.AsyncFunctionDef):
            for (name, field) in ast.iter_fields(node):
                if name in ('name', 'body'):
                    pass
                elif name in ('bases', 'decorator_list'):
                    for child in field:
                        _check(child)
                elif name == 'args':
                    for child in field.defaults:
                        _check(child)
                    if python_version >= 768:
                        for child in node.args.kw_defaults:
                            if child is not None:
                                _check(child)
                        for child in node.args.args:
                            if child.annotation is not None:
                                _check(child.annotation)
                elif name == 'returns':
                    if field is not None:
                        _check(field)
                elif name == 'type_comment':
                    assert field is None or type(field) is str
                else:
                    assert False, (name, field, ast.dump(node))
        elif node_class is ast.GeneratorExp:
            for (name, field) in ast.iter_fields(node):
                if name == 'name':
                    pass
                elif name in ('body', 'comparators', 'elt'):
                    if python_version >= 880:
                        _checkCoroutine(field)
                elif name == 'generators':
                    _check(field[0].iter)
                    if python_version >= 880 and node in nodes:
                        for gen in field:
                            if gen.is_async:
                                indications.add('Coroutine')
                                break
                            if _checkCoroutine(gen):
                                break
                else:
                    assert False, (name, field, ast.dump(node))
        elif node_class is ast.ListComp and python_version >= 768:
            for (name, field) in ast.iter_fields(node):
                if name in ('name', 'body', 'comparators'):
                    pass
                elif name == 'generators':
                    _check(field[0].iter)
                elif name in ('body', 'elt'):
                    _check(field)
                else:
                    assert False, (name, field, ast.dump(node))
        elif python_version >= 624 and node_class is ast.SetComp:
            for (name, field) in ast.iter_fields(node):
                if name in ('name', 'body', 'comparators', 'elt'):
                    pass
                elif name == 'generators':
                    _check(field[0].iter)
                else:
                    assert False, (name, field, ast.dump(node))
        elif python_version >= 624 and node_class is ast.DictComp:
            for (name, field) in ast.iter_fields(node):
                if name in ('name', 'body', 'comparators', 'key', 'value'):
                    pass
                elif name == 'generators':
                    _check(field[0].iter)
                else:
                    assert False, (name, field, ast.dump(node))
        elif python_version >= 880 and node_class is ast.comprehension:
            for (name, field) in ast.iter_fields(node):
                if name in ('name', 'target'):
                    pass
                elif name == 'iter':
                    if node not in nodes:
                        _check(field)
                elif name == 'ifs':
                    for child in field:
                        _check(child)
                elif name == 'is_async':
                    if field:
                        indications.add('Coroutine')
                else:
                    assert False, (name, field, ast.dump(node))
        elif node_class is ast.Name:
            if python_version >= 768 and node.id == 'super':
                flags.add('has_super')
        elif python_version < 768 and node_class is ast.Exec:
            flags.add('has_exec')
            if node.globals is None:
                flags.add('has_unqualified_exec')
            for child in ast.iter_child_nodes(node):
                _check(child)
        elif python_version < 768 and node_class is ast.ImportFrom:
            for import_desc in node.names:
                if import_desc.name[0] == '*':
                    flags.add('has_exec')
            for child in ast.iter_child_nodes(node):
                _check(child)
        else:
            for child in ast.iter_child_nodes(node):
                _check(child)
    for node in nodes:
        _check(node)
    if indications:
        if 'Coroutine' in indications and 'Generator' in indications:
            function_kind = 'Asyncgen'
        else:
            assert len(indications) == 1, indications
            function_kind = indications.pop()
    else:
        function_kind = 'Function'
    return (function_kind, flags)
build_nodes_args3 = None
build_nodes_args2 = None
build_nodes_args1 = None

def setBuildingDispatchers(path_args3, path_args2, path_args1):
    if False:
        return 10
    global build_nodes_args3, build_nodes_args2, build_nodes_args1
    build_nodes_args3 = path_args3
    build_nodes_args2 = path_args2
    build_nodes_args1 = path_args1

def buildNode(provider, node, source_ref, allow_none=False):
    if False:
        return 10
    if node is None and allow_none:
        return None
    try:
        kind = getKind(node)
        if hasattr(node, 'lineno'):
            source_ref = source_ref.atLineNumber(node.lineno)
        if kind in build_nodes_args3:
            result = build_nodes_args3[kind](provider=provider, node=node, source_ref=source_ref)
        elif kind in build_nodes_args2:
            result = build_nodes_args2[kind](node=node, source_ref=source_ref)
        elif kind in build_nodes_args1:
            result = build_nodes_args1[kind](source_ref=source_ref)
        elif kind == 'Pass':
            result = None
        else:
            assert False, ast.dump(node)
        if result is None and allow_none:
            return None
        assert isinstance(result, NodeBase), result
        return result
    except SyntaxError:
        raise
    except RuntimeError:
        raise
    except KeyboardInterrupt:
        optimization_logger.info("Interrupted at '%s'." % source_ref)
        raise
    except:
        optimization_logger.warning("Problem at '%s' with %s." % (source_ref.getAsString(), ast.dump(node)))
        raise

def buildNodeList(provider, nodes, source_ref, allow_none=False):
    if False:
        return 10
    if nodes is not None:
        result = []
        for node in nodes:
            if hasattr(node, 'lineno'):
                node_source_ref = source_ref.atLineNumber(node.lineno)
            else:
                node_source_ref = source_ref
            entry = buildNode(provider, node, node_source_ref, allow_none)
            if entry is not None:
                result.append(entry)
        return result
    else:
        return []

def buildNodeTuple(provider, nodes, source_ref, allow_none=False):
    if False:
        while True:
            i = 10
    if nodes is not None:
        result = []
        for node in nodes:
            if hasattr(node, 'lineno'):
                node_source_ref = source_ref.atLineNumber(node.lineno)
            else:
                node_source_ref = source_ref
            entry = buildNode(provider, node, node_source_ref, allow_none)
            if entry is not None:
                result.append(entry)
        return tuple(result)
    else:
        return ()
_host_node = None

def buildAnnotationNode(provider, node, source_ref):
    if False:
        for i in range(10):
            print('nop')
    if python_version >= 880 and provider.getParentModule().getFutureSpec().isFutureAnnotations():
        global _host_node
        if _host_node is None:
            _host_node = ast.parse('x:1')
        _host_node.body[0].annotation = node
        r = compile(_host_node, '<annotations>', 'exec', __future__.CO_FUTURE_ANNOTATIONS, dont_inherit=True)
        m = {}
        exec(r, m)
        value = m['__annotations__']['x']
        if Options.is_debug and python_version >= 912:
            assert value == ast.unparse(node)
        return makeConstantRefNode(constant=value, source_ref=source_ref)
    return buildNode(provider, node, source_ref)

def makeModuleFrame(module, statements, source_ref):
    if False:
        for i in range(10):
            print('nop')
    assert module.isCompiledPythonModule()
    if Options.is_full_compat:
        code_name = '<module>'
    elif module.isMainModule():
        code_name = '<module>'
    else:
        code_name = '<module %s>' % module.getFullName()
    return StatementsFrameModule(statements=tuple(statements), code_object=CodeObjectSpec(co_name=code_name, co_qualname=code_name, co_kind='Module', co_varnames=(), co_freevars=(), co_argcount=0, co_posonlyargcount=0, co_kwonlyargcount=0, co_has_starlist=False, co_has_stardict=False, co_filename=module.getRunTimeFilename(), co_lineno=source_ref.getLineNumber(), future_spec=module.getFutureSpec()), source_ref=source_ref)

def buildStatementsNode(provider, nodes, source_ref):
    if False:
        i = 10
        return i + 15
    if nodes is None:
        return None
    statements = buildNodeList(provider, nodes, source_ref, allow_none=True)
    statements = mergeStatements(statements)
    if not statements:
        return None
    else:
        return StatementsSequence(statements=statements, source_ref=source_ref)

def buildFrameNode(provider, nodes, code_object, source_ref):
    if False:
        i = 10
        return i + 15
    if nodes is None:
        return None
    statements = buildNodeList(provider, nodes, source_ref, allow_none=True)
    statements = mergeStatements(statements)
    if not statements:
        return None
    if provider.isExpressionOutlineFunction():
        provider = provider.getParentVariableProvider()
    if provider.isExpressionFunctionBody():
        result = StatementsFrameFunction(statements=statements, code_object=code_object, source_ref=source_ref)
    elif provider.isExpressionClassBodyBase():
        result = StatementsFrameClass(statements=statements, code_object=code_object, locals_scope=provider.getLocalsScope(), source_ref=source_ref)
    elif provider.isExpressionGeneratorObjectBody():
        result = StatementsFrameGenerator(statements=statements, code_object=code_object, source_ref=source_ref)
    elif provider.isExpressionCoroutineObjectBody():
        result = StatementsFrameCoroutine(statements=statements, code_object=code_object, source_ref=source_ref)
    elif provider.isExpressionAsyncgenObjectBody():
        result = StatementsFrameAsyncgen(statements=statements, code_object=code_object, source_ref=source_ref)
    else:
        assert False, provider
    return result

def makeStatementsSequenceOrStatement(statements, source_ref):
    if False:
        for i in range(10):
            print('nop')
    'Make a statement sequence, but only if more than one statement\n\n    Useful for when we can unroll constructs already here, but are not sure if\n    we actually did that. This avoids the branch or the pollution of doing it\n    always.\n    '
    if len(statements) > 1:
        return StatementsSequence(statements=mergeStatements(statements), source_ref=source_ref)
    else:
        return statements[0]

def makeStatementsSequence(statements, allow_none, source_ref):
    if False:
        i = 10
        return i + 15
    if allow_none:
        statements = tuple((statement for statement in statements if statement is not None))
    if statements:
        return StatementsSequence(statements=mergeStatements(statements, allow_none=allow_none), source_ref=source_ref)
    else:
        return None

def makeStatementsSequenceFromStatement(statement):
    if False:
        while True:
            i = 10
    return StatementsSequence(statements=mergeStatements((statement,)), source_ref=statement.getSourceReference())

def makeStatementsSequenceFromStatements(*statements):
    if False:
        while True:
            i = 10
    assert statements
    assert None not in statements
    statements = mergeStatements(statements, allow_none=False)
    return StatementsSequence(statements=statements, source_ref=statements[0].getSourceReference())

def makeDictCreationOrConstant2(keys, values, source_ref):
    if False:
        return 10
    assert len(keys) == len(values)
    for value in values:
        if not value.isExpressionConstantRef():
            constant = False
            break
    else:
        constant = True
    if constant:
        result = makeConstantRefNode(constant=Constants.createConstantDict(keys=keys, values=[value.getCompileTimeConstant() for value in values]), user_provided=True, source_ref=source_ref)
    else:
        result = makeExpressionMakeDict(pairs=makeKeyValuePairExpressionsFromKwArgs(zip(keys, values)), source_ref=source_ref)
    if values:
        result.setCompatibleSourceReference(source_ref=values[-1].getCompatibleSourceReference())
    return result

def getStatementsAppended(statement_sequence, statements):
    if False:
        return 10
    return makeStatementsSequence(statements=(statement_sequence, statements), allow_none=False, source_ref=statement_sequence.getSourceReference())

def getStatementsPrepended(statement_sequence, statements):
    if False:
        for i in range(10):
            print('nop')
    return makeStatementsSequence(statements=(statements, statement_sequence), allow_none=False, source_ref=statement_sequence.getSourceReference())

def makeReraiseExceptionStatement(source_ref):
    if False:
        for i in range(10):
            print('nop')
    return StatementReraiseException(source_ref=source_ref)

def mangleName(name, owner):
    if False:
        while True:
            i = 10
    'Mangle names with leading "__" for usage in a class owner.\n\n    Notes: The is the private name handling for Python classes.\n    '
    if not name.startswith('__') or name.endswith('__'):
        return name
    else:
        class_container = owner.getContainingClassDictCreation()
        if class_container is None:
            return name
        else:
            return '_%s%s' % (class_container.getName().lstrip('_'), name)

def makeCallNode(called, *args, **kwargs):
    if False:
        while True:
            i = 10
    source_ref = args[-1]
    if len(args) > 1:
        args = makeExpressionMakeTupleOrConstant(elements=args[:-1], user_provided=True, source_ref=source_ref)
    else:
        args = None
    if kwargs:
        kwargs = makeDictCreationOrConstant2(keys=tuple(kwargs.keys()), values=tuple(kwargs.values()), source_ref=source_ref)
    else:
        kwargs = None
    return makeExpressionCall(called=called, args=args, kw=kwargs, source_ref=source_ref)
build_contexts = [None]

def pushBuildContext(value):
    if False:
        return 10
    build_contexts.append(value)

def popBuildContext():
    if False:
        print('Hello World!')
    del build_contexts[-1]

def getBuildContext():
    if False:
        for i in range(10):
            print('nop')
    return build_contexts[-1]