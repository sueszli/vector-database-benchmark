""" Reformulation of "yield" and "yield from" expressions.

Consult the Developer Manual for information. TODO: Add ability to sync
source code comments with Developer Manual sections.

"""
import ast
from nuitka.nodes.ConstantRefNodes import ExpressionConstantNoneRef
from nuitka.nodes.CoroutineNodes import ExpressionAsyncWait
from nuitka.nodes.YieldNodes import ExpressionYield, ExpressionYieldFrom, ExpressionYieldFromAwaitable
from nuitka.PythonVersions import python_version
from .SyntaxErrors import raiseSyntaxError
from .TreeHelpers import buildNode

def _checkInsideGenerator(construct_name, provider, node, source_ref):
    if False:
        while True:
            i = 10
    if provider.isCompiledPythonModule():
        raiseSyntaxError("'%s' outside function" % (construct_name if construct_name == 'await' else 'yield'), source_ref.atColumnNumber(node.col_offset))
    if provider.isExpressionAsyncgenObjectBody() and construct_name == 'yield_from':
        raiseSyntaxError("'%s' inside async function" % ('yield' if node.__class__ is ast.Yield else 'yield from',), source_ref.atColumnNumber(node.col_offset))
    if python_version >= 896 and provider.isExpressionGeneratorObjectBody() and (provider.name == '<genexpr>') and (construct_name != 'await'):
        raiseSyntaxError("'%s' inside generator expression" % ('yield' if node.__class__ is ast.Yield else 'yield from',), provider.getSourceReference())
    while provider.isExpressionOutlineFunction():
        provider = provider.getParentVariableProvider()
    assert provider.isExpressionGeneratorObjectBody() or provider.isExpressionAsyncgenObjectBody() or provider.isExpressionCoroutineObjectBody(), provider

def buildYieldNode(provider, node, source_ref):
    if False:
        print('Hello World!')
    _checkInsideGenerator('yield', provider, node, source_ref)
    if node.value is not None:
        return ExpressionYield(expression=buildNode(provider, node.value, source_ref), source_ref=source_ref)
    else:
        return ExpressionYield(expression=ExpressionConstantNoneRef(source_ref=source_ref), source_ref=source_ref)

def buildYieldFromNode(provider, node, source_ref):
    if False:
        return 10
    assert python_version >= 768
    _checkInsideGenerator('yield_from', provider, node, source_ref)
    return ExpressionYieldFrom(expression=buildNode(provider, node.value, source_ref), source_ref=source_ref)

def buildAwaitNode(provider, node, source_ref):
    if False:
        print('Hello World!')
    _checkInsideGenerator('await', provider, node, source_ref)
    return ExpressionYieldFromAwaitable(expression=ExpressionAsyncWait(expression=buildNode(provider, node.value, source_ref), source_ref=source_ref), source_ref=source_ref)