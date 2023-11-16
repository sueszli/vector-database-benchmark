""" Reformulation of call expressions.

Consult the Developer Manual for information. TODO: Add ability to sync
source code comments with Developer Manual sections.

"""
from nuitka.nodes.CallNodes import makeExpressionCall
from nuitka.nodes.ConstantRefNodes import makeConstantRefNode
from nuitka.nodes.ContainerMakingNodes import makeExpressionMakeTuple, makeExpressionMakeTupleOrConstant
from nuitka.nodes.DictionaryNodes import makeExpressionMakeDictOrConstant
from nuitka.nodes.FunctionNodes import ExpressionFunctionRef, makeExpressionFunctionCall, makeExpressionFunctionCreation
from nuitka.nodes.KeyValuePairNodes import makeExpressionPairs
from nuitka.nodes.OutlineNodes import ExpressionOutlineBody
from nuitka.nodes.ReturnNodes import StatementReturn
from nuitka.nodes.VariableAssignNodes import makeStatementAssignmentVariable
from nuitka.nodes.VariableRefNodes import ExpressionTempVariableRef
from nuitka.PythonVersions import python_version
from .ComplexCallHelperFunctions import getFunctionCallHelperDictionaryUnpacking, getFunctionCallHelperKeywordsStarDict, getFunctionCallHelperKeywordsStarList, getFunctionCallHelperKeywordsStarListStarDict, getFunctionCallHelperPosKeywordsStarDict, getFunctionCallHelperPosKeywordsStarList, getFunctionCallHelperPosKeywordsStarListStarDict, getFunctionCallHelperPosStarDict, getFunctionCallHelperPosStarList, getFunctionCallHelperPosStarListStarDict, getFunctionCallHelperStarDict, getFunctionCallHelperStarList, getFunctionCallHelperStarListStarDict
from .ReformulationDictionaryCreation import buildDictionaryUnpackingArgs
from .ReformulationSequenceCreation import buildListUnpacking
from .TreeHelpers import buildNode, buildNodeTuple, getKind, makeStatementsSequenceFromStatements

def buildCallNode(provider, node, source_ref):
    if False:
        for i in range(10):
            print('nop')
    called = buildNode(provider, node.func, source_ref)
    if python_version >= 848:
        list_star_arg = None
        dict_star_arg = None
    positional_args = []
    for node_arg in node.args[:-1]:
        if getKind(node_arg) == 'Starred':
            assert python_version >= 848
            list_star_arg = buildListUnpacking(provider, node.args, source_ref)
            positional_args = ()
            break
    else:
        if node.args and getKind(node.args[-1]) == 'Starred':
            assert python_version >= 848
            list_star_arg = buildNode(provider, node.args[-1].value, source_ref)
            positional_args = buildNodeTuple(provider, node.args[:-1], source_ref)
        else:
            positional_args = buildNodeTuple(provider, node.args, source_ref)
    keys = []
    values = []
    for keyword in node.keywords[:-1]:
        if keyword.arg is None:
            assert python_version >= 848
            outline_body = ExpressionOutlineBody(provider=provider, name='dict_unpacking_call', source_ref=source_ref)
            tmp_called = outline_body.allocateTempVariable(temp_scope=None, name='called', temp_type='object')
            helper_args = [ExpressionTempVariableRef(variable=tmp_called, source_ref=source_ref), makeExpressionMakeTuple(elements=buildDictionaryUnpackingArgs(provider=provider, keys=(keyword.arg for keyword in node.keywords), values=(keyword.value for keyword in node.keywords), source_ref=source_ref), source_ref=source_ref)]
            dict_star_arg = makeExpressionFunctionCall(function=makeExpressionFunctionCreation(function_ref=ExpressionFunctionRef(function_body=getFunctionCallHelperDictionaryUnpacking(), source_ref=source_ref), defaults=(), kw_defaults=None, annotations=None, source_ref=source_ref), values=helper_args, source_ref=source_ref)
            outline_body.setChildBody(makeStatementsSequenceFromStatements(makeStatementAssignmentVariable(variable=tmp_called, source=called, source_ref=source_ref), StatementReturn(expression=_makeCallNode(called=ExpressionTempVariableRef(variable=tmp_called, source_ref=source_ref), positional_args=positional_args, keys=keys, values=values, list_star_arg=list_star_arg, dict_star_arg=dict_star_arg, source_ref=source_ref), source_ref=source_ref)))
            return outline_body
    if node.keywords and node.keywords[-1].arg is None:
        assert python_version >= 848
        dict_star_arg = buildNode(provider, node.keywords[-1].value, source_ref)
        keywords = node.keywords[:-1]
    else:
        keywords = node.keywords
    for keyword in keywords:
        keys.append(makeConstantRefNode(constant=keyword.arg, source_ref=source_ref, user_provided=True))
        values.append(buildNode(provider, keyword.value, source_ref))
    if python_version < 848:
        list_star_arg = buildNode(provider, node.starargs, source_ref, True)
        dict_star_arg = buildNode(provider, node.kwargs, source_ref, True)
    return _makeCallNode(called=called, positional_args=positional_args, keys=keys, values=values, list_star_arg=list_star_arg, dict_star_arg=dict_star_arg, source_ref=source_ref)

def _makeCallNode(called, positional_args, keys, values, list_star_arg, dict_star_arg, source_ref):
    if False:
        print('Hello World!')
    if list_star_arg is None and dict_star_arg is None:
        result = makeExpressionCall(called=called, args=makeExpressionMakeTupleOrConstant(elements=positional_args, user_provided=True, source_ref=source_ref), kw=makeExpressionMakeDictOrConstant(makeExpressionPairs(keys=keys, values=values), user_provided=True, source_ref=source_ref), source_ref=source_ref)
        if python_version < 896:
            if values:
                result.setCompatibleSourceReference(source_ref=values[-1].getCompatibleSourceReference())
            elif positional_args:
                result.setCompatibleSourceReference(source_ref=positional_args[-1].getCompatibleSourceReference())
        return result
    else:
        key = (bool(positional_args), bool(keys), list_star_arg is not None, dict_star_arg is not None)
        table = {(True, True, True, False): getFunctionCallHelperPosKeywordsStarList, (True, False, True, False): getFunctionCallHelperPosStarList, (False, True, True, False): getFunctionCallHelperKeywordsStarList, (False, False, True, False): getFunctionCallHelperStarList, (True, True, False, True): getFunctionCallHelperPosKeywordsStarDict, (True, False, False, True): getFunctionCallHelperPosStarDict, (False, True, False, True): getFunctionCallHelperKeywordsStarDict, (False, False, False, True): getFunctionCallHelperStarDict, (True, True, True, True): getFunctionCallHelperPosKeywordsStarListStarDict, (True, False, True, True): getFunctionCallHelperPosStarListStarDict, (False, True, True, True): getFunctionCallHelperKeywordsStarListStarDict, (False, False, True, True): getFunctionCallHelperStarListStarDict}
        get_helper = table[key]
        helper_args = [called]
        if positional_args:
            helper_args.append(makeExpressionMakeTupleOrConstant(elements=positional_args, user_provided=True, source_ref=source_ref))
        if python_version >= 848 and list_star_arg is not None:
            helper_args.append(list_star_arg)
        if keys:
            helper_args.append(makeExpressionMakeDictOrConstant(pairs=makeExpressionPairs(keys=keys, values=values), user_provided=True, source_ref=source_ref))
        if python_version < 848 and list_star_arg is not None:
            helper_args.append(list_star_arg)
        if dict_star_arg is not None:
            helper_args.append(dict_star_arg)
        result = makeExpressionFunctionCall(function=makeExpressionFunctionCreation(function_ref=ExpressionFunctionRef(function_body=get_helper(), source_ref=source_ref), defaults=(), kw_defaults=None, annotations=None, source_ref=source_ref), values=helper_args, source_ref=source_ref)
        if python_version < 896:
            result.setCompatibleSourceReference(source_ref=helper_args[-1].getCompatibleSourceReference())
        return result