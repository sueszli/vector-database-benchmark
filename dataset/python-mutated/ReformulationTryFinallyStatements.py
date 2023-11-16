""" Reformulation of try/finally statements.

Consult the Developer Manual for information. TODO: Add ability to sync
source code comments with Developer Manual sections.

"""
from nuitka.nodes.LoopNodes import StatementLoopBreak, StatementLoopContinue
from nuitka.nodes.ReturnNodes import StatementReturnReturnedValue
from nuitka.nodes.StatementNodes import StatementPreserveFrameException, StatementPublishException, StatementRestoreFrameException, StatementsSequence
from nuitka.nodes.TryNodes import StatementTry
from nuitka.PythonVersions import python_version
from .TreeHelpers import buildStatementsNode, getStatementsAppended, getStatementsPrepended, makeReraiseExceptionStatement, makeStatementsSequence, makeStatementsSequenceFromStatement, makeStatementsSequenceFromStatements, mergeStatements, popBuildContext, pushBuildContext

def _checkCloning(final, provider):
    if False:
        return 10
    final2 = final.makeClone()
    final2.parent = provider
    import nuitka.TreeXML
    if nuitka.TreeXML.Element is not None:
        f1 = final.asXml()
        f2 = final2.asXml()

        def compare(a, b):
            if False:
                return 10
            for (c1, c2) in zip(a, b):
                compare(c1, c2)
            assert a.attrib == b.attrib, (a.attrib, b.attrib)
        compare(f1, f2)

def makeTryFinallyStatement(provider, tried, final, source_ref, public_exc=False):
    if False:
        i = 10
        return i + 15
    if type(tried) in (tuple, list):
        if tried:
            tried = makeStatementsSequenceFromStatements(*tried)
        else:
            tried = None
    if type(final) in (tuple, list):
        if final:
            final = StatementsSequence(statements=mergeStatements(final, False), source_ref=source_ref)
        else:
            final = None
    if tried is not None and (not tried.isStatementsSequence()):
        tried = makeStatementsSequenceFromStatement(tried)
    if final is not None and (not final.isStatementsSequence()):
        final = makeStatementsSequenceFromStatement(final)
    if tried is None:
        return final
    if final is None:
        return tried
    if provider is not None:
        tried.parent = provider
        final.parent = provider

    def getFinal():
        if False:
            for i in range(10):
                print('nop')
        if hasattr(getFinal, 'used'):
            return final.makeClone()
        else:
            getFinal.used = True
            return final
    if tried.mayRaiseException(BaseException):
        except_handler = getStatementsAppended(statement_sequence=getFinal(), statements=makeReraiseExceptionStatement(source_ref=source_ref))
        if public_exc:
            preserver_id = provider.allocatePreserverId()
            except_handler = getStatementsPrepended(statement_sequence=except_handler, statements=(StatementPreserveFrameException(preserver_id=preserver_id, source_ref=source_ref.atInternal()), StatementPublishException(source_ref=source_ref)))
            except_handler = makeTryFinallyStatement(provider=provider, tried=except_handler, final=StatementRestoreFrameException(preserver_id=preserver_id, source_ref=source_ref.atInternal()), public_exc=False, source_ref=source_ref)
            except_handler = makeStatementsSequenceFromStatement(statement=except_handler)
    else:
        except_handler = None
    if tried.mayBreak():
        break_handler = getStatementsAppended(statement_sequence=getFinal(), statements=StatementLoopBreak(source_ref=source_ref))
    else:
        break_handler = None
    if tried.mayContinue():
        continue_handler = getStatementsAppended(statement_sequence=getFinal(), statements=StatementLoopContinue(source_ref=source_ref))
    else:
        continue_handler = None
    if tried.mayReturn():
        return_handler = getStatementsAppended(statement_sequence=getFinal(), statements=StatementReturnReturnedValue(source_ref=source_ref))
    else:
        return_handler = None
    result = StatementTry(tried=tried, except_handler=except_handler, break_handler=break_handler, continue_handler=continue_handler, return_handler=return_handler, source_ref=source_ref)
    if result.isStatementAborting():
        return result
    else:
        return makeStatementsSequence(statements=(result, getFinal()), allow_none=False, source_ref=source_ref)

def buildTryFinallyNode(provider, build_tried, node, source_ref):
    if False:
        print('Hello World!')
    if python_version < 768:
        pushBuildContext('finally')
        final = buildStatementsNode(provider=provider, nodes=node.finalbody, source_ref=source_ref)
        popBuildContext()
        return makeTryFinallyStatement(provider=provider, tried=build_tried(), final=final, source_ref=source_ref, public_exc=False)
    else:
        tried = build_tried()
        pushBuildContext('finally')
        final = buildStatementsNode(provider=provider, nodes=node.finalbody, source_ref=source_ref)
        popBuildContext()
        return makeTryFinallyStatement(provider=provider, tried=tried, final=final, public_exc=True, source_ref=source_ref)