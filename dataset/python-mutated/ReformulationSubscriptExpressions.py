""" Reformulation of subscript into slicing.

For Python2, there is a difference between x[a], x[a:b], x[a:b:c] whereas
Python3 treats the later by making a slice object, Python2 tries to have
special slice access, if available, or building a slice object only at the
end.

Consult the Developer Manual for information. TODO: Add ability to sync
source code comments with Developer Manual sections.
"""
from nuitka.nodes.ConstantRefNodes import ExpressionConstantEllipsisRef
from nuitka.nodes.SliceNodes import ExpressionSliceLookup, makeExpressionBuiltinSlice
from nuitka.nodes.SubscriptNodes import ExpressionSubscriptLookup
from nuitka.PythonVersions import python_version
from .ReformulationAssignmentStatements import buildExtSliceNode
from .TreeHelpers import buildNode, getKind

def buildSubscriptNode(provider, node, source_ref):
    if False:
        return 10
    assert getKind(node.ctx) == 'Load', source_ref
    kind = getKind(node.slice)
    if kind == 'Index':
        return ExpressionSubscriptLookup(expression=buildNode(provider, node.value, source_ref), subscript=buildNode(provider, node.slice.value, source_ref), source_ref=source_ref)
    elif kind == 'Slice':
        lower = buildNode(provider=provider, node=node.slice.lower, source_ref=source_ref, allow_none=True)
        upper = buildNode(provider=provider, node=node.slice.upper, source_ref=source_ref, allow_none=True)
        step = buildNode(provider=provider, node=node.slice.step, source_ref=source_ref, allow_none=True)
        use_slice_object = step is not None or python_version >= 768
        if use_slice_object:
            return ExpressionSubscriptLookup(expression=buildNode(provider, node.value, source_ref), subscript=makeExpressionBuiltinSlice(start=lower, stop=upper, step=step, source_ref=source_ref), source_ref=source_ref)
        else:
            return ExpressionSliceLookup(expression=buildNode(provider, node.value, source_ref), lower=lower, upper=upper, source_ref=source_ref)
    elif kind == 'ExtSlice':
        return ExpressionSubscriptLookup(expression=buildNode(provider, node.value, source_ref), subscript=buildExtSliceNode(provider, node, source_ref), source_ref=source_ref)
    elif kind == 'Ellipsis':
        return ExpressionSubscriptLookup(expression=buildNode(provider, node.value, source_ref), subscript=ExpressionConstantEllipsisRef(source_ref=source_ref), source_ref=source_ref)
    elif python_version >= 912:
        return ExpressionSubscriptLookup(expression=buildNode(provider, node.value, source_ref), subscript=buildNode(provider, node.slice, source_ref), source_ref=source_ref)
    else:
        assert False, kind