""" Format related nodes format/bin/oct/hex/ascii.

These will most often be used for outputs, and the hope is, the type prediction or the
result prediction will help to be smarter, but generally these should not be that much
about performance critical.

"""
from nuitka.PythonVersions import python_version
from nuitka.specs import BuiltinParameterSpecs
from .ChildrenHavingMixins import ChildrenHavingValueFormatSpecOptionalAutoNoneEmptyStrMixin
from .ExpressionBases import ExpressionBase, ExpressionBuiltinSingleArgBase
from .ExpressionShapeMixins import ExpressionIntOrLongExactMixin, ExpressionStrOrUnicodeExactMixin, ExpressionStrShapeExactMixin
from .NodeMakingHelpers import makeStatementExpressionOnlyReplacementNode

class ExpressionBuiltinFormat(ExpressionStrOrUnicodeExactMixin, ChildrenHavingValueFormatSpecOptionalAutoNoneEmptyStrMixin, ExpressionBase):
    kind = 'EXPRESSION_BUILTIN_FORMAT'
    named_children = ('value', 'format_spec|auto_none_empty_str')

    def __init__(self, value, format_spec, source_ref):
        if False:
            print('Hello World!')
        ChildrenHavingValueFormatSpecOptionalAutoNoneEmptyStrMixin.__init__(self, value=value, format_spec=format_spec)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        value = self.subnode_value
        format_spec = self.subnode_format_spec
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        if format_spec is None:
            if value.hasShapeStrOrUnicodeExact():
                return (value, 'new_expression', "Removed useless 'format' on '%s' value." % value.getTypeShape().getTypeName())
        return (self, None, None)

class ExpressionBuiltinAscii(ExpressionStrShapeExactMixin, ExpressionBuiltinSingleArgBase):
    kind = 'EXPRESSION_BUILTIN_ASCII'
    if python_version >= 768:
        builtin_spec = BuiltinParameterSpecs.builtin_ascii_spec

class ExpressionBuiltinBin(ExpressionStrShapeExactMixin, ExpressionBuiltinSingleArgBase):
    kind = 'EXPRESSION_BUILTIN_BIN'
    builtin_spec = BuiltinParameterSpecs.builtin_bin_spec

class ExpressionBuiltinOct(ExpressionStrShapeExactMixin, ExpressionBuiltinSingleArgBase):
    kind = 'EXPRESSION_BUILTIN_OCT'
    builtin_spec = BuiltinParameterSpecs.builtin_oct_spec

class ExpressionBuiltinHex(ExpressionStrShapeExactMixin, ExpressionBuiltinSingleArgBase):
    kind = 'EXPRESSION_BUILTIN_HEX'
    builtin_spec = BuiltinParameterSpecs.builtin_hex_spec

class ExpressionBuiltinId(ExpressionIntOrLongExactMixin, ExpressionBuiltinSingleArgBase):
    kind = 'EXPRESSION_BUILTIN_ID'
    builtin_spec = BuiltinParameterSpecs.builtin_id_spec

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            while True:
                i = 10
        return self.subnode_value.mayRaiseException(exception_type)

    def getIntValue(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def computeExpressionDrop(self, statement, trace_collection):
        if False:
            while True:
                i = 10
        result = makeStatementExpressionOnlyReplacementNode(expression=self.subnode_value, node=self)
        del self.parent
        return (result, 'new_statements', 'Removed id taking for unused result.')

    def mayHaveSideEffects(self):
        if False:
            i = 10
            return i + 15
        return self.subnode_value.mayHaveSideEffects()

    def extractSideEffects(self):
        if False:
            return 10
        return self.subnode_value.extractSideEffects()