""" The type nodes.

These ones deal with types and they are great for optimization. We need to know
them, their relationship or check for them in re-formulations.

"""
from nuitka.__past__ import GenericAlias
from nuitka.Builtins import builtin_names
from nuitka.Options import isExperimental
from .BuiltinRefNodes import ExpressionBuiltinAnonymousRef, ExpressionBuiltinRef, makeExpressionBuiltinRef
from .ChildrenHavingMixins import ChildHavingClsMixin, ChildrenExpressionBuiltinIssubclassMixin, ChildrenExpressionBuiltinSuper0Mixin, ChildrenExpressionBuiltinSuper1Mixin, ChildrenExpressionBuiltinSuper2Mixin, ChildrenHavingInstanceClassesMixin
from .ExpressionBases import ExpressionBase, ExpressionBuiltinSingleArgBase
from .ExpressionBasesGenerated import ExpressionSubtypeCheckBase
from .ExpressionShapeMixins import ExpressionBoolShapeExactMixin
from .NodeBases import SideEffectsFromChildrenMixin
from .NodeMakingHelpers import makeConstantReplacementNode, wrapExpressionWithNodeSideEffects
from .shapes.BuiltinTypeShapes import tshape_type

class ExpressionBuiltinType1(ExpressionBuiltinSingleArgBase):
    kind = 'EXPRESSION_BUILTIN_TYPE1'

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        value = self.subnode_value
        type_shape = value.getTypeShape()
        if type_shape is not None:
            type_name = type_shape.getTypeName()
            if type_name is not None:
                if isExperimental('assume-type-complete') and hasattr(type_shape, 'typical_value'):
                    result = makeConstantReplacementNode(constant=type(getattr(type_shape, 'typical_value')), node=self, user_provided=False)
                elif type_name in __builtins__:
                    result = ExpressionBuiltinRef(builtin_name=type_name, source_ref=value.getSourceReference())
                else:
                    result = None
                if result is not None:
                    result = wrapExpressionWithNodeSideEffects(new_node=result, old_node=value)
                    return (result, 'new_builtin', "Replaced predictable type lookup with builtin type '%s'." % type_name)
        if value.isCompileTimeConstant():
            value = value.getCompileTimeConstant()
            if type(value) is GenericAlias:
                type_name = 'GenericAlias'
            else:
                type_name = value.__class__.__name__
            if type_name in builtin_names:
                new_node = makeExpressionBuiltinRef(builtin_name=type_name, locals_scope=None, source_ref=self.source_ref)
            else:
                new_node = ExpressionBuiltinAnonymousRef(builtin_name=type_name, source_ref=self.source_ref)
            return (new_node, 'new_builtin', "Replaced predictable type lookup with builtin type '%s'." % type_name)
        return (self, None, None)

    @staticmethod
    def getTypeShape():
        if False:
            return 10
        return tshape_type

    def computeExpressionDrop(self, statement, trace_collection):
        if False:
            while True:
                i = 10
        from .NodeMakingHelpers import makeStatementExpressionOnlyReplacementNode
        result = makeStatementExpressionOnlyReplacementNode(expression=self.subnode_value, node=statement)
        return (result, 'new_statements', 'Removed type taking for unused result.')

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        return self.subnode_value.mayRaiseException(exception_type)

    def mayHaveSideEffects(self):
        if False:
            print('Hello World!')
        return self.subnode_value.mayHaveSideEffects()

class ExpressionBuiltinSuper1(ChildrenExpressionBuiltinSuper1Mixin, ExpressionBase):
    """Two arguments form of super."""
    kind = 'EXPRESSION_BUILTIN_SUPER1'
    named_children = ('type_arg',)

    def __init__(self, type_arg, source_ref):
        if False:
            print('Hello World!')
        ChildrenExpressionBuiltinSuper1Mixin.__init__(self, type_arg=type_arg)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

class ExpressionBuiltinSuper2(ChildrenExpressionBuiltinSuper2Mixin, ExpressionBase):
    """Two arguments form of super."""
    kind = 'EXPRESSION_BUILTIN_SUPER2'
    named_children = ('type_arg', 'object_arg')

    def __init__(self, type_arg, object_arg, source_ref):
        if False:
            return 10
        ChildrenExpressionBuiltinSuper2Mixin.__init__(self, type_arg=type_arg, object_arg=object_arg)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

class ExpressionBuiltinSuper0(ChildrenExpressionBuiltinSuper0Mixin, ExpressionBase):
    """Python3 form of super, arguments determined from cells and function arguments."""
    kind = 'EXPRESSION_BUILTIN_SUPER0'
    named_children = ('type_arg', 'object_arg')

    def __init__(self, type_arg, object_arg, source_ref):
        if False:
            print('Hello World!')
        ChildrenExpressionBuiltinSuper0Mixin.__init__(self, type_arg=type_arg, object_arg=object_arg)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

class ExpressionBuiltinIsinstance(ChildrenHavingInstanceClassesMixin, ExpressionBase):
    kind = 'EXPRESSION_BUILTIN_ISINSTANCE'
    named_children = ('instance', 'classes')

    def __init__(self, instance, classes, source_ref):
        if False:
            while True:
                i = 10
        ChildrenHavingInstanceClassesMixin.__init__(self, instance=instance, classes=classes)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            return 10
        instance = self.subnode_instance
        if not instance.isCompileTimeConstant():
            trace_collection.onExceptionRaiseExit(BaseException)
            return (self, None, None)
        classes = self.subnode_classes
        if not classes.isCompileTimeConstant():
            trace_collection.onExceptionRaiseExit(BaseException)
            return (self, None, None)
        return trace_collection.getCompileTimeComputationResult(node=self, computation=lambda : isinstance(instance.getCompileTimeConstant(), classes.getCompileTimeConstant()), description="Built-in call to 'isinstance' computed.")

class ExpressionBuiltinIssubclass(ChildrenExpressionBuiltinIssubclassMixin, ExpressionBase):
    kind = 'EXPRESSION_BUILTIN_ISSUBCLASS'
    named_children = ('cls', 'classes')

    def __init__(self, cls, classes, source_ref):
        if False:
            i = 10
            return i + 15
        ChildrenExpressionBuiltinIssubclassMixin.__init__(self, cls=cls, classes=classes)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        cls = self.subnode_cls
        if not cls.isCompileTimeConstant():
            trace_collection.onExceptionRaiseExit(BaseException)
            return (self, None, None)
        classes = self.subnode_classes
        if not classes.isCompileTimeConstant():
            trace_collection.onExceptionRaiseExit(BaseException)
            return (self, None, None)
        return trace_collection.getCompileTimeComputationResult(node=self, computation=lambda : issubclass(cls.getCompileTimeConstant(), classes.getCompileTimeConstant()), description="Built-in call to 'issubclass' computed.")

class ExpressionTypeCheck(ExpressionBoolShapeExactMixin, SideEffectsFromChildrenMixin, ChildHavingClsMixin, ExpressionBase):
    kind = 'EXPRESSION_TYPE_CHECK'
    named_children = ('cls',)

    def __init__(self, cls, source_ref):
        if False:
            return 10
        ChildHavingClsMixin.__init__(self, cls=cls)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        return (self, None, None)

class ExpressionSubtypeCheck(ExpressionBoolShapeExactMixin, SideEffectsFromChildrenMixin, ExpressionSubtypeCheckBase):
    kind = 'EXPRESSION_SUBTYPE_CHECK'
    named_children = ('left', 'right')
    auto_compute_handling = 'final,no_raise'

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        return (self, None, None)