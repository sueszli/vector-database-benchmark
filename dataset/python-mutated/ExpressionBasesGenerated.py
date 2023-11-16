"""Children having expression bases

WARNING, this code is GENERATED. Modify the template ChildrenHavingMixin.py.j2 instead!

spell-checker: ignore append capitalize casefold center clear copy count decode encode endswith expandtabs extend find format formatmap fromkeys get haskey index insert isalnum isalpha isascii isdecimal isdigit isidentifier islower isnumeric isprintable isspace istitle isupper items iteritems iterkeys itervalues join keys ljust lower lstrip maketrans partition pop popitem prepare remove replace reverse rfind rindex rjust rpartition rsplit rstrip setdefault sort split splitlines startswith strip swapcase title translate update upper values viewitems viewkeys viewvalues zfill
spell-checker: ignore args chars count default delete encoding end errors fillchar index item iterable keepends key kwargs maxsplit new old pairs prefix sep start stop sub suffix table tabsize value width
"""
from abc import abstractmethod
from .ExpressionBases import ExpressionBase
from .NodeMakingHelpers import wrapExpressionWithSideEffects

class NoChildHavingFinalNoRaiseMixin(ExpressionBase):
    __slots__ = ()

    def finalize(self):
        if False:
            while True:
                i = 10
        del self.parent

    def computeExpressionRaw(self, trace_collection):
        if False:
            return 10
        'Compute an expression.\n\n        Default behavior is to just visit the child expressions first, and\n        then the node "computeExpression". For a few cases this needs to\n        be overloaded, e.g. conditional expressions.\n        '
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            print('Hello World!')
        return False

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            i = 10
            return i + 15
        'Collect variable reads and writes of child nodes.'
ExpressionImportlibMetadataBackportEntryPointValueRefBase = NoChildHavingFinalNoRaiseMixin
ExpressionImportlibMetadataEntryPointValueRefBase = NoChildHavingFinalNoRaiseMixin

class ChildHavingArgsTupleFinalNoRaiseMixin(ExpressionBase):
    __slots__ = ()

    def __init__(self, args, source_ref):
        if False:
            return 10
        assert type(args) is tuple
        for val in args:
            val.parent = self
        self.subnode_args = args
        ExpressionBase.__init__(self, source_ref)

    def getVisitableNodes(self):
        if False:
            for i in range(10):
                print('nop')
        'The visitable nodes, with tuple values flattened.'
        return self.subnode_args

    def getVisitableNodesNamed(self):
        if False:
            print('Hello World!')
        'Named children dictionary.\n\n        For use in cloning nodes, debugging and XML output.\n        '
        return (('args', self.subnode_args),)

    def replaceChild(self, old_node, new_node):
        if False:
            while True:
                i = 10
        value = self.subnode_args
        if old_node in value:
            if new_node is not None:
                new_node.parent = self
                self.subnode_args = tuple((val if val is not old_node else new_node for val in value))
            else:
                self.subnode_args = tuple((val for val in value if val is not old_node))
            return
        raise AssertionError("Didn't find child", old_node, 'in', self)

    def getCloneArgs(self):
        if False:
            i = 10
            return i + 15
        'Get clones of all children to pass for a new node.\n\n        Needs to make clones of child nodes too.\n        '
        values = {'args': tuple((v.makeClone() for v in self.subnode_args))}
        values.update(self.getDetails())
        return values

    def finalize(self):
        if False:
            return 10
        del self.parent
        for c in self.subnode_args:
            c.finalize()
        del self.subnode_args

    def computeExpressionRaw(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        'Compute an expression.\n\n        Default behavior is to just visit the child expressions first, and\n        then the node "computeExpression". For a few cases this needs to\n        be overloaded, e.g. conditional expressions.\n        '
        old_subnode_args = self.subnode_args
        for sub_expression in old_subnode_args:
            expression = trace_collection.onExpression(sub_expression)
            if expression.willRaiseAnyException():
                wrapped_expression = wrapExpressionWithSideEffects(side_effects=self.subnode_args[:old_subnode_args.index(sub_expression)], old_node=sub_expression, new_node=expression)
                return (wrapped_expression, 'new_raise', lambda : "For '%s' the child expression '%s' will raise." % (self.getChildNameNice(), expression.getChildNameNice()))
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            return 10
        return False

    def mayRaiseException(self, exception_type):
        if False:
            for i in range(10):
                print('nop')
        return any((value.mayRaiseException(exception_type) for value in self.subnode_args))

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            i = 10
            return i + 15
        'Collect variable reads and writes of child nodes.'
        for element in self.subnode_args:
            element.collectVariableAccesses(emit_read, emit_write)
ExpressionBuiltinMakeExceptionBase = ChildHavingArgsTupleFinalNoRaiseMixin

class ChildrenHavingArgsTupleNameOptionalPathOptionalFinalNoRaiseMixin(ExpressionBase):
    __slots__ = ()

    def __init__(self, args, name, path, source_ref):
        if False:
            for i in range(10):
                print('nop')
        assert type(args) is tuple
        for val in args:
            val.parent = self
        self.subnode_args = args
        if name is not None:
            name.parent = self
        self.subnode_name = name
        if path is not None:
            path.parent = self
        self.subnode_path = path
        ExpressionBase.__init__(self, source_ref)

    def getVisitableNodes(self):
        if False:
            while True:
                i = 10
        'The visitable nodes, with tuple values flattened.'
        result = []
        result.extend(self.subnode_args)
        value = self.subnode_name
        if value is None:
            pass
        else:
            result.append(value)
        value = self.subnode_path
        if value is None:
            pass
        else:
            result.append(value)
        return tuple(result)

    def getVisitableNodesNamed(self):
        if False:
            for i in range(10):
                print('nop')
        'Named children dictionary.\n\n        For use in cloning nodes, debugging and XML output.\n        '
        return (('args', self.subnode_args), ('name', self.subnode_name), ('path', self.subnode_path))

    def replaceChild(self, old_node, new_node):
        if False:
            print('Hello World!')
        value = self.subnode_args
        if old_node in value:
            if new_node is not None:
                new_node.parent = self
                self.subnode_args = tuple((val if val is not old_node else new_node for val in value))
            else:
                self.subnode_args = tuple((val for val in value if val is not old_node))
            return
        value = self.subnode_name
        if old_node is value:
            if new_node is not None:
                new_node.parent = self
            self.subnode_name = new_node
            return
        value = self.subnode_path
        if old_node is value:
            if new_node is not None:
                new_node.parent = self
            self.subnode_path = new_node
            return
        raise AssertionError("Didn't find child", old_node, 'in', self)

    def getCloneArgs(self):
        if False:
            i = 10
            return i + 15
        'Get clones of all children to pass for a new node.\n\n        Needs to make clones of child nodes too.\n        '
        values = {'args': tuple((v.makeClone() for v in self.subnode_args)), 'name': self.subnode_name.makeClone() if self.subnode_name is not None else None, 'path': self.subnode_path.makeClone() if self.subnode_path is not None else None}
        values.update(self.getDetails())
        return values

    def finalize(self):
        if False:
            return 10
        del self.parent
        for c in self.subnode_args:
            c.finalize()
        del self.subnode_args
        if self.subnode_name is not None:
            self.subnode_name.finalize()
        del self.subnode_name
        if self.subnode_path is not None:
            self.subnode_path.finalize()
        del self.subnode_path

    def computeExpressionRaw(self, trace_collection):
        if False:
            i = 10
            return i + 15
        'Compute an expression.\n\n        Default behavior is to just visit the child expressions first, and\n        then the node "computeExpression". For a few cases this needs to\n        be overloaded, e.g. conditional expressions.\n        '
        for (count, sub_expression) in enumerate(self.getVisitableNodes()):
            expression = trace_collection.onExpression(sub_expression)
            if expression.willRaiseAnyException():
                sub_expressions = self.getVisitableNodes()
                wrapped_expression = wrapExpressionWithSideEffects(side_effects=sub_expressions[:count], old_node=sub_expression, new_node=expression)
                return (wrapped_expression, 'new_raise', lambda : "For '%s' the child expression '%s' will raise." % (self.getChildNameNice(), expression.getChildNameNice()))
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            for i in range(10):
                print('nop')
        return False

    def mayRaiseException(self, exception_type):
        if False:
            i = 10
            return i + 15
        return any((value.mayRaiseException(exception_type) for value in self.subnode_args)) or (self.subnode_name is not None and self.subnode_name.mayRaiseException(exception_type)) or (self.subnode_path is not None and self.subnode_path.mayRaiseException(exception_type))

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            i = 10
            return i + 15
        'Collect variable reads and writes of child nodes.'
        for element in self.subnode_args:
            element.collectVariableAccesses(emit_read, emit_write)
        subnode_name = self.subnode_name
        if subnode_name is not None:
            self.subnode_name.collectVariableAccesses(emit_read, emit_write)
        subnode_path = self.subnode_path
        if subnode_path is not None:
            self.subnode_path.collectVariableAccesses(emit_read, emit_write)
ExpressionBuiltinMakeExceptionImportErrorBase = ChildrenHavingArgsTupleNameOptionalPathOptionalFinalNoRaiseMixin

class ChildrenHavingCallableArgSentinelFinalMixin(ExpressionBase):
    __slots__ = ()

    def __init__(self, callable_arg, sentinel, source_ref):
        if False:
            while True:
                i = 10
        callable_arg.parent = self
        self.subnode_callable_arg = callable_arg
        sentinel.parent = self
        self.subnode_sentinel = sentinel
        ExpressionBase.__init__(self, source_ref)

    def getVisitableNodes(self):
        if False:
            i = 10
            return i + 15
        'The visitable nodes, with tuple values flattened.'
        return (self.subnode_callable_arg, self.subnode_sentinel)

    def getVisitableNodesNamed(self):
        if False:
            while True:
                i = 10
        'Named children dictionary.\n\n        For use in cloning nodes, debugging and XML output.\n        '
        return (('callable_arg', self.subnode_callable_arg), ('sentinel', self.subnode_sentinel))

    def replaceChild(self, old_node, new_node):
        if False:
            for i in range(10):
                print('nop')
        value = self.subnode_callable_arg
        if old_node is value:
            new_node.parent = self
            self.subnode_callable_arg = new_node
            return
        value = self.subnode_sentinel
        if old_node is value:
            new_node.parent = self
            self.subnode_sentinel = new_node
            return
        raise AssertionError("Didn't find child", old_node, 'in', self)

    def getCloneArgs(self):
        if False:
            i = 10
            return i + 15
        'Get clones of all children to pass for a new node.\n\n        Needs to make clones of child nodes too.\n        '
        values = {'callable_arg': self.subnode_callable_arg.makeClone(), 'sentinel': self.subnode_sentinel.makeClone()}
        values.update(self.getDetails())
        return values

    def finalize(self):
        if False:
            while True:
                i = 10
        del self.parent
        self.subnode_callable_arg.finalize()
        del self.subnode_callable_arg
        self.subnode_sentinel.finalize()
        del self.subnode_sentinel

    def computeExpressionRaw(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        'Compute an expression.\n\n        Default behavior is to just visit the child expressions first, and\n        then the node "computeExpression". For a few cases this needs to\n        be overloaded, e.g. conditional expressions.\n        '
        for (count, sub_expression) in enumerate(self.getVisitableNodes()):
            expression = trace_collection.onExpression(sub_expression)
            if expression.willRaiseAnyException():
                sub_expressions = self.getVisitableNodes()
                wrapped_expression = wrapExpressionWithSideEffects(side_effects=sub_expressions[:count], old_node=sub_expression, new_node=expression)
                return (wrapped_expression, 'new_raise', lambda : "For '%s' the child expression '%s' will raise." % (self.getChildNameNice(), expression.getChildNameNice()))
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            return 10
        'Collect variable reads and writes of child nodes.'
        self.subnode_callable_arg.collectVariableAccesses(emit_read, emit_write)
        self.subnode_sentinel.collectVariableAccesses(emit_read, emit_write)
ExpressionBuiltinIter2Base = ChildrenHavingCallableArgSentinelFinalMixin

class ChildHavingDistributionNameFinalChildrenMixin(ExpressionBase):
    __slots__ = ()

    def __init__(self, distribution_name, source_ref):
        if False:
            print('Hello World!')
        distribution_name.parent = self
        self.subnode_distribution_name = distribution_name
        ExpressionBase.__init__(self, source_ref)

    def getVisitableNodes(self):
        if False:
            i = 10
            return i + 15
        'The visitable nodes, with tuple values flattened.'
        return (self.subnode_distribution_name,)

    def getVisitableNodesNamed(self):
        if False:
            print('Hello World!')
        'Named children dictionary.\n\n        For use in cloning nodes, debugging and XML output.\n        '
        return (('distribution_name', self.subnode_distribution_name),)

    def replaceChild(self, old_node, new_node):
        if False:
            while True:
                i = 10
        value = self.subnode_distribution_name
        if old_node is value:
            new_node.parent = self
            self.subnode_distribution_name = new_node
            return
        raise AssertionError("Didn't find child", old_node, 'in', self)

    def getCloneArgs(self):
        if False:
            for i in range(10):
                print('nop')
        'Get clones of all children to pass for a new node.\n\n        Needs to make clones of child nodes too.\n        '
        values = {'distribution_name': self.subnode_distribution_name.makeClone()}
        values.update(self.getDetails())
        return values

    def finalize(self):
        if False:
            i = 10
            return i + 15
        del self.parent
        self.subnode_distribution_name.finalize()
        del self.subnode_distribution_name

    def computeExpressionRaw(self, trace_collection):
        if False:
            print('Hello World!')
        'Compute an expression.\n\n        Default behavior is to just visit the child expressions first, and\n        then the node "computeExpression". For a few cases this needs to\n        be overloaded, e.g. conditional expressions.\n        '
        return self.computeExpression(trace_collection)

    @abstractmethod
    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        'Must be overloaded for non-final node.'

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            return 10
        'Collect variable reads and writes of child nodes.'
        self.subnode_distribution_name.collectVariableAccesses(emit_read, emit_write)
ExpressionImportlibMetadataBackportDistributionFailedCallBase = ChildHavingDistributionNameFinalChildrenMixin
ExpressionImportlibMetadataDistributionFailedCallBase = ChildHavingDistributionNameFinalChildrenMixin

class ChildHavingElementsTupleFinalNoRaiseMixin(ExpressionBase):
    __slots__ = ()

    def __init__(self, elements, source_ref):
        if False:
            i = 10
            return i + 15
        assert type(elements) is tuple
        for val in elements:
            val.parent = self
        self.subnode_elements = elements
        ExpressionBase.__init__(self, source_ref)

    def getVisitableNodes(self):
        if False:
            while True:
                i = 10
        'The visitable nodes, with tuple values flattened.'
        return self.subnode_elements

    def getVisitableNodesNamed(self):
        if False:
            i = 10
            return i + 15
        'Named children dictionary.\n\n        For use in cloning nodes, debugging and XML output.\n        '
        return (('elements', self.subnode_elements),)

    def replaceChild(self, old_node, new_node):
        if False:
            i = 10
            return i + 15
        value = self.subnode_elements
        if old_node in value:
            if new_node is not None:
                new_node.parent = self
                self.subnode_elements = tuple((val if val is not old_node else new_node for val in value))
            else:
                self.subnode_elements = tuple((val for val in value if val is not old_node))
            return
        raise AssertionError("Didn't find child", old_node, 'in', self)

    def getCloneArgs(self):
        if False:
            return 10
        'Get clones of all children to pass for a new node.\n\n        Needs to make clones of child nodes too.\n        '
        values = {'elements': tuple((v.makeClone() for v in self.subnode_elements))}
        values.update(self.getDetails())
        return values

    def finalize(self):
        if False:
            for i in range(10):
                print('nop')
        del self.parent
        for c in self.subnode_elements:
            c.finalize()
        del self.subnode_elements

    def computeExpressionRaw(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        'Compute an expression.\n\n        Default behavior is to just visit the child expressions first, and\n        then the node "computeExpression". For a few cases this needs to\n        be overloaded, e.g. conditional expressions.\n        '
        old_subnode_elements = self.subnode_elements
        for sub_expression in old_subnode_elements:
            expression = trace_collection.onExpression(sub_expression)
            if expression.willRaiseAnyException():
                wrapped_expression = wrapExpressionWithSideEffects(side_effects=self.subnode_elements[:old_subnode_elements.index(sub_expression)], old_node=sub_expression, new_node=expression)
                return (wrapped_expression, 'new_raise', lambda : "For '%s' the child expression '%s' will raise." % (self.getChildNameNice(), expression.getChildNameNice()))
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            for i in range(10):
                print('nop')
        return False

    def mayRaiseException(self, exception_type):
        if False:
            while True:
                i = 10
        return any((value.mayRaiseException(exception_type) for value in self.subnode_elements))

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            return 10
        'Collect variable reads and writes of child nodes.'
        for element in self.subnode_elements:
            element.collectVariableAccesses(emit_read, emit_write)
ExpressionImportlibMetadataBackportEntryPointsValueRefBase = ChildHavingElementsTupleFinalNoRaiseMixin
ExpressionImportlibMetadataEntryPointsValueRefBase = ChildHavingElementsTupleFinalNoRaiseMixin

class ChildHavingExpressionAttributeNameMixin(ExpressionBase):
    __slots__ = ()

    def __init__(self, expression, attribute_name, source_ref):
        if False:
            print('Hello World!')
        expression.parent = self
        self.subnode_expression = expression
        self.attribute_name = attribute_name
        ExpressionBase.__init__(self, source_ref)

    def getDetails(self):
        if False:
            for i in range(10):
                print('nop')
        return {'attribute_name': self.attribute_name}

    def getVisitableNodes(self):
        if False:
            print('Hello World!')
        'The visitable nodes, with tuple values flattened.'
        return (self.subnode_expression,)

    def getVisitableNodesNamed(self):
        if False:
            while True:
                i = 10
        'Named children dictionary.\n\n        For use in cloning nodes, debugging and XML output.\n        '
        return (('expression', self.subnode_expression),)

    def replaceChild(self, old_node, new_node):
        if False:
            i = 10
            return i + 15
        value = self.subnode_expression
        if old_node is value:
            new_node.parent = self
            self.subnode_expression = new_node
            return
        raise AssertionError("Didn't find child", old_node, 'in', self)

    def getCloneArgs(self):
        if False:
            for i in range(10):
                print('nop')
        'Get clones of all children to pass for a new node.\n\n        Needs to make clones of child nodes too.\n        '
        values = {'expression': self.subnode_expression.makeClone()}
        values.update(self.getDetails())
        return values

    def finalize(self):
        if False:
            for i in range(10):
                print('nop')
        del self.parent
        self.subnode_expression.finalize()
        del self.subnode_expression

    def computeExpressionRaw(self, trace_collection):
        if False:
            print('Hello World!')
        'Compute an expression.\n\n        Default behavior is to just visit the child expressions first, and\n        then the node "computeExpression". For a few cases this needs to\n        be overloaded, e.g. conditional expressions.\n        '
        expression = trace_collection.onExpression(self.subnode_expression)
        if expression.willRaiseAnyException():
            return (expression, 'new_raise', lambda : "For '%s' the child expression '%s' will raise." % (self.getChildNameNice(), expression.getChildNameNice()))
        return self.computeExpression(trace_collection)

    @abstractmethod
    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        'Must be overloaded for non-final node.'

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            for i in range(10):
                print('nop')
        'Collect variable reads and writes of child nodes.'
        self.subnode_expression.collectVariableAccesses(emit_read, emit_write)
ExpressionAttributeLookupBase = ChildHavingExpressionAttributeNameMixin
ExpressionAttributeLookupSpecialBase = ChildHavingExpressionAttributeNameMixin

class ChildrenHavingExpressionNameRaiseWaitConstantNameMixin(ExpressionBase):
    __slots__ = ()

    def __init__(self, expression, name, source_ref):
        if False:
            for i in range(10):
                print('nop')
        expression.parent = self
        self.subnode_expression = expression
        name.parent = self
        self.subnode_name = name
        ExpressionBase.__init__(self, source_ref)

    def getVisitableNodes(self):
        if False:
            while True:
                i = 10
        'The visitable nodes, with tuple values flattened.'
        return (self.subnode_expression, self.subnode_name)

    def getVisitableNodesNamed(self):
        if False:
            return 10
        'Named children dictionary.\n\n        For use in cloning nodes, debugging and XML output.\n        '
        return (('expression', self.subnode_expression), ('name', self.subnode_name))

    def replaceChild(self, old_node, new_node):
        if False:
            for i in range(10):
                print('nop')
        value = self.subnode_expression
        if old_node is value:
            new_node.parent = self
            self.subnode_expression = new_node
            return
        value = self.subnode_name
        if old_node is value:
            new_node.parent = self
            self.subnode_name = new_node
            return
        raise AssertionError("Didn't find child", old_node, 'in', self)

    def getCloneArgs(self):
        if False:
            for i in range(10):
                print('nop')
        'Get clones of all children to pass for a new node.\n\n        Needs to make clones of child nodes too.\n        '
        values = {'expression': self.subnode_expression.makeClone(), 'name': self.subnode_name.makeClone()}
        values.update(self.getDetails())
        return values

    def finalize(self):
        if False:
            print('Hello World!')
        del self.parent
        self.subnode_expression.finalize()
        del self.subnode_expression
        self.subnode_name.finalize()
        del self.subnode_name

    def computeExpressionRaw(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        'Compute an expression.\n\n        Default behavior is to just visit the child expressions first, and\n        then the node "computeExpression". For a few cases this needs to\n        be overloaded, e.g. conditional expressions.\n        '
        for (count, sub_expression) in enumerate(self.getVisitableNodes()):
            expression = trace_collection.onExpression(sub_expression)
            if expression.willRaiseAnyException():
                sub_expressions = self.getVisitableNodes()
                wrapped_expression = wrapExpressionWithSideEffects(side_effects=sub_expressions[:count], old_node=sub_expression, new_node=expression)
                return (wrapped_expression, 'new_raise', lambda : "For '%s' the child expression '%s' will raise." % (self.getChildNameNice(), expression.getChildNameNice()))
        if self.subnode_name.isCompileTimeConstant():
            try:
                return self.computeExpressionConstantName(trace_collection)
            finally:
                trace_collection.onExceptionRaiseExit(BaseException)
        return self.computeExpression(trace_collection)

    @abstractmethod
    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        'Must be overloaded for non-final node.'

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            for i in range(10):
                print('nop')
        'Collect variable reads and writes of child nodes.'
        self.subnode_expression.collectVariableAccesses(emit_read, emit_write)
        self.subnode_name.collectVariableAccesses(emit_read, emit_write)

    @abstractmethod
    def computeExpressionConstantName(self, trace_collection):
        if False:
            i = 10
            return i + 15
        'Called when attribute name is constant.'
ExpressionBuiltinHasattrBase = ChildrenHavingExpressionNameRaiseWaitConstantNameMixin

class ChildrenHavingLeftRightFinalNoRaiseMixin(ExpressionBase):
    __slots__ = ()

    def __init__(self, left, right, source_ref):
        if False:
            print('Hello World!')
        left.parent = self
        self.subnode_left = left
        right.parent = self
        self.subnode_right = right
        ExpressionBase.__init__(self, source_ref)

    def getVisitableNodes(self):
        if False:
            while True:
                i = 10
        'The visitable nodes, with tuple values flattened.'
        return (self.subnode_left, self.subnode_right)

    def getVisitableNodesNamed(self):
        if False:
            while True:
                i = 10
        'Named children dictionary.\n\n        For use in cloning nodes, debugging and XML output.\n        '
        return (('left', self.subnode_left), ('right', self.subnode_right))

    def replaceChild(self, old_node, new_node):
        if False:
            while True:
                i = 10
        value = self.subnode_left
        if old_node is value:
            new_node.parent = self
            self.subnode_left = new_node
            return
        value = self.subnode_right
        if old_node is value:
            new_node.parent = self
            self.subnode_right = new_node
            return
        raise AssertionError("Didn't find child", old_node, 'in', self)

    def getCloneArgs(self):
        if False:
            i = 10
            return i + 15
        'Get clones of all children to pass for a new node.\n\n        Needs to make clones of child nodes too.\n        '
        values = {'left': self.subnode_left.makeClone(), 'right': self.subnode_right.makeClone()}
        values.update(self.getDetails())
        return values

    def finalize(self):
        if False:
            print('Hello World!')
        del self.parent
        self.subnode_left.finalize()
        del self.subnode_left
        self.subnode_right.finalize()
        del self.subnode_right

    def computeExpressionRaw(self, trace_collection):
        if False:
            i = 10
            return i + 15
        'Compute an expression.\n\n        Default behavior is to just visit the child expressions first, and\n        then the node "computeExpression". For a few cases this needs to\n        be overloaded, e.g. conditional expressions.\n        '
        for (count, sub_expression) in enumerate(self.getVisitableNodes()):
            expression = trace_collection.onExpression(sub_expression)
            if expression.willRaiseAnyException():
                sub_expressions = self.getVisitableNodes()
                wrapped_expression = wrapExpressionWithSideEffects(side_effects=sub_expressions[:count], old_node=sub_expression, new_node=expression)
                return (wrapped_expression, 'new_raise', lambda : "For '%s' the child expression '%s' will raise." % (self.getChildNameNice(), expression.getChildNameNice()))
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            i = 10
            return i + 15
        return False

    def mayRaiseException(self, exception_type):
        if False:
            for i in range(10):
                print('nop')
        return self.subnode_left.mayRaiseException(exception_type) or self.subnode_right.mayRaiseException(exception_type)

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            return 10
        'Collect variable reads and writes of child nodes.'
        self.subnode_left.collectVariableAccesses(emit_read, emit_write)
        self.subnode_right.collectVariableAccesses(emit_read, emit_write)
ExpressionSubtypeCheckBase = ChildrenHavingLeftRightFinalNoRaiseMixin

class ChildHavingListArgNoRaiseMixin(ExpressionBase):
    __slots__ = ()

    def __init__(self, list_arg, source_ref):
        if False:
            while True:
                i = 10
        list_arg.parent = self
        self.subnode_list_arg = list_arg
        ExpressionBase.__init__(self, source_ref)

    def getVisitableNodes(self):
        if False:
            return 10
        'The visitable nodes, with tuple values flattened.'
        return (self.subnode_list_arg,)

    def getVisitableNodesNamed(self):
        if False:
            print('Hello World!')
        'Named children dictionary.\n\n        For use in cloning nodes, debugging and XML output.\n        '
        return (('list_arg', self.subnode_list_arg),)

    def replaceChild(self, old_node, new_node):
        if False:
            while True:
                i = 10
        value = self.subnode_list_arg
        if old_node is value:
            new_node.parent = self
            self.subnode_list_arg = new_node
            return
        raise AssertionError("Didn't find child", old_node, 'in', self)

    def getCloneArgs(self):
        if False:
            print('Hello World!')
        'Get clones of all children to pass for a new node.\n\n        Needs to make clones of child nodes too.\n        '
        values = {'list_arg': self.subnode_list_arg.makeClone()}
        values.update(self.getDetails())
        return values

    def finalize(self):
        if False:
            return 10
        del self.parent
        self.subnode_list_arg.finalize()
        del self.subnode_list_arg

    def computeExpressionRaw(self, trace_collection):
        if False:
            while True:
                i = 10
        'Compute an expression.\n\n        Default behavior is to just visit the child expressions first, and\n        then the node "computeExpression". For a few cases this needs to\n        be overloaded, e.g. conditional expressions.\n        '
        expression = trace_collection.onExpression(self.subnode_list_arg)
        if expression.willRaiseAnyException():
            return (expression, 'new_raise', lambda : "For '%s' the child expression '%s' will raise." % (self.getChildNameNice(), expression.getChildNameNice()))
        return self.computeExpression(trace_collection)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            return 10
        return False

    def mayRaiseException(self, exception_type):
        if False:
            print('Hello World!')
        return self.subnode_list_arg.mayRaiseException(exception_type)

    @abstractmethod
    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        'Must be overloaded for non-final node.'

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            i = 10
            return i + 15
        'Collect variable reads and writes of child nodes.'
        self.subnode_list_arg.collectVariableAccesses(emit_read, emit_write)
ExpressionListOperationClearBase = ChildHavingListArgNoRaiseMixin
ExpressionListOperationReverseBase = ChildHavingListArgNoRaiseMixin

class ChildrenHavingListArgItemNoRaiseMixin(ExpressionBase):
    __slots__ = ()

    def __init__(self, list_arg, item, source_ref):
        if False:
            print('Hello World!')
        list_arg.parent = self
        self.subnode_list_arg = list_arg
        item.parent = self
        self.subnode_item = item
        ExpressionBase.__init__(self, source_ref)

    def getVisitableNodes(self):
        if False:
            while True:
                i = 10
        'The visitable nodes, with tuple values flattened.'
        return (self.subnode_list_arg, self.subnode_item)

    def getVisitableNodesNamed(self):
        if False:
            return 10
        'Named children dictionary.\n\n        For use in cloning nodes, debugging and XML output.\n        '
        return (('list_arg', self.subnode_list_arg), ('item', self.subnode_item))

    def replaceChild(self, old_node, new_node):
        if False:
            return 10
        value = self.subnode_list_arg
        if old_node is value:
            new_node.parent = self
            self.subnode_list_arg = new_node
            return
        value = self.subnode_item
        if old_node is value:
            new_node.parent = self
            self.subnode_item = new_node
            return
        raise AssertionError("Didn't find child", old_node, 'in', self)

    def getCloneArgs(self):
        if False:
            for i in range(10):
                print('nop')
        'Get clones of all children to pass for a new node.\n\n        Needs to make clones of child nodes too.\n        '
        values = {'list_arg': self.subnode_list_arg.makeClone(), 'item': self.subnode_item.makeClone()}
        values.update(self.getDetails())
        return values

    def finalize(self):
        if False:
            return 10
        del self.parent
        self.subnode_list_arg.finalize()
        del self.subnode_list_arg
        self.subnode_item.finalize()
        del self.subnode_item

    def computeExpressionRaw(self, trace_collection):
        if False:
            return 10
        'Compute an expression.\n\n        Default behavior is to just visit the child expressions first, and\n        then the node "computeExpression". For a few cases this needs to\n        be overloaded, e.g. conditional expressions.\n        '
        for (count, sub_expression) in enumerate(self.getVisitableNodes()):
            expression = trace_collection.onExpression(sub_expression)
            if expression.willRaiseAnyException():
                sub_expressions = self.getVisitableNodes()
                wrapped_expression = wrapExpressionWithSideEffects(side_effects=sub_expressions[:count], old_node=sub_expression, new_node=expression)
                return (wrapped_expression, 'new_raise', lambda : "For '%s' the child expression '%s' will raise." % (self.getChildNameNice(), expression.getChildNameNice()))
        return self.computeExpression(trace_collection)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            return 10
        return False

    def mayRaiseException(self, exception_type):
        if False:
            for i in range(10):
                print('nop')
        return self.subnode_list_arg.mayRaiseException(exception_type) or self.subnode_item.mayRaiseException(exception_type)

    @abstractmethod
    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        'Must be overloaded for non-final node.'

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            for i in range(10):
                print('nop')
        'Collect variable reads and writes of child nodes.'
        self.subnode_list_arg.collectVariableAccesses(emit_read, emit_write)
        self.subnode_item.collectVariableAccesses(emit_read, emit_write)
ExpressionListOperationAppendBase = ChildrenHavingListArgItemNoRaiseMixin

class ChildrenHavingListArgValueFinalNoRaiseMixin(ExpressionBase):
    __slots__ = ()

    def __init__(self, list_arg, value, source_ref):
        if False:
            return 10
        list_arg.parent = self
        self.subnode_list_arg = list_arg
        value.parent = self
        self.subnode_value = value
        ExpressionBase.__init__(self, source_ref)

    def getVisitableNodes(self):
        if False:
            print('Hello World!')
        'The visitable nodes, with tuple values flattened.'
        return (self.subnode_list_arg, self.subnode_value)

    def getVisitableNodesNamed(self):
        if False:
            i = 10
            return i + 15
        'Named children dictionary.\n\n        For use in cloning nodes, debugging and XML output.\n        '
        return (('list_arg', self.subnode_list_arg), ('value', self.subnode_value))

    def replaceChild(self, old_node, new_node):
        if False:
            print('Hello World!')
        value = self.subnode_list_arg
        if old_node is value:
            new_node.parent = self
            self.subnode_list_arg = new_node
            return
        value = self.subnode_value
        if old_node is value:
            new_node.parent = self
            self.subnode_value = new_node
            return
        raise AssertionError("Didn't find child", old_node, 'in', self)

    def getCloneArgs(self):
        if False:
            for i in range(10):
                print('nop')
        'Get clones of all children to pass for a new node.\n\n        Needs to make clones of child nodes too.\n        '
        values = {'list_arg': self.subnode_list_arg.makeClone(), 'value': self.subnode_value.makeClone()}
        values.update(self.getDetails())
        return values

    def finalize(self):
        if False:
            i = 10
            return i + 15
        del self.parent
        self.subnode_list_arg.finalize()
        del self.subnode_list_arg
        self.subnode_value.finalize()
        del self.subnode_value

    def computeExpressionRaw(self, trace_collection):
        if False:
            print('Hello World!')
        'Compute an expression.\n\n        Default behavior is to just visit the child expressions first, and\n        then the node "computeExpression". For a few cases this needs to\n        be overloaded, e.g. conditional expressions.\n        '
        for (count, sub_expression) in enumerate(self.getVisitableNodes()):
            expression = trace_collection.onExpression(sub_expression)
            if expression.willRaiseAnyException():
                sub_expressions = self.getVisitableNodes()
                wrapped_expression = wrapExpressionWithSideEffects(side_effects=sub_expressions[:count], old_node=sub_expression, new_node=expression)
                return (wrapped_expression, 'new_raise', lambda : "For '%s' the child expression '%s' will raise." % (self.getChildNameNice(), expression.getChildNameNice()))
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            i = 10
            return i + 15
        return False

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        return self.subnode_list_arg.mayRaiseException(exception_type) or self.subnode_value.mayRaiseException(exception_type)

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            i = 10
            return i + 15
        'Collect variable reads and writes of child nodes.'
        self.subnode_list_arg.collectVariableAccesses(emit_read, emit_write)
        self.subnode_value.collectVariableAccesses(emit_read, emit_write)
ExpressionListOperationCountBase = ChildrenHavingListArgValueFinalNoRaiseMixin

class ChildHavingPairsTupleFinalNoRaiseMixin(ExpressionBase):
    __slots__ = ()

    def __init__(self, pairs, source_ref):
        if False:
            print('Hello World!')
        assert type(pairs) is tuple
        for val in pairs:
            val.parent = self
        self.subnode_pairs = pairs
        ExpressionBase.__init__(self, source_ref)

    def getVisitableNodes(self):
        if False:
            return 10
        'The visitable nodes, with tuple values flattened.'
        return self.subnode_pairs

    def getVisitableNodesNamed(self):
        if False:
            return 10
        'Named children dictionary.\n\n        For use in cloning nodes, debugging and XML output.\n        '
        return (('pairs', self.subnode_pairs),)

    def replaceChild(self, old_node, new_node):
        if False:
            print('Hello World!')
        value = self.subnode_pairs
        if old_node in value:
            if new_node is not None:
                new_node.parent = self
                self.subnode_pairs = tuple((val if val is not old_node else new_node for val in value))
            else:
                self.subnode_pairs = tuple((val for val in value if val is not old_node))
            return
        raise AssertionError("Didn't find child", old_node, 'in', self)

    def getCloneArgs(self):
        if False:
            while True:
                i = 10
        'Get clones of all children to pass for a new node.\n\n        Needs to make clones of child nodes too.\n        '
        values = {'pairs': tuple((v.makeClone() for v in self.subnode_pairs))}
        values.update(self.getDetails())
        return values

    def finalize(self):
        if False:
            while True:
                i = 10
        del self.parent
        for c in self.subnode_pairs:
            c.finalize()
        del self.subnode_pairs

    def computeExpressionRaw(self, trace_collection):
        if False:
            while True:
                i = 10
        'Compute an expression.\n\n        Default behavior is to just visit the child expressions first, and\n        then the node "computeExpression". For a few cases this needs to\n        be overloaded, e.g. conditional expressions.\n        '
        old_subnode_pairs = self.subnode_pairs
        for sub_expression in old_subnode_pairs:
            expression = trace_collection.onExpression(sub_expression)
            if expression.willRaiseAnyException():
                wrapped_expression = wrapExpressionWithSideEffects(side_effects=self.subnode_pairs[:old_subnode_pairs.index(sub_expression)], old_node=sub_expression, new_node=expression)
                return (wrapped_expression, 'new_raise', lambda : "For '%s' the child expression '%s' will raise." % (self.getChildNameNice(), expression.getChildNameNice()))
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            i = 10
            return i + 15
        return False

    def mayRaiseException(self, exception_type):
        if False:
            for i in range(10):
                print('nop')
        return any((value.mayRaiseException(exception_type) for value in self.subnode_pairs))

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            print('Hello World!')
        'Collect variable reads and writes of child nodes.'
        for element in self.subnode_pairs:
            element.collectVariableAccesses(emit_read, emit_write)
ExpressionImportlibMetadataBackportSelectableGroupsValueRefBase = ChildHavingPairsTupleFinalNoRaiseMixin
ExpressionImportlibMetadataSelectableGroupsValueRefBase = ChildHavingPairsTupleFinalNoRaiseMixin

class ChildHavingPromptOptionalFinalMixin(ExpressionBase):
    __slots__ = ()

    def __init__(self, prompt, source_ref):
        if False:
            i = 10
            return i + 15
        if prompt is not None:
            prompt.parent = self
        self.subnode_prompt = prompt
        ExpressionBase.__init__(self, source_ref)

    def getVisitableNodes(self):
        if False:
            while True:
                i = 10
        'The visitable nodes, with tuple values flattened.'
        value = self.subnode_prompt
        if value is None:
            return ()
        else:
            return (value,)

    def getVisitableNodesNamed(self):
        if False:
            for i in range(10):
                print('nop')
        'Named children dictionary.\n\n        For use in cloning nodes, debugging and XML output.\n        '
        return (('prompt', self.subnode_prompt),)

    def replaceChild(self, old_node, new_node):
        if False:
            print('Hello World!')
        value = self.subnode_prompt
        if old_node is value:
            if new_node is not None:
                new_node.parent = self
            self.subnode_prompt = new_node
            return
        raise AssertionError("Didn't find child", old_node, 'in', self)

    def getCloneArgs(self):
        if False:
            while True:
                i = 10
        'Get clones of all children to pass for a new node.\n\n        Needs to make clones of child nodes too.\n        '
        values = {'prompt': self.subnode_prompt.makeClone() if self.subnode_prompt is not None else None}
        values.update(self.getDetails())
        return values

    def finalize(self):
        if False:
            while True:
                i = 10
        del self.parent
        if self.subnode_prompt is not None:
            self.subnode_prompt.finalize()
        del self.subnode_prompt

    def computeExpressionRaw(self, trace_collection):
        if False:
            return 10
        'Compute an expression.\n\n        Default behavior is to just visit the child expressions first, and\n        then the node "computeExpression". For a few cases this needs to\n        be overloaded, e.g. conditional expressions.\n        '
        expression = self.subnode_prompt
        if expression is not None:
            expression = trace_collection.onExpression(expression)
            if expression.willRaiseAnyException():
                return (expression, 'new_raise', lambda : "For '%s' the child expression '%s' will raise." % (self.getChildNameNice(), expression.getChildNameNice()))
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            i = 10
            return i + 15
        'Collect variable reads and writes of child nodes.'
        subnode_prompt = self.subnode_prompt
        if subnode_prompt is not None:
            self.subnode_prompt.collectVariableAccesses(emit_read, emit_write)
ExpressionBuiltinInputBase = ChildHavingPromptOptionalFinalMixin

class ChildHavingValueFinalNoRaiseMixin(ExpressionBase):
    __slots__ = ()

    def __init__(self, value, source_ref):
        if False:
            for i in range(10):
                print('nop')
        value.parent = self
        self.subnode_value = value
        ExpressionBase.__init__(self, source_ref)

    def getVisitableNodes(self):
        if False:
            print('Hello World!')
        'The visitable nodes, with tuple values flattened.'
        return (self.subnode_value,)

    def getVisitableNodesNamed(self):
        if False:
            for i in range(10):
                print('nop')
        'Named children dictionary.\n\n        For use in cloning nodes, debugging and XML output.\n        '
        return (('value', self.subnode_value),)

    def replaceChild(self, old_node, new_node):
        if False:
            return 10
        value = self.subnode_value
        if old_node is value:
            new_node.parent = self
            self.subnode_value = new_node
            return
        raise AssertionError("Didn't find child", old_node, 'in', self)

    def getCloneArgs(self):
        if False:
            for i in range(10):
                print('nop')
        'Get clones of all children to pass for a new node.\n\n        Needs to make clones of child nodes too.\n        '
        values = {'value': self.subnode_value.makeClone()}
        values.update(self.getDetails())
        return values

    def finalize(self):
        if False:
            return 10
        del self.parent
        self.subnode_value.finalize()
        del self.subnode_value

    def computeExpressionRaw(self, trace_collection):
        if False:
            while True:
                i = 10
        'Compute an expression.\n\n        Default behavior is to just visit the child expressions first, and\n        then the node "computeExpression". For a few cases this needs to\n        be overloaded, e.g. conditional expressions.\n        '
        expression = trace_collection.onExpression(self.subnode_value)
        if expression.willRaiseAnyException():
            return (expression, 'new_raise', lambda : "For '%s' the child expression '%s' will raise." % (self.getChildNameNice(), expression.getChildNameNice()))
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            while True:
                i = 10
        return False

    def mayRaiseException(self, exception_type):
        if False:
            for i in range(10):
                print('nop')
        return self.subnode_value.mayRaiseException(exception_type)

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            for i in range(10):
                print('nop')
        'Collect variable reads and writes of child nodes.'
        self.subnode_value.collectVariableAccesses(emit_read, emit_write)
ExpressionBuiltinClassmethodBase = ChildHavingValueFinalNoRaiseMixin
ExpressionBuiltinStaticmethodBase = ChildHavingValueFinalNoRaiseMixin