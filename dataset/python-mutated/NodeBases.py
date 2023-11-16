""" Node base classes.

These classes provide the generic base classes available for nodes,
statements or expressions alike. There is a dedicated module for
expression only stuff.

"""
import ast
from abc import abstractmethod
from nuitka import Options, Tracing, TreeXML
from nuitka.__past__ import iterItems
from nuitka.Errors import NuitkaNodeError
from nuitka.PythonVersions import python_version
from nuitka.SourceCodeReferences import SourceCodeReference
from nuitka.utils.InstanceCounters import counted_del, counted_init, isCountingInstances
from nuitka.Variables import TempVariable
from .FutureSpecs import fromFlags
from .NodeMakingHelpers import makeStatementOnlyNodesFromExpressions
from .NodeMetaClasses import NodeCheckMetaClass, NodeMetaClassBase

class NodeBase(NodeMetaClassBase):
    __slots__ = ('parent', 'source_ref')
    assert Options.is_full_compat is not None
    if Options.is_full_compat:
        __slots__ += ('effective_source_ref',)
    kind = None

    @counted_init
    def __init__(self, source_ref):
        if False:
            i = 10
            return i + 15
        assert source_ref is not None
        assert source_ref.line is not None
        self.parent = None
        self.source_ref = source_ref
    if isCountingInstances():
        __del__ = counted_del()

    @abstractmethod
    def finalize(self):
        if False:
            return 10
        pass

    def __repr__(self):
        if False:
            return 10
        return '<Node %s>' % self.getDescription()

    def getDescription(self):
        if False:
            i = 10
            return i + 15
        'Description of the node, intended for use in __repr__ and\n        graphical display.\n\n        '
        details = self.getDetailsForDisplay()
        if details:
            return "'%s' with %s" % (self.kind, details)
        else:
            return "'%s'" % self.kind

    def getDetails(self):
        if False:
            for i in range(10):
                print('nop')
        'Details of the node, intended for re-creation.\n\n        We are not using the pickle mechanisms, but this is basically\n        part of what the constructor call needs. Real children will\n        also be added.\n\n        '
        return {}

    def getDetailsForDisplay(self):
        if False:
            return 10
        'Details of the node, intended for use in __repr__ and dumps.\n\n        This is also used for XML.\n        '
        return self.getDetails()

    def getCloneArgs(self):
        if False:
            while True:
                i = 10
        return self.getDetails()

    def makeClone(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            result = self.__class__(source_ref=self.source_ref, **self.getCloneArgs())
        except TypeError as e:
            raise NuitkaNodeError('Problem cloning node', self, e)
        effective_source_ref = self.getCompatibleSourceReference()
        if effective_source_ref is not self.source_ref:
            result.setCompatibleSourceReference(effective_source_ref)
        return result

    def makeCloneShallow(self):
        if False:
            return 10
        args = self.getDetails()
        args.update(self.getVisitableNodesNamed())
        try:
            result = self.__class__(source_ref=self.source_ref, **args)
        except TypeError as e:
            raise NuitkaNodeError('Problem cloning node', self, e)
        effective_source_ref = self.getCompatibleSourceReference()
        if effective_source_ref is not self.source_ref:
            result.setCompatibleSourceReference(effective_source_ref)
        return result

    def getParent(self):
        if False:
            while True:
                i = 10
        'Parent of the node. Every node except modules has to have a parent.'
        if self.parent is None and (not self.isCompiledPythonModule()):
            assert False, (self, self.source_ref)
        return self.parent

    def getChildName(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the role in the current parent, subject to changes.'
        parent = self.getParent()
        for (key, value) in parent.getVisitableNodesNamed():
            if self is value:
                return key
            if type(value) is tuple:
                if self in value:
                    return (key, value.index(self))
        return None

    def getChildNameNice(self):
        if False:
            i = 10
            return i + 15
        child_name = self.getChildName()
        if hasattr(self.parent, 'nice_children_dict'):
            return self.parent.nice_children_dict[child_name]
        else:
            return child_name

    def getParentFunction(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the parent that is a function.'
        parent = self.getParent()
        while parent is not None and (not parent.isExpressionFunctionBodyBase()):
            parent = parent.getParent()
        return parent

    def getParentModule(self):
        if False:
            print('Hello World!')
        'Return the parent that is module.'
        parent = self
        while not parent.isCompiledPythonModule():
            if hasattr(parent, 'provider'):
                parent = parent.provider
            else:
                parent = parent.getParent()
        return parent

    def isParentVariableProvider(self):
        if False:
            print('Hello World!')
        return isinstance(self, ClosureGiverNodeMixin)

    def getParentVariableProvider(self):
        if False:
            for i in range(10):
                print('nop')
        parent = self.getParent()
        while not parent.isParentVariableProvider():
            parent = parent.getParent()
        return parent

    def getParentReturnConsumer(self):
        if False:
            for i in range(10):
                print('nop')
        parent = self.getParent()
        while not parent.isParentVariableProvider() and (not parent.isExpressionOutlineBody()):
            parent = parent.getParent()
        return parent

    def getParentStatementsFrame(self):
        if False:
            for i in range(10):
                print('nop')
        current = self.getParent()
        while True:
            if current.isStatementsFrame():
                return current
            if current.isParentVariableProvider():
                return None
            if current.isExpressionOutlineBody():
                return None
            current = current.getParent()

    def getSourceReference(self):
        if False:
            return 10
        return self.source_ref

    def setCompatibleSourceReference(self, source_ref):
        if False:
            i = 10
            return i + 15
        'Bug compatible line numbers information.\n\n        As CPython outputs the last bit of bytecode executed, and not the\n        line of the operation. For example calls, output the line of the\n        last argument, as opposed to the line of the operation start.\n\n        For tests, we wants to be compatible. In improved more, we are\n        not being fully compatible, and just drop it altogether.\n        '
        if self.source_ref is not source_ref and Options.is_full_compat and (self.source_ref != source_ref):
            self.effective_source_ref = source_ref

    def getCompatibleSourceReference(self):
        if False:
            while True:
                i = 10
        'Bug compatible line numbers information.\n\n        See above.\n        '
        return getattr(self, 'effective_source_ref', self.source_ref)

    def asXml(self):
        if False:
            return 10
        line = self.source_ref.getLineNumber()
        result = TreeXML.Element('node', kind=self.__class__.__name__, line=str(line))
        compat_line = self.getCompatibleSourceReference().getLineNumber()
        if compat_line != line:
            result.attrib['compat_line'] = str(compat_line)
        for (key, value) in iterItems(self.getDetailsForDisplay()):
            result.set(key, str(value))
        for (name, children) in self.getVisitableNodesNamed():
            role = TreeXML.Element('role', name=name)
            result.append(role)
            if children is None:
                role.attrib['type'] = 'none'
            elif type(children) not in (list, tuple):
                role.append(children.asXml())
            else:
                role.attrib['type'] = 'list'
                for child in children:
                    role.append(child.asXml())
        return result

    @classmethod
    def fromXML(cls, provider, source_ref, **args):
        if False:
            i = 10
            return i + 15
        return cls(source_ref=source_ref, **args)

    def asXmlText(self):
        if False:
            for i in range(10):
                print('nop')
        xml = self.asXml()
        return TreeXML.toString(xml)

    def dump(self, level=0):
        if False:
            return 10
        Tracing.printIndented(level, self)
        Tracing.printSeparator(level)
        for visitable in self.getVisitableNodes():
            visitable.dump(level + 1)
        Tracing.printSeparator(level)

    @staticmethod
    def isStatementsSequence():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def isStatementsFrame():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def isCompiledPythonModule():
        if False:
            i = 10
            return i + 15
        return False

    def isExpressionBuiltin(self):
        if False:
            for i in range(10):
                print('nop')
        return self.kind.startswith('EXPRESSION_BUILTIN_')

    @staticmethod
    def isStatementAssignmentVariable():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def isStatementDelVariable():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def isStatementReleaseVariable():
        if False:
            return 10
        return False

    @staticmethod
    def isExpressionConstantRef():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def isExpressionConstantBoolRef():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def isExpressionOperationUnary():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def isExpressionOperationBinary():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def isExpressionOperationInplace():
        if False:
            return 10
        return False

    @staticmethod
    def isExpressionComparison():
        if False:
            return 10
        return False

    @staticmethod
    def isExpressionSideEffects():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def isExpressionMakeSequence():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def isNumberConstant():
        if False:
            return 10
        return False

    @staticmethod
    def isExpressionCall():
        if False:
            return 10
        return False

    @staticmethod
    def isExpressionFunctionBodyBase():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def isExpressionOutlineFunctionBase():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def isExpressionClassBodyBase():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def isExpressionFunctionCreation():
        if False:
            for i in range(10):
                print('nop')
        return False

    def visit(self, context, visitor):
        if False:
            while True:
                i = 10
        visitor(self)
        for visitable in self.getVisitableNodes():
            visitable.visit(context, visitor)

    @staticmethod
    def getVisitableNodes():
        if False:
            print('Hello World!')
        return ()

    @staticmethod
    def getVisitableNodesNamed():
        if False:
            return 10
        'Named children dictionary.\n\n        For use in debugging and XML output.\n        '
        return ()

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            for i in range(10):
                print('nop')
        'Collect variable reads and writes of child nodes.'

    @staticmethod
    def getName():
        if False:
            i = 10
            return i + 15
        'Name of the node if any.'
        return None

    @staticmethod
    def mayHaveSideEffects():
        if False:
            while True:
                i = 10
        'Unless we are told otherwise, everything may have a side effect.'
        return True

    def isOrderRelevant(self):
        if False:
            return 10
        return self.mayHaveSideEffects()

    def extractSideEffects(self):
        if False:
            i = 10
            return i + 15
        'Unless defined otherwise, the expression is the side effect.'
        return (self,)

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            return 10
        'Unless we are told otherwise, everything may raise everything.'
        return True

    @staticmethod
    def mayReturn():
        if False:
            return 10
        'May this node do a return exit, to be overloaded for things that might.'
        return False

    @staticmethod
    def mayBreak():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def mayContinue():
        if False:
            i = 10
            return i + 15
        return False

    def needsFrame(self):
        if False:
            i = 10
            return i + 15
        'Unless we are tolder otherwise, this depends on exception raise.'
        return self.mayRaiseException(BaseException)

    @staticmethod
    def willRaiseAnyException():
        if False:
            return 10
        return False

    @staticmethod
    def isStatementAborting():
        if False:
            for i in range(10):
                print('nop')
        "Is the node aborting, control flow doesn't continue after this node."
        return False

class CodeNodeMixin(object):
    __slots__ = ()

    def __init__(self, name, code_prefix):
        if False:
            return 10
        assert name is not None
        self.name = name
        self.code_prefix = code_prefix
        self.code_name = None
        self.uids = {}

    def getName(self):
        if False:
            i = 10
            return i + 15
        return self.name

    def getCodeName(self):
        if False:
            print('Hello World!')
        if self.code_name is None:
            provider = self.getParentVariableProvider().getEntryPoint()
            parent_name = provider.getCodeName()
            uid = '_%d' % provider.getChildUID(self)
            assert isinstance(self, CodeNodeMixin)
            if self.name:
                name = uid + '_' + self.name.strip('<>')
            else:
                name = uid
            if str is not bytes:
                name = name.encode('ascii', 'c_identifier').decode()
            self.code_name = '%s$$$%s_%s' % (parent_name, self.code_prefix, name)
        return self.code_name

    def getChildUID(self, node):
        if False:
            print('Hello World!')
        if node.kind not in self.uids:
            self.uids[node.kind] = 0
        self.uids[node.kind] += 1
        return self.uids[node.kind]

class ClosureGiverNodeMixin(CodeNodeMixin):
    """Base class for nodes that provide variables for closure takers."""
    __slots__ = ()

    def __init__(self, name, code_prefix):
        if False:
            print('Hello World!')
        CodeNodeMixin.__init__(self, name=name, code_prefix=code_prefix)
        self.temp_variables = {}
        self.temp_scopes = {}
        self.preserver_id = 0

    def hasProvidedVariable(self, variable_name):
        if False:
            i = 10
            return i + 15
        return self.locals_scope.hasProvidedVariable(variable_name)

    def getProvidedVariable(self, variable_name):
        if False:
            while True:
                i = 10
        if not self.locals_scope.hasProvidedVariable(variable_name):
            variable = self.createProvidedVariable(variable_name=variable_name)
            self.locals_scope.registerProvidedVariable(variable)
        return self.locals_scope.getProvidedVariable(variable_name)

    @abstractmethod
    def createProvidedVariable(self, variable_name):
        if False:
            for i in range(10):
                print('nop')
        'Create a variable provided by this function.'

    def allocateTempScope(self, name):
        if False:
            print('Hello World!')
        self.temp_scopes[name] = self.temp_scopes.get(name, 0) + 1
        return '%s_%d' % (name, self.temp_scopes[name])

    def allocateTempVariable(self, temp_scope, name, temp_type):
        if False:
            i = 10
            return i + 15
        if temp_scope is not None:
            full_name = '%s__%s' % (temp_scope, name)
        else:
            assert name != 'result'
            full_name = name
        assert full_name not in self.temp_variables, full_name
        result = self.createTempVariable(temp_name=full_name, temp_type=temp_type)
        if self.trace_collection is not None:
            self.trace_collection.initVariableUnknown(result).addUsage()
        return result

    def createTempVariable(self, temp_name, temp_type):
        if False:
            print('Hello World!')
        if temp_name in self.temp_variables:
            return self.temp_variables[temp_name]
        result = TempVariable(owner=self, variable_name=temp_name, variable_type=temp_type)
        self.temp_variables[temp_name] = result
        return result

    def getTempVariable(self, temp_scope, name):
        if False:
            for i in range(10):
                print('nop')
        if temp_scope is not None:
            full_name = '%s__%s' % (temp_scope, name)
        else:
            full_name = name
        return self.temp_variables[full_name]

    def getTempVariables(self):
        if False:
            print('Hello World!')
        return self.temp_variables.values()

    def _removeTempVariable(self, variable):
        if False:
            i = 10
            return i + 15
        del self.temp_variables[variable.getName()]

    def optimizeUnusedTempVariables(self):
        if False:
            print('Hello World!')
        remove = []
        for temp_variable in self.getTempVariables():
            empty = self.trace_collection.hasEmptyTraces(variable=temp_variable)
            if empty:
                remove.append(temp_variable)
        for temp_variable in remove:
            self._removeTempVariable(temp_variable)

    def allocatePreserverId(self):
        if False:
            return 10
        if python_version >= 768:
            self.preserver_id += 1
        return self.preserver_id

class ClosureTakerMixin(object):
    """Mixin for nodes that accept variables from closure givers."""
    __slots__ = ()

    def __init__(self, provider):
        if False:
            print('Hello World!')
        self.provider = provider
        self.taken = set()

    def getParentVariableProvider(self):
        if False:
            print('Hello World!')
        return self.provider

    def getClosureVariable(self, variable_name):
        if False:
            while True:
                i = 10
        result = self.provider.getVariableForClosure(variable_name=variable_name)
        assert result is not None, variable_name
        if not result.isModuleVariable():
            self.addClosureVariable(result)
        return result

    def addClosureVariable(self, variable):
        if False:
            i = 10
            return i + 15
        self.taken.add(variable)
        return variable

    def getClosureVariables(self):
        if False:
            print('Hello World!')
        return tuple(sorted([take for take in self.taken if not take.isModuleVariable()], key=lambda x: x.getName()))

    def getClosureVariableIndex(self, variable):
        if False:
            while True:
                i = 10
        closure_variables = self.getClosureVariables()
        for (count, closure_variable) in enumerate(closure_variables):
            if variable is closure_variable:
                return count
        raise IndexError(variable)

    def hasTakenVariable(self, variable_name):
        if False:
            return 10
        for variable in self.taken:
            if variable.getName() == variable_name:
                return True
        return False

    def getTakenVariable(self, variable_name):
        if False:
            for i in range(10):
                print('nop')
        for variable in self.taken:
            if variable.getName() == variable_name:
                return variable
        return None

class StatementBase(NodeBase):
    """Base class for all statement nodes."""

    @staticmethod
    def getStatementNiceName():
        if False:
            print('Hello World!')
        return 'un-described statement'

    def computeStatementSubExpressions(self, trace_collection):
        if False:
            return 10
        'Compute a statement.\n\n        Default behavior is to just visit the child expressions first, and\n        then the node "computeStatement". For a few cases this needs to\n        be overloaded.\n        '
        expressions = self.getVisitableNodes()
        for (count, expression) in enumerate(expressions):
            expression = trace_collection.onExpression(expression)
            if expression.willRaiseAnyException():
                wrapped_expression = makeStatementOnlyNodesFromExpressions(expressions[:count + 1])
                assert wrapped_expression is not None
                return (wrapped_expression, 'new_raise', lambda : "For %s the child expression '%s' will raise." % (self.getStatementNiceName(), expression.getChildNameNice()))
        return (self, None, None)

class SideEffectsFromChildrenMixin(object):
    __slots__ = ()

    def mayHaveSideEffects(self):
        if False:
            print('Hello World!')
        for child in self.getVisitableNodes():
            if child.mayHaveSideEffects():
                return True
        return False

    def extractSideEffects(self):
        if False:
            i = 10
            return i + 15
        result = []
        for child in self.getVisitableNodes():
            result.extend(child.extractSideEffects())
        return tuple(result)

    def computeExpressionDrop(self, statement, trace_collection):
        if False:
            while True:
                i = 10
        side_effects = self.extractSideEffects()
        if side_effects:
            return (makeStatementOnlyNodesFromExpressions(side_effects), 'new_statements', 'Lowered unused expression %s to its side effects.' % self.kind)
        else:
            return (None, 'new_statements', 'Removed %s without side effects.' % self.kind)

def makeChild(provider, child, source_ref):
    if False:
        print('Hello World!')
    child_type = child.attrib.get('type')
    if child_type == 'list':
        return [fromXML(provider=provider, xml=sub_child, source_ref=source_ref) for sub_child in child]
    elif child_type == 'none':
        return None
    else:
        return fromXML(provider=provider, xml=child[0], source_ref=source_ref)

def getNodeClassFromKind(kind):
    if False:
        print('Hello World!')
    return NodeCheckMetaClass.kinds[kind]

def extractKindAndArgsFromXML(xml, source_ref):
    if False:
        return 10
    kind = xml.attrib['kind']
    args = dict(xml.attrib)
    del args['kind']
    if source_ref is None:
        source_ref = SourceCodeReference.fromFilenameAndLine(args['filename'], int(args['line']))
        del args['filename']
        del args['line']
    else:
        source_ref = source_ref.atLineNumber(int(args['line']))
        del args['line']
    node_class = getNodeClassFromKind(kind)
    return (kind, node_class, args, source_ref)

def fromXML(provider, xml, source_ref=None):
    if False:
        i = 10
        return i + 15
    assert xml.tag == 'node', xml
    (kind, node_class, args, source_ref) = extractKindAndArgsFromXML(xml, source_ref)
    if 'constant' in args:
        args['constant'] = ast.literal_eval(args['constant'])
    if kind in ('ExpressionFunctionBody', 'PythonMainModule', 'PythonCompiledModule', 'PythonCompiledPackage', 'PythonInternalModule'):
        delayed = node_class.named_children
        if 'code_flags' in args:
            args['future_spec'] = fromFlags(args['code_flags'])
    else:
        delayed = ()
    for child in xml:
        assert child.tag == 'role', child.tag
        child_name = child.attrib['name']
        if child_name not in delayed:
            args[child_name] = makeChild(provider, child, source_ref)
        else:
            args[child_name] = child
    try:
        return node_class.fromXML(provider=provider, source_ref=source_ref, **args)
    except (TypeError, AttributeError):
        Tracing.printLine(node_class, args, source_ref)
        raise