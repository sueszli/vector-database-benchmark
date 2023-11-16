""" Value trace objects.

Value traces indicate the flow of values and merges their versions for
the SSA (Single State Assignment) form being used in Nuitka.

Values can be seen as:

* Unknown (maybe initialized, maybe not, we cannot know)
* Uninitialized (definitely not initialized, first version)
* Init (definitely initialized, e.g. parameter variables)
* Assign (assignment was done)
* Deleted (del was done, now unassigned, uninitialized)
* Merge (result of diverged code paths, loop potentially)
* LoopIncomplete (aggregation during loops, not yet fully known)
* LoopComplete (complete knowledge of loop types)
"""
from nuitka.nodes.shapes.BuiltinTypeShapes import tshape_dict, tshape_tuple
from nuitka.nodes.shapes.ControlFlowDescriptions import ControlFlowDescriptionElementBasedEscape, ControlFlowDescriptionFullEscape, ControlFlowDescriptionNoEscape
from nuitka.nodes.shapes.StandardShapes import ShapeLoopCompleteAlternative, ShapeLoopInitialAlternative, tshape_uninitialized, tshape_unknown
from nuitka.Tracing import my_print
from nuitka.utils.InstanceCounters import counted_del, counted_init, isCountingInstances

class ValueTraceBase(object):
    __slots__ = ('owner', 'usage_count', 'name_usage_count', 'merge_usage_count', 'closure_usages', 'previous')

    @counted_init
    def __init__(self, owner, previous):
        if False:
            print('Hello World!')
        self.owner = owner
        self.usage_count = 0
        self.name_usage_count = 0
        self.merge_usage_count = 0
        self.closure_usages = False
        self.previous = previous
    if isCountingInstances():
        __del__ = counted_del()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%s of %s>' % (self.__class__.__name__, self.owner.getCodeName())

    def dump(self, indent):
        if False:
            print('Hello World!')
        my_print('%s%s %s:' % (indent, self.__class__.__name__, id(self)))

    def getOwner(self):
        if False:
            return 10
        return self.owner

    @staticmethod
    def isLoopTrace():
        if False:
            for i in range(10):
                print('nop')
        return False

    def addUsage(self):
        if False:
            return 10
        self.usage_count += 1

    def addNameUsage(self):
        if False:
            return 10
        self.usage_count += 1
        self.name_usage_count += 1
        if self.name_usage_count <= 2 and self.previous is not None:
            self.previous.addNameUsage()

    def addMergeUsage(self):
        if False:
            for i in range(10):
                print('nop')
        self.usage_count += 1
        self.merge_usage_count += 1

    def getUsageCount(self):
        if False:
            return 10
        return self.usage_count

    def getNameUsageCount(self):
        if False:
            print('Hello World!')
        return self.name_usage_count

    def getMergeUsageCount(self):
        if False:
            for i in range(10):
                print('nop')
        return self.merge_usage_count

    def getMergeOrNameUsageCount(self):
        if False:
            return 10
        return self.merge_usage_count + self.name_usage_count

    def getPrevious(self):
        if False:
            while True:
                i = 10
        return self.previous

    @staticmethod
    def isAssignTrace():
        if False:
            return 10
        return False

    @staticmethod
    def isUnassignedTrace():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def isDeletedTrace():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def isUninitializedTrace():
        if False:
            return 10
        return False

    @staticmethod
    def isInitTrace():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def isUnknownTrace():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def isUnknownOrVeryTrustedTrace():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def isEscapeTrace():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def isTraceThatNeedsEscape():
        if False:
            return 10
        return True

    @staticmethod
    def isMergeTrace():
        if False:
            return 10
        return False

    def mustHaveValue(self):
        if False:
            while True:
                i = 10
        'Will this definitely have a value.\n\n        Every trace has this overloaded.\n        '
        assert False, self

    def mustNotHaveValue(self):
        if False:
            print('Hello World!')
        'Will this definitely have a value.\n\n        Every trace has this overloaded.\n        '
        assert False, self

    def getReplacementNode(self, usage):
        if False:
            i = 10
            return i + 15
        return None

    @staticmethod
    def hasShapeListExact():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def hasShapeDictionaryExact():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def hasShapeStrExact():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def hasShapeUnicodeExact():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def hasShapeTupleExact():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def hasShapeBoolExact():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def getTruthValue():
        if False:
            return 10
        return None

    @staticmethod
    def getComparisonValue():
        if False:
            for i in range(10):
                print('nop')
        return (False, None)

    @staticmethod
    def getAttributeNode():
        if False:
            return 10
        'Node to use for attribute lookups.'
        return None

    @staticmethod
    def getAttributeNodeTrusted():
        if False:
            return 10
        'Node to use for attribute lookups, with increased trust.\n\n        Used with hard imports mainly.\n        '
        return None

    @staticmethod
    def getAttributeNodeVeryTrusted():
        if False:
            print('Hello World!')
        'Node to use for attribute lookups, with highest trust.\n\n        Used for hard imports mainly.\n        '
        return None

    @staticmethod
    def getIterationSourceNode():
        if False:
            for i in range(10):
                print('nop')
        'Node to use for iteration decisions.'
        return None

    @staticmethod
    def getDictInValue(key):
        if False:
            i = 10
            return i + 15
        'Value to use for dict in decisions.'
        return None

    @staticmethod
    def inhibitsClassScopeForwardPropagation():
        if False:
            for i in range(10):
                print('nop')
        return True

class ValueTraceUnassignedBase(ValueTraceBase):
    __slots__ = ()

    @staticmethod
    def isUnassignedTrace():
        if False:
            return 10
        return True

    @staticmethod
    def getTypeShape():
        if False:
            i = 10
            return i + 15
        return tshape_uninitialized

    @staticmethod
    def getReleaseEscape():
        if False:
            while True:
                i = 10
        return ControlFlowDescriptionNoEscape

    def compareValueTrace(self, other):
        if False:
            return 10
        return other.isUnassignedTrace()

    @staticmethod
    def mustHaveValue():
        if False:
            return 10
        return False

    @staticmethod
    def mustNotHaveValue():
        if False:
            i = 10
            return i + 15
        return True

class ValueTraceUninitialized(ValueTraceUnassignedBase):
    __slots__ = ()

    def __init__(self, owner, previous):
        if False:
            i = 10
            return i + 15
        ValueTraceUnassignedBase.__init__(self, owner=owner, previous=previous)

    @staticmethod
    def isUninitializedTrace():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def isTraceThatNeedsEscape():
        if False:
            return 10
        return False

    def inhibitsClassScopeForwardPropagation(self):
        if False:
            while True:
                i = 10
        return False

class ValueTraceDeleted(ValueTraceUnassignedBase):
    """Trace caused by a deletion."""
    __slots__ = ('del_node',)

    def __init__(self, owner, previous, del_node):
        if False:
            i = 10
            return i + 15
        ValueTraceUnassignedBase.__init__(self, owner=owner, previous=previous)
        self.del_node = del_node

    @staticmethod
    def isDeletedTrace():
        if False:
            i = 10
            return i + 15
        return True

    def getDelNode(self):
        if False:
            i = 10
            return i + 15
        return self.del_node

class ValueTraceInit(ValueTraceBase):
    __slots__ = ()

    def __init__(self, owner):
        if False:
            return 10
        ValueTraceBase.__init__(self, owner=owner, previous=None)

    @staticmethod
    def getTypeShape():
        if False:
            while True:
                i = 10
        return tshape_unknown

    @staticmethod
    def getReleaseEscape():
        if False:
            return 10
        return ControlFlowDescriptionFullEscape

    def compareValueTrace(self, other):
        if False:
            i = 10
            return i + 15
        return other.isInitTrace()

    @staticmethod
    def isInitTrace():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def mustHaveValue():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def mustNotHaveValue():
        if False:
            while True:
                i = 10
        return False

class ValueTraceInitStarArgs(ValueTraceInit):

    @staticmethod
    def getTypeShape():
        if False:
            i = 10
            return i + 15
        return tshape_tuple

    @staticmethod
    def getReleaseEscape():
        if False:
            i = 10
            return i + 15
        return ControlFlowDescriptionElementBasedEscape

    @staticmethod
    def hasShapeTupleExact():
        if False:
            for i in range(10):
                print('nop')
        return True

class ValueTraceInitStarDict(ValueTraceInit):

    @staticmethod
    def getTypeShape():
        if False:
            print('Hello World!')
        return tshape_dict

    @staticmethod
    def getReleaseEscape():
        if False:
            i = 10
            return i + 15
        return ControlFlowDescriptionElementBasedEscape

    @staticmethod
    def hasShapeDictionaryExact():
        if False:
            for i in range(10):
                print('nop')
        return True

class ValueTraceUnknown(ValueTraceBase):
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            i = 10
            return i + 15
        return tshape_unknown

    @staticmethod
    def getReleaseEscape():
        if False:
            return 10
        return ControlFlowDescriptionFullEscape

    def addUsage(self):
        if False:
            return 10
        self.usage_count += 1
        if self.previous:
            self.previous.addUsage()

    def addMergeUsage(self):
        if False:
            return 10
        self.usage_count += 1
        self.merge_usage_count += 1
        if self.previous:
            self.previous.addMergeUsage()

    def compareValueTrace(self, other):
        if False:
            while True:
                i = 10
        return other.isUnknownTrace()

    @staticmethod
    def isUnknownTrace():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def isUnknownOrVeryTrustedTrace():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def isTraceThatNeedsEscape():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def mustHaveValue():
        if False:
            return 10
        return False

    @staticmethod
    def mustNotHaveValue():
        if False:
            while True:
                i = 10
        return False

    def getAttributeNode(self):
        if False:
            return 10
        if self.previous is not None:
            return self.previous.getAttributeNodeVeryTrusted()

    def getAttributeNodeTrusted(self):
        if False:
            return 10
        if self.previous is not None:
            return self.previous.getAttributeNodeVeryTrusted()

    def getAttributeNodeVeryTrusted(self):
        if False:
            return 10
        if self.previous is not None:
            return self.previous.getAttributeNodeVeryTrusted()

class ValueTraceEscaped(ValueTraceUnknown):
    __slots__ = ()

    def addUsage(self):
        if False:
            while True:
                i = 10
        self.usage_count += 1
        if self.usage_count <= 2:
            self.previous.addNameUsage()

    def addMergeUsage(self):
        if False:
            print('Hello World!')
        self.usage_count += 1
        if self.usage_count <= 2:
            self.previous.addNameUsage()
        self.merge_usage_count += 1
        if self.merge_usage_count <= 2:
            self.previous.addMergeUsage()

    def mustHaveValue(self):
        if False:
            print('Hello World!')
        return self.previous.mustHaveValue()

    def mustNotHaveValue(self):
        if False:
            print('Hello World!')
        return self.previous.mustNotHaveValue()

    def getReplacementNode(self, usage):
        if False:
            for i in range(10):
                print('nop')
        return self.previous.getReplacementNode(usage)

    @staticmethod
    def isUnknownTrace():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def isUnknownOrVeryTrustedTrace():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def isEscapeTrace():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def isTraceThatNeedsEscape():
        if False:
            for i in range(10):
                print('nop')
        return False

    def getAttributeNode(self):
        if False:
            while True:
                i = 10
        return self.previous.getAttributeNodeTrusted()

    def getAttributeNodeTrusted(self):
        if False:
            i = 10
            return i + 15
        return self.previous.getAttributeNodeTrusted()

    def getAttributeNodeVeryTrusted(self):
        if False:
            return 10
        return self.previous.getAttributeNodeVeryTrusted()

    def hasShapeListExact(self):
        if False:
            return 10
        trusted_node = self.previous.getAttributeNodeTrusted()
        return trusted_node is not None and trusted_node.hasShapeListExact()

    def hasShapeDictionaryExact(self):
        if False:
            while True:
                i = 10
        trusted_node = self.previous.getAttributeNodeTrusted()
        return trusted_node is not None and trusted_node.hasShapeDictionaryExact()

class ValueTraceAssign(ValueTraceBase):
    __slots__ = ('assign_node',)

    def __init__(self, owner, assign_node, previous):
        if False:
            return 10
        ValueTraceBase.__init__(self, owner=owner, previous=previous)
        self.assign_node = assign_node

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<%s at %s of %s>' % (self.__class__.__name__, self.assign_node.getSourceReference().getAsString(), self.assign_node.subnode_source)

    @staticmethod
    def isAssignTrace():
        if False:
            print('Hello World!')
        return True

    def compareValueTrace(self, other):
        if False:
            return 10
        return other.isAssignTrace() and self.assign_node is other.assign_node

    @staticmethod
    def mustHaveValue():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def mustNotHaveValue():
        if False:
            while True:
                i = 10
        return False

    def getTypeShape(self):
        if False:
            return 10
        return self.assign_node.getTypeShape()

    def getReleaseEscape(self):
        if False:
            while True:
                i = 10
        return self.assign_node.getReleaseEscape()

    def getAssignNode(self):
        if False:
            i = 10
            return i + 15
        return self.assign_node

    def hasShapeListExact(self):
        if False:
            for i in range(10):
                print('nop')
        return self.assign_node.subnode_source.hasShapeListExact()

    def hasShapeDictionaryExact(self):
        if False:
            i = 10
            return i + 15
        return self.assign_node.subnode_source.hasShapeDictionaryExact()

    def hasShapeStrExact(self):
        if False:
            for i in range(10):
                print('nop')
        return self.assign_node.subnode_source.hasShapeStrExact()

    def hasShapeUnicodeExact(self):
        if False:
            while True:
                i = 10
        return self.assign_node.subnode_source.hasShapeUnicodeExact()

    def hasShapeBoolExact(self):
        if False:
            for i in range(10):
                print('nop')
        return self.assign_node.subnode_source.hasShapeBoolExact()

    def getTruthValue(self):
        if False:
            print('Hello World!')
        return self.assign_node.subnode_source.getTruthValue()

    def getComparisonValue(self):
        if False:
            i = 10
            return i + 15
        return self.assign_node.subnode_source.getComparisonValue()

    def getAttributeNode(self):
        if False:
            while True:
                i = 10
        return self.assign_node.subnode_source

    def getAttributeNodeTrusted(self):
        if False:
            print('Hello World!')
        source_node = self.assign_node.subnode_source
        if source_node.hasShapeTrustedAttributes():
            return source_node
        else:
            return None

    def getAttributeNodeVeryTrusted(self):
        if False:
            return 10
        if self.assign_node.hasVeryTrustedValue():
            return self.assign_node.subnode_source
        else:
            return None

    def getIterationSourceNode(self):
        if False:
            for i in range(10):
                print('nop')
        return self.assign_node.subnode_source

    def getDictInValue(self, key):
        if False:
            for i in range(10):
                print('nop')
        'Value to use for dict in decisions.'
        return self.assign_node.subnode_source.getExpressionDictInConstant(key)

    def inhibitsClassScopeForwardPropagation(self):
        if False:
            for i in range(10):
                print('nop')
        return self.assign_node.subnode_source.mayHaveSideEffects()

class ValueTraceAssignUnescapable(ValueTraceAssign):

    @staticmethod
    def isTraceThatNeedsEscape():
        if False:
            i = 10
            return i + 15
        return False

class ValueTraceAssignVeryTrusted(ValueTraceAssignUnescapable):

    @staticmethod
    def isUnknownOrVeryTrustedTrace():
        if False:
            while True:
                i = 10
        return True

class ValueTraceAssignUnescapablePropagated(ValueTraceAssignUnescapable):
    """Assignment from value where it is not that escaping doesn't matter."""
    __slots__ = ('replacement',)

    def __init__(self, owner, assign_node, previous, replacement):
        if False:
            while True:
                i = 10
        ValueTraceAssignUnescapable.__init__(self, owner=owner, assign_node=assign_node, previous=previous)
        self.replacement = replacement

    def getReplacementNode(self, usage):
        if False:
            return 10
        return self.replacement(usage)

class ValueTraceMergeBase(ValueTraceBase):
    """Merge of two or more traces or start of loops."""
    __slots__ = ()

    def addNameUsage(self):
        if False:
            i = 10
            return i + 15
        self.usage_count += 1
        self.name_usage_count += 1
        if self.name_usage_count <= 2 and self.previous is not None:
            for previous in self.previous:
                previous.addNameUsage()

    def addUsage(self):
        if False:
            print('Hello World!')
        self.usage_count += 1
        if self.usage_count == 1:
            for trace in self.previous:
                trace.addMergeUsage()

    def addMergeUsage(self):
        if False:
            print('Hello World!')
        self.addUsage()
        self.merge_usage_count += 1

    def dump(self, indent):
        if False:
            print('Hello World!')
        ValueTraceBase.dump(self, indent)
        for trace in self.previous:
            trace.dump(indent + '  ')

class ValueTraceMerge(ValueTraceMergeBase):
    """Merge of two or more traces.

    Happens at the end of conditional blocks. This is "phi" in
    SSA theory. Also used for merging multiple "return", "break" or
    "continue" exits.
    """
    __slots__ = ()

    def __init__(self, traces):
        if False:
            for i in range(10):
                print('nop')
        shorted = []
        for trace in traces:
            if type(trace) is ValueTraceMerge:
                for trace2 in trace.previous:
                    if trace2 not in shorted:
                        shorted.append(trace2)
            elif trace not in shorted:
                shorted.append(trace)
        traces = tuple(shorted)
        assert len(traces) > 1
        ValueTraceMergeBase.__init__(self, owner=traces[0].owner, previous=traces)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<ValueTraceMerge of {previous}>'.format(previous=self.previous)

    def getTypeShape(self):
        if False:
            for i in range(10):
                print('nop')
        type_shape_found = None
        for trace in self.previous:
            type_shape = trace.getTypeShape()
            if type_shape is tshape_unknown:
                return tshape_unknown
            if type_shape_found is None:
                type_shape_found = type_shape
            elif type_shape is not type_shape_found:
                return tshape_unknown
        return type_shape_found

    def getReleaseEscape(self):
        if False:
            while True:
                i = 10
        release_escape_found = None
        for trace in self.previous:
            release_escape = trace.getReleaseEscape()
            if release_escape is ControlFlowDescriptionFullEscape:
                return ControlFlowDescriptionFullEscape
            if release_escape_found is None:
                release_escape_found = release_escape
            elif release_escape is not release_escape_found:
                return ControlFlowDescriptionFullEscape
        return release_escape_found

    @staticmethod
    def isMergeTrace():
        if False:
            print('Hello World!')
        return True

    def compareValueTrace(self, other):
        if False:
            print('Hello World!')
        if not other.isMergeTrace():
            return False
        if len(self.previous) != len(other.previous):
            return False
        for (a, b) in zip(self.previous, other.previous):
            if not a.compareValueTrace(b):
                return False
        return True

    def mustHaveValue(self):
        if False:
            return 10
        for previous in self.previous:
            if not previous.isInitTrace() and (not previous.isAssignTrace()):
                return False
        return True

    def mustNotHaveValue(self):
        if False:
            for i in range(10):
                print('nop')
        for previous in self.previous:
            if not previous.mustNotHaveValue():
                return False
        return True

    def hasShapeListExact(self):
        if False:
            i = 10
            return i + 15
        return all((previous.hasShapeListExact() for previous in self.previous))

    def hasShapeDictionaryExact(self):
        if False:
            for i in range(10):
                print('nop')
        return all((previous.hasShapeDictionaryExact() for previous in self.previous))

    def getTruthValue(self):
        if False:
            while True:
                i = 10
        any_false = False
        any_true = False
        for previous in self.previous:
            truth_value = previous.getTruthValue()
            if truth_value is None:
                return None
            elif truth_value is True:
                if any_false:
                    return None
                any_true = True
            else:
                if any_true:
                    return None
                any_false = True
        return any_true

    def getComparisonValue(self):
        if False:
            print('Hello World!')
        return (False, None)

class ValueTraceLoopBase(ValueTraceMergeBase):
    __slots__ = ('loop_node', 'type_shapes', 'type_shape', 'recursion')

    def __init__(self, loop_node, previous, type_shapes):
        if False:
            i = 10
            return i + 15
        ValueTraceMergeBase.__init__(self, owner=previous.owner, previous=(previous,))
        self.loop_node = loop_node
        self.type_shapes = type_shapes
        self.type_shape = None
        self.recursion = False

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<%s shapes %s of %s>' % (self.__class__.__name__, self.type_shapes, self.owner.getCodeName())

    @staticmethod
    def isLoopTrace():
        if False:
            print('Hello World!')
        return True

    def getTypeShape(self):
        if False:
            print('Hello World!')
        if self.type_shape is None:
            if len(self.type_shapes) > 1:
                self.type_shape = ShapeLoopCompleteAlternative(self.type_shapes)
            else:
                self.type_shape = next(iter(self.type_shapes))
        return self.type_shape

    def addLoopContinueTraces(self, continue_traces):
        if False:
            i = 10
            return i + 15
        self.previous += tuple(continue_traces)
        for previous in continue_traces:
            previous.addMergeUsage()

    def mustHaveValue(self):
        if False:
            for i in range(10):
                print('nop')
        if self.recursion:
            return True
        self.recursion = True
        for previous in self.previous:
            if not previous.mustHaveValue():
                self.recursion = False
                return False
        self.recursion = False
        return True

class ValueTraceLoopComplete(ValueTraceLoopBase):
    __slots__ = ()

    @staticmethod
    def getReleaseEscape():
        if False:
            i = 10
            return i + 15
        return ControlFlowDescriptionFullEscape

    def compareValueTrace(self, other):
        if False:
            return 10
        return self.__class__ is other.__class__ and self.loop_node == other.loop_node and (self.type_shapes == other.type_shapes)

    @staticmethod
    def mustHaveValue():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def mustNotHaveValue():
        if False:
            return 10
        return False

    @staticmethod
    def getTruthValue():
        if False:
            for i in range(10):
                print('nop')
        return None

    @staticmethod
    def getComparisonValue():
        if False:
            return 10
        return (False, None)

class ValueTraceLoopIncomplete(ValueTraceLoopBase):
    __slots__ = ()

    def getTypeShape(self):
        if False:
            i = 10
            return i + 15
        if self.type_shape is None:
            self.type_shape = ShapeLoopInitialAlternative(self.type_shapes)
        return self.type_shape

    @staticmethod
    def getReleaseEscape():
        if False:
            for i in range(10):
                print('nop')
        return ControlFlowDescriptionFullEscape

    def compareValueTrace(self, other):
        if False:
            return 10
        return self.__class__ is other.__class__ and self.loop_node == other.loop_node

    @staticmethod
    def mustHaveValue():
        if False:
            return 10
        return False

    @staticmethod
    def mustNotHaveValue():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def getTruthValue():
        if False:
            print('Hello World!')
        return None

    @staticmethod
    def getComparisonValue():
        if False:
            for i in range(10):
                print('nop')
        return (False, None)