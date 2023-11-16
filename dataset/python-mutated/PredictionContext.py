from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.error.Errors import IllegalStateException
from io import StringIO
_trace_atn_sim = False

class PredictionContext(object):
    EMPTY = None
    EMPTY_RETURN_STATE = 2147483647
    globalNodeCount = 1
    id = globalNodeCount

    def __init__(self, cachedHashCode: int):
        if False:
            return 10
        self.cachedHashCode = cachedHashCode

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return 0

    def isEmpty(self):
        if False:
            return 10
        return self is self.EMPTY

    def hasEmptyPath(self):
        if False:
            for i in range(10):
                print('nop')
        return self.getReturnState(len(self) - 1) == self.EMPTY_RETURN_STATE

    def getReturnState(self, index: int):
        if False:
            print('Hello World!')
        raise IllegalStateException('illegal!')

    def __hash__(self):
        if False:
            print('Hello World!')
        return self.cachedHashCode

def calculateHashCode(parent: PredictionContext, returnState: int):
    if False:
        i = 10
        return i + 15
    return hash('') if parent is None else hash((hash(parent), returnState))

def calculateListsHashCode(parents: [], returnStates: []):
    if False:
        i = 10
        return i + 15
    h = 0
    for (parent, returnState) in zip(parents, returnStates):
        h = hash((h, calculateHashCode(parent, returnState)))
    return h

class PredictionContextCache(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.cache = dict()

    def add(self, ctx: PredictionContext):
        if False:
            while True:
                i = 10
        if ctx == PredictionContext.EMPTY:
            return PredictionContext.EMPTY
        existing = self.cache.get(ctx, None)
        if existing is not None:
            return existing
        self.cache[ctx] = ctx
        return ctx

    def get(self, ctx: PredictionContext):
        if False:
            print('Hello World!')
        return self.cache.get(ctx, None)

    def __len__(self):
        if False:
            return 10
        return len(self.cache)

class SingletonPredictionContext(PredictionContext):

    @staticmethod
    def create(parent: PredictionContext, returnState: int):
        if False:
            return 10
        if returnState == PredictionContext.EMPTY_RETURN_STATE and parent is None:
            return SingletonPredictionContext.EMPTY
        else:
            return SingletonPredictionContext(parent, returnState)

    def __init__(self, parent: PredictionContext, returnState: int):
        if False:
            for i in range(10):
                print('nop')
        hashCode = calculateHashCode(parent, returnState)
        super().__init__(hashCode)
        self.parentCtx = parent
        self.returnState = returnState

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return 1

    def getParent(self, index: int):
        if False:
            while True:
                i = 10
        return self.parentCtx

    def getReturnState(self, index: int):
        if False:
            while True:
                i = 10
        return self.returnState

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if self is other:
            return True
        elif other is None:
            return False
        elif not isinstance(other, SingletonPredictionContext):
            return False
        else:
            return self.returnState == other.returnState and self.parentCtx == other.parentCtx

    def __hash__(self):
        if False:
            return 10
        return self.cachedHashCode

    def __str__(self):
        if False:
            print('Hello World!')
        up = '' if self.parentCtx is None else str(self.parentCtx)
        if len(up) == 0:
            if self.returnState == self.EMPTY_RETURN_STATE:
                return '$'
            else:
                return str(self.returnState)
        else:
            return str(self.returnState) + ' ' + up

class EmptyPredictionContext(SingletonPredictionContext):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(None, PredictionContext.EMPTY_RETURN_STATE)

    def isEmpty(self):
        if False:
            print('Hello World!')
        return True

    def __eq__(self, other):
        if False:
            return 10
        return self is other

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.cachedHashCode

    def __str__(self):
        if False:
            return 10
        return '$'
PredictionContext.EMPTY = EmptyPredictionContext()

class ArrayPredictionContext(PredictionContext):

    def __init__(self, parents: list, returnStates: list):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(calculateListsHashCode(parents, returnStates))
        self.parents = parents
        self.returnStates = returnStates

    def isEmpty(self):
        if False:
            print('Hello World!')
        return self.returnStates[0] == PredictionContext.EMPTY_RETURN_STATE

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.returnStates)

    def getParent(self, index: int):
        if False:
            return 10
        return self.parents[index]

    def getReturnState(self, index: int):
        if False:
            return 10
        return self.returnStates[index]

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if self is other:
            return True
        elif not isinstance(other, ArrayPredictionContext):
            return False
        elif hash(self) != hash(other):
            return False
        else:
            return self.returnStates == other.returnStates and self.parents == other.parents

    def __str__(self):
        if False:
            while True:
                i = 10
        if self.isEmpty():
            return '[]'
        with StringIO() as buf:
            buf.write('[')
            for i in range(0, len(self.returnStates)):
                if i > 0:
                    buf.write(', ')
                if self.returnStates[i] == PredictionContext.EMPTY_RETURN_STATE:
                    buf.write('$')
                    continue
                buf.write(str(self.returnStates[i]))
                if self.parents[i] is not None:
                    buf.write(' ')
                    buf.write(str(self.parents[i]))
                else:
                    buf.write('null')
            buf.write(']')
            return buf.getvalue()

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return self.cachedHashCode

def PredictionContextFromRuleContext(atn: ATN, outerContext: RuleContext=None):
    if False:
        i = 10
        return i + 15
    if outerContext is None:
        outerContext = RuleContext.EMPTY
    if outerContext.parentCtx is None or outerContext is RuleContext.EMPTY:
        return PredictionContext.EMPTY
    parent = PredictionContextFromRuleContext(atn, outerContext.parentCtx)
    state = atn.states[outerContext.invokingState]
    transition = state.transitions[0]
    return SingletonPredictionContext.create(parent, transition.followState.stateNumber)

def merge(a: PredictionContext, b: PredictionContext, rootIsWildcard: bool, mergeCache: dict):
    if False:
        return 10
    if a == b:
        return a
    if isinstance(a, SingletonPredictionContext) and isinstance(b, SingletonPredictionContext):
        return mergeSingletons(a, b, rootIsWildcard, mergeCache)
    if rootIsWildcard:
        if isinstance(a, EmptyPredictionContext):
            return a
        if isinstance(b, EmptyPredictionContext):
            return b
    if isinstance(a, SingletonPredictionContext):
        a = ArrayPredictionContext([a.parentCtx], [a.returnState])
    if isinstance(b, SingletonPredictionContext):
        b = ArrayPredictionContext([b.parentCtx], [b.returnState])
    return mergeArrays(a, b, rootIsWildcard, mergeCache)

def mergeSingletons(a: SingletonPredictionContext, b: SingletonPredictionContext, rootIsWildcard: bool, mergeCache: dict):
    if False:
        while True:
            i = 10
    if mergeCache is not None:
        previous = mergeCache.get((a, b), None)
        if previous is not None:
            return previous
        previous = mergeCache.get((b, a), None)
        if previous is not None:
            return previous
    merged = mergeRoot(a, b, rootIsWildcard)
    if merged is not None:
        if mergeCache is not None:
            mergeCache[a, b] = merged
        return merged
    if a.returnState == b.returnState:
        parent = merge(a.parentCtx, b.parentCtx, rootIsWildcard, mergeCache)
        if parent == a.parentCtx:
            return a
        if parent == b.parentCtx:
            return b
        merged = SingletonPredictionContext.create(parent, a.returnState)
        if mergeCache is not None:
            mergeCache[a, b] = merged
        return merged
    else:
        singleParent = None
        if a is b or (a.parentCtx is not None and a.parentCtx == b.parentCtx):
            singleParent = a.parentCtx
        if singleParent is not None:
            payloads = [a.returnState, b.returnState]
            if a.returnState > b.returnState:
                payloads = [b.returnState, a.returnState]
            parents = [singleParent, singleParent]
            merged = ArrayPredictionContext(parents, payloads)
            if mergeCache is not None:
                mergeCache[a, b] = merged
            return merged
        payloads = [a.returnState, b.returnState]
        parents = [a.parentCtx, b.parentCtx]
        if a.returnState > b.returnState:
            payloads = [b.returnState, a.returnState]
            parents = [b.parentCtx, a.parentCtx]
        merged = ArrayPredictionContext(parents, payloads)
        if mergeCache is not None:
            mergeCache[a, b] = merged
        return merged

def mergeRoot(a: SingletonPredictionContext, b: SingletonPredictionContext, rootIsWildcard: bool):
    if False:
        i = 10
        return i + 15
    if rootIsWildcard:
        if a == PredictionContext.EMPTY:
            return PredictionContext.EMPTY
        if b == PredictionContext.EMPTY:
            return PredictionContext.EMPTY
    elif a == PredictionContext.EMPTY and b == PredictionContext.EMPTY:
        return PredictionContext.EMPTY
    elif a == PredictionContext.EMPTY:
        payloads = [b.returnState, PredictionContext.EMPTY_RETURN_STATE]
        parents = [b.parentCtx, None]
        return ArrayPredictionContext(parents, payloads)
    elif b == PredictionContext.EMPTY:
        payloads = [a.returnState, PredictionContext.EMPTY_RETURN_STATE]
        parents = [a.parentCtx, None]
        return ArrayPredictionContext(parents, payloads)
    return None

def mergeArrays(a: ArrayPredictionContext, b: ArrayPredictionContext, rootIsWildcard: bool, mergeCache: dict):
    if False:
        print('Hello World!')
    if mergeCache is not None:
        previous = mergeCache.get((a, b), None)
        if previous is not None:
            if _trace_atn_sim:
                print('mergeArrays a=' + str(a) + ',b=' + str(b) + ' -> previous')
            return previous
        previous = mergeCache.get((b, a), None)
        if previous is not None:
            if _trace_atn_sim:
                print('mergeArrays a=' + str(a) + ',b=' + str(b) + ' -> previous')
            return previous
    i = 0
    j = 0
    k = 0
    mergedReturnStates = [None] * (len(a.returnStates) + len(b.returnStates))
    mergedParents = [None] * len(mergedReturnStates)
    while i < len(a.returnStates) and j < len(b.returnStates):
        a_parent = a.parents[i]
        b_parent = b.parents[j]
        if a.returnStates[i] == b.returnStates[j]:
            payload = a.returnStates[i]
            bothDollars = payload == PredictionContext.EMPTY_RETURN_STATE and a_parent is None and (b_parent is None)
            ax_ax = (a_parent is not None and b_parent is not None) and a_parent == b_parent
            if bothDollars or ax_ax:
                mergedParents[k] = a_parent
                mergedReturnStates[k] = payload
            else:
                mergedParent = merge(a_parent, b_parent, rootIsWildcard, mergeCache)
                mergedParents[k] = mergedParent
                mergedReturnStates[k] = payload
            i += 1
            j += 1
        elif a.returnStates[i] < b.returnStates[j]:
            mergedParents[k] = a_parent
            mergedReturnStates[k] = a.returnStates[i]
            i += 1
        else:
            mergedParents[k] = b_parent
            mergedReturnStates[k] = b.returnStates[j]
            j += 1
        k += 1
    if i < len(a.returnStates):
        for p in range(i, len(a.returnStates)):
            mergedParents[k] = a.parents[p]
            mergedReturnStates[k] = a.returnStates[p]
            k += 1
    else:
        for p in range(j, len(b.returnStates)):
            mergedParents[k] = b.parents[p]
            mergedReturnStates[k] = b.returnStates[p]
            k += 1
    if k < len(mergedParents):
        if k == 1:
            merged = SingletonPredictionContext.create(mergedParents[0], mergedReturnStates[0])
            if mergeCache is not None:
                mergeCache[a, b] = merged
            return merged
        mergedParents = mergedParents[0:k]
        mergedReturnStates = mergedReturnStates[0:k]
    merged = ArrayPredictionContext(mergedParents, mergedReturnStates)
    if merged == a:
        if mergeCache is not None:
            mergeCache[a, b] = a
        if _trace_atn_sim:
            print('mergeArrays a=' + str(a) + ',b=' + str(b) + ' -> a')
        return a
    if merged == b:
        if mergeCache is not None:
            mergeCache[a, b] = b
        if _trace_atn_sim:
            print('mergeArrays a=' + str(a) + ',b=' + str(b) + ' -> b')
        return b
    combineCommonParents(mergedParents)
    if mergeCache is not None:
        mergeCache[a, b] = merged
    if _trace_atn_sim:
        print('mergeArrays a=' + str(a) + ',b=' + str(b) + ' -> ' + str(M))
    return merged

def combineCommonParents(parents: list):
    if False:
        print('Hello World!')
    uniqueParents = dict()
    for p in range(0, len(parents)):
        parent = parents[p]
        if uniqueParents.get(parent, None) is None:
            uniqueParents[parent] = parent
    for p in range(0, len(parents)):
        parents[p] = uniqueParents[parents[p]]

def getCachedPredictionContext(context: PredictionContext, contextCache: PredictionContextCache, visited: dict):
    if False:
        while True:
            i = 10
    if context.isEmpty():
        return context
    existing = visited.get(context)
    if existing is not None:
        return existing
    existing = contextCache.get(context)
    if existing is not None:
        visited[context] = existing
        return existing
    changed = False
    parents = [None] * len(context)
    for i in range(0, len(parents)):
        parent = getCachedPredictionContext(context.getParent(i), contextCache, visited)
        if changed or parent is not context.getParent(i):
            if not changed:
                parents = [context.getParent(j) for j in range(len(context))]
                changed = True
            parents[i] = parent
    if not changed:
        contextCache.add(context)
        visited[context] = context
        return context
    updated = None
    if len(parents) == 0:
        updated = PredictionContext.EMPTY
    elif len(parents) == 1:
        updated = SingletonPredictionContext.create(parents[0], context.getReturnState(0))
    else:
        updated = ArrayPredictionContext(parents, context.returnStates)
    contextCache.add(updated)
    visited[updated] = updated
    visited[context] = updated
    return updated

def getAllContextNodes(context: PredictionContext, nodes: list=None, visited: dict=None):
    if False:
        return 10
    if nodes is None:
        nodes = list()
        return getAllContextNodes(context, nodes, visited)
    elif visited is None:
        visited = dict()
        return getAllContextNodes(context, nodes, visited)
    else:
        if context is None or visited.get(context, None) is not None:
            return nodes
        visited.put(context, context)
        nodes.add(context)
        for i in range(0, len(context)):
            getAllContextNodes(context.getParent(i), nodes, visited)
        return nodes