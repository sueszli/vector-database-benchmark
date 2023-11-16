from antlr4.Recognizer import Recognizer
from antlr4.RuleContext import RuleContext
from io import StringIO

class SemanticContext(object):
    NONE = None

    def eval(self, parser: Recognizer, outerContext: RuleContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def evalPrecedence(self, parser: Recognizer, outerContext: RuleContext):
        if False:
            i = 10
            return i + 15
        return self
AND = None

def andContext(a: SemanticContext, b: SemanticContext):
    if False:
        for i in range(10):
            print('nop')
    if a is None or a is SemanticContext.NONE:
        return b
    if b is None or b is SemanticContext.NONE:
        return a
    result = AND(a, b)
    if len(result.opnds) == 1:
        return result.opnds[0]
    else:
        return result
OR = None

def orContext(a: SemanticContext, b: SemanticContext):
    if False:
        for i in range(10):
            print('nop')
    if a is None:
        return b
    if b is None:
        return a
    if a is SemanticContext.NONE or b is SemanticContext.NONE:
        return SemanticContext.NONE
    result = OR(a, b)
    if len(result.opnds) == 1:
        return result.opnds[0]
    else:
        return result

def filterPrecedencePredicates(collection: set):
    if False:
        print('Hello World!')
    return [context for context in collection if isinstance(context, PrecedencePredicate)]

class EmptySemanticContext(SemanticContext):
    pass

class Predicate(SemanticContext):
    __slots__ = ('ruleIndex', 'predIndex', 'isCtxDependent')

    def __init__(self, ruleIndex: int=-1, predIndex: int=-1, isCtxDependent: bool=False):
        if False:
            for i in range(10):
                print('nop')
        self.ruleIndex = ruleIndex
        self.predIndex = predIndex
        self.isCtxDependent = isCtxDependent

    def eval(self, parser: Recognizer, outerContext: RuleContext):
        if False:
            while True:
                i = 10
        localctx = outerContext if self.isCtxDependent else None
        return parser.sempred(localctx, self.ruleIndex, self.predIndex)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash((self.ruleIndex, self.predIndex, self.isCtxDependent))

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if self is other:
            return True
        elif not isinstance(other, Predicate):
            return False
        return self.ruleIndex == other.ruleIndex and self.predIndex == other.predIndex and (self.isCtxDependent == other.isCtxDependent)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '{' + str(self.ruleIndex) + ':' + str(self.predIndex) + '}?'

class PrecedencePredicate(SemanticContext):

    def __init__(self, precedence: int=0):
        if False:
            print('Hello World!')
        self.precedence = precedence

    def eval(self, parser: Recognizer, outerContext: RuleContext):
        if False:
            for i in range(10):
                print('nop')
        return parser.precpred(outerContext, self.precedence)

    def evalPrecedence(self, parser: Recognizer, outerContext: RuleContext):
        if False:
            for i in range(10):
                print('nop')
        if parser.precpred(outerContext, self.precedence):
            return SemanticContext.NONE
        else:
            return None

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        return self.precedence < other.precedence

    def __hash__(self):
        if False:
            print('Hello World!')
        return 31

    def __eq__(self, other):
        if False:
            return 10
        if self is other:
            return True
        elif not isinstance(other, PrecedencePredicate):
            return False
        else:
            return self.precedence == other.precedence

    def __str__(self):
        if False:
            return 10
        return '{' + str(self.precedence) + '>=prec}?'
del AND

class AND(SemanticContext):
    __slots__ = 'opnds'

    def __init__(self, a: SemanticContext, b: SemanticContext):
        if False:
            while True:
                i = 10
        operands = set()
        if isinstance(a, AND):
            operands.update(a.opnds)
        else:
            operands.add(a)
        if isinstance(b, AND):
            operands.update(b.opnds)
        else:
            operands.add(b)
        precedencePredicates = filterPrecedencePredicates(operands)
        if len(precedencePredicates) > 0:
            reduced = min(precedencePredicates)
            operands.add(reduced)
        self.opnds = list(operands)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if self is other:
            return True
        elif not isinstance(other, AND):
            return False
        else:
            return self.opnds == other.opnds

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        h = 0
        for o in self.opnds:
            h = hash((h, o))
        return hash((h, 'AND'))

    def eval(self, parser: Recognizer, outerContext: RuleContext):
        if False:
            i = 10
            return i + 15
        return all((opnd.eval(parser, outerContext) for opnd in self.opnds))

    def evalPrecedence(self, parser: Recognizer, outerContext: RuleContext):
        if False:
            for i in range(10):
                print('nop')
        differs = False
        operands = []
        for context in self.opnds:
            evaluated = context.evalPrecedence(parser, outerContext)
            differs |= evaluated is not context
            if evaluated is None:
                return None
            elif evaluated is not SemanticContext.NONE:
                operands.append(evaluated)
        if not differs:
            return self
        if len(operands) == 0:
            return SemanticContext.NONE
        result = None
        for o in operands:
            result = o if result is None else andContext(result, o)
        return result

    def __str__(self):
        if False:
            while True:
                i = 10
        with StringIO() as buf:
            first = True
            for o in self.opnds:
                if not first:
                    buf.write('&&')
                buf.write(str(o))
                first = False
            return buf.getvalue()
del OR

class OR(SemanticContext):
    __slots__ = 'opnds'

    def __init__(self, a: SemanticContext, b: SemanticContext):
        if False:
            print('Hello World!')
        operands = set()
        if isinstance(a, OR):
            operands.update(a.opnds)
        else:
            operands.add(a)
        if isinstance(b, OR):
            operands.update(b.opnds)
        else:
            operands.add(b)
        precedencePredicates = filterPrecedencePredicates(operands)
        if len(precedencePredicates) > 0:
            s = sorted(precedencePredicates)
            reduced = s[-1]
            operands.add(reduced)
        self.opnds = list(operands)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if self is other:
            return True
        elif not isinstance(other, OR):
            return False
        else:
            return self.opnds == other.opnds

    def __hash__(self):
        if False:
            while True:
                i = 10
        h = 0
        for o in self.opnds:
            h = hash((h, o))
        return hash((h, 'OR'))

    def eval(self, parser: Recognizer, outerContext: RuleContext):
        if False:
            return 10
        return any((opnd.eval(parser, outerContext) for opnd in self.opnds))

    def evalPrecedence(self, parser: Recognizer, outerContext: RuleContext):
        if False:
            for i in range(10):
                print('nop')
        differs = False
        operands = []
        for context in self.opnds:
            evaluated = context.evalPrecedence(parser, outerContext)
            differs |= evaluated is not context
            if evaluated is SemanticContext.NONE:
                return SemanticContext.NONE
            elif evaluated is not None:
                operands.append(evaluated)
        if not differs:
            return self
        if len(operands) == 0:
            return None
        result = None
        for o in operands:
            result = o if result is None else orContext(result, o)
        return result

    def __str__(self):
        if False:
            while True:
                i = 10
        with StringIO() as buf:
            first = True
            for o in self.opnds:
                if not first:
                    buf.write('||')
                buf.write(str(o))
                first = False
            return buf.getvalue()
SemanticContext.NONE = EmptySemanticContext()