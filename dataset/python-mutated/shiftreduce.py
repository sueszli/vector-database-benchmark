from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import Tree

class ShiftReduceParser(ParserI):
    """
    A simple bottom-up CFG parser that uses two operations, "shift"
    and "reduce", to find a single parse for a text.

    ``ShiftReduceParser`` maintains a stack, which records the
    structure of a portion of the text.  This stack is a list of
    strings and Trees that collectively cover a portion of
    the text.  For example, while parsing the sentence "the dog saw
    the man" with a typical grammar, ``ShiftReduceParser`` will produce
    the following stack, which covers "the dog saw"::

       [(NP: (Det: 'the') (N: 'dog')), (V: 'saw')]

    ``ShiftReduceParser`` attempts to extend the stack to cover the
    entire text, and to combine the stack elements into a single tree,
    producing a complete parse for the sentence.

    Initially, the stack is empty.  It is extended to cover the text,
    from left to right, by repeatedly applying two operations:

      - "shift" moves a token from the beginning of the text to the
        end of the stack.
      - "reduce" uses a CFG production to combine the rightmost stack
        elements into a single Tree.

    Often, more than one operation can be performed on a given stack.
    In this case, ``ShiftReduceParser`` uses the following heuristics
    to decide which operation to perform:

      - Only shift if no reductions are available.
      - If multiple reductions are available, then apply the reduction
        whose CFG production is listed earliest in the grammar.

    Note that these heuristics are not guaranteed to choose an
    operation that leads to a parse of the text.  Also, if multiple
    parses exists, ``ShiftReduceParser`` will return at most one of
    them.

    :see: ``nltk.grammar``
    """

    def __init__(self, grammar, trace=0):
        if False:
            print('Hello World!')
        '\n        Create a new ``ShiftReduceParser``, that uses ``grammar`` to\n        parse texts.\n\n        :type grammar: Grammar\n        :param grammar: The grammar used to parse texts.\n        :type trace: int\n        :param trace: The level of tracing that should be used when\n            parsing a text.  ``0`` will generate no tracing output;\n            and higher numbers will produce more verbose tracing\n            output.\n        '
        self._grammar = grammar
        self._trace = trace
        self._check_grammar()

    def grammar(self):
        if False:
            for i in range(10):
                print('nop')
        return self._grammar

    def parse(self, tokens):
        if False:
            for i in range(10):
                print('nop')
        tokens = list(tokens)
        self._grammar.check_coverage(tokens)
        stack = []
        remaining_text = tokens
        if self._trace:
            print('Parsing %r' % ' '.join(tokens))
            self._trace_stack(stack, remaining_text)
        while len(remaining_text) > 0:
            self._shift(stack, remaining_text)
            while self._reduce(stack, remaining_text):
                pass
        if len(stack) == 1:
            if stack[0].label() == self._grammar.start().symbol():
                yield stack[0]

    def _shift(self, stack, remaining_text):
        if False:
            return 10
        '\n        Move a token from the beginning of ``remaining_text`` to the\n        end of ``stack``.\n\n        :type stack: list(str and Tree)\n        :param stack: A list of strings and Trees, encoding\n            the structure of the text that has been parsed so far.\n        :type remaining_text: list(str)\n        :param remaining_text: The portion of the text that is not yet\n            covered by ``stack``.\n        :rtype: None\n        '
        stack.append(remaining_text[0])
        remaining_text.remove(remaining_text[0])
        if self._trace:
            self._trace_shift(stack, remaining_text)

    def _match_rhs(self, rhs, rightmost_stack):
        if False:
            while True:
                i = 10
        "\n        :rtype: bool\n        :return: true if the right hand side of a CFG production\n            matches the rightmost elements of the stack.  ``rhs``\n            matches ``rightmost_stack`` if they are the same length,\n            and each element of ``rhs`` matches the corresponding\n            element of ``rightmost_stack``.  A nonterminal element of\n            ``rhs`` matches any Tree whose node value is equal\n            to the nonterminal's symbol.  A terminal element of ``rhs``\n            matches any string whose type is equal to the terminal.\n        :type rhs: list(terminal and Nonterminal)\n        :param rhs: The right hand side of a CFG production.\n        :type rightmost_stack: list(string and Tree)\n        :param rightmost_stack: The rightmost elements of the parser's\n            stack.\n        "
        if len(rightmost_stack) != len(rhs):
            return False
        for i in range(len(rightmost_stack)):
            if isinstance(rightmost_stack[i], Tree):
                if not isinstance(rhs[i], Nonterminal):
                    return False
                if rightmost_stack[i].label() != rhs[i].symbol():
                    return False
            else:
                if isinstance(rhs[i], Nonterminal):
                    return False
                if rightmost_stack[i] != rhs[i]:
                    return False
        return True

    def _reduce(self, stack, remaining_text, production=None):
        if False:
            print('Hello World!')
        "\n        Find a CFG production whose right hand side matches the\n        rightmost stack elements; and combine those stack elements\n        into a single Tree, with the node specified by the\n        production's left-hand side.  If more than one CFG production\n        matches the stack, then use the production that is listed\n        earliest in the grammar.  The new Tree replaces the\n        elements in the stack.\n\n        :rtype: Production or None\n        :return: If a reduction is performed, then return the CFG\n            production that the reduction is based on; otherwise,\n            return false.\n        :type stack: list(string and Tree)\n        :param stack: A list of strings and Trees, encoding\n            the structure of the text that has been parsed so far.\n        :type remaining_text: list(str)\n        :param remaining_text: The portion of the text that is not yet\n            covered by ``stack``.\n        "
        if production is None:
            productions = self._grammar.productions()
        else:
            productions = [production]
        for production in productions:
            rhslen = len(production.rhs())
            if self._match_rhs(production.rhs(), stack[-rhslen:]):
                tree = Tree(production.lhs().symbol(), stack[-rhslen:])
                stack[-rhslen:] = [tree]
                if self._trace:
                    self._trace_reduce(stack, production, remaining_text)
                return production
        return None

    def trace(self, trace=2):
        if False:
            print('Hello World!')
        '\n        Set the level of tracing output that should be generated when\n        parsing a text.\n\n        :type trace: int\n        :param trace: The trace level.  A trace level of ``0`` will\n            generate no tracing output; and higher trace levels will\n            produce more verbose tracing output.\n        :rtype: None\n        '
        self._trace = trace

    def _trace_stack(self, stack, remaining_text, marker=' '):
        if False:
            return 10
        "\n        Print trace output displaying the given stack and text.\n\n        :rtype: None\n        :param marker: A character that is printed to the left of the\n            stack.  This is used with trace level 2 to print 'S'\n            before shifted stacks and 'R' before reduced stacks.\n        "
        s = '  ' + marker + ' [ '
        for elt in stack:
            if isinstance(elt, Tree):
                s += repr(Nonterminal(elt.label())) + ' '
            else:
                s += repr(elt) + ' '
        s += '* ' + ' '.join(remaining_text) + ']'
        print(s)

    def _trace_shift(self, stack, remaining_text):
        if False:
            i = 10
            return i + 15
        '\n        Print trace output displaying that a token has been shifted.\n\n        :rtype: None\n        '
        if self._trace > 2:
            print('Shift %r:' % stack[-1])
        if self._trace == 2:
            self._trace_stack(stack, remaining_text, 'S')
        elif self._trace > 0:
            self._trace_stack(stack, remaining_text)

    def _trace_reduce(self, stack, production, remaining_text):
        if False:
            print('Hello World!')
        '\n        Print trace output displaying that ``production`` was used to\n        reduce ``stack``.\n\n        :rtype: None\n        '
        if self._trace > 2:
            rhs = ' '.join(production.rhs())
            print(f'Reduce {production.lhs()!r} <- {rhs}')
        if self._trace == 2:
            self._trace_stack(stack, remaining_text, 'R')
        elif self._trace > 1:
            self._trace_stack(stack, remaining_text)

    def _check_grammar(self):
        if False:
            i = 10
            return i + 15
        '\n        Check to make sure that all of the CFG productions are\n        potentially useful.  If any productions can never be used,\n        then print a warning.\n\n        :rtype: None\n        '
        productions = self._grammar.productions()
        for i in range(len(productions)):
            for j in range(i + 1, len(productions)):
                rhs1 = productions[i].rhs()
                rhs2 = productions[j].rhs()
                if rhs1[:len(rhs2)] == rhs2:
                    print('Warning: %r will never be used' % productions[i])

class SteppingShiftReduceParser(ShiftReduceParser):
    """
    A ``ShiftReduceParser`` that allows you to setp through the parsing
    process, performing a single operation at a time.  It also allows
    you to change the parser's grammar midway through parsing a text.

    The ``initialize`` method is used to start parsing a text.
    ``shift`` performs a single shift operation, and ``reduce`` performs
    a single reduce operation.  ``step`` will perform a single reduce
    operation if possible; otherwise, it will perform a single shift
    operation.  ``parses`` returns the set of parses that have been
    found by the parser.

    :ivar _history: A list of ``(stack, remaining_text)`` pairs,
        containing all of the previous states of the parser.  This
        history is used to implement the ``undo`` operation.
    :see: ``nltk.grammar``
    """

    def __init__(self, grammar, trace=0):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(grammar, trace)
        self._stack = None
        self._remaining_text = None
        self._history = []

    def parse(self, tokens):
        if False:
            for i in range(10):
                print('nop')
        tokens = list(tokens)
        self.initialize(tokens)
        while self.step():
            pass
        return self.parses()

    def stack(self):
        if False:
            i = 10
            return i + 15
        "\n        :return: The parser's stack.\n        :rtype: list(str and Tree)\n        "
        return self._stack

    def remaining_text(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :return: The portion of the text that is not yet covered by the\n            stack.\n        :rtype: list(str)\n        '
        return self._remaining_text

    def initialize(self, tokens):
        if False:
            print('Hello World!')
        "\n        Start parsing a given text.  This sets the parser's stack to\n        ``[]`` and sets its remaining text to ``tokens``.\n        "
        self._stack = []
        self._remaining_text = tokens
        self._history = []

    def step(self):
        if False:
            while True:
                i = 10
        '\n        Perform a single parsing operation.  If a reduction is\n        possible, then perform that reduction, and return the\n        production that it is based on.  Otherwise, if a shift is\n        possible, then perform it, and return True.  Otherwise,\n        return False.\n\n        :return: False if no operation was performed; True if a shift was\n            performed; and the CFG production used to reduce if a\n            reduction was performed.\n        :rtype: Production or bool\n        '
        return self.reduce() or self.shift()

    def shift(self):
        if False:
            while True:
                i = 10
        '\n        Move a token from the beginning of the remaining text to the\n        end of the stack.  If there are no more tokens in the\n        remaining text, then do nothing.\n\n        :return: True if the shift operation was successful.\n        :rtype: bool\n        '
        if len(self._remaining_text) == 0:
            return False
        self._history.append((self._stack[:], self._remaining_text[:]))
        self._shift(self._stack, self._remaining_text)
        return True

    def reduce(self, production=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Use ``production`` to combine the rightmost stack elements into\n        a single Tree.  If ``production`` does not match the\n        rightmost stack elements, then do nothing.\n\n        :return: The production used to reduce the stack, if a\n            reduction was performed.  If no reduction was performed,\n            return None.\n\n        :rtype: Production or None\n        '
        self._history.append((self._stack[:], self._remaining_text[:]))
        return_val = self._reduce(self._stack, self._remaining_text, production)
        if not return_val:
            self._history.pop()
        return return_val

    def undo(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the parser to its state before the most recent\n        shift or reduce operation.  Calling ``undo`` repeatedly return\n        the parser to successively earlier states.  If no shift or\n        reduce operations have been performed, ``undo`` will make no\n        changes.\n\n        :return: true if an operation was successfully undone.\n        :rtype: bool\n        '
        if len(self._history) == 0:
            return False
        (self._stack, self._remaining_text) = self._history.pop()
        return True

    def reducible_productions(self):
        if False:
            while True:
                i = 10
        '\n        :return: A list of the productions for which reductions are\n            available for the current parser state.\n        :rtype: list(Production)\n        '
        productions = []
        for production in self._grammar.productions():
            rhslen = len(production.rhs())
            if self._match_rhs(production.rhs(), self._stack[-rhslen:]):
                productions.append(production)
        return productions

    def parses(self):
        if False:
            i = 10
            return i + 15
        '\n        :return: An iterator of the parses that have been found by this\n            parser so far.\n        :rtype: iter(Tree)\n        '
        if len(self._remaining_text) == 0 and len(self._stack) == 1 and (self._stack[0].label() == self._grammar.start().symbol()):
            yield self._stack[0]

    def set_grammar(self, grammar):
        if False:
            while True:
                i = 10
        '\n        Change the grammar used to parse texts.\n\n        :param grammar: The new grammar.\n        :type grammar: CFG\n        '
        self._grammar = grammar

def demo():
    if False:
        while True:
            i = 10
    '\n    A demonstration of the shift-reduce parser.\n    '
    from nltk import CFG, parse
    grammar = CFG.fromstring("\n    S -> NP VP\n    NP -> Det N | Det N PP\n    VP -> V NP | V NP PP\n    PP -> P NP\n    NP -> 'I'\n    N -> 'man' | 'park' | 'telescope' | 'dog'\n    Det -> 'the' | 'a'\n    P -> 'in' | 'with'\n    V -> 'saw'\n    ")
    sent = 'I saw a man in the park'.split()
    parser = parse.ShiftReduceParser(grammar, trace=2)
    for p in parser.parse(sent):
        print(p)
if __name__ == '__main__':
    demo()