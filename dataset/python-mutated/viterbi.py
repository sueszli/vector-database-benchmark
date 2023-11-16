from functools import reduce
from nltk.parse.api import ParserI
from nltk.tree import ProbabilisticTree, Tree

class ViterbiParser(ParserI):
    """
    A bottom-up ``PCFG`` parser that uses dynamic programming to find
    the single most likely parse for a text.  The ``ViterbiParser`` parser
    parses texts by filling in a "most likely constituent table".
    This table records the most probable tree representation for any
    given span and node value.  In particular, it has an entry for
    every start index, end index, and node value, recording the most
    likely subtree that spans from the start index to the end index,
    and has the given node value.

    The ``ViterbiParser`` parser fills in this table incrementally.  It starts
    by filling in all entries for constituents that span one element
    of text (i.e., entries where the end index is one greater than the
    start index).  After it has filled in all table entries for
    constituents that span one element of text, it fills in the
    entries for constitutants that span two elements of text.  It
    continues filling in the entries for constituents spanning larger
    and larger portions of the text, until the entire table has been
    filled.  Finally, it returns the table entry for a constituent
    spanning the entire text, whose node value is the grammar's start
    symbol.

    In order to find the most likely constituent with a given span and
    node value, the ``ViterbiParser`` parser considers all productions that
    could produce that node value.  For each production, it finds all
    children that collectively cover the span and have the node values
    specified by the production's right hand side.  If the probability
    of the tree formed by applying the production to the children is
    greater than the probability of the current entry in the table,
    then the table is updated with this new tree.

    A pseudo-code description of the algorithm used by
    ``ViterbiParser`` is:

    | Create an empty most likely constituent table, *MLC*.
    | For width in 1...len(text):
    |   For start in 1...len(text)-width:
    |     For prod in grammar.productions:
    |       For each sequence of subtrees [t[1], t[2], ..., t[n]] in MLC,
    |         where t[i].label()==prod.rhs[i],
    |         and the sequence covers [start:start+width]:
    |           old_p = MLC[start, start+width, prod.lhs]
    |           new_p = P(t[1])P(t[1])...P(t[n])P(prod)
    |           if new_p > old_p:
    |             new_tree = Tree(prod.lhs, t[1], t[2], ..., t[n])
    |             MLC[start, start+width, prod.lhs] = new_tree
    | Return MLC[0, len(text), start_symbol]

    :type _grammar: PCFG
    :ivar _grammar: The grammar used to parse sentences.
    :type _trace: int
    :ivar _trace: The level of tracing output that should be generated
        when parsing a text.
    """

    def __init__(self, grammar, trace=0):
        if False:
            print('Hello World!')
        '\n        Create a new ``ViterbiParser`` parser, that uses ``grammar`` to\n        parse texts.\n\n        :type grammar: PCFG\n        :param grammar: The grammar used to parse texts.\n        :type trace: int\n        :param trace: The level of tracing that should be used when\n            parsing a text.  ``0`` will generate no tracing output;\n            and higher numbers will produce more verbose tracing\n            output.\n        '
        self._grammar = grammar
        self._trace = trace

    def grammar(self):
        if False:
            return 10
        return self._grammar

    def trace(self, trace=2):
        if False:
            return 10
        '\n        Set the level of tracing output that should be generated when\n        parsing a text.\n\n        :type trace: int\n        :param trace: The trace level.  A trace level of ``0`` will\n            generate no tracing output; and higher trace levels will\n            produce more verbose tracing output.\n        :rtype: None\n        '
        self._trace = trace

    def parse(self, tokens):
        if False:
            while True:
                i = 10
        tokens = list(tokens)
        self._grammar.check_coverage(tokens)
        constituents = {}
        if self._trace:
            print('Inserting tokens into the most likely' + ' constituents table...')
        for index in range(len(tokens)):
            token = tokens[index]
            constituents[index, index + 1, token] = token
            if self._trace > 1:
                self._trace_lexical_insertion(token, index, len(tokens))
        for length in range(1, len(tokens) + 1):
            if self._trace:
                print('Finding the most likely constituents' + ' spanning %d text elements...' % length)
            for start in range(len(tokens) - length + 1):
                span = (start, start + length)
                self._add_constituents_spanning(span, constituents, tokens)
        tree = constituents.get((0, len(tokens), self._grammar.start()))
        if tree is not None:
            yield tree

    def _add_constituents_spanning(self, span, constituents, tokens):
        if False:
            return 10
        '\n        Find any constituents that might cover ``span``, and add them\n        to the most likely constituents table.\n\n        :rtype: None\n        :type span: tuple(int, int)\n        :param span: The section of the text for which we are\n            trying to find possible constituents.  The span is\n            specified as a pair of integers, where the first integer\n            is the index of the first token that should be included in\n            the constituent; and the second integer is the index of\n            the first token that should not be included in the\n            constituent.  I.e., the constituent should cover\n            ``text[span[0]:span[1]]``, where ``text`` is the text\n            that we are parsing.\n\n        :type constituents: dict(tuple(int,int,Nonterminal) -> ProbabilisticToken or ProbabilisticTree)\n        :param constituents: The most likely constituents table.  This\n            table records the most probable tree representation for\n            any given span and node value.  In particular,\n            ``constituents(s,e,nv)`` is the most likely\n            ``ProbabilisticTree`` that covers ``text[s:e]``\n            and has a node value ``nv.symbol()``, where ``text``\n            is the text that we are parsing.  When\n            ``_add_constituents_spanning`` is called, ``constituents``\n            should contain all possible constituents that are shorter\n            than ``span``.\n\n        :type tokens: list of tokens\n        :param tokens: The text we are parsing.  This is only used for\n            trace output.\n        '
        changed = True
        while changed:
            changed = False
            instantiations = self._find_instantiations(span, constituents)
            for (production, children) in instantiations:
                subtrees = [c for c in children if isinstance(c, Tree)]
                p = reduce(lambda pr, t: pr * t.prob(), subtrees, production.prob())
                node = production.lhs().symbol()
                tree = ProbabilisticTree(node, children, prob=p)
                c = constituents.get((span[0], span[1], production.lhs()))
                if self._trace > 1:
                    if c is None or c != tree:
                        if c is None or c.prob() < tree.prob():
                            print('   Insert:', end=' ')
                        else:
                            print('  Discard:', end=' ')
                        self._trace_production(production, p, span, len(tokens))
                if c is None or c.prob() < tree.prob():
                    constituents[span[0], span[1], production.lhs()] = tree
                    changed = True

    def _find_instantiations(self, span, constituents):
        if False:
            return 10
        '\n        :return: a list of the production instantiations that cover a\n            given span of the text.  A "production instantiation" is\n            a tuple containing a production and a list of children,\n            where the production\'s right hand side matches the list of\n            children; and the children cover ``span``.  :rtype: list\n            of ``pair`` of ``Production``, (list of\n            (``ProbabilisticTree`` or token.\n\n        :type span: tuple(int, int)\n        :param span: The section of the text for which we are\n            trying to find production instantiations.  The span is\n            specified as a pair of integers, where the first integer\n            is the index of the first token that should be covered by\n            the production instantiation; and the second integer is\n            the index of the first token that should not be covered by\n            the production instantiation.\n        :type constituents: dict(tuple(int,int,Nonterminal) -> ProbabilisticToken or ProbabilisticTree)\n        :param constituents: The most likely constituents table.  This\n            table records the most probable tree representation for\n            any given span and node value.  See the module\n            documentation for more information.\n        '
        rv = []
        for production in self._grammar.productions():
            childlists = self._match_rhs(production.rhs(), span, constituents)
            for childlist in childlists:
                rv.append((production, childlist))
        return rv

    def _match_rhs(self, rhs, span, constituents):
        if False:
            for i in range(10):
                print('nop')
        "\n        :return: a set of all the lists of children that cover ``span``\n            and that match ``rhs``.\n        :rtype: list(list(ProbabilisticTree or token)\n\n        :type rhs: list(Nonterminal or any)\n        :param rhs: The list specifying what kinds of children need to\n            cover ``span``.  Each nonterminal in ``rhs`` specifies\n            that the corresponding child should be a tree whose node\n            value is that nonterminal's symbol.  Each terminal in ``rhs``\n            specifies that the corresponding child should be a token\n            whose type is that terminal.\n        :type span: tuple(int, int)\n        :param span: The section of the text for which we are\n            trying to find child lists.  The span is specified as a\n            pair of integers, where the first integer is the index of\n            the first token that should be covered by the child list;\n            and the second integer is the index of the first token\n            that should not be covered by the child list.\n        :type constituents: dict(tuple(int,int,Nonterminal) -> ProbabilisticToken or ProbabilisticTree)\n        :param constituents: The most likely constituents table.  This\n            table records the most probable tree representation for\n            any given span and node value.  See the module\n            documentation for more information.\n        "
        (start, end) = span
        if start >= end and rhs == ():
            return [[]]
        if start >= end or rhs == ():
            return []
        childlists = []
        for split in range(start, end + 1):
            l = constituents.get((start, split, rhs[0]))
            if l is not None:
                rights = self._match_rhs(rhs[1:], (split, end), constituents)
                childlists += [[l] + r for r in rights]
        return childlists

    def _trace_production(self, production, p, span, width):
        if False:
            print('Hello World!')
        '\n        Print trace output indicating that a given production has been\n        applied at a given location.\n\n        :param production: The production that has been applied\n        :type production: Production\n        :param p: The probability of the tree produced by the production.\n        :type p: float\n        :param span: The span of the production\n        :type span: tuple\n        :rtype: None\n        '
        str = '|' + '.' * span[0]
        str += '=' * (span[1] - span[0])
        str += '.' * (width - span[1]) + '| '
        str += '%s' % production
        if self._trace > 2:
            str = f'{str:<40} {p:12.10f} '
        print(str)

    def _trace_lexical_insertion(self, token, index, width):
        if False:
            print('Hello World!')
        str = '   Insert: |' + '.' * index + '=' + '.' * (width - index - 1) + '| '
        str += f'{token}'
        print(str)

    def __repr__(self):
        if False:
            return 10
        return '<ViterbiParser for %r>' % self._grammar

def demo():
    if False:
        return 10
    '\n    A demonstration of the probabilistic parsers.  The user is\n    prompted to select which demo to run, and how many parses should\n    be found; and then each parser is run on the same demo, and a\n    summary of the results are displayed.\n    '
    import sys
    import time
    from nltk import tokenize
    from nltk.grammar import PCFG
    from nltk.parse import ViterbiParser
    toy_pcfg1 = PCFG.fromstring("\n    S -> NP VP [1.0]\n    NP -> Det N [0.5] | NP PP [0.25] | 'John' [0.1] | 'I' [0.15]\n    Det -> 'the' [0.8] | 'my' [0.2]\n    N -> 'man' [0.5] | 'telescope' [0.5]\n    VP -> VP PP [0.1] | V NP [0.7] | V [0.2]\n    V -> 'ate' [0.35] | 'saw' [0.65]\n    PP -> P NP [1.0]\n    P -> 'with' [0.61] | 'under' [0.39]\n    ")
    toy_pcfg2 = PCFG.fromstring("\n    S    -> NP VP         [1.0]\n    VP   -> V NP          [.59]\n    VP   -> V             [.40]\n    VP   -> VP PP         [.01]\n    NP   -> Det N         [.41]\n    NP   -> Name          [.28]\n    NP   -> NP PP         [.31]\n    PP   -> P NP          [1.0]\n    V    -> 'saw'         [.21]\n    V    -> 'ate'         [.51]\n    V    -> 'ran'         [.28]\n    N    -> 'boy'         [.11]\n    N    -> 'cookie'      [.12]\n    N    -> 'table'       [.13]\n    N    -> 'telescope'   [.14]\n    N    -> 'hill'        [.5]\n    Name -> 'Jack'        [.52]\n    Name -> 'Bob'         [.48]\n    P    -> 'with'        [.61]\n    P    -> 'under'       [.39]\n    Det  -> 'the'         [.41]\n    Det  -> 'a'           [.31]\n    Det  -> 'my'          [.28]\n    ")
    demos = [('I saw the man with my telescope', toy_pcfg1), ('the boy saw Jack with Bob under the table with a telescope', toy_pcfg2)]
    print()
    for i in range(len(demos)):
        print(f'{i + 1:>3}: {demos[i][0]}')
        print('     %r' % demos[i][1])
        print()
    print('Which demo (%d-%d)? ' % (1, len(demos)), end=' ')
    try:
        snum = int(sys.stdin.readline().strip()) - 1
        (sent, grammar) = demos[snum]
    except:
        print('Bad sentence number')
        return
    tokens = sent.split()
    parser = ViterbiParser(grammar)
    all_parses = {}
    print(f'\nsent: {sent}\nparser: {parser}\ngrammar: {grammar}')
    parser.trace(3)
    t = time.time()
    parses = parser.parse_all(tokens)
    time = time.time() - t
    average = reduce(lambda a, b: a + b.prob(), parses, 0) / len(parses) if parses else 0
    num_parses = len(parses)
    for p in parses:
        all_parses[p.freeze()] = 1
    print()
    print('Time (secs)   # Parses   Average P(parse)')
    print('-----------------------------------------')
    print('%11.4f%11d%19.14f' % (time, num_parses, average))
    parses = all_parses.keys()
    if parses:
        p = reduce(lambda a, b: a + b.prob(), parses, 0) / len(parses)
    else:
        p = 0
    print('------------------------------------------')
    print('%11s%11d%19.14f' % ('n/a', len(parses), p))
    print()
    print('Draw parses (y/n)? ', end=' ')
    if sys.stdin.readline().strip().lower().startswith('y'):
        from nltk.draw.tree import draw_trees
        print('  please wait...')
        draw_trees(*parses)
    print()
    print('Print parses (y/n)? ', end=' ')
    if sys.stdin.readline().strip().lower().startswith('y'):
        for parse in parses:
            print(parse)
if __name__ == '__main__':
    demo()