"""
Classes and interfaces for associating probabilities with tree
structures that represent the internal organization of a text.  The
probabilistic parser module defines ``BottomUpProbabilisticChartParser``.

``BottomUpProbabilisticChartParser`` is an abstract class that implements
a bottom-up chart parser for ``PCFG`` grammars.  It maintains a queue of edges,
and adds them to the chart one at a time.  The ordering of this queue
is based on the probabilities associated with the edges, allowing the
parser to expand more likely edges before less likely ones.  Each
subclass implements a different queue ordering, producing different
search strategies.  Currently the following subclasses are defined:

  - ``InsideChartParser`` searches edges in decreasing order of
    their trees' inside probabilities.
  - ``RandomChartParser`` searches edges in random order.
  - ``LongestChartParser`` searches edges in decreasing order of their
    location's length.

The ``BottomUpProbabilisticChartParser`` constructor has an optional
argument beam_size.  If non-zero, this controls the size of the beam
(aka the edge queue).  This option is most useful with InsideChartParser.
"""
import random
from functools import reduce
from nltk.grammar import PCFG, Nonterminal
from nltk.parse.api import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, LeafEdge, TreeEdge
from nltk.tree import ProbabilisticTree, Tree

class ProbabilisticLeafEdge(LeafEdge):

    def prob(self):
        if False:
            i = 10
            return i + 15
        return 1.0

class ProbabilisticTreeEdge(TreeEdge):

    def __init__(self, prob, *args, **kwargs):
        if False:
            print('Hello World!')
        TreeEdge.__init__(self, *args, **kwargs)
        self._prob = prob
        self._comparison_key = (self._comparison_key, prob)

    def prob(self):
        if False:
            print('Hello World!')
        return self._prob

    @staticmethod
    def from_production(production, index, p):
        if False:
            for i in range(10):
                print('nop')
        return ProbabilisticTreeEdge(p, (index, index), production.lhs(), production.rhs(), 0)

class ProbabilisticBottomUpInitRule(AbstractChartRule):
    NUM_EDGES = 0

    def apply(self, chart, grammar):
        if False:
            print('Hello World!')
        for index in range(chart.num_leaves()):
            new_edge = ProbabilisticLeafEdge(chart.leaf(index), index)
            if chart.insert(new_edge, ()):
                yield new_edge

class ProbabilisticBottomUpPredictRule(AbstractChartRule):
    NUM_EDGES = 1

    def apply(self, chart, grammar, edge):
        if False:
            for i in range(10):
                print('nop')
        if edge.is_incomplete():
            return
        for prod in grammar.productions():
            if edge.lhs() == prod.rhs()[0]:
                new_edge = ProbabilisticTreeEdge.from_production(prod, edge.start(), prod.prob())
                if chart.insert(new_edge, ()):
                    yield new_edge

class ProbabilisticFundamentalRule(AbstractChartRule):
    NUM_EDGES = 2

    def apply(self, chart, grammar, left_edge, right_edge):
        if False:
            i = 10
            return i + 15
        if not (left_edge.end() == right_edge.start() and left_edge.nextsym() == right_edge.lhs() and left_edge.is_incomplete() and right_edge.is_complete()):
            return
        p = left_edge.prob() * right_edge.prob()
        new_edge = ProbabilisticTreeEdge(p, span=(left_edge.start(), right_edge.end()), lhs=left_edge.lhs(), rhs=left_edge.rhs(), dot=left_edge.dot() + 1)
        changed_chart = False
        for cpl1 in chart.child_pointer_lists(left_edge):
            if chart.insert(new_edge, cpl1 + (right_edge,)):
                changed_chart = True
        if changed_chart:
            yield new_edge

class SingleEdgeProbabilisticFundamentalRule(AbstractChartRule):
    NUM_EDGES = 1
    _fundamental_rule = ProbabilisticFundamentalRule()

    def apply(self, chart, grammar, edge1):
        if False:
            for i in range(10):
                print('nop')
        fr = self._fundamental_rule
        if edge1.is_incomplete():
            for edge2 in chart.select(start=edge1.end(), is_complete=True, lhs=edge1.nextsym()):
                yield from fr.apply(chart, grammar, edge1, edge2)
        else:
            for edge2 in chart.select(end=edge1.start(), is_complete=False, nextsym=edge1.lhs()):
                yield from fr.apply(chart, grammar, edge2, edge1)

    def __str__(self):
        if False:
            print('Hello World!')
        return 'Fundamental Rule'

class BottomUpProbabilisticChartParser(ParserI):
    """
    An abstract bottom-up parser for ``PCFG`` grammars that uses a ``Chart`` to
    record partial results.  ``BottomUpProbabilisticChartParser`` maintains
    a queue of edges that can be added to the chart.  This queue is
    initialized with edges for each token in the text that is being
    parsed.  ``BottomUpProbabilisticChartParser`` inserts these edges into
    the chart one at a time, starting with the most likely edges, and
    proceeding to less likely edges.  For each edge that is added to
    the chart, it may become possible to insert additional edges into
    the chart; these are added to the queue.  This process continues
    until enough complete parses have been generated, or until the
    queue is empty.

    The sorting order for the queue is not specified by
    ``BottomUpProbabilisticChartParser``.  Different sorting orders will
    result in different search strategies.  The sorting order for the
    queue is defined by the method ``sort_queue``; subclasses are required
    to provide a definition for this method.

    :type _grammar: PCFG
    :ivar _grammar: The grammar used to parse sentences.
    :type _trace: int
    :ivar _trace: The level of tracing output that should be generated
        when parsing a text.
    """

    def __init__(self, grammar, beam_size=0, trace=0):
        if False:
            return 10
        "\n        Create a new ``BottomUpProbabilisticChartParser``, that uses\n        ``grammar`` to parse texts.\n\n        :type grammar: PCFG\n        :param grammar: The grammar used to parse texts.\n        :type beam_size: int\n        :param beam_size: The maximum length for the parser's edge queue.\n        :type trace: int\n        :param trace: The level of tracing that should be used when\n            parsing a text.  ``0`` will generate no tracing output;\n            and higher numbers will produce more verbose tracing\n            output.\n        "
        if not isinstance(grammar, PCFG):
            raise ValueError('The grammar must be probabilistic PCFG')
        self._grammar = grammar
        self.beam_size = beam_size
        self._trace = trace

    def grammar(self):
        if False:
            return 10
        return self._grammar

    def trace(self, trace=2):
        if False:
            while True:
                i = 10
        '\n        Set the level of tracing output that should be generated when\n        parsing a text.\n\n        :type trace: int\n        :param trace: The trace level.  A trace level of ``0`` will\n            generate no tracing output; and higher trace levels will\n            produce more verbose tracing output.\n        :rtype: None\n        '
        self._trace = trace

    def parse(self, tokens):
        if False:
            i = 10
            return i + 15
        self._grammar.check_coverage(tokens)
        chart = Chart(list(tokens))
        grammar = self._grammar
        bu_init = ProbabilisticBottomUpInitRule()
        bu = ProbabilisticBottomUpPredictRule()
        fr = SingleEdgeProbabilisticFundamentalRule()
        queue = []
        for edge in bu_init.apply(chart, grammar):
            if self._trace > 1:
                print('  %-50s [%s]' % (chart.pretty_format_edge(edge, width=2), edge.prob()))
            queue.append(edge)
        while len(queue) > 0:
            self.sort_queue(queue, chart)
            if self.beam_size:
                self._prune(queue, chart)
            edge = queue.pop()
            if self._trace > 0:
                print('  %-50s [%s]' % (chart.pretty_format_edge(edge, width=2), edge.prob()))
            queue.extend(bu.apply(chart, grammar, edge))
            queue.extend(fr.apply(chart, grammar, edge))
        parses = list(chart.parses(grammar.start(), ProbabilisticTree))
        prod_probs = {}
        for prod in grammar.productions():
            prod_probs[prod.lhs(), prod.rhs()] = prod.prob()
        for parse in parses:
            self._setprob(parse, prod_probs)
        parses.sort(reverse=True, key=lambda tree: tree.prob())
        return iter(parses)

    def _setprob(self, tree, prod_probs):
        if False:
            for i in range(10):
                print('nop')
        if tree.prob() is not None:
            return
        lhs = Nonterminal(tree.label())
        rhs = []
        for child in tree:
            if isinstance(child, Tree):
                rhs.append(Nonterminal(child.label()))
            else:
                rhs.append(child)
        prob = prod_probs[lhs, tuple(rhs)]
        for child in tree:
            if isinstance(child, Tree):
                self._setprob(child, prod_probs)
                prob *= child.prob()
        tree.set_prob(prob)

    def sort_queue(self, queue, chart):
        if False:
            i = 10
            return i + 15
        '\n        Sort the given queue of ``Edge`` objects, placing the edge that should\n        be tried first at the beginning of the queue.  This method\n        will be called after each ``Edge`` is added to the queue.\n\n        :param queue: The queue of ``Edge`` objects to sort.  Each edge in\n            this queue is an edge that could be added to the chart by\n            the fundamental rule; but that has not yet been added.\n        :type queue: list(Edge)\n        :param chart: The chart being used to parse the text.  This\n            chart can be used to provide extra information for sorting\n            the queue.\n        :type chart: Chart\n        :rtype: None\n        '
        raise NotImplementedError()

    def _prune(self, queue, chart):
        if False:
            print('Hello World!')
        'Discard items in the queue if the queue is longer than the beam.'
        if len(queue) > self.beam_size:
            split = len(queue) - self.beam_size
            if self._trace > 2:
                for edge in queue[:split]:
                    print('  %-50s [DISCARDED]' % chart.pretty_format_edge(edge, 2))
            del queue[:split]

class InsideChartParser(BottomUpProbabilisticChartParser):
    """
    A bottom-up parser for ``PCFG`` grammars that tries edges in descending
    order of the inside probabilities of their trees.  The "inside
    probability" of a tree is simply the
    probability of the entire tree, ignoring its context.  In
    particular, the inside probability of a tree generated by
    production *p* with children *c[1], c[2], ..., c[n]* is
    *P(p)P(c[1])P(c[2])...P(c[n])*; and the inside
    probability of a token is 1 if it is present in the text, and 0 if
    it is absent.

    This sorting order results in a type of lowest-cost-first search
    strategy.
    """

    def sort_queue(self, queue, chart):
        if False:
            print('Hello World!')
        "\n        Sort the given queue of edges, in descending order of the\n        inside probabilities of the edges' trees.\n\n        :param queue: The queue of ``Edge`` objects to sort.  Each edge in\n            this queue is an edge that could be added to the chart by\n            the fundamental rule; but that has not yet been added.\n        :type queue: list(Edge)\n        :param chart: The chart being used to parse the text.  This\n            chart can be used to provide extra information for sorting\n            the queue.\n        :type chart: Chart\n        :rtype: None\n        "
        queue.sort(key=lambda edge: edge.prob())

class RandomChartParser(BottomUpProbabilisticChartParser):
    """
    A bottom-up parser for ``PCFG`` grammars that tries edges in random order.
    This sorting order results in a random search strategy.
    """

    def sort_queue(self, queue, chart):
        if False:
            while True:
                i = 10
        i = random.randint(0, len(queue) - 1)
        (queue[-1], queue[i]) = (queue[i], queue[-1])

class UnsortedChartParser(BottomUpProbabilisticChartParser):
    """
    A bottom-up parser for ``PCFG`` grammars that tries edges in whatever order.
    """

    def sort_queue(self, queue, chart):
        if False:
            while True:
                i = 10
        return

class LongestChartParser(BottomUpProbabilisticChartParser):
    """
    A bottom-up parser for ``PCFG`` grammars that tries longer edges before
    shorter ones.  This sorting order results in a type of best-first
    search strategy.
    """

    def sort_queue(self, queue, chart):
        if False:
            while True:
                i = 10
        queue.sort(key=lambda edge: edge.length())

def demo(choice=None, draw_parses=None, print_parses=None):
    if False:
        i = 10
        return i + 15
    '\n    A demonstration of the probabilistic parsers.  The user is\n    prompted to select which demo to run, and how many parses should\n    be found; and then each parser is run on the same demo, and a\n    summary of the results are displayed.\n    '
    import sys
    import time
    from nltk import tokenize
    from nltk.parse import pchart
    toy_pcfg1 = PCFG.fromstring("\n    S -> NP VP [1.0]\n    NP -> Det N [0.5] | NP PP [0.25] | 'John' [0.1] | 'I' [0.15]\n    Det -> 'the' [0.8] | 'my' [0.2]\n    N -> 'man' [0.5] | 'telescope' [0.5]\n    VP -> VP PP [0.1] | V NP [0.7] | V [0.2]\n    V -> 'ate' [0.35] | 'saw' [0.65]\n    PP -> P NP [1.0]\n    P -> 'with' [0.61] | 'under' [0.39]\n    ")
    toy_pcfg2 = PCFG.fromstring("\n    S    -> NP VP         [1.0]\n    VP   -> V NP          [.59]\n    VP   -> V             [.40]\n    VP   -> VP PP         [.01]\n    NP   -> Det N         [.41]\n    NP   -> Name          [.28]\n    NP   -> NP PP         [.31]\n    PP   -> P NP          [1.0]\n    V    -> 'saw'         [.21]\n    V    -> 'ate'         [.51]\n    V    -> 'ran'         [.28]\n    N    -> 'boy'         [.11]\n    N    -> 'cookie'      [.12]\n    N    -> 'table'       [.13]\n    N    -> 'telescope'   [.14]\n    N    -> 'hill'        [.5]\n    Name -> 'Jack'        [.52]\n    Name -> 'Bob'         [.48]\n    P    -> 'with'        [.61]\n    P    -> 'under'       [.39]\n    Det  -> 'the'         [.41]\n    Det  -> 'a'           [.31]\n    Det  -> 'my'          [.28]\n    ")
    demos = [('I saw John with my telescope', toy_pcfg1), ('the boy saw Jack with Bob under the table with a telescope', toy_pcfg2)]
    if choice is None:
        print()
        for i in range(len(demos)):
            print(f'{i + 1:>3}: {demos[i][0]}')
            print('     %r' % demos[i][1])
            print()
        print('Which demo (%d-%d)? ' % (1, len(demos)), end=' ')
        choice = int(sys.stdin.readline().strip()) - 1
    try:
        (sent, grammar) = demos[choice]
    except:
        print('Bad sentence number')
        return
    tokens = sent.split()
    parsers = [pchart.InsideChartParser(grammar), pchart.RandomChartParser(grammar), pchart.UnsortedChartParser(grammar), pchart.LongestChartParser(grammar), pchart.InsideChartParser(grammar, beam_size=len(tokens) + 1)]
    times = []
    average_p = []
    num_parses = []
    all_parses = {}
    for parser in parsers:
        print(f'\ns: {sent}\nparser: {parser}\ngrammar: {grammar}')
        parser.trace(3)
        t = time.time()
        parses = list(parser.parse(tokens))
        times.append(time.time() - t)
        p = reduce(lambda a, b: a + b.prob(), parses, 0) / len(parses) if parses else 0
        average_p.append(p)
        num_parses.append(len(parses))
        for p in parses:
            all_parses[p.freeze()] = 1
    print()
    print('       Parser      Beam | Time (secs)   # Parses   Average P(parse)')
    print('------------------------+------------------------------------------')
    for i in range(len(parsers)):
        print('%18s %4d |%11.4f%11d%19.14f' % (parsers[i].__class__.__name__, parsers[i].beam_size, times[i], num_parses[i], average_p[i]))
    parses = all_parses.keys()
    if parses:
        p = reduce(lambda a, b: a + b.prob(), parses, 0) / len(parses)
    else:
        p = 0
    print('------------------------+------------------------------------------')
    print('%18s      |%11s%11d%19.14f' % ('(All Parses)', 'n/a', len(parses), p))
    if draw_parses is None:
        print()
        print('Draw parses (y/n)? ', end=' ')
        draw_parses = sys.stdin.readline().strip().lower().startswith('y')
    if draw_parses:
        from nltk.draw.tree import draw_trees
        print('  please wait...')
        draw_trees(*parses)
    if print_parses is None:
        print()
        print('Print parses (y/n)? ', end=' ')
        print_parses = sys.stdin.readline().strip().lower().startswith('y')
    if print_parses:
        for parse in parses:
            print(parse)
if __name__ == '__main__':
    demo()