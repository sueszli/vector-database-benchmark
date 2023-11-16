"""
Extension of chart parsing implementation to handle grammars with
feature structures as nodes.
"""
from time import perf_counter
from nltk.featstruct import TYPE, FeatStruct, find_variables, unify
from nltk.grammar import CFG, FeatStructNonterminal, Nonterminal, Production, is_nonterminal, is_terminal
from nltk.parse.chart import BottomUpPredictCombineRule, BottomUpPredictRule, CachedTopDownPredictRule, Chart, ChartParser, EdgeI, EmptyPredictRule, FundamentalRule, LeafInitRule, SingleEdgeFundamentalRule, TopDownInitRule, TreeEdge
from nltk.sem import logic
from nltk.tree import Tree

class FeatureTreeEdge(TreeEdge):
    """
    A specialized tree edge that allows shared variable bindings
    between nonterminals on the left-hand side and right-hand side.

    Each ``FeatureTreeEdge`` contains a set of ``bindings``, i.e., a
    dictionary mapping from variables to values.  If the edge is not
    complete, then these bindings are simply stored.  However, if the
    edge is complete, then the constructor applies these bindings to
    every nonterminal in the edge whose symbol implements the
    interface ``SubstituteBindingsI``.
    """

    def __init__(self, span, lhs, rhs, dot=0, bindings=None):
        if False:
            print('Hello World!')
        '\n        Construct a new edge.  If the edge is incomplete (i.e., if\n        ``dot<len(rhs)``), then store the bindings as-is.  If the edge\n        is complete (i.e., if ``dot==len(rhs)``), then apply the\n        bindings to all nonterminals in ``lhs`` and ``rhs``, and then\n        clear the bindings.  See ``TreeEdge`` for a description of\n        the other arguments.\n        '
        if bindings is None:
            bindings = {}
        if dot == len(rhs) and bindings:
            lhs = self._bind(lhs, bindings)
            rhs = [self._bind(elt, bindings) for elt in rhs]
            bindings = {}
        TreeEdge.__init__(self, span, lhs, rhs, dot)
        self._bindings = bindings
        self._comparison_key = (self._comparison_key, tuple(sorted(bindings.items())))

    @staticmethod
    def from_production(production, index):
        if False:
            while True:
                i = 10
        "\n        :return: A new ``TreeEdge`` formed from the given production.\n            The new edge's left-hand side and right-hand side will\n            be taken from ``production``; its span will be\n            ``(index,index)``; and its dot position will be ``0``.\n        :rtype: TreeEdge\n        "
        return FeatureTreeEdge(span=(index, index), lhs=production.lhs(), rhs=production.rhs(), dot=0)

    def move_dot_forward(self, new_end, bindings=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        :return: A new ``FeatureTreeEdge`` formed from this edge.\n            The new edge's dot position is increased by ``1``,\n            and its end index will be replaced by ``new_end``.\n        :rtype: FeatureTreeEdge\n        :param new_end: The new end index.\n        :type new_end: int\n        :param bindings: Bindings for the new edge.\n        :type bindings: dict\n        "
        return FeatureTreeEdge(span=(self._span[0], new_end), lhs=self._lhs, rhs=self._rhs, dot=self._dot + 1, bindings=bindings)

    def _bind(self, nt, bindings):
        if False:
            while True:
                i = 10
        if not isinstance(nt, FeatStructNonterminal):
            return nt
        return nt.substitute_bindings(bindings)

    def next_with_bindings(self):
        if False:
            i = 10
            return i + 15
        return self._bind(self.nextsym(), self._bindings)

    def bindings(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a copy of this edge's bindings dictionary.\n        "
        return self._bindings.copy()

    def variables(self):
        if False:
            i = 10
            return i + 15
        '\n        :return: The set of variables used by this edge.\n        :rtype: set(Variable)\n        '
        return find_variables([self._lhs] + list(self._rhs) + list(self._bindings.keys()) + list(self._bindings.values()), fs_class=FeatStruct)

    def __str__(self):
        if False:
            while True:
                i = 10
        if self.is_complete():
            return super().__str__()
        else:
            bindings = '{%s}' % ', '.join(('%s: %r' % item for item in sorted(self._bindings.items())))
            return f'{super().__str__()} {bindings}'

class FeatureChart(Chart):
    """
    A Chart for feature grammars.
    :see: ``Chart`` for more information.
    """

    def select(self, **restrictions):
        if False:
            print('Hello World!')
        '\n        Returns an iterator over the edges in this chart.\n        See ``Chart.select`` for more information about the\n        ``restrictions`` on the edges.\n        '
        if restrictions == {}:
            return iter(self._edges)
        restr_keys = sorted(restrictions.keys())
        restr_keys = tuple(restr_keys)
        if restr_keys not in self._indexes:
            self._add_index(restr_keys)
        vals = tuple((self._get_type_if_possible(restrictions[key]) for key in restr_keys))
        return iter(self._indexes[restr_keys].get(vals, []))

    def _add_index(self, restr_keys):
        if False:
            return 10
        '\n        A helper function for ``select``, which creates a new index for\n        a given set of attributes (aka restriction keys).\n        '
        for key in restr_keys:
            if not hasattr(EdgeI, key):
                raise ValueError('Bad restriction: %s' % key)
        index = self._indexes[restr_keys] = {}
        for edge in self._edges:
            vals = tuple((self._get_type_if_possible(getattr(edge, key)()) for key in restr_keys))
            index.setdefault(vals, []).append(edge)

    def _register_with_indexes(self, edge):
        if False:
            i = 10
            return i + 15
        '\n        A helper function for ``insert``, which registers the new\n        edge with all existing indexes.\n        '
        for (restr_keys, index) in self._indexes.items():
            vals = tuple((self._get_type_if_possible(getattr(edge, key)()) for key in restr_keys))
            index.setdefault(vals, []).append(edge)

    def _get_type_if_possible(self, item):
        if False:
            while True:
                i = 10
        '\n        Helper function which returns the ``TYPE`` feature of the ``item``,\n        if it exists, otherwise it returns the ``item`` itself\n        '
        if isinstance(item, dict) and TYPE in item:
            return item[TYPE]
        else:
            return item

    def parses(self, start, tree_class=Tree):
        if False:
            return 10
        for edge in self.select(start=0, end=self._num_leaves):
            if isinstance(edge, FeatureTreeEdge) and edge.lhs()[TYPE] == start[TYPE] and unify(edge.lhs(), start, rename_vars=True):
                yield from self.trees(edge, complete=True, tree_class=tree_class)

class FeatureFundamentalRule(FundamentalRule):
    """
    A specialized version of the fundamental rule that operates on
    nonterminals whose symbols are ``FeatStructNonterminal``s.  Rather
    than simply comparing the nonterminals for equality, they are
    unified.  Variable bindings from these unifications are collected
    and stored in the chart using a ``FeatureTreeEdge``.  When a
    complete edge is generated, these bindings are applied to all
    nonterminals in the edge.

    The fundamental rule states that:

    - ``[A -> alpha \\* B1 beta][i:j]``
    - ``[B2 -> gamma \\*][j:k]``

    licenses the edge:

    - ``[A -> alpha B3 \\* beta][i:j]``

    assuming that B1 and B2 can be unified to generate B3.
    """

    def apply(self, chart, grammar, left_edge, right_edge):
        if False:
            i = 10
            return i + 15
        if not (left_edge.end() == right_edge.start() and left_edge.is_incomplete() and right_edge.is_complete() and isinstance(left_edge, FeatureTreeEdge)):
            return
        found = right_edge.lhs()
        nextsym = left_edge.nextsym()
        if isinstance(right_edge, FeatureTreeEdge):
            if not is_nonterminal(nextsym):
                return
            if left_edge.nextsym()[TYPE] != right_edge.lhs()[TYPE]:
                return
            bindings = left_edge.bindings()
            found = found.rename_variables(used_vars=left_edge.variables())
            result = unify(nextsym, found, bindings, rename_vars=False)
            if result is None:
                return
        else:
            if nextsym != found:
                return
            bindings = left_edge.bindings()
        new_edge = left_edge.move_dot_forward(right_edge.end(), bindings)
        if chart.insert_with_backpointer(new_edge, left_edge, right_edge):
            yield new_edge

class FeatureSingleEdgeFundamentalRule(SingleEdgeFundamentalRule):
    """
    A specialized version of the completer / single edge fundamental rule
    that operates on nonterminals whose symbols are ``FeatStructNonterminal``.
    Rather than simply comparing the nonterminals for equality, they are
    unified.
    """
    _fundamental_rule = FeatureFundamentalRule()

    def _apply_complete(self, chart, grammar, right_edge):
        if False:
            print('Hello World!')
        fr = self._fundamental_rule
        for left_edge in chart.select(end=right_edge.start(), is_complete=False, nextsym=right_edge.lhs()):
            yield from fr.apply(chart, grammar, left_edge, right_edge)

    def _apply_incomplete(self, chart, grammar, left_edge):
        if False:
            print('Hello World!')
        fr = self._fundamental_rule
        for right_edge in chart.select(start=left_edge.end(), is_complete=True, lhs=left_edge.nextsym()):
            yield from fr.apply(chart, grammar, left_edge, right_edge)

class FeatureTopDownInitRule(TopDownInitRule):

    def apply(self, chart, grammar):
        if False:
            i = 10
            return i + 15
        for prod in grammar.productions(lhs=grammar.start()):
            new_edge = FeatureTreeEdge.from_production(prod, 0)
            if chart.insert(new_edge, ()):
                yield new_edge

class FeatureTopDownPredictRule(CachedTopDownPredictRule):
    """
    A specialized version of the (cached) top down predict rule that operates
    on nonterminals whose symbols are ``FeatStructNonterminal``.  Rather
    than simply comparing the nonterminals for equality, they are
    unified.

    The top down expand rule states that:

    - ``[A -> alpha \\* B1 beta][i:j]``

    licenses the edge:

    - ``[B2 -> \\* gamma][j:j]``

    for each grammar production ``B2 -> gamma``, assuming that B1
    and B2 can be unified.
    """

    def apply(self, chart, grammar, edge):
        if False:
            print('Hello World!')
        if edge.is_complete():
            return
        (nextsym, index) = (edge.nextsym(), edge.end())
        if not is_nonterminal(nextsym):
            return
        nextsym_with_bindings = edge.next_with_bindings()
        done = self._done.get((nextsym_with_bindings, index), (None, None))
        if done[0] is chart and done[1] is grammar:
            return
        for prod in grammar.productions(lhs=nextsym):
            if prod.rhs():
                first = prod.rhs()[0]
                if is_terminal(first):
                    if index >= chart.num_leaves():
                        continue
                    if first != chart.leaf(index):
                        continue
            if unify(prod.lhs(), nextsym_with_bindings, rename_vars=True):
                new_edge = FeatureTreeEdge.from_production(prod, edge.end())
                if chart.insert(new_edge, ()):
                    yield new_edge
        self._done[nextsym_with_bindings, index] = (chart, grammar)

class FeatureBottomUpPredictRule(BottomUpPredictRule):

    def apply(self, chart, grammar, edge):
        if False:
            print('Hello World!')
        if edge.is_incomplete():
            return
        for prod in grammar.productions(rhs=edge.lhs()):
            if isinstance(edge, FeatureTreeEdge):
                _next = prod.rhs()[0]
                if not is_nonterminal(_next):
                    continue
            new_edge = FeatureTreeEdge.from_production(prod, edge.start())
            if chart.insert(new_edge, ()):
                yield new_edge

class FeatureBottomUpPredictCombineRule(BottomUpPredictCombineRule):

    def apply(self, chart, grammar, edge):
        if False:
            for i in range(10):
                print('nop')
        if edge.is_incomplete():
            return
        found = edge.lhs()
        for prod in grammar.productions(rhs=found):
            bindings = {}
            if isinstance(edge, FeatureTreeEdge):
                _next = prod.rhs()[0]
                if not is_nonterminal(_next):
                    continue
                used_vars = find_variables((prod.lhs(),) + prod.rhs(), fs_class=FeatStruct)
                found = found.rename_variables(used_vars=used_vars)
                result = unify(_next, found, bindings, rename_vars=False)
                if result is None:
                    continue
            new_edge = FeatureTreeEdge.from_production(prod, edge.start()).move_dot_forward(edge.end(), bindings)
            if chart.insert(new_edge, (edge,)):
                yield new_edge

class FeatureEmptyPredictRule(EmptyPredictRule):

    def apply(self, chart, grammar):
        if False:
            for i in range(10):
                print('nop')
        for prod in grammar.productions(empty=True):
            for index in range(chart.num_leaves() + 1):
                new_edge = FeatureTreeEdge.from_production(prod, index)
                if chart.insert(new_edge, ()):
                    yield new_edge
TD_FEATURE_STRATEGY = [LeafInitRule(), FeatureTopDownInitRule(), FeatureTopDownPredictRule(), FeatureSingleEdgeFundamentalRule()]
BU_FEATURE_STRATEGY = [LeafInitRule(), FeatureEmptyPredictRule(), FeatureBottomUpPredictRule(), FeatureSingleEdgeFundamentalRule()]
BU_LC_FEATURE_STRATEGY = [LeafInitRule(), FeatureEmptyPredictRule(), FeatureBottomUpPredictCombineRule(), FeatureSingleEdgeFundamentalRule()]

class FeatureChartParser(ChartParser):

    def __init__(self, grammar, strategy=BU_LC_FEATURE_STRATEGY, trace_chart_width=20, chart_class=FeatureChart, **parser_args):
        if False:
            for i in range(10):
                print('nop')
        ChartParser.__init__(self, grammar, strategy=strategy, trace_chart_width=trace_chart_width, chart_class=chart_class, **parser_args)

class FeatureTopDownChartParser(FeatureChartParser):

    def __init__(self, grammar, **parser_args):
        if False:
            for i in range(10):
                print('nop')
        FeatureChartParser.__init__(self, grammar, TD_FEATURE_STRATEGY, **parser_args)

class FeatureBottomUpChartParser(FeatureChartParser):

    def __init__(self, grammar, **parser_args):
        if False:
            while True:
                i = 10
        FeatureChartParser.__init__(self, grammar, BU_FEATURE_STRATEGY, **parser_args)

class FeatureBottomUpLeftCornerChartParser(FeatureChartParser):

    def __init__(self, grammar, **parser_args):
        if False:
            while True:
                i = 10
        FeatureChartParser.__init__(self, grammar, BU_LC_FEATURE_STRATEGY, **parser_args)

class InstantiateVarsChart(FeatureChart):
    """
    A specialized chart that 'instantiates' variables whose names
    start with '@', by replacing them with unique new variables.
    In particular, whenever a complete edge is added to the chart, any
    variables in the edge's ``lhs`` whose names start with '@' will be
    replaced by unique new ``Variable``.
    """

    def __init__(self, tokens):
        if False:
            return 10
        FeatureChart.__init__(self, tokens)

    def initialize(self):
        if False:
            i = 10
            return i + 15
        self._instantiated = set()
        FeatureChart.initialize(self)

    def insert(self, edge, child_pointer_list):
        if False:
            return 10
        if edge in self._instantiated:
            return False
        self.instantiate_edge(edge)
        return FeatureChart.insert(self, edge, child_pointer_list)

    def instantiate_edge(self, edge):
        if False:
            i = 10
            return i + 15
        "\n        If the edge is a ``FeatureTreeEdge``, and it is complete,\n        then instantiate all variables whose names start with '@',\n        by replacing them with unique new variables.\n\n        Note that instantiation is done in-place, since the\n        parsing algorithms might already hold a reference to\n        the edge for future use.\n        "
        if not isinstance(edge, FeatureTreeEdge):
            return
        if not edge.is_complete():
            return
        if edge in self._edge_to_cpls:
            return
        inst_vars = self.inst_vars(edge)
        if not inst_vars:
            return
        self._instantiated.add(edge)
        edge._lhs = edge.lhs().substitute_bindings(inst_vars)

    def inst_vars(self, edge):
        if False:
            print('Hello World!')
        return {var: logic.unique_variable() for var in edge.lhs().variables() if var.name.startswith('@')}

def demo_grammar():
    if False:
        for i in range(10):
            print('nop')
    from nltk.grammar import FeatureGrammar
    return FeatureGrammar.fromstring('\nS  -> NP VP\nPP -> Prep NP\nNP -> NP PP\nVP -> VP PP\nVP -> Verb NP\nVP -> Verb\nNP -> Det[pl=?x] Noun[pl=?x]\nNP -> "John"\nNP -> "I"\nDet -> "the"\nDet -> "my"\nDet[-pl] -> "a"\nNoun[-pl] -> "dog"\nNoun[-pl] -> "cookie"\nVerb -> "ate"\nVerb -> "saw"\nPrep -> "with"\nPrep -> "under"\n')

def demo(print_times=True, print_grammar=True, print_trees=True, print_sentence=True, trace=1, parser=FeatureChartParser, sent='I saw John with a dog with my cookie'):
    if False:
        print('Hello World!')
    import sys
    import time
    print()
    grammar = demo_grammar()
    if print_grammar:
        print(grammar)
        print()
    print('*', parser.__name__)
    if print_sentence:
        print('Sentence:', sent)
    tokens = sent.split()
    t = perf_counter()
    cp = parser(grammar, trace=trace)
    chart = cp.chart_parse(tokens)
    trees = list(chart.parses(grammar.start()))
    if print_times:
        print('Time: %s' % (perf_counter() - t))
    if print_trees:
        for tree in trees:
            print(tree)
    else:
        print('Nr trees:', len(trees))

def run_profile():
    if False:
        i = 10
        return i + 15
    import profile
    profile.run('for i in range(1): demo()', '/tmp/profile.out')
    import pstats
    p = pstats.Stats('/tmp/profile.out')
    p.strip_dirs().sort_stats('time', 'cum').print_stats(60)
    p.strip_dirs().sort_stats('cum', 'time').print_stats(60)
if __name__ == '__main__':
    from nltk.data import load
    demo()
    print()
    grammar = load('grammars/book_grammars/feat0.fcfg')
    cp = FeatureChartParser(grammar, trace=2)
    sent = 'Kim likes children'
    tokens = sent.split()
    trees = cp.parse(tokens)
    for tree in trees:
        print(tree)