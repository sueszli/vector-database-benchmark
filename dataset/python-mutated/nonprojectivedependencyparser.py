import logging
import math
from nltk.parse.dependencygraph import DependencyGraph
logger = logging.getLogger(__name__)

class DependencyScorerI:
    """
    A scorer for calculated the weights on the edges of a weighted
    dependency graph.  This is used by a
    ``ProbabilisticNonprojectiveParser`` to initialize the edge
    weights of a ``DependencyGraph``.  While typically this would be done
    by training a binary classifier, any class that can return a
    multidimensional list representation of the edge weights can
    implement this interface.  As such, it has no necessary
    fields.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        if self.__class__ == DependencyScorerI:
            raise TypeError('DependencyScorerI is an abstract interface')

    def train(self, graphs):
        if False:
            i = 10
            return i + 15
        '\n        :type graphs: list(DependencyGraph)\n        :param graphs: A list of dependency graphs to train the scorer.\n            Typically the edges present in the graphs can be used as\n            positive training examples, and the edges not present as negative\n            examples.\n        '
        raise NotImplementedError()

    def score(self, graph):
        if False:
            while True:
                i = 10
        "\n        :type graph: DependencyGraph\n        :param graph: A dependency graph whose set of edges need to be\n            scored.\n        :rtype: A three-dimensional list of numbers.\n        :return: The score is returned in a multidimensional(3) list, such\n            that the outer-dimension refers to the head, and the\n            inner-dimension refers to the dependencies.  For instance,\n            scores[0][1] would reference the list of scores corresponding to\n            arcs from node 0 to node 1.  The node's 'address' field can be used\n            to determine its number identification.\n\n        For further illustration, a score list corresponding to Fig.2 of\n        Keith Hall's 'K-best Spanning Tree Parsing' paper::\n\n              scores = [[[], [5],  [1],  [1]],\n                       [[], [],   [11], [4]],\n                       [[], [10], [],   [5]],\n                       [[], [8],  [8],  []]]\n\n        When used in conjunction with a MaxEntClassifier, each score would\n        correspond to the confidence of a particular edge being classified\n        with the positive training examples.\n        "
        raise NotImplementedError()

class NaiveBayesDependencyScorer(DependencyScorerI):
    """
    A dependency scorer built around a MaxEnt classifier.  In this
    particular class that classifier is a ``NaiveBayesClassifier``.
    It uses head-word, head-tag, child-word, and child-tag features
    for classification.

    >>> from nltk.parse.dependencygraph import DependencyGraph, conll_data2

    >>> graphs = [DependencyGraph(entry) for entry in conll_data2.split('\\n\\n') if entry]
    >>> npp = ProbabilisticNonprojectiveParser()
    >>> npp.train(graphs, NaiveBayesDependencyScorer())
    >>> parses = npp.parse(['Cathy', 'zag', 'hen', 'zwaaien', '.'], ['N', 'V', 'Pron', 'Adj', 'N', 'Punc'])
    >>> len(list(parses))
    1

    """

    def __init__(self):
        if False:
            return 10
        pass

    def train(self, graphs):
        if False:
            print('Hello World!')
        '\n        Trains a ``NaiveBayesClassifier`` using the edges present in\n        graphs list as positive examples, the edges not present as\n        negative examples.  Uses a feature vector of head-word,\n        head-tag, child-word, and child-tag.\n\n        :type graphs: list(DependencyGraph)\n        :param graphs: A list of dependency graphs to train the scorer.\n        '
        from nltk.classify import NaiveBayesClassifier
        labeled_examples = []
        for graph in graphs:
            for head_node in graph.nodes.values():
                for (child_index, child_node) in graph.nodes.items():
                    if child_index in head_node['deps']:
                        label = 'T'
                    else:
                        label = 'F'
                    labeled_examples.append((dict(a=head_node['word'], b=head_node['tag'], c=child_node['word'], d=child_node['tag']), label))
        self.classifier = NaiveBayesClassifier.train(labeled_examples)

    def score(self, graph):
        if False:
            print('Hello World!')
        '\n        Converts the graph into a feature-based representation of\n        each edge, and then assigns a score to each based on the\n        confidence of the classifier in assigning it to the\n        positive label.  Scores are returned in a multidimensional list.\n\n        :type graph: DependencyGraph\n        :param graph: A dependency graph to score.\n        :rtype: 3 dimensional list\n        :return: Edge scores for the graph parameter.\n        '
        edges = []
        for head_node in graph.nodes.values():
            for child_node in graph.nodes.values():
                edges.append(dict(a=head_node['word'], b=head_node['tag'], c=child_node['word'], d=child_node['tag']))
        edge_scores = []
        row = []
        count = 0
        for pdist in self.classifier.prob_classify_many(edges):
            logger.debug('%.4f %.4f', pdist.prob('T'), pdist.prob('F'))
            row.append([math.log(pdist.prob('T') + 1e-11)])
            count += 1
            if count == len(graph.nodes):
                edge_scores.append(row)
                row = []
                count = 0
        return edge_scores

class DemoScorer(DependencyScorerI):

    def train(self, graphs):
        if False:
            return 10
        print('Training...')

    def score(self, graph):
        if False:
            print('Hello World!')
        return [[[], [5], [1], [1]], [[], [], [11], [4]], [[], [10], [], [5]], [[], [8], [8], []]]

class ProbabilisticNonprojectiveParser:
    """A probabilistic non-projective dependency parser.

    Nonprojective dependencies allows for "crossing branches" in the parse tree
    which is necessary for representing particular linguistic phenomena, or even
    typical parses in some languages.  This parser follows the MST parsing
    algorithm, outlined in McDonald(2005), which likens the search for the best
    non-projective parse to finding the maximum spanning tree in a weighted
    directed graph.

    >>> class Scorer(DependencyScorerI):
    ...     def train(self, graphs):
    ...         pass
    ...
    ...     def score(self, graph):
    ...         return [
    ...             [[], [5],  [1],  [1]],
    ...             [[], [],   [11], [4]],
    ...             [[], [10], [],   [5]],
    ...             [[], [8],  [8],  []],
    ...         ]


    >>> npp = ProbabilisticNonprojectiveParser()
    >>> npp.train([], Scorer())

    >>> parses = npp.parse(['v1', 'v2', 'v3'], [None, None, None])
    >>> len(list(parses))
    1

    Rule based example

    >>> from nltk.grammar import DependencyGrammar

    >>> grammar = DependencyGrammar.fromstring('''
    ... 'taught' -> 'play' | 'man'
    ... 'man' -> 'the' | 'in'
    ... 'in' -> 'corner'
    ... 'corner' -> 'the'
    ... 'play' -> 'golf' | 'dachshund' | 'to'
    ... 'dachshund' -> 'his'
    ... ''')

    >>> ndp = NonprojectiveDependencyParser(grammar)
    >>> parses = ndp.parse(['the', 'man', 'in', 'the', 'corner', 'taught', 'his', 'dachshund', 'to', 'play', 'golf'])
    >>> len(list(parses))
    4

    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new non-projective parser.\n        '
        logging.debug('initializing prob. nonprojective...')

    def train(self, graphs, dependency_scorer):
        if False:
            print('Hello World!')
        "\n        Trains a ``DependencyScorerI`` from a set of ``DependencyGraph`` objects,\n        and establishes this as the parser's scorer.  This is used to\n        initialize the scores on a ``DependencyGraph`` during the parsing\n        procedure.\n\n        :type graphs: list(DependencyGraph)\n        :param graphs: A list of dependency graphs to train the scorer.\n        :type dependency_scorer: DependencyScorerI\n        :param dependency_scorer: A scorer which implements the\n            ``DependencyScorerI`` interface.\n        "
        self._scorer = dependency_scorer
        self._scorer.train(graphs)

    def initialize_edge_scores(self, graph):
        if False:
            print('Hello World!')
        "\n        Assigns a score to every edge in the ``DependencyGraph`` graph.\n        These scores are generated via the parser's scorer which\n        was assigned during the training process.\n\n        :type graph: DependencyGraph\n        :param graph: A dependency graph to assign scores to.\n        "
        self.scores = self._scorer.score(graph)

    def collapse_nodes(self, new_node, cycle_path, g_graph, b_graph, c_graph):
        if False:
            while True:
                i = 10
        '\n        Takes a list of nodes that have been identified to belong to a cycle,\n        and collapses them into on larger node.  The arcs of all nodes in\n        the graph must be updated to account for this.\n\n        :type new_node: Node.\n        :param new_node: A Node (Dictionary) to collapse the cycle nodes into.\n        :type cycle_path: A list of integers.\n        :param cycle_path: A list of node addresses, each of which is in the cycle.\n        :type g_graph, b_graph, c_graph: DependencyGraph\n        :param g_graph, b_graph, c_graph: Graphs which need to be updated.\n        '
        logger.debug('Collapsing nodes...')
        for cycle_node_index in cycle_path:
            g_graph.remove_by_address(cycle_node_index)
        g_graph.add_node(new_node)
        g_graph.redirect_arcs(cycle_path, new_node['address'])

    def update_edge_scores(self, new_node, cycle_path):
        if False:
            i = 10
            return i + 15
        '\n        Updates the edge scores to reflect a collapse operation into\n        new_node.\n\n        :type new_node: A Node.\n        :param new_node: The node which cycle nodes are collapsed into.\n        :type cycle_path: A list of integers.\n        :param cycle_path: A list of node addresses that belong to the cycle.\n        '
        logger.debug('cycle %s', cycle_path)
        cycle_path = self.compute_original_indexes(cycle_path)
        logger.debug('old cycle %s', cycle_path)
        logger.debug('Prior to update: %s', self.scores)
        for (i, row) in enumerate(self.scores):
            for (j, column) in enumerate(self.scores[i]):
                logger.debug(self.scores[i][j])
                if j in cycle_path and i not in cycle_path and self.scores[i][j]:
                    subtract_val = self.compute_max_subtract_score(j, cycle_path)
                    logger.debug('%s - %s', self.scores[i][j], subtract_val)
                    new_vals = []
                    for cur_val in self.scores[i][j]:
                        new_vals.append(cur_val - subtract_val)
                    self.scores[i][j] = new_vals
        for (i, row) in enumerate(self.scores):
            for (j, cell) in enumerate(self.scores[i]):
                if i in cycle_path and j in cycle_path:
                    self.scores[i][j] = []
        logger.debug('After update: %s', self.scores)

    def compute_original_indexes(self, new_indexes):
        if False:
            for i in range(10):
                print('nop')
        "\n        As nodes are collapsed into others, they are replaced\n        by the new node in the graph, but it's still necessary\n        to keep track of what these original nodes were.  This\n        takes a list of node addresses and replaces any collapsed\n        node addresses with their original addresses.\n\n        :type new_indexes: A list of integers.\n        :param new_indexes: A list of node addresses to check for\n            subsumed nodes.\n        "
        swapped = True
        while swapped:
            originals = []
            swapped = False
            for new_index in new_indexes:
                if new_index in self.inner_nodes:
                    for old_val in self.inner_nodes[new_index]:
                        if old_val not in originals:
                            originals.append(old_val)
                            swapped = True
                else:
                    originals.append(new_index)
            new_indexes = originals
        return new_indexes

    def compute_max_subtract_score(self, column_index, cycle_indexes):
        if False:
            while True:
                i = 10
        '\n        When updating scores the score of the highest-weighted incoming\n        arc is subtracted upon collapse.  This returns the correct\n        amount to subtract from that edge.\n\n        :type column_index: integer.\n        :param column_index: A index representing the column of incoming arcs\n            to a particular node being updated\n        :type cycle_indexes: A list of integers.\n        :param cycle_indexes: Only arcs from cycle nodes are considered.  This\n            is a list of such nodes addresses.\n        '
        max_score = -100000
        for row_index in cycle_indexes:
            for subtract_val in self.scores[row_index][column_index]:
                if subtract_val > max_score:
                    max_score = subtract_val
        return max_score

    def best_incoming_arc(self, node_index):
        if False:
            return 10
        "\n        Returns the source of the best incoming arc to the\n        node with address: node_index\n\n        :type node_index: integer.\n        :param node_index: The address of the 'destination' node,\n            the node that is arced to.\n        "
        originals = self.compute_original_indexes([node_index])
        logger.debug('originals: %s', originals)
        max_arc = None
        max_score = None
        for row_index in range(len(self.scores)):
            for col_index in range(len(self.scores[row_index])):
                if col_index in originals and (max_score is None or self.scores[row_index][col_index] > max_score):
                    max_score = self.scores[row_index][col_index]
                    max_arc = row_index
                    logger.debug('%s, %s', row_index, col_index)
        logger.debug(max_score)
        for key in self.inner_nodes:
            replaced_nodes = self.inner_nodes[key]
            if max_arc in replaced_nodes:
                return key
        return max_arc

    def original_best_arc(self, node_index):
        if False:
            for i in range(10):
                print('nop')
        originals = self.compute_original_indexes([node_index])
        max_arc = None
        max_score = None
        max_orig = None
        for row_index in range(len(self.scores)):
            for col_index in range(len(self.scores[row_index])):
                if col_index in originals and (max_score is None or self.scores[row_index][col_index] > max_score):
                    max_score = self.scores[row_index][col_index]
                    max_arc = row_index
                    max_orig = col_index
        return [max_arc, max_orig]

    def parse(self, tokens, tags):
        if False:
            i = 10
            return i + 15
        '\n        Parses a list of tokens in accordance to the MST parsing algorithm\n        for non-projective dependency parses.  Assumes that the tokens to\n        be parsed have already been tagged and those tags are provided.  Various\n        scoring methods can be used by implementing the ``DependencyScorerI``\n        interface and passing it to the training algorithm.\n\n        :type tokens: list(str)\n        :param tokens: A list of words or punctuation to be parsed.\n        :type tags: list(str)\n        :param tags: A list of tags corresponding by index to the words in the tokens list.\n        :return: An iterator of non-projective parses.\n        :rtype: iter(DependencyGraph)\n        '
        self.inner_nodes = {}
        g_graph = DependencyGraph()
        for (index, token) in enumerate(tokens):
            g_graph.nodes[index + 1].update({'word': token, 'tag': tags[index], 'rel': 'NTOP', 'address': index + 1})
        g_graph.connect_graph()
        original_graph = DependencyGraph()
        for (index, token) in enumerate(tokens):
            original_graph.nodes[index + 1].update({'word': token, 'tag': tags[index], 'rel': 'NTOP', 'address': index + 1})
        b_graph = DependencyGraph()
        c_graph = DependencyGraph()
        for (index, token) in enumerate(tokens):
            c_graph.nodes[index + 1].update({'word': token, 'tag': tags[index], 'rel': 'NTOP', 'address': index + 1})
        self.initialize_edge_scores(g_graph)
        logger.debug(self.scores)
        unvisited_vertices = [vertex['address'] for vertex in c_graph.nodes.values()]
        nr_vertices = len(tokens)
        betas = {}
        while unvisited_vertices:
            current_vertex = unvisited_vertices.pop(0)
            logger.debug('current_vertex: %s', current_vertex)
            current_node = g_graph.get_by_address(current_vertex)
            logger.debug('current_node: %s', current_node)
            best_in_edge = self.best_incoming_arc(current_vertex)
            betas[current_vertex] = self.original_best_arc(current_vertex)
            logger.debug('best in arc: %s --> %s', best_in_edge, current_vertex)
            for new_vertex in [current_vertex, best_in_edge]:
                b_graph.nodes[new_vertex].update({'word': 'TEMP', 'rel': 'NTOP', 'address': new_vertex})
            b_graph.add_arc(best_in_edge, current_vertex)
            cycle_path = b_graph.contains_cycle()
            if cycle_path:
                new_node = {'word': 'NONE', 'rel': 'NTOP', 'address': nr_vertices + 1}
                c_graph.add_node(new_node)
                self.update_edge_scores(new_node, cycle_path)
                self.collapse_nodes(new_node, cycle_path, g_graph, b_graph, c_graph)
                for cycle_index in cycle_path:
                    c_graph.add_arc(new_node['address'], cycle_index)
                self.inner_nodes[new_node['address']] = cycle_path
                unvisited_vertices.insert(0, nr_vertices + 1)
                nr_vertices += 1
                for cycle_node_address in cycle_path:
                    b_graph.remove_by_address(cycle_node_address)
            logger.debug('g_graph: %s', g_graph)
            logger.debug('b_graph: %s', b_graph)
            logger.debug('c_graph: %s', c_graph)
            logger.debug('Betas: %s', betas)
            logger.debug('replaced nodes %s', self.inner_nodes)
        logger.debug('Final scores: %s', self.scores)
        logger.debug('Recovering parse...')
        for i in range(len(tokens) + 1, nr_vertices + 1):
            betas[betas[i][1]] = betas[i]
        logger.debug('Betas: %s', betas)
        for node in original_graph.nodes.values():
            node['deps'] = {}
        for i in range(1, len(tokens) + 1):
            original_graph.add_arc(betas[i][0], betas[i][1])
        logger.debug('Done.')
        yield original_graph

class NonprojectiveDependencyParser:
    """
    A non-projective, rule-based, dependency parser.  This parser
    will return the set of all possible non-projective parses based on
    the word-to-word relations defined in the parser's dependency
    grammar, and will allow the branches of the parse tree to cross
    in order to capture a variety of linguistic phenomena that a
    projective parser will not.
    """

    def __init__(self, dependency_grammar):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new ``NonprojectiveDependencyParser``.\n\n        :param dependency_grammar: a grammar of word-to-word relations.\n        :type dependency_grammar: DependencyGrammar\n        '
        self._grammar = dependency_grammar

    def parse(self, tokens):
        if False:
            i = 10
            return i + 15
        "\n        Parses the input tokens with respect to the parser's grammar.  Parsing\n        is accomplished by representing the search-space of possible parses as\n        a fully-connected directed graph.  Arcs that would lead to ungrammatical\n        parses are removed and a lattice is constructed of length n, where n is\n        the number of input tokens, to represent all possible grammatical\n        traversals.  All possible paths through the lattice are then enumerated\n        to produce the set of non-projective parses.\n\n        param tokens: A list of tokens to parse.\n        type tokens: list(str)\n        return: An iterator of non-projective parses.\n        rtype: iter(DependencyGraph)\n        "
        self._graph = DependencyGraph()
        for (index, token) in enumerate(tokens):
            self._graph.nodes[index] = {'word': token, 'deps': [], 'rel': 'NTOP', 'address': index}
        for head_node in self._graph.nodes.values():
            deps = []
            for dep_node in self._graph.nodes.values():
                if self._grammar.contains(head_node['word'], dep_node['word']) and head_node['word'] != dep_node['word']:
                    deps.append(dep_node['address'])
            head_node['deps'] = deps
        roots = []
        possible_heads = []
        for (i, word) in enumerate(tokens):
            heads = []
            for (j, head) in enumerate(tokens):
                if i != j and self._grammar.contains(head, word):
                    heads.append(j)
            if len(heads) == 0:
                roots.append(i)
            possible_heads.append(heads)
        if len(roots) < 2:
            if len(roots) == 0:
                for i in range(len(tokens)):
                    roots.append(i)
            analyses = []
            for _ in roots:
                stack = []
                analysis = [[] for i in range(len(possible_heads))]
            i = 0
            forward = True
            while i >= 0:
                if forward:
                    if len(possible_heads[i]) == 1:
                        analysis[i] = possible_heads[i][0]
                    elif len(possible_heads[i]) == 0:
                        analysis[i] = -1
                    else:
                        head = possible_heads[i].pop()
                        analysis[i] = head
                        stack.append([i, head])
                if not forward:
                    index_on_stack = False
                    for stack_item in stack:
                        if stack_item[0] == i:
                            index_on_stack = True
                    orig_length = len(possible_heads[i])
                    if index_on_stack and orig_length == 0:
                        for j in range(len(stack) - 1, -1, -1):
                            stack_item = stack[j]
                            if stack_item[0] == i:
                                possible_heads[i].append(stack.pop(j)[1])
                    elif index_on_stack and orig_length > 0:
                        head = possible_heads[i].pop()
                        analysis[i] = head
                        stack.append([i, head])
                        forward = True
                if i + 1 == len(possible_heads):
                    analyses.append(analysis[:])
                    forward = False
                if forward:
                    i += 1
                else:
                    i -= 1
        for analysis in analyses:
            if analysis.count(-1) > 1:
                continue
            graph = DependencyGraph()
            graph.root = graph.nodes[analysis.index(-1) + 1]
            for (address, (token, head_index)) in enumerate(zip(tokens, analysis), start=1):
                head_address = head_index + 1
                node = graph.nodes[address]
                node.update({'word': token, 'address': address})
                if head_address == 0:
                    rel = 'ROOT'
                else:
                    rel = ''
                graph.nodes[head_index + 1]['deps'][rel].append(address)
            yield graph

def demo():
    if False:
        while True:
            i = 10
    nonprojective_conll_parse_demo()
    rule_based_demo()

def hall_demo():
    if False:
        for i in range(10):
            print('nop')
    npp = ProbabilisticNonprojectiveParser()
    npp.train([], DemoScorer())
    for parse_graph in npp.parse(['v1', 'v2', 'v3'], [None, None, None]):
        print(parse_graph)

def nonprojective_conll_parse_demo():
    if False:
        while True:
            i = 10
    from nltk.parse.dependencygraph import conll_data2
    graphs = [DependencyGraph(entry) for entry in conll_data2.split('\n\n') if entry]
    npp = ProbabilisticNonprojectiveParser()
    npp.train(graphs, NaiveBayesDependencyScorer())
    for parse_graph in npp.parse(['Cathy', 'zag', 'hen', 'zwaaien', '.'], ['N', 'V', 'Pron', 'Adj', 'N', 'Punc']):
        print(parse_graph)

def rule_based_demo():
    if False:
        i = 10
        return i + 15
    from nltk.grammar import DependencyGrammar
    grammar = DependencyGrammar.fromstring("\n    'taught' -> 'play' | 'man'\n    'man' -> 'the' | 'in'\n    'in' -> 'corner'\n    'corner' -> 'the'\n    'play' -> 'golf' | 'dachshund' | 'to'\n    'dachshund' -> 'his'\n    ")
    print(grammar)
    ndp = NonprojectiveDependencyParser(grammar)
    graphs = ndp.parse(['the', 'man', 'in', 'the', 'corner', 'taught', 'his', 'dachshund', 'to', 'play', 'golf'])
    print('Graphs:')
    for graph in graphs:
        print(graph)
if __name__ == '__main__':
    demo()