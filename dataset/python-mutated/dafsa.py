"""
Main module for computing DAFSA/DAWG graphs from list of strings.

The library computes a Deterministic Acyclic Finite State Automata from a
list of sequences in a non incremental way, with no plans to expand to
incremental computation. The library was originally based on public domain
code by `Steve Hanov (2011) <http://stevehanov.ca/blog/?id=115>`__.

Adapted from dafsa/dafsa.py of
`DAFSA <https://github.com/tresoldi/dafsa>`_.
"""
from collections import Counter
import copy
import itertools

def common_prefix_length(seq_a, seq_b):
    if False:
        return 10
    '\n    Return the length of the common prefix between two sequences.\n    Parameters\n    ----------\n    seq_a : iter\n        An iterable holding the first sequence.\n    seq_b : iter\n        An iterable holding the second sequence.\n    Returns\n    -------\n    length: int\n        The length of the common prefix between `seq_a` and `seq_b`.\n    Examples\n    --------\n    >>> import dafsa\n    >>> dafsa.utils.common_prefix_length("abcde", "abcDE")\n    3\n    >>> dafsa.utils.common_prefix_length("abcde", "ABCDE")\n    0\n    '
    common_prefix_len = 0
    for i in range(min(len(seq_a), len(seq_b))):
        if seq_a[i] != seq_b[i]:
            break
        common_prefix_len += 1
    return common_prefix_len

def pairwise(iterable):
    if False:
        for i in range(10):
            print('nop')
    "\n    Iterate pairwise over an iterable.\n    The function follows the recipe offered on Python's `itertools`\n    documentation.\n    Parameters\n    ----------\n    iterable : iter\n        The iterable to be iterate pairwise.\n    Examples\n    --------\n    >>> import dafsa\n    >>> list(dafsa.utils.pairwise([1,2,3,4,5]))\n    [(1, 2), (2, 3), (3, 4), (4, 5)]\n    "
    (elem_a, elem_b) = itertools.tee(iterable)
    next(elem_b, None)
    return zip(elem_a, elem_b)

class DAFSANode:
    """
    Class representing node objects in a DAFSA.

    Each object carries an internal ``node_id`` integer identifier which must
    be locally unique within a DAFSA, but is meaningless. There is no
    implicit order nor a sequential progression must be observed.

    As in previous implementation by Hanov (2011), minimization is performed
    by comparing nodes, with equivalence determined by the standard
    Python ``.__eq__()`` method which overloads the equality operator. Nodes
    are considered identical if they have identical edges, which edges
    pointing from or to the same node. In particular, edge weight and node
    finalness, respectively expressed by the ``.weight`` and ``.final``
    properties, are *not* considered. This allows to correctly count edges
    after minimization and to have final pass-through nodes.

    Parameters
    ----------
    node_id : int
        The global unique ID for the current node.
    """

    def __init__(self, node_id):
        if False:
            print('Hello World!')
        '\n        Initializes a DAFSANode.\n        '
        self.edges = {}
        self.final = False
        self.weight = 0
        self.node_id = node_id

    def __str__(self):
        if False:
            print('Hello World!')
        '\n        Return a textual representation of the node.\n\n        The representation lists any edge, with ``id`` and ``attr``ibute. The\n        edge dictionary is sorted at every call, so that, even if\n        more expansive computationally, the function is guaranteed to be\n        idempotent in all implementations.\n\n        Please note that, as counts and final state are not accounted for,\n        the value returned by this method might be ambiguous, with different\n        nodes returning the same value. For unambigous representation,\n        the ``.__repr__()`` method must be used.\n\n.. code:: python\n        >>> from dafsa import DAFSANode, DAFSAEdge\n        >>> node = DAFSANode(0)\n        >>> node.final = True\n        >>> node.edges["x"] = DAFSAEdge(DAFSANode(1), 1)\n        >>> str(node)\n        \'x|1\'\n\n        Returns\n        -------\n        string : str\n            The (potentially ambiguous) textual representation of the\n            current node.\n        '
        buf = ';'.join(['%s|%i' % (label, self.edges[label].node.node_id) for label in sorted(self.edges)])
        return buf

    def __repr__(self):
        if False:
            while True:
                i = 10
        '\n        Return an unambigous textual representation of the node.\n\n        The representation lists any edge, with all properties. The\n        edge dictionary is sorted at every call, so that, even if\n        more expansive computationally, the function is guaranteed to be\n        idempotent in all implementations.\n\n        Please note that, as the return value includes information such as\n        edge weight, it cannot be used for minimization. For such purposes,\n        the potentially ambiguous ``.__str__()`` method must be used.\n\n.. code:: python\n        >>> from dafsa import DAFSANode, DAFSAEdge\n        >>> node = DAFSANode(0)\n        >>> node.final = True\n        >>> node.edges["x"] = DAFSAEdge(DAFSANode(1), 1)\n        >>> repr(node)\n        \'0(#1/0:<x>/1)\'\n\n        Returns\n        -------\n        string : str\n            The unambiguous textual representation of the current node.\n        '
        buf = ';'.join(['|'.join(['#%i/%i:<%s>/%i' % (self.edges[label].node.node_id, self.weight, label, self.edges[label].weight) for label in sorted(self.edges)])])
        if self.node_id == 0:
            buf = '0(%s)' % buf
        elif self.final:
            buf = 'F(%s)' % buf
        else:
            buf = 'n(%s)' % buf
        return buf

    def __eq__(self, other):
        if False:
            return 10
        '\n        Checks whether two nodes are equivalent.\n\n        Please note that this method checks for *equivalence* (in particular,\n        disregarding edge weight), and not for *equality*.\n\n        Paremeters\n        ----------\n        other : DAFSANode\n            The DAFSANode to be compared with the current one.\n\n        Returns\n        -------\n        eq : bool\n            A boolean indicating if the two nodes are equivalent.\n        '
        if len(self.edges) != len(other.edges):
            return False
        if self.final != other.final:
            return False
        for label in self.edges:
            if label not in other.edges:
                return False
            if self.edges[label].node.node_id != other.edges[label].node.node_id:
                return False
        return True

    def __gt__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Return a "greater than" comparison between two nodes.\n\n        Internally, the method reuses the ``.__str__()`` method, so that\n        the logic for comparison is implemented in a single place. As such,\n        while it guarantees idempotency when sorting nodes, it does not\n        check for properties suc like "node length", "entropy", or\n        "information amount", only providing a convenient complementary\n        method to ``.__eq__()``.\n\n        Paremeters\n        ----------\n        other : DAFSANode\n            The DAFSANode to be compared with the current one.\n\n        Returns\n        -------\n        gt : bool\n            A boolean indicating if the current node is greater than the one\n            it is compared with (that is, if it should be placed after it\n            in an ordered sequence).\n        '
        return self.__str__() > other.__str__()

    def __hash__(self):
        if False:
            return 10
        '\n        Return a hash for the node.\n\n        The returned has is based on the potentially ambigous string\n        representation provided by the ``.__str__()`` method, allowing to\n        use nodes as, among others, dictionary keys. The choice of the\n        potentially ambiguous ``.__str__()`` over ``.__repr__()`` is intentional\n        and by design and complemented by the ``.repr_hash()`` method.\n\n        Returns\n        -------\n        hash : number\n            The hash from the (potentially ambigous) textual representation of\n            the current node.\n        '
        return self.__str__().__hash__()

    def repr_hash(self):
        if False:
            return 10
        '\n        Return a hash for the node.\n\n        The returned has is based on the unambigous string\n        representation provided by the ``.__repr__()`` method, allowing to\n        use nodes as, among others, dictionary keys. The method is\n        complemented by the ``.__hash__()`` one.\n\n        Returns\n        -------\n        hash : number\n            The hash from the unambigous textual representation of the\n            current node.\n        '
        return self.__repr__().__hash__()

class DAFSAEdge(dict):
    """
    Class representing edge objects in a DAFSA.

    This class overloads a normal Python dictionary, and in simpler
    implementations could potentially be replaced with a pure dictionary.
    It was implemented as its own object for homogeneity and for planned
    future expansions, particularly in terms of fuzzy automata.

    Parameters
    ----------
    node : DAFSANode
        Reference to the target node, mandatory. Please note that it
        must be a DAFSANode object and *not* a node id.
    weight : int
        Edge weight as collected from training data. Defaults to 0.
    """

    def __init__(self, node, weight=0):
        if False:
            return 10
        '\n        Initializes a DAFSA edge.\n        '
        super().__init__()
        if not isinstance(node, DAFSANode):
            raise TypeError('`node` must be a DAFSANode (perhaps a `node_id` was passed?).')
        self.node = node
        self.weight = weight

    def __str__(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a textual representation of the node.\n\n        The representation only include the ``node_id``, without information\n        on the node actual contents.\n\n        Returns\n        -------\n        string : str\n            The (potentially ambiguous) textual representation of the\n            current edge.\n        '
        return '{node_id: %i, weight: %i}' % (self.node.node_id, self.weight)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a full textual representation of the node.\n\n        The representation includes information on the entire contents of\n        the node.\n\n        Returns\n        -------\n        string : str\n            The unambiguous textual representation of the current edge.\n        '
        return '{node: <%s>, weight: %i}' % (repr(self.node), self.weight)

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a hash for the edge.\n\n        The returned has is based on the potentially ambigous string\n        representation provided by the ``.__str__()`` method, allowing to\n        use edges as, among others, dictionary keys. The choice of the\n        potentially ambiguous ``.__str__()`` over ``.__repr__()`` is intentional\n        and by design and complemented by the ``.repr_hash()`` method.\n\n        Returns\n        -------\n        hash : number\n            The hash from the (potentially ambigous) textual representation of\n            the current edge.\n        '
        return self.__str__().__hash__()

    def repr_hash(self):
        if False:
            return 10
        '\n        Return a hash for the edge.\n\n        The returned has is based on the unambigous string\n        representation provided by the ``.__repr__()`` method, allowing to\n        use edges as, among others, dictionary keys. The method is\n        complemented by the ``.__hash__()`` one.\n\n        Returns\n        -------\n        hash : number\n            The hash from the unambigous textual representation of the\n            current edge.\n        '
        return self.__repr__().__hash__()

class DAFSA:
    """
    Class representing a DAFSA object.

    Parameters
    ----------
    sequences : list
        List of sequences to be added to the DAFSA object.
    weight : bool
        Whether to collect edge weights after minimization. Defaults
        to ``True``.
    condense: bool
        Whether to join sequences of transitions into single compound
        transitions whenever possible. Defaults to ``False``.
    delimiter : str
        The delimiter to use in case of joining single path transitions.
        Defaults to a single white space (`" "`).
    minimize : bool
        Whether to minimize the trie into a DAFSA. Defaults to ``True``; this
        option is implemented for development and testing purposes and
        it is not intended for users (there are specific and better libraries
        and algorithms to build tries).
    """

    def __init__(self, sequences, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initializes a DAFSA object.\n        '
        self._delimiter = kwargs.get('delimiter', ' ')
        minimize = kwargs.get('minimize', True)
        self._iditer = itertools.count()
        self.nodes = {0: DAFSANode(next(self._iditer))}
        self.lookup_nodes = None
        self._unchecked_nodes = []
        self._num_sequences = None
        sequences = sorted(sequences)
        self._num_sequences = len(sequences)
        for (previous_seq, seq) in pairwise([''] + sequences):
            self._insert_single_seq(seq, previous_seq, minimize)
        self._minimize(0, minimize)
        if kwargs.get('weight', True):
            self._collect_weights(sequences)
        self.lookup_nodes = copy.deepcopy(self.nodes)
        if kwargs.get('condense', False):
            self.condense()

    def _insert_single_seq(self, seq, previous_seq, minimize):
        if False:
            print('Hello World!')
        '\n        Internal method for single sequence insertion.\n\n        Parameters\n        ----------\n        seq: sequence\n            The sequence being inserted.\n        previous_seq : sequence\n            The previous sequence from the sorted list of sequences,\n            for common prefix length computation.\n        minimize : bool\n            Flag indicating whether to perform minimization or not.\n        '
        prefix_len = common_prefix_length(seq, previous_seq)
        self._minimize(prefix_len, minimize)
        if not self._unchecked_nodes:
            node = self.nodes[0]
        else:
            node = self._unchecked_nodes[-1]['child']
        for token in seq[prefix_len:]:
            child = DAFSANode(next(self._iditer))
            node.edges[token] = DAFSAEdge(child)
            self._unchecked_nodes.append({'parent': node, 'token': token, 'child': child})
            node = child
        node.final = True

    def _minimize(self, index, minimize):
        if False:
            for i in range(10):
                print('nop')
        '\n        Internal method for graph minimization.\n\n        Minimize the graph from the last unchecked item until ``index``.\n        Final minimization, with ``index`` equal to zero, will traverse the\n        entire data structure.\n\n        The method allows the minimization to be overridden by setting to\n        ``False`` the ``minimize`` flag (returning a trie). Due to the logic in\n        place for the DAFSA minimization, this ends up executed as a\n        non-efficient code, where all comparisons fail, but it is\n        necessary to do it this way to clean the list of unchecked nodes.\n        This is not an implementation problem: this class is not supposed\n        to be used for generating tries (there are more efficient ways of\n        doing that), but it worth having the flag in place for experiments.\n\n        Parameters\n        ----------\n        index : int\n            The index until the sequence minimization, right to left.\n        minimize : bool\n            Flag indicating whether to perform minimization or not.\n        '
        while True:
            graph_changed = False
            for _ in range(len(self._unchecked_nodes) - index):
                unchecked_node = self._unchecked_nodes.pop()
                parent = unchecked_node['parent']
                token = unchecked_node['token']
                child = unchecked_node['child']
                if not minimize:
                    self.nodes[child.node_id] = child
                else:
                    child_idx = None
                    for (node_idx, node) in self.nodes.items():
                        if node == child:
                            child_idx = node_idx
                            break
                    if child_idx:
                        if parent.edges[token].node.final:
                            self.nodes[child_idx].final = True
                        parent.edges[token].node = self.nodes[child_idx]
                        graph_changed = True
                    else:
                        self.nodes[child.node_id] = child
            if not graph_changed:
                break

    def condense(self):
        if False:
            while True:
                i = 10
        '\n        Condenses the automaton, merging single-child nodes with their parents.\n\n        The function joins paths of unique edges into single edges with\n        compound transitions, removing redundant nodes. A redundant node\n        is defined as one that (a) is not final, (b) emits a single transition,\n        (b) receives a single transition, and (d) its source emits a single\n        transition.\n\n        Internally, the function will call the ``._joining_round()``\n        method until no more candidates for joining are available.\n        Performing everything in a single step would require a more complex\n        logic.\n        '
        while True:
            if self._joining_round() == 0:
                break

    def _joining_round(self):
        if False:
            return 10
        '\n        Internal method for the unique-edge joining algorithm.\n\n        This function will be called a successive number of times by\n        ``._join_transitions()``, until no more candidates for unique-edge\n        joining are available (as informed by its return value).\n\n        Returns\n        -------\n        num_operations: int\n            The number of joining operations that was performed. When zero,\n            it signals that no more joining is possible.\n        '
        edges = []
        for (source_id, node) in self.nodes.items():
            edges += [{'source': source_id, 'target': node.edges[label].node.node_id} for label in node.edges]
        sources = Counter([edge['source'] for edge in edges])
        targets = Counter([edge['target'] for edge in edges])
        transitions = []
        transitions_nodes = []
        for (node_id, node) in self.nodes.items():
            if targets[node_id] > 1:
                continue
            if sources[node_id] > 1:
                continue
            if node.final:
                continue
            edge_info = [edge for edge in edges if edge['target'] == node_id][0]
            label_from = [label for label in self.nodes[edge_info['source']].edges if self.nodes[edge_info['source']].edges[label].node.node_id == edge_info['target']][0]
            label_to = list(node.edges.keys())[0]
            if all([node_id not in transitions_nodes for node_id in edge_info]):
                transitions_nodes += edge_info
                transitions.append({'edge': edge_info, 'label_from': label_from, 'label_to': label_to})
        for transition in transitions:
            new_label = self._delimiter.join([transition['label_from'], transition['label_to']])
            self.nodes[transition['edge']['source']].edges[new_label] = DAFSAEdge(self.nodes[transition['edge']['target']].edges[transition['label_to']].node, self.nodes[transition['edge']['target']].edges[transition['label_to']].weight)
            self.nodes[transition['edge']['source']].edges.pop(transition['label_from'])
            self.nodes.pop(transition['edge']['target'])
        return len(transitions)

    def _collect_weights(self, sequences):
        if False:
            while True:
                i = 10
        '\n        Internal method for collecting node and edge weights from sequences.\n\n        This method requires the minimized graph to be already in place.\n\n        Parameters\n        ----------\n        sequences : list\n            List of sequences whose node and edge weights will be collected.\n        '
        for seq in sequences:
            node = self.nodes[0]
            node.weight += 1
            for token in seq:
                node.edges[token].weight += 1
                node = node.edges[token].node
                node.weight += 1

    def lookup(self, sequence, stop_on_prefix=False):
        if False:
            return 10
        '\n        Check if a sequence can be expressed by the DAFSA.\n\n        The method does not return all possible potential paths, nor\n        the cumulative weight: if this is needed, the DAFSA object should\n        be converted to a Graph and other libraries, such as ``networkx``,\n        should be used.\n\n        Parameters\n        ----------\n        sequence : sequence\n            Sequence to be checked for presence/absence.\n\n        Returns\n        -------\n        node : tuple of DAFSANode and int, or None\n            Either a tuple with a DAFSANode referring to the final state\n            that can be reached by following the specified sequence,\n            plus the cumulative weight for reaching it, or None if no path\n            can be found.\n        '
        node = self.lookup_nodes[0]
        cum_weight = 0
        for token in sequence:
            if token not in node.edges:
                return None
            cum_weight += node.edges[token].weight
            node = node.edges[token].node
            if stop_on_prefix and node.final:
                break
        if not node.final:
            return None
        return (node, cum_weight)

    def count_nodes(self):
        if False:
            while True:
                i = 10
        '\n        Return the number of minimized nodes in the structure.\n\n        Returns\n        -------\n        node_count : int\n            Number of minimized nodes in the structure.\n        '
        return len(self.nodes)

    def count_edges(self):
        if False:
            print('Hello World!')
        '\n        Return the number of minimized edges in the structure.\n\n        Returns\n        -------\n        edge_count : int\n            Number of minimized edges in the structure.\n        '
        return sum([len(node.edges) for node in self.nodes.values()])

    def count_sequences(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the number of sequences inserted in the structure.\n\n        Please note that the return value mirrors the number of sequences\n        provided during initialization, and *not* a set of it: repeated\n        sequences are accounted, as each will be added a single time to\n        the object.\n\n        Returns\n        -------\n        seq_count : int\n            Number of sequences in the structure.\n        '
        return self._num_sequences

    def __str__(self):
        if False:
            return 10
        '\n        Return a readable multiline textual representation of the object.\n\n        Returns\n        -------\n        string : str\n            The textual representation of the object.\n        '
        buf = ['DAFSA with %i nodes and %i edges (%i inserted sequences)' % (self.count_nodes(), self.count_edges(), self.count_sequences())]
        for node_id in sorted(self.nodes):
            node = self.nodes[node_id]
            buf += ['  +-- #%i: %s %s' % (node_id, repr(node), [(attr, n.node.node_id) for (attr, n) in node.edges.items()])]
        return '\n'.join(buf)