"""
Tools for reading and writing dependency trees.
The input is assumed to be in Malt-TAB format
(https://stp.lingfil.uu.se/~nivre/research/MaltXML.html).
"""
import subprocess
import warnings
from collections import defaultdict
from itertools import chain
from pprint import pformat
from nltk.internals import find_binary
from nltk.tree import Tree

class DependencyGraph:
    """
    A container for the nodes and labelled edges of a dependency structure.
    """

    def __init__(self, tree_str=None, cell_extractor=None, zero_based=False, cell_separator=None, top_relation_label='ROOT'):
        if False:
            while True:
                i = 10
        'Dependency graph.\n\n        We place a dummy `TOP` node with the index 0, since the root node is\n        often assigned 0 as its head. This also means that the indexing of the\n        nodes corresponds directly to the Malt-TAB format, which starts at 1.\n\n        If zero-based is True, then Malt-TAB-like input with node numbers\n        starting at 0 and the root node assigned -1 (as produced by, e.g.,\n        zpar).\n\n        :param str cell_separator: the cell separator. If not provided, cells\n            are split by whitespace.\n\n        :param str top_relation_label: the label by which the top relation is\n            identified, for examlple, `ROOT`, `null` or `TOP`.\n        '
        self.nodes = defaultdict(lambda : {'address': None, 'word': None, 'lemma': None, 'ctag': None, 'tag': None, 'feats': None, 'head': None, 'deps': defaultdict(list), 'rel': None})
        self.nodes[0].update({'ctag': 'TOP', 'tag': 'TOP', 'address': 0})
        self.root = None
        if tree_str:
            self._parse(tree_str, cell_extractor=cell_extractor, zero_based=zero_based, cell_separator=cell_separator, top_relation_label=top_relation_label)

    def remove_by_address(self, address):
        if False:
            for i in range(10):
                print('nop')
        '\n        Removes the node with the given address.  References\n        to this node in others will still exist.\n        '
        del self.nodes[address]

    def redirect_arcs(self, originals, redirect):
        if False:
            for i in range(10):
                print('nop')
        '\n        Redirects arcs to any of the nodes in the originals list\n        to the redirect node address.\n        '
        for node in self.nodes.values():
            new_deps = []
            for dep in node['deps']:
                if dep in originals:
                    new_deps.append(redirect)
                else:
                    new_deps.append(dep)
            node['deps'] = new_deps

    def add_arc(self, head_address, mod_address):
        if False:
            i = 10
            return i + 15
        '\n        Adds an arc from the node specified by head_address to the\n        node specified by the mod address.\n        '
        relation = self.nodes[mod_address]['rel']
        self.nodes[head_address]['deps'].setdefault(relation, [])
        self.nodes[head_address]['deps'][relation].append(mod_address)

    def connect_graph(self):
        if False:
            i = 10
            return i + 15
        '\n        Fully connects all non-root nodes.  All nodes are set to be dependents\n        of the root node.\n        '
        for node1 in self.nodes.values():
            for node2 in self.nodes.values():
                if node1['address'] != node2['address'] and node2['rel'] != 'TOP':
                    relation = node2['rel']
                    node1['deps'].setdefault(relation, [])
                    node1['deps'][relation].append(node2['address'])

    def get_by_address(self, node_address):
        if False:
            return 10
        'Return the node with the given address.'
        return self.nodes[node_address]

    def contains_address(self, node_address):
        if False:
            i = 10
            return i + 15
        '\n        Returns true if the graph contains a node with the given node\n        address, false otherwise.\n        '
        return node_address in self.nodes

    def to_dot(self):
        if False:
            i = 10
            return i + 15
        'Return a dot representation suitable for using with Graphviz.\n\n        >>> dg = DependencyGraph(\n        ...     \'John N 2\\n\'\n        ...     \'loves V 0\\n\'\n        ...     \'Mary N 2\'\n        ... )\n        >>> print(dg.to_dot())\n        digraph G{\n        edge [dir=forward]\n        node [shape=plaintext]\n        <BLANKLINE>\n        0 [label="0 (None)"]\n        0 -> 2 [label="ROOT"]\n        1 [label="1 (John)"]\n        2 [label="2 (loves)"]\n        2 -> 1 [label=""]\n        2 -> 3 [label=""]\n        3 [label="3 (Mary)"]\n        }\n\n        '
        s = 'digraph G{\n'
        s += 'edge [dir=forward]\n'
        s += 'node [shape=plaintext]\n'
        for node in sorted(self.nodes.values(), key=lambda v: v['address']):
            s += '\n{} [label="{} ({})"]'.format(node['address'], node['address'], node['word'])
            for (rel, deps) in node['deps'].items():
                for dep in deps:
                    if rel is not None:
                        s += '\n{} -> {} [label="{}"]'.format(node['address'], dep, rel)
                    else:
                        s += '\n{} -> {} '.format(node['address'], dep)
        s += '\n}'
        return s

    def _repr_svg_(self):
        if False:
            i = 10
            return i + 15
        'Show SVG representation of the transducer (IPython magic).\n        >>> from nltk.test.setup_fixt import check_binary\n        >>> check_binary(\'dot\')\n        >>> dg = DependencyGraph(\n        ...     \'John N 2\\n\'\n        ...     \'loves V 0\\n\'\n        ...     \'Mary N 2\'\n        ... )\n        >>> dg._repr_svg_().split(\'\\n\')[0]\n        \'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\'\n\n        '
        dot_string = self.to_dot()
        return dot2img(dot_string)

    def __str__(self):
        if False:
            print('Hello World!')
        return pformat(self.nodes)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'<DependencyGraph with {len(self.nodes)} nodes>'

    @staticmethod
    def load(filename, zero_based=False, cell_separator=None, top_relation_label='ROOT'):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param filename: a name of a file in Malt-TAB format\n        :param zero_based: nodes in the input file are numbered starting from 0\n            rather than 1 (as produced by, e.g., zpar)\n        :param str cell_separator: the cell separator. If not provided, cells\n            are split by whitespace.\n        :param str top_relation_label: the label by which the top relation is\n            identified, for examlple, `ROOT`, `null` or `TOP`.\n\n        :return: a list of DependencyGraphs\n\n        '
        with open(filename) as infile:
            return [DependencyGraph(tree_str, zero_based=zero_based, cell_separator=cell_separator, top_relation_label=top_relation_label) for tree_str in infile.read().split('\n\n')]

    def left_children(self, node_index):
        if False:
            while True:
                i = 10
        '\n        Returns the number of left children under the node specified\n        by the given address.\n        '
        children = chain.from_iterable(self.nodes[node_index]['deps'].values())
        index = self.nodes[node_index]['address']
        return sum((1 for c in children if c < index))

    def right_children(self, node_index):
        if False:
            return 10
        '\n        Returns the number of right children under the node specified\n        by the given address.\n        '
        children = chain.from_iterable(self.nodes[node_index]['deps'].values())
        index = self.nodes[node_index]['address']
        return sum((1 for c in children if c > index))

    def add_node(self, node):
        if False:
            print('Hello World!')
        if not self.contains_address(node['address']):
            self.nodes[node['address']].update(node)

    def _parse(self, input_, cell_extractor=None, zero_based=False, cell_separator=None, top_relation_label='ROOT'):
        if False:
            while True:
                i = 10
        'Parse a sentence.\n\n        :param extractor: a function that given a tuple of cells returns a\n        7-tuple, where the values are ``word, lemma, ctag, tag, feats, head,\n        rel``.\n\n        :param str cell_separator: the cell separator. If not provided, cells\n        are split by whitespace.\n\n        :param str top_relation_label: the label by which the top relation is\n        identified, for examlple, `ROOT`, `null` or `TOP`.\n\n        '

        def extract_3_cells(cells, index):
            if False:
                for i in range(10):
                    print('nop')
            (word, tag, head) = cells
            return (index, word, word, tag, tag, '', head, '')

        def extract_4_cells(cells, index):
            if False:
                while True:
                    i = 10
            (word, tag, head, rel) = cells
            return (index, word, word, tag, tag, '', head, rel)

        def extract_7_cells(cells, index):
            if False:
                return 10
            (line_index, word, lemma, tag, _, head, rel) = cells
            try:
                index = int(line_index)
            except ValueError:
                pass
            return (index, word, lemma, tag, tag, '', head, rel)

        def extract_10_cells(cells, index):
            if False:
                for i in range(10):
                    print('nop')
            (line_index, word, lemma, ctag, tag, feats, head, rel, _, _) = cells
            try:
                index = int(line_index)
            except ValueError:
                pass
            return (index, word, lemma, ctag, tag, feats, head, rel)
        extractors = {3: extract_3_cells, 4: extract_4_cells, 7: extract_7_cells, 10: extract_10_cells}
        if isinstance(input_, str):
            input_ = (line for line in input_.split('\n'))
        lines = (l.rstrip() for l in input_)
        lines = (l for l in lines if l)
        cell_number = None
        for (index, line) in enumerate(lines, start=1):
            cells = line.split(cell_separator)
            if cell_number is None:
                cell_number = len(cells)
            else:
                assert cell_number == len(cells)
            if cell_extractor is None:
                try:
                    cell_extractor = extractors[cell_number]
                except KeyError as e:
                    raise ValueError('Number of tab-delimited fields ({}) not supported by CoNLL(10) or Malt-Tab(4) format'.format(cell_number)) from e
            try:
                (index, word, lemma, ctag, tag, feats, head, rel) = cell_extractor(cells, index)
            except (TypeError, ValueError):
                (word, lemma, ctag, tag, feats, head, rel) = cell_extractor(cells)
            if head == '_':
                continue
            head = int(head)
            if zero_based:
                head += 1
            self.nodes[index].update({'address': index, 'word': word, 'lemma': lemma, 'ctag': ctag, 'tag': tag, 'feats': feats, 'head': head, 'rel': rel})
            if cell_number == 3 and head == 0:
                rel = top_relation_label
            self.nodes[head]['deps'][rel].append(index)
        if self.nodes[0]['deps'][top_relation_label]:
            root_address = self.nodes[0]['deps'][top_relation_label][0]
            self.root = self.nodes[root_address]
            self.top_relation_label = top_relation_label
        else:
            warnings.warn("The graph doesn't contain a node that depends on the root element.")

    def _word(self, node, filter=True):
        if False:
            while True:
                i = 10
        w = node['word']
        if filter:
            if w != ',':
                return w
        return w

    def _tree(self, i):
        if False:
            for i in range(10):
                print('nop')
        'Turn dependency graphs into NLTK trees.\n\n        :param int i: index of a node\n        :return: either a word (if the indexed node is a leaf) or a ``Tree``.\n        '
        node = self.get_by_address(i)
        word = node['word']
        deps = sorted(chain.from_iterable(node['deps'].values()))
        if deps:
            return Tree(word, [self._tree(dep) for dep in deps])
        else:
            return word

    def tree(self):
        if False:
            while True:
                i = 10
        '\n        Starting with the ``root`` node, build a dependency tree using the NLTK\n        ``Tree`` constructor. Dependency labels are omitted.\n        '
        node = self.root
        word = node['word']
        deps = sorted(chain.from_iterable(node['deps'].values()))
        return Tree(word, [self._tree(dep) for dep in deps])

    def triples(self, node=None):
        if False:
            i = 10
            return i + 15
        '\n        Extract dependency triples of the form:\n        ((head word, head tag), rel, (dep word, dep tag))\n        '
        if not node:
            node = self.root
        head = (node['word'], node['ctag'])
        for i in sorted(chain.from_iterable(node['deps'].values())):
            dep = self.get_by_address(i)
            yield (head, dep['rel'], (dep['word'], dep['ctag']))
            yield from self.triples(node=dep)

    def _hd(self, i):
        if False:
            i = 10
            return i + 15
        try:
            return self.nodes[i]['head']
        except IndexError:
            return None

    def _rel(self, i):
        if False:
            return 10
        try:
            return self.nodes[i]['rel']
        except IndexError:
            return None

    def contains_cycle(self):
        if False:
            for i in range(10):
                print('nop')
        "Check whether there are cycles.\n\n        >>> dg = DependencyGraph(treebank_data)\n        >>> dg.contains_cycle()\n        False\n\n        >>> cyclic_dg = DependencyGraph()\n        >>> top = {'word': None, 'deps': [1], 'rel': 'TOP', 'address': 0}\n        >>> child1 = {'word': None, 'deps': [2], 'rel': 'NTOP', 'address': 1}\n        >>> child2 = {'word': None, 'deps': [4], 'rel': 'NTOP', 'address': 2}\n        >>> child3 = {'word': None, 'deps': [1], 'rel': 'NTOP', 'address': 3}\n        >>> child4 = {'word': None, 'deps': [3], 'rel': 'NTOP', 'address': 4}\n        >>> cyclic_dg.nodes = {\n        ...     0: top,\n        ...     1: child1,\n        ...     2: child2,\n        ...     3: child3,\n        ...     4: child4,\n        ... }\n        >>> cyclic_dg.root = top\n\n        >>> cyclic_dg.contains_cycle()\n        [1, 2, 4, 3]\n\n        "
        distances = {}
        for node in self.nodes.values():
            for dep in node['deps']:
                key = tuple([node['address'], dep])
                distances[key] = 1
        for _ in self.nodes:
            new_entries = {}
            for pair1 in distances:
                for pair2 in distances:
                    if pair1[1] == pair2[0]:
                        key = tuple([pair1[0], pair2[1]])
                        new_entries[key] = distances[pair1] + distances[pair2]
            for pair in new_entries:
                distances[pair] = new_entries[pair]
                if pair[0] == pair[1]:
                    path = self.get_cycle_path(self.get_by_address(pair[0]), pair[0])
                    return path
        return False

    def get_cycle_path(self, curr_node, goal_node_index):
        if False:
            while True:
                i = 10
        for dep in curr_node['deps']:
            if dep == goal_node_index:
                return [curr_node['address']]
        for dep in curr_node['deps']:
            path = self.get_cycle_path(self.get_by_address(dep), goal_node_index)
            if len(path) > 0:
                path.insert(0, curr_node['address'])
                return path
        return []

    def to_conll(self, style):
        if False:
            print('Hello World!')
        '\n        The dependency graph in CoNLL format.\n\n        :param style: the style to use for the format (3, 4, 10 columns)\n        :type style: int\n        :rtype: str\n        '
        if style == 3:
            template = '{word}\t{tag}\t{head}\n'
        elif style == 4:
            template = '{word}\t{tag}\t{head}\t{rel}\n'
        elif style == 10:
            template = '{i}\t{word}\t{lemma}\t{ctag}\t{tag}\t{feats}\t{head}\t{rel}\t_\t_\n'
        else:
            raise ValueError('Number of tab-delimited fields ({}) not supported by CoNLL(10) or Malt-Tab(4) format'.format(style))
        return ''.join((template.format(i=i, **node) for (i, node) in sorted(self.nodes.items()) if node['tag'] != 'TOP'))

    def nx_graph(self):
        if False:
            for i in range(10):
                print('nop')
        'Convert the data in a ``nodelist`` into a networkx labeled directed graph.'
        import networkx
        nx_nodelist = list(range(1, len(self.nodes)))
        nx_edgelist = [(n, self._hd(n), self._rel(n)) for n in nx_nodelist if self._hd(n)]
        self.nx_labels = {}
        for n in nx_nodelist:
            self.nx_labels[n] = self.nodes[n]['word']
        g = networkx.MultiDiGraph()
        g.add_nodes_from(nx_nodelist)
        g.add_edges_from(nx_edgelist)
        return g

def dot2img(dot_string, t='svg'):
    if False:
        print('Hello World!')
    '\n    Create image representation fom dot_string, using the \'dot\' program\n    from the Graphviz package.\n\n    Use the \'t\' argument to specify the image file format, for ex. \'jpeg\', \'eps\',\n    \'json\', \'png\' or \'webp\' (Running \'dot -T:\' lists all available formats).\n\n    Note that the "capture_output" option of subprocess.run() is only available\n    with text formats (like svg), but not with binary image formats (like png).\n    '
    try:
        find_binary('dot')
        try:
            if t in ['dot', 'dot_json', 'json', 'svg']:
                proc = subprocess.run(['dot', '-T%s' % t], capture_output=True, input=dot_string, text=True)
            else:
                proc = subprocess.run(['dot', '-T%s' % t], input=bytes(dot_string, encoding='utf8'))
            return proc.stdout
        except:
            raise Exception('Cannot create image representation by running dot from string: {}'.format(dot_string))
    except OSError as e:
        raise Exception('Cannot find the dot binary from Graphviz package') from e

class DependencyGraphError(Exception):
    """Dependency graph exception."""

def demo():
    if False:
        while True:
            i = 10
    malt_demo()
    conll_demo()
    conll_file_demo()
    cycle_finding_demo()

def malt_demo(nx=False):
    if False:
        print('Hello World!')
    '\n    A demonstration of the result of reading a dependency\n    version of the first sentence of the Penn Treebank.\n    '
    dg = DependencyGraph('Pierre  NNP     2       NMOD\nVinken  NNP     8       SUB\n,       ,       2       P\n61      CD      5       NMOD\nyears   NNS     6       AMOD\nold     JJ      2       NMOD\n,       ,       2       P\nwill    MD      0       ROOT\njoin    VB      8       VC\nthe     DT      11      NMOD\nboard   NN      9       OBJ\nas      IN      9       VMOD\na       DT      15      NMOD\nnonexecutive    JJ      15      NMOD\ndirector        NN      12      PMOD\nNov.    NNP     9       VMOD\n29      CD      16      NMOD\n.       .       9       VMOD\n')
    tree = dg.tree()
    tree.pprint()
    if nx:
        import networkx
        from matplotlib import pylab
        g = dg.nx_graph()
        g.info()
        pos = networkx.spring_layout(g, dim=1)
        networkx.draw_networkx_nodes(g, pos, node_size=50)
        networkx.draw_networkx_labels(g, pos, dg.nx_labels)
        pylab.xticks([])
        pylab.yticks([])
        pylab.savefig('tree.png')
        pylab.show()

def conll_demo():
    if False:
        while True:
            i = 10
    '\n    A demonstration of how to read a string representation of\n    a CoNLL format dependency tree.\n    '
    dg = DependencyGraph(conll_data1)
    tree = dg.tree()
    tree.pprint()
    print(dg)
    print(dg.to_conll(4))

def conll_file_demo():
    if False:
        for i in range(10):
            print('nop')
    print('Mass conll_read demo...')
    graphs = [DependencyGraph(entry) for entry in conll_data2.split('\n\n') if entry]
    for graph in graphs:
        tree = graph.tree()
        print('\n')
        tree.pprint()

def cycle_finding_demo():
    if False:
        print('Hello World!')
    dg = DependencyGraph(treebank_data)
    print(dg.contains_cycle())
    cyclic_dg = DependencyGraph()
    cyclic_dg.add_node({'word': None, 'deps': [1], 'rel': 'TOP', 'address': 0})
    cyclic_dg.add_node({'word': None, 'deps': [2], 'rel': 'NTOP', 'address': 1})
    cyclic_dg.add_node({'word': None, 'deps': [4], 'rel': 'NTOP', 'address': 2})
    cyclic_dg.add_node({'word': None, 'deps': [1], 'rel': 'NTOP', 'address': 3})
    cyclic_dg.add_node({'word': None, 'deps': [3], 'rel': 'NTOP', 'address': 4})
    print(cyclic_dg.contains_cycle())
treebank_data = 'Pierre  NNP     2       NMOD\nVinken  NNP     8       SUB\n,       ,       2       P\n61      CD      5       NMOD\nyears   NNS     6       AMOD\nold     JJ      2       NMOD\n,       ,       2       P\nwill    MD      0       ROOT\njoin    VB      8       VC\nthe     DT      11      NMOD\nboard   NN      9       OBJ\nas      IN      9       VMOD\na       DT      15      NMOD\nnonexecutive    JJ      15      NMOD\ndirector        NN      12      PMOD\nNov.    NNP     9       VMOD\n29      CD      16      NMOD\n.       .       9       VMOD\n'
conll_data1 = '\n1   Ze                ze                Pron  Pron  per|3|evofmv|nom                 2   su      _  _\n2   had               heb               V     V     trans|ovt|1of2of3|ev             0   ROOT    _  _\n3   met               met               Prep  Prep  voor                             8   mod     _  _\n4   haar              haar              Pron  Pron  bez|3|ev|neut|attr               5   det     _  _\n5   moeder            moeder            N     N     soort|ev|neut                    3   obj1    _  _\n6   kunnen            kan               V     V     hulp|ott|1of2of3|mv              2   vc      _  _\n7   gaan              ga                V     V     hulp|inf                         6   vc      _  _\n8   winkelen          winkel            V     V     intrans|inf                      11  cnj     _  _\n9   ,                 ,                 Punc  Punc  komma                            8   punct   _  _\n10  zwemmen           zwem              V     V     intrans|inf                      11  cnj     _  _\n11  of                of                Conj  Conj  neven                            7   vc      _  _\n12  terrassen         terras            N     N     soort|mv|neut                    11  cnj     _  _\n13  .                 .                 Punc  Punc  punt                             12  punct   _  _\n'
conll_data2 = '1   Cathy             Cathy             N     N     eigen|ev|neut                    2   su      _  _\n2   zag               zie               V     V     trans|ovt|1of2of3|ev             0   ROOT    _  _\n3   hen               hen               Pron  Pron  per|3|mv|datofacc                2   obj1    _  _\n4   wild              wild              Adj   Adj   attr|stell|onverv                5   mod     _  _\n5   zwaaien           zwaai             N     N     soort|mv|neut                    2   vc      _  _\n6   .                 .                 Punc  Punc  punt                             5   punct   _  _\n\n1   Ze                ze                Pron  Pron  per|3|evofmv|nom                 2   su      _  _\n2   had               heb               V     V     trans|ovt|1of2of3|ev             0   ROOT    _  _\n3   met               met               Prep  Prep  voor                             8   mod     _  _\n4   haar              haar              Pron  Pron  bez|3|ev|neut|attr               5   det     _  _\n5   moeder            moeder            N     N     soort|ev|neut                    3   obj1    _  _\n6   kunnen            kan               V     V     hulp|ott|1of2of3|mv              2   vc      _  _\n7   gaan              ga                V     V     hulp|inf                         6   vc      _  _\n8   winkelen          winkel            V     V     intrans|inf                      11  cnj     _  _\n9   ,                 ,                 Punc  Punc  komma                            8   punct   _  _\n10  zwemmen           zwem              V     V     intrans|inf                      11  cnj     _  _\n11  of                of                Conj  Conj  neven                            7   vc      _  _\n12  terrassen         terras            N     N     soort|mv|neut                    11  cnj     _  _\n13  .                 .                 Punc  Punc  punt                             12  punct   _  _\n\n1   Dat               dat               Pron  Pron  aanw|neut|attr                   2   det     _  _\n2   werkwoord         werkwoord         N     N     soort|ev|neut                    6   obj1    _  _\n3   had               heb               V     V     hulp|ovt|1of2of3|ev              0   ROOT    _  _\n4   ze                ze                Pron  Pron  per|3|evofmv|nom                 6   su      _  _\n5   zelf              zelf              Pron  Pron  aanw|neut|attr|wzelf             3   predm   _  _\n6   uitgevonden       vind              V     V     trans|verldw|onverv              3   vc      _  _\n7   .                 .                 Punc  Punc  punt                             6   punct   _  _\n\n1   Het               het               Pron  Pron  onbep|neut|zelfst                2   su      _  _\n2   hoorde            hoor              V     V     trans|ovt|1of2of3|ev             0   ROOT    _  _\n3   bij               bij               Prep  Prep  voor                             2   ld      _  _\n4   de                de                Art   Art   bep|zijdofmv|neut                6   det     _  _\n5   warme             warm              Adj   Adj   attr|stell|vervneut              6   mod     _  _\n6   zomerdag          zomerdag          N     N     soort|ev|neut                    3   obj1    _  _\n7   die               die               Pron  Pron  betr|neut|zelfst                 6   mod     _  _\n8   ze                ze                Pron  Pron  per|3|evofmv|nom                 12  su      _  _\n9   ginds             ginds             Adv   Adv   gew|aanw                         12  mod     _  _\n10  achter            achter            Adv   Adv   gew|geenfunc|stell|onverv        12  svp     _  _\n11  had               heb               V     V     hulp|ovt|1of2of3|ev              7   body    _  _\n12  gelaten           laat              V     V     trans|verldw|onverv              11  vc      _  _\n13  .                 .                 Punc  Punc  punt                             12  punct   _  _\n\n1   Ze                ze                Pron  Pron  per|3|evofmv|nom                 2   su      _  _\n2   hadden            heb               V     V     trans|ovt|1of2of3|mv             0   ROOT    _  _\n3   languit           languit           Adv   Adv   gew|geenfunc|stell|onverv        11  mod     _  _\n4   naast             naast             Prep  Prep  voor                             11  mod     _  _\n5   elkaar            elkaar            Pron  Pron  rec|neut                         4   obj1    _  _\n6   op                op                Prep  Prep  voor                             11  ld      _  _\n7   de                de                Art   Art   bep|zijdofmv|neut                8   det     _  _\n8   strandstoelen     strandstoel       N     N     soort|mv|neut                    6   obj1    _  _\n9   kunnen            kan               V     V     hulp|inf                         2   vc      _  _\n10  gaan              ga                V     V     hulp|inf                         9   vc      _  _\n11  liggen            lig               V     V     intrans|inf                      10  vc      _  _\n12  .                 .                 Punc  Punc  punt                             11  punct   _  _\n\n1   Zij               zij               Pron  Pron  per|3|evofmv|nom                 2   su      _  _\n2   zou               zal               V     V     hulp|ovt|1of2of3|ev              7   cnj     _  _\n3   mams              mams              N     N     soort|ev|neut                    4   det     _  _\n4   rug               rug               N     N     soort|ev|neut                    5   obj1    _  _\n5   ingewreven        wrijf             V     V     trans|verldw|onverv              6   vc      _  _\n6   hebben            heb               V     V     hulp|inf                         2   vc      _  _\n7   en                en                Conj  Conj  neven                            0   ROOT    _  _\n8   mam               mam               V     V     trans|ovt|1of2of3|ev             7   cnj     _  _\n9   de                de                Art   Art   bep|zijdofmv|neut                10  det     _  _\n10  hare              hare              Pron  Pron  bez|3|ev|neut|attr               8   obj1    _  _\n11  .                 .                 Punc  Punc  punt                             10  punct   _  _\n\n1   Of                of                Conj  Conj  onder|metfin                     0   ROOT    _  _\n2   ze                ze                Pron  Pron  per|3|evofmv|nom                 3   su      _  _\n3   had               heb               V     V     hulp|ovt|1of2of3|ev              0   ROOT    _  _\n4   gewoon            gewoon            Adj   Adj   adv|stell|onverv                 10  mod     _  _\n5   met               met               Prep  Prep  voor                             10  mod     _  _\n6   haar              haar              Pron  Pron  bez|3|ev|neut|attr               7   det     _  _\n7   vriendinnen       vriendin          N     N     soort|mv|neut                    5   obj1    _  _\n8   rond              rond              Adv   Adv   deelv                            10  svp     _  _\n9   kunnen            kan               V     V     hulp|inf                         3   vc      _  _\n10  slenteren         slenter           V     V     intrans|inf                      9   vc      _  _\n11  in                in                Prep  Prep  voor                             10  mod     _  _\n12  de                de                Art   Art   bep|zijdofmv|neut                13  det     _  _\n13  buurt             buurt             N     N     soort|ev|neut                    11  obj1    _  _\n14  van               van               Prep  Prep  voor                             13  mod     _  _\n15  Trafalgar_Square  Trafalgar_Square  MWU   N_N   eigen|ev|neut_eigen|ev|neut      14  obj1    _  _\n16  .                 .                 Punc  Punc  punt                             15  punct   _  _\n'
if __name__ == '__main__':
    demo()