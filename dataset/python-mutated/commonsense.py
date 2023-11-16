from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from builtins import str, bytes, dict, int
from builtins import map, zip, filter
from builtins import object, range
from codecs import BOM_UTF8
from itertools import chain
from functools import cmp_to_key
from io import open
try:
    from urllib import urlopen
except ImportError:
    from urllib.request import urlopen
from .__init__ import Graph, Node, Edge, bfs
from .__init__ import WEIGHT, CENTRALITY, EIGENVECTOR, BETWEENNESS
import os
import sys
try:
    MODULE = os.path.dirname(os.path.realpath(__file__))
except:
    MODULE = ''
if sys.version > '3':
    BOM_UTF8 = str(BOM_UTF8.decode('utf-8'))
else:
    BOM_UTF8 = BOM_UTF8.decode('utf-8')

class Concept(Node):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        ' A concept in the sematic network.\n        '
        Node.__init__(self, *args, **kwargs)
        self._properties = None

    @property
    def halo(self, depth=2):
        if False:
            return 10
        ' Returns the concept halo: a list with this concept + surrounding concepts.\n            This is useful to reason more fluidly about the concept,\n            since the halo will include latent properties linked to nearby concepts.\n        '
        return self.flatten(depth=depth)

    @property
    def properties(self):
        if False:
            print('Hello World!')
        " Returns the top properties in the concept halo, sorted by betweenness centrality.\n            The return value is a list of concept id's instead of Concepts (for performance).\n        "
        if self._properties is None:
            g = self.graph.copy(nodes=self.halo)
            p = (n for n in g.nodes if n.id in self.graph.properties)
            p = [n.id for n in reversed(sorted(p, key=lambda n: n.centrality))]
            self._properties = p
        return self._properties

def halo(concept, depth=2):
    if False:
        while True:
            i = 10
    return concept.flatten(depth=depth)

def properties(concept, depth=2, centrality=BETWEENNESS):
    if False:
        return 10
    g = concept.graph.copy(nodes=halo(concept, depth))
    p = (n for n in g.nodes if n.id in concept.graph.properties)
    p = [n.id for n in reversed(sorted(p, key=lambda n: getattr(n, centrality)))]
    return p

class Relation(Edge):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        ' A relation between two concepts, with an optional context.\n            For example, "Felix is-a cat" is in the "media" context, "tiger is-a cat" in "nature".\n        '
        self.context = kwargs.pop('context', None)
        Edge.__init__(self, *args, **kwargs)
COMMONALITY = (lambda concept: concept.properties, lambda edge: 1 - int(edge.context == 'properties' and edge.type != 'is-opposite-of'))

class Commonsense(Graph):

    def __init__(self, data=os.path.join(MODULE, 'commonsense.csv'), **kwargs):
        if False:
            return 10
        ' A semantic network of commonsense, using different relation types:\n            - is-a,\n            - is-part-of,\n            - is-opposite-of,\n            - is-property-of,\n            - is-related-to,\n            - is-same-as,\n            - is-effect-of.\n        '
        Graph.__init__(self, **kwargs)
        self._properties = None
        if data is not None:
            s = open(data, encoding='utf-8').read()
            s = s.strip(BOM_UTF8)
            s = ((v.strip('"') for v in r.split(',')) for r in s.splitlines())
            for (concept1, relation, concept2, context, weight) in s:
                self.add_edge(concept1, concept2, type=relation, context=context, weight=min(int(weight) * 0.1, 1.0))

    @property
    def concepts(self):
        if False:
            return 10
        return self.nodes

    @property
    def relations(self):
        if False:
            while True:
                i = 10
        return self.edges

    @property
    def properties(self):
        if False:
            for i in range(10):
                print('nop')
        ' Yields all concepts that are properties (i.e., adjectives).\n            For example: "cold is-property-of winter" => "cold".\n        '
        if self._properties is None:
            self._properties = (e for e in self.edges if e.context == 'properties')
            self._properties = set(chain(*((e.node1.id, e.node2.id) for e in self._properties)))
        return self._properties

    def add_node(self, id, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Returns a Concept (Node subclass).\n        '
        self._properties = None
        kwargs.setdefault('base', Concept)
        return Graph.add_node(self, id, *args, **kwargs)

    def add_edge(self, id1, id2, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Returns a Relation between two concepts (Edge subclass).\n        '
        self._properties = None
        kwargs.setdefault('base', Relation)
        return Graph.add_edge(self, id1, id2, *args, **kwargs)

    def remove(self, x):
        if False:
            i = 10
            return i + 15
        self._properties = None
        Graph.remove(self, x)

    def similarity(self, concept1, concept2, k=3, heuristic=COMMONALITY):
        if False:
            i = 10
            return i + 15
        ' Returns the similarity of the given concepts,\n            by cross-comparing shortest path distance between k concept properties.\n            A given concept can also be a flat list of properties, e.g. ["creepy"].\n            The given heuristic is a tuple of two functions:\n            1) function(concept) returns a list of salient properties,\n            2) function(edge) returns the cost for traversing this edge (0.0-1.0).\n        '
        if isinstance(concept1, str):
            concept1 = self[concept1]
        if isinstance(concept2, str):
            concept2 = self[concept2]
        if isinstance(concept1, Node):
            concept1 = heuristic[0](concept1)
        if isinstance(concept2, Node):
            concept2 = heuristic[0](concept2)
        if isinstance(concept1, list):
            concept1 = [isinstance(n, Node) and n or self[n] for n in concept1]
        if isinstance(concept2, list):
            concept2 = [isinstance(n, Node) and n or self[n] for n in concept2]
        h = lambda id1, id2: heuristic[1](self.edge(id1, id2))
        w = 0.0
        for p1 in concept1[:k]:
            for p2 in concept2[:k]:
                p = self.shortest_path(p1, p2, heuristic=h)
                w += 1.0 / (p is None and 10000000000.0 or len(p))
        return w / k

    def nearest_neighbors(self, concept, concepts=[], k=3):
        if False:
            print('Hello World!')
        ' Returns the k most similar concepts from the given list.\n        '
        return sorted(concepts, key=lambda candidate: self.similarity(concept, candidate, k), reverse=True)
    similar = neighbors = nn = nearest_neighbors

    def taxonomy(self, concept, depth=3, fringe=2):
        if False:
            return 10
        ' Returns a list of concepts that are descendants of the given concept, using "is-a" relations.\n            Creates a subgraph of "is-a" related concepts up to the given depth,\n            then takes the fringe (i.e., leaves) of the subgraph.\n        '

        def traversable(node, edge):
            if False:
                return 10
            return edge.node2 == node and edge.type == 'is-a'
        if not isinstance(concept, Node):
            concept = self[concept]
        g = self.copy(nodes=concept.flatten(depth, traversable))
        g = g.fringe(depth=fringe)
        g = [self[n.id] for n in g if n != concept]
        return g
    field = semantic_field = taxonomy

def download(path=os.path.join(MODULE, 'commonsense.csv'), threshold=50):
    if False:
        i = 10
        return i + 15
    ' Downloads commonsense data from http://nodebox.net/perception.\n        Saves the data as commonsense.csv which can be the input for Commonsense.load().\n    '
    s = 'http://nodebox.net/perception?format=txt&robots=1'
    s = urlopen(s).read()
    s = s.decode('utf-8')
    s = s.replace("\\'", "'")
    a = {}
    for r in ([v.strip("'") for v in r.split(', ')] for r in s.split('\n')):
        if len(r) == 7:
            a.setdefault(r[-2], []).append(r)
    a = sorted(a.items(), key=cmp_to_key(lambda v1, v2: len(v2[1]) - len(v1[1])))
    r = {}
    for (author, relations) in a:
        if author == '' or author.startswith('robots@'):
            continue
        if len(relations) < threshold:
            break
        relations = sorted(relations, key=cmp_to_key(lambda r1, r2: r1[-1] > r2[-1]))
        for (concept1, relation, concept2, context, weight, author, date) in relations:
            id = (concept1, relation, concept2)
            if id not in r:
                r[id] = [None, 0]
            if r[id][0] is None and context is not None:
                r[id][0] = context
    for (author, relations) in a:
        for (concept1, relation, concept2, context, weight, author, date) in relations:
            id = (concept1, relation, concept2)
            if id in r:
                r[id][1] += int(weight)
    s = []
    for ((concept1, relation, concept2), (context, weight)) in r.items():
        s.append('"%s","%s","%s","%s",%s' % (concept1, relation, concept2, context, weight))
    f = open(path, 'w', encoding='utf-8')
    f.write(BOM_UTF8)
    f.write('\n'.join(s))
    f.close()

def json():
    if False:
        for i in range(10):
            print('nop')
    ' Returns a JSON-string with the data from commonsense.csv.\n        Each relation is encoded as a [concept1, relation, concept2, context, weight] list.\n    '
    f = lambda s: s.replace("'", "\\'").encode('utf-8')
    s = []
    g = Commonsense()
    for e in g.edges:
        s.append("\n\t['%s', '%s', '%s', '%s', %.2f]" % (f(e.node1.id), f(e.type), f(e.node2.id), f(e.context), e.weight))
    return 'commonsense = [%s];' % ', '.join(s)