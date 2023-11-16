"""
Graph module tests
"""
import os
import itertools
import tempfile
import unittest
from txtai.embeddings import Embeddings
from txtai.graph import Graph, GraphFactory

class TestGraph(unittest.TestCase):
    """
    Graph tests.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        '\n        Initialize test data.\n        '
        cls.data = ['US tops 5 million confirmed virus cases', "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg", 'Beijing mobilises invasion craft along coast as Taiwan tensions escalate', 'The National Park Service warns against sacrificing slower friends in a bear attack', 'Maine man wins $1M from $25 lottery ticket', 'Make huge profits without work, earn up to $100,000 a day']
        cls.config = {'path': 'sentence-transformers/nli-mpnet-base-v2', 'content': True, 'functions': [{'name': 'graph', 'function': 'graph.attribute'}], 'expressions': [{'name': 'category', 'expression': "graph(indexid, 'category')"}, {'name': 'topic', 'expression': "graph(indexid, 'topic')"}, {'name': 'topicrank', 'expression': "graph(indexid, 'topicrank')"}], 'graph': {'limit': 5, 'minscore': 0.2, 'batchsize': 4, 'approximate': False, 'topics': {'categories': ['News'], 'stopwords': ['the']}}}
        cls.embeddings = Embeddings(cls.config)

    def testAnalysis(self):
        if False:
            print('Hello World!')
        '\n        Test analysis methods\n        '
        self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
        graph = self.embeddings.graph
        centrality = graph.centrality()
        self.assertEqual(list(centrality.keys())[0], 5)
        pagerank = graph.pagerank()
        self.assertEqual(list(pagerank.keys())[0], 5)
        path = graph.showpath(4, 5)
        self.assertEqual(len(path), 2)

    def testCommunity(self):
        if False:
            i = 10
            return i + 15
        '\n        Test community detection\n        '
        self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
        graph = self.embeddings.graph
        graph.config = {'topics': {'algorithm': 'greedy'}}
        graph.addtopics()
        self.assertEqual(sum((len(graph.topics[x]) for x in graph.topics)), 6)
        graph.config = {'topics': {'algorithm': 'lpa'}}
        graph.addtopics()
        self.assertEqual(sum((len(graph.topics[x]) for x in graph.topics)), 4)

    def testCustomBackend(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test resolving a custom backend\n        '
        graph = GraphFactory.create({'backend': 'txtai.graph.NetworkX'})
        graph.initialize()
        self.assertIsNotNone(graph)

    def testCustomBackendNotFound(self):
        if False:
            while True:
                i = 10
        '\n        Test resolving an unresolvable backend\n        '
        with self.assertRaises(ImportError):
            graph = GraphFactory.create({'backend': 'notfound.graph'})
            graph.initialize()

    def testDelete(self):
        if False:
            return 10
        '\n        Test delete\n        '
        self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
        self.embeddings.delete([4])
        graph = self.embeddings.graph
        self.assertEqual(graph.count(), 5)
        self.assertEqual(graph.edgecount(), 1)
        self.assertEqual(sum((len(graph.topics[x]) for x in graph.topics)), 5)
        self.assertEqual(len(graph.categories), 6)

    def testFunction(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test running graph functions with SQL\n        '
        self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
        result = self.embeddings.search('select category, topic, topicrank from txtai where id = 0', 1)[0]
        self.assertIsNotNone(result['category'])
        self.assertIsNotNone(result['topic'])
        self.assertIsNotNone(result['topicrank'])

    def testFunctionReindex(self):
        if False:
            print('Hello World!')
        '\n        Test running graph functions with SQL after reindex\n        '
        self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
        self.embeddings.reindex(self.embeddings.config)
        result = self.embeddings.search('select category, topic, topicrank from txtai where id = 0', 1)[0]
        self.assertIsNotNone(result['category'])
        self.assertIsNotNone(result['topic'])
        self.assertIsNotNone(result['topicrank'])

    def testIndex(self):
        if False:
            return 10
        '\n        Test index\n        '
        self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
        graph = self.embeddings.graph
        self.assertEqual(graph.count(), 6)
        self.assertEqual(graph.edgecount(), 2)
        self.assertEqual(len(graph.topics), 6)
        self.assertEqual(len(graph.categories), 6)

    def testNotImplemented(self):
        if False:
            while True:
                i = 10
        '\n        Test exceptions for non-implemented methods\n        '
        graph = Graph({})
        self.assertRaises(NotImplementedError, graph.create)
        self.assertRaises(NotImplementedError, graph.count)
        self.assertRaises(NotImplementedError, graph.scan, None)
        self.assertRaises(NotImplementedError, graph.node, None)
        self.assertRaises(NotImplementedError, graph.addnode, None)
        self.assertRaises(NotImplementedError, graph.removenode, None)
        self.assertRaises(NotImplementedError, graph.hasnode, None)
        self.assertRaises(NotImplementedError, graph.attribute, None, None)
        self.assertRaises(NotImplementedError, graph.addattribute, None, None, None)
        self.assertRaises(NotImplementedError, graph.removeattribute, None, None)
        self.assertRaises(NotImplementedError, graph.edgecount)
        self.assertRaises(NotImplementedError, graph.edges, None)
        self.assertRaises(NotImplementedError, graph.addedge, None, None)
        self.assertRaises(NotImplementedError, graph.hasedge, None, None)
        self.assertRaises(NotImplementedError, graph.centrality)
        self.assertRaises(NotImplementedError, graph.pagerank)
        self.assertRaises(NotImplementedError, graph.showpath, None, None)
        self.assertRaises(NotImplementedError, graph.communities, None)
        self.assertRaises(NotImplementedError, graph.loadgraph, None)
        self.assertRaises(NotImplementedError, graph.savegraph, None)

    def testResetTopics(self):
        if False:
            i = 10
            return i + 15
        '\n        Test resetting of topics\n        '
        self.embeddings.index([(1, 'text', None)])
        self.embeddings.upsert([(1, 'graph', None)])
        self.assertEqual(list(self.embeddings.graph.topics.keys()), ['graph'])

    def testSave(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test save\n        '
        self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
        index = os.path.join(tempfile.gettempdir(), 'graph')
        self.embeddings.save(index)
        self.embeddings.load(index)
        graph = self.embeddings.graph
        self.assertEqual(graph.count(), 6)
        self.assertEqual(graph.edgecount(), 2)
        self.assertEqual(sum((len(graph.topics[x]) for x in graph.topics)), 6)
        self.assertEqual(len(graph.categories), 6)

    def testSimple(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test creating a simple graph\n        '
        graph = GraphFactory.create({'topics': {}})
        graph.initialize()
        for x in range(5):
            graph.addnode(x)
        for (x, y) in itertools.combinations(range(5), 2):
            graph.addedge(x, y)
        self.assertEqual(graph.count(), 5)
        self.assertEqual(graph.edgecount(), 10)
        self.assertIsNone(graph.edges(100))
        graph.addtopics()
        self.assertEqual(len(graph.topics), 5)

    def testSubindex(self):
        if False:
            i = 10
            return i + 15
        '\n        Test subindex\n        '
        data = [(uid, text, None) for (uid, text) in enumerate(self.data)]
        embeddings = Embeddings({'content': True, 'functions': [{'name': 'graph', 'function': 'indexes.index1.graph.attribute'}], 'expressions': [{'name': 'category', 'expression': "graph(indexid, 'category')"}, {'name': 'topic', 'expression': "graph(indexid, 'topic')"}, {'name': 'topicrank', 'expression': "graph(indexid, 'topicrank')"}], 'indexes': {'index1': {'path': 'sentence-transformers/nli-mpnet-base-v2', 'graph': {'limit': 5, 'minscore': 0.2, 'batchsize': 4, 'approximate': False, 'topics': {'categories': ['News'], 'stopwords': ['the']}}}}})
        embeddings.index(data)
        result = embeddings.search('select id, category, topic, topicrank from txtai where id = 0', 1)[0]
        self.assertIsNotNone(result['category'])
        self.assertIsNotNone(result['topic'])
        self.assertIsNotNone(result['topicrank'])
        data[0] = (0, 'Feel good story: lottery winner announced', None)
        embeddings.upsert([data[0]])
        result = embeddings.search('select id, category, topic, topicrank from txtai where id = 0', 1)[0]
        self.assertIsNotNone(result['category'])
        self.assertIsNotNone(result['topic'])
        self.assertIsNotNone(result['topicrank'])

    def testUpsert(self):
        if False:
            while True:
                i = 10
        '\n        Test upsert\n        '
        self.embeddings.upsert([(0, {'text': 'Canadian ice shelf collapses'.split()}, None)])
        graph = self.embeddings.graph
        self.assertEqual(graph.count(), 6)
        self.assertEqual(graph.edgecount(), 2)
        self.assertEqual(sum((len(graph.topics[x]) for x in graph.topics)), 6)
        self.assertEqual(len(graph.categories), 6)