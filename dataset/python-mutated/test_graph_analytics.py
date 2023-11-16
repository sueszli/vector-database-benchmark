from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from . import util
import unittest
import pandas as pd
from pandas.util.testing import assert_frame_equal
import turicreate as tc
from turicreate._connect.main import get_unity
from turicreate.toolkits._main import ToolkitError
from turicreate.data_structures.sgraph import SGraph
from turicreate.data_structures.sframe import SFrame
import sys
if sys.version_info.major == 3:
    unittest.TestCase.assertItemsEqual = unittest.TestCase.assertCountEqual
dataset_server = 'http://testdatasets.s3-website-us-west-2.amazonaws.com/'

class GraphAnalyticsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        url = dataset_server + 'p2p-Gnutella04.txt.gz'
        cls.graph = tc.load_sgraph(url, format='snap')

    def __test_model_save_load_helper__(self, model):
        if False:
            for i in range(10):
                print('nop')
        with util.TempDirectory() as f:
            model.save(f)
            m2 = tc.load_model(f)
            self.assertItemsEqual(model._list_fields(), m2._list_fields())
            for key in model._list_fields():
                if type(model._get(key)) is SGraph:
                    self.assertItemsEqual(model._get(key).summary(), m2._get(key).summary())
                    self.assertItemsEqual(model._get(key).get_fields(), m2._get(key).get_fields())
                elif type(model._get(key)) is SFrame:
                    sf1 = model._get(key)
                    sf2 = m2._get(key)
                    self.assertEqual(len(sf1), len(sf2))
                    self.assertItemsEqual(sf1.column_names(), sf2.column_names())
                    df1 = sf1.to_dataframe()
                    print(df1)
                    df2 = sf2.to_dataframe()
                    print(df2)
                    df1 = df1.set_index(df1.columns[0])
                    df2 = df2.set_index(df2.columns[0])
                    assert_frame_equal(df1, df2)
                elif type(model._get(key)) is pd.DataFrame:
                    assert_frame_equal(model._get(key), m2._get(key))
                else:
                    self.assertEqual(model._get(key), m2._get(key))

    def test_degree_count(self):
        if False:
            while True:
                i = 10
        if 'degree_count' in get_unity().list_toolkit_functions():
            m = tc.degree_counting.create(self.graph)
            m.summary()
            self.__test_model_save_load_helper__(m)
            g = m.graph
            expected_out_deg = g.edges.groupby('__src_id', {'expected': tc.aggregate.COUNT})
            expected_out_deg = expected_out_deg.join(g.vertices[['__id']], on={'__src_id': '__id'}, how='right').fillna('expected', 0)
            expected_out_deg = expected_out_deg.sort('__src_id')['expected']
            expected_in_deg = g.edges.groupby('__dst_id', {'expected': tc.aggregate.COUNT})
            expected_in_deg = expected_in_deg.join(g.vertices[['__id']], on={'__dst_id': '__id'}, how='right').fillna('expected', 0)
            expected_in_deg = expected_in_deg.sort('__dst_id')['expected']
            sf = g.vertices.sort('__id')
            actual_out_deg = sf['out_degree']
            actual_in_deg = sf['in_degree']
            actual_all_deg = sf['total_degree']
            self.assertEqual((expected_in_deg - actual_in_deg).sum(), 0)
            self.assertEqual((expected_out_deg - actual_out_deg).sum(), 0)
            self.assertEqual((actual_all_deg - (actual_out_deg + actual_in_deg)).sum(), 0)

    def test_label_propagation(self):
        if False:
            for i in range(10):
                print('nop')
        if 'label_propagation' in get_unity().list_toolkit_functions():
            g = self.graph.copy()
            num_vertices = len(g.vertices)
            num_classes = 2

            def get_label(vid):
                if False:
                    print('Hello World!')
                if vid < 100:
                    return 0
                elif vid > num_vertices - 100:
                    return 1
                else:
                    return None
            g.vertices['label'] = g.vertices['__id'].apply(get_label, int)
            m = tc.label_propagation.create(g, label_field='label')
            m.summary()
            self.__test_model_save_load_helper__(m)
            for row in m.graph.vertices:
                predicted_label = row['predicted_label']
                if predicted_label is None:
                    for k in ['P%d' % i for i in range(num_classes)]:
                        self.assertAlmostEqual(row[k], 1.0 / num_classes)
                else:
                    sum_of_prob = 0.0
                    for k in ['P%d' % i for i in range(num_classes)]:
                        sum_of_prob += row[k]
                        self.assertGreaterEqual(row['P%d' % predicted_label], row[k])
                    self.assertAlmostEqual(sum_of_prob, 1.0)

            def get_edge_weight(vid):
                if False:
                    for i in range(10):
                        print('nop')
                return float(vid) * 10 / num_vertices
            g.edges['weight'] = g.edges['__src_id'].apply(get_edge_weight, float)
            m = tc.label_propagation.create(g, label_field='label', threshold=0.01, weight_field='weight', self_weight=0.5, undirected=True)
            max_iter = 3
            m = tc.label_propagation.create(g, label_field='label', threshold=1e-10, max_iterations=max_iter)
            self.assertEqual(m.num_iterations, max_iter)
            g = g.add_vertices(tc.SFrame({'__id': [-1]}))
            m = tc.label_propagation.create(g, label_field='label', threshold=1e-10, max_iterations=max_iter)
            result = m.graph.vertices
            self.assertEqual(result[result['__id'] == -1]['predicted_label'][0], None)

    def test_pagerank(self):
        if False:
            for i in range(10):
                print('nop')
        if 'pagerank' in get_unity().list_toolkit_functions():
            m = tc.pagerank.create(self.graph)
            print(m)
            m.summary()
            self.assertEqual((m.pagerank.num_rows(), m.pagerank.num_columns()), (self.graph.summary()['num_vertices'], 3))
            self.assertEqual(int(m.pagerank['pagerank'].sum()), 2727)
            self.__test_model_save_load_helper__(m)
            m2 = tc.pagerank.create(self.graph, reset_probability=0.5)
            print(m2)
            self.assertEqual((m2.pagerank.num_rows(), m2.pagerank.num_columns()), (self.graph.summary()['num_vertices'], 3))
            self.assertAlmostEqual(m2.pagerank['pagerank'].sum(), 7087.08, delta=0.01)
            with self.assertRaises(Exception):
                assert_frame_equal(m.pagerank.topk('pagerank'), m2.pagerank.topk('pagerank'))
            pr_out = m2['pagerank']
            with self.assertRaises(Exception):
                assert_frame_equal(m.pagerank.topk('pagerank'), pr_out.topk('pagerank'))
            self.__test_model_save_load_helper__(m2)

    def test_triangle_counting(self):
        if False:
            print('Hello World!')
        if 'triangle_counting' in get_unity().list_toolkit_functions():
            m = tc.triangle_counting.create(self.graph)
            print(m)
            m.summary()
            self.__test_model_save_load_helper__(m)
            self.assertEqual(m.num_triangles, 934)

    def test_connected_component(self):
        if False:
            i = 10
            return i + 15
        if 'connected_component' in get_unity().list_toolkit_functions():
            m = tc.connected_components.create(self.graph)
            print(m)
            m.summary()
            print(m.component_id)
            print(m.component_size)
            self.assertEqual(m.component_size.num_rows(), 1)
            self.__test_model_save_load_helper__(m)

    def test_graph_coloring(self):
        if False:
            return 10
        if 'graph_coloring' in get_unity().list_toolkit_functions():
            m = tc.graph_coloring.create(self.graph)
            print(m)
            m.summary()
            self.__test_model_save_load_helper__(m)

    def test_kcore(self):
        if False:
            while True:
                i = 10
        if 'kcore' in get_unity().list_toolkit_functions():
            m = tc.kcore.create(self.graph)
            print(m)
            m.summary()
            biggest_core = m.core_id.groupby('core_id', tc.aggregate.COUNT).topk('Count').head(1)
            self.assertEqual(biggest_core['core_id'][0], 6)
            self.assertEqual(biggest_core['Count'][0], 4492)
            self.__test_model_save_load_helper__(m)

    def test_shortest_path(self):
        if False:
            while True:
                i = 10
        if 'sssp' in get_unity().list_toolkit_functions():
            m = tc.shortest_path.create(self.graph, source_vid=0)
            print(m)
            m.summary()
            self.__test_model_save_load_helper__(m)
            m2 = tc.shortest_path.create(self.graph, source_vid=0)
            print(m2)
            self.__test_model_save_load_helper__(m2)
            chain_graph = tc.SGraph().add_edges([tc.Edge(i, i + 1) for i in range(10)])
            m3 = tc.shortest_path.create(chain_graph, source_vid=0)
            for i in range(10):
                self.assertSequenceEqual(m3.get_path(i), [(j, float(j)) for j in range(i + 1)])
            star_graph = tc.SGraph().add_edges([tc.Edge(0, i + 1) for i in range(10)])
            m4 = tc.shortest_path.create(star_graph, source_vid=0)
            for i in range(1, 11):
                self.assertSequenceEqual(m4.get_path(i), [(0, 0.0), (i, 1.0)])
            star_graph.vertices['distance'] = 0
            m5 = tc.shortest_path.create(star_graph, source_vid=0)
            for i in range(1, 11):
                self.assertSequenceEqual(m5.get_path(i), [(0, 0.0), (i, 1.0)])

    def test_compute_shortest_path(self):
        if False:
            for i in range(10):
                print('nop')
        edge_src_ids = ['src1', 'src2', 'a', 'b', 'c']
        edge_dst_ids = ['a', 'b', 'dst', 'c', 'dst']
        edges = tc.SFrame({'__src_id': edge_src_ids, '__dst_id': edge_dst_ids})
        g = tc.SGraph().add_edges(edges)
        res = tc.shortest_path._compute_shortest_path(g, ['src1', 'src2'], 'dst')
        self.assertEqual(res, [['src1', 'a', 'dst']])
        res = tc.shortest_path._compute_shortest_path(g, 'src2', 'dst')
        self.assertEqual(res, [['src2', 'b', 'c', 'dst']])
        edge_src_ids = [0, 1, 2, 3, 4]
        edge_dst_ids = [2, 3, 5, 4, 5]
        edge_weights = [1, 0.1, 1, 0.1, 0.1]
        g = tc.SFrame({'__src_id': edge_src_ids, '__dst_id': edge_dst_ids, 'weights': edge_weights})
        g = tc.SGraph(edges=g)
        t = tc.shortest_path._compute_shortest_path(g, [0, 1], [5], 'weights')
        self.assertEqual(t, [[1, 3, 4, 5]])