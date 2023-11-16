import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal

class BaseGraphML:

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        cls.simple_directed_data = '<?xml version="1.0" encoding="UTF-8"?>\n<!-- This file was written by the JAVA GraphML Library.-->\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G" edgedefault="directed">\n    <node id="n0"/>\n    <node id="n1"/>\n    <node id="n2"/>\n    <node id="n3"/>\n    <node id="n4"/>\n    <node id="n5"/>\n    <node id="n6"/>\n    <node id="n7"/>\n    <node id="n8"/>\n    <node id="n9"/>\n    <node id="n10"/>\n    <edge id="foo" source="n0" target="n2"/>\n    <edge source="n1" target="n2"/>\n    <edge source="n2" target="n3"/>\n    <edge source="n3" target="n5"/>\n    <edge source="n3" target="n4"/>\n    <edge source="n4" target="n6"/>\n    <edge source="n6" target="n5"/>\n    <edge source="n5" target="n7"/>\n    <edge source="n6" target="n8"/>\n    <edge source="n8" target="n7"/>\n    <edge source="n8" target="n9"/>\n  </graph>\n</graphml>'
        cls.simple_directed_graph = nx.DiGraph()
        cls.simple_directed_graph.add_node('n10')
        cls.simple_directed_graph.add_edge('n0', 'n2', id='foo')
        cls.simple_directed_graph.add_edge('n0', 'n2')
        cls.simple_directed_graph.add_edges_from([('n1', 'n2'), ('n2', 'n3'), ('n3', 'n5'), ('n3', 'n4'), ('n4', 'n6'), ('n6', 'n5'), ('n5', 'n7'), ('n6', 'n8'), ('n8', 'n7'), ('n8', 'n9')])
        cls.simple_directed_fh = io.BytesIO(cls.simple_directed_data.encode('UTF-8'))
        cls.attribute_data = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n      xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n        http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key id="d0" for="node" attr.name="color" attr.type="string">\n    <default>yellow</default>\n  </key>\n  <key id="d1" for="edge" attr.name="weight" attr.type="double"/>\n  <graph id="G" edgedefault="directed">\n    <node id="n0">\n      <data key="d0">green</data>\n    </node>\n    <node id="n1"/>\n    <node id="n2">\n      <data key="d0">blue</data>\n    </node>\n    <node id="n3">\n      <data key="d0">red</data>\n    </node>\n    <node id="n4"/>\n    <node id="n5">\n      <data key="d0">turquoise</data>\n    </node>\n    <edge id="e0" source="n0" target="n2">\n      <data key="d1">1.0</data>\n    </edge>\n    <edge id="e1" source="n0" target="n1">\n      <data key="d1">1.0</data>\n    </edge>\n    <edge id="e2" source="n1" target="n3">\n      <data key="d1">2.0</data>\n    </edge>\n    <edge id="e3" source="n3" target="n2"/>\n    <edge id="e4" source="n2" target="n4"/>\n    <edge id="e5" source="n3" target="n5"/>\n    <edge id="e6" source="n5" target="n4">\n      <data key="d1">1.1</data>\n    </edge>\n  </graph>\n</graphml>\n'
        cls.attribute_graph = nx.DiGraph(id='G')
        cls.attribute_graph.graph['node_default'] = {'color': 'yellow'}
        cls.attribute_graph.add_node('n0', color='green')
        cls.attribute_graph.add_node('n2', color='blue')
        cls.attribute_graph.add_node('n3', color='red')
        cls.attribute_graph.add_node('n4')
        cls.attribute_graph.add_node('n5', color='turquoise')
        cls.attribute_graph.add_edge('n0', 'n2', id='e0', weight=1.0)
        cls.attribute_graph.add_edge('n0', 'n1', id='e1', weight=1.0)
        cls.attribute_graph.add_edge('n1', 'n3', id='e2', weight=2.0)
        cls.attribute_graph.add_edge('n3', 'n2', id='e3')
        cls.attribute_graph.add_edge('n2', 'n4', id='e4')
        cls.attribute_graph.add_edge('n3', 'n5', id='e5')
        cls.attribute_graph.add_edge('n5', 'n4', id='e6', weight=1.1)
        cls.attribute_fh = io.BytesIO(cls.attribute_data.encode('UTF-8'))
        cls.node_attribute_default_data = '<?xml version="1.0" encoding="UTF-8"?>\n        <graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n              xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n              xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n                http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n          <key id="d0" for="node" attr.name="boolean_attribute" attr.type="boolean"><default>false</default></key>\n          <key id="d1" for="node" attr.name="int_attribute" attr.type="int"><default>0</default></key>\n          <key id="d2" for="node" attr.name="long_attribute" attr.type="long"><default>0</default></key>\n          <key id="d3" for="node" attr.name="float_attribute" attr.type="float"><default>0.0</default></key>\n          <key id="d4" for="node" attr.name="double_attribute" attr.type="double"><default>0.0</default></key>\n          <key id="d5" for="node" attr.name="string_attribute" attr.type="string"><default>Foo</default></key>\n          <graph id="G" edgedefault="directed">\n            <node id="n0"/>\n            <node id="n1"/>\n            <edge id="e0" source="n0" target="n1"/>\n          </graph>\n        </graphml>\n        '
        cls.node_attribute_default_graph = nx.DiGraph(id='G')
        cls.node_attribute_default_graph.graph['node_default'] = {'boolean_attribute': False, 'int_attribute': 0, 'long_attribute': 0, 'float_attribute': 0.0, 'double_attribute': 0.0, 'string_attribute': 'Foo'}
        cls.node_attribute_default_graph.add_node('n0')
        cls.node_attribute_default_graph.add_node('n1')
        cls.node_attribute_default_graph.add_edge('n0', 'n1', id='e0')
        cls.node_attribute_default_fh = io.BytesIO(cls.node_attribute_default_data.encode('UTF-8'))
        cls.attribute_named_key_ids_data = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key id="edge_prop" for="edge" attr.name="edge_prop" attr.type="string"/>\n  <key id="prop2" for="node" attr.name="prop2" attr.type="string"/>\n  <key id="prop1" for="node" attr.name="prop1" attr.type="string"/>\n  <graph edgedefault="directed">\n    <node id="0">\n      <data key="prop1">val1</data>\n      <data key="prop2">val2</data>\n    </node>\n    <node id="1">\n      <data key="prop1">val_one</data>\n      <data key="prop2">val2</data>\n    </node>\n    <edge source="0" target="1">\n      <data key="edge_prop">edge_value</data>\n    </edge>\n  </graph>\n</graphml>\n'
        cls.attribute_named_key_ids_graph = nx.DiGraph()
        cls.attribute_named_key_ids_graph.add_node('0', prop1='val1', prop2='val2')
        cls.attribute_named_key_ids_graph.add_node('1', prop1='val_one', prop2='val2')
        cls.attribute_named_key_ids_graph.add_edge('0', '1', edge_prop='edge_value')
        fh = io.BytesIO(cls.attribute_named_key_ids_data.encode('UTF-8'))
        cls.attribute_named_key_ids_fh = fh
        cls.attribute_numeric_type_data = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key attr.name="weight" attr.type="double" for="node" id="d1" />\n  <key attr.name="weight" attr.type="double" for="edge" id="d0" />\n  <graph edgedefault="directed">\n    <node id="n0">\n      <data key="d1">1</data>\n    </node>\n    <node id="n1">\n      <data key="d1">2.0</data>\n    </node>\n    <edge source="n0" target="n1">\n      <data key="d0">1</data>\n    </edge>\n    <edge source="n1" target="n0">\n      <data key="d0">k</data>\n    </edge>\n    <edge source="n1" target="n1">\n      <data key="d0">1.0</data>\n    </edge>\n  </graph>\n</graphml>\n'
        cls.attribute_numeric_type_graph = nx.DiGraph()
        cls.attribute_numeric_type_graph.add_node('n0', weight=1)
        cls.attribute_numeric_type_graph.add_node('n1', weight=2.0)
        cls.attribute_numeric_type_graph.add_edge('n0', 'n1', weight=1)
        cls.attribute_numeric_type_graph.add_edge('n1', 'n1', weight=1.0)
        fh = io.BytesIO(cls.attribute_numeric_type_data.encode('UTF-8'))
        cls.attribute_numeric_type_fh = fh
        cls.simple_undirected_data = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G">\n    <node id="n0"/>\n    <node id="n1"/>\n    <node id="n2"/>\n    <node id="n10"/>\n    <edge id="foo" source="n0" target="n2"/>\n    <edge source="n1" target="n2"/>\n    <edge source="n2" target="n3"/>\n  </graph>\n</graphml>'
        cls.simple_undirected_graph = nx.Graph()
        cls.simple_undirected_graph.add_node('n10')
        cls.simple_undirected_graph.add_edge('n0', 'n2', id='foo')
        cls.simple_undirected_graph.add_edges_from([('n1', 'n2'), ('n2', 'n3')])
        fh = io.BytesIO(cls.simple_undirected_data.encode('UTF-8'))
        cls.simple_undirected_fh = fh
        cls.undirected_multigraph_data = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G">\n    <node id="n0"/>\n    <node id="n1"/>\n    <node id="n2"/>\n    <node id="n10"/>\n    <edge id="e0" source="n0" target="n2"/>\n    <edge id="e1" source="n1" target="n2"/>\n    <edge id="e2" source="n2" target="n1"/>\n  </graph>\n</graphml>'
        cls.undirected_multigraph = nx.MultiGraph()
        cls.undirected_multigraph.add_node('n10')
        cls.undirected_multigraph.add_edge('n0', 'n2', id='e0')
        cls.undirected_multigraph.add_edge('n1', 'n2', id='e1')
        cls.undirected_multigraph.add_edge('n2', 'n1', id='e2')
        fh = io.BytesIO(cls.undirected_multigraph_data.encode('UTF-8'))
        cls.undirected_multigraph_fh = fh
        cls.undirected_multigraph_no_multiedge_data = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G">\n    <node id="n0"/>\n    <node id="n1"/>\n    <node id="n2"/>\n    <node id="n10"/>\n    <edge id="e0" source="n0" target="n2"/>\n    <edge id="e1" source="n1" target="n2"/>\n    <edge id="e2" source="n2" target="n3"/>\n  </graph>\n</graphml>'
        cls.undirected_multigraph_no_multiedge = nx.MultiGraph()
        cls.undirected_multigraph_no_multiedge.add_node('n10')
        cls.undirected_multigraph_no_multiedge.add_edge('n0', 'n2', id='e0')
        cls.undirected_multigraph_no_multiedge.add_edge('n1', 'n2', id='e1')
        cls.undirected_multigraph_no_multiedge.add_edge('n2', 'n3', id='e2')
        fh = io.BytesIO(cls.undirected_multigraph_no_multiedge_data.encode('UTF-8'))
        cls.undirected_multigraph_no_multiedge_fh = fh
        cls.multigraph_only_ids_for_multiedges_data = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G">\n    <node id="n0"/>\n    <node id="n1"/>\n    <node id="n2"/>\n    <node id="n10"/>\n    <edge source="n0" target="n2"/>\n    <edge id="e1" source="n1" target="n2"/>\n    <edge id="e2" source="n2" target="n1"/>\n  </graph>\n</graphml>'
        cls.multigraph_only_ids_for_multiedges = nx.MultiGraph()
        cls.multigraph_only_ids_for_multiedges.add_node('n10')
        cls.multigraph_only_ids_for_multiedges.add_edge('n0', 'n2')
        cls.multigraph_only_ids_for_multiedges.add_edge('n1', 'n2', id='e1')
        cls.multigraph_only_ids_for_multiedges.add_edge('n2', 'n1', id='e2')
        fh = io.BytesIO(cls.multigraph_only_ids_for_multiedges_data.encode('UTF-8'))
        cls.multigraph_only_ids_for_multiedges_fh = fh

class TestReadGraphML(BaseGraphML):

    def test_read_simple_directed_graphml(self):
        if False:
            i = 10
            return i + 15
        G = self.simple_directed_graph
        H = nx.read_graphml(self.simple_directed_fh)
        assert sorted(G.nodes()) == sorted(H.nodes())
        assert sorted(G.edges()) == sorted(H.edges())
        assert sorted(G.edges(data=True)) == sorted(H.edges(data=True))
        self.simple_directed_fh.seek(0)
        PG = nx.parse_graphml(self.simple_directed_data)
        assert sorted(G.nodes()) == sorted(PG.nodes())
        assert sorted(G.edges()) == sorted(PG.edges())
        assert sorted(G.edges(data=True)) == sorted(PG.edges(data=True))

    def test_read_simple_undirected_graphml(self):
        if False:
            i = 10
            return i + 15
        G = self.simple_undirected_graph
        H = nx.read_graphml(self.simple_undirected_fh)
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        self.simple_undirected_fh.seek(0)
        PG = nx.parse_graphml(self.simple_undirected_data)
        assert nodes_equal(G.nodes(), PG.nodes())
        assert edges_equal(G.edges(), PG.edges())

    def test_read_undirected_multigraph_graphml(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.undirected_multigraph
        H = nx.read_graphml(self.undirected_multigraph_fh)
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        self.undirected_multigraph_fh.seek(0)
        PG = nx.parse_graphml(self.undirected_multigraph_data)
        assert nodes_equal(G.nodes(), PG.nodes())
        assert edges_equal(G.edges(), PG.edges())

    def test_read_undirected_multigraph_no_multiedge_graphml(self):
        if False:
            while True:
                i = 10
        G = self.undirected_multigraph_no_multiedge
        H = nx.read_graphml(self.undirected_multigraph_no_multiedge_fh)
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        self.undirected_multigraph_no_multiedge_fh.seek(0)
        PG = nx.parse_graphml(self.undirected_multigraph_no_multiedge_data)
        assert nodes_equal(G.nodes(), PG.nodes())
        assert edges_equal(G.edges(), PG.edges())

    def test_read_undirected_multigraph_only_ids_for_multiedges_graphml(self):
        if False:
            print('Hello World!')
        G = self.multigraph_only_ids_for_multiedges
        H = nx.read_graphml(self.multigraph_only_ids_for_multiedges_fh)
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        self.multigraph_only_ids_for_multiedges_fh.seek(0)
        PG = nx.parse_graphml(self.multigraph_only_ids_for_multiedges_data)
        assert nodes_equal(G.nodes(), PG.nodes())
        assert edges_equal(G.edges(), PG.edges())

    def test_read_attribute_graphml(self):
        if False:
            i = 10
            return i + 15
        G = self.attribute_graph
        H = nx.read_graphml(self.attribute_fh)
        assert nodes_equal(G.nodes(True), sorted(H.nodes(data=True)))
        ge = sorted(G.edges(data=True))
        he = sorted(H.edges(data=True))
        for (a, b) in zip(ge, he):
            assert a == b
        self.attribute_fh.seek(0)
        PG = nx.parse_graphml(self.attribute_data)
        assert sorted(G.nodes(True)) == sorted(PG.nodes(data=True))
        ge = sorted(G.edges(data=True))
        he = sorted(PG.edges(data=True))
        for (a, b) in zip(ge, he):
            assert a == b

    def test_node_default_attribute_graphml(self):
        if False:
            i = 10
            return i + 15
        G = self.node_attribute_default_graph
        H = nx.read_graphml(self.node_attribute_default_fh)
        assert G.graph['node_default'] == H.graph['node_default']

    def test_directed_edge_in_undirected(self):
        if False:
            return 10
        s = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G">\n    <node id="n0"/>\n    <node id="n1"/>\n    <node id="n2"/>\n    <edge source="n0" target="n1"/>\n    <edge source="n1" target="n2" directed=\'true\'/>\n  </graph>\n</graphml>'
        fh = io.BytesIO(s.encode('UTF-8'))
        pytest.raises(nx.NetworkXError, nx.read_graphml, fh)
        pytest.raises(nx.NetworkXError, nx.parse_graphml, s)

    def test_undirected_edge_in_directed(self):
        if False:
            for i in range(10):
                print('nop')
        s = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G" edgedefault=\'directed\'>\n    <node id="n0"/>\n    <node id="n1"/>\n    <node id="n2"/>\n    <edge source="n0" target="n1"/>\n    <edge source="n1" target="n2" directed=\'false\'/>\n  </graph>\n</graphml>'
        fh = io.BytesIO(s.encode('UTF-8'))
        pytest.raises(nx.NetworkXError, nx.read_graphml, fh)
        pytest.raises(nx.NetworkXError, nx.parse_graphml, s)

    def test_key_raise(self):
        if False:
            i = 10
            return i + 15
        s = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key id="d0" for="node" attr.name="color" attr.type="string">\n    <default>yellow</default>\n  </key>\n  <key id="d1" for="edge" attr.name="weight" attr.type="double"/>\n  <graph id="G" edgedefault="directed">\n    <node id="n0">\n      <data key="d0">green</data>\n    </node>\n    <node id="n1"/>\n    <node id="n2">\n      <data key="d0">blue</data>\n    </node>\n    <edge id="e0" source="n0" target="n2">\n      <data key="d2">1.0</data>\n    </edge>\n  </graph>\n</graphml>\n'
        fh = io.BytesIO(s.encode('UTF-8'))
        pytest.raises(nx.NetworkXError, nx.read_graphml, fh)
        pytest.raises(nx.NetworkXError, nx.parse_graphml, s)

    def test_hyperedge_raise(self):
        if False:
            i = 10
            return i + 15
        s = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key id="d0" for="node" attr.name="color" attr.type="string">\n    <default>yellow</default>\n  </key>\n  <key id="d1" for="edge" attr.name="weight" attr.type="double"/>\n  <graph id="G" edgedefault="directed">\n    <node id="n0">\n      <data key="d0">green</data>\n    </node>\n    <node id="n1"/>\n    <node id="n2">\n      <data key="d0">blue</data>\n    </node>\n    <hyperedge id="e0" source="n0" target="n2">\n       <endpoint node="n0"/>\n       <endpoint node="n1"/>\n       <endpoint node="n2"/>\n    </hyperedge>\n  </graph>\n</graphml>\n'
        fh = io.BytesIO(s.encode('UTF-8'))
        pytest.raises(nx.NetworkXError, nx.read_graphml, fh)
        pytest.raises(nx.NetworkXError, nx.parse_graphml, s)

    def test_multigraph_keys(self):
        if False:
            print('Hello World!')
        s = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G" edgedefault="directed">\n    <node id="n0"/>\n    <node id="n1"/>\n    <edge id="e0" source="n0" target="n1"/>\n    <edge id="e1" source="n0" target="n1"/>\n  </graph>\n</graphml>\n'
        fh = io.BytesIO(s.encode('UTF-8'))
        G = nx.read_graphml(fh)
        expected = [('n0', 'n1', 'e0'), ('n0', 'n1', 'e1')]
        assert sorted(G.edges(keys=True)) == expected
        fh.seek(0)
        H = nx.parse_graphml(s)
        assert sorted(H.edges(keys=True)) == expected

    def test_preserve_multi_edge_data(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that data and keys of edges are preserved on consequent\n        write and reads\n        '
        G = nx.MultiGraph()
        G.add_node(1)
        G.add_node(2)
        G.add_edges_from([(1, 2), (1, 2, {'key': 'data_key1'}), (1, 2, {'id': 'data_id2'}), (1, 2, {'key': 'data_key3', 'id': 'data_id3'}), (1, 2, 103, {'key': 'data_key4'}), (1, 2, 104, {'id': 'data_id5'}), (1, 2, 105, {'key': 'data_key6', 'id': 'data_id7'})])
        fh = io.BytesIO()
        nx.write_graphml(G, fh)
        fh.seek(0)
        H = nx.read_graphml(fh, node_type=int)
        assert edges_equal(G.edges(data=True, keys=True), H.edges(data=True, keys=True))
        assert G._adj == H._adj
        Gadj = {str(node): {str(nbr): {str(ekey): dd for (ekey, dd) in key_dict.items()} for (nbr, key_dict) in nbr_dict.items()} for (node, nbr_dict) in G._adj.items()}
        fh.seek(0)
        HH = nx.read_graphml(fh, node_type=str, edge_key_type=str)
        assert Gadj == HH._adj
        fh.seek(0)
        string_fh = fh.read()
        HH = nx.parse_graphml(string_fh, node_type=str, edge_key_type=str)
        assert Gadj == HH._adj

    def test_yfiles_extension(self):
        if False:
            print('Hello World!')
        data = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xmlns:y="http://www.yworks.com/xml/graphml"\n         xmlns:yed="http://www.yworks.com/xml/yed/3"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <!--Created by yFiles for Java 2.7-->\n  <key for="graphml" id="d0" yfiles.type="resources"/>\n  <key attr.name="url" attr.type="string" for="node" id="d1"/>\n  <key attr.name="description" attr.type="string" for="node" id="d2"/>\n  <key for="node" id="d3" yfiles.type="nodegraphics"/>\n  <key attr.name="Description" attr.type="string" for="graph" id="d4">\n    <default/>\n  </key>\n  <key attr.name="url" attr.type="string" for="edge" id="d5"/>\n  <key attr.name="description" attr.type="string" for="edge" id="d6"/>\n  <key for="edge" id="d7" yfiles.type="edgegraphics"/>\n  <graph edgedefault="directed" id="G">\n    <node id="n0">\n      <data key="d3">\n        <y:ShapeNode>\n          <y:Geometry height="30.0" width="30.0" x="125.0" y="100.0"/>\n          <y:Fill color="#FFCC00" transparent="false"/>\n          <y:BorderStyle color="#000000" type="line" width="1.0"/>\n          <y:NodeLabel alignment="center" autoSizePolicy="content"\n           borderDistance="0.0" fontFamily="Dialog" fontSize="13"\n           fontStyle="plain" hasBackgroundColor="false" hasLineColor="false"\n           height="19.1328125" modelName="internal" modelPosition="c"\n           textColor="#000000" visible="true" width="12.27099609375"\n           x="8.864501953125" y="5.43359375">1</y:NodeLabel>\n          <y:Shape type="rectangle"/>\n        </y:ShapeNode>\n      </data>\n    </node>\n    <node id="n1">\n      <data key="d3">\n        <y:ShapeNode>\n          <y:Geometry height="30.0" width="30.0" x="183.0" y="205.0"/>\n          <y:Fill color="#FFCC00" transparent="false"/>\n          <y:BorderStyle color="#000000" type="line" width="1.0"/>\n          <y:NodeLabel alignment="center" autoSizePolicy="content"\n          borderDistance="0.0" fontFamily="Dialog" fontSize="13"\n          fontStyle="plain" hasBackgroundColor="false" hasLineColor="false"\n          height="19.1328125" modelName="internal" modelPosition="c"\n          textColor="#000000" visible="true" width="12.27099609375"\n          x="8.864501953125" y="5.43359375">2</y:NodeLabel>\n          <y:Shape type="rectangle"/>\n        </y:ShapeNode>\n      </data>\n    </node>\n    <node id="n2">\n      <data key="d6" xml:space="preserve"><![CDATA[description\nline1\nline2]]></data>\n      <data key="d3">\n        <y:GenericNode configuration="com.yworks.flowchart.terminator">\n          <y:Geometry height="40.0" width="80.0" x="950.0" y="286.0"/>\n          <y:Fill color="#E8EEF7" color2="#B7C9E3" transparent="false"/>\n          <y:BorderStyle color="#000000" type="line" width="1.0"/>\n          <y:NodeLabel alignment="center" autoSizePolicy="content"\n          fontFamily="Dialog" fontSize="12" fontStyle="plain"\n          hasBackgroundColor="false" hasLineColor="false" height="17.96875"\n          horizontalTextPosition="center" iconTextGap="4" modelName="custom"\n          textColor="#000000" verticalTextPosition="bottom" visible="true"\n          width="67.984375" x="6.0078125" xml:space="preserve"\n          y="11.015625">3<y:LabelModel>\n          <y:SmartNodeLabelModel distance="4.0"/></y:LabelModel>\n          <y:ModelParameter><y:SmartNodeLabelModelParameter labelRatioX="0.0"\n          labelRatioY="0.0" nodeRatioX="0.0" nodeRatioY="0.0" offsetX="0.0"\n          offsetY="0.0" upX="0.0" upY="-1.0"/></y:ModelParameter></y:NodeLabel>\n        </y:GenericNode>\n      </data>\n    </node>\n    <edge id="e0" source="n0" target="n1">\n      <data key="d7">\n        <y:PolyLineEdge>\n          <y:Path sx="0.0" sy="0.0" tx="0.0" ty="0.0"/>\n          <y:LineStyle color="#000000" type="line" width="1.0"/>\n          <y:Arrows source="none" target="standard"/>\n          <y:BendStyle smoothed="false"/>\n        </y:PolyLineEdge>\n      </data>\n    </edge>\n  </graph>\n  <data key="d0">\n    <y:Resources/>\n  </data>\n</graphml>\n'
        fh = io.BytesIO(data.encode('UTF-8'))
        G = nx.read_graphml(fh, force_multigraph=True)
        assert list(G.edges()) == [('n0', 'n1')]
        assert G.has_edge('n0', 'n1', key='e0')
        assert G.nodes['n0']['label'] == '1'
        assert G.nodes['n1']['label'] == '2'
        assert G.nodes['n2']['label'] == '3'
        assert G.nodes['n0']['shape_type'] == 'rectangle'
        assert G.nodes['n1']['shape_type'] == 'rectangle'
        assert G.nodes['n2']['shape_type'] == 'com.yworks.flowchart.terminator'
        assert G.nodes['n2']['description'] == 'description\nline1\nline2'
        fh.seek(0)
        G = nx.read_graphml(fh)
        assert list(G.edges()) == [('n0', 'n1')]
        assert G['n0']['n1']['id'] == 'e0'
        assert G.nodes['n0']['label'] == '1'
        assert G.nodes['n1']['label'] == '2'
        assert G.nodes['n2']['label'] == '3'
        assert G.nodes['n0']['shape_type'] == 'rectangle'
        assert G.nodes['n1']['shape_type'] == 'rectangle'
        assert G.nodes['n2']['shape_type'] == 'com.yworks.flowchart.terminator'
        assert G.nodes['n2']['description'] == 'description\nline1\nline2'
        H = nx.parse_graphml(data, force_multigraph=True)
        assert list(H.edges()) == [('n0', 'n1')]
        assert H.has_edge('n0', 'n1', key='e0')
        assert H.nodes['n0']['label'] == '1'
        assert H.nodes['n1']['label'] == '2'
        assert H.nodes['n2']['label'] == '3'
        H = nx.parse_graphml(data)
        assert list(H.edges()) == [('n0', 'n1')]
        assert H['n0']['n1']['id'] == 'e0'
        assert H.nodes['n0']['label'] == '1'
        assert H.nodes['n1']['label'] == '2'
        assert H.nodes['n2']['label'] == '3'

    def test_bool(self):
        if False:
            while True:
                i = 10
        s = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key id="d0" for="node" attr.name="test" attr.type="boolean">\n    <default>false</default>\n  </key>\n  <graph id="G" edgedefault="directed">\n    <node id="n0">\n      <data key="d0">true</data>\n    </node>\n    <node id="n1"/>\n    <node id="n2">\n      <data key="d0">false</data>\n    </node>\n    <node id="n3">\n      <data key="d0">FaLsE</data>\n    </node>\n    <node id="n4">\n      <data key="d0">True</data>\n    </node>\n    <node id="n5">\n      <data key="d0">0</data>\n    </node>\n    <node id="n6">\n      <data key="d0">1</data>\n    </node>\n  </graph>\n</graphml>\n'
        fh = io.BytesIO(s.encode('UTF-8'))
        G = nx.read_graphml(fh)
        H = nx.parse_graphml(s)
        for graph in [G, H]:
            assert graph.nodes['n0']['test']
            assert not graph.nodes['n2']['test']
            assert not graph.nodes['n3']['test']
            assert graph.nodes['n4']['test']
            assert not graph.nodes['n5']['test']
            assert graph.nodes['n6']['test']

    def test_graphml_header_line(self):
        if False:
            i = 10
            return i + 15
        good = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key id="d0" for="node" attr.name="test" attr.type="boolean">\n    <default>false</default>\n  </key>\n  <graph id="G">\n    <node id="n0">\n      <data key="d0">true</data>\n    </node>\n  </graph>\n</graphml>\n'
        bad = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<graphml>\n  <key id="d0" for="node" attr.name="test" attr.type="boolean">\n    <default>false</default>\n  </key>\n  <graph id="G">\n    <node id="n0">\n      <data key="d0">true</data>\n    </node>\n  </graph>\n</graphml>\n'
        ugly = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<graphml xmlns="https://ghghgh">\n  <key id="d0" for="node" attr.name="test" attr.type="boolean">\n    <default>false</default>\n  </key>\n  <graph id="G">\n    <node id="n0">\n      <data key="d0">true</data>\n    </node>\n  </graph>\n</graphml>\n'
        for s in (good, bad):
            fh = io.BytesIO(s.encode('UTF-8'))
            G = nx.read_graphml(fh)
            H = nx.parse_graphml(s)
            for graph in [G, H]:
                assert graph.nodes['n0']['test']
        fh = io.BytesIO(ugly.encode('UTF-8'))
        pytest.raises(nx.NetworkXError, nx.read_graphml, fh)
        pytest.raises(nx.NetworkXError, nx.parse_graphml, ugly)

    def test_read_attributes_with_groups(self):
        if False:
            for i in range(10):
                print('nop')
        data = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:java="http://www.yworks.com/xml/yfiles-common/1.0/java" xmlns:sys="http://www.yworks.com/xml/yfiles-common/markup/primitives/2.0" xmlns:x="http://www.yworks.com/xml/yfiles-common/markup/2.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:y="http://www.yworks.com/xml/graphml" xmlns:yed="http://www.yworks.com/xml/yed/3" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd">\n  <!--Created by yEd 3.17-->\n  <key attr.name="Description" attr.type="string" for="graph" id="d0"/>\n  <key for="port" id="d1" yfiles.type="portgraphics"/>\n  <key for="port" id="d2" yfiles.type="portgeometry"/>\n  <key for="port" id="d3" yfiles.type="portuserdata"/>\n  <key attr.name="CustomProperty" attr.type="string" for="node" id="d4">\n    <default/>\n  </key>\n  <key attr.name="url" attr.type="string" for="node" id="d5"/>\n  <key attr.name="description" attr.type="string" for="node" id="d6"/>\n  <key for="node" id="d7" yfiles.type="nodegraphics"/>\n  <key for="graphml" id="d8" yfiles.type="resources"/>\n  <key attr.name="url" attr.type="string" for="edge" id="d9"/>\n  <key attr.name="description" attr.type="string" for="edge" id="d10"/>\n  <key for="edge" id="d11" yfiles.type="edgegraphics"/>\n  <graph edgedefault="directed" id="G">\n    <data key="d0"/>\n    <node id="n0">\n      <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n      <data key="d6"/>\n      <data key="d7">\n        <y:ShapeNode>\n          <y:Geometry height="30.0" width="30.0" x="125.0" y="-255.4611111111111"/>\n          <y:Fill color="#FFCC00" transparent="false"/>\n          <y:BorderStyle color="#000000" raised="false" type="line" width="1.0"/>\n          <y:NodeLabel alignment="center" autoSizePolicy="content" fontFamily="Dialog" fontSize="12" fontStyle="plain" hasBackgroundColor="false" hasLineColor="false" height="17.96875" horizontalTextPosition="center" iconTextGap="4" modelName="custom" textColor="#000000" verticalTextPosition="bottom" visible="true" width="11.634765625" x="9.1826171875" y="6.015625">2<y:LabelModel>\n              <y:SmartNodeLabelModel distance="4.0"/>\n            </y:LabelModel>\n            <y:ModelParameter>\n              <y:SmartNodeLabelModelParameter labelRatioX="0.0" labelRatioY="0.0" nodeRatioX="0.0" nodeRatioY="0.0" offsetX="0.0" offsetY="0.0" upX="0.0" upY="-1.0"/>\n            </y:ModelParameter>\n          </y:NodeLabel>\n          <y:Shape type="rectangle"/>\n        </y:ShapeNode>\n      </data>\n    </node>\n    <node id="n1" yfiles.foldertype="group">\n      <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n      <data key="d5"/>\n      <data key="d6"/>\n      <data key="d7">\n        <y:ProxyAutoBoundsNode>\n          <y:Realizers active="0">\n            <y:GroupNode>\n              <y:Geometry height="250.38333333333333" width="140.0" x="-30.0" y="-330.3833333333333"/>\n              <y:Fill color="#F5F5F5" transparent="false"/>\n              <y:BorderStyle color="#000000" type="dashed" width="1.0"/>\n              <y:NodeLabel alignment="right" autoSizePolicy="node_width" backgroundColor="#EBEBEB" borderDistance="0.0" fontFamily="Dialog" fontSize="15" fontStyle="plain" hasLineColor="false" height="21.4609375" horizontalTextPosition="center" iconTextGap="4" modelName="internal" modelPosition="t" textColor="#000000" verticalTextPosition="bottom" visible="true" width="140.0" x="0.0" y="0.0">Group 3</y:NodeLabel>\n              <y:Shape type="roundrectangle"/>\n              <y:State closed="false" closedHeight="50.0" closedWidth="50.0" innerGraphDisplayEnabled="false"/>\n              <y:Insets bottom="15" bottomF="15.0" left="15" leftF="15.0" right="15" rightF="15.0" top="15" topF="15.0"/>\n              <y:BorderInsets bottom="1" bottomF="1.0" left="0" leftF="0.0" right="0" rightF="0.0" top="1" topF="1.0001736111111086"/>\n            </y:GroupNode>\n            <y:GroupNode>\n              <y:Geometry height="50.0" width="50.0" x="0.0" y="60.0"/>\n              <y:Fill color="#F5F5F5" transparent="false"/>\n              <y:BorderStyle color="#000000" type="dashed" width="1.0"/>\n              <y:NodeLabel alignment="right" autoSizePolicy="node_width" backgroundColor="#EBEBEB" borderDistance="0.0" fontFamily="Dialog" fontSize="15" fontStyle="plain" hasLineColor="false" height="21.4609375" horizontalTextPosition="center" iconTextGap="4" modelName="internal" modelPosition="t" textColor="#000000" verticalTextPosition="bottom" visible="true" width="65.201171875" x="-7.6005859375" y="0.0">Folder 3</y:NodeLabel>\n              <y:Shape type="roundrectangle"/>\n              <y:State closed="true" closedHeight="50.0" closedWidth="50.0" innerGraphDisplayEnabled="false"/>\n              <y:Insets bottom="5" bottomF="5.0" left="5" leftF="5.0" right="5" rightF="5.0" top="5" topF="5.0"/>\n              <y:BorderInsets bottom="0" bottomF="0.0" left="0" leftF="0.0" right="0" rightF="0.0" top="0" topF="0.0"/>\n            </y:GroupNode>\n          </y:Realizers>\n        </y:ProxyAutoBoundsNode>\n      </data>\n      <graph edgedefault="directed" id="n1:">\n        <node id="n1::n0" yfiles.foldertype="group">\n          <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n          <data key="d5"/>\n          <data key="d6"/>\n          <data key="d7">\n            <y:ProxyAutoBoundsNode>\n              <y:Realizers active="0">\n                <y:GroupNode>\n                  <y:Geometry height="83.46111111111111" width="110.0" x="-15.0" y="-292.9222222222222"/>\n                  <y:Fill color="#F5F5F5" transparent="false"/>\n                  <y:BorderStyle color="#000000" type="dashed" width="1.0"/>\n                  <y:NodeLabel alignment="right" autoSizePolicy="node_width" backgroundColor="#EBEBEB" borderDistance="0.0" fontFamily="Dialog" fontSize="15" fontStyle="plain" hasLineColor="false" height="21.4609375" horizontalTextPosition="center" iconTextGap="4" modelName="internal" modelPosition="t" textColor="#000000" verticalTextPosition="bottom" visible="true" width="110.0" x="0.0" y="0.0">Group 1</y:NodeLabel>\n                  <y:Shape type="roundrectangle"/>\n                  <y:State closed="false" closedHeight="50.0" closedWidth="50.0" innerGraphDisplayEnabled="false"/>\n                  <y:Insets bottom="15" bottomF="15.0" left="15" leftF="15.0" right="15" rightF="15.0" top="15" topF="15.0"/>\n                  <y:BorderInsets bottom="1" bottomF="1.0" left="0" leftF="0.0" right="0" rightF="0.0" top="1" topF="1.0001736111111086"/>\n                </y:GroupNode>\n                <y:GroupNode>\n                  <y:Geometry height="50.0" width="50.0" x="0.0" y="60.0"/>\n                  <y:Fill color="#F5F5F5" transparent="false"/>\n                  <y:BorderStyle color="#000000" type="dashed" width="1.0"/>\n                  <y:NodeLabel alignment="right" autoSizePolicy="node_width" backgroundColor="#EBEBEB" borderDistance="0.0" fontFamily="Dialog" fontSize="15" fontStyle="plain" hasLineColor="false" height="21.4609375" horizontalTextPosition="center" iconTextGap="4" modelName="internal" modelPosition="t" textColor="#000000" verticalTextPosition="bottom" visible="true" width="65.201171875" x="-7.6005859375" y="0.0">Folder 1</y:NodeLabel>\n                  <y:Shape type="roundrectangle"/>\n                  <y:State closed="true" closedHeight="50.0" closedWidth="50.0" innerGraphDisplayEnabled="false"/>\n                  <y:Insets bottom="5" bottomF="5.0" left="5" leftF="5.0" right="5" rightF="5.0" top="5" topF="5.0"/>\n                  <y:BorderInsets bottom="0" bottomF="0.0" left="0" leftF="0.0" right="0" rightF="0.0" top="0" topF="0.0"/>\n                </y:GroupNode>\n              </y:Realizers>\n            </y:ProxyAutoBoundsNode>\n          </data>\n          <graph edgedefault="directed" id="n1::n0:">\n            <node id="n1::n0::n0">\n              <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n              <data key="d6"/>\n              <data key="d7">\n                <y:ShapeNode>\n                  <y:Geometry height="30.0" width="30.0" x="50.0" y="-255.4611111111111"/>\n                  <y:Fill color="#FFCC00" transparent="false"/>\n                  <y:BorderStyle color="#000000" raised="false" type="line" width="1.0"/>\n                  <y:NodeLabel alignment="center" autoSizePolicy="content" fontFamily="Dialog" fontSize="12" fontStyle="plain" hasBackgroundColor="false" hasLineColor="false" height="17.96875" horizontalTextPosition="center" iconTextGap="4" modelName="custom" textColor="#000000" verticalTextPosition="bottom" visible="true" width="11.634765625" x="9.1826171875" y="6.015625">1<y:LabelModel>\n                      <y:SmartNodeLabelModel distance="4.0"/>\n                    </y:LabelModel>\n                    <y:ModelParameter>\n                      <y:SmartNodeLabelModelParameter labelRatioX="0.0" labelRatioY="0.0" nodeRatioX="0.0" nodeRatioY="0.0" offsetX="0.0" offsetY="0.0" upX="0.0" upY="-1.0"/>\n                    </y:ModelParameter>\n                  </y:NodeLabel>\n                  <y:Shape type="rectangle"/>\n                </y:ShapeNode>\n              </data>\n            </node>\n            <node id="n1::n0::n1">\n              <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n              <data key="d6"/>\n              <data key="d7">\n                <y:ShapeNode>\n                  <y:Geometry height="30.0" width="30.0" x="0.0" y="-255.4611111111111"/>\n                  <y:Fill color="#FFCC00" transparent="false"/>\n                  <y:BorderStyle color="#000000" raised="false" type="line" width="1.0"/>\n                  <y:NodeLabel alignment="center" autoSizePolicy="content" fontFamily="Dialog" fontSize="12" fontStyle="plain" hasBackgroundColor="false" hasLineColor="false" height="17.96875" horizontalTextPosition="center" iconTextGap="4" modelName="custom" textColor="#000000" verticalTextPosition="bottom" visible="true" width="11.634765625" x="9.1826171875" y="6.015625">3<y:LabelModel>\n                      <y:SmartNodeLabelModel distance="4.0"/>\n                    </y:LabelModel>\n                    <y:ModelParameter>\n                      <y:SmartNodeLabelModelParameter labelRatioX="0.0" labelRatioY="0.0" nodeRatioX="0.0" nodeRatioY="0.0" offsetX="0.0" offsetY="0.0" upX="0.0" upY="-1.0"/>\n                    </y:ModelParameter>\n                  </y:NodeLabel>\n                  <y:Shape type="rectangle"/>\n                </y:ShapeNode>\n              </data>\n            </node>\n          </graph>\n        </node>\n        <node id="n1::n1" yfiles.foldertype="group">\n          <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n          <data key="d5"/>\n          <data key="d6"/>\n          <data key="d7">\n            <y:ProxyAutoBoundsNode>\n              <y:Realizers active="0">\n                <y:GroupNode>\n                  <y:Geometry height="83.46111111111111" width="110.0" x="-15.0" y="-179.4611111111111"/>\n                  <y:Fill color="#F5F5F5" transparent="false"/>\n                  <y:BorderStyle color="#000000" type="dashed" width="1.0"/>\n                  <y:NodeLabel alignment="right" autoSizePolicy="node_width" backgroundColor="#EBEBEB" borderDistance="0.0" fontFamily="Dialog" fontSize="15" fontStyle="plain" hasLineColor="false" height="21.4609375" horizontalTextPosition="center" iconTextGap="4" modelName="internal" modelPosition="t" textColor="#000000" verticalTextPosition="bottom" visible="true" width="110.0" x="0.0" y="0.0">Group 2</y:NodeLabel>\n                  <y:Shape type="roundrectangle"/>\n                  <y:State closed="false" closedHeight="50.0" closedWidth="50.0" innerGraphDisplayEnabled="false"/>\n                  <y:Insets bottom="15" bottomF="15.0" left="15" leftF="15.0" right="15" rightF="15.0" top="15" topF="15.0"/>\n                  <y:BorderInsets bottom="1" bottomF="1.0" left="0" leftF="0.0" right="0" rightF="0.0" top="1" topF="1.0001736111111086"/>\n                </y:GroupNode>\n                <y:GroupNode>\n                  <y:Geometry height="50.0" width="50.0" x="0.0" y="60.0"/>\n                  <y:Fill color="#F5F5F5" transparent="false"/>\n                  <y:BorderStyle color="#000000" type="dashed" width="1.0"/>\n                  <y:NodeLabel alignment="right" autoSizePolicy="node_width" backgroundColor="#EBEBEB" borderDistance="0.0" fontFamily="Dialog" fontSize="15" fontStyle="plain" hasLineColor="false" height="21.4609375" horizontalTextPosition="center" iconTextGap="4" modelName="internal" modelPosition="t" textColor="#000000" verticalTextPosition="bottom" visible="true" width="65.201171875" x="-7.6005859375" y="0.0">Folder 2</y:NodeLabel>\n                  <y:Shape type="roundrectangle"/>\n                  <y:State closed="true" closedHeight="50.0" closedWidth="50.0" innerGraphDisplayEnabled="false"/>\n                  <y:Insets bottom="5" bottomF="5.0" left="5" leftF="5.0" right="5" rightF="5.0" top="5" topF="5.0"/>\n                  <y:BorderInsets bottom="0" bottomF="0.0" left="0" leftF="0.0" right="0" rightF="0.0" top="0" topF="0.0"/>\n                </y:GroupNode>\n              </y:Realizers>\n            </y:ProxyAutoBoundsNode>\n          </data>\n          <graph edgedefault="directed" id="n1::n1:">\n            <node id="n1::n1::n0">\n              <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n              <data key="d6"/>\n              <data key="d7">\n                <y:ShapeNode>\n                  <y:Geometry height="30.0" width="30.0" x="0.0" y="-142.0"/>\n                  <y:Fill color="#FFCC00" transparent="false"/>\n                  <y:BorderStyle color="#000000" raised="false" type="line" width="1.0"/>\n                  <y:NodeLabel alignment="center" autoSizePolicy="content" fontFamily="Dialog" fontSize="12" fontStyle="plain" hasBackgroundColor="false" hasLineColor="false" height="17.96875" horizontalTextPosition="center" iconTextGap="4" modelName="custom" textColor="#000000" verticalTextPosition="bottom" visible="true" width="11.634765625" x="9.1826171875" y="6.015625">5<y:LabelModel>\n                      <y:SmartNodeLabelModel distance="4.0"/>\n                    </y:LabelModel>\n                    <y:ModelParameter>\n                      <y:SmartNodeLabelModelParameter labelRatioX="0.0" labelRatioY="0.0" nodeRatioX="0.0" nodeRatioY="0.0" offsetX="0.0" offsetY="0.0" upX="0.0" upY="-1.0"/>\n                    </y:ModelParameter>\n                  </y:NodeLabel>\n                  <y:Shape type="rectangle"/>\n                </y:ShapeNode>\n              </data>\n            </node>\n            <node id="n1::n1::n1">\n              <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n              <data key="d6"/>\n              <data key="d7">\n                <y:ShapeNode>\n                  <y:Geometry height="30.0" width="30.0" x="50.0" y="-142.0"/>\n                  <y:Fill color="#FFCC00" transparent="false"/>\n                  <y:BorderStyle color="#000000" raised="false" type="line" width="1.0"/>\n                  <y:NodeLabel alignment="center" autoSizePolicy="content" fontFamily="Dialog" fontSize="12" fontStyle="plain" hasBackgroundColor="false" hasLineColor="false" height="17.96875" horizontalTextPosition="center" iconTextGap="4" modelName="custom" textColor="#000000" verticalTextPosition="bottom" visible="true" width="11.634765625" x="9.1826171875" y="6.015625">6<y:LabelModel>\n                      <y:SmartNodeLabelModel distance="4.0"/>\n                    </y:LabelModel>\n                    <y:ModelParameter>\n                      <y:SmartNodeLabelModelParameter labelRatioX="0.0" labelRatioY="0.0" nodeRatioX="0.0" nodeRatioY="0.0" offsetX="0.0" offsetY="0.0" upX="0.0" upY="-1.0"/>\n                    </y:ModelParameter>\n                  </y:NodeLabel>\n                  <y:Shape type="rectangle"/>\n                </y:ShapeNode>\n              </data>\n            </node>\n          </graph>\n        </node>\n      </graph>\n    </node>\n    <node id="n2">\n      <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n      <data key="d6"/>\n      <data key="d7">\n        <y:ShapeNode>\n          <y:Geometry height="30.0" width="30.0" x="125.0" y="-142.0"/>\n          <y:Fill color="#FFCC00" transparent="false"/>\n          <y:BorderStyle color="#000000" raised="false" type="line" width="1.0"/>\n          <y:NodeLabel alignment="center" autoSizePolicy="content" fontFamily="Dialog" fontSize="12" fontStyle="plain" hasBackgroundColor="false" hasLineColor="false" height="17.96875" horizontalTextPosition="center" iconTextGap="4" modelName="custom" textColor="#000000" verticalTextPosition="bottom" visible="true" width="11.634765625" x="9.1826171875" y="6.015625">9<y:LabelModel>\n              <y:SmartNodeLabelModel distance="4.0"/>\n            </y:LabelModel>\n            <y:ModelParameter>\n              <y:SmartNodeLabelModelParameter labelRatioX="0.0" labelRatioY="0.0" nodeRatioX="0.0" nodeRatioY="0.0" offsetX="0.0" offsetY="0.0" upX="0.0" upY="-1.0"/>\n            </y:ModelParameter>\n          </y:NodeLabel>\n          <y:Shape type="rectangle"/>\n        </y:ShapeNode>\n      </data>\n    </node>\n    <edge id="n1::n1::e0" source="n1::n1::n0" target="n1::n1::n1">\n      <data key="d10"/>\n      <data key="d11">\n        <y:PolyLineEdge>\n          <y:Path sx="15.0" sy="-0.0" tx="-15.0" ty="-0.0"/>\n          <y:LineStyle color="#000000" type="line" width="1.0"/>\n          <y:Arrows source="none" target="standard"/>\n          <y:BendStyle smoothed="false"/>\n        </y:PolyLineEdge>\n      </data>\n    </edge>\n    <edge id="n1::n0::e0" source="n1::n0::n1" target="n1::n0::n0">\n      <data key="d10"/>\n      <data key="d11">\n        <y:PolyLineEdge>\n          <y:Path sx="15.0" sy="-0.0" tx="-15.0" ty="-0.0"/>\n          <y:LineStyle color="#000000" type="line" width="1.0"/>\n          <y:Arrows source="none" target="standard"/>\n          <y:BendStyle smoothed="false"/>\n        </y:PolyLineEdge>\n      </data>\n    </edge>\n    <edge id="e0" source="n1::n0::n0" target="n0">\n      <data key="d10"/>\n      <data key="d11">\n        <y:PolyLineEdge>\n          <y:Path sx="15.0" sy="-0.0" tx="-15.0" ty="-0.0"/>\n          <y:LineStyle color="#000000" type="line" width="1.0"/>\n          <y:Arrows source="none" target="standard"/>\n          <y:BendStyle smoothed="false"/>\n        </y:PolyLineEdge>\n      </data>\n    </edge>\n    <edge id="e1" source="n1::n1::n1" target="n2">\n      <data key="d10"/>\n      <data key="d11">\n        <y:PolyLineEdge>\n          <y:Path sx="15.0" sy="-0.0" tx="-15.0" ty="-0.0"/>\n          <y:LineStyle color="#000000" type="line" width="1.0"/>\n          <y:Arrows source="none" target="standard"/>\n          <y:BendStyle smoothed="false"/>\n        </y:PolyLineEdge>\n      </data>\n    </edge>\n  </graph>\n  <data key="d8">\n    <y:Resources/>\n  </data>\n</graphml>\n'
        fh = io.BytesIO(data.encode('UTF-8'))
        G = nx.read_graphml(fh)
        data = [x for (_, x) in G.nodes(data=True)]
        assert len(data) == 9
        for node_data in data:
            assert node_data['CustomProperty'] != ''

    def test_long_attribute_type(self):
        if False:
            print('Hello World!')
        s = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key attr.name="cudfversion" attr.type="long" for="node" id="d6" />\n  <graph edgedefault="directed">\n    <node id="n1">\n      <data key="d6">4284</data>\n    </node>\n  </graph>\n</graphml>'
        fh = io.BytesIO(s.encode('UTF-8'))
        G = nx.read_graphml(fh)
        expected = [('n1', {'cudfversion': 4284})]
        assert sorted(G.nodes(data=True)) == expected
        fh.seek(0)
        H = nx.parse_graphml(s)
        assert sorted(H.nodes(data=True)) == expected

class TestWriteGraphML(BaseGraphML):
    writer = staticmethod(nx.write_graphml_lxml)

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        BaseGraphML.setup_class()
        _ = pytest.importorskip('lxml.etree')

    def test_write_interface(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            import lxml.etree
            assert nx.write_graphml == nx.write_graphml_lxml
        except ImportError:
            assert nx.write_graphml == nx.write_graphml_xml

    def test_write_read_simple_directed_graphml(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.simple_directed_graph
        G.graph['hi'] = 'there'
        fh = io.BytesIO()
        self.writer(G, fh)
        fh.seek(0)
        H = nx.read_graphml(fh)
        assert sorted(G.nodes()) == sorted(H.nodes())
        assert sorted(G.edges()) == sorted(H.edges())
        assert sorted(G.edges(data=True)) == sorted(H.edges(data=True))
        self.simple_directed_fh.seek(0)

    def test_GraphMLWriter_add_graphs(self):
        if False:
            return 10
        gmlw = GraphMLWriter()
        G = self.simple_directed_graph
        H = G.copy()
        gmlw.add_graphs([G, H])

    def test_write_read_simple_no_prettyprint(self):
        if False:
            i = 10
            return i + 15
        G = self.simple_directed_graph
        G.graph['hi'] = 'there'
        G.graph['id'] = '1'
        fh = io.BytesIO()
        self.writer(G, fh, prettyprint=False)
        fh.seek(0)
        H = nx.read_graphml(fh)
        assert sorted(G.nodes()) == sorted(H.nodes())
        assert sorted(G.edges()) == sorted(H.edges())
        assert sorted(G.edges(data=True)) == sorted(H.edges(data=True))
        self.simple_directed_fh.seek(0)

    def test_write_read_attribute_named_key_ids_graphml(self):
        if False:
            i = 10
            return i + 15
        from xml.etree.ElementTree import parse
        G = self.attribute_named_key_ids_graph
        fh = io.BytesIO()
        self.writer(G, fh, named_key_ids=True)
        fh.seek(0)
        H = nx.read_graphml(fh)
        fh.seek(0)
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        assert edges_equal(G.edges(data=True), H.edges(data=True))
        self.attribute_named_key_ids_fh.seek(0)
        xml = parse(fh)
        children = list(xml.getroot())
        assert len(children) == 4
        keys = [child.items() for child in children[:3]]
        assert len(keys) == 3
        assert ('id', 'edge_prop') in keys[0]
        assert ('attr.name', 'edge_prop') in keys[0]
        assert ('id', 'prop2') in keys[1]
        assert ('attr.name', 'prop2') in keys[1]
        assert ('id', 'prop1') in keys[2]
        assert ('attr.name', 'prop1') in keys[2]
        default_behavior_fh = io.BytesIO()
        nx.write_graphml(G, default_behavior_fh)
        default_behavior_fh.seek(0)
        H = nx.read_graphml(default_behavior_fh)
        named_key_ids_behavior_fh = io.BytesIO()
        nx.write_graphml(G, named_key_ids_behavior_fh, named_key_ids=True)
        named_key_ids_behavior_fh.seek(0)
        J = nx.read_graphml(named_key_ids_behavior_fh)
        assert all((n1 == n2 for (n1, n2) in zip(H.nodes, J.nodes)))
        assert all((e1 == e2 for (e1, e2) in zip(H.edges, J.edges)))

    def test_write_read_attribute_numeric_type_graphml(self):
        if False:
            while True:
                i = 10
        from xml.etree.ElementTree import parse
        G = self.attribute_numeric_type_graph
        fh = io.BytesIO()
        self.writer(G, fh, infer_numeric_types=True)
        fh.seek(0)
        H = nx.read_graphml(fh)
        fh.seek(0)
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        assert edges_equal(G.edges(data=True), H.edges(data=True))
        self.attribute_numeric_type_fh.seek(0)
        xml = parse(fh)
        children = list(xml.getroot())
        assert len(children) == 3
        keys = [child.items() for child in children[:2]]
        assert len(keys) == 2
        assert ('attr.type', 'double') in keys[0]
        assert ('attr.type', 'double') in keys[1]

    def test_more_multigraph_keys(self):
        if False:
            while True:
                i = 10
        'Writing keys as edge id attributes means keys become strings.\n        The original keys are stored as data, so read them back in\n        if `str(key) == edge_id`\n        This allows the adjacency to remain the same.\n        '
        G = nx.MultiGraph()
        G.add_edges_from([('a', 'b', 2), ('a', 'b', 3)])
        (fd, fname) = tempfile.mkstemp()
        self.writer(G, fname)
        H = nx.read_graphml(fname)
        assert H.is_multigraph()
        assert edges_equal(G.edges(keys=True), H.edges(keys=True))
        assert G._adj == H._adj
        os.close(fd)
        os.unlink(fname)

    def test_default_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        G = nx.Graph(name='Fred')
        G.add_node(1, label=1, color='green')
        nx.add_path(G, [0, 1, 2, 3])
        G.add_edge(1, 2, weight=3)
        G.graph['node_default'] = {'color': 'yellow'}
        G.graph['edge_default'] = {'weight': 7}
        fh = io.BytesIO()
        self.writer(G, fh)
        fh.seek(0)
        H = nx.read_graphml(fh, node_type=int)
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        assert G.graph == H.graph

    def test_mixed_type_attributes(self):
        if False:
            i = 10
            return i + 15
        G = nx.MultiGraph()
        G.add_node('n0', special=False)
        G.add_node('n1', special=0)
        G.add_edge('n0', 'n1', special=False)
        G.add_edge('n0', 'n1', special=0)
        fh = io.BytesIO()
        self.writer(G, fh)
        fh.seek(0)
        H = nx.read_graphml(fh)
        assert not H.nodes['n0']['special']
        assert H.nodes['n1']['special'] == 0
        assert not H.edges['n0', 'n1', 0]['special']
        assert H.edges['n0', 'n1', 1]['special'] == 0

    def test_str_number_mixed_type_attributes(self):
        if False:
            i = 10
            return i + 15
        G = nx.MultiGraph()
        G.add_node('n0', special='hello')
        G.add_node('n1', special=0)
        G.add_edge('n0', 'n1', special='hello')
        G.add_edge('n0', 'n1', special=0)
        fh = io.BytesIO()
        self.writer(G, fh)
        fh.seek(0)
        H = nx.read_graphml(fh)
        assert H.nodes['n0']['special'] == 'hello'
        assert H.nodes['n1']['special'] == 0
        assert H.edges['n0', 'n1', 0]['special'] == 'hello'
        assert H.edges['n0', 'n1', 1]['special'] == 0

    def test_mixed_int_type_number_attributes(self):
        if False:
            while True:
                i = 10
        np = pytest.importorskip('numpy')
        G = nx.MultiGraph()
        G.add_node('n0', special=np.int64(0))
        G.add_node('n1', special=1)
        G.add_edge('n0', 'n1', special=np.int64(2))
        G.add_edge('n0', 'n1', special=3)
        fh = io.BytesIO()
        self.writer(G, fh)
        fh.seek(0)
        H = nx.read_graphml(fh)
        assert H.nodes['n0']['special'] == 0
        assert H.nodes['n1']['special'] == 1
        assert H.edges['n0', 'n1', 0]['special'] == 2
        assert H.edges['n0', 'n1', 1]['special'] == 3

    def test_multigraph_to_graph(self):
        if False:
            print('Hello World!')
        G = nx.MultiGraph()
        G.add_edges_from([('a', 'b', 2), ('b', 'c', 3)])
        (fd, fname) = tempfile.mkstemp()
        self.writer(G, fname)
        H = nx.read_graphml(fname)
        assert not H.is_multigraph()
        H = nx.read_graphml(fname, force_multigraph=True)
        assert H.is_multigraph()
        os.close(fd)
        os.unlink(fname)
        G.add_edge('a', 'b', 'e-id')
        (fd, fname) = tempfile.mkstemp()
        self.writer(G, fname)
        H = nx.read_graphml(fname)
        assert H.is_multigraph()
        H = nx.read_graphml(fname, force_multigraph=True)
        assert H.is_multigraph()
        os.close(fd)
        os.unlink(fname)

    def test_write_generate_edge_id_from_attribute(self):
        if False:
            return 10
        from xml.etree.ElementTree import parse
        G = nx.Graph()
        G.add_edges_from([('a', 'b'), ('b', 'c'), ('a', 'c')])
        edge_attributes = {e: str(e) for e in G.edges}
        nx.set_edge_attributes(G, edge_attributes, 'eid')
        (fd, fname) = tempfile.mkstemp()
        self.writer(G, fname, edge_id_from_attribute='eid')
        generator = nx.generate_graphml(G, edge_id_from_attribute='eid')
        H = nx.read_graphml(fname)
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        nx.set_edge_attributes(G, edge_attributes, 'id')
        assert edges_equal(G.edges(data=True), H.edges(data=True))
        tree = parse(fname)
        children = list(tree.getroot())
        assert len(children) == 2
        edge_ids = [edge.attrib['id'] for edge in tree.getroot().findall('.//{http://graphml.graphdrawing.org/xmlns}edge')]
        assert sorted(edge_ids) == sorted(edge_attributes.values())
        data = ''.join(generator)
        J = nx.parse_graphml(data)
        assert sorted(G.nodes()) == sorted(J.nodes())
        assert sorted(G.edges()) == sorted(J.edges())
        nx.set_edge_attributes(G, edge_attributes, 'id')
        assert edges_equal(G.edges(data=True), J.edges(data=True))
        os.close(fd)
        os.unlink(fname)

    def test_multigraph_write_generate_edge_id_from_attribute(self):
        if False:
            i = 10
            return i + 15
        from xml.etree.ElementTree import parse
        G = nx.MultiGraph()
        G.add_edges_from([('a', 'b'), ('b', 'c'), ('a', 'c'), ('a', 'b')])
        edge_attributes = {e: str(e) for e in G.edges}
        nx.set_edge_attributes(G, edge_attributes, 'eid')
        (fd, fname) = tempfile.mkstemp()
        self.writer(G, fname, edge_id_from_attribute='eid')
        generator = nx.generate_graphml(G, edge_id_from_attribute='eid')
        H = nx.read_graphml(fname)
        assert H.is_multigraph()
        H = nx.read_graphml(fname, force_multigraph=True)
        assert H.is_multigraph()
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        assert sorted((data.get('eid') for (u, v, data) in H.edges(data=True))) == sorted(edge_attributes.values())
        assert sorted((key for (u, v, key) in H.edges(keys=True))) == sorted(edge_attributes.values())
        tree = parse(fname)
        children = list(tree.getroot())
        assert len(children) == 2
        edge_ids = [edge.attrib['id'] for edge in tree.getroot().findall('.//{http://graphml.graphdrawing.org/xmlns}edge')]
        assert sorted(edge_ids) == sorted(edge_attributes.values())
        graphml_data = ''.join(generator)
        J = nx.parse_graphml(graphml_data)
        assert J.is_multigraph()
        assert nodes_equal(G.nodes(), J.nodes())
        assert edges_equal(G.edges(), J.edges())
        assert sorted((data.get('eid') for (u, v, data) in J.edges(data=True))) == sorted(edge_attributes.values())
        assert sorted((key for (u, v, key) in J.edges(keys=True))) == sorted(edge_attributes.values())
        os.close(fd)
        os.unlink(fname)

    def test_numpy_float64(self):
        if False:
            for i in range(10):
                print('nop')
        np = pytest.importorskip('numpy')
        wt = np.float64(3.4)
        G = nx.Graph([(1, 2, {'weight': wt})])
        (fd, fname) = tempfile.mkstemp()
        self.writer(G, fname)
        H = nx.read_graphml(fname, node_type=int)
        assert G.edges == H.edges
        wtG = G[1][2]['weight']
        wtH = H[1][2]['weight']
        assert wtG == pytest.approx(wtH, abs=1e-06)
        assert type(wtG) == np.float64
        assert type(wtH) == float
        os.close(fd)
        os.unlink(fname)

    def test_numpy_float32(self):
        if False:
            i = 10
            return i + 15
        np = pytest.importorskip('numpy')
        wt = np.float32(3.4)
        G = nx.Graph([(1, 2, {'weight': wt})])
        (fd, fname) = tempfile.mkstemp()
        self.writer(G, fname)
        H = nx.read_graphml(fname, node_type=int)
        assert G.edges == H.edges
        wtG = G[1][2]['weight']
        wtH = H[1][2]['weight']
        assert wtG == pytest.approx(wtH, abs=1e-06)
        assert type(wtG) == np.float32
        assert type(wtH) == float
        os.close(fd)
        os.unlink(fname)

    def test_numpy_float64_inference(self):
        if False:
            while True:
                i = 10
        np = pytest.importorskip('numpy')
        G = self.attribute_numeric_type_graph
        G.edges['n1', 'n1']['weight'] = np.float64(1.1)
        (fd, fname) = tempfile.mkstemp()
        self.writer(G, fname, infer_numeric_types=True)
        H = nx.read_graphml(fname)
        assert G._adj == H._adj
        os.close(fd)
        os.unlink(fname)

    def test_unicode_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        G = nx.Graph()
        name1 = chr(2344) + chr(123) + chr(6543)
        name2 = chr(5543) + chr(1543) + chr(324)
        node_type = str
        G.add_edge(name1, 'Radiohead', foo=name2)
        (fd, fname) = tempfile.mkstemp()
        self.writer(G, fname)
        H = nx.read_graphml(fname, node_type=node_type)
        assert G._adj == H._adj
        os.close(fd)
        os.unlink(fname)

    def test_unicode_escape(self):
        if False:
            print('Hello World!')
        import json
        a = {'a': '{"a": "123"}'}
        sa = json.dumps(a)
        G = nx.Graph()
        G.graph['test'] = sa
        fh = io.BytesIO()
        self.writer(G, fh)
        fh.seek(0)
        H = nx.read_graphml(fh)
        assert G.graph['test'] == H.graph['test']

class TestXMLGraphML(TestWriteGraphML):
    writer = staticmethod(nx.write_graphml_xml)

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        TestWriteGraphML.setup_class()

def test_exception_for_unsupported_datatype_node_attr():
    if False:
        print('Hello World!')
    'Test that a detailed exception is raised when an attribute is of a type\n    not supported by GraphML, e.g. a list'
    pytest.importorskip('lxml.etree')
    G = nx.Graph()
    G.add_node(0, my_list_attribute=[0, 1, 2])
    fh = io.BytesIO()
    with pytest.raises(TypeError, match='GraphML does not support'):
        nx.write_graphml(G, fh)

def test_exception_for_unsupported_datatype_edge_attr():
    if False:
        i = 10
        return i + 15
    'Test that a detailed exception is raised when an attribute is of a type\n    not supported by GraphML, e.g. a list'
    pytest.importorskip('lxml.etree')
    G = nx.Graph()
    G.add_edge(0, 1, my_list_attribute=[0, 1, 2])
    fh = io.BytesIO()
    with pytest.raises(TypeError, match='GraphML does not support'):
        nx.write_graphml(G, fh)

def test_exception_for_unsupported_datatype_graph_attr():
    if False:
        return 10
    'Test that a detailed exception is raised when an attribute is of a type\n    not supported by GraphML, e.g. a list'
    pytest.importorskip('lxml.etree')
    G = nx.Graph()
    G.graph['my_list_attribute'] = [0, 1, 2]
    fh = io.BytesIO()
    with pytest.raises(TypeError, match='GraphML does not support'):
        nx.write_graphml(G, fh)